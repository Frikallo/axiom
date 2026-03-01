// GPU Full-Graph Compiler
//
// Compiles an arbitrary lazy computation DAG into a single MPSGraph.
// This replaces hundreds of individual GPU dispatches with one fused graph
// execution, dramatically improving both performance and numerical accuracy.
//
// The compiler walks the lazy DAG in topological order, mapping each GraphNode
// to its MPSGraph equivalent. Compiled graphs are cached by structural
// signature (FNV-1a hash of ops, shapes, dtypes, connectivity).

#import "axiom/graph/graph_node.hpp"
#import "axiom/graph/graph_registry.hpp"
#import "axiom/graph/graph_signature.hpp"
#import "axiom/error.hpp"
#import "axiom/operations.hpp"
#import "metal_common.hpp"
#import "metal_buffer_provider.hpp"
#import "metal_storage.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace axiom {
namespace graph {

// ============================================================================
// MPS helpers
// ============================================================================

static MPSDataType toMPS(DType dtype) {
    switch (dtype) {
    case DType::Float32: return MPSDataTypeFloat32;
    case DType::Float16: return MPSDataTypeFloat16;
    case DType::Int32: return MPSDataTypeInt32;
    case DType::Int64: return MPSDataTypeInt64;
    case DType::Int16: return MPSDataTypeInt16;
    case DType::Int8: return MPSDataTypeInt8;
    case DType::UInt8: return MPSDataTypeUInt8;
    case DType::UInt16: return MPSDataTypeUInt16;
    case DType::UInt32: return MPSDataTypeUInt32;
    case DType::UInt64: return MPSDataTypeUInt64;
    case DType::Bool: return MPSDataTypeBool;
    case DType::BFloat16: return MPSDataTypeBFloat16;
    default: return static_cast<MPSDataType>(0);
    }
}

static MPSShape *toMPSShape(const Shape &shape) {
    NSMutableArray<NSNumber *> *s =
        [NSMutableArray arrayWithCapacity:shape.size()];
    for (size_t d : shape)
        [s addObject:@(static_cast<NSInteger>(d))];
    return s;
}

static MPSGraphTensorData *makeTensorData(const Tensor &t) {
    auto *ms = backends::metal::as_metal_buffer_provider(t.storage().get());
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)ms->buffer();
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                   shape:toMPSShape(t.shape())
                                                dataType:toMPS(t.dtype())];
}

static void encodeMPSGraphAsync(
    MPSGraph *graph,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *targets) {
    auto &stream = backends::metal::MetalExecutionStream::instance();
    MPSCommandBuffer *cmdBuf =
        (__bridge MPSCommandBuffer *)stream.current_mps_buffer();
    MPSGraphExecutionDescriptor *desc =
        (__bridge MPSGraphExecutionDescriptor *)stream.execution_descriptor();
    [graph encodeToCommandBuffer:cmdBuf
                           feeds:feeds
                targetOperations:nil
               resultsDictionary:targets
             executionDescriptor:desc];
    stream.increment_batch();
}

// ============================================================================
// Topological sort of a lazy DAG
// ============================================================================

static std::vector<const GraphNode *> topo_sort(const GraphNode *root) {
    std::vector<const GraphNode *> result;
    std::unordered_set<const GraphNode *> visited;
    std::vector<std::pair<const GraphNode *, size_t>> stack;
    stack.push_back({root, 0});

    while (!stack.empty()) {
        auto &[node, idx] = stack.back();
        if (visited.count(node)) {
            stack.pop_back();
            continue;
        }
        // Constant/materialized nodes are leaves — don't recurse
        if (node != root && (node->is_constant || node->is_materialized_)) {
            visited.insert(node);
            result.push_back(node);
            stack.pop_back();
            continue;
        }
        if (idx < node->inputs.size()) {
            const GraphNode *child = node->inputs[idx].get();
            idx++;
            if (!visited.count(child))
                stack.push_back({child, 0});
        } else {
            visited.insert(node);
            result.push_back(node);
            stack.pop_back();
        }
    }
    return result;
}

// ============================================================================
// Collect leaf input tensors from a lazy DAG
// ============================================================================

static std::vector<Tensor>
collect_leaf_tensors(const std::vector<const GraphNode *> &sorted) {
    std::vector<Tensor> inputs;
    std::unordered_set<const GraphNode *> seen;
    for (const auto *n : sorted) {
        if (n->is_constant || n->is_materialized_) {
            // Skip empty tensors (e.g., no-bias placeholder)
            if (n->output_shape.empty() && !n->constant_storage)
                continue;
            if (seen.insert(n).second) {
                if (n->is_constant) {
                    inputs.emplace_back(n->constant_storage, n->output_shape,
                                        n->constant_strides, n->output_dtype,
                                        n->constant_offset);
                } else {
                    inputs.emplace_back(n->cached_result_, n->cached_shape_,
                                        n->cached_strides_, n->output_dtype);
                }
            }
        }
    }
    return inputs;
}

// ============================================================================
// MPSGraph node builders for each OpType
// ============================================================================

// Elementwise unary
static MPSGraphTensor *build_unary(MPSGraph *g, ops::OpType op,
                                   MPSGraphTensor *a) {
    using ops::OpType;
    switch (op) {
    case OpType::Negate: return [g negativeWithTensor:a name:nil];
    case OpType::Abs: return [g absoluteWithTensor:a name:nil];
    case OpType::Sqrt: return [g squareRootWithTensor:a name:nil];
    case OpType::Exp: return [g exponentWithTensor:a name:nil];
    case OpType::Log: return [g logarithmWithTensor:a name:nil];
    case OpType::Sin: return [g sinWithTensor:a name:nil];
    case OpType::Cos: return [g cosWithTensor:a name:nil];
    case OpType::Tan: return [g tanWithTensor:a name:nil];
    case OpType::Tanh: return [g tanhWithTensor:a name:nil];
    case OpType::Erf: return [g erfWithTensor:a name:nil];
    case OpType::Square: return [g squareWithTensor:a name:nil];
    case OpType::Reciprocal: return [g reciprocalWithTensor:a name:nil];
    case OpType::Sign: return [g signWithTensor:a name:nil];
    case OpType::Floor: return [g floorWithTensor:a name:nil];
    case OpType::Ceil: return [g ceilWithTensor:a name:nil];
    case OpType::Round: return [g roundWithTensor:a name:nil];
    case OpType::Trunc: return [g truncateWithTensor:a name:nil];
    case OpType::ReLU: return [g reLUWithTensor:a name:nil];
    case OpType::Sigmoid: return [g sigmoidWithTensor:a name:nil];
    case OpType::SiLU: {
        auto *sig = [g sigmoidWithTensor:a name:nil];
        return [g multiplicationWithPrimaryTensor:a secondaryTensor:sig name:nil];
    }
    case OpType::GELU: {
        auto *half = [g constantWithScalar:0.5 dataType:a.dataType];
        auto *inv_sqrt2 = [g constantWithScalar:0.7071067811865475 dataType:a.dataType];
        auto *scaled = [g multiplicationWithPrimaryTensor:a secondaryTensor:inv_sqrt2 name:nil];
        auto *e = [g erfWithTensor:scaled name:nil];
        auto *one = [g constantWithScalar:1.0 dataType:a.dataType];
        auto *inner = [g additionWithPrimaryTensor:one secondaryTensor:e name:nil];
        auto *h = [g multiplicationWithPrimaryTensor:half secondaryTensor:inner name:nil];
        return [g multiplicationWithPrimaryTensor:a secondaryTensor:h name:nil];
    }
    case OpType::LogicalNot: {
        auto *b = [g castTensor:a toType:MPSDataTypeBool name:nil];
        return [g notWithTensor:b name:nil];
    }
    case OpType::Cbrt: {
        auto *third = [g constantWithScalar:1.0 / 3.0 dataType:a.dataType];
        // cbrt handles negative values: sign(x) * |x|^(1/3)
        auto *sign_a = [g signWithTensor:a name:nil];
        auto *abs_a = [g absoluteWithTensor:a name:nil];
        auto *pow_a = [g powerWithPrimaryTensor:abs_a secondaryTensor:third name:nil];
        return [g multiplicationWithPrimaryTensor:sign_a secondaryTensor:pow_a name:nil];
    }
    case OpType::IsNaN: return [g isNaNWithTensor:a name:nil];
    case OpType::IsInf: return [g isInfiniteWithTensor:a name:nil];
    case OpType::IsFinite: return [g isFiniteWithTensor:a name:nil];
    default: return nil;
    }
}

// Elementwise binary
static MPSGraphTensor *build_binary(MPSGraph *g, ops::OpType op,
                                    MPSGraphTensor *a, MPSGraphTensor *b) {
    using ops::OpType;
    switch (op) {
    case OpType::Add: return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Subtract: return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Multiply: return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Divide: return [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Maximum: return [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Minimum: return [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Power: return [g powerWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Modulo: return [g moduloWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Equal: return [g equalWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::NotEqual: return [g notEqualWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Less: return [g lessThanWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::LessEqual: return [g lessThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Greater: return [g greaterThanWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::GreaterEqual: return [g greaterThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::LogicalAnd: {
        auto *ba = [g castTensor:a toType:MPSDataTypeBool name:nil];
        auto *bb = [g castTensor:b toType:MPSDataTypeBool name:nil];
        return [g logicalANDWithPrimaryTensor:ba secondaryTensor:bb name:nil];
    }
    case OpType::LogicalOr: {
        auto *ba = [g castTensor:a toType:MPSDataTypeBool name:nil];
        auto *bb = [g castTensor:b toType:MPSDataTypeBool name:nil];
        return [g logicalORWithPrimaryTensor:ba secondaryTensor:bb name:nil];
    }
    case OpType::LogicalXor: {
        auto *ba = [g castTensor:a toType:MPSDataTypeBool name:nil];
        auto *bb = [g castTensor:b toType:MPSDataTypeBool name:nil];
        return [g logicalXORWithPrimaryTensor:ba secondaryTensor:bb name:nil];
    }
    case OpType::BitwiseAnd: return [g bitwiseANDWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::BitwiseOr: return [g bitwiseORWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::BitwiseXor: return [g bitwiseXORWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::LeftShift: return [g bitwiseLeftShiftWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::RightShift: return [g bitwiseRightShiftWithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Atan2: return [g atan2WithPrimaryTensor:a secondaryTensor:b name:nil];
    case OpType::Hypot: {
        // hypot(a, b) = sqrt(a*a + b*b)
        auto *a2 = [g squareWithTensor:a name:nil];
        auto *b2 = [g squareWithTensor:b name:nil];
        auto *sum = [g additionWithPrimaryTensor:a2 secondaryTensor:b2 name:nil];
        return [g squareRootWithTensor:sum name:nil];
    }
    default: return nil;
    }
}

// MatMul
static MPSGraphTensor *build_matmul(MPSGraph *g, MPSGraphTensor *a,
                                    MPSGraphTensor *b, bool transpose_a,
                                    bool transpose_b) {
    // Handle transposes via MPSGraph transpose ops
    if (transpose_a) {
        NSUInteger ndim = a.shape.count;
        a = [g transposeTensor:a dimension:(ndim - 2) withDimension:(ndim - 1) name:nil];
    }
    if (transpose_b) {
        NSUInteger ndim = b.shape.count;
        b = [g transposeTensor:b dimension:(ndim - 2) withDimension:(ndim - 1) name:nil];
    }
    return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
}

// Softmax
static MPSGraphTensor *build_softmax(MPSGraph *g, MPSGraphTensor *input,
                                     int axis) {
    return [g softMaxWithTensor:input axis:axis name:nil];
}

// LogSoftmax
static MPSGraphTensor *build_log_softmax(MPSGraph *g, MPSGraphTensor *input,
                                         int axis) {
    // logSoftmax = log(softmax(x)) — MPSGraph doesn't have a direct API,
    // but log(softmax) is numerically stable via the identity:
    // log_softmax(x) = x - log(sum(exp(x)))
    auto *sm = [g softMaxWithTensor:input axis:axis name:nil];
    return [g logarithmWithTensor:sm name:nil];
}

// LayerNorm: normalize(input) * weight + bias
static MPSGraphTensor *build_layernorm(MPSGraph *g, MPSGraphTensor *input,
                                       MPSGraphTensor *weight,
                                       MPSGraphTensor *bias, int axis,
                                       float eps) {
    // Normalize axis
    NSUInteger ndim = input.shape.count;
    int norm_axis = axis;
    if (norm_axis < 0)
        norm_axis += static_cast<int>(ndim);

    // Compute mean along axis
    auto *mean = [g meanOfTensor:input axes:@[@(norm_axis)] name:nil];

    // Compute variance: E[(x-mean)^2]
    auto *centered = [g subtractionWithPrimaryTensor:input secondaryTensor:mean name:nil];
    auto *sq = [g squareWithTensor:centered name:nil];
    auto *var = [g meanOfTensor:sq axes:@[@(norm_axis)] name:nil];

    // Normalize: (x - mean) / sqrt(var + eps)
    auto *eps_t = [g constantWithScalar:eps dataType:input.dataType];
    auto *var_eps = [g additionWithPrimaryTensor:var secondaryTensor:eps_t name:nil];
    auto *inv_std = [g reciprocalWithTensor:[g squareRootWithTensor:var_eps name:nil] name:nil];
    auto *normalized = [g multiplicationWithPrimaryTensor:centered secondaryTensor:inv_std name:nil];

    // Scale and shift
    auto *scaled = [g multiplicationWithPrimaryTensor:normalized secondaryTensor:weight name:nil];
    return [g additionWithPrimaryTensor:scaled secondaryTensor:bias name:nil];
}

// BatchNorm1D: (input - running_mean) / sqrt(running_var + eps) * weight + bias
// Upcasts to f32 for numerical stability (matching PyTorch), casts back to
// input dtype. All within a single fused MPSGraph — no eager materialization.
static MPSGraphTensor *build_batchnorm(MPSGraph *g, MPSGraphTensor *input,
                                       MPSGraphTensor *weight,
                                       MPSGraphTensor *bias,
                                       MPSGraphTensor *running_mean,
                                       MPSGraphTensor *running_var,
                                       float eps) {
    MPSDataType orig_dtype = input.dataType;
    // Upcast to f32 for numerical stability
    if (orig_dtype != MPSDataTypeFloat32) {
        input = [g castTensor:input toType:MPSDataTypeFloat32 name:nil];
        running_mean = [g castTensor:running_mean toType:MPSDataTypeFloat32 name:nil];
        running_var = [g castTensor:running_var toType:MPSDataTypeFloat32 name:nil];
        if (weight) weight = [g castTensor:weight toType:MPSDataTypeFloat32 name:nil];
        if (bias) bias = [g castTensor:bias toType:MPSDataTypeFloat32 name:nil];
    }
    auto *eps_t = [g constantWithScalar:eps dataType:MPSDataTypeFloat32];
    auto *var_eps = [g additionWithPrimaryTensor:running_var secondaryTensor:eps_t name:nil];
    auto *inv_std = [g reciprocalWithTensor:[g squareRootWithTensor:var_eps name:nil] name:nil];
    auto *centered = [g subtractionWithPrimaryTensor:input secondaryTensor:running_mean name:nil];
    auto *result = [g multiplicationWithPrimaryTensor:centered secondaryTensor:inv_std name:nil];
    if (weight) result = [g multiplicationWithPrimaryTensor:result secondaryTensor:weight name:nil];
    if (bias) result = [g additionWithPrimaryTensor:result secondaryTensor:bias name:nil];
    // Cast back to input dtype
    if (orig_dtype != MPSDataTypeFloat32) {
        result = [g castTensor:result toType:orig_dtype name:nil];
    }
    return result;
}

// Conv1D via Conv2D (MPSGraph only has 2D convolution)
static MPSGraphTensor *build_conv1d(MPSGraph *g, MPSGraphTensor *input,
                                    MPSGraphTensor *weight,
                                    MPSGraphTensor *bias,
                                    const ConvParams &params) {
    // input: (N, C_in, L) → unsqueeze to (N, C_in, 1, L)
    auto *input_4d = [g expandDimsOfTensor:input axis:2 name:nil];

    // weight: (C_out, C_in/groups, K) → unsqueeze to (C_out, C_in/groups, 1, K)
    auto *weight_4d = [g expandDimsOfTensor:weight axis:2 name:nil];

    MPSGraphConvolution2DOpDescriptor *desc = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:params.stride[0]
                      strideInY:1
              dilationRateInX:params.dilation[0]
              dilationRateInY:1
                        groups:params.groups
                 paddingLeft:params.padding[0]
                paddingRight:params.padding[0]
                  paddingTop:0
               paddingBottom:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                  dataLayout:MPSGraphTensorNamedDataLayoutNCHW
              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    auto *conv = [g convolution2DWithSourceTensor:input_4d
                                    weightsTensor:weight_4d
                                       descriptor:desc
                                             name:nil];

    // Squeeze back to 3D: (N, C_out, 1, L_out) → (N, C_out, L_out)
    conv = [g squeezeTensor:conv axis:2 name:nil];

    // Add bias: reshape (C_out,) → (1, C_out, 1) for NCHW broadcast
    if (bias) {
        auto *bias_3d = [g reshapeTensor:bias
                               withShape:@[@1, @-1, @1]
                                    name:nil];
        conv = [g additionWithPrimaryTensor:conv secondaryTensor:bias_3d name:nil];
    }
    return conv;
}

// Conv2D
static MPSGraphTensor *build_conv2d(MPSGraph *g, MPSGraphTensor *input,
                                    MPSGraphTensor *weight,
                                    MPSGraphTensor *bias,
                                    const ConvParams &params) {
    int sx = params.stride.size() > 1 ? params.stride[1] : params.stride[0];
    int sy = params.stride[0];
    int dx = params.dilation.size() > 1 ? params.dilation[1] : params.dilation[0];
    int dy = params.dilation[0];
    int pl = params.padding.size() > 1 ? params.padding[1] : params.padding[0];
    int pt = params.padding[0];

    MPSGraphConvolution2DOpDescriptor *desc = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:sx
                      strideInY:sy
              dilationRateInX:dx
              dilationRateInY:dy
                        groups:params.groups
                 paddingLeft:pl
                paddingRight:pl
                  paddingTop:pt
               paddingBottom:pt
                paddingStyle:MPSGraphPaddingStyleExplicit
                  dataLayout:MPSGraphTensorNamedDataLayoutNCHW
              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    auto *conv = [g convolution2DWithSourceTensor:input
                                    weightsTensor:weight
                                       descriptor:desc
                                             name:nil];
    // Add bias: reshape (C_out,) → (1, C_out, 1, 1) for NCHW broadcast
    if (bias) {
        auto *bias_4d = [g reshapeTensor:bias
                               withShape:@[@1, @-1, @1, @1]
                                    name:nil];
        conv = [g additionWithPrimaryTensor:conv secondaryTensor:bias_4d name:nil];
    }
    return conv;
}

// Reshape
static MPSGraphTensor *build_reshape(MPSGraph *g, MPSGraphTensor *input,
                                     const Shape &new_shape) {
    return [g reshapeTensor:input withShape:toMPSShape(new_shape) name:nil];
}

// Transpose (arbitrary axis permutation)
static MPSGraphTensor *build_transpose(MPSGraph *g, MPSGraphTensor *input,
                                       const std::vector<int> &axes) {
    // MPSGraph's transposeTensor only swaps two dims at a time.
    // For arbitrary permutations, we chain swaps.
    // Build the permutation as a sequence of transpositions.
    MPSGraphTensor *result = input;

    // Convert the permutation to a sequence of swaps using selection sort
    std::vector<int> perm = axes;
    for (size_t i = 0; i < perm.size(); ++i) {
        int target = perm[i] < 0 ? perm[i] + static_cast<int>(perm.size())
                                 : perm[i];
        if (static_cast<size_t>(target) != i) {
            // Find where 'i' currently is
            size_t j = i;
            for (size_t k = i + 1; k < perm.size(); ++k) {
                int pk = perm[k] < 0 ? perm[k] + static_cast<int>(perm.size())
                                     : perm[k];
                if (static_cast<size_t>(pk) == i) {
                    j = k;
                    break;
                }
            }
            if (j != i) {
                result = [g transposeTensor:result
                                  dimension:i
                              withDimension:j
                                       name:nil];
                std::swap(perm[i], perm[j]);
            }
        }
    }
    return result;
}

// Pad
static MPSGraphTensor *build_pad(MPSGraph *g, MPSGraphTensor *input,
                                 const PadParams &params) {
    NSMutableArray<NSNumber *> *left =
        [NSMutableArray arrayWithCapacity:params.pad_widths.size()];
    NSMutableArray<NSNumber *> *right =
        [NSMutableArray arrayWithCapacity:params.pad_widths.size()];
    for (const auto &[l, r] : params.pad_widths) {
        [left addObject:@(static_cast<NSInteger>(l))];
        [right addObject:@(static_cast<NSInteger>(r))];
    }

    return [g padTensor:input
         withPaddingMode:MPSGraphPaddingModeConstant
             leftPadding:left
            rightPadding:right
           constantValue:params.value
                    name:nil];
}

// Slice
static MPSGraphTensor *build_slice(MPSGraph *g, MPSGraphTensor *input,
                                   const SliceParams &params) {
    NSMutableArray<NSNumber *> *starts =
        [NSMutableArray arrayWithCapacity:params.starts.size()];
    NSMutableArray<NSNumber *> *ends =
        [NSMutableArray arrayWithCapacity:params.ends.size()];
    NSMutableArray<NSNumber *> *strides =
        [NSMutableArray arrayWithCapacity:params.strides.size()];
    for (auto s : params.starts)
        [starts addObject:@(static_cast<NSInteger>(s))];
    for (auto e : params.ends)
        [ends addObject:@(static_cast<NSInteger>(e))];
    for (auto st : params.strides)
        [strides addObject:@(static_cast<NSInteger>(st))];

    return [g sliceTensor:input starts:starts ends:ends strides:strides name:nil];
}

// MaskedFill: where(mask, fill_value, input)
static MPSGraphTensor *build_masked_fill(MPSGraph *g, MPSGraphTensor *input,
                                         MPSGraphTensor *mask, float value) {
    auto *mask_bool = [g castTensor:mask toType:MPSDataTypeBool name:nil];
    auto *fill_val = [g constantWithScalar:value dataType:input.dataType];
    return [g selectWithPredicateTensor:mask_bool
                    truePredicateTensor:fill_val
                   falsePredicateTensor:input
                                   name:nil];
}

// GLU: split along dim, return first_half * sigmoid(second_half)
static MPSGraphTensor *build_glu(MPSGraph *g, MPSGraphTensor *input, int dim) {
    NSUInteger ndim = input.shape.count;
    int norm_dim = dim < 0 ? dim + static_cast<int>(ndim) : dim;

    NSInteger half_size = [input.shape[norm_dim] integerValue] / 2;
    auto *half_shape = [NSNumber numberWithInteger:half_size];

    // Split using slice: first_half and second_half
    NSMutableArray<NSNumber *> *starts1 = [NSMutableArray array];
    NSMutableArray<NSNumber *> *ends1 = [NSMutableArray array];
    NSMutableArray<NSNumber *> *strides1 = [NSMutableArray array];
    NSMutableArray<NSNumber *> *starts2 = [NSMutableArray array];
    NSMutableArray<NSNumber *> *ends2 = [NSMutableArray array];

    for (NSUInteger i = 0; i < ndim; ++i) {
        [strides1 addObject:@1];
        if (static_cast<int>(i) == norm_dim) {
            [starts1 addObject:@0];
            [ends1 addObject:half_shape];
            [starts2 addObject:half_shape];
            [ends2 addObject:input.shape[i]];
        } else {
            [starts1 addObject:@0];
            [ends1 addObject:input.shape[i]];
            [starts2 addObject:@0];
            [ends2 addObject:input.shape[i]];
        }
    }

    auto *first = [g sliceTensor:input starts:starts1 ends:ends1 strides:strides1 name:nil];
    auto *second = [g sliceTensor:input starts:starts2 ends:ends2 strides:strides1 name:nil];
    auto *sig = [g sigmoidWithTensor:second name:nil];
    return [g multiplicationWithPrimaryTensor:first secondaryTensor:sig name:nil];
}

// Reduction ops
static MPSGraphTensor *build_reduction(MPSGraph *g, ops::OpType op,
                                       MPSGraphTensor *input,
                                       const ReductionParams &params) {
    int ndim = static_cast<int>(input.shape.count);

    // Build axes array — empty means reduce ALL axes
    NSMutableArray<NSNumber *> *axes = [NSMutableArray array];
    if (params.axes.empty()) {
        for (int i = 0; i < ndim; ++i)
            [axes addObject:@(i)];
    } else {
        for (int ax : params.axes) {
            int norm = ax < 0 ? ax + ndim : ax;
            [axes addObject:@(norm)];
        }
    }

    MPSGraphTensor *result = nil;
    switch (op) {
    case ops::OpType::Sum:
        result = [g reductionSumWithTensor:input axes:axes name:nil];
        break;
    case ops::OpType::Mean:
        result = [g meanOfTensor:input axes:axes name:nil];
        break;
    case ops::OpType::Max:
        result = [g reductionMaximumWithTensor:input axes:axes name:nil];
        break;
    case ops::OpType::Min:
        result = [g reductionMinimumWithTensor:input axes:axes name:nil];
        break;
    case ops::OpType::Prod:
        result = [g reductionProductWithTensor:input axes:axes name:nil];
        break;
    case ops::OpType::Any: {
        // any = sum(cast_bool) > 0
        auto *b = [g castTensor:input toType:MPSDataTypeBool name:nil];
        auto *s = [g castTensor:b toType:MPSDataTypeInt32 name:nil];
        auto *sum = [g reductionSumWithTensor:s axes:axes name:nil];
        auto *zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
        result = [g greaterThanWithPrimaryTensor:sum secondaryTensor:zero name:nil];
        break;
    }
    case ops::OpType::All: {
        // all = min(cast_bool) > 0
        auto *b = [g castTensor:input toType:MPSDataTypeBool name:nil];
        auto *s = [g castTensor:b toType:MPSDataTypeInt32 name:nil];
        auto *m = [g reductionMinimumWithTensor:s axes:axes name:nil];
        auto *zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
        result = [g greaterThanWithPrimaryTensor:m secondaryTensor:zero name:nil];
        break;
    }
    case ops::OpType::ArgMax: {
        // MPSGraph argmax takes a single axis; for full reduction, flatten first
        int axis;
        MPSGraphTensor *arg_input = input;
        if (params.axes.empty()) {
            // Full reduction: flatten to 1D, argmax on axis 0
            auto *neg1 = @[@-1];
            arg_input = [g reshapeTensor:input withShape:neg1 name:nil];
            axis = 0;
        } else {
            axis = params.axes[0] < 0 ? params.axes[0] + ndim : params.axes[0];
        }
        result = [g reductionArgMaximumWithTensor:arg_input axis:(NSInteger)axis name:nil];
        result = [g castTensor:result toType:MPSDataTypeInt64 name:nil];
        break;
    }
    case ops::OpType::ArgMin: {
        int axis;
        MPSGraphTensor *arg_input = input;
        if (params.axes.empty()) {
            auto *neg1 = @[@-1];
            arg_input = [g reshapeTensor:input withShape:neg1 name:nil];
            axis = 0;
        } else {
            axis = params.axes[0] < 0 ? params.axes[0] + ndim : params.axes[0];
        }
        result = [g reductionArgMinimumWithTensor:arg_input axis:(NSInteger)axis name:nil];
        result = [g castTensor:result toType:MPSDataTypeInt64 name:nil];
        break;
    }
    default:
        return nil;
    }

    if (!params.keep_dims && result) {
        // Squeeze reduced dimensions
        std::set<int> reduced_set;
        if (params.axes.empty()) {
            for (int i = 0; i < ndim; ++i)
                reduced_set.insert(i);
        } else {
            for (int ax : params.axes) {
                int norm = ax < 0 ? ax + ndim : ax;
                reduced_set.insert(norm);
            }
        }
        NSMutableArray<NSNumber *> *new_shape = [NSMutableArray array];
        for (int i = 0; i < ndim; ++i) {
            if (!reduced_set.count(i))
                [new_shape addObject:input.shape[i]];
        }
        if (new_shape.count == 0)
            [new_shape addObject:@1]; // scalar → shape {1}
        result = [g reshapeTensor:result withShape:new_shape name:nil];
    }
    return result;
}

// ============================================================================
// Compiled GPU Graph cache
// ============================================================================

struct CachedGPUFullGraph {
    MPSGraph *graph;
    NSArray<MPSGraphTensor *> *input_placeholders; // leaf inputs
    MPSGraphTensor *output;
};

static std::unordered_map<GraphSignature, CachedGPUFullGraph, GraphSignatureHash> g_gpu_graph_cache;
static std::mutex g_gpu_graph_cache_mutex;
static constexpr size_t MAX_GPU_GRAPH_CACHE = 128;

// ============================================================================
// Main compilation function: compile a lazy DAG → MPSGraph
// ============================================================================

// Materialize a GPU lazy graph as a single MPSGraph.
// Called from GraphRegistry::materialize() when target_device == GPU.
void materialize_gpu_graph(GraphNode *root) {
    @autoreleasepool {
        // 1. Walk the subgraph and materialize shared nodes first
        {
            std::vector<GraphNode *> shared_topo;
            std::unordered_set<GraphNode *> visited;
            std::vector<std::pair<GraphNode *, size_t>> stk;
            stk.push_back({root, 0});

            while (!stk.empty()) {
                auto &[n, idx] = stk.back();
                if (visited.count(n)) {
                    stk.pop_back();
                    continue;
                }
                if (idx < n->inputs.size()) {
                    GraphNode *child = n->inputs[idx].get();
                    idx++;
                    if (!visited.count(child) && !child->is_constant &&
                        !child->is_materialized_) {
                        stk.push_back({child, 0});
                    }
                } else {
                    visited.insert(n);
                    if (n != root && !n->is_constant && !n->is_materialized_ &&
                        n->ref_count > 1) {
                        shared_topo.push_back(n);
                    }
                    stk.pop_back();
                }
            }

            for (auto *shared : shared_topo) {
                if (!shared->is_materialized_) {
                    // Recursively materialize shared sub-graphs
                    materialize_gpu_graph(shared);
                }
            }
        }

        // 2. Topological sort of the remaining DAG
        auto sorted = topo_sort(root);

        // 3. Collect leaf input tensors
        auto leaf_tensors = collect_leaf_tensors(sorted);

        // 4. Compute structural signature for caching
        auto sig = compute_signature(root);

        // 5. Look up or compile the MPSGraph
        CachedGPUFullGraph cached{};
        bool cache_hit = false;
        {
            std::lock_guard<std::mutex> lock(g_gpu_graph_cache_mutex);
            auto it = g_gpu_graph_cache.find(sig);
            if (it != g_gpu_graph_cache.end()) {
                cached = it->second;
                cache_hit = true;
            }
        }

        if (!cache_hit) {
            // Build a new MPSGraph from the lazy DAG
            MPSGraph *graph = [[MPSGraph alloc] init];

            // Map from GraphNode* to MPSGraphTensor*
            std::unordered_map<const GraphNode *, MPSGraphTensor *> node_map;

            // Create placeholders for leaf inputs
            NSMutableArray<MPSGraphTensor *> *placeholders = [NSMutableArray array];
            std::unordered_set<const GraphNode *> leaf_seen;
            // First pass: create placeholders for all leaves
            for (const auto *node : sorted) {
                if (node->is_constant || node->is_materialized_) {
                    // Skip empty tensors (e.g., no-bias placeholder)
                    if (node->output_shape.empty() && !node->constant_storage)
                        continue;
                    if (leaf_seen.insert(node).second) {
                        MPSGraphTensor *ph =
                            [graph placeholderWithShape:toMPSShape(node->output_shape)
                                              dataType:toMPS(node->output_dtype)
                                                  name:nil];
                        [placeholders addObject:ph];
                        node_map[node] = ph;
                    }
                }
            }

            // Second pass: build graph ops for non-leaf nodes
            for (const auto *node : sorted) {
                if (node->is_constant || node->is_materialized_)
                    continue; // already handled as placeholders

                // Resolve inputs — returns nil for missing or empty tensors
                auto get_input = [&](size_t input_idx) -> MPSGraphTensor * {
                    if (input_idx >= node->inputs.size())
                        return nil;
                    auto *inp_node = node->inputs[input_idx].get();
                    // Empty tensor (e.g., no-bias Conv): no storage, empty shape
                    if (inp_node->output_shape.empty() && !inp_node->constant_storage)
                        return nil;
                    auto it = node_map.find(inp_node);
                    return (it != node_map.end()) ? it->second : nil;
                };

                MPSGraphTensor *result = nil;
                ops::OpType op = node->op_type;

                if (is_unary_op(op)) {
                    result = build_unary(graph, op, get_input(0));
                } else if (is_binary_op(op)) {
                    result = build_binary(graph, op, get_input(0), get_input(1));
                } else if (op == ops::OpType::MatMul || op == ops::OpType::BatchMatMul) {
                    const auto &mp = get_params<MatMulParams>(node->params);
                    result = build_matmul(graph, get_input(0), get_input(1),
                                          mp.transpose_a, mp.transpose_b);
                } else if (op == ops::OpType::Softmax) {
                    const auto &ap = get_params<ActivationParams>(node->params);
                    result = build_softmax(graph, get_input(0), ap.axis);
                } else if (op == ops::OpType::LogSoftmax) {
                    const auto &ap = get_params<ActivationParams>(node->params);
                    result = build_log_softmax(graph, get_input(0), ap.axis);
                } else if (op == ops::OpType::LayerNorm) {
                    const auto &np = get_params<NormParams>(node->params);
                    result = build_layernorm(graph, get_input(0), get_input(1),
                                             get_input(2), np.axis, np.eps);
                } else if (op == ops::OpType::BatchNorm1D) {
                    const auto &np = get_params<NormParams>(node->params);
                    result = build_batchnorm(graph, get_input(0), get_input(1),
                                             get_input(2), get_input(3),
                                             get_input(4), np.eps);
                } else if (op == ops::OpType::Conv1D) {
                    const auto &cp = get_params<ConvParams>(node->params);
                    MPSGraphTensor *bias_t = (node->inputs.size() > 2) ? get_input(2) : nil;
                    result = build_conv1d(graph, get_input(0), get_input(1),
                                          bias_t, cp);
                } else if (op == ops::OpType::Conv2D) {
                    const auto &cp = get_params<ConvParams>(node->params);
                    MPSGraphTensor *bias_t = (node->inputs.size() > 2) ? get_input(2) : nil;
                    result = build_conv2d(graph, get_input(0), get_input(1),
                                          bias_t, cp);
                } else if (op == ops::OpType::Reshape) {
                    const auto &rp = get_params<ReshapeParams>(node->params);
                    result = build_reshape(graph, get_input(0), rp.new_shape);
                } else if (op == ops::OpType::Transpose) {
                    const auto &tp = get_params<TransposeParams>(node->params);
                    result = build_transpose(graph, get_input(0), tp.axes);
                } else if (op == ops::OpType::Pad) {
                    const auto &pp = get_params<PadParams>(node->params);
                    result = build_pad(graph, get_input(0), pp);
                } else if (op == ops::OpType::Slice) {
                    const auto &sp = get_params<SliceParams>(node->params);
                    result = build_slice(graph, get_input(0), sp);
                } else if (op == ops::OpType::MaskedFill) {
                    const auto &mf = get_params<MaskedFillParams>(node->params);
                    result = build_masked_fill(graph, get_input(0), get_input(1),
                                               mf.value);
                } else if (op == ops::OpType::GLU) {
                    const auto &ap = get_params<ActivationParams>(node->params);
                    result = build_glu(graph, get_input(0), ap.axis);
                } else if (is_reduction_op(op)) {
                    const auto &rp = get_params<ReductionParams>(node->params);
                    result = build_reduction(graph, op, get_input(0), rp);
                } else {
                    // Unsupported op — fall back to eager materialization
                    // This shouldn't happen if all used ops are captured
                    throw RuntimeError(
                        "GPU graph compiler: unsupported op type " +
                        std::to_string(static_cast<int>(op)));
                }

                if (!result) {
                    throw RuntimeError(
                        "GPU graph compiler: failed to build MPSGraph node for op " +
                        std::to_string(static_cast<int>(op)));
                }

                node_map[node] = result;
            }

            // Find the root output
            auto root_it = node_map.find(root);
            if (root_it == node_map.end()) {
                throw RuntimeError("GPU graph compiler: root node not in node_map");
            }

            cached.graph = graph;
            cached.input_placeholders = [placeholders copy];
            cached.output = root_it->second;

            // Store in cache
            {
                std::lock_guard<std::mutex> lock(g_gpu_graph_cache_mutex);
                if (g_gpu_graph_cache.size() >= MAX_GPU_GRAPH_CACHE)
                    g_gpu_graph_cache.clear(); // simple eviction
                g_gpu_graph_cache[sig] = cached;
            }
        }

        // 6. Ensure all leaf inputs are on GPU and contiguous
        for (auto &t : leaf_tensors) {
            if (t.device() != Device::GPU)
                t = t.to(Device::GPU);
            if (!t.is_contiguous())
                t = t.ascontiguousarray(); // eager — this is a materialized tensor
        }

        // 7. Allocate output tensor
        Tensor output(root->output_shape, root->output_dtype, Device::GPU);

        // 8. Build feeds dictionary
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionaryWithCapacity:cached.input_placeholders.count];
        for (NSUInteger i = 0; i < cached.input_placeholders.count && i < leaf_tensors.size(); ++i) {
            feeds[cached.input_placeholders[i]] = makeTensorData(leaf_tensors[i]);
        }

        // 9. Build targets
        MPSGraphTensorData *out_data = makeTensorData(output);
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *targets =
            @{cached.output : out_data};

        // 10. Encode and execute
        encodeMPSGraphAsync(cached.graph, feeds, targets);

        // 11. Store result in node
        root->cached_result_ = output.storage();
        root->cached_shape_ = output.shape();
        root->cached_strides_ = output.strides();
        root->is_materialized_ = true;

        // 12. Decrement ref counts
        for (auto &inp : root->inputs) {
            if (inp && !inp->is_constant && inp->ref_count > 0) {
                inp->ref_count--;
            }
        }
    }
}

} // namespace graph
} // namespace axiom
