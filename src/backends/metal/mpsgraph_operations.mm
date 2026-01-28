#import "mpsgraph_operations.hpp"
#import "metal_common.hpp"
#import "metal_storage.hpp"
#import "axiom/error.hpp"
#import "axiom/shape.hpp"
#import "axiom/dtype.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <set>
#include <numeric>
#include <map>

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// Constants and Structs
// ============================================================================

constexpr int kMaxDims = 8;

// Must match GatherStridedParams in kernels.metal
struct GatherStridedParams {
    uint32_t ndim;
    uint32_t numel;
    uint32_t offset;      // Byte offset into source buffer
    uint32_t itemsize;    // Size of each element in bytes
    uint32_t shape[kMaxDims];
    uint32_t src_strides[kMaxDims];  // Strides in ELEMENTS (not bytes)
};

// ============================================================================
// GPU Gather Kernel Infrastructure
// ============================================================================

// Cache for gather kernel pipeline states
static std::map<DType, id<MTLComputePipelineState>> g_gather_pipeline_states;

static id<MTLComputePipelineState> getGatherPipelineState(DType dtype) {
    auto it = g_gather_pipeline_states.find(dtype);
    if (it != g_gather_pipeline_states.end()) {
        return it->second;
    }
    
    std::string type_suffix;
    switch (dtype) {
        case DType::Float32: type_suffix = "float"; break;
        case DType::Float16: type_suffix = "half"; break;
        case DType::Int32:   type_suffix = "int"; break;
        case DType::UInt32:  type_suffix = "uint"; break;
        case DType::Int16:   type_suffix = "short"; break;
        case DType::UInt16:  type_suffix = "ushort"; break;
        case DType::Int8:    type_suffix = "char"; break;
        case DType::UInt8:   type_suffix = "uchar"; break;
        case DType::Int64:   type_suffix = "long"; break;
        case DType::UInt64:  type_suffix = "ulong"; break;
        default:
            throw TypeError::unsupported_dtype(dtype_name(dtype), "GPU gather kernel");
    }
    
    std::string kernel_name = "gather_strided_" + type_suffix;
    
    id<MTLDevice> device = (__bridge id<MTLDevice>)MetalContext::instance().device();
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)get_default_library();
    
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    if (!function) {
        throw DeviceError("Failed to find Metal kernel: " + kernel_name);
    }
    
    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline_state) {
        throw DeviceError("Failed to create Metal pipeline state for kernel: " + kernel_name);
    }
    
    g_gather_pipeline_states[dtype] = pipeline_state;
    return pipeline_state;
}

// Copies non-contiguous GPU tensor to contiguous using GPU gather kernel
static Tensor makeContiguousViaGatherKernel(const Tensor& tensor) {
    if (tensor.is_contiguous()) {
        return tensor;
    }
    
    // Create output tensor
    Tensor result(tensor.shape(), tensor.dtype(), Device::GPU);
    
    // Get pipeline state for this dtype
    id<MTLComputePipelineState> pipeline_state = getGatherPipelineState(tensor.dtype());
    
    // Get storage buffers
    auto* src_storage = static_cast<const MetalStorage*>(tensor.storage().get());
    auto* dst_storage = static_cast<MetalStorage*>(result.storage().get());
    
    id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src_storage->buffer();
    id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst_storage->buffer();
    
    // Prepare kernel parameters
    GatherStridedParams params;
    params.ndim = static_cast<uint32_t>(tensor.ndim());
    params.numel = static_cast<uint32_t>(tensor.size());
    params.offset = 0;  // offset is applied via buffer offset
    params.itemsize = static_cast<uint32_t>(tensor.itemsize());
    
    std::fill_n(params.shape, kMaxDims, 0);
    std::fill_n(params.src_strides, kMaxDims, 0);
    
    for (size_t i = 0; i < tensor.ndim(); ++i) {
        params.shape[i] = static_cast<uint32_t>(tensor.shape()[i]);
        // Strides in elements, not bytes
        params.src_strides[i] = static_cast<uint32_t>(tensor.strides()[i] / tensor.itemsize());
    }
    
    // Create command buffer and encoder
    id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue();
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];
    
    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:src_buffer offset:tensor.offset() atIndex:0];
    [command_encoder setBuffer:dst_buffer offset:0 atIndex:1];
    [command_encoder setBytes:&params length:sizeof(params) atIndex:2];
    
    // Dispatch threads - one thread per output element
    NSUInteger width = params.numel;
    NSUInteger max_threads = [pipeline_state maxTotalThreadsPerThreadgroup];
    MTLSize threads_per_group = MTLSizeMake(std::min(width, max_threads), 1, 1);
    MTLSize thread_groups = MTLSizeMake((width + threads_per_group.width - 1) / threads_per_group.width, 1, 1);
    
    [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
    [command_encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSLog(@"Gather kernel error: %@", [command_buffer error]);
        throw DeviceError("GPU gather kernel execution failed");
    }
    
    return result;
}

// Helper to ensure a GPU tensor is contiguous (uses GPU gather kernel)
static inline Tensor ensureContiguous(const Tensor& tensor) {
    return makeContiguousViaGatherKernel(tensor);
}

// ============================================================================
// Helper Functions
// ============================================================================

static MPSDataType getMPSDataType(DType dtype) {
    switch (dtype) {
        case DType::Float32:    return MPSDataTypeFloat32;
        case DType::Float16:    return MPSDataTypeFloat16;
        case DType::Int32:      return MPSDataTypeInt32;
        case DType::Int64:      return MPSDataTypeInt64;
        case DType::Int16:      return MPSDataTypeInt16;
        case DType::Int8:       return MPSDataTypeInt8;
        case DType::UInt8:      return MPSDataTypeUInt8;
        case DType::UInt16:     return MPSDataTypeUInt16;
        case DType::UInt32:     return MPSDataTypeUInt32;
        case DType::UInt64:     return MPSDataTypeUInt64;
        case DType::Bool:       return MPSDataTypeBool;
        default:
            throw TypeError::unsupported_dtype(dtype_name(dtype), "MPSGraph");
    }
}

static MPSShape* getMPSShape(const Shape& shape) {
    NSMutableArray<NSNumber*>* mps_shape = [NSMutableArray arrayWithCapacity:shape.size()];
    for (size_t i = 0; i < shape.size(); ++i) {
        [mps_shape addObject:@(shape[i])];
    }
    return mps_shape;
}

static MPSGraphTensor* createPlaceholder(MPSGraph* graph, const Tensor& tensor) {
    MPSDataType dtype = getMPSDataType(tensor.dtype());
    MPSShape* shape = getMPSShape(tensor.shape());
    return [graph placeholderWithShape:shape dataType:dtype name:nil];
}

static MPSGraphTensorData* createTensorData(const Tensor& tensor) {
    if (tensor.device() != Device::GPU) {
        throw DeviceError("MPSGraph operations require GPU tensors");
    }
    
    auto* storage = static_cast<const MetalStorage*>(tensor.storage().get());
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage->buffer();
    
    MPSDataType dtype = getMPSDataType(tensor.dtype());
    MPSShape* shape = getMPSShape(tensor.shape());
    
    // Create tensor data with proper offset (ARC handles memory management)
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                   shape:shape
                                                dataType:dtype];
}

static MPSGraphTensor* castToCommonType(MPSGraph* graph, MPSGraphTensor* tensor, 
                                        DType target_dtype) {
    MPSDataType target_mps_dtype = getMPSDataType(target_dtype);
    if (tensor.dataType != target_mps_dtype) {
        return [graph castTensor:tensor toType:target_mps_dtype name:nil];
    }
    return tensor;
}

// ============================================================================
// Core Execution Helper
// ============================================================================

static Tensor executeMPSGraphBinaryOp(const Tensor& lhs_raw, const Tensor& rhs_raw,
                                      DType output_dtype,
                                      MPSGraphBinaryOpBlock op_block) {
    @autoreleasepool {
        // Make non-contiguous tensors contiguous using GPU gather kernel
        Tensor lhs = ensureContiguous(lhs_raw);
        Tensor rhs = ensureContiguous(rhs_raw);
        
        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create placeholders
        MPSGraphTensor* lhs_placeholder = createPlaceholder(graph, lhs);
        MPSGraphTensor* rhs_placeholder = createPlaceholder(graph, rhs);
        
        // Type promotion if needed
        DType common_dtype = ops::promote_types(lhs.dtype(), rhs.dtype());
        MPSGraphTensor* lhs_cast = castToCommonType(graph, lhs_placeholder, common_dtype);
        MPSGraphTensor* rhs_cast = castToCommonType(graph, rhs_placeholder, common_dtype);
        
        // Execute the operation
        MPSGraphTensor* result_tensor = op_block(graph, lhs_cast, rhs_cast);
        
        // Cast to output dtype if needed
        result_tensor = castToCommonType(graph, result_tensor, output_dtype);
        
        // Infer output shape (MPSGraph handles broadcasting)
        Shape output_shape = ShapeUtils::broadcast_shape(lhs.shape(), rhs.shape());
        
        // Create output tensor
        Tensor output(output_shape, output_dtype, Device::GPU);
        
        // Create tensor data
        MPSGraphTensorData* lhs_data = createTensorData(lhs);
        MPSGraphTensorData* rhs_data = createTensorData(rhs);
        MPSGraphTensorData* output_data = createTensorData(output);
        
        // Create feeds dictionary
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            lhs_placeholder: lhs_data,
            rhs_placeholder: rhs_data
        };
        
        // Create targets dictionary  
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: output_data
        };
        
        // Execute the graph using synchronous API
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return output;
    }
}

static Tensor executeMPSGraphUnaryOp(const Tensor& input_raw,
                                     DType output_dtype,
                                     MPSGraphUnaryOpBlock op_block) {
    @autoreleasepool {
        // Make non-contiguous tensor contiguous using GPU gather kernel
        Tensor input = ensureContiguous(input_raw);
        
        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create placeholder
        MPSGraphTensor* input_placeholder = createPlaceholder(graph, input);
        
        // Execute the operation
        MPSGraphTensor* result_tensor = op_block(graph, input_placeholder);
        
        // Cast to output dtype if needed
        result_tensor = castToCommonType(graph, result_tensor, output_dtype);
        
        // Create output tensor (same shape as input)
        Tensor output(input.shape(), output_dtype, Device::GPU);
        
        // Create tensor data
        MPSGraphTensorData* input_data = createTensorData(input);
        MPSGraphTensorData* output_data = createTensorData(output);
        
        // Create feeds and targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            input_placeholder: input_data
        };
        
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: output_data
        };
        
        // Execute the graph using synchronous API
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return output;
    }
}

static Tensor executeMPSGraphTernaryOp(const Tensor& cond_raw, const Tensor& a_raw, const Tensor& b_raw,
                                       DType output_dtype,
                                       MPSGraphTernaryOpBlock op_block) {
    @autoreleasepool {
        // Make non-contiguous tensors contiguous using GPU gather kernel
        Tensor cond = ensureContiguous(cond_raw);
        Tensor a = ensureContiguous(a_raw);
        Tensor b = ensureContiguous(b_raw);
        
        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create placeholders
        MPSGraphTensor* cond_placeholder = createPlaceholder(graph, cond);
        MPSGraphTensor* a_placeholder = createPlaceholder(graph, a);
        MPSGraphTensor* b_placeholder = createPlaceholder(graph, b);
        
        // Always use comparison to zero for reliable Bool semantics
        // MPSGraph's Bool handling is inconsistent with Axiom's 1-byte Bool,
        // so we compare != 0 which works correctly for all input dtypes
        MPSGraphTensor* zero = [graph constantWithScalar:0.0
                                                dataType:getMPSDataType(cond.dtype())];
        MPSGraphTensor* cond_bool = [graph notEqualWithPrimaryTensor:cond_placeholder
                                                    secondaryTensor:zero
                                                               name:nil];
        
        // Type promotion for a and b
        DType common_dtype = ops::promote_types(a.dtype(), b.dtype());
        MPSGraphTensor* a_cast = castToCommonType(graph, a_placeholder, common_dtype);
        MPSGraphTensor* b_cast = castToCommonType(graph, b_placeholder, common_dtype);
        
        // Execute the operation
        MPSGraphTensor* result_tensor = op_block(graph, cond_bool, a_cast, b_cast);
        
        // Cast to output dtype if needed
        result_tensor = castToCommonType(graph, result_tensor, output_dtype);
        
        // Infer output shape (broadcast all three inputs)
        Shape temp_shape = ShapeUtils::broadcast_shape(cond.shape(), a.shape());
        Shape output_shape = ShapeUtils::broadcast_shape(temp_shape, b.shape());
        
        // Create output tensor
        Tensor output(output_shape, output_dtype, Device::GPU);
        
        // Create tensor data
        MPSGraphTensorData* cond_data = createTensorData(cond);
        MPSGraphTensorData* a_data = createTensorData(a);
        MPSGraphTensorData* b_data = createTensorData(b);
        MPSGraphTensorData* output_data = createTensorData(output);
        
        // Create feeds and targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            cond_placeholder: cond_data,
            a_placeholder: a_data,
            b_placeholder: b_data
        };
        
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: output_data
        };
        
        // Execute the graph using synchronous API
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return output;
    }
}

// ============================================================================
// MPSGraphBinaryOperation Implementation
// ============================================================================

MPSGraphBinaryOperation::MPSGraphBinaryOperation(ops::OpType op_type, 
                                                 std::string op_name,
                                                 MPSGraphBinaryOpBlock op_block)
    : MPSGraphOperation(op_type, std::move(op_name)), op_block_(op_block) {}

Tensor MPSGraphBinaryOperation::execute_binary(const Tensor& lhs, const Tensor& rhs) const {
    // Determine output dtype based on operation type
    DType output_dtype;
    
    // Comparison and logical operations output Bool
    if (op_type_ >= ops::OpType::Equal && op_type_ <= ops::OpType::GreaterEqual) {
        output_dtype = DType::Bool;
    } else if (op_type_ >= ops::OpType::LogicalAnd && op_type_ <= ops::OpType::LogicalXor) {
        output_dtype = DType::Bool;
    } else {
        // Arithmetic operations preserve type promotion
        output_dtype = ops::promote_types(lhs.dtype(), rhs.dtype());
    }
    
    return executeMPSGraphBinaryOp(lhs, rhs, output_dtype, op_block_);
}

// ============================================================================
// MPSGraphUnaryOperation Implementation
// ============================================================================

MPSGraphUnaryOperation::MPSGraphUnaryOperation(ops::OpType op_type,
                                               std::string op_name,
                                               MPSGraphUnaryOpBlock op_block)
    : MPSGraphOperation(op_type, std::move(op_name)), op_block_(op_block) {}

Tensor MPSGraphUnaryOperation::execute_unary(const Tensor& input) const {
    // Logical NOT outputs Bool, others preserve input dtype
    DType output_dtype = (op_type_ == ops::OpType::LogicalNot) ? DType::Bool : input.dtype();
    return executeMPSGraphUnaryOp(input, output_dtype, op_block_);
}

// ============================================================================
// MPSGraphTernaryOperation Implementation
// ============================================================================

MPSGraphTernaryOperation::MPSGraphTernaryOperation(ops::OpType op_type,
                                                   std::string op_name,
                                                   MPSGraphTernaryOpBlock op_block)
    : MPSGraphOperation(op_type, std::move(op_name)), op_block_(op_block) {}

Tensor MPSGraphTernaryOperation::execute_where(const Tensor& condition, 
                                               const Tensor& a, 
                                               const Tensor& b) const {
    DType output_dtype = ops::promote_types(a.dtype(), b.dtype());
    return executeMPSGraphTernaryOp(condition, a, b, output_dtype, op_block_);
}

// ============================================================================
// MPSGraph Operation Blocks
// ============================================================================

// Binary arithmetic operations (migrated from custom Metal kernels for fusion benefits)
static MPSGraphTensor* add_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* subtract_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* multiply_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* divide_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
}

// Comparison operations
static MPSGraphTensor* equal_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph equalWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* not_equal_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph notEqualWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* less_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph lessThanWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* less_equal_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph lessThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* greater_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph greaterThanWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* greater_equal_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph greaterThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
}

// Logical operations
static MPSGraphTensor* logical_and_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    // Cast inputs to Bool first
    MPSGraphTensor* a_bool = [graph castTensor:a toType:MPSDataTypeBool name:nil];
    MPSGraphTensor* b_bool = [graph castTensor:b toType:MPSDataTypeBool name:nil];
    return [graph logicalANDWithPrimaryTensor:a_bool secondaryTensor:b_bool name:nil];
}

static MPSGraphTensor* logical_or_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    MPSGraphTensor* a_bool = [graph castTensor:a toType:MPSDataTypeBool name:nil];
    MPSGraphTensor* b_bool = [graph castTensor:b toType:MPSDataTypeBool name:nil];
    return [graph logicalORWithPrimaryTensor:a_bool secondaryTensor:b_bool name:nil];
}

static MPSGraphTensor* logical_xor_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    MPSGraphTensor* a_bool = [graph castTensor:a toType:MPSDataTypeBool name:nil];
    MPSGraphTensor* b_bool = [graph castTensor:b toType:MPSDataTypeBool name:nil];
    return [graph logicalXORWithPrimaryTensor:a_bool secondaryTensor:b_bool name:nil];
}

// Unary logical operation
static MPSGraphTensor* logical_not_op(MPSGraph* graph, MPSGraphTensor* a) {
    // Cast to bool first, then negate
    MPSGraphTensor* a_bool = [graph castTensor:a toType:MPSDataTypeBool name:nil];
    return [graph notWithTensor:a_bool name:nil];
}

// Math binary operations
static MPSGraphTensor* maximum_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* minimum_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* atan2_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph atan2WithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* power_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph powerWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* hypot_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    // hypot(a,b) = sqrt(a² + b²)
    MPSGraphTensor* a_sq = [graph multiplicationWithPrimaryTensor:a secondaryTensor:a name:nil];
    MPSGraphTensor* b_sq = [graph multiplicationWithPrimaryTensor:b secondaryTensor:b name:nil];
    MPSGraphTensor* sum_sq = [graph additionWithPrimaryTensor:a_sq secondaryTensor:b_sq name:nil];
    return [graph squareRootWithTensor:sum_sq name:nil];
}

static MPSGraphTensor* modulo_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph moduloWithPrimaryTensor:a secondaryTensor:b name:nil];
}

// Bitwise operations (integer types only)
static MPSGraphTensor* bitwise_and_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph bitwiseANDWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* bitwise_or_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph bitwiseORWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* bitwise_xor_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph bitwiseXORWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* left_shift_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph bitwiseLeftShiftWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* right_shift_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph bitwiseRightShiftWithPrimaryTensor:a secondaryTensor:b name:nil];
}

// Ternary operation (where/select)
static MPSGraphTensor* where_op(MPSGraph* graph, MPSGraphTensor* cond, 
                                MPSGraphTensor* a, MPSGraphTensor* b) {
    return [graph selectWithPredicateTensor:cond
                        truePredicateTensor:a
                       falsePredicateTensor:b
                                       name:nil];
}

// Unary math operations (migrated from custom Metal kernels)
static MPSGraphTensor* negate_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph negativeWithTensor:a name:nil];
}

static MPSGraphTensor* abs_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph absoluteWithTensor:a name:nil];
}

static MPSGraphTensor* sqrt_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph squareRootWithTensor:a name:nil];
}

static MPSGraphTensor* exp_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph exponentWithTensor:a name:nil];
}

static MPSGraphTensor* log_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph logarithmWithTensor:a name:nil];
}

static MPSGraphTensor* sin_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph sinWithTensor:a name:nil];
}

static MPSGraphTensor* cos_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph cosWithTensor:a name:nil];
}

static MPSGraphTensor* tan_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph tanWithTensor:a name:nil];
}

// ============================================================================
// Reduction Operations (migrated from custom Metal kernels)
// ============================================================================

static Tensor executeReduction(const Tensor& input_raw, const std::vector<int>& raw_axes, 
                               bool keep_dims, ops::OpType op_type) {
    @autoreleasepool {
        // For non-contiguous GPU tensors, make them contiguous using GPU gather kernel
        Tensor input = ensureContiguous(input_raw);
        
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create input placeholder
        MPSGraphTensor* input_tensor = createPlaceholder(graph, input);
        
        // Handle empty axes (reduce over all dimensions)
        std::vector<int> axes = raw_axes;
        if (axes.empty()) {
            axes.resize(input.ndim());
            std::iota(axes.begin(), axes.end(), 0);
        }
        
        // Normalize negative axes
        for (int& axis : axes) {
            if (axis < 0) {
                axis += input.ndim();
            }
        }
        
        // Convert axes to NSArray
        NSMutableArray<NSNumber*>* mps_axes = [NSMutableArray arrayWithCapacity:axes.size()];
        for (int axis : axes) {
            [mps_axes addObject:@(axis)];
        }
        
        // Perform reduction
        MPSGraphTensor* result_tensor;
        switch (op_type) {
            case ops::OpType::Sum:
                result_tensor = [graph reductionSumWithTensor:input_tensor axes:mps_axes name:nil];
                break;
            case ops::OpType::Mean:
                result_tensor = [graph meanOfTensor:input_tensor axes:mps_axes name:nil];
                break;
            case ops::OpType::Max:
                result_tensor = [graph reductionMaximumWithTensor:input_tensor axes:mps_axes name:nil];
                break;
            case ops::OpType::Min:
                result_tensor = [graph reductionMinimumWithTensor:input_tensor axes:mps_axes name:nil];
                break;
            case ops::OpType::Any: {
                // any = max > 0 (cast to numeric first if bool)
                MPSGraphTensor* numeric_tensor = input_tensor;
                if (input.dtype() == DType::Bool) {
                    numeric_tensor = [graph castTensor:input_tensor toType:MPSDataTypeInt32 name:nil];
                }
                result_tensor = [graph reductionMaximumWithTensor:numeric_tensor axes:mps_axes name:nil];
                // Convert to bool: > 0
                MPSGraphTensor* zero = [graph constantWithScalar:0.0
                                                        dataType:result_tensor.dataType];
                result_tensor = [graph greaterThanWithPrimaryTensor:result_tensor
                                                   secondaryTensor:zero
                                                              name:nil];
                break;
            }
            case ops::OpType::All: {
                // all = min > 0 (cast to numeric first if bool)
                MPSGraphTensor* numeric_tensor = input_tensor;
                if (input.dtype() == DType::Bool) {
                    numeric_tensor = [graph castTensor:input_tensor toType:MPSDataTypeInt32 name:nil];
                }
                result_tensor = [graph reductionMinimumWithTensor:numeric_tensor axes:mps_axes name:nil];
                // Convert to bool: > 0
                MPSGraphTensor* zero = [graph constantWithScalar:0.0
                                                        dataType:result_tensor.dataType];
                result_tensor = [graph greaterThanWithPrimaryTensor:result_tensor
                                                   secondaryTensor:zero
                                                              name:nil];
                break;
            }
            default:
                throw RuntimeError::not_implemented("Reduction operation");
        }
        
        // MPSGraph reductions produce output with reduced dimensions as size 1 (like keep_dims=true)
        // We need to use this "apparent shape" for MPSGraphTensorData
        
        // Build apparent shape (always with 1s for reduced dimensions)
        NSMutableArray<NSNumber*>* apparent_shape = [NSMutableArray arrayWithCapacity:input.ndim()];
        std::set<int> axes_set(axes.begin(), axes.end());
        for (size_t i = 0; i < input.ndim(); ++i) {
            if (axes_set.count(i)) {
                [apparent_shape addObject:@(1)];
            } else {
                [apparent_shape addObject:@(input.shape()[i])];
            }
        }
        
        // Create feeds and execute
        MPSGraphTensorData* input_data = createTensorData(input);
        NSDictionary* feeds = @{input_tensor: input_data};
        
        // Calculate final output shape for Axiom tensor
        Shape output_shape;
        if (keep_dims) {
            output_shape = input.shape();
            for (int axis : axes) {
                output_shape[axis] = 1;
            }
        } else {
            for (size_t i = 0; i < input.ndim(); ++i) {
                if (!axes_set.count(i)) {
                    output_shape.push_back(input.shape()[i]);
                }
            }
            if (output_shape.empty()) {
                output_shape = {1};  // Scalar result (Axiom convention)
            }
        }
        
        // Create output tensor with Axiom's expected shape
        Tensor result = Tensor(output_shape, input.dtype(), Device::GPU);
        auto* result_storage = static_cast<MetalStorage*>(result.storage().get());
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)result_storage->buffer();
        
        // Use apparent shape (with 1s) for MPSGraphTensorData to match MPSGraph output
        NSArray<NSNumber*>* mps_output_shape = apparent_shape;
        
        MPSGraphTensorData* result_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:result_buffer
            shape:mps_output_shape
            dataType:getMPSDataType(input.dtype())];
        
        // Execute the graph using synchronous API
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:@{result_tensor: result_data}];
        
        return result;
    }
}

// Reduction operation wrapper class
class MPSGraphReductionOperation : public ops::Operation {
private:
    ops::OpType op_type_;
    std::string op_name_;
    
public:
    MPSGraphReductionOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}
    
    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }
    
    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw RuntimeError::internal("execute_binary called on reduction operation");
    }
    
    Tensor execute_unary(const Tensor& input) const override {
        (void)input;
        throw RuntimeError::internal("execute_unary called on reduction operation");
    }
    
    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axes, bool keep_dims) const override {
        if (input.device() != Device::GPU) {
            throw DeviceError("MPSGraph reduction requires GPU tensor");
        }
        return executeReduction(input, axes, keep_dims, op_type_);
    }
    
    void execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw RuntimeError::internal("execute_binary_inplace called on reduction operation");
    }
};

// ============================================================================
// Matrix Multiplication Operation (MPSGraph)
// ============================================================================

static Tensor executeMatMul(const Tensor& a_raw, const Tensor& b_raw,
                            bool transpose_a, bool transpose_b) {
    @autoreleasepool {
        // Make non-contiguous tensors contiguous using GPU gather kernel
        Tensor a = ensureContiguous(a_raw);
        Tensor b = ensureContiguous(b_raw);
        
        // Type promotion
        DType result_dtype = ops::promote_types(a.dtype(), b.dtype());
        
        // MPSGraph matmul supports Float16 and Float32
        if (result_dtype != DType::Float32 && result_dtype != DType::Float16) {
            result_dtype = DType::Float32;
        }
        
        // Cast to common dtype if needed
        if (a.dtype() != result_dtype) {
            // Create a simple cast operation via MPSGraph
            a = a.astype(result_dtype);
        }
        if (b.dtype() != result_dtype) {
            b = b.astype(result_dtype);
        }
        
        // Get matrix dimensions
        size_t a_ndim = a.ndim();
        size_t b_ndim = b.ndim();
        
        if (a_ndim == 0 || b_ndim == 0) {
            throw ShapeError("MatMul does not support 0-dimensional tensors");
        }
        
        // Compute M, N, K dimensions and output shape
        size_t a_rows, a_cols, b_rows, b_cols;
        
        if (a_ndim == 1) {
            a_rows = 1;
            a_cols = a.shape()[0];
        } else {
            a_rows = a.shape()[a_ndim - 2];
            a_cols = a.shape()[a_ndim - 1];
        }
        
        if (b_ndim == 1) {
            b_rows = b.shape()[0];
            b_cols = 1;
        } else {
            b_rows = b.shape()[b_ndim - 2];
            b_cols = b.shape()[b_ndim - 1];
        }
        
        if (transpose_a) std::swap(a_rows, a_cols);
        if (transpose_b) std::swap(b_rows, b_cols);
        
        size_t M = a_rows;
        size_t K = a_cols;
        size_t K_b = b_rows;
        size_t N = b_cols;
        
        if (K != K_b) {
            throw ShapeError(
                "MatMul dimension mismatch: A has " + std::to_string(K) +
                " columns but B has " + std::to_string(K_b) + " rows");
        }
        
        // Compute batch shape if needed
        Shape batch_shape;
        if (a_ndim > 2 || b_ndim > 2) {
            Shape a_batch, b_batch;
            for (size_t i = 0; i < (a_ndim > 2 ? a_ndim - 2 : 0); ++i) {
                a_batch.push_back(a.shape()[i]);
            }
            for (size_t i = 0; i < (b_ndim > 2 ? b_ndim - 2 : 0); ++i) {
                b_batch.push_back(b.shape()[i]);
            }
            batch_shape = ShapeUtils::broadcast_shape(a_batch, b_batch);
        }
        
        // Compute output shape
        Shape result_shape = batch_shape;
        if (a_ndim == 1 && b_ndim == 1) {
            result_shape = {1};  // Dot product
        } else if (a_ndim == 1) {
            result_shape.push_back(N);
        } else if (b_ndim == 1) {
            result_shape.push_back(M);
        } else {
            result_shape.push_back(M);
            result_shape.push_back(N);
        }
        if (result_shape.empty()) result_shape = {1};
        
        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // For MPSGraph matmul, we need 2D or higher tensors
        // If input is 1D, reshape to 2D
        Shape a_mps_shape = a.shape();
        Shape b_mps_shape = b.shape();
        
        if (a_ndim == 1) {
            a_mps_shape = {1, a.shape()[0]};  // Row vector
        }
        if (b_ndim == 1) {
            b_mps_shape = {b.shape()[0], 1};  // Column vector
        }
        
        // Create placeholders with adjusted shapes
        MPSDataType mps_dtype = getMPSDataType(result_dtype);
        MPSGraphTensor* a_placeholder = [graph placeholderWithShape:getMPSShape(a_mps_shape)
                                                           dataType:mps_dtype
                                                               name:nil];
        MPSGraphTensor* b_placeholder = [graph placeholderWithShape:getMPSShape(b_mps_shape)
                                                           dataType:mps_dtype
                                                               name:nil];
        
        MPSGraphTensor* a_tensor = a_placeholder;
        MPSGraphTensor* b_tensor = b_placeholder;
        
        // Handle transpose via MPSGraph transpose operation
        if (transpose_a) {
            NSInteger a_rank = [a_tensor.shape count];
            a_tensor = [graph transposeTensor:a_tensor
                                    dimension:a_rank - 2
                                withDimension:a_rank - 1
                                         name:nil];
        }
        if (transpose_b) {
            NSInteger b_rank = [b_tensor.shape count];
            b_tensor = [graph transposeTensor:b_tensor
                                    dimension:b_rank - 2
                                withDimension:b_rank - 1
                                         name:nil];
        }
        
        // Perform matrix multiplication using MPSGraph
        MPSGraphTensor* result_tensor = [graph matrixMultiplicationWithPrimaryTensor:a_tensor
                                                                    secondaryTensor:b_tensor
                                                                               name:nil];
        
        // Create output tensor
        Tensor result(result_shape, result_dtype, Device::GPU);
        
        // Create tensor data - reshape inputs if needed
        auto* a_storage = static_cast<const MetalStorage*>(a.storage().get());
        auto* b_storage = static_cast<const MetalStorage*>(b.storage().get());
        auto* result_storage = static_cast<MetalStorage*>(result.storage().get());
        
        id<MTLBuffer> a_buffer = (__bridge id<MTLBuffer>)a_storage->buffer();
        id<MTLBuffer> b_buffer = (__bridge id<MTLBuffer>)b_storage->buffer();
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)result_storage->buffer();
        
        MPSGraphTensorData* a_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:a_buffer
                        shape:getMPSShape(a_mps_shape)
                     dataType:mps_dtype];
        
        MPSGraphTensorData* b_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:b_buffer
                        shape:getMPSShape(b_mps_shape)
                     dataType:mps_dtype];
        
        // For output, we need the shape that MPSGraph produces
        // This accounts for the possible reshape of inputs
        Shape result_mps_shape = result_shape;
        if (a_ndim == 1 && b_ndim == 1) {
            result_mps_shape = {1, 1};  // (1,K) @ (K,1) = (1,1)
        } else if (a_ndim == 1) {
            // (1,K) @ (K,N) = (1,N), but we want just (N)
            result_mps_shape = batch_shape;
            result_mps_shape.push_back(1);
            result_mps_shape.push_back(N);
        } else if (b_ndim == 1) {
            // (M,K) @ (K,1) = (M,1), but we want just (M)
            result_mps_shape = batch_shape;
            result_mps_shape.push_back(M);
            result_mps_shape.push_back(1);
        }
        
        MPSGraphTensorData* result_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:result_buffer
                        shape:getMPSShape(result_mps_shape)
                     dataType:mps_dtype];
        
        // Create feeds and targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            a_placeholder: a_data,
            b_placeholder: b_data
        };
        
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: result_data
        };
        
        // Execute the graph
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return result;
    }
}

// MatMul operation wrapper class
class MPSGraphMatMulOperation : public ops::Operation {
public:
    ops::OpType type() const override { return ops::OpType::MatMul; }
    std::string name() const override { return "matmul"; }
    Device device() const override { return Device::GPU; }
    
    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw RuntimeError::internal("Use execute_matmul for MatMul operations");
    }
    
    Tensor execute_unary(const Tensor& input) const override {
        (void)input;
        throw RuntimeError::internal("execute_unary called on matmul operation");
    }
    
    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axes, bool keep_dims) const override {
        (void)input; (void)axes; (void)keep_dims;
        throw RuntimeError::internal("execute_reduction called on matmul operation");
    }
    
    Tensor execute_matmul(const Tensor& a, const Tensor& b,
                         bool transpose_a, bool transpose_b) const override {
        if (a.device() != Device::GPU || b.device() != Device::GPU) {
            throw DeviceError("MPSGraph MatMul requires GPU tensors");
        }
        return executeMatMul(a, b, transpose_a, transpose_b);
    }
};

// ============================================================================
// ArgMax/ArgMin Operations (MPSGraph)
// ============================================================================

static Tensor executeArgMaxMin(const Tensor& input_raw, int axis, bool keep_dims, bool is_max) {
    @autoreleasepool {
        // Make non-contiguous tensor contiguous using GPU gather kernel
        Tensor input = ensureContiguous(input_raw);
        
        // Normalize axis
        int normalized_axis = axis;
        if (normalized_axis < 0) {
            normalized_axis += static_cast<int>(input.ndim());
        }
        
        if (normalized_axis < 0 || normalized_axis >= static_cast<int>(input.ndim())) {
            throw ShapeError("Axis " + std::to_string(axis) + " is out of range for tensor with " +
                           std::to_string(input.ndim()) + " dimensions");
        }
        
        // Create MPSGraph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create input placeholder
        MPSGraphTensor* input_tensor = createPlaceholder(graph, input);
        
        // Perform ArgMax or ArgMin reduction
        MPSGraphTensor* result_tensor;
        if (is_max) {
            result_tensor = [graph reductionArgMaximumWithTensor:input_tensor
                                                           axis:(NSInteger)normalized_axis
                                                           name:nil];
        } else {
            result_tensor = [graph reductionArgMinimumWithTensor:input_tensor
                                                           axis:(NSInteger)normalized_axis
                                                           name:nil];
        }
        
        // MPSGraph argmax/argmin produces Int32 output by default
        // We need to cast to Int64 to match Axiom's convention
        result_tensor = [graph castTensor:result_tensor toType:MPSDataTypeInt64 name:nil];
        
        // Create feeds
        MPSGraphTensorData* input_data = createTensorData(input);
        NSDictionary* feeds = @{input_tensor: input_data};
        
        // Calculate output shape
        // ArgMax/ArgMin reduces along one axis, similar to other reductions
        Shape output_shape;
        for (size_t i = 0; i < input.ndim(); ++i) {
            if (static_cast<int>(i) == normalized_axis) {
                if (keep_dims) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input.shape()[i]);
            }
        }
        if (output_shape.empty()) {
            output_shape = {1};  // Scalar result
        }
        
        // Create output tensor with Int64 dtype
        Tensor result = Tensor(output_shape, DType::Int64, Device::GPU);
        auto* result_storage = static_cast<MetalStorage*>(result.storage().get());
        id<MTLBuffer> result_buffer = (__bridge id<MTLBuffer>)result_storage->buffer();
        
        // Build apparent shape (with 1 at reduced dimension) for MPSGraph
        Shape apparent_shape;
        for (size_t i = 0; i < input.ndim(); ++i) {
            if (static_cast<int>(i) == normalized_axis) {
                apparent_shape.push_back(1);
            } else {
                apparent_shape.push_back(input.shape()[i]);
            }
        }
        
        MPSGraphTensorData* result_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:result_buffer
                        shape:getMPSShape(apparent_shape)
                     dataType:MPSDataTypeInt64];
        
        // Execute the graph
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:@{result_tensor: result_data}];
        
        return result;
    }
}

// ArgMax operation wrapper class
class MPSGraphArgMaxOperation : public ops::Operation {
public:
    ops::OpType type() const override { return ops::OpType::ArgMax; }
    std::string name() const override { return "argmax"; }
    Device device() const override { return Device::GPU; }
    
    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw RuntimeError::internal("execute_binary called on ArgMax operation");
    }
    
    Tensor execute_unary(const Tensor& input) const override {
        (void)input;
        throw RuntimeError::internal("execute_unary called on ArgMax operation");
    }
    
    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axes, bool keep_dims) const override {
        if (input.device() != Device::GPU) {
            throw DeviceError("MPSGraph ArgMax requires GPU tensor");
        }
        
        // ArgMax operates on a single axis
        int axis = axes.empty() ? -1 : axes[0];
        
        // For full reduction (axis=-1), flatten first then argmax on axis 0
        if (axis == -1 && axes.empty()) {
            // This case means no axis specified - reduce over all elements
            auto flat = input.flatten();
            return executeArgMaxMin(flat, 0, keep_dims, true);
        }
        
        return executeArgMaxMin(input, axis, keep_dims, true);
    }
};

// ArgMin operation wrapper class
class MPSGraphArgMinOperation : public ops::Operation {
public:
    ops::OpType type() const override { return ops::OpType::ArgMin; }
    std::string name() const override { return "argmin"; }
    Device device() const override { return Device::GPU; }
    
    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw RuntimeError::internal("execute_binary called on ArgMin operation");
    }
    
    Tensor execute_unary(const Tensor& input) const override {
        (void)input;
        throw RuntimeError::internal("execute_unary called on ArgMin operation");
    }
    
    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axes, bool keep_dims) const override {
        if (input.device() != Device::GPU) {
            throw DeviceError("MPSGraph ArgMin requires GPU tensor");
        }
        
        // ArgMin operates on a single axis
        int axis = axes.empty() ? -1 : axes[0];
        
        // For full reduction (axis=-1), flatten first then argmin on axis 0
        if (axis == -1 && axes.empty()) {
            // This case means no axis specified - reduce over all elements
            auto flat = input.flatten();
            return executeArgMaxMin(flat, 0, keep_dims, false);
        }
        
        return executeArgMaxMin(input, axis, keep_dims, false);
    }
};

// ============================================================================
// Softmax/LogSoftmax Operations (MPSGraph)
// ============================================================================

class MPSGraphSoftmaxOperation : public ops::Operation {
    bool is_log_;
public:
    MPSGraphSoftmaxOperation(bool is_log) : is_log_(is_log) {}

    ops::OpType type() const override { return is_log_ ? ops::OpType::LogSoftmax : ops::OpType::Softmax; }
    std::string name() const override { return is_log_ ? "log_softmax" : "softmax"; }
    Device device() const override { return Device::GPU; }

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw RuntimeError::internal("execute_binary called on Softmax operation");
    }

    Tensor execute_unary(const Tensor& input) const override {
        (void)input;
        throw RuntimeError::internal("Use execute_reduction for Softmax operations");
    }

    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axes, bool keep_dims) const override {
        (void)keep_dims;  // Softmax preserves shape
        if (input.device() != Device::GPU) {
            throw DeviceError("MPSGraph Softmax requires GPU tensor");
        }

        int axis = axes.empty() ? -1 : axes[0];
        return executeSoftmax(input, axis);
    }

private:
    Tensor executeSoftmax(const Tensor& input_raw, int axis) const {
        @autoreleasepool {
            Tensor input = ensureContiguous(input_raw);

            // Normalize axis
            int norm_axis = axis;
            if (norm_axis < 0) {
                norm_axis += static_cast<int>(input.ndim());
            }

            MPSGraph* graph = [[MPSGraph alloc] init];

            // Create input placeholder
            MPSGraphTensor* x = createPlaceholder(graph, input);

            // MPSGraph has built-in numerically stable softmax
            MPSGraphTensor* result_tensor = [graph softMaxWithTensor:x axis:norm_axis name:nil];

            if (is_log_) {
                result_tensor = [graph logarithmWithTensor:result_tensor name:nil];
            }

            // Create output tensor (same shape as input)
            Tensor result(input.shape(), input.dtype(), Device::GPU);

            // Create tensor data
            MPSGraphTensorData* input_data = createTensorData(input);
            MPSGraphTensorData* result_data = createTensorData(result);

            // Create feeds and targets
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
                x: input_data
            };

            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
                result_tensor: result_data
            };

            // Execute the graph
            [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                    feeds:feeds
                         targetOperations:nil
                        resultsDictionary:targets];

            return result;
        }
    }
};

// ============================================================================
// Erf and GELU Operations (MPSGraph)
// ============================================================================

static MPSGraphTensor* erf_op(MPSGraph* graph, MPSGraphTensor* a) {
    return [graph erfWithTensor:a name:nil];
}

static MPSGraphTensor* gelu_op(MPSGraph* graph, MPSGraphTensor* x) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    MPSGraphTensor* sqrt2 = [graph constantWithScalar:M_SQRT2 dataType:x.dataType];
    MPSGraphTensor* half = [graph constantWithScalar:0.5 dataType:x.dataType];
    MPSGraphTensor* one = [graph constantWithScalar:1.0 dataType:x.dataType];

    MPSGraphTensor* x_scaled = [graph divisionWithPrimaryTensor:x secondaryTensor:sqrt2 name:nil];
    MPSGraphTensor* erf_val = [graph erfWithTensor:x_scaled name:nil];
    MPSGraphTensor* inner = [graph additionWithPrimaryTensor:one secondaryTensor:erf_val name:nil];
    MPSGraphTensor* half_x = [graph multiplicationWithPrimaryTensor:half secondaryTensor:x name:nil];
    return [graph multiplicationWithPrimaryTensor:half_x secondaryTensor:inner name:nil];
}

static MPSGraphTensor* relu_op(MPSGraph* graph, MPSGraphTensor* x) {
    return [graph reLUWithTensor:x name:nil];
}

static MPSGraphTensor* leaky_relu_op(MPSGraph* graph, MPSGraphTensor* x) {
    // Default alpha = 0.01
    return [graph leakyReLUWithTensor:x alpha:0.01 name:nil];
}

static MPSGraphTensor* sigmoid_op(MPSGraph* graph, MPSGraphTensor* x) {
    return [graph sigmoidWithTensor:x name:nil];
}

static MPSGraphTensor* tanh_op(MPSGraph* graph, MPSGraphTensor* x) {
    return [graph tanhWithTensor:x name:nil];
}

static MPSGraphTensor* silu_op(MPSGraph* graph, MPSGraphTensor* x) {
    // SiLU(x) = x * sigmoid(x)
    MPSGraphTensor* sig = [graph sigmoidWithTensor:x name:nil];
    return [graph multiplicationWithPrimaryTensor:x secondaryTensor:sig name:nil];
}

// ============================================================================
// MPSGraph Masking Operations Implementation
// ============================================================================

Tensor MPSGraphMaskedFillOperation::execute_masked_fill(const Tensor& input,
                                                        const Tensor& mask,
                                                        const Tensor& value) const {
    @autoreleasepool {
        // Ensure inputs are contiguous and on GPU
        Tensor input_cont = ensureContiguous(input);
        Tensor mask_cont = ensureContiguous(mask);
        Tensor value_cont = ensureContiguous(value);
        
        // Handle broadcasting
        auto broadcast_info = ops::compute_broadcast_info(input_cont.shape(), mask_cont.shape());
        const auto& result_shape = broadcast_info.result_shape;
        
        // Create the graph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create placeholders using the tensors
        MPSGraphTensor* input_placeholder = createPlaceholder(graph, input_cont);
        MPSGraphTensor* mask_placeholder = createPlaceholder(graph, mask_cont);
        MPSGraphTensor* value_placeholder = createPlaceholder(graph, value_cont);
        
        // Convert mask to bool
        MPSGraphTensor* mask_bool = [graph castTensor:mask_placeholder toType:MPSDataTypeBool name:nil];
        
        // Cast value to input dtype
        MPSGraphTensor* value_casted = [graph castTensor:value_placeholder 
                                                  toType:getMPSDataType(input_cont.dtype()) 
                                                    name:nil];
        
        // Broadcast value to result shape
        NSMutableArray<NSNumber*>* result_ns_shape = [NSMutableArray arrayWithCapacity:result_shape.size()];
        for (size_t dim : result_shape) {
            [result_ns_shape addObject:@(dim)];
        }
        MPSGraphTensor* value_broadcast = [graph broadcastTensor:value_casted
                                                         toShape:result_ns_shape
                                                            name:nil];
        
        // Use select: where mask is true, use value; otherwise use input
        MPSGraphTensor* result_tensor = [graph selectWithPredicateTensor:mask_bool
                                                     truePredicateTensor:value_broadcast
                                                    falsePredicateTensor:input_placeholder
                                                                    name:nil];
        
        // Create output tensor
        Tensor result(result_shape, input_cont.dtype(), Device::GPU);
        
        // Create tensor data
        MPSGraphTensorData* input_data = createTensorData(input_cont);
        MPSGraphTensorData* mask_data = createTensorData(mask_cont);
        MPSGraphTensorData* value_data = createTensorData(value_cont);
        MPSGraphTensorData* result_data = createTensorData(result);
        
        // Create feeds and targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            input_placeholder: input_data,
            mask_placeholder: mask_data,
            value_placeholder: value_data
        };
        
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: result_data
        };
        
        // Execute
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return result;
    }
}

Tensor MPSGraphMaskedSelectOperation::execute_masked_select(const Tensor& input,
                                                            const Tensor& mask) const {
    // MaskedSelect is complex for GPU - fallback to CPU for now
    // A proper GPU implementation would require a two-pass approach:
    // 1. Count non-zero elements in mask (prefix sum)
    // 2. Compact elements using scatter
    
    // For now, execute on CPU
    Tensor input_cpu = input.cpu();
    Tensor mask_cpu = mask.cpu();
    
    auto cpu_op = ops::OperationRegistry::get_operation(ops::OpType::MaskedSelect, Device::CPU);
    if (!cpu_op) {
        throw DeviceError("MaskedSelect not available on CPU");
    }
    
    Tensor result_cpu = cpu_op->execute_masked_select(input_cpu, mask_cpu);
    return result_cpu.gpu();
}

// ============================================================================
// MPSGraph Indexing Operations Implementation
// ============================================================================

Tensor MPSGraphGatherOperation::execute_gather(const Tensor& input, int dim,
                                               const Tensor& indices) const {
    @autoreleasepool {
        // Ensure inputs are contiguous and on GPU
        Tensor input_cont = ensureContiguous(input);
        Tensor indices_cont = ensureContiguous(indices.astype(DType::Int32)).gpu();
        
        // Normalize dim
        int norm_dim = dim;
        if (norm_dim < 0) {
            norm_dim += static_cast<int>(input_cont.ndim());
        }
        
        // Create the graph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create placeholders using tensors
        MPSGraphTensor* input_placeholder = createPlaceholder(graph, input_cont);
        MPSGraphTensor* indices_placeholder = createPlaceholder(graph, indices_cont);
        
        // Use gatherAlongAxis
        MPSGraphTensor* result_tensor = [graph gatherAlongAxis:norm_dim
                                               withUpdatesTensor:input_placeholder
                                                   indicesTensor:indices_placeholder
                                                            name:nil];
        
        // Output shape is same as indices
        Tensor result(indices_cont.shape(), input_cont.dtype(), Device::GPU);
        
        // Create tensor data
        MPSGraphTensorData* input_data = createTensorData(input_cont);
        MPSGraphTensorData* indices_data = createTensorData(indices_cont);
        MPSGraphTensorData* result_data = createTensorData(result);
        
        // Create feeds and targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            input_placeholder: input_data,
            indices_placeholder: indices_data
        };
        
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: result_data
        };
        
        // Execute
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return result;
    }
}

Tensor MPSGraphScatterOperation::execute_scatter(const Tensor& input, int dim,
                                                 const Tensor& indices,
                                                 const Tensor& src) const {
    @autoreleasepool {
        // Ensure inputs are contiguous and on GPU
        Tensor input_cont = ensureContiguous(input);
        Tensor indices_cont = ensureContiguous(indices.astype(DType::Int32)).gpu();
        Tensor src_cont = ensureContiguous(src.astype(input.dtype())).gpu();
        
        // Normalize dim
        int norm_dim = dim;
        if (norm_dim < 0) {
            norm_dim += static_cast<int>(input_cont.ndim());
        }
        
        // Create the graph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create placeholders using tensors
        MPSGraphTensor* input_placeholder = createPlaceholder(graph, input_cont);
        MPSGraphTensor* indices_placeholder = createPlaceholder(graph, indices_cont);
        MPSGraphTensor* src_placeholder = createPlaceholder(graph, src_cont);
        
        // Use scatterAlongAxis
        MPSGraphTensor* result_tensor = [graph scatterAlongAxis:norm_dim
                                                   withDataTensor:input_placeholder
                                                    updatesTensor:src_placeholder
                                                    indicesTensor:indices_placeholder
                                                             mode:MPSGraphScatterModeSet
                                                             name:nil];
        
        // Output shape is same as input
        Tensor result(input_cont.shape(), input_cont.dtype(), Device::GPU);
        
        // Create tensor data
        MPSGraphTensorData* input_data = createTensorData(input_cont);
        MPSGraphTensorData* indices_data = createTensorData(indices_cont);
        MPSGraphTensorData* src_data = createTensorData(src_cont);
        MPSGraphTensorData* result_data = createTensorData(result);
        
        // Create feeds and targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            input_placeholder: input_data,
            indices_placeholder: indices_data,
            src_placeholder: src_data
        };
        
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: result_data
        };
        
        // Execute
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return result;
    }
}

Tensor MPSGraphIndexSelectOperation::execute_index_select(const Tensor& input, int dim,
                                                          const Tensor& indices) const {
    @autoreleasepool {
        // Ensure inputs are contiguous and on GPU
        Tensor input_cont = ensureContiguous(input);
        Tensor indices_cont = ensureContiguous(indices.astype(DType::Int32)).gpu();
        
        // Normalize dim
        int norm_dim = dim;
        if (norm_dim < 0) {
            norm_dim += static_cast<int>(input_cont.ndim());
        }
        
        // Compute output shape: replace dim size with num_indices
        Shape output_shape = input_cont.shape();
        output_shape[norm_dim] = indices_cont.size();
        
        // Create the graph
        MPSGraph* graph = [[MPSGraph alloc] init];
        
        // Create placeholders using tensors
        MPSGraphTensor* input_placeholder = createPlaceholder(graph, input_cont);
        MPSGraphTensor* indices_placeholder = createPlaceholder(graph, indices_cont);
        
        // Use gatherAlongAxis - for index_select, indices are 1D and we select whole slices
        // First reshape indices to match expected broadcast shape
        NSMutableArray<NSNumber*>* indices_reshape = [NSMutableArray arrayWithCapacity:input_cont.ndim()];
        for (size_t i = 0; i < input_cont.ndim(); ++i) {
            if (static_cast<int>(i) == norm_dim) {
                [indices_reshape addObject:@(indices_cont.size())];
            } else {
                [indices_reshape addObject:@(1)];
            }
        }
        
        MPSGraphTensor* indices_reshaped = [graph reshapeTensor:indices_placeholder
                                                       withShape:indices_reshape
                                                            name:nil];
        
        // Broadcast indices to output shape
        NSMutableArray<NSNumber*>* output_ns_shape = [NSMutableArray arrayWithCapacity:output_shape.size()];
        for (size_t d : output_shape) {
            [output_ns_shape addObject:@(d)];
        }
        
        MPSGraphTensor* indices_broadcast = [graph broadcastTensor:indices_reshaped
                                                           toShape:output_ns_shape
                                                              name:nil];
        
        // Use gatherAlongAxis
        MPSGraphTensor* result_tensor = [graph gatherAlongAxis:norm_dim
                                               withUpdatesTensor:input_placeholder
                                                   indicesTensor:indices_broadcast
                                                            name:nil];
        
        // Create output tensor
        Tensor result(output_shape, input_cont.dtype(), Device::GPU);
        
        // Create tensor data
        MPSGraphTensorData* input_data = createTensorData(input_cont);
        MPSGraphTensorData* indices_data = createTensorData(indices_cont);
        MPSGraphTensorData* result_data = createTensorData(result);
        
        // Create feeds and targets
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            input_placeholder: input_data,
            indices_placeholder: indices_data
        };
        
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* targets = @{
            result_tensor: result_data
        };
        
        // Execute
        [graph runWithMTLCommandQueue:(__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:targets];
        
        return result;
    }
}

// ============================================================================
// Registration
// ============================================================================

void register_mpsgraph_operations() {
    using namespace ops;
    
    // Binary arithmetic operations (migrated for automatic fusion benefits)
    OperationRegistry::register_operation(OpType::Add, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Add, "add", add_op));
    
    OperationRegistry::register_operation(OpType::Subtract, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Subtract, "subtract", subtract_op));
    
    OperationRegistry::register_operation(OpType::Multiply, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Multiply, "multiply", multiply_op));
    
    OperationRegistry::register_operation(OpType::Divide, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Divide, "divide", divide_op));
    
    // Comparison operations
    OperationRegistry::register_operation(OpType::Equal, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Equal, "equal", equal_op));
    
    OperationRegistry::register_operation(OpType::NotEqual, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::NotEqual, "not_equal", not_equal_op));
    
    OperationRegistry::register_operation(OpType::Less, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Less, "less", less_op));
    
    OperationRegistry::register_operation(OpType::LessEqual, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::LessEqual, "less_equal", less_equal_op));
    
    OperationRegistry::register_operation(OpType::Greater, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Greater, "greater", greater_op));
    
    OperationRegistry::register_operation(OpType::GreaterEqual, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::GreaterEqual, "greater_equal", greater_equal_op));
    
    // Logical operations
    OperationRegistry::register_operation(OpType::LogicalAnd, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::LogicalAnd, "logical_and", logical_and_op));
    
    OperationRegistry::register_operation(OpType::LogicalOr, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::LogicalOr, "logical_or", logical_or_op));
    
    OperationRegistry::register_operation(OpType::LogicalXor, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::LogicalXor, "logical_xor", logical_xor_op));
    
    // Math binary operations
    OperationRegistry::register_operation(OpType::Maximum, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Maximum, "maximum", maximum_op));
    
    OperationRegistry::register_operation(OpType::Minimum, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Minimum, "minimum", minimum_op));
    
    OperationRegistry::register_operation(OpType::Atan2, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Atan2, "atan2", atan2_op));
    
    OperationRegistry::register_operation(OpType::Power, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Power, "power", power_op));
    
    OperationRegistry::register_operation(OpType::Modulo, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Modulo, "modulo", modulo_op));

    OperationRegistry::register_operation(OpType::Hypot, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::Hypot, "hypot", hypot_op));
    
    // Bitwise operations
    OperationRegistry::register_operation(OpType::BitwiseAnd, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::BitwiseAnd, "bitwise_and", bitwise_and_op));
    
    OperationRegistry::register_operation(OpType::BitwiseOr, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::BitwiseOr, "bitwise_or", bitwise_or_op));
    
    OperationRegistry::register_operation(OpType::BitwiseXor, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::BitwiseXor, "bitwise_xor", bitwise_xor_op));
    
    OperationRegistry::register_operation(OpType::LeftShift, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::LeftShift, "left_shift", left_shift_op));
    
    OperationRegistry::register_operation(OpType::RightShift, Device::GPU,
        std::make_unique<MPSGraphBinaryOperation>(OpType::RightShift, "right_shift", right_shift_op));
    
    // Unary operations (migrated from custom Metal kernels to MPSGraph)
    OperationRegistry::register_operation(OpType::Negate, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Negate, "negate", negate_op));
    
    OperationRegistry::register_operation(OpType::Abs, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Abs, "abs", abs_op));
    
    OperationRegistry::register_operation(OpType::Sqrt, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Sqrt, "sqrt", sqrt_op));
    
    OperationRegistry::register_operation(OpType::Exp, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Exp, "exp", exp_op));
    
    OperationRegistry::register_operation(OpType::Log, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Log, "log", log_op));
    
    OperationRegistry::register_operation(OpType::Sin, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Sin, "sin", sin_op));
    
    OperationRegistry::register_operation(OpType::Cos, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Cos, "cos", cos_op));
    
    OperationRegistry::register_operation(OpType::Tan, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Tan, "tan", tan_op));
    
    // Reduction operations (migrated from custom Metal kernels to MPSGraph)
    OperationRegistry::register_operation(OpType::Sum, Device::GPU,
        std::make_unique<MPSGraphReductionOperation>(OpType::Sum, "sum"));
    
    OperationRegistry::register_operation(OpType::Mean, Device::GPU,
        std::make_unique<MPSGraphReductionOperation>(OpType::Mean, "mean"));
    
    OperationRegistry::register_operation(OpType::Max, Device::GPU,
        std::make_unique<MPSGraphReductionOperation>(OpType::Max, "max"));
    
    OperationRegistry::register_operation(OpType::Min, Device::GPU,
        std::make_unique<MPSGraphReductionOperation>(OpType::Min, "min"));

    OperationRegistry::register_operation(OpType::Any, Device::GPU,
        std::make_unique<MPSGraphReductionOperation>(OpType::Any, "any"));

    OperationRegistry::register_operation(OpType::All, Device::GPU,
        std::make_unique<MPSGraphReductionOperation>(OpType::All, "all"));
    
    // Matrix Multiplication (migrated from custom Metal kernels to MPSGraph)
    OperationRegistry::register_operation(OpType::MatMul, Device::GPU,
        std::make_unique<MPSGraphMatMulOperation>());
    
    // ArgMax/ArgMin operations
    OperationRegistry::register_operation(OpType::ArgMax, Device::GPU,
        std::make_unique<MPSGraphArgMaxOperation>());
    
    OperationRegistry::register_operation(OpType::ArgMin, Device::GPU,
        std::make_unique<MPSGraphArgMinOperation>());
    
    // Where (conditional selection) operation
    OperationRegistry::register_operation(OpType::Where, Device::GPU,
        std::make_unique<MPSGraphTernaryOperation>(OpType::Where, "where", where_op));

    // Masking operations
    OperationRegistry::register_operation(OpType::MaskedFill, Device::GPU,
        std::make_unique<MPSGraphMaskedFillOperation>());
    
    OperationRegistry::register_operation(OpType::MaskedSelect, Device::GPU,
        std::make_unique<MPSGraphMaskedSelectOperation>());
    
    // Indexing operations
    OperationRegistry::register_operation(OpType::Gather, Device::GPU,
        std::make_unique<MPSGraphGatherOperation>());
    
    OperationRegistry::register_operation(OpType::Scatter, Device::GPU,
        std::make_unique<MPSGraphScatterOperation>());
    
    OperationRegistry::register_operation(OpType::IndexSelect, Device::GPU,
        std::make_unique<MPSGraphIndexSelectOperation>());

    // Softmax operations
    OperationRegistry::register_operation(OpType::Softmax, Device::GPU,
        std::make_unique<MPSGraphSoftmaxOperation>(false));

    OperationRegistry::register_operation(OpType::LogSoftmax, Device::GPU,
        std::make_unique<MPSGraphSoftmaxOperation>(true));

    // Erf operation
    OperationRegistry::register_operation(OpType::Erf, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Erf, "erf", erf_op));

    // GELU operation
    OperationRegistry::register_operation(OpType::GELU, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::GELU, "gelu", gelu_op));

    // Activation operations
    OperationRegistry::register_operation(OpType::ReLU, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::ReLU, "relu", relu_op));
    OperationRegistry::register_operation(OpType::LeakyReLU, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::LeakyReLU, "leaky_relu", leaky_relu_op));
    OperationRegistry::register_operation(OpType::Sigmoid, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Sigmoid, "sigmoid", sigmoid_op));
    OperationRegistry::register_operation(OpType::Tanh, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::Tanh, "tanh", tanh_op));
    OperationRegistry::register_operation(OpType::SiLU, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::SiLU, "silu", silu_op));

    // LogicalNot operation
    OperationRegistry::register_operation(OpType::LogicalNot, Device::GPU,
        std::make_unique<MPSGraphUnaryOperation>(OpType::LogicalNot, "logical_not", logical_not_op));
}

} // namespace metal
} // namespace backends
} // namespace axiom
