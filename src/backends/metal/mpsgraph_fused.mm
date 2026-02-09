#import "axiom/graph/gpu_fusion.hpp"
#import "axiom/graph/graph_node.hpp"
#import "axiom/error.hpp"
#import "axiom/operations.hpp"
#import "graph_cache.hpp"
#import "metal_common.hpp"
#import "metal_storage.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <functional>
#include <numeric>
#include <unordered_map>

namespace axiom {
namespace graph {

// ============================================================================
// MPS helpers (duplicated locally to avoid exposing ObjC in public headers)
// ============================================================================

static MPSDataType toMPS(DType dtype) {
    switch (dtype) {
    case DType::Float32:
        return MPSDataTypeFloat32;
    case DType::Float16:
        return MPSDataTypeFloat16;
    case DType::Int32:
        return MPSDataTypeInt32;
    case DType::Int64:
        return MPSDataTypeInt64;
    case DType::Int16:
        return MPSDataTypeInt16;
    case DType::Int8:
        return MPSDataTypeInt8;
    case DType::UInt8:
        return MPSDataTypeUInt8;
    case DType::UInt16:
        return MPSDataTypeUInt16;
    case DType::UInt32:
        return MPSDataTypeUInt32;
    case DType::UInt64:
        return MPSDataTypeUInt64;
    case DType::Bool:
        return MPSDataTypeBool;
    default:
        // Float64, Complex, BFloat16 are not supported by MPS.
        // Return 0 as a sentinel — callers must check before using.
        return static_cast<MPSDataType>(0);
    }
}

// Check if a dtype is supported for GPU fusion via MPS
static bool is_mps_supported_dtype(DType dtype) {
    return toMPS(dtype) != static_cast<MPSDataType>(0);
}

static MPSShape *toMPSShape(const Shape &shape) {
    NSMutableArray<NSNumber *> *s =
        [NSMutableArray arrayWithCapacity:shape.size()];
    for (size_t d : shape)
        [s addObject:@(d)];
    return s;
}

static MPSGraphTensorData *makeTensorData(const Tensor &t) {
    auto *ms = static_cast<const backends::metal::MetalStorage *>(
        t.storage().get());
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
// OpType -> MPSGraph operation mapping
// ============================================================================

using UnaryBuilder = std::function<MPSGraphTensor *(MPSGraph *,
                                                    MPSGraphTensor *)>;
using BinaryBuilder = std::function<MPSGraphTensor *(
    MPSGraph *, MPSGraphTensor *, MPSGraphTensor *)>;

static UnaryBuilder get_mps_unary(ops::OpType op) {
    using ops::OpType;
    switch (op) {
    case OpType::Negate:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g negativeWithTensor:a name:nil];
        };
    case OpType::Abs:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g absoluteWithTensor:a name:nil];
        };
    case OpType::Sqrt:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g squareRootWithTensor:a name:nil];
        };
    case OpType::Exp:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g exponentWithTensor:a name:nil];
        };
    case OpType::Log:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g logarithmWithTensor:a name:nil];
        };
    case OpType::Sin:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g sinWithTensor:a name:nil];
        };
    case OpType::Cos:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g cosWithTensor:a name:nil];
        };
    case OpType::Tan:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g tanWithTensor:a name:nil];
        };
    case OpType::Tanh:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g tanhWithTensor:a name:nil];
        };
    case OpType::Erf:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g erfWithTensor:a name:nil];
        };
    case OpType::Cbrt:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            MPSGraphTensor *abs_a = [g absoluteWithTensor:a name:nil];
            MPSGraphTensor *third =
                [g constantWithScalar:(1.0 / 3.0) dataType:a.dataType];
            MPSGraphTensor *r =
                [g powerWithPrimaryTensor:abs_a secondaryTensor:third name:nil];
            MPSGraphTensor *s = [g signWithTensor:a name:nil];
            return [g multiplicationWithPrimaryTensor:s
                                      secondaryTensor:r
                                                 name:nil];
        };
    case OpType::Square:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g squareWithTensor:a name:nil];
        };
    case OpType::Reciprocal:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g reciprocalWithTensor:a name:nil];
        };
    case OpType::Sign:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g signWithTensor:a name:nil];
        };
    case OpType::Floor:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g floorWithTensor:a name:nil];
        };
    case OpType::Ceil:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g ceilWithTensor:a name:nil];
        };
    case OpType::Round:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g roundWithTensor:a name:nil];
        };
    case OpType::Trunc:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g truncateWithTensor:a name:nil];
        };
    case OpType::ReLU:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g reLUWithTensor:a name:nil];
        };
    case OpType::Sigmoid:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            return [g sigmoidWithTensor:a name:nil];
        };
    case OpType::SiLU:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            MPSGraphTensor *sig = [g sigmoidWithTensor:a name:nil];
            return [g multiplicationWithPrimaryTensor:a
                                      secondaryTensor:sig
                                                 name:nil];
        };
    case OpType::GELU:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
            MPSGraphTensor *half =
                [g constantWithScalar:0.5 dataType:a.dataType];
            MPSGraphTensor *inv_sqrt2 =
                [g constantWithScalar:0.7071067811865475 dataType:a.dataType];
            MPSGraphTensor *scaled =
                [g multiplicationWithPrimaryTensor:a
                                   secondaryTensor:inv_sqrt2
                                              name:nil];
            MPSGraphTensor *e = [g erfWithTensor:scaled name:nil];
            MPSGraphTensor *one =
                [g constantWithScalar:1.0 dataType:a.dataType];
            MPSGraphTensor *inner =
                [g additionWithPrimaryTensor:one secondaryTensor:e name:nil];
            MPSGraphTensor *h =
                [g multiplicationWithPrimaryTensor:half
                                   secondaryTensor:inner
                                              name:nil];
            return [g multiplicationWithPrimaryTensor:a
                                      secondaryTensor:h
                                                 name:nil];
        };
    case OpType::LogicalNot:
        return [](MPSGraph *g, MPSGraphTensor *a) {
            MPSGraphTensor *b =
                [g castTensor:a toType:MPSDataTypeBool name:nil];
            return [g notWithTensor:b name:nil];
        };
    default:
        return nullptr;
    }
}

static BinaryBuilder get_mps_binary(ops::OpType op) {
    using ops::OpType;
    switch (op) {
    case OpType::Add:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
    case OpType::Subtract:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g subtractionWithPrimaryTensor:a
                                   secondaryTensor:b
                                              name:nil];
        };
    case OpType::Multiply:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g multiplicationWithPrimaryTensor:a
                                      secondaryTensor:b
                                                 name:nil];
        };
    case OpType::Divide:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
    case OpType::Maximum:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
    case OpType::Minimum:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
    case OpType::Power:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g powerWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
    case OpType::Atan2:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g atan2WithPrimaryTensor:a secondaryTensor:b name:nil];
        };
    case OpType::Modulo:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            return [g moduloWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
    case OpType::Hypot:
        return [](MPSGraph *g, MPSGraphTensor *a, MPSGraphTensor *b) {
            MPSGraphTensor *a2 =
                [g multiplicationWithPrimaryTensor:a secondaryTensor:a name:nil];
            MPSGraphTensor *b2 =
                [g multiplicationWithPrimaryTensor:b secondaryTensor:b name:nil];
            MPSGraphTensor *s =
                [g additionWithPrimaryTensor:a2 secondaryTensor:b2 name:nil];
            return [g squareRootWithTensor:s name:nil];
        };
    default:
        return nullptr;
    }
}

// ============================================================================
// Cache key for fused chain graphs
// ============================================================================

struct FusedChainCacheKey {
    std::vector<ops::OpType> ops;
    std::vector<Shape> input_shapes;
    std::vector<DType> input_dtypes;
    DType output_dtype;
    // For reductions
    ops::OpType reduction_op = ops::OpType::Add; // sentinel
    std::vector<int> reduction_axes;
    bool keep_dims = false;

    bool operator==(const FusedChainCacheKey &o) const {
        return ops == o.ops && input_shapes == o.input_shapes &&
               input_dtypes == o.input_dtypes &&
               output_dtype == o.output_dtype &&
               reduction_op == o.reduction_op &&
               reduction_axes == o.reduction_axes &&
               keep_dims == o.keep_dims;
    }
};

struct FusedChainCacheKeyHash {
    size_t operator()(const FusedChainCacheKey &k) const {
        size_t h = 0xcbf29ce484222325ULL;
        auto mix = [&](uint64_t v) {
            h ^= v;
            h *= 0x100000001b3ULL;
        };
        for (auto op : k.ops)
            mix(static_cast<uint64_t>(op));
        for (auto &s : k.input_shapes)
            for (auto d : s)
                mix(d);
        for (auto dt : k.input_dtypes)
            mix(static_cast<uint64_t>(dt));
        mix(static_cast<uint64_t>(k.output_dtype));
        mix(static_cast<uint64_t>(k.reduction_op));
        for (auto a : k.reduction_axes)
            mix(static_cast<uint64_t>(a));
        mix(k.keep_dims ? 1ULL : 0ULL);
        return h;
    }
};

struct CachedFusedGraph {
    MPSGraph *graph;
    NSArray<MPSGraphTensor *> *placeholders; // external inputs
    MPSGraphTensor *output;
};

// Simple LRU cache for fused MPSGraph chains
static std::unordered_map<FusedChainCacheKey, CachedFusedGraph,
                          FusedChainCacheKeyHash>
    g_fused_cache;
static std::mutex g_fused_cache_mutex;
static constexpr size_t MAX_FUSED_CACHE = 256;

// Make a GPU tensor contiguous if needed (calls into metal backend)
static Tensor ensureGPUContiguous(const Tensor &t) {
    if (t.is_contiguous())
        return t;
    // Copy to contiguous via to() which triggers a gather
    Tensor c(t.shape(), t.dtype(), Device::GPU);
    // Use the storage copy path
    auto *src =
        static_cast<const backends::metal::MetalStorage *>(t.storage().get());
    auto *dst = static_cast<backends::metal::MetalStorage *>(c.storage().get());
    id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)src->buffer();
    id<MTLBuffer> dst_buf = (__bridge id<MTLBuffer>)dst->buffer();
    size_t bytes = ShapeUtils::size(t.shape()) * dtype_size(t.dtype());
    auto &stream = backends::metal::MetalExecutionStream::instance();
    id<MTLCommandBuffer> cmd =
        (__bridge id<MTLCommandBuffer>)stream.current_buffer();
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:src_buf
            sourceOffset:0
                toBuffer:dst_buf
       destinationOffset:0
                    size:bytes];
    [blit endEncoding];
    stream.increment_batch();
    return c;
}

// ============================================================================
// Build and execute a fused elementwise chain as a single MPSGraph
// ============================================================================

bool execute_gpu_fused_chain(const ExecutionStep &step,
                             std::vector<Tensor> &buffers) {
    @autoreleasepool {
        if (step.op_chain.empty())
            return false;

        // Bail out for dtypes not supported by MPS (Float64, Complex, etc.)
        if (!is_mps_supported_dtype(step.output_dtype))
            return false;

        // Collect unique external input slots (not chain-internal -1)
        std::vector<int> external_slots;
        for (const auto &per_op : step.input_slot_indices) {
            for (int s : per_op) {
                if (s >= 0) {
                    bool found = false;
                    for (int e : external_slots) {
                        if (e == s) {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        external_slots.push_back(s);
                }
            }
        }

        // Ensure GPU inputs are available and contiguous
        std::vector<Tensor> inputs;
        inputs.reserve(external_slots.size());
        for (int slot : external_slots) {
            if (slot < 0 || slot >= static_cast<int>(buffers.size()))
                return false;
            Tensor t = buffers[slot];
            if (t.device() != Device::GPU)
                t = t.to(Device::GPU);
            if (!t.is_contiguous())
                t = ensureGPUContiguous(t);
            inputs.push_back(t);
        }

        // Build cache key
        FusedChainCacheKey key;
        key.ops = step.op_chain;
        for (auto &t : inputs) {
            key.input_shapes.push_back(t.shape());
            key.input_dtypes.push_back(t.dtype());
        }
        key.output_dtype = step.output_dtype;

        // Map from buffer slot index to position in external_slots
        auto slot_to_input = [&](int slot) -> int {
            for (size_t i = 0; i < external_slots.size(); ++i) {
                if (external_slots[i] == slot)
                    return static_cast<int>(i);
            }
            return -1;
        };

        // Try cache
        CachedFusedGraph cached{};
        bool cache_hit = false;
        {
            std::lock_guard<std::mutex> lock(g_fused_cache_mutex);
            auto it = g_fused_cache.find(key);
            if (it != g_fused_cache.end()) {
                cached = it->second;
                cache_hit = true;
            }
        }

        if (!cache_hit) {
            // Build the MPSGraph
            MPSGraph *graph = [[MPSGraph alloc] init];

            // Create placeholders for each external input
            NSMutableArray<MPSGraphTensor *> *placeholders =
                [NSMutableArray arrayWithCapacity:inputs.size()];
            for (size_t i = 0; i < inputs.size(); ++i) {
                MPSGraphTensor *ph =
                    [graph placeholderWithShape:toMPSShape(inputs[i].shape())
                                      dataType:toMPS(inputs[i].dtype())
                                          name:nil];
                [placeholders addObject:ph];
            }

            // Walk the op chain, threading results through
            MPSGraphTensor *prev = nil;
            for (size_t oi = 0; oi < step.op_chain.size(); ++oi) {
                ops::OpType op = step.op_chain[oi];
                const auto &indices = step.input_slot_indices[oi];

                auto resolve = [&](int idx) -> MPSGraphTensor * {
                    if (idx == -1)
                        return prev;
                    int ext_idx = slot_to_input(idx);
                    if (ext_idx >= 0 &&
                        ext_idx < static_cast<int>(placeholders.count))
                        return placeholders[ext_idx];
                    return prev; // fallback
                };

                if (is_unary_op(op)) {
                    auto builder = get_mps_unary(op);
                    if (!builder)
                        return false;
                    MPSGraphTensor *in =
                        resolve(indices.empty() ? -1 : indices[0]);
                    prev = builder(graph, in);
                } else if (is_binary_op(op)) {
                    auto builder = get_mps_binary(op);
                    if (!builder)
                        return false;
                    MPSGraphTensor *a =
                        resolve(indices.empty() ? -1 : indices[0]);
                    MPSGraphTensor *b =
                        resolve(indices.size() > 1 ? indices[1] : -1);
                    prev = builder(graph, a, b);
                } else {
                    return false; // unsupported op kind in chain
                }
            }

            if (!prev)
                return false;

            // Cast to output dtype if needed
            if (toMPS(step.output_dtype) != prev.dataType) {
                prev = [graph castTensor:prev
                                  toType:toMPS(step.output_dtype)
                                    name:nil];
            }

            cached.graph = graph;
            cached.placeholders = [placeholders copy];
            cached.output = prev;

            // Store in cache
            {
                std::lock_guard<std::mutex> lock(g_fused_cache_mutex);
                if (g_fused_cache.size() >= MAX_FUSED_CACHE)
                    g_fused_cache.clear(); // simple eviction
                g_fused_cache[key] = cached;
            }
        }

        // Allocate output
        Tensor output(step.output_shape, step.output_dtype, Device::GPU);

        // Build feeds dictionary
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary
                dictionaryWithCapacity:cached.placeholders.count];
        for (NSUInteger i = 0; i < cached.placeholders.count; ++i) {
            feeds[cached.placeholders[i]] = makeTensorData(inputs[i]);
        }

        // Build targets
        MPSGraphTensorData *out_data = makeTensorData(output);
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *targets =
            @{cached.output : out_data};

        // Encode and execute
        encodeMPSGraphAsync(cached.graph, feeds, targets);

        if (step.output_slot >= 0 &&
            step.output_slot < static_cast<int>(buffers.size())) {
            buffers[step.output_slot] = output;
        }

        return true;
    }
}

// ============================================================================
// Build and execute a fused elementwise + reduction as single MPSGraph
// ============================================================================

bool execute_gpu_fused_reduction(const ExecutionStep &step,
                                 std::vector<Tensor> &buffers) {
    @autoreleasepool {
        if (step.op_chain.empty())
            return false;

        // Bail out for dtypes not supported by MPS (Float64, Complex, etc.)
        if (!is_mps_supported_dtype(step.output_dtype))
            return false;

        // Collect external inputs
        std::vector<int> external_slots;
        for (const auto &per_op : step.input_slot_indices) {
            for (int s : per_op) {
                if (s >= 0) {
                    bool found = false;
                    for (int e : external_slots) {
                        if (e == s) {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        external_slots.push_back(s);
                }
            }
        }

        std::vector<Tensor> inputs;
        inputs.reserve(external_slots.size());
        for (int slot : external_slots) {
            if (slot < 0 || slot >= static_cast<int>(buffers.size()))
                return false;
            Tensor t = buffers[slot];
            if (t.device() != Device::GPU)
                t = t.to(Device::GPU);
            if (!t.is_contiguous())
                t = ensureGPUContiguous(t);
            inputs.push_back(t);
        }

        // Build cache key (includes reduction info)
        FusedChainCacheKey key;
        key.ops = step.op_chain;
        for (auto &t : inputs) {
            key.input_shapes.push_back(t.shape());
            key.input_dtypes.push_back(t.dtype());
        }
        key.output_dtype = step.output_dtype;
        key.reduction_op = step.reduction_op;
        key.reduction_axes = step.params.axes;
        key.keep_dims = step.params.keep_dims;

        auto slot_to_input = [&](int slot) -> int {
            for (size_t i = 0; i < external_slots.size(); ++i) {
                if (external_slots[i] == slot)
                    return static_cast<int>(i);
            }
            return -1;
        };

        CachedFusedGraph cached{};
        bool cache_hit = false;
        {
            std::lock_guard<std::mutex> lock(g_fused_cache_mutex);
            auto it = g_fused_cache.find(key);
            if (it != g_fused_cache.end()) {
                cached = it->second;
                cache_hit = true;
            }
        }

        if (!cache_hit) {
            MPSGraph *graph = [[MPSGraph alloc] init];

            NSMutableArray<MPSGraphTensor *> *placeholders =
                [NSMutableArray arrayWithCapacity:inputs.size()];
            for (size_t i = 0; i < inputs.size(); ++i) {
                MPSGraphTensor *ph =
                    [graph placeholderWithShape:toMPSShape(inputs[i].shape())
                                      dataType:toMPS(inputs[i].dtype())
                                          name:nil];
                [placeholders addObject:ph];
            }

            // Build elementwise chain
            MPSGraphTensor *prev = nil;
            for (size_t oi = 0; oi < step.op_chain.size(); ++oi) {
                ops::OpType op = step.op_chain[oi];
                const auto &indices = step.input_slot_indices[oi];

                auto resolve = [&](int idx) -> MPSGraphTensor * {
                    if (idx == -1)
                        return prev;
                    int ext_idx = slot_to_input(idx);
                    if (ext_idx >= 0 &&
                        ext_idx < static_cast<int>(placeholders.count))
                        return placeholders[ext_idx];
                    return prev;
                };

                if (is_unary_op(op)) {
                    auto builder = get_mps_unary(op);
                    if (!builder)
                        return false;
                    prev = builder(graph,
                                   resolve(indices.empty() ? -1 : indices[0]));
                } else if (is_binary_op(op)) {
                    auto builder = get_mps_binary(op);
                    if (!builder)
                        return false;
                    prev = builder(
                        graph, resolve(indices.empty() ? -1 : indices[0]),
                        resolve(indices.size() > 1 ? indices[1] : -1));
                } else {
                    return false;
                }
            }

            if (!prev)
                return false;

            // Build reduction axes
            std::vector<int> axes = step.params.axes;
            if (axes.empty()) {
                // Full reduction: all axes
                // Infer rank from the first input
                if (!inputs.empty()) {
                    for (size_t i = 0; i < inputs[0].ndim(); ++i)
                        axes.push_back(static_cast<int>(i));
                }
            }
            NSMutableArray<NSNumber *> *mps_axes =
                [NSMutableArray arrayWithCapacity:axes.size()];
            for (int a : axes)
                [mps_axes addObject:@(a)];

            // Apply reduction
            MPSGraphTensor *reduced = nil;
            switch (step.reduction_op) {
            case ops::OpType::Sum:
                reduced = [graph reductionSumWithTensor:prev
                                                  axes:mps_axes
                                                  name:nil];
                break;
            case ops::OpType::Mean:
                reduced =
                    [graph meanOfTensor:prev axes:mps_axes name:nil];
                break;
            case ops::OpType::Max:
                reduced = [graph reductionMaximumWithTensor:prev
                                                      axes:mps_axes
                                                      name:nil];
                break;
            case ops::OpType::Min:
                reduced = [graph reductionMinimumWithTensor:prev
                                                      axes:mps_axes
                                                      name:nil];
                break;
            case ops::OpType::Prod:
                reduced = [graph reductionProductWithTensor:prev
                                                      axes:mps_axes
                                                      name:nil];
                break;
            default:
                return false;
            }

            if (!reduced)
                return false;

            // Cast to output dtype if needed
            if (toMPS(step.output_dtype) != reduced.dataType) {
                reduced = [graph castTensor:reduced
                                    toType:toMPS(step.output_dtype)
                                      name:nil];
            }

            cached.graph = graph;
            cached.placeholders = [placeholders copy];
            cached.output = reduced;

            {
                std::lock_guard<std::mutex> lock(g_fused_cache_mutex);
                if (g_fused_cache.size() >= MAX_FUSED_CACHE)
                    g_fused_cache.clear();
                g_fused_cache[key] = cached;
            }
        }

        // Allocate output
        Tensor output(step.output_shape, step.output_dtype, Device::GPU);

        // Feeds
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary
                dictionaryWithCapacity:cached.placeholders.count];
        for (NSUInteger i = 0; i < cached.placeholders.count; ++i) {
            feeds[cached.placeholders[i]] = makeTensorData(inputs[i]);
        }

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *targets =
            @{cached.output : makeTensorData(output)};

        encodeMPSGraphAsync(cached.graph, feeds, targets);

        if (step.output_slot >= 0 &&
            step.output_slot < static_cast<int>(buffers.size())) {
            buffers[step.output_slot] = output;
        }

        return true;
    }
}

} // namespace graph
} // namespace axiom
