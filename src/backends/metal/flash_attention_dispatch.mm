#import "metal_operations.hpp"
#import "metal_common.hpp"
#import "metal_buffer_provider.hpp"

#import "axiom/error.hpp"
#import "axiom/tensor.hpp"

#import <Metal/Metal.h>
#import <map>
#import <string>
#import <utility>

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// FlashAttentionParams — must match the Metal kernel struct exactly
// ============================================================================

struct FlashAttentionParams {
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t head_dim;
    float scale;
    uint32_t is_causal;
    uint32_t has_mask;
    uint32_t q_batch_stride;
    uint32_t q_head_stride;
    uint32_t k_batch_stride;
    uint32_t k_head_stride;
    uint32_t v_batch_stride;
    uint32_t v_head_stride;
    uint32_t o_batch_stride;
    uint32_t o_head_stride;
    uint32_t mask_batch_stride;
    uint32_t mask_head_stride;
};

// ============================================================================
// Pipeline Cache
// ============================================================================

// Key: (dtype, head_dim) -> pipeline state
static std::map<std::pair<DType, int>, id<MTLComputePipelineState>>
    g_flash_attn_pipelines;

// Block size config for a given (dtype, head_dim)
struct BlockConfig {
    int block_q;
    int block_kv;
    int threads;
};

static BlockConfig get_block_config(DType dtype, int head_dim) {
    if (dtype == DType::Float16) {
        switch (head_dim) {
        case 32:
            return {64, 64, 128};
        case 64:
            return {64, 64, 128};
        case 128:
            return {32, 64, 128};
        default:
            return {0, 0, 0}; // unsupported
        }
    } else { // Float32
        switch (head_dim) {
        case 32:
            return {32, 32, 128};
        case 64:
            return {32, 32, 128};
        case 128:
            return {16, 32, 128};
        default:
            return {0, 0, 0}; // unsupported
        }
    }
}

static std::string get_kernel_name(DType dtype, int head_dim) {
    std::string type_str = (dtype == DType::Float16) ? "half" : "float";
    return "flash_attention_v2_" + type_str + "_d" + std::to_string(head_dim);
}

static id<MTLComputePipelineState>
getFlashAttentionPipeline(DType dtype, int head_dim) {
    auto key = std::make_pair(dtype, head_dim);
    auto it = g_flash_attn_pipelines.find(key);
    if (it != g_flash_attn_pipelines.end()) {
        return it->second;
    }

    std::string kernel_name = get_kernel_name(dtype, head_dim);

    id<MTLDevice> device =
        (__bridge id<MTLDevice>)MetalContext::instance().device();
    id<MTLLibrary> library =
        (__bridge id<MTLLibrary>)get_default_library();

    id<MTLFunction> function = [library
        newFunctionWithName:[NSString
                                stringWithUTF8String:kernel_name.c_str()]];
    if (!function) {
        return nil; // Kernel not available for this config
    }

    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        return nil;
    }

    g_flash_attn_pipelines[key] = pipeline;
    return pipeline;
}

// ============================================================================
// Decomposed fallback: matmul + scale + mask + softmax + matmul
// ============================================================================

static Tensor decomposed_attention(const Tensor &query, const Tensor &key,
                                   const Tensor &value, const Tensor &mask,
                                   float scale, bool is_causal) {
    // S = Q @ K^T * scale
    auto scores = ops::matmul(query, key, false, true);
    auto scale_tensor =
        Tensor::full({1}, scale, scores.device()).astype(scores.dtype());
    scores = ops::multiply(scores, scale_tensor);

    // Apply causal mask
    if (is_causal) {
        auto seq_q = static_cast<int>(query.shape()[2]);
        auto seq_kv = static_cast<int>(key.shape()[2]);
        // Create upper-triangular mask (true = masked out)
        auto causal_mask =
            Tensor::zeros({static_cast<size_t>(seq_q),
                           static_cast<size_t>(seq_kv)},
                          DType::Bool, Device::CPU);
        auto *mask_data = causal_mask.typed_data<uint8_t>();
        for (int i = 0; i < seq_q; ++i) {
            for (int j = i + 1; j < seq_kv; ++j) {
                mask_data[i * seq_kv + j] = 1;
            }
        }
        causal_mask = causal_mask.to(scores.device());
        scores =
            ops::masked_fill(scores.ascontiguousarray(), causal_mask, -1e9f);
    }

    // Apply explicit mask
    if (mask.storage()) {
        scores =
            ops::masked_fill(scores.ascontiguousarray(), mask, -1e9f);
    }

    auto attn_weights = ops::softmax(scores, -1);
    return ops::matmul(attn_weights, value);
}

// ============================================================================
// GPU Dispatch
// ============================================================================

Tensor gpu_scaled_dot_product_attention(const Tensor &query, const Tensor &key,
                                        const Tensor &value,
                                        const Tensor &mask, float scale,
                                        bool is_causal) {
    auto batch = static_cast<int>(query.shape()[0]);
    auto heads = static_cast<int>(query.shape()[1]);
    auto seq_q = static_cast<int>(query.shape()[2]);
    auto seq_kv = static_cast<int>(key.shape()[2]);
    auto head_dim = static_cast<int>(query.shape()[3]);

    DType dtype = query.dtype();

    // Only Float32 and Float16 supported by custom kernel
    if (dtype != DType::Float32 && dtype != DType::Float16) {
        return decomposed_attention(query, key, value, mask, scale, is_causal);
    }

    // Check if we have a custom kernel for this head_dim
    BlockConfig config = get_block_config(dtype, head_dim);
    if (config.block_q == 0) {
        // No custom kernel for this head_dim — fall back to decomposed
        return decomposed_attention(query, key, value, mask, scale, is_causal);
    }

    id<MTLComputePipelineState> pipeline =
        getFlashAttentionPipeline(dtype, head_dim);
    if (!pipeline) {
        return decomposed_attention(query, key, value, mask, scale, is_causal);
    }

    // Ensure inputs are contiguous
    Tensor q = query.is_contiguous() ? query : query.ascontiguousarray();
    Tensor k = key.is_contiguous() ? key : key.ascontiguousarray();
    Tensor v = value.is_contiguous() ? value : value.ascontiguousarray();

    // Allocate output
    Tensor output({static_cast<size_t>(batch), static_cast<size_t>(heads),
                   static_cast<size_t>(seq_q),
                   static_cast<size_t>(head_dim)},
                  dtype, Device::GPU);

    // Fill params struct
    FlashAttentionParams params;
    params.batch_size = static_cast<uint32_t>(batch);
    params.num_heads = static_cast<uint32_t>(heads);
    params.seq_q = static_cast<uint32_t>(seq_q);
    params.seq_kv = static_cast<uint32_t>(seq_kv);
    params.head_dim = static_cast<uint32_t>(head_dim);
    params.scale = scale;
    params.is_causal = is_causal ? 1 : 0;
    params.has_mask = mask.storage() ? 1 : 0;

    // Element strides for contiguous 4D tensor (batch, heads, seq, head_dim)
    params.q_batch_stride = static_cast<uint32_t>(heads * seq_q * head_dim);
    params.q_head_stride = static_cast<uint32_t>(seq_q * head_dim);
    params.k_batch_stride = static_cast<uint32_t>(heads * seq_kv * head_dim);
    params.k_head_stride = static_cast<uint32_t>(seq_kv * head_dim);
    params.v_batch_stride = static_cast<uint32_t>(heads * seq_kv * head_dim);
    params.v_head_stride = static_cast<uint32_t>(seq_kv * head_dim);
    params.o_batch_stride = static_cast<uint32_t>(heads * seq_q * head_dim);
    params.o_head_stride = static_cast<uint32_t>(seq_q * head_dim);

    params.mask_batch_stride = 0;
    params.mask_head_stride = 0;

    // Get Metal buffers
    auto *q_storage = as_metal_buffer_provider(q.storage().get());
    auto *k_storage = as_metal_buffer_provider(k.storage().get());
    auto *v_storage = as_metal_buffer_provider(v.storage().get());
    auto *o_storage = as_metal_buffer_provider(output.storage().get());

    id<MTLBuffer> q_buffer = (__bridge id<MTLBuffer>)q_storage->buffer();
    id<MTLBuffer> k_buffer = (__bridge id<MTLBuffer>)k_storage->buffer();
    id<MTLBuffer> v_buffer = (__bridge id<MTLBuffer>)v_storage->buffer();
    id<MTLBuffer> o_buffer = (__bridge id<MTLBuffer>)o_storage->buffer();

    // Handle mask buffer
    id<MTLBuffer> mask_buffer = nil;
    Tensor mask_contig;
    if (mask.storage()) {
        mask_contig =
            mask.is_contiguous() ? mask : mask.ascontiguousarray();
        auto *mask_storage =
            as_metal_buffer_provider(mask_contig.storage().get());
        mask_buffer = (__bridge id<MTLBuffer>)mask_storage->buffer();

        if (mask_contig.ndim() == 4) {
            params.mask_batch_stride = static_cast<uint32_t>(
                mask_contig.shape()[1] * mask_contig.shape()[2] *
                mask_contig.shape()[3]);
            params.mask_head_stride = static_cast<uint32_t>(
                mask_contig.shape()[2] * mask_contig.shape()[3]);
        }
    }

    // End any pending MPSGraph encoder before custom kernel
    auto &stream = MetalExecutionStream::instance();
    stream.end_kernel_coalescing();

    // Encode kernel
    id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)stream.compute_encoder();

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:q_buffer offset:q_storage->offset() atIndex:0];
    [encoder setBuffer:k_buffer offset:k_storage->offset() atIndex:1];
    [encoder setBuffer:v_buffer offset:v_storage->offset() atIndex:2];
    [encoder setBuffer:o_buffer offset:o_storage->offset() atIndex:3];

    if (mask_buffer) {
        auto *ms = as_metal_buffer_provider(mask_contig.storage().get());
        [encoder setBuffer:mask_buffer offset:ms->offset() atIndex:4];
    } else {
        // Bind dummy buffer (use Q buffer, kernel checks has_mask flag)
        [encoder setBuffer:q_buffer offset:0 atIndex:4];
    }

    [encoder setBytes:&params length:sizeof(params) atIndex:5];

    // Grid dimensions
    uint32_t num_q_blocks =
        (static_cast<uint32_t>(seq_q) + config.block_q - 1) / config.block_q;
    uint32_t num_batch_head = static_cast<uint32_t>(batch * heads);

    MTLSize grid_size = MTLSizeMake(num_q_blocks, num_batch_head, 1);
    MTLSize threadgroup_size =
        MTLSizeMake(static_cast<NSUInteger>(config.threads), 1, 1);

    [encoder dispatchThreadgroups:grid_size
            threadsPerThreadgroup:threadgroup_size];

    stream.increment_batch();
    stream.end_kernel_coalescing();
    stream.synchronize();

    return output;
}

} // namespace metal
} // namespace backends
} // namespace axiom
