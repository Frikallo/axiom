//
//  flash_attention.metal
//  axiom
//
//  Flash Attention v2 Metal kernel
//  Tiled online softmax with fused Q@K^T, masking, softmax, and P@V
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Parameters
// ============================================================================

struct FlashAttentionParams {
    uint batch_size;
    uint num_heads;
    uint seq_q;
    uint seq_kv;
    uint head_dim;
    float scale;
    uint is_causal;
    uint has_mask;
    // Q/K/V/O strides in elements (not bytes)
    uint q_batch_stride;
    uint q_head_stride;
    uint k_batch_stride;
    uint k_head_stride;
    uint v_batch_stride;
    uint v_head_stride;
    uint o_batch_stride;
    uint o_head_stride;
    // Mask strides (elements)
    uint mask_batch_stride;
    uint mask_head_stride;
};

// ============================================================================
// Flash Attention v2 Kernel
// ============================================================================
//
// Grid: (ceil(seq_q / BLOCK_Q), batch_size * num_heads, 1)
// Threadgroup: (THREADS_PER_GROUP, 1, 1)
//
// Each threadgroup processes one Q-block for one (batch, head) pair.
// Outer loop = Q blocks (parallel across threadgroups)
// Inner loop = K/V blocks (sequential within threadgroup)

template <typename T, ushort HEAD_DIM, ushort BLOCK_Q, ushort BLOCK_KV, ushort THREADS>
kernel void flash_attention_v2(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    device const uint8_t* mask [[buffer(4)]],
    constant FlashAttentionParams& params [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    // Decode threadgroup ID
    uint q_block_idx = tgid.x;
    uint batch_head = tgid.y;
    uint b = batch_head / params.num_heads;
    uint h = batch_head % params.num_heads;

    uint q_start = q_block_idx * BLOCK_Q;
    if (q_start >= params.seq_q) return;
    uint actual_bq = min((uint)BLOCK_Q, params.seq_q - q_start);

    // Pointers for this (batch, head) slice
    device const T* q_base = Q + b * params.q_batch_stride + h * params.q_head_stride;
    device const T* k_base = K + b * params.k_batch_stride + h * params.k_head_stride;
    device const T* v_base = V + b * params.v_batch_stride + h * params.v_head_stride;
    device T* o_base = O + b * params.o_batch_stride + h * params.o_head_stride;

    device const uint8_t* mask_base = nullptr;
    if (params.has_mask) {
        mask_base = mask + b * params.mask_batch_stride + h * params.mask_head_stride;
    }

    // Shared memory layout:
    // q_shared: (BLOCK_Q, HEAD_DIM) — loaded once
    // kv_shared: (BLOCK_KV, HEAD_DIM) — reused for K then V
    // s_shared: (BLOCK_Q, BLOCK_KV) — attention scores
    threadgroup float q_shared[BLOCK_Q * HEAD_DIM];
    threadgroup float kv_shared[BLOCK_KV * HEAD_DIM];
    threadgroup float s_shared[BLOCK_Q * BLOCK_KV];

    // Per-row accumulators in registers (distributed across threads)
    // Each thread handles a subset of Q rows
    constexpr ushort ROWS_PER_THREAD = (BLOCK_Q + THREADS - 1) / THREADS;

    float o_acc[ROWS_PER_THREAD * HEAD_DIM];
    float m_val[ROWS_PER_THREAD];
    float l_val[ROWS_PER_THREAD];

    for (ushort r = 0; r < ROWS_PER_THREAD; ++r) {
        m_val[r] = -INFINITY;
        l_val[r] = 0.0f;
        for (ushort d = 0; d < HEAD_DIM; ++d) {
            o_acc[r * HEAD_DIM + d] = 0.0f;
        }
    }

    // Load Q block into shared memory (collaborative load)
    uint total_q_elems = actual_bq * HEAD_DIM;
    for (uint idx = tid; idx < total_q_elems; idx += THREADS) {
        uint row = idx / HEAD_DIM;
        uint col = idx % HEAD_DIM;
        q_shared[row * HEAD_DIM + col] = float(q_base[(q_start + row) * params.head_dim + col]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inner loop over K/V blocks
    for (uint kv_start = 0; kv_start < params.seq_kv; kv_start += BLOCK_KV) {
        uint actual_bkv = min((uint)BLOCK_KV, params.seq_kv - kv_start);

        // Load K block into shared memory
        uint total_kv_elems = actual_bkv * HEAD_DIM;
        for (uint idx = tid; idx < total_kv_elems; idx += THREADS) {
            uint row = idx / HEAD_DIM;
            uint col = idx % HEAD_DIM;
            kv_shared[row * HEAD_DIM + col] = float(k_base[(kv_start + row) * params.head_dim + col]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute S = Q_block @ K_block^T * scale
        // Each thread computes a subset of rows of S
        for (ushort r = 0; r < ROWS_PER_THREAD; ++r) {
            uint qi = tid + r * THREADS;
            if (qi >= actual_bq) continue;

            for (uint kj = 0; kj < actual_bkv; ++kj) {
                float dot = 0.0f;
                for (ushort d = 0; d < HEAD_DIM; ++d) {
                    dot += q_shared[qi * HEAD_DIM + d] * kv_shared[kj * HEAD_DIM + d];
                }
                dot *= params.scale;

                // Apply causal mask
                if (params.is_causal && (kv_start + kj) > (q_start + qi)) {
                    dot = -INFINITY;
                }

                // Apply explicit mask
                if (params.has_mask && mask_base) {
                    if (mask_base[(q_start + qi) * params.seq_kv + (kv_start + kj)]) {
                        dot = -INFINITY;
                    }
                }

                s_shared[qi * BLOCK_KV + kj] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: update m, l, rescale O, compute P
        for (ushort r = 0; r < ROWS_PER_THREAD; ++r) {
            uint qi = tid + r * THREADS;
            if (qi >= actual_bq) continue;

            // Find row max
            float row_max = -INFINITY;
            for (uint kj = 0; kj < actual_bkv; ++kj) {
                row_max = max(row_max, s_shared[qi * BLOCK_KV + kj]);
            }

            float m_new = max(m_val[r], row_max);
            float alpha = exp(m_val[r] - m_new);

            // Compute exp(S - m_new) and row sum
            float row_sum = 0.0f;
            for (uint kj = 0; kj < actual_bkv; ++kj) {
                float p = exp(s_shared[qi * BLOCK_KV + kj] - m_new);
                s_shared[qi * BLOCK_KV + kj] = p;  // Reuse s_shared for P
                row_sum += p;
            }

            // Rescale O accumulator
            for (ushort d = 0; d < HEAD_DIM; ++d) {
                o_acc[r * HEAD_DIM + d] *= alpha;
            }

            l_val[r] = alpha * l_val[r] + row_sum;
            m_val[r] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V block into shared memory (reusing kv_shared)
        for (uint idx = tid; idx < total_kv_elems; idx += THREADS) {
            uint row = idx / HEAD_DIM;
            uint col = idx % HEAD_DIM;
            kv_shared[row * HEAD_DIM + col] = float(v_base[(kv_start + row) * params.head_dim + col]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P @ V_block
        for (ushort r = 0; r < ROWS_PER_THREAD; ++r) {
            uint qi = tid + r * THREADS;
            if (qi >= actual_bq) continue;

            for (ushort d = 0; d < HEAD_DIM; ++d) {
                float sum = 0.0f;
                for (uint kj = 0; kj < actual_bkv; ++kj) {
                    sum += s_shared[qi * BLOCK_KV + kj] * kv_shared[kj * HEAD_DIM + d];
                }
                o_acc[r * HEAD_DIM + d] += sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization: O = O / l and write to global memory
    for (ushort r = 0; r < ROWS_PER_THREAD; ++r) {
        uint qi = tid + r * THREADS;
        if (qi >= actual_bq) continue;

        float inv_l = (l_val[r] > 0.0f) ? (1.0f / l_val[r]) : 0.0f;
        for (ushort d = 0; d < HEAD_DIM; ++d) {
            o_base[(q_start + qi) * params.head_dim + d] = T(o_acc[r * HEAD_DIM + d] * inv_l);
        }
    }
}

// ============================================================================
// Explicit Instantiations
// ============================================================================

// Float32, head_dim=32
template [[host_name("flash_attention_v2_float_d32")]]
kernel void flash_attention_v2<float, 32, 32, 32, 128>(
    device const float*, device const float*, device const float*,
    device float*, device const uint8_t*,
    constant FlashAttentionParams&,
    uint2, uint);

// Float32, head_dim=64
template [[host_name("flash_attention_v2_float_d64")]]
kernel void flash_attention_v2<float, 64, 32, 32, 128>(
    device const float*, device const float*, device const float*,
    device float*, device const uint8_t*,
    constant FlashAttentionParams&,
    uint2, uint);

// Float32, head_dim=128
template [[host_name("flash_attention_v2_float_d128")]]
kernel void flash_attention_v2<float, 128, 16, 32, 128>(
    device const float*, device const float*, device const float*,
    device float*, device const uint8_t*,
    constant FlashAttentionParams&,
    uint2, uint);

// Float16, head_dim=32
template [[host_name("flash_attention_v2_half_d32")]]
kernel void flash_attention_v2<half, 32, 64, 64, 128>(
    device const half*, device const half*, device const half*,
    device half*, device const uint8_t*,
    constant FlashAttentionParams&,
    uint2, uint);

// Float16, head_dim=64
template [[host_name("flash_attention_v2_half_d64")]]
kernel void flash_attention_v2<half, 64, 64, 64, 128>(
    device const half*, device const half*, device const half*,
    device half*, device const uint8_t*,
    constant FlashAttentionParams&,
    uint2, uint);

// Float16, head_dim=128
template [[host_name("flash_attention_v2_half_d128")]]
kernel void flash_attention_v2<half, 128, 32, 64, 128>(
    device const half*, device const half*, device const half*,
    device half*, device const uint8_t*,
    constant FlashAttentionParams&,
    uint2, uint);
