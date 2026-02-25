#include "axiom/operations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "axiom/dispatch.hpp"
#include "axiom/error.hpp"
#include "axiom/tensor.hpp"
#include "blas/blas_backend.hpp"

namespace axiom {
namespace ops {

// Block sizes for CPU tiled flash attention
// Larger tiles = better BLAS utilization, CPU has plenty of stack/heap
static constexpr int CPU_BLOCK_Q = 64;
static constexpr int CPU_BLOCK_KV = 64;

namespace {

// Core flash attention v2 tiled algorithm for a single (batch, head) slice.
// Q: (seq_q, head_dim), K: (seq_kv, head_dim), V: (seq_kv, head_dim)
// O: (seq_q, head_dim) output buffer (must be pre-allocated)
template <typename T>
void flash_attention_tile(const T *q_ptr, const T *k_ptr, const T *v_ptr,
                          T *o_ptr, int seq_q, int seq_kv, int head_dim,
                          float scale, bool is_causal, const uint8_t *mask_ptr,
                          int mask_seq_stride, int q_block_offset) {
    auto &blas = backends::cpu::blas::get_blas_backend();

    // Scratch buffers for one Q-block
    // S: (BLOCK_Q, BLOCK_KV) — attention scores
    // m: (BLOCK_Q,) — running row-wise max
    // l: (BLOCK_Q,) — running row-wise sum of exp
    // O_acc: (BLOCK_Q, head_dim) — accumulator for output
    const int bq = std::min(CPU_BLOCK_Q, seq_q - q_block_offset);

    std::vector<float> s_buf(static_cast<size_t>(bq) * CPU_BLOCK_KV);
    std::vector<float> o_acc(static_cast<size_t>(bq) * head_dim, 0.0f);
    std::vector<float> m_vec(bq, -std::numeric_limits<float>::infinity());
    std::vector<float> l_vec(bq, 0.0f);

    // Upcast Q block to float for computation
    std::vector<float> q_block(static_cast<size_t>(bq) * head_dim);
    for (int i = 0; i < bq; ++i) {
        for (int j = 0; j < head_dim; ++j) {
            q_block[static_cast<size_t>(i) * head_dim + j] = static_cast<float>(
                q_ptr[static_cast<size_t>(q_block_offset + i) * head_dim + j]);
        }
    }

    // Inner loop over K/V blocks
    for (int kv_offset = 0; kv_offset < seq_kv; kv_offset += CPU_BLOCK_KV) {
        const int bkv = std::min(CPU_BLOCK_KV, seq_kv - kv_offset);

        // Upcast K block to float
        std::vector<float> k_block(static_cast<size_t>(bkv) * head_dim);
        for (int i = 0; i < bkv; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                k_block[static_cast<size_t>(i) * head_dim + j] =
                    static_cast<float>(
                        k_ptr[static_cast<size_t>(kv_offset + i) * head_dim +
                              j]);
            }
        }

        // S = Q_block @ K_block^T * scale
        // S is (bq, bkv), Q_block is (bq, head_dim), K_block is (bkv,
        // head_dim) So: S = Q_block * K_block^T => sgemm(N, T, bq, bkv,
        // head_dim)
        blas.sgemm(false, true, bq, bkv, head_dim, scale, q_block.data(),
                   head_dim, k_block.data(), head_dim, 0.0f, s_buf.data(), bkv);

        // Apply causal mask: zero out S[i][j] where (q_block_offset + i) <
        // (kv_offset + j)
        if (is_causal) {
            for (int i = 0; i < bq; ++i) {
                int q_pos = q_block_offset + i;
                for (int j = 0; j < bkv; ++j) {
                    int kv_pos = kv_offset + j;
                    if (kv_pos > q_pos) {
                        s_buf[static_cast<size_t>(i) * bkv + j] =
                            -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }

        // Apply explicit mask: mask_ptr[q_pos, kv_pos] != 0 means masked out
        if (mask_ptr) {
            for (int i = 0; i < bq; ++i) {
                int q_pos = q_block_offset + i;
                for (int j = 0; j < bkv; ++j) {
                    int kv_pos = kv_offset + j;
                    if (mask_ptr[static_cast<size_t>(q_pos) * mask_seq_stride +
                                 kv_pos]) {
                        s_buf[static_cast<size_t>(i) * bkv + j] =
                            -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }

        // Online softmax update
        // For each row i:
        //   m_new = max(m[i], max(S[i,:]))
        //   alpha = exp(m[i] - m_new)
        //   P[i,j] = exp(S[i,j] - m_new)
        //   l[i] = alpha * l[i] + sum(P[i,:])
        //   O[i,:] = alpha * O[i,:] + P[i,:] @ V_block
        std::vector<float> p_buf(static_cast<size_t>(bq) * bkv);

        for (int i = 0; i < bq; ++i) {
            // Find row max
            float row_max = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < bkv; ++j) {
                row_max =
                    std::max(row_max, s_buf[static_cast<size_t>(i) * bkv + j]);
            }
            float m_new = std::max(m_vec[i], row_max);

            // Rescaling factor for old accumulators
            float alpha = std::exp(m_vec[i] - m_new);

            // Compute P = exp(S - m_new) and row sum
            float row_sum = 0.0f;
            for (int j = 0; j < bkv; ++j) {
                float p =
                    std::exp(s_buf[static_cast<size_t>(i) * bkv + j] - m_new);
                p_buf[static_cast<size_t>(i) * bkv + j] = p;
                row_sum += p;
            }

            // Rescale existing output accumulator
            for (int d = 0; d < head_dim; ++d) {
                o_acc[static_cast<size_t>(i) * head_dim + d] *= alpha;
            }

            // Update running statistics
            l_vec[i] = alpha * l_vec[i] + row_sum;
            m_vec[i] = m_new;
        }

        // Upcast V block to float
        std::vector<float> v_block(static_cast<size_t>(bkv) * head_dim);
        for (int i = 0; i < bkv; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                v_block[static_cast<size_t>(i) * head_dim + j] =
                    static_cast<float>(
                        v_ptr[static_cast<size_t>(kv_offset + i) * head_dim +
                              j]);
            }
        }

        // O_acc += P @ V_block
        // P is (bq, bkv), V_block is (bkv, head_dim)
        // O_acc is (bq, head_dim)
        blas.sgemm(false, false, bq, head_dim, bkv, 1.0f, p_buf.data(), bkv,
                   v_block.data(), head_dim, 1.0f, o_acc.data(), head_dim);
    }

    // Final normalization: O = O_acc / l
    for (int i = 0; i < bq; ++i) {
        float inv_l = (l_vec[i] > 0.0f) ? (1.0f / l_vec[i]) : 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            o_ptr[static_cast<size_t>(q_block_offset + i) * head_dim + d] =
                static_cast<T>(o_acc[static_cast<size_t>(i) * head_dim + d] *
                               inv_l);
        }
    }
}

} // anonymous namespace

Tensor cpu_flash_attention(const Tensor &query, const Tensor &key,
                           const Tensor &value, const Tensor &mask, float scale,
                           bool is_causal) {
    auto batch = static_cast<int>(query.shape()[0]);
    auto heads = static_cast<int>(query.shape()[1]);
    auto seq_q = static_cast<int>(query.shape()[2]);
    auto seq_kv = static_cast<int>(key.shape()[2]);
    auto head_dim = static_cast<int>(query.shape()[3]);

    // Ensure inputs are contiguous for direct pointer access
    Tensor q = query.is_contiguous() ? query : query.ascontiguousarray();
    Tensor k = key.is_contiguous() ? key : key.ascontiguousarray();
    Tensor v = value.is_contiguous() ? value : value.ascontiguousarray();

    // Determine output dtype — compute in float32, output matches input
    DType out_dtype = q.dtype();
    Tensor result = Tensor::zeros(
        {static_cast<size_t>(batch), static_cast<size_t>(heads),
         static_cast<size_t>(seq_q), static_cast<size_t>(head_dim)},
        out_dtype, Device::CPU);

    // Handle mask: ensure contiguous, get pointer
    const uint8_t *mask_data = nullptr;
    int mask_seq_stride = 0;
    Tensor mask_contig;
    if (mask.storage()) {
        mask_contig = mask.is_contiguous() ? mask : mask.ascontiguousarray();
        // Mask can be (batch, heads, seq_q, seq_kv), (1, 1, seq_q, seq_kv),
        // or (seq_q, seq_kv). We handle the last two dims as (seq_q, seq_kv)
        // per (batch, head) slice.
        mask_data = mask_contig.typed_data<uint8_t>();
        mask_seq_stride = seq_kv;
    }

    // Total number of (batch, head) slices for parallelism
    int total_slices = batch * heads;

    dispatch_float(
        q.dtype(), "scaled_dot_product_attention", [&]<typename DT>(DT) {
            using T = typename DT::value_type;

            const T *q_data = q.typed_data<T>();
            const T *k_data = k.typed_data<T>();
            const T *v_data = v.typed_data<T>();
            T *o_data = result.typed_data<T>();

            // Strides in elements for (batch, head) indexing
            auto q_batch_stride = heads * seq_q * head_dim;
            auto q_head_stride = seq_q * head_dim;
            auto k_batch_stride = heads * seq_kv * head_dim;
            auto k_head_stride = seq_kv * head_dim;
            auto v_batch_stride = heads * seq_kv * head_dim;
            auto v_head_stride = seq_kv * head_dim;
            auto o_batch_stride = heads * seq_q * head_dim;
            auto o_head_stride = seq_q * head_dim;

            // Mask strides (if present)
            int mask_batch_stride = 0;
            int mask_head_stride = 0;
            if (mask_data && mask_contig.ndim() == 4) {
                mask_batch_stride = static_cast<int>(mask_contig.shape()[1] *
                                                     mask_contig.shape()[2] *
                                                     mask_contig.shape()[3]);
                mask_head_stride = static_cast<int>(mask_contig.shape()[2] *
                                                    mask_contig.shape()[3]);
            } else if (mask_data && mask_contig.ndim() == 2) {
                // (seq_q, seq_kv) — shared across batch and heads
                mask_batch_stride = 0;
                mask_head_stride = 0;
            }

#ifdef AXIOM_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int slice = 0; slice < total_slices; ++slice) {
                int b = slice / heads;
                int h = slice % heads;

                const T *q_slice =
                    q_data + b * q_batch_stride + h * q_head_stride;
                const T *k_slice =
                    k_data + b * k_batch_stride + h * k_head_stride;
                const T *v_slice =
                    v_data + b * v_batch_stride + h * v_head_stride;
                T *o_slice = o_data + b * o_batch_stride + h * o_head_stride;

                const uint8_t *mask_slice = nullptr;
                if (mask_data) {
                    mask_slice = mask_data + b * mask_batch_stride +
                                 h * mask_head_stride;
                }

                // Outer loop over Q blocks (flash attention v2
                // structure)
                for (int q_off = 0; q_off < seq_q; q_off += CPU_BLOCK_Q) {
                    flash_attention_tile(q_slice, k_slice, v_slice, o_slice,
                                         seq_q, seq_kv, head_dim, scale,
                                         is_causal, mask_slice, mask_seq_stride,
                                         q_off);
                }
            }
        });

    return result;
}

} // namespace ops
} // namespace axiom
