#include "axiom/nn/attention.hpp"

#include <cmath>

#include "axiom/error.hpp"
#include "axiom/operations.hpp"

namespace axiom::nn {

MultiHeadAttention::MultiHeadAttention(int num_heads) : num_heads_(num_heads) {
    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
}

Tensor MultiHeadAttention::forward(const Tensor &query, const Tensor &key,
                                   const Tensor &value,
                                   const Tensor &mask) const {
    if (query.ndim() != 3) {
        throw ShapeError("MultiHeadAttention expects 3D input (batch, seq, "
                         "d_model), got " +
                         std::to_string(query.ndim()) + "D");
    }

    auto q = q_proj_(query);
    auto k = k_proj_(key);
    auto v = v_proj_(value);

    auto d_model = static_cast<int>(q.shape().back());
    if (d_model % num_heads_ != 0) {
        throw ValueError("d_model (" + std::to_string(d_model) +
                         ") must be divisible by num_heads (" +
                         std::to_string(num_heads_) + ")");
    }
    int head_dim = d_model / num_heads_;

    auto batch = q.shape()[0];
    auto seq_q = q.shape()[1];
    auto seq_k = k.shape()[1];

    // Reshape to (batch, seq, num_heads, head_dim) then transpose to
    // (batch, num_heads, seq, head_dim)
    q = q.reshape({batch, seq_q, static_cast<size_t>(num_heads_),
                   static_cast<size_t>(head_dim)})
            .transpose({0, 2, 1, 3});
    k = k.reshape({batch, seq_k, static_cast<size_t>(num_heads_),
                   static_cast<size_t>(head_dim)})
            .transpose({0, 2, 1, 3});
    v = v.reshape({batch, seq_k, static_cast<size_t>(num_heads_),
                   static_cast<size_t>(head_dim)})
            .transpose({0, 2, 1, 3});

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto scores = ops::matmul(q, k, false, true);
    auto scale_tensor =
        Tensor::full({1}, scale, scores.device()).astype(scores.dtype());
    scores = ops::multiply(scores, scale_tensor);

    // Apply mask if provided (ensure contiguous for masked_fill)
    if (mask.storage()) {
        scores = ops::masked_fill(scores.ascontiguousarray(), mask, -1e9f);
    }

    auto attn_weights = ops::softmax(scores, -1);

    auto attn_output = ops::matmul(attn_weights, v);

    // Transpose back: (batch, num_heads, seq, head_dim) -> (batch, seq,
    // num_heads, head_dim)
    attn_output = attn_output.transpose({0, 2, 1, 3});
    attn_output =
        attn_output.reshape({batch, seq_q, static_cast<size_t>(d_model)});

    return out_proj_(attn_output);
}

} // namespace axiom::nn
