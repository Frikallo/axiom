#pragma once

#include "axiom/nn/linear.hpp"
#include "axiom/nn/module.hpp"

namespace axiom::nn {

class MultiHeadAttention : public Module {
  public:
    using Module::forward; // multi-arg forward hides base; re-expose

    explicit MultiHeadAttention(int num_heads);

    // query, key, value: (batch, seq, d_model)
    // mask: optional attention mask, or empty Tensor() to skip
    // is_causal: apply lower-triangular causal mask inside the fused kernel
    // Returns: (batch, seq, d_model)
    Tensor forward(const Tensor &query, const Tensor &key, const Tensor &value,
                   const Tensor &mask = Tensor(), bool is_causal = false) const;

  private:
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear out_proj_;
    int num_heads_;
};

} // namespace axiom::nn
