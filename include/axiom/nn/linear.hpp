#pragma once

#include "axiom/nn/module.hpp"

namespace axiom::nn {

class Linear : public Module {
  public:
    explicit Linear(bool bias = true);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

    // Accessors for fused GPU operations
    const Tensor &weight() const { return weight_; }
    const Tensor &bias() const { return bias_; }
    bool has_bias() const { return has_bias_; }

  private:
    Tensor weight_; // (out_features, in_features)
    Tensor bias_;   // (out_features,)
    bool has_bias_;
};

} // namespace axiom::nn
