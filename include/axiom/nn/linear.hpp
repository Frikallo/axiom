#pragma once

#include "axiom/nn/module.hpp"

namespace axiom::nn {

class Linear : public Module {
  public:
    explicit Linear(bool bias = true);

    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_; // (out_features, in_features)
    Tensor bias_;   // (out_features,)
    bool has_bias_;
};

} // namespace axiom::nn
