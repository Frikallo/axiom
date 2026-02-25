#pragma once

#include "axiom/nn/module.hpp"

namespace axiom::nn {

class LayerNorm : public Module {
  public:
    explicit LayerNorm(float eps = 1e-5f);

    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    float eps_;
};

class RMSNorm : public Module {
  public:
    explicit RMSNorm(float eps = 1e-5f);

    Tensor forward(const Tensor &input) const;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    float eps_;
};

} // namespace axiom::nn
