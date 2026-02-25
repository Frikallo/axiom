#pragma once

#include "axiom/nn/module.hpp"

namespace axiom::nn {

class Embedding : public Module {
  public:
    Embedding();

    Tensor forward(const Tensor &indices) const override;
    Tensor operator()(const Tensor &indices) const { return forward(indices); }

  private:
    Tensor weight_; // (vocab_size, embed_dim)
};

} // namespace axiom::nn
