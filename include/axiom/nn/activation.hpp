#pragma once

#include <vector>

#include "axiom/nn/module.hpp"
#include "axiom/operations.hpp"

namespace axiom::nn {

class ReLU : public Module {
  public:
    ReLU() = default;
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }
};

class GELU : public Module {
  public:
    GELU() = default;
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }
};

class SiLU : public Module {
  public:
    SiLU() = default;
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }
};

class Sigmoid : public Module {
  public:
    Sigmoid() = default;
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }
};

class Tanh : public Module {
  public:
    Tanh() = default;
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }
};

class LeakyReLU : public Module {
  public:
    explicit LeakyReLU(float negative_slope = 0.01f);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    float negative_slope_;
};

// Inference-only: forward() returns input unchanged.
// Stores p_ for API compatibility with PyTorch state dicts.
class Dropout : public Module {
  public:
    explicit Dropout(float p = 0.5f);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    float p_;
};

class Flatten : public Module {
  public:
    explicit Flatten(int start_dim = 1, int end_dim = -1);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    int start_dim_;
    int end_dim_;
};

// Upsamples input using nearest/bilinear/bicubic interpolation.
// Constructed with either a target size or scale factor.
// InterpolateMode is defined in axiom/operations.hpp as ops::InterpolateMode.
using ops::InterpolateMode;

class Upsample : public Module {
  public:
    // Size-based: specify target spatial dimensions
    explicit Upsample(std::vector<size_t> size,
                      InterpolateMode mode = InterpolateMode::Nearest,
                      bool align_corners = false);

    // Scale-based: specify scale factors per spatial dimension
    explicit Upsample(std::vector<float> scale_factor,
                      InterpolateMode mode = InterpolateMode::Nearest,
                      bool align_corners = false);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    std::vector<size_t> size_;
    std::vector<float> scale_factor_;
    InterpolateMode mode_;
    bool align_corners_;
};

} // namespace axiom::nn
