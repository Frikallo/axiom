#pragma once

#include "axiom/nn/module.hpp"

namespace axiom::nn {

class LayerNorm : public Module {
  public:
    explicit LayerNorm(float eps = 1e-5f);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    float eps_;
};

class RMSNorm : public Module {
  public:
    explicit RMSNorm(float eps = 1e-5f);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    float eps_;
};

class BatchNorm1d : public Module {
  public:
    explicit BatchNorm1d(float eps = 1e-5f);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
    Tensor num_batches_tracked_;
    float eps_;
};

class BatchNorm2d : public Module {
  public:
    explicit BatchNorm2d(float eps = 1e-5f);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
    Tensor num_batches_tracked_;
    float eps_;
};

class GroupNorm : public Module {
  public:
    explicit GroupNorm(int num_groups, float eps = 1e-5f);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    int num_groups_;
    Tensor weight_;
    Tensor bias_;
    float eps_;
};

class InstanceNorm1d : public Module {
  public:
    explicit InstanceNorm1d(float eps = 1e-5f, bool affine = true);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    float eps_;
    bool affine_;
};

class InstanceNorm2d : public Module {
  public:
    explicit InstanceNorm2d(float eps = 1e-5f, bool affine = true);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    float eps_;
    bool affine_;
};

} // namespace axiom::nn
