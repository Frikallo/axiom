#pragma once

#include <array>

#include "axiom/nn/module.hpp"

namespace axiom::nn {

struct Conv1dConfig {
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    bool bias = true;
};

struct Conv2dConfig {
    std::array<int, 2> stride = {1, 1};
    std::array<int, 2> padding = {0, 0};
    std::array<int, 2> dilation = {1, 1};
    int groups = 1;
    bool bias = true;
};

class Conv1d : public Module {
  public:
    Conv1d(int stride = 1, int padding = 0, int dilation = 1, int groups = 1,
           bool bias = true);
    explicit Conv1d(const Conv1dConfig &config);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    int stride_;
    int padding_;
    int dilation_;
    int groups_;
    bool has_bias_;
};

class Conv2d : public Module {
  public:
    Conv2d(std::array<int, 2> stride = {1, 1},
           std::array<int, 2> padding = {0, 0},
           std::array<int, 2> dilation = {1, 1}, int groups = 1,
           bool bias = true);
    explicit Conv2d(const Conv2dConfig &config);

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    Tensor weight_;
    Tensor bias_;
    std::array<int, 2> stride_;
    std::array<int, 2> padding_;
    std::array<int, 2> dilation_;
    int groups_;
    bool has_bias_;
};

} // namespace axiom::nn
