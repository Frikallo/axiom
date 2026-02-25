#pragma once

#include <vector>

#include "axiom/nn/module.hpp"

namespace axiom::nn {

class MaxPool1d : public Module {
  public:
    explicit MaxPool1d(int kernel_size, int stride = 0, int padding = 0);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    int kernel_size_;
    int stride_;
    int padding_;
};

class MaxPool2d : public Module {
  public:
    explicit MaxPool2d(std::vector<int> kernel_size,
                       std::vector<int> stride = {},
                       std::vector<int> padding = {});
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
};

class AvgPool1d : public Module {
  public:
    explicit AvgPool1d(int kernel_size, int stride = 0, int padding = 0,
                       bool count_include_pad = true);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    int kernel_size_;
    int stride_;
    int padding_;
    bool count_include_pad_;
};

class AvgPool2d : public Module {
  public:
    explicit AvgPool2d(std::vector<int> kernel_size,
                       std::vector<int> stride = {},
                       std::vector<int> padding = {},
                       bool count_include_pad = true);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    std::vector<int> kernel_size_;
    std::vector<int> stride_;
    std::vector<int> padding_;
    bool count_include_pad_;
};

class AdaptiveAvgPool1d : public Module {
  public:
    explicit AdaptiveAvgPool1d(int output_size);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    int output_size_;
};

class AdaptiveAvgPool2d : public Module {
  public:
    explicit AdaptiveAvgPool2d(std::vector<int> output_size);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    std::vector<int> output_size_;
};

class AdaptiveMaxPool1d : public Module {
  public:
    explicit AdaptiveMaxPool1d(int output_size);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    int output_size_;
};

class AdaptiveMaxPool2d : public Module {
  public:
    explicit AdaptiveMaxPool2d(std::vector<int> output_size);
    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

  private:
    std::vector<int> output_size_;
};

} // namespace axiom::nn
