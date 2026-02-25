#include "axiom/nn/pooling.hpp"

#include "axiom/operations.hpp"

namespace axiom::nn {

// ============================================================================
// MaxPool1d
// ============================================================================

MaxPool1d::MaxPool1d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride),
      padding_(padding) {}

Tensor MaxPool1d::forward(const Tensor &input) const {
    return ops::max_pool1d(input, kernel_size_, stride_, padding_);
}

// ============================================================================
// MaxPool2d
// ============================================================================

MaxPool2d::MaxPool2d(std::vector<int> kernel_size, std::vector<int> stride,
                     std::vector<int> padding)
    : kernel_size_(std::move(kernel_size)),
      stride_(stride.empty() ? kernel_size_ : std::move(stride)),
      padding_(std::move(padding)) {}

Tensor MaxPool2d::forward(const Tensor &input) const {
    return ops::max_pool2d(input, kernel_size_, stride_, padding_);
}

// ============================================================================
// AvgPool1d
// ============================================================================

AvgPool1d::AvgPool1d(int kernel_size, int stride, int padding,
                     bool count_include_pad)
    : kernel_size_(kernel_size), stride_(stride == 0 ? kernel_size : stride),
      padding_(padding), count_include_pad_(count_include_pad) {}

Tensor AvgPool1d::forward(const Tensor &input) const {
    return ops::avg_pool1d(input, kernel_size_, stride_, padding_,
                           count_include_pad_);
}

// ============================================================================
// AvgPool2d
// ============================================================================

AvgPool2d::AvgPool2d(std::vector<int> kernel_size, std::vector<int> stride,
                     std::vector<int> padding, bool count_include_pad)
    : kernel_size_(std::move(kernel_size)),
      stride_(stride.empty() ? kernel_size_ : std::move(stride)),
      padding_(std::move(padding)), count_include_pad_(count_include_pad) {}

Tensor AvgPool2d::forward(const Tensor &input) const {
    return ops::avg_pool2d(input, kernel_size_, stride_, padding_,
                           count_include_pad_);
}

// ============================================================================
// Adaptive pooling
// ============================================================================

AdaptiveAvgPool1d::AdaptiveAvgPool1d(int output_size)
    : output_size_(output_size) {}

Tensor AdaptiveAvgPool1d::forward(const Tensor &input) const {
    return ops::adaptive_avg_pool1d(input, output_size_);
}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(std::vector<int> output_size)
    : output_size_(std::move(output_size)) {}

Tensor AdaptiveAvgPool2d::forward(const Tensor &input) const {
    return ops::adaptive_avg_pool2d(input, output_size_);
}

AdaptiveMaxPool1d::AdaptiveMaxPool1d(int output_size)
    : output_size_(output_size) {}

Tensor AdaptiveMaxPool1d::forward(const Tensor &input) const {
    return ops::adaptive_max_pool1d(input, output_size_);
}

AdaptiveMaxPool2d::AdaptiveMaxPool2d(std::vector<int> output_size)
    : output_size_(std::move(output_size)) {}

Tensor AdaptiveMaxPool2d::forward(const Tensor &input) const {
    return ops::adaptive_max_pool2d(input, output_size_);
}

} // namespace axiom::nn
