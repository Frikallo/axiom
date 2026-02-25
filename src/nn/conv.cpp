#include "axiom/nn/conv.hpp"

#include "axiom/error.hpp"
#include "axiom/operations.hpp"

namespace axiom::nn {

Conv1d::Conv1d(int stride, int padding, int dilation, int groups, bool bias)
    : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups),
      has_bias_(bias) {
    register_parameter("weight", weight_);
    if (has_bias_) {
        register_parameter("bias", bias_);
    }
}

Conv1d::Conv1d(const Conv1dConfig &config)
    : Conv1d(config.stride, config.padding, config.dilation, config.groups,
             config.bias) {}

Tensor Conv1d::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("Conv1d: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::conv1d(input, weight_, has_bias_ ? bias_ : Tensor(), stride_,
                       padding_, dilation_, groups_);
}

Conv2d::Conv2d(std::array<int, 2> stride, std::array<int, 2> padding,
               std::array<int, 2> dilation, int groups, bool bias)
    : stride_(stride), padding_(padding), dilation_(dilation), groups_(groups),
      has_bias_(bias) {
    register_parameter("weight", weight_);
    if (has_bias_) {
        register_parameter("bias", bias_);
    }
}

Conv2d::Conv2d(const Conv2dConfig &config)
    : Conv2d(config.stride, config.padding, config.dilation, config.groups,
             config.bias) {}

Tensor Conv2d::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("Conv2d: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::conv2d(input, weight_, has_bias_ ? bias_ : Tensor(), stride_,
                       padding_, dilation_, groups_);
}

// ============================================================================
// ConvTranspose1d
// ============================================================================

ConvTranspose1d::ConvTranspose1d(int stride, int padding, int output_padding,
                                 int dilation, int groups, bool bias)
    : stride_(stride), padding_(padding), output_padding_(output_padding),
      dilation_(dilation), groups_(groups), has_bias_(bias) {
    register_parameter("weight", weight_);
    if (has_bias_) {
        register_parameter("bias", bias_);
    }
}

ConvTranspose1d::ConvTranspose1d(const ConvTranspose1dConfig &config)
    : ConvTranspose1d(config.stride, config.padding, config.output_padding,
                      config.dilation, config.groups, config.bias) {}

Tensor ConvTranspose1d::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("ConvTranspose1d: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::conv_transpose1d(input, weight_, has_bias_ ? bias_ : Tensor(),
                                 stride_, padding_, output_padding_, dilation_,
                                 groups_);
}

// ============================================================================
// ConvTranspose2d
// ============================================================================

ConvTranspose2d::ConvTranspose2d(std::array<int, 2> stride,
                                 std::array<int, 2> padding,
                                 std::array<int, 2> output_padding,
                                 std::array<int, 2> dilation, int groups,
                                 bool bias)
    : stride_(stride), padding_(padding), output_padding_(output_padding),
      dilation_(dilation), groups_(groups), has_bias_(bias) {
    register_parameter("weight", weight_);
    if (has_bias_) {
        register_parameter("bias", bias_);
    }
}

ConvTranspose2d::ConvTranspose2d(const ConvTranspose2dConfig &config)
    : ConvTranspose2d(config.stride, config.padding, config.output_padding,
                      config.dilation, config.groups, config.bias) {}

Tensor ConvTranspose2d::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("ConvTranspose2d: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::conv_transpose2d(input, weight_, has_bias_ ? bias_ : Tensor(),
                                 stride_, padding_, output_padding_, dilation_,
                                 groups_);
}

} // namespace axiom::nn
