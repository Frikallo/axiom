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

Tensor Conv2d::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("Conv2d: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::conv2d(input, weight_, has_bias_ ? bias_ : Tensor(), stride_,
                       padding_, dilation_, groups_);
}

} // namespace axiom::nn
