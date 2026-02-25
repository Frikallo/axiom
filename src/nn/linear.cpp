#include "axiom/nn/linear.hpp"

#include "axiom/error.hpp"
#include "axiom/operations.hpp"

namespace axiom::nn {

Linear::Linear(bool bias) : has_bias_(bias) {
    register_parameter("weight", weight_);
    if (has_bias_) {
        register_parameter("bias", bias_);
    }
}

Tensor Linear::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("Linear: weight not initialized (call "
                           "load_state_dict first)");
    }
    auto out = ops::matmul(input, weight_, false, true);
    if (has_bias_ && bias_.storage()) {
        out = ops::add(out, bias_);
    }
    return out;
}

} // namespace axiom::nn
