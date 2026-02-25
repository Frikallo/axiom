#include "axiom/nn/normalization.hpp"

#include "axiom/error.hpp"
#include "axiom/operations.hpp"

namespace axiom::nn {

LayerNorm::LayerNorm(float eps) : eps_(eps) {
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
}

Tensor LayerNorm::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("LayerNorm: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::layer_norm(input, weight_, bias_, -1, eps_);
}

RMSNorm::RMSNorm(float eps) : eps_(eps) {
    register_parameter("weight", weight_);
}

Tensor RMSNorm::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("RMSNorm: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::rms_norm(input, weight_, -1, eps_);
}

} // namespace axiom::nn
