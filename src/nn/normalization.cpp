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

// ============================================================================
// BatchNorm1d — inference-only (eval mode)
// ============================================================================

BatchNorm1d::BatchNorm1d(float eps) : eps_(eps) {
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
    register_parameter("running_mean", running_mean_);
    register_parameter("running_var", running_var_);
    register_parameter("num_batches_tracked", num_batches_tracked_);
}

Tensor BatchNorm1d::forward(const Tensor &input) const {
    if (!running_mean_.storage() || !running_var_.storage()) {
        throw RuntimeError("BatchNorm1d: running stats not initialized (call "
                           "load_state_dict first)");
    }
    if (input.ndim() < 2 || input.ndim() > 3) {
        throw ShapeError("BatchNorm1d: expected 2D or 3D input (N,C) or "
                         "(N,C,L), got " +
                         std::to_string(input.ndim()) + "D");
    }

    // Stats are per-channel: reshape to broadcast over (N, C, ...) dims
    // running_mean/var are shape (C,) — reshape to (1, C) or (1, C, 1)
    Shape stat_shape(input.ndim(), 1);
    stat_shape[1] = input.shape()[1];

    auto mean = running_mean_.reshape(stat_shape);
    auto var = running_var_.reshape(stat_shape);
    auto eps_tensor = Tensor::full<float>({1}, eps_, input.device());
    auto inv_std = ops::reciprocal(ops::sqrt(ops::add(var, eps_tensor)));

    auto result = ops::multiply(ops::subtract(input, mean), inv_std);

    if (weight_.storage()) {
        result = ops::multiply(result, weight_.reshape(stat_shape));
    }
    if (bias_.storage()) {
        result = ops::add(result, bias_.reshape(stat_shape));
    }
    return result;
}

// ============================================================================
// BatchNorm2d — inference-only (eval mode)
// ============================================================================

BatchNorm2d::BatchNorm2d(float eps) : eps_(eps) {
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
    register_parameter("running_mean", running_mean_);
    register_parameter("running_var", running_var_);
    register_parameter("num_batches_tracked", num_batches_tracked_);
}

Tensor BatchNorm2d::forward(const Tensor &input) const {
    if (!running_mean_.storage() || !running_var_.storage()) {
        throw RuntimeError("BatchNorm2d: running stats not initialized (call "
                           "load_state_dict first)");
    }
    if (input.ndim() != 4) {
        throw ShapeError("BatchNorm2d: expected 4D input (N,C,H,W), got " +
                         std::to_string(input.ndim()) + "D");
    }

    // Stats are per-channel: reshape to (1, C, 1, 1)
    Shape stat_shape = {1, input.shape()[1], 1, 1};

    auto mean = running_mean_.reshape(stat_shape);
    auto var = running_var_.reshape(stat_shape);
    auto eps_tensor = Tensor::full<float>({1}, eps_, input.device());
    auto inv_std = ops::reciprocal(ops::sqrt(ops::add(var, eps_tensor)));

    auto result = ops::multiply(ops::subtract(input, mean), inv_std);

    if (weight_.storage()) {
        result = ops::multiply(result, weight_.reshape(stat_shape));
    }
    if (bias_.storage()) {
        result = ops::add(result, bias_.reshape(stat_shape));
    }
    return result;
}

} // namespace axiom::nn
