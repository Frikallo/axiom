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

// Shared implementation: (x - mean) / sqrt(var + eps) * weight + bias
// Upcasts to Float32 for numerical stability, casts back to input dtype.
// Moves stats to input device once at the top to avoid per-op device moves.
static Tensor batch_norm_forward(const Tensor &input,
                                 const Tensor &running_mean,
                                 const Tensor &running_var,
                                 const Tensor &weight, const Tensor &bias,
                                 float eps, const Shape &stat_shape) {
    DType input_dtype = input.dtype();
    Device device = input.device();

    // Upcast to at least Float32 for numerical stability (matches PyTorch)
    DType compute_dtype =
        (input_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    auto x =
        (input_dtype == compute_dtype) ? input : input.astype(compute_dtype);

    // Cast to compute dtype first (avoids GPU rejecting Float64), then move
    auto mean =
        running_mean.astype(compute_dtype).to(device).reshape(stat_shape);
    auto var = running_var.astype(compute_dtype).to(device).reshape(stat_shape);

    auto eps_tensor =
        Tensor::full<float>({1}, eps).astype(compute_dtype).to(device);
    auto inv_std = ops::reciprocal(ops::sqrt(ops::add(var, eps_tensor)));
    auto result = ops::multiply(ops::subtract(x, mean), inv_std);

    if (weight.storage()) {
        result = ops::multiply(
            result,
            weight.astype(compute_dtype).to(device).reshape(stat_shape));
    }
    if (bias.storage()) {
        result = ops::add(
            result, bias.astype(compute_dtype).to(device).reshape(stat_shape));
    }

    // Cast back to input dtype
    if (result.dtype() != input_dtype) {
        result = result.astype(input_dtype);
    }
    return result;
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

    Shape stat_shape(input.ndim(), 1);
    stat_shape[1] = input.shape()[1];
    return batch_norm_forward(input, running_mean_, running_var_, weight_,
                              bias_, eps_, stat_shape);
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

    Shape stat_shape = {1, input.shape()[1], 1, 1};
    return batch_norm_forward(input, running_mean_, running_var_, weight_,
                              bias_, eps_, stat_shape);
}

} // namespace axiom::nn
