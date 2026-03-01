#include "axiom/nn/normalization.hpp"

#include "axiom/error.hpp"
#include "axiom/graph/graph_registry.hpp"
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
    Device device = input.device();

    // GPU fast-path: route through lazy graph compiler (handles dtype
    // internally) to avoid eager astype calls that fragment the lazy graph.
    if (device == Device::GPU) {
        // Reshape stats to (1, C, 1, ...) on CPU for broadcast, then transfer
        auto mean_r = running_mean.reshape(stat_shape);
        auto var_r = running_var.reshape(stat_shape);
        if (mean_r.device() != device)
            mean_r = mean_r.to(device);
        if (var_r.device() != device)
            var_r = var_r.to(device);
        auto w = weight.storage()
                     ? ((weight.device() == device)
                            ? weight.reshape(stat_shape)
                            : weight.reshape(stat_shape).to(device))
                     : Tensor();
        auto b = bias.storage() ? ((bias.device() == device)
                                       ? bias.reshape(stat_shape)
                                       : bias.reshape(stat_shape).to(device))
                                : Tensor();
        return graph::GraphRegistry::create_lazy_batchnorm(input, w, b, mean_r,
                                                           var_r, eps);
    }

    DType input_dtype = input.dtype();

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

// ============================================================================
// GroupNorm
// ============================================================================

GroupNorm::GroupNorm(int num_groups, float eps)
    : num_groups_(num_groups), eps_(eps) {
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
}

Tensor GroupNorm::forward(const Tensor &input) const {
    if (!weight_.storage()) {
        throw RuntimeError("GroupNorm: weight not initialized (call "
                           "load_state_dict first)");
    }
    if (input.ndim() < 2) {
        throw ShapeError("GroupNorm: expected at least 2D input, got " +
                         std::to_string(input.ndim()) + "D");
    }
    size_t C = input.shape()[1];
    if (C % static_cast<size_t>(num_groups_) != 0) {
        throw ShapeError("GroupNorm: channels (" + std::to_string(C) +
                         ") must be divisible by num_groups (" +
                         std::to_string(num_groups_) + ")");
    }

    DType input_dtype = input.dtype();
    DType compute_dtype =
        (input_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    auto x =
        (input_dtype == compute_dtype) ? input : input.astype(compute_dtype);
    Device device = input.device();

    // Reshape (N, C, ...) → (N, G, C/G * spatial...)
    size_t N = input.shape()[0];
    size_t G = static_cast<size_t>(num_groups_);
    size_t channels_per_group = C / G;

    // Compute total spatial size
    size_t spatial_size = 1;
    for (size_t i = 2; i < input.ndim(); ++i) {
        spatial_size *= input.shape()[i];
    }

    auto reshaped = x.reshape({N, G, channels_per_group * spatial_size});

    // Compute mean and var over last dim
    auto mean = ops::mean(reshaped, {2}, true);
    auto centered = ops::subtract(reshaped, mean);
    auto var = ops::mean(ops::multiply(centered, centered), {2}, true);

    auto eps_tensor =
        Tensor::full<float>({1}, eps_).astype(compute_dtype).to(device);
    auto inv_std = ops::reciprocal(ops::sqrt(ops::add(var, eps_tensor)));
    auto normed = ops::multiply(centered, inv_std);

    // Reshape back to original shape
    auto result = normed.reshape(input.shape());

    // Apply affine: weight and bias are (C,) — reshape for broadcast
    Shape affine_shape(input.ndim(), 1);
    affine_shape[1] = C;

    if (weight_.storage()) {
        result = ops::multiply(
            result,
            weight_.astype(compute_dtype).to(device).reshape(affine_shape));
    }
    if (bias_.storage()) {
        result = ops::add(
            result,
            bias_.astype(compute_dtype).to(device).reshape(affine_shape));
    }

    if (result.dtype() != input_dtype) {
        result = result.astype(input_dtype);
    }
    return result;
}

// ============================================================================
// Instance normalization — shared helper
// ============================================================================

static Tensor instance_norm_forward(const Tensor &input, const Tensor &weight,
                                    const Tensor &bias, float eps, bool affine,
                                    const std::vector<int> &spatial_axes) {
    DType input_dtype = input.dtype();
    DType compute_dtype =
        (input_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    auto x =
        (input_dtype == compute_dtype) ? input : input.astype(compute_dtype);
    Device device = input.device();

    auto mean = ops::mean(x, spatial_axes, true);
    auto centered = ops::subtract(x, mean);
    auto var = ops::mean(ops::multiply(centered, centered), spatial_axes, true);

    auto eps_tensor =
        Tensor::full<float>({1}, eps).astype(compute_dtype).to(device);
    auto inv_std = ops::reciprocal(ops::sqrt(ops::add(var, eps_tensor)));
    auto result = ops::multiply(centered, inv_std);

    if (affine && weight.storage()) {
        Shape affine_shape(input.ndim(), 1);
        affine_shape[1] = input.shape()[1];
        result = ops::multiply(
            result,
            weight.astype(compute_dtype).to(device).reshape(affine_shape));
    }
    if (affine && bias.storage()) {
        Shape affine_shape(input.ndim(), 1);
        affine_shape[1] = input.shape()[1];
        result = ops::add(
            result,
            bias.astype(compute_dtype).to(device).reshape(affine_shape));
    }

    if (result.dtype() != input_dtype) {
        result = result.astype(input_dtype);
    }
    return result;
}

// ============================================================================
// InstanceNorm1d
// ============================================================================

InstanceNorm1d::InstanceNorm1d(float eps, bool affine)
    : eps_(eps), affine_(affine) {
    if (affine_) {
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }
}

Tensor InstanceNorm1d::forward(const Tensor &input) const {
    if (input.ndim() != 3) {
        throw ShapeError("InstanceNorm1d: expected 3D input (N,C,L), got " +
                         std::to_string(input.ndim()) + "D");
    }
    return instance_norm_forward(input, weight_, bias_, eps_, affine_, {2});
}

// ============================================================================
// InstanceNorm2d
// ============================================================================

InstanceNorm2d::InstanceNorm2d(float eps, bool affine)
    : eps_(eps), affine_(affine) {
    if (affine_) {
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }
}

Tensor InstanceNorm2d::forward(const Tensor &input) const {
    if (input.ndim() != 4) {
        throw ShapeError("InstanceNorm2d: expected 4D input (N,C,H,W), got " +
                         std::to_string(input.ndim()) + "D");
    }
    return instance_norm_forward(input, weight_, bias_, eps_, affine_, {2, 3});
}

} // namespace axiom::nn
