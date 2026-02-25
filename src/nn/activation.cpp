#include "axiom/nn/activation.hpp"

#include "axiom/error.hpp"
#include "axiom/operations.hpp"

namespace axiom::nn {

// ============================================================================
// Stateless activations
// ============================================================================

Tensor ReLU::forward(const Tensor &input) const { return ops::relu(input); }

Tensor GELU::forward(const Tensor &input) const { return ops::gelu(input); }

Tensor SiLU::forward(const Tensor &input) const { return ops::silu(input); }

Tensor Sigmoid::forward(const Tensor &input) const {
    return ops::sigmoid(input);
}

Tensor Tanh::forward(const Tensor &input) const { return ops::tanh(input); }

// ============================================================================
// LeakyReLU
// ============================================================================

LeakyReLU::LeakyReLU(float negative_slope) : negative_slope_(negative_slope) {}

Tensor LeakyReLU::forward(const Tensor &input) const {
    return ops::leaky_relu(input, negative_slope_);
}

// ============================================================================
// Dropout — inference-only identity
// ============================================================================

Dropout::Dropout(float p) : p_(p) { (void)p_; }

Tensor Dropout::forward(const Tensor &input) const { return input; }

// ============================================================================
// Flatten
// ============================================================================

Flatten::Flatten(int start_dim, int end_dim)
    : start_dim_(start_dim), end_dim_(end_dim) {}

Tensor Flatten::forward(const Tensor &input) const {
    return input.flatten(start_dim_, end_dim_);
}

// ============================================================================
// Upsample
// ============================================================================

Upsample::Upsample(std::vector<size_t> size, InterpolateMode mode,
                   bool align_corners)
    : size_(std::move(size)), mode_(mode), align_corners_(align_corners) {}

Upsample::Upsample(std::vector<float> scale_factor, InterpolateMode mode,
                   bool align_corners)
    : scale_factor_(std::move(scale_factor)), mode_(mode),
      align_corners_(align_corners) {}

Tensor Upsample::forward(const Tensor &input) const {
    std::vector<size_t> target_size = size_;

    if (target_size.empty() && !scale_factor_.empty()) {
        // Compute target size from input spatial dims and scale factors
        // Input: (N, C, ..spatial..)
        size_t spatial_start = 2;
        size_t num_spatial = input.ndim() - spatial_start;
        if (scale_factor_.size() != num_spatial) {
            throw ValueError("Upsample: scale_factor length (" +
                             std::to_string(scale_factor_.size()) +
                             ") must match number of spatial dimensions (" +
                             std::to_string(num_spatial) + ")");
        }
        target_size.resize(num_spatial);
        for (size_t i = 0; i < num_spatial; ++i) {
            target_size[i] = static_cast<size_t>(
                static_cast<float>(input.shape()[spatial_start + i]) *
                scale_factor_[i]);
        }
    }

    return ops::interpolate(input, target_size, {}, mode_, align_corners_);
}

} // namespace axiom::nn
