#pragma once

#include <array>

#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace metal {

void add(Tensor &a, const Tensor &b);

// Register all Metal backend operations (both custom kernels and MPSGraph)
void register_metal_operations();

// Check if Metal is available on this system
bool is_metal_available();

// Fused GPU normalization (single graph submission instead of multiple kernels)
Tensor gpu_layer_norm(const Tensor &input, const Tensor &weight,
                      const Tensor &bias, int axis, float eps);
Tensor gpu_rms_norm(const Tensor &input, const Tensor &weight, int axis,
                    float eps);

// Fused GPU convolution (single MPSGraph submission)
Tensor gpu_conv1d(const Tensor &input, const Tensor &weight, const Tensor &bias,
                  int stride, int padding, int dilation, int groups);
Tensor gpu_conv2d(const Tensor &input, const Tensor &weight, const Tensor &bias,
                  std::array<int, 2> stride, std::array<int, 2> padding,
                  std::array<int, 2> dilation, int groups);

// Fused GPU scaled dot-product attention (Flash Attention v2)
Tensor gpu_scaled_dot_product_attention(const Tensor &query, const Tensor &key,
                                        const Tensor &value, const Tensor &mask,
                                        float scale, bool is_causal);

} // namespace metal
} // namespace backends
} // namespace axiom