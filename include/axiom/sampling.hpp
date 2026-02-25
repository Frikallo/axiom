#pragma once

#include "tensor.hpp"

namespace axiom {
namespace sampling {

// Scale logits by temperature (divide by temperature)
// temperature=1.0 is identity
Tensor temperature_scale(const Tensor &logits, float temperature);

// Mask logits below the k-th largest value to -inf
// top_k=1 keeps only the maximum
Tensor top_k(const Tensor &logits, int k);

// Nucleus sampling: mask logits outside cumulative probability p to -inf
// top_p=1.0 keeps all logits
Tensor top_p(const Tensor &logits, float p);

} // namespace sampling
} // namespace axiom
