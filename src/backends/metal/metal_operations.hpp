#pragma once

#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace metal {

void add(Tensor &a, const Tensor &b);

// Register all Metal backend operations (both custom kernels and MPSGraph)
void register_metal_operations();

// Check if Metal is available on this system
bool is_metal_available();

} // namespace metal
} // namespace backends
} // namespace axiom