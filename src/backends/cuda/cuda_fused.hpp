#pragma once

#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace cuda {

// Register fused CUDA kernel patterns (AddReLU, MulAdd, etc.).
void register_cuda_fused_operations();

} // namespace cuda
} // namespace backends
} // namespace axiom
