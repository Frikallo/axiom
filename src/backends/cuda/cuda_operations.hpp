#pragma once

#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace cuda {

// Register all CUDA backend operations with the operation registry.
void register_cuda_operations();

// Check if CUDA is available on this system.
bool is_cuda_available();

} // namespace cuda
} // namespace backends
} // namespace axiom
