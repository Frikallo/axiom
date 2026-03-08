#pragma once

#include "backends/cpu/lapack/lapack_backend.hpp"

namespace axiom {
namespace backends {
namespace cuda {

// Get the cuSOLVER-backed LAPACK backend for GPU tensors.
// Returns nullptr if CUDA is not available.
cpu::lapack::LapackBackend *get_cusolver_backend();

} // namespace cuda
} // namespace backends
} // namespace axiom
