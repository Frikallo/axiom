#pragma once

#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace cuda {

// Register cuBLAS-accelerated operations (matmul, gemm, etc.).
void register_cublas_operations();

} // namespace cuda
} // namespace backends
} // namespace axiom
