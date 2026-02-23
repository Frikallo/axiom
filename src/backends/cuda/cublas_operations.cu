#include "cublas_operations.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cublas_v2.h>
#endif

namespace axiom {
namespace backends {
namespace cuda {

void register_cublas_operations() {
    // TODO: register matmul, gemm, and other cuBLAS-backed operations
}

} // namespace cuda
} // namespace backends
} // namespace axiom
