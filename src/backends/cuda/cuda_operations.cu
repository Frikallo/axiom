#include "cuda_operations.hpp"
#include "cuda_context.hpp"
#include "cublas_operations.hpp"

namespace axiom {
namespace backends {
namespace cuda {

void register_cuda_operations() {
    if (!is_cuda_available()) return;

    register_cublas_operations();

    // TODO: register element-wise, reduction, and custom kernel operations
}

} // namespace cuda
} // namespace backends
} // namespace axiom
