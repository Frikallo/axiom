#include "cuda_fused.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace axiom {
namespace backends {
namespace cuda {

void register_cuda_fused_operations() {
    // TODO: register fused kernel patterns (AddReLU, MulAdd, etc.)
}

} // namespace cuda
} // namespace backends
} // namespace axiom
