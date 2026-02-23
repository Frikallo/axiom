#pragma once

#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

namespace axiom {
namespace backends {
namespace cuda {

// Register all CUDA backend operations with the operation registry.
void register_cuda_operations();

// Check if CUDA is available on this system.
bool is_cuda_available();

// Return a contiguous GPU tensor.  If the input is already contiguous,
// returns it unchanged.  Otherwise launches a gather_strided kernel to
// compact the data.  Every CUDA operation should call this on its inputs.
Tensor ensure_gpu_contiguous(const Tensor &t);

} // namespace cuda
} // namespace backends
} // namespace axiom
