#pragma once

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace axiom {
namespace backends {
namespace cuda {

static constexpr int MAX_DIMS = 8;

struct GatherStridedParams {
    unsigned int ndim;
    unsigned int numel;
    unsigned int offset;    // Byte offset into source buffer
    unsigned int itemsize;  // Size of each element in bytes
    unsigned int shape[MAX_DIMS];
    unsigned int src_strides[MAX_DIMS]; // Strides in ELEMENTS, always positive
    unsigned int flip_mask; // Bitmask: bit i set if axis i has negative stride
};

#ifdef AXIOM_CUDA_SUPPORT
// Launch gather_strided for the given dtype.  Dispatches to the correct
// template instantiation defined in cuda_kernels.cu.
void launch_gather_strided(const void *src, void *dst,
                           const GatherStridedParams &params,
                           size_t element_size, cudaStream_t stream);
#endif

} // namespace cuda
} // namespace backends
} // namespace axiom
