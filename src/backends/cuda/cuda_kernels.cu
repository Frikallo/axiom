#include "cuda_kernels.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace cuda {

// ============================================================================
// Gather Strided Kernel
// ============================================================================
// Copies non-contiguous (strided) tensor data to a contiguous buffer.
// Converts each output index to N-dimensional coordinates, then uses
// the source strides to compute the correct input offset.
// ============================================================================

template <typename T>
__global__ void gather_strided(const T *src, T *dst,
                               GatherStridedParams p) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.numel) return;

    // Convert linear index to N-dimensional coordinates
    unsigned int coords[MAX_DIMS];
    unsigned int temp = gid;

    #pragma unroll
    for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
        coords[i] = temp % p.shape[i];
        temp /= p.shape[i];
    }

    // Compute strided source offset using element strides.
    // For flipped axes (negative stride), transform coord to
    // (shape - 1 - coord).
    unsigned int src_idx = 0;
    #pragma unroll
    for (unsigned int i = 0; i < p.ndim; ++i) {
        unsigned int coord = coords[i];
        if (p.flip_mask & (1u << i)) {
            coord = p.shape[i] - 1 - coord;
        }
        src_idx += coord * p.src_strides[i];
    }

    dst[gid] = src[src_idx];
}

// Explicit template instantiations
template __global__ void gather_strided<float>(const float *, float *,
                                               GatherStridedParams);
template __global__ void gather_strided<double>(const double *, double *,
                                                GatherStridedParams);
template __global__ void gather_strided<__half>(const __half *, __half *,
                                                GatherStridedParams);
template __global__ void gather_strided<int32_t>(const int32_t *, int32_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<int64_t>(const int64_t *, int64_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<int16_t>(const int16_t *, int16_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<int8_t>(const int8_t *, int8_t *,
                                                GatherStridedParams);
template __global__ void gather_strided<uint8_t>(const uint8_t *, uint8_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<uint16_t>(const uint16_t *,
                                                  uint16_t *,
                                                  GatherStridedParams);
template __global__ void gather_strided<uint32_t>(const uint32_t *,
                                                  uint32_t *,
                                                  GatherStridedParams);

// ============================================================================
// Launcher — dispatches to the correct template by element size
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

void launch_gather_strided(const void *src, void *dst,
                           const GatherStridedParams &params,
                           size_t element_size, cudaStream_t stream) {
    unsigned int grid =
        (params.numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (element_size) {
    case 1:
        gather_strided<uint8_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint8_t *>(src),
            static_cast<uint8_t *>(dst), params);
        break;
    case 2:
        gather_strided<uint16_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint16_t *>(src),
            static_cast<uint16_t *>(dst), params);
        break;
    case 4:
        gather_strided<uint32_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint32_t *>(src),
            static_cast<uint32_t *>(dst), params);
        break;
    case 8:
        gather_strided<int64_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const int64_t *>(src),
            static_cast<int64_t *>(dst), params);
        break;
    default:
        throw std::runtime_error(
            "gather_strided: unsupported element size " +
            std::to_string(element_size));
    }
}

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif // AXIOM_CUDA_SUPPORT
