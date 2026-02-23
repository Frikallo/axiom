#pragma once

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace axiom {
namespace backends {
namespace cuda {

// Cached launch configuration for a CUDA kernel.
struct LaunchParams {
    unsigned int grid;
    int block;
};

// Caches the optimal block size per kernel function pointer via
// cudaOccupancyMaxPotentialBlockSize, then computes grid from numel.
// Thread-safe singleton — the occupancy query runs once per unique
// kernel pointer and the result is reused for all subsequent launches.
class CudaLaunchCache {
  public:
    static CudaLaunchCache &instance() {
        static CudaLaunchCache cache;
        return cache;
    }

    // Look up (or compute and cache) the optimal block size for `kernel`,
    // then return {grid, block} for the given element count.
    // The template is required because cudaOccupancyMaxPotentialBlockSize
    // is itself a template that needs the typed kernel pointer.
    template <typename KernelFunc>
    LaunchParams params_for(KernelFunc kernel, size_t numel,
                            size_t dynamic_smem = 0) {
        const void *key = reinterpret_cast<const void *>(kernel);
        int block;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                block = it->second;
                unsigned int grid = static_cast<unsigned int>(
                    (numel + block - 1) / block);
                return {grid, block};
            }
        }

        // Query optimal block size — this is the expensive call we cache.
        int min_grid = 0;
        block = 0;
        cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
            &min_grid, &block, kernel, dynamic_smem);
        (void)min_grid;

        if (err != cudaSuccess || block <= 0)
            block = 256; // safe fallback

        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_.emplace(key, block);
        }

        unsigned int grid =
            static_cast<unsigned int>((numel + block - 1) / block);
        return {grid, block};
    }

  private:
    CudaLaunchCache() = default;

    CudaLaunchCache(const CudaLaunchCache &) = delete;
    CudaLaunchCache &operator=(const CudaLaunchCache &) = delete;

    std::unordered_map<const void *, int> cache_;
    std::mutex mutex_;
};

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif
