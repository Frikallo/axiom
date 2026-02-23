#include "cuda_allocator.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

#include <bit>
#include <stdexcept>
#include <string>

namespace axiom {
namespace backends {
namespace cuda {

// Minimum allocation size — avoids degenerate tiny buckets.
static constexpr size_t MIN_BLOCK_SIZE = 512;

CudaAllocator &CudaAllocator::instance() {
    static CudaAllocator alloc;
    return alloc;
}

size_t CudaAllocator::round_up(size_t size) {
    if (size <= MIN_BLOCK_SIZE) return MIN_BLOCK_SIZE;
    return std::bit_ceil(size);
}

void *CudaAllocator::allocate(size_t size_bytes) {
    size_t bucket = round_up(size_bytes);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = free_blocks_.find(bucket);
        if (it != free_blocks_.end()) {
            void *ptr = it->second;
            cached_bytes_ -= bucket;
            free_blocks_.erase(it);
            return ptr;
        }
    }

    // Cache miss — fall back to cudaMalloc.
#ifdef AXIOM_CUDA_SUPPORT
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bucket);
    if (err != cudaSuccess) {
        // Try to free cached blocks and retry once.
        release_pool();
        err = cudaMalloc(&ptr, bucket);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                     cudaGetErrorString(err));
        }
    }
    return ptr;
#else
    (void)bucket;
    throw std::runtime_error("CUDA support not compiled");
#endif
}

void CudaAllocator::deallocate(void *ptr, size_t alloc_size) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);
    if (cached_bytes_ + alloc_size <= pool_limit_) {
        free_blocks_.emplace(alloc_size, ptr);
        cached_bytes_ += alloc_size;
    } else {
        // Pool full — free immediately.
#ifdef AXIOM_CUDA_SUPPORT
        cudaFree(ptr);
#endif
    }
}

void CudaAllocator::release_pool() {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef AXIOM_CUDA_SUPPORT
    for (auto &[size, ptr] : free_blocks_) {
        cudaFree(ptr);
    }
#endif
    free_blocks_.clear();
    cached_bytes_ = 0;
}

size_t CudaAllocator::cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cached_bytes_;
}

CudaAllocator::CudaAllocator()
    : cached_bytes_(0), pool_limit_(DEFAULT_POOL_LIMIT) {}

CudaAllocator::~CudaAllocator() { release_pool(); }

} // namespace cuda
} // namespace backends
} // namespace axiom
