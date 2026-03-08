#pragma once

#include <cstddef>
#include <map>
#include <mutex>

namespace axiom {
namespace backends {
namespace cuda {

// Block-caching allocator that sits in front of cudaMalloc/cudaFree.
// Sizes are rounded up to the next power-of-two so that similarly-sized
// allocations can reuse freed blocks.  A configurable pool limit caps
// how much free memory is kept cached (default 512 MB).
class CudaAllocator {
  public:
    static CudaAllocator &instance();

    // Allocate at least `size_bytes` of device memory.
    // Returns a cached block when one of the right bucket size exists,
    // otherwise falls back to cudaMalloc.
    void *allocate(size_t size_bytes);

    // Return a block to the cache.  `alloc_size` must be the value
    // originally returned by round_up() (i.e. the bucket size, NOT the
    // raw request size).  If the pool is already at its limit the block
    // is freed immediately via cudaFree.
    void deallocate(void *ptr, size_t alloc_size);

    // Free every cached block.  Called automatically on destruction.
    void release_pool();

    // Current bytes held in the free-block cache.
    size_t cached_bytes() const;

    // Maximum bytes the cache will hold before evicting to cudaFree.
    static constexpr size_t DEFAULT_POOL_LIMIT = 512ULL * 1024 * 1024;

    // Round `size` up to the next power of two (bucket key).
    static size_t round_up(size_t size);

  private:
    CudaAllocator();
    ~CudaAllocator();

    CudaAllocator(const CudaAllocator &) = delete;
    CudaAllocator &operator=(const CudaAllocator &) = delete;

    // Free blocks keyed by bucket size.  multimap allows many blocks of
    // the same size to coexist.
    std::multimap<size_t, void *> free_blocks_;
    size_t cached_bytes_;
    size_t pool_limit_;
    mutable std::mutex mutex_;
};

} // namespace cuda
} // namespace backends
} // namespace axiom
