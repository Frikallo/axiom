#pragma once

#include "axiom/storage.hpp"

namespace axiom {
namespace backends {
namespace cuda {

// Interface for any storage type that can provide a CUDA device pointer.
// Mirrors MetalBufferProvider — allows GPU backend code to work with
// different storage implementations uniformly.
class CudaBufferProvider {
  public:
    virtual ~CudaBufferProvider() = default;

    // Raw device pointer (void* on device).
    virtual void *device_ptr() const = 0;

    // Byte offset into the allocation (for views).
    virtual size_t offset() const = 0;

    // True if the backing memory is host-accessible (e.g. cudaMallocManaged).
    virtual bool is_host_accessible() const = 0;
};

// Extract CudaBufferProvider from any Storage pointer.
// Returns nullptr if the storage is not CUDA-backed.
inline const CudaBufferProvider *
as_cuda_buffer_provider(const Storage *storage) {
    return dynamic_cast<const CudaBufferProvider *>(storage);
}

inline CudaBufferProvider *as_cuda_buffer_provider(Storage *storage) {
    return dynamic_cast<CudaBufferProvider *>(storage);
}

} // namespace cuda
} // namespace backends
} // namespace axiom
