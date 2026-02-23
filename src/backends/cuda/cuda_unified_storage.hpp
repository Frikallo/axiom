#pragma once

#include <memory>

#include "axiom/storage.hpp"
#include "cuda_buffer_provider.hpp"

namespace axiom {
namespace backends {
namespace cuda {

// Storage backed by cudaMallocManaged (unified / managed memory).
// The managed pointer is valid on both CPU and GPU after synchronization.
// The device_tag_ indicates which device should be used for compute,
// not where the memory physically resides — mirroring Metal's
// UnifiedStorage.
class CudaUnifiedStorage : public Storage, public CudaBufferProvider {
  private:
    void *managed_ptr_;
    size_t size_bytes_;
    size_t offset_;
    Device device_tag_;
    bool owns_memory_; // False for zero-copy aliases

    // Private alias constructor — shares existing managed pointer.
    CudaUnifiedStorage(void *managed_ptr, size_t size_bytes, size_t offset,
                       Device tag);

  public:
    CudaUnifiedStorage(size_t size_bytes, Device device_tag);
    ~CudaUnifiedStorage() override;

    // Storage interface — managed pointer is CPU-accessible after sync.
    void *data() override;
    const void *data() const override;
    size_t size_bytes() const override;
    Device device() const override { return device_tag_; }
    void copy_to(Storage &other) const override;
    void copy_from(const Storage &other) override;
    std::unique_ptr<Storage> clone() const override;

    // CudaBufferProvider interface — same pointer works on GPU.
    void *device_ptr() const override { return managed_ptr_; }
    size_t offset() const override { return offset_; }
    bool is_host_accessible() const override { return true; }

    // Zero-copy alias with a different device tag.
    std::unique_ptr<CudaUnifiedStorage> with_device_tag(Device tag) const;
};

// True if device 0 supports managed memory (Pascal+).
bool is_cuda_unified_memory_available();

std::unique_ptr<Storage> make_cuda_unified_storage(size_t size_bytes,
                                                   Device device_tag);

} // namespace cuda
} // namespace backends
} // namespace axiom
