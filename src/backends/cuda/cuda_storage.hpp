#pragma once

#include <memory>

#include "axiom/storage.hpp"
#include "cuda_buffer_provider.hpp"

namespace axiom {
namespace backends {
namespace cuda {

class CudaStorage : public Storage, public CudaBufferProvider {
  private:
    void *device_ptr_;   // Device memory pointer
    size_t size_bytes_;  // Requested size
    size_t alloc_size_;  // Rounded-up bucket size used by the allocator
    size_t offset_;

  public:
    explicit CudaStorage(size_t size_bytes);
    ~CudaStorage() override;

    // Storage interface
    void *data() override;
    const void *data() const override;
    size_t size_bytes() const override;
    Device device() const override { return Device::GPU; }
    void copy_to(Storage &other) const override;
    void copy_from(const Storage &other) override;
    std::unique_ptr<Storage> clone() const override;

    // CudaBufferProvider interface
    void *device_ptr() const override { return device_ptr_; }
    size_t offset() const override { return offset_; }
    bool is_host_accessible() const override { return false; }
};

std::unique_ptr<Storage> make_cuda_storage(size_t size_bytes);

} // namespace cuda
} // namespace backends
} // namespace axiom
