#pragma once

#include <memory>

#include "axiom/storage.hpp"

namespace axiom {
namespace backends {
namespace metal {

class MetalStorage : public Storage {
  private:
    void *device_; // id<MTLDevice>
    void *buffer_; // id<MTLBuffer>
    size_t size_bytes_;
    std::shared_ptr<Storage> base_storage_;
    size_t offset_;

  public:
    explicit MetalStorage(void *device, size_t size_bytes);
    ~MetalStorage();

    void *data() override;
    const void *data() const override;
    size_t size_bytes() const override;
    Device device() const override { return Device::GPU; }
    void copy_to(Storage &other) const override;
    void copy_from(const Storage &other) override;
    std::unique_ptr<Storage> clone() const override;

    void *buffer() const { return buffer_; } // id<MTLBuffer>
    size_t offset() const { return offset_; }
};

std::unique_ptr<Storage> make_metal_storage(size_t size_bytes);
bool is_metal_available();

} // namespace metal
} // namespace backends
} // namespace axiom