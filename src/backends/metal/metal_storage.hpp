#pragma once

#include <memory>

#include "axiom/storage.hpp"
#include "metal_buffer_provider.hpp"

namespace axiom {
namespace backends {
namespace metal {

// Metal buffer storage modes
enum class MetalStorageMode {
    Shared, // MTLResourceStorageModeShared - CPU and GPU can access
    Private // MTLResourceStorageModePrivate - GPU-only, faster for
            // intermediates
};

class MetalStorage : public Storage, public MetalBufferProvider {
  private:
    void *device_; // id<MTLDevice>
    void *buffer_; // id<MTLBuffer>
    size_t size_bytes_;
    std::shared_ptr<Storage> base_storage_;
    size_t offset_;
    MetalStorageMode storage_mode_;

  public:
    explicit MetalStorage(void *device, size_t size_bytes,
                          MetalStorageMode mode = MetalStorageMode::Shared);
    ~MetalStorage();

    void *data() override;
    const void *data() const override;
    size_t size_bytes() const override;
    Device device() const override { return Device::GPU; }
    void copy_to(Storage &other) const override;
    void copy_from(const Storage &other) override;
    std::unique_ptr<Storage> clone() const override;

    void *buffer() const override { return buffer_; } // id<MTLBuffer>
    size_t offset() const override { return offset_; }
    MetalStorageMode storage_mode() const { return storage_mode_; }
    bool is_private() const override {
        return storage_mode_ == MetalStorageMode::Private;
    }
};

std::unique_ptr<Storage> make_metal_storage(size_t size_bytes);
std::unique_ptr<Storage> make_metal_storage_private(size_t size_bytes);
bool is_metal_available();

} // namespace metal
} // namespace backends
} // namespace axiom