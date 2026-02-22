#pragma once

#include <memory>

#include "axiom/storage.hpp"
#include "metal_buffer_provider.hpp"

namespace axiom {
namespace backends {
namespace metal {

// Storage backed by a single shared MTLBuffer in unified memory.
// On Apple Silicon, CPU and GPU share the same physical memory, so this
// provides zero-copy access from both sides. The device_tag_ indicates
// which device should be used for compute, not where the memory lives.
class UnifiedStorage : public Storage, public MetalBufferProvider {
  private:
    void *device_;          // id<MTLDevice>
    void *buffer_;          // id<MTLBuffer> (StorageModeShared)
    void *cached_contents_; // Cached [MTLBuffer contents] pointer
    size_t size_bytes_;
    size_t offset_;
    Device device_tag_; // CPU or GPU — indicates compute target

    // Private sharing constructor — shares an existing MTLBuffer (ARC retain)
    UnifiedStorage(void *device, void *buffer, void *cached_contents,
                   size_t size_bytes, size_t offset, Device tag);

  public:
    UnifiedStorage(void *device, size_t size_bytes, Device device_tag);
    ~UnifiedStorage();

    // Storage interface — CPU-accessible data pointer
    void *data() override;
    const void *data() const override;
    size_t size_bytes() const override;
    Device device() const override { return device_tag_; }
    void copy_to(Storage &other) const override;
    void copy_from(const Storage &other) override;
    std::unique_ptr<Storage> clone() const override;

    // MetalBufferProvider interface — GPU-accessible buffer
    void *buffer() const override { return buffer_; }
    size_t offset() const override { return offset_; }
    bool is_private() const override { return false; }

    // Returns a new UnifiedStorage sharing the same MTLBuffer but with a
    // different device tag. Zero-copy — just changes which device is targeted.
    std::unique_ptr<UnifiedStorage> with_device_tag(Device tag) const;
};

// Check if the current device supports unified memory (Apple Silicon).
bool is_unified_memory_available();

// Factory: create a UnifiedStorage backed by a shared MTLBuffer.
std::unique_ptr<Storage> make_unified_storage(size_t size_bytes,
                                              Device device_tag);

} // namespace metal
} // namespace backends
} // namespace axiom
