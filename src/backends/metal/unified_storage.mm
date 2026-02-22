#import "unified_storage.hpp"
#import "metal_common.hpp"

#import <Metal/Metal.h>
#import "axiom/error.hpp"

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// Unified memory detection
// ============================================================================

bool is_unified_memory_available() {
    static const bool cached = [] {
        id<MTLDevice> device =
            (__bridge id<MTLDevice>)MetalContext::instance().device();
        if (!device)
            return false;
        if (@available(macOS 12.0, iOS 15.0, *)) {
            return static_cast<bool>([device hasUnifiedMemory]);
        }
        return false;
    }();
    return cached;
}

// ============================================================================
// Construction / Destruction
// ============================================================================

UnifiedStorage::UnifiedStorage(void *device, size_t size_bytes,
                               Device device_tag)
    : device_(device), cached_contents_(nullptr), size_bytes_(size_bytes),
      offset_(0), device_tag_(device_tag) {
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_;
    id<MTLBuffer> mtl_buffer =
        [mtl_device newBufferWithLength:size_bytes
                                options:MTLResourceStorageModeShared];
    if (!mtl_buffer) {
        throw MemoryError::allocation_failed(size_bytes);
    }
    buffer_ = (__bridge_retained void *)mtl_buffer;
    cached_contents_ = [mtl_buffer contents];
}

UnifiedStorage::UnifiedStorage(void *device, void *buffer,
                               void *cached_contents, size_t size_bytes,
                               size_t offset, Device tag)
    : device_(device), cached_contents_(cached_contents),
      size_bytes_(size_bytes), offset_(offset), device_tag_(tag) {
    // ARC-retain the shared buffer
    id<MTLBuffer> shared = (__bridge id<MTLBuffer>)buffer;
    buffer_ = (__bridge_retained void *)shared;
}

UnifiedStorage::~UnifiedStorage() {
    if (buffer_) {
        id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)buffer_;
        (void)mtl_buffer; // ARC releases
    }
}

// ============================================================================
// Storage interface
// ============================================================================

void *UnifiedStorage::data() {
    return static_cast<uint8_t *>(cached_contents_) + offset_;
}

const void *UnifiedStorage::data() const {
    return static_cast<const uint8_t *>(cached_contents_) + offset_;
}

size_t UnifiedStorage::size_bytes() const { return size_bytes_; }

void UnifiedStorage::copy_to(Storage &other) const {
    if (device_tag_ == Device::GPU) {
        // Synchronize GPU before reading from the buffer
        MetalExecutionStream::instance().synchronize();
    }

    if (other.device() == Device::CPU ||
        dynamic_cast<UnifiedStorage *>(&other) != nullptr) {
        // Both sides are CPU-accessible — plain memcpy
        std::memcpy(other.data(), data(), size_bytes_);
    } else {
        // Target is a private Metal buffer — use blit
        auto *target = dynamic_cast<MetalBufferProvider *>(&other);
        if (!target)
            throw DeviceError("Cannot copy unified storage to unknown device");

        id<MTLBuffer> src = (__bridge id<MTLBuffer>)buffer_;
        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)target->buffer();

        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];

        [enc copyFromBuffer:src
               sourceOffset:offset_
                   toBuffer:dst
          destinationOffset:target->offset()
                       size:std::min(size_bytes_, other.size_bytes())];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void UnifiedStorage::copy_from(const Storage &other) {
    if (other.device() == Device::GPU) {
        // Source might have pending GPU work — synchronize
        MetalExecutionStream::instance().synchronize();
    }

    auto *other_unified =
        dynamic_cast<const UnifiedStorage *>(&other);
    if (other.device() == Device::CPU || other_unified != nullptr) {
        std::memcpy(data(), other.data(), size_bytes_);
    } else {
        // Source is a private Metal buffer — use blit
        auto *source =
            dynamic_cast<const MetalBufferProvider *>(&other);
        if (!source)
            throw DeviceError("Cannot copy to unified storage from unknown device");

        id<MTLBuffer> src = (__bridge id<MTLBuffer>)source->buffer();
        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)buffer_;

        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];

        [enc copyFromBuffer:src
               sourceOffset:source->offset()
                   toBuffer:dst
          destinationOffset:offset_
                       size:std::min(other.size_bytes(), size_bytes_)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

std::unique_ptr<Storage> UnifiedStorage::clone() const {
    if (device_tag_ == Device::GPU) {
        MetalExecutionStream::instance().synchronize();
    }
    auto copy = std::make_unique<UnifiedStorage>(device_, size_bytes_,
                                                  device_tag_);
    std::memcpy(copy->data(), data(), size_bytes_);
    return copy;
}

// ============================================================================
// Zero-copy device tag switch
// ============================================================================

std::unique_ptr<UnifiedStorage>
UnifiedStorage::with_device_tag(Device tag) const {
    return std::unique_ptr<UnifiedStorage>(new UnifiedStorage(
        device_, buffer_, cached_contents_, size_bytes_, offset_, tag));
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Storage> make_unified_storage(size_t size_bytes,
                                              Device device_tag) {
    void *device = MetalContext::instance().device();
    if (!device)
        throw DeviceError::not_available("Metal");
    return std::make_unique<UnifiedStorage>(device, size_bytes, device_tag);
}

} // namespace metal
} // namespace backends
} // namespace axiom
