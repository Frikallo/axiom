#include "metal_storage.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>

#ifdef __APPLE__
#include <Metal/Metal.h>

namespace axiom {
namespace backends {
namespace metal {

// Metal GPU storage implementation
class MetalStorage : public Storage {
private:
    id<MTLBuffer> buffer_;
    size_t size_bytes_;
    size_t offset_;
    std::shared_ptr<Storage> base_storage_;
    
    static id<MTLDevice> get_default_device();
    
public:
    // Create new Metal storage
    explicit MetalStorage(size_t size_bytes);
    
    // Create view of existing storage
    MetalStorage(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes);
    
    void* data() override { return nullptr; } // GPU memory not directly accessible
    const void* data() const override { return nullptr; }
    size_t size_bytes() const override;
    Device device() const override { return Device::GPU; }
    void copy_to(Storage& other) const override;
    void copy_from(const Storage& other) override;
    std::unique_ptr<Storage> clone() const override;
    bool is_view() const override;
    std::shared_ptr<Storage> base() const override;
    
    // Metal-specific methods
    id<MTLBuffer> metal_buffer() const { return buffer_; }
    size_t offset() const { return offset_; }
};

// ============================================================================
// MetalStorage Implementation
// ============================================================================

id<MTLDevice> MetalStorage::get_default_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Failed to create Metal device");
    }
    return device;
}

MetalStorage::MetalStorage(size_t size_bytes)
    : buffer_(nil)
    , size_bytes_(size_bytes)
    , offset_(0)
    , base_storage_(nullptr) {
    
    id<MTLDevice> device = get_default_device();
    buffer_ = [device newBufferWithLength:size_bytes options:MTLResourceStorageModeShared];
    
    if (!buffer_) {
        throw std::runtime_error("Failed to allocate Metal buffer");
    }
}

MetalStorage::MetalStorage(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes)
    : buffer_(nil)
    , size_bytes_(size_bytes)
    , offset_(offset)
    , base_storage_(base) {
    
    if (base->device() != Device::GPU) {
        throw std::runtime_error("Cannot create Metal view of non-Metal storage");
    }
    
    // Get the underlying Metal storage
    auto metal_base = std::dynamic_pointer_cast<MetalStorage>(base);
    if (!metal_base) {
        throw std::runtime_error("Invalid Metal storage cast");
    }
    
    if (offset + size_bytes > metal_base->size_bytes()) {
        throw std::runtime_error("View exceeds base storage bounds");
    }
    
    // Share the underlying buffer
    buffer_ = metal_base->buffer_;
    offset_ += metal_base->offset_; // Compound the offset
}

size_t MetalStorage::size_bytes() const {
    return size_bytes_;
}

void MetalStorage::copy_to(Storage& other) const {
    if (other.device() == Device::GPU) {
        // Metal to Metal copy
        auto other_metal = dynamic_cast<MetalStorage*>(&other);
        if (!other_metal) {
            throw std::runtime_error("Invalid Metal storage cast");
        }
        
        id<MTLDevice> device = get_default_device();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd_buffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buffer blitCommandEncoder];
        
        [encoder copyFromBuffer:buffer_
                   sourceOffset:offset_
                       toBuffer:other_metal->buffer_
              destinationOffset:other_metal->offset_
                           size:std::min(size_bytes_, other.size_bytes())];
        
        [encoder endEncoding];
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];
    } else {
        // Metal to CPU copy
        const void* buffer_data = [buffer_ contents];
        std::memcpy(other.data(), 
                   static_cast<const uint8_t*>(buffer_data) + offset_,
                   std::min(size_bytes_, other.size_bytes()));
    }
}

void MetalStorage::copy_from(const Storage& other) {
    if (other.device() == Device::GPU) {
        // Metal to Metal copy
        const auto* other_metal = dynamic_cast<const MetalStorage*>(&other);
        if (!other_metal) {
            throw std::runtime_error("Invalid Metal storage cast");
        }
        
        id<MTLDevice> device = get_default_device();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd_buffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buffer blitCommandEncoder];
        
        [encoder copyFromBuffer:other_metal->buffer_
                   sourceOffset:other_metal->offset_
                       toBuffer:buffer_
              destinationOffset:offset_
                           size:std::min(size_bytes_, other.size_bytes())];
        
        [encoder endEncoding];
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];
    } else {
        // CPU to Metal copy
        void* buffer_data = [buffer_ contents];
        std::memcpy(static_cast<uint8_t*>(buffer_data) + offset_,
                   other.data(),
                   std::min(size_bytes_, other.size_bytes()));
    }
}

std::unique_ptr<Storage> MetalStorage::clone() const {
    auto new_storage = std::make_unique<MetalStorage>(size_bytes_);
    new_storage->copy_from(*this);
    return new_storage;
}

bool MetalStorage::is_view() const {
    return base_storage_ != nullptr;
}

std::shared_ptr<Storage> MetalStorage::base() const {
    return base_storage_ ? base_storage_ : nullptr;
}

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<Storage> make_metal_storage(size_t size_bytes) {
    return std::make_unique<MetalStorage>(size_bytes);
}

std::unique_ptr<Storage> make_metal_storage_view(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes) {
    return std::make_unique<MetalStorage>(base, offset, size_bytes);
}

bool is_metal_available() {
    @try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    } @catch (NSException* exception) {
        return false;
    }
}

} // namespace metal
} // namespace backends
} // namespace axiom

#endif // __APPLE__