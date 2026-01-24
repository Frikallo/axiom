#import "metal_storage.hpp"

#import <Metal/Metal.h>
#import "axiom/error.hpp"
#include "backends/cpu/cpu_storage.hpp"

namespace axiom {
namespace backends {
namespace metal {

static id<MTLDevice> g_metal_device = nil;
static dispatch_once_t g_metal_device_once;

void init_metal_device() {
    dispatch_once(&g_metal_device_once, ^{
        g_metal_device = MTLCreateSystemDefaultDevice();
    });
}

bool is_metal_available() {
    init_metal_device();
    return g_metal_device != nil;
}

MetalStorage::MetalStorage(void* device, size_t size_bytes)
    : device_(device), size_bytes_(size_bytes), offset_(0), base_storage_(nullptr) {
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_;
    id<MTLBuffer> mtl_buffer = [mtl_device newBufferWithLength:size_bytes options:MTLResourceStorageModeShared];
    if (!mtl_buffer) {
        throw MemoryError::allocation_failed(size_bytes);
    }
    buffer_ = (__bridge_retained void*)mtl_buffer;
}

MetalStorage::~MetalStorage() {
    if (buffer_) {
        // The buffer was retained in the constructor, so it needs to be released.
        id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)buffer_;
        (void)mtl_buffer; // ARC will release it.
    }
}

void* MetalStorage::data() {
    throw DeviceError("Cannot directly access GPU memory from CPU. Use copy_to to transfer data");
}

const void* MetalStorage::data() const {
    throw DeviceError("Cannot directly access GPU memory from CPU. Use copy_to to transfer data");
}

size_t MetalStorage::size_bytes() const {
    return size_bytes_;
}

void MetalStorage::copy_to(Storage& other) const {
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer_;
    if (other.device() == Device::CPU) {
        void* cpu_ptr = other.data();
        const void* metal_ptr = [mtl_buffer contents];
        std::memcpy(cpu_ptr, static_cast<const uint8_t*>(metal_ptr) + offset_, size_bytes());
    } else if (other.device() == Device::GPU) {
        auto& other_metal = static_cast<MetalStorage&>(other);
        id<MTLBuffer> other_mtl_buffer = (__bridge id<MTLBuffer>)other_metal.buffer_;

        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmd_buffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buffer blitCommandEncoder];

        [encoder copyFromBuffer:mtl_buffer
                   sourceOffset:offset_
                       toBuffer:other_mtl_buffer
              destinationOffset:other_metal.offset_
                           size:std::min(size_bytes_, other.size_bytes())];
        
        [encoder endEncoding];
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];
    }
}

void MetalStorage::copy_from(const Storage& other) {
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer_;
    if (other.device() == Device::CPU) {
        const void* cpu_ptr = other.data();
        void* metal_ptr = [mtl_buffer contents];
        std::memcpy(static_cast<uint8_t*>(metal_ptr) + offset_, cpu_ptr, size_bytes());
    } else if (other.device() == Device::GPU) {
        const_cast<Storage&>(other).copy_to(*this);
    }
}

std::unique_ptr<Storage> MetalStorage::clone() const {
    auto new_storage = std::make_unique<MetalStorage>(device_, size_bytes_);
    new_storage->copy_from(*this);
    return new_storage;
}

std::unique_ptr<Storage> make_metal_storage(size_t size_bytes) {
    init_metal_device();
    if (!is_metal_available()) {
        throw DeviceError::not_available("Metal");
    }
    return std::make_unique<MetalStorage>((__bridge void*)g_metal_device, size_bytes);
}

} // namespace metal
} // namespace backends
} // namespace axiom