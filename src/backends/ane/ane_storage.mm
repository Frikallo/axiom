#import "ane_storage.hpp"
#import "ane_bridge.h"
#import "ane_iosurface.h"

#import <Accelerate/Accelerate.h>
#import <IOSurface/IOSurface.h>

#include "axiom/error.hpp"
#include "backends/cpu/cpu_storage.hpp"

namespace axiom {
namespace backends {
namespace ane {

// ============================================================================
// ANEStorage implementation
// ============================================================================

ANEStorage::ANEStorage(int channels, int spatial_size)
    : channels_(channels), spatial_size_(spatial_size),
      logical_bytes_((size_t)channels * spatial_size * 2 /* FP16 */) {
    IOSurfaceRef surface = ane_create_surface(channels, spatial_size);
    if (!surface) {
        throw MemoryError::allocation_failed(logical_bytes_);
    }
    surface_ = (void *)surface;
}

ANEStorage::~ANEStorage() {
    if (surface_) {
        CFRelease((IOSurfaceRef)surface_);
        surface_ = nullptr;
    }
}

void *ANEStorage::data() {
    // Lock the IOSurface and return base address.
    // Caller is responsible for understanding the FP16 channel-first layout.
    IOSurfaceRef surface = (IOSurfaceRef)surface_;
    IOSurfaceLock(surface, 0, NULL);
    void *base = IOSurfaceGetBaseAddress(surface);
    IOSurfaceUnlock(surface, 0, NULL);
    return base;
}

const void *ANEStorage::data() const {
    IOSurfaceRef surface = (IOSurfaceRef)surface_;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const void *base = IOSurfaceGetBaseAddress(surface);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return base;
}

size_t ANEStorage::size_bytes() const { return logical_bytes_; }

void ANEStorage::copy_to(Storage &other) const {
    if (other.device() == Device::CPU) {
        // ANE (FP16 channel-first) -> CPU (FP32 row-major)
        // Allocate temp buffer for FP32 data
        size_t num_elements = (size_t)channels_ * spatial_size_;
        std::vector<float> tmp(num_elements);

        int rc = ane_surface_read_f32((IOSurfaceRef)surface_, tmp.data(),
                                      channels_, spatial_size_);
        if (rc != 0) {
            throw RuntimeError::internal("Failed to read ANE IOSurface");
        }

        // Copy FP32 data into CPU storage
        void *dst = other.data();
        size_t copy_bytes = std::min(other.size_bytes(), num_elements * sizeof(float));
        std::memcpy(dst, tmp.data(), copy_bytes);

    } else if (other.device() == Device::ANE) {
        // ANE -> ANE: copy IOSurface contents
        auto *other_ane = dynamic_cast<ANEStorage *>(&other);
        if (!other_ane) {
            throw DeviceError("Cannot copy ANE storage to non-ANE storage");
        }

        IOSurfaceRef src = (IOSurfaceRef)surface_;
        IOSurfaceRef dst = (IOSurfaceRef)other_ane->surface_;

        IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(dst, 0, NULL);

        size_t src_alloc = IOSurfaceGetAllocSize(src);
        size_t dst_alloc = IOSurfaceGetAllocSize(dst);
        size_t copy_size = std::min(src_alloc, dst_alloc);

        std::memcpy(IOSurfaceGetBaseAddress(dst),
                     IOSurfaceGetBaseAddress(src), copy_size);

        IOSurfaceUnlock(dst, 0, NULL);
        IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);

    } else {
        // ANE -> GPU: go through CPU as intermediary
        // (Could potentially optimize with IOSurface sharing later)
        size_t num_elements = (size_t)channels_ * spatial_size_;
        auto cpu_storage = cpu::make_cpu_storage(num_elements * sizeof(float));
        copy_to(*cpu_storage);
        cpu_storage->copy_to(other);
    }
}

void ANEStorage::copy_from(const Storage &other) {
    if (other.device() == Device::CPU) {
        // CPU (FP32 row-major) -> ANE (FP16 channel-first)
        const void *src = other.data();
        size_t num_elements = (size_t)channels_ * spatial_size_;

        int rc = ane_surface_write_f32((IOSurfaceRef)surface_,
                                       static_cast<const float *>(src),
                                       channels_, spatial_size_);
        if (rc != 0) {
            throw RuntimeError::internal("Failed to write to ANE IOSurface");
        }

    } else if (other.device() == Device::ANE) {
        // ANE -> ANE: delegate to copy_to
        other.copy_to(*this);

    } else {
        // GPU -> ANE: go through CPU as intermediary
        size_t num_elements = (size_t)channels_ * spatial_size_;
        auto cpu_storage = cpu::make_cpu_storage(num_elements * sizeof(float));
        other.copy_to(*cpu_storage);
        copy_from(*cpu_storage);
    }
}

std::unique_ptr<Storage> ANEStorage::clone() const {
    auto new_storage = std::make_unique<ANEStorage>(channels_, spatial_size_);
    new_storage->copy_from(*this);
    return new_storage;
}

void *ANEStorage::surface() const { return surface_; }

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<Storage> make_ane_storage(int channels, int spatial_size) {
    if (!is_ane_available()) {
        throw DeviceError::not_available("ANE");
    }
    return std::make_unique<ANEStorage>(channels, spatial_size);
}

bool is_ane_available() { return ane_is_available(); }

void initialize_ane_backend() {
    ane_init();
    if (is_ane_available()) {
        // ANE does NOT register per-op operations in the OperationRegistry.
        // ANE works at the graph level via ANECompiledModel (Phase 2).
        // This initialization just ensures the bridge is ready.
    }
}

} // namespace ane
} // namespace backends
} // namespace axiom
