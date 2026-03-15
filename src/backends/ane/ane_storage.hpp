#pragma once

#include <memory>

#include "axiom/storage.hpp"

// Forward-declare IOSurfaceRef to avoid pulling in the framework header
// in pure C++ translation units.
typedef struct __IOSurface *IOSurfaceRef;

namespace axiom {
namespace backends {
namespace ane {

// Storage backed by an IOSurface in ANE's native [1, C, 1, S] FP16 layout.
//
// ANE requires data in channel-first format with 64-byte row alignment.
// This storage handles:
//   - IOSurface allocation with correct dimensions
//   - FP32 <-> FP16 conversion on copy_from/copy_to (CPU path)
//   - Row alignment (64 bytes per row)
//
// Data access via data()/data() const is supported but requires the caller
// to understand the FP16 channel-first layout with padding.
class ANEStorage : public Storage {
  public:
    // Create storage for a tensor with the given logical shape.
    // channels: number of channels (rows in IOSurface)
    // spatial_size: spatial extent (columns in IOSurface)
    // total_elements: channels * spatial_size
    ANEStorage(int channels, int spatial_size);
    ~ANEStorage() override;

    // Storage interface
    void *data() override;
    const void *data() const override;
    size_t size_bytes() const override;
    Device device() const override { return Device::ANE; }
    void copy_to(Storage &other) const override;
    void copy_from(const Storage &other) override;
    std::unique_ptr<Storage> clone() const override;

    // ANE-specific accessors
    void *surface() const; // Returns IOSurfaceRef (as void* for C++ compat)
    int channels() const { return channels_; }
    int spatial_size() const { return spatial_size_; }

  private:
    void *surface_; // IOSurfaceRef (stored as void* to avoid ObjC in header)
    int channels_;
    int spatial_size_;
    size_t logical_bytes_; // channels * spatial_size * sizeof(fp16)
};

// Factory function matching the pattern of make_cpu_storage / make_metal_storage.
// For ANE, the caller must specify the channel-first dimensions.
std::unique_ptr<Storage> make_ane_storage(int channels, int spatial_size);

// Check if ANE is available on this system.
bool is_ane_available();

// Initialize the ANE backend (bridge + availability check).
void initialize_ane_backend();

} // namespace ane
} // namespace backends
} // namespace axiom
