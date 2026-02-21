#pragma once

#include "axiom/storage.hpp"

namespace axiom {
namespace backends {
namespace metal {

// Interface for any storage type that can provide a Metal buffer.
// Both MetalStorage and UnifiedStorage implement this, allowing GPU backend
// code to work with either without knowing the concrete type.
class MetalBufferProvider {
  public:
    virtual ~MetalBufferProvider() = default;

    virtual void *buffer() const = 0;
    virtual size_t offset() const = 0;
    virtual bool is_private() const = 0;
};

// Extracts MetalBufferProvider from any Storage pointer.
// Returns nullptr if the storage doesn't provide a Metal buffer.
inline const MetalBufferProvider *
as_metal_buffer_provider(const Storage *storage) {
    return dynamic_cast<const MetalBufferProvider *>(storage);
}

inline MetalBufferProvider *as_metal_buffer_provider(Storage *storage) {
    return dynamic_cast<MetalBufferProvider *>(storage);
}

} // namespace metal
} // namespace backends
} // namespace axiom
