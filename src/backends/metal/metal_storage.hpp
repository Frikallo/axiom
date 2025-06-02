#pragma once

#include "axiom/storage.hpp"
#include <memory>

#ifdef __APPLE__

namespace axiom {
namespace backends {
namespace metal {

// Forward declare Metal storage class
// Implementation in .mm file to avoid Metal includes in C++
class MetalStorage;

// Metal backend factory functions
std::unique_ptr<Storage> make_metal_storage(size_t size_bytes);
std::unique_ptr<Storage> make_metal_storage_view(std::shared_ptr<Storage> base, size_t offset, size_t size_bytes);

// Check if Metal is available
bool is_metal_available();

} // namespace metal
} // namespace backends
} // namespace axiom

#endif // __APPLE__