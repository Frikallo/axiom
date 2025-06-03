#pragma once

#include <memory>

#include "axiom/storage.hpp"

#ifdef __APPLE__

namespace axiom {
namespace backends {
namespace metal {

class MetalStorage;

std::unique_ptr<Storage> make_metal_storage(size_t size_bytes);
std::unique_ptr<Storage> make_metal_storage_view(std::shared_ptr<Storage> base,
                                                 size_t offset,
                                                 size_t size_bytes);

bool is_metal_available();

}  // namespace metal
}  // namespace backends
}  // namespace axiom

#endif  // __APPLE__