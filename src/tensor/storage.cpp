#include "axiom/storage.hpp"

#include <stdexcept>

#include "backends/cpu/cpu_storage.hpp"

#ifdef __APPLE__
namespace axiom {
namespace backends {
namespace metal {
std::unique_ptr<Storage> make_metal_storage(size_t size_bytes);
std::unique_ptr<Storage> make_metal_storage_view(std::shared_ptr<Storage> base,
                                                 size_t offset,
                                                 size_t size_bytes);
bool is_metal_available();
}  // namespace metal
}  // namespace backends
}  // namespace axiom
#endif

namespace axiom {

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<Storage> make_storage(size_t size_bytes, Device device) {
  switch (device) {
    case Device::CPU:
      return backends::cpu::make_cpu_storage(size_bytes);

    case Device::GPU:
#ifdef __APPLE__
      if (backends::metal::is_metal_available()) {
        return backends::metal::make_metal_storage(size_bytes);
      } else {
        throw std::runtime_error("Metal GPU not available on this system");
      }
#else
      throw std::runtime_error("GPU storage not available on this platform");
#endif
  }
  throw std::runtime_error("Unknown device type");
}

std::unique_ptr<Storage> make_storage_view(std::shared_ptr<Storage> base,
                                           size_t offset, size_t size_bytes) {
  switch (base->device()) {
    case Device::CPU:
      return backends::cpu::make_cpu_storage_view(base, offset, size_bytes);

    case Device::GPU:
#ifdef __APPLE__
      return backends::metal::make_metal_storage_view(base, offset, size_bytes);
#else
      throw std::runtime_error("GPU storage not available on this platform");
#endif
  }
  throw std::runtime_error("Unknown device type");
}

}  // namespace axiom