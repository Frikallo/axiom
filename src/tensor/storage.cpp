#include "axiom/storage.hpp"

#include "axiom/error.hpp"
#include "backends/cpu/cpu_storage.hpp"

#ifdef __APPLE__
namespace axiom {
namespace backends {
namespace metal {
std::unique_ptr<Storage> make_metal_storage(size_t size_bytes);
bool is_metal_available();
} // namespace metal
} // namespace backends
} // namespace axiom
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
            throw DeviceError::not_available("Metal GPU");
        }
#else
        throw DeviceError::not_available("GPU storage on this platform");
#endif
    }
    throw DeviceError("Unknown device type");
}

} // namespace axiom