#include "axiom/system.hpp"

// Forward-declare the internal function from the Metal backend.
// The actual implementation is in an Objective-C++ file.
namespace axiom::backends::metal {
bool is_metal_available();
}

namespace axiom::system {
bool is_metal_available() {
#ifdef __APPLE__
    return backends::metal::is_metal_available();
#else
    return false;
#endif
}

std::string device_to_string(Device device) {
    switch (device) {
    case Device::CPU:
        return "CPU";
    case Device::GPU:
        return "GPU";
    default:
        return "Unknown";
    }
}
} // namespace axiom::system