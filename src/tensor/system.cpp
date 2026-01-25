#include "axiom/system.hpp"

#include <cstdlib>

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

bool should_run_gpu_tests() {
    // Check environment variable first
    const char* skip_env = std::getenv("AXIOM_SKIP_GPU_TESTS");
    if (skip_env && std::string(skip_env) == "1") {
        return false;
    }
    
    // Then check if Metal is available
    return is_metal_available();
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