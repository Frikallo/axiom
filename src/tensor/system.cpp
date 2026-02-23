#include "axiom/system.hpp"

#include <cstdlib>

#ifdef __APPLE__
namespace axiom::backends::metal {
bool is_metal_available();
}
#elif defined(AXIOM_CUDA_SUPPORT)
namespace axiom::backends::cuda {
bool is_cuda_available();
}
#endif

namespace axiom::system {
bool is_metal_available() {
#ifdef __APPLE__
    return backends::metal::is_metal_available();
#else
    return false;
#endif
}

bool is_gpu_available() {
#ifdef __APPLE__
    return backends::metal::is_metal_available();
#elif defined(AXIOM_CUDA_SUPPORT)
    return backends::cuda::is_cuda_available();
#else
    return false;
#endif
}

bool should_run_gpu_tests() {
    // Check environment variable first
    const char *skip_env = std::getenv("AXIOM_SKIP_GPU_TESTS");
    if (skip_env && std::string(skip_env) == "1") {
        return false;
    }

    return is_gpu_available();
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
