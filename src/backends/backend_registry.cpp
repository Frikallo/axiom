#include "backend_registry.hpp"

// Forward declare Metal function to avoid including Metal headers in C++
#ifdef __APPLE__
namespace axiom {
namespace backends {
namespace metal {
bool is_metal_available();
}
}  // namespace backends
}  // namespace axiom
#endif

namespace axiom {
namespace backends {

std::vector<BackendInfo> BackendRegistry::available_backends() {
  std::vector<BackendInfo> backends;

  // CPU backend is always available
  backends.push_back({.name = "CPU",
                      .device_type = Device::CPU,
                      .available = true,
                      .description = "CPU backend using system memory"});

#ifdef __APPLE__
  // Check Metal availability
  bool metal_available = metal::is_metal_available();
  backends.push_back(
      {.name = "Metal",
       .device_type = Device::GPU,
       .available = metal_available,
       .description = metal_available ? "Metal GPU backend for Apple Silicon"
                                      : "Metal GPU backend (not available)"});
#endif

  return backends;
}

bool BackendRegistry::is_device_available(Device device) {
  switch (device) {
    case Device::CPU:
      return true;
    case Device::GPU:
#ifdef __APPLE__
      return metal::is_metal_available();
#else
      return false;
#endif
  }
  return false;
}

Device BackendRegistry::default_device() {
  // Prefer GPU if available, fallback to CPU
  if (is_device_available(Device::GPU)) {
    return Device::GPU;
  }
  return Device::CPU;
}

std::string BackendRegistry::device_name(Device device) {
  switch (device) {
    case Device::CPU:
      return "CPU";
    case Device::GPU:
      return "GPU";
  }
  return "Unknown";
}

}  // namespace backends
}  // namespace axiom