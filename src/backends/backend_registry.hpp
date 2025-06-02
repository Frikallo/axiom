#pragma once

#include "axiom/storage.hpp"
#include <vector>
#include <string>

namespace axiom {
namespace backends {

// Backend capability information
struct BackendInfo {
    std::string name;
    Device device_type;
    bool available;
    std::string description;
};

// Backend registry for listing available backends
class BackendRegistry {
public:
    // Get list of all available backends
    static std::vector<BackendInfo> available_backends();
    
    // Check if a specific device type is available
    static bool is_device_available(Device device);
    
    // Get default device (prefer GPU if available)
    static Device default_device();
    
    // Get device name string
    static std::string device_name(Device device);
};

} // namespace backends
} // namespace axiom