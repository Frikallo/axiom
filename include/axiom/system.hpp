#pragma once

#include <string>

#include "tensor.hpp"

namespace axiom {
namespace system {

/**
 * @brief Checks if a Metal-capable GPU is available on the system.
 *
 * @return True if a Metal device is found, false otherwise.
 * This will always be false on non-Apple platforms.
 */
bool is_metal_available();

std::string device_to_string(Device device);

} // namespace system
} // namespace axiom