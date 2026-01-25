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

/**
 * @brief Checks if GPU tests should be run.
 *
 * Returns false if:
 * - Metal is not available
 * - The AXIOM_SKIP_GPU_TESTS environment variable is set to "1"
 *
 * @return True if GPU tests should be executed, false otherwise.
 */
bool should_run_gpu_tests();

std::string device_to_string(Device device);

} // namespace system
} // namespace axiom