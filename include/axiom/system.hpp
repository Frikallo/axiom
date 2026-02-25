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
 * @brief Checks if any GPU backend is available (Metal or CUDA).
 *
 * @return True if a GPU device is found, false otherwise.
 */
bool is_gpu_available();

/**
 * @brief Checks if GPU tests should be run.
 *
 * Returns false if:
 * - No GPU backend is available (Metal or CUDA)
 * - The AXIOM_SKIP_GPU_TESTS environment variable is set to "1"
 *
 * @return True if GPU tests should be executed, false otherwise.
 */
bool should_run_gpu_tests();

std::string device_to_string(Device device);

/**
 * @brief Synchronize the GPU device, blocking until all pending work completes.
 *
 * Equivalent to torch.cuda.synchronize() / torch.mps.synchronize().
 * This does NOT transfer data — use tensor.cpu() for that.
 * No-op if no GPU backend is active.
 */
void synchronize();

} // namespace system
} // namespace axiom
