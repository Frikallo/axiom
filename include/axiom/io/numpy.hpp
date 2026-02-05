#pragma once

#include <string>

#include "axiom/tensor.hpp"

namespace axiom {
namespace io {
namespace numpy {

// NumPy .npy format magic bytes
constexpr uint8_t NPY_MAGIC[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
constexpr size_t NPY_MAGIC_SIZE = 6;

/**
 * Check if file has NumPy magic bytes.
 * @param filename Path to file
 * @return True if file starts with NumPy magic (\x93NUMPY)
 */
bool is_npy_file(const std::string &filename);

/**
 * Load tensor from NumPy .npy file.
 *
 * Supports:
 * - All standard NumPy dtypes (float, int, uint, bool, complex)
 * - Both little-endian and big-endian (with automatic conversion)
 * - C-order (row major) and Fortran-order (column major) arrays
 * - NPY format versions 1.0, 2.0, and 3.0
 *
 * @param filename Path to .npy file
 * @param device Target device (default: CPU)
 * @return Loaded tensor
 */
Tensor load(const std::string &filename, Device device = Device::CPU);

} // namespace numpy
} // namespace io
} // namespace axiom
