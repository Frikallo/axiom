#pragma once

#include <map>
#include <string>
#include <vector>

#include "axiom/io/numpy.hpp"
#include "axiom/tensor.hpp"

namespace axiom {
namespace io {

// ============================================================================
// File Format Detection
// ============================================================================

enum class FileFormat { Unknown, Axiom, NumPy };

/**
 * Detect file format by examining magic bytes.
 * @param filename Path to file
 * @return Detected file format
 */
FileFormat detect_format(const std::string &filename);

/**
 * Get human-readable name for file format.
 */
std::string format_name(FileFormat format);

// ============================================================================
// Exceptions
// ============================================================================

class SerializationError : public std::runtime_error {
  public:
    explicit SerializationError(const std::string &message)
        : std::runtime_error("Axiom serialization error: " + message) {}
};

class FileFormatError : public SerializationError {
  public:
    explicit FileFormatError(const std::string &message)
        : SerializationError("Invalid file format: " + message) {}
};

// ============================================================================
// Universal Load Functions (Auto-detect format)
// ============================================================================

/**
 * Load tensor from file, auto-detecting format.
 * Supports: .axfb (FlatBuffers), .npy (NumPy)
 * @param filename Path to input file
 * @param device Target device (default: CPU)
 * @return Loaded tensor
 */
Tensor load(const std::string &filename, Device device = Device::CPU);

/**
 * Load multiple tensors from archive file.
 * For NumPy files, returns single tensor with filename as key.
 * @param filename Path to input file
 * @param device Target device
 * @return Map of tensor names to loaded tensors
 */
std::map<std::string, Tensor> load_archive(const std::string &filename,
                                           Device device = Device::CPU);

// ============================================================================
// Save Functions (FlatBuffers format)
// ============================================================================

/**
 * Save tensor to .axfb file (FlatBuffers format).
 * @param tensor The tensor to save
 * @param filename Path to output file
 */
void save(const Tensor &tensor, const std::string &filename);

/**
 * Save multiple tensors to .axfb archive file.
 * @param tensors Map of tensor names to tensors
 * @param filename Path to output archive
 */
void save_archive(const std::map<std::string, Tensor> &tensors,
                  const std::string &filename);

// ============================================================================
// NumPy Format Support
// ============================================================================

// numpy namespace is defined in io/numpy.hpp (included above)

// ============================================================================
// FlatBuffers Format Support
// ============================================================================

namespace flatbuffers {

/**
 * Check if file is a valid Axiom FlatBuffers file.
 * @param filename Path to file
 * @return True if file has AXFB magic bytes
 */
bool is_axfb_file(const std::string &filename);

/**
 * Load tensor from .axfb file.
 * @param filename Path to .axfb file
 * @param device Target device
 * @return Loaded tensor (first tensor in archive)
 */
Tensor load(const std::string &filename, Device device = Device::CPU);

/**
 * Load all tensors from .axfb archive.
 * @param filename Path to .axfb file
 * @param device Target device
 * @return Map of tensor names to loaded tensors
 */
std::map<std::string, Tensor> load_archive(const std::string &filename,
                                           Device device = Device::CPU);

/**
 * Save tensor to .axfb file.
 * @param tensor The tensor to save
 * @param filename Path to output file
 */
void save(const Tensor &tensor, const std::string &filename);

/**
 * Save multiple tensors to .axfb archive.
 * @param tensors Map of tensor names to tensors
 * @param filename Path to output file
 */
void save_archive(const std::map<std::string, Tensor> &tensors,
                  const std::string &filename);

/**
 * List tensor names in archive without loading data.
 * @param filename Path to archive file
 * @return Vector of tensor names
 */
std::vector<std::string> list_archive(const std::string &filename);

/**
 * Load specific tensor from archive by name.
 * @param filename Path to archive file
 * @param tensor_name Name of tensor to load
 * @param device Target device
 * @return Loaded tensor
 */
Tensor load_from_archive(const std::string &filename,
                         const std::string &tensor_name,
                         Device device = Device::CPU);

} // namespace flatbuffers

// ============================================================================
// String Formatting
// ============================================================================

/**
 * Get a NumPy-style string representation of the tensor.
 * @param tensor The tensor to format.
 * @return A string with the formatted tensor.
 */
std::string to_string(const Tensor &tensor);

// ============================================================================
// Convenience Aliases
// ============================================================================

inline void save_tensor(const Tensor &tensor, const std::string &filename) {
    save(tensor, filename);
}

inline Tensor load_tensor(const std::string &filename,
                          Device device = Device::CPU) {
    return load(filename, device);
}

inline void save_tensors(const std::map<std::string, Tensor> &tensors,
                         const std::string &filename) {
    save_archive(tensors, filename);
}

inline std::map<std::string, Tensor> load_tensors(const std::string &filename,
                                                  Device device = Device::CPU) {
    return load_archive(filename, device);
}

} // namespace io
} // namespace axiom
