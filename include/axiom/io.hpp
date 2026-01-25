#pragma once

#include <map>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace axiom {
namespace io {

// Version information for file format compatibility
constexpr uint32_t AXIOM_FILE_VERSION = 1;
constexpr uint32_t AXIOM_MAGIC_NUMBER = 0x41584D00; // "AXMA"

// File format header structure
struct AxiomFileHeader {
    uint32_t magic;          // Magic number for file identification
    uint32_t version;        // File format version
    uint32_t dtype;          // Data type (cast from DType enum)
    uint32_t ndim;           // Number of dimensions
    uint32_t memory_order;   // Memory order (0=RowMajor, 1=ColMajor)
    uint64_t total_elements; // Total number of elements
    uint64_t data_size;      // Size of data in bytes
    uint64_t shape_offset;   // Offset to shape data
    uint64_t data_offset;    // Offset to tensor data
    uint8_t reserved[32];    // Reserved for future use
};

// Serialization options
struct SerializationOptions {
    MemoryOrder force_order =
        MemoryOrder::RowMajor;  // Force specific memory order
    bool preserve_order = true; // Preserve original memory order
};

// Load/Save exceptions
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
// Core serialization functions
// ============================================================================

/**
 * Save tensor to .axm file
 * @param tensor The tensor to save
 * @param filename Path to output file
 * @param options Serialization options
 */
void save(const Tensor &tensor, const std::string &filename,
          const SerializationOptions &options = SerializationOptions{});

/**
 * Load tensor from .axm file
 * @param filename Path to input file
 * @param device Target device (default: CPU)
 * @return Loaded tensor
 */
Tensor load(const std::string &filename, Device device = Device::CPU);

/**
 * Save tensor data to binary stream
 * @param tensor The tensor to save
 * @param stream Output stream
 * @param options Serialization options
 */
void save_stream(const Tensor &tensor, std::ostream &stream,
                 const SerializationOptions &options = SerializationOptions{});

/**
 * Load tensor from binary stream
 * @param stream Input stream
 * @param device Target device
 * @return Loaded tensor
 */
Tensor load_stream(std::istream &stream, Device device = Device::CPU);

// ============================================================================
// Multi-tensor serialization (archive format)
// ============================================================================

/**
 * Save multiple tensors to a single .axm archive file
 * @param tensors Map of tensor names to tensors
 * @param filename Path to output archive
 * @param options Serialization options
 */
void save_archive(const std::map<std::string, Tensor> &tensors,
                  const std::string &filename,
                  const SerializationOptions &options = SerializationOptions{});

/**
 * Load all tensors from .axm archive file
 * @param filename Path to input archive
 * @param device Target device
 * @return Map of tensor names to loaded tensors
 */
std::map<std::string, Tensor> load_archive(const std::string &filename,
                                           Device device = Device::CPU);

/**
 * List tensor names in archive without loading data
 * @param filename Path to archive file
 * @return Vector of tensor names
 */
std::vector<std::string> list_archive(const std::string &filename);

/**
 * Load specific tensor from archive
 * @param filename Path to archive file
 * @param tensor_name Name of tensor to load
 * @param device Target device
 * @return Loaded tensor
 */
Tensor load_from_archive(const std::string &filename,
                         const std::string &tensor_name,
                         Device device = Device::CPU);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Get file information without loading the tensor
 * @param filename Path to .axm file
 * @return Header information
 */
AxiomFileHeader get_file_info(const std::string &filename);

/**
 * Check if file is a valid .axm file
 * @param filename Path to file
 * @return True if valid .axm file
 */
bool is_axiom_file(const std::string &filename);

/**
 * Validate tensor data integrity
 * @param tensor Tensor to validate
 * @return True if tensor data is valid
 */
bool validate_tensor(const Tensor &tensor);

/**
 * Get human-readable file format version
 * @param version Version number from file
 * @return Version string
 */
std::string version_string(uint32_t version);

// ============================================================================
// String formatting
// ============================================================================

/**
 * Get a NumPy-style string representation of the tensor.
 * @param tensor The tensor to format.
 * @return A string with the formatted tensor.
 */
std::string to_string(const Tensor &tensor);

// ============================================================================
// Convenience functions (NumPy-style API)
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