#include "axiom/io.hpp"

#include <cstring>
#include <fstream>
#include <map>
#include <iomanip>
#include <sstream>

namespace axiom {
namespace io {

// ============================================================================
// Helper functions
// ============================================================================

namespace {

// Write value to stream with endianness handling
template <typename T>
void write_value(std::ostream& stream, const T& value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Read value from stream with endianness handling
template <typename T>
T read_value(std::istream& stream) {
  T value;
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream.good()) {
    throw SerializationError("Failed to read value from stream");
  }
  return value;
}

// Convert DType to uint32_t for serialization
uint32_t dtype_to_uint32(DType dtype) { return static_cast<uint32_t>(dtype); }

// Convert uint32_t back to DType
DType uint32_to_dtype(uint32_t value) { return static_cast<DType>(value); }

// Convert MemoryOrder to uint32_t
uint32_t memory_order_to_uint32(MemoryOrder order) {
  return (order == MemoryOrder::RowMajor) ? 0 : 1;
}

// Convert uint32_t to MemoryOrder
MemoryOrder uint32_to_memory_order(uint32_t value) {
  return (value == 0) ? MemoryOrder::RowMajor : MemoryOrder::ColMajor;
}

// Create file header from tensor
AxiomFileHeader create_header(const Tensor& tensor) {
  AxiomFileHeader header = {};

  header.magic = AXIOM_MAGIC_NUMBER;
  header.version = AXIOM_FILE_VERSION;
  header.dtype = dtype_to_uint32(tensor.dtype());
  header.ndim = static_cast<uint32_t>(tensor.ndim());
  header.memory_order = memory_order_to_uint32(tensor.memory_order());
  header.total_elements = tensor.size();
  header.data_size = tensor.nbytes();

  // Calculate offsets
  header.shape_offset = sizeof(AxiomFileHeader);
  header.data_offset = header.shape_offset + tensor.ndim() * sizeof(uint64_t);

  return header;
}

// Validate header
void validate_header(const AxiomFileHeader& header) {
  if (header.magic != AXIOM_MAGIC_NUMBER) {
    throw FileFormatError("Invalid magic number - not an Axiom file");
  }

  if (header.version > AXIOM_FILE_VERSION) {
    throw FileFormatError("Unsupported file version: " +
                          std::to_string(header.version));
  }

  if (header.ndim > 32) {  // Reasonable limit
    throw FileFormatError("Too many dimensions: " +
                          std::to_string(header.ndim));
  }

  if (header.total_elements == 0 && header.ndim > 0) {
    throw FileFormatError("Zero elements with non-zero dimensions");
  }
}

// Copy tensor data to CPU if needed
Tensor ensure_cpu_tensor(const Tensor& tensor) {
  if (tensor.device() == Device::CPU) {
    return tensor;
  } else {
    return tensor.cpu();
  }
}

// Helper to convert a single element to string based on dtype
template<typename T>
std::string element_to_string(const void* data, size_t index) {
    std::stringstream ss;
    if constexpr (std::is_floating_point_v<T>) {
        ss << std::fixed << std::setprecision(4) << static_cast<const T*>(data)[index];
    } else {
        ss << static_cast<const T*>(data)[index];
    }
    return ss.str();
}

std::string dispatch_element_to_string(const Tensor& t, size_t index) {
    const void* data = t.data();
    switch (t.dtype()) {
        case DType::Float32:  return element_to_string<float>(data, index);
        case DType::Float64:  return element_to_string<double>(data, index);
        case DType::Float16:  return element_to_string<half_float::half>(data, index);
        case DType::Int8:     return std::to_string(static_cast<const int8_t*>(data)[index]);
        case DType::Int16:    return element_to_string<int16_t>(data, index);
        case DType::Int32:    return element_to_string<int32_t>(data, index);
        case DType::Int64:    return element_to_string<int64_t>(data, index);
        case DType::UInt8:    return std::to_string(static_cast<const uint8_t*>(data)[index]);
        case DType::UInt16:   return element_to_string<uint16_t>(data, index);
        case DType::UInt32:   return element_to_string<uint32_t>(data, index);
        case DType::UInt64:   return element_to_string<uint64_t>(data, index);
        case DType::Bool:     return static_cast<const bool*>(data)[index] ? "true" : "false";
        default: throw std::runtime_error("Unsupported dtype for printing");
    }
}

void print_recursive(std::stringstream& ss, const Tensor& t, std::vector<size_t>& coords, int dim,
                     size_t edge_items) {
    ss << "[";
    size_t dim_size = t.shape()[dim];

    if (dim == t.ndim() - 1) {
        if (dim_size > 2 * edge_items) {
            for (size_t i = 0; i < edge_items; ++i) {
                coords[dim] = i;
                size_t offset = ShapeUtils::linear_index(coords, t.strides()) / t.itemsize();
                ss << dispatch_element_to_string(t, offset) << " ";
            }
            ss << "... ";
            for (size_t i = dim_size - edge_items; i < dim_size; ++i) {
                coords[dim] = i;
                size_t offset = ShapeUtils::linear_index(coords, t.strides()) / t.itemsize();
                ss << dispatch_element_to_string(t, offset);
                if (i < dim_size - 1) ss << " ";
            }
        } else {
            for (size_t i = 0; i < dim_size; ++i) {
                coords[dim] = i;
                size_t offset = ShapeUtils::linear_index(coords, t.strides()) / t.itemsize();
                ss << dispatch_element_to_string(t, offset);
                if (i < dim_size - 1) ss << " ";
            }
        }
    } else {
        if (dim_size > 2 * edge_items) {
            for (size_t i = 0; i < edge_items; ++i) {
                 if (i > 0) {
                    ss << "\n";
                    for (int j = 0; j <= dim; ++j) ss << " ";
                }
                coords[dim] = i;
                print_recursive(ss, t, coords, dim + 1, edge_items);
            }
            ss << "\n";
            for (int j = 0; j <= dim; ++j) ss << " ";
            ss << "...";
            for (size_t i = dim_size - edge_items; i < dim_size; ++i) {
                ss << "\n";
                for (int j = 0; j <= dim; ++j) ss << " ";
                coords[dim] = i;
                print_recursive(ss, t, coords, dim + 1, edge_items);
            }

        } else {
            for (size_t i = 0; i < dim_size; ++i) {
                if (i > 0) {
                    ss << "\n";
                    for (int j = 0; j <= dim; ++j) {
                        ss << " ";
                    }
                }
                coords[dim] = i;
                print_recursive(ss, t, coords, dim + 1, edge_items);
            }
        }
    }
    ss << "]";
}

}  // anonymous namespace

// ============================================================================
// Core serialization implementation
// ============================================================================

void save(const Tensor& tensor, const std::string& filename,
          const SerializationOptions& options) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw SerializationError("Cannot open file for writing: " + filename);
  }

  save_stream(tensor, file, options);

  if (!file.good()) {
    throw SerializationError("Error writing to file: " + filename);
  }
}

void save_stream(const Tensor& tensor, std::ostream& stream,
                 const SerializationOptions& options) {
  // Ensure we have CPU data for serialization
  Tensor cpu_tensor = ensure_cpu_tensor(tensor);

  // Handle memory order options
  if (!options.preserve_order) {
    if (options.force_order != cpu_tensor.memory_order()) {
      if (options.force_order == MemoryOrder::RowMajor) {
        cpu_tensor = cpu_tensor.ascontiguousarray();
      } else {
        cpu_tensor = cpu_tensor.asfortranarray();
      }
    }
  }

  // Create and write header
  AxiomFileHeader header = create_header(cpu_tensor);
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

  // Write shape data
  const auto& shape = cpu_tensor.shape();
  for (size_t dim : shape) {
    uint64_t dim64 = static_cast<uint64_t>(dim);
    write_value(stream, dim64);
  }

  // Write tensor data
  if (cpu_tensor.size() > 0) {
    const void* data = cpu_tensor.data();
    stream.write(static_cast<const char*>(data), cpu_tensor.nbytes());
  }

  if (!stream.good()) {
    throw SerializationError("Failed to write tensor data to stream");
  }
}

Tensor load(const std::string& filename, Device device) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw SerializationError("Cannot open file for reading: " + filename);
  }

  return load_stream(file, device);
}

Tensor load_stream(std::istream& stream, Device device) {
  // Read and validate header
  AxiomFileHeader header;
  stream.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!stream.good()) {
    throw SerializationError("Failed to read file header");
  }

  validate_header(header);

  // Read shape data
  Shape shape(header.ndim);
  for (uint32_t i = 0; i < header.ndim; ++i) {
    uint64_t dim = read_value<uint64_t>(stream);
    shape[i] = static_cast<size_t>(dim);
  }

  // Convert header fields back to enums
  DType dtype = uint32_to_dtype(header.dtype);
  MemoryOrder memory_order = uint32_to_memory_order(header.memory_order);

  // Create tensor with appropriate memory order
  Tensor tensor(shape, dtype, Device::CPU, memory_order);

  // Read tensor data
  if (header.total_elements > 0) {
    void* data = tensor.data();
    stream.read(static_cast<char*>(data), header.data_size);
    if (!stream.good()) {
      throw SerializationError("Failed to read tensor data");
    }
  }

  // Transfer to target device if needed
  if (device != Device::CPU) {
    return tensor.to(device);
  }

  return tensor;
}

// ============================================================================
// Multi-tensor archive implementation
// ============================================================================

namespace {

struct ArchiveHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t num_tensors;
  uint64_t index_offset;
  uint8_t reserved[48];
};

struct TensorIndexEntry {
  char name[64];    // Tensor name (null-terminated)
  uint64_t offset;  // Offset to tensor data in file
  uint64_t size;    // Size of tensor data in bytes
  uint8_t reserved[16];
};

constexpr uint32_t ARCHIVE_MAGIC = 0x41584D41;  // "AXMA"

}  // anonymous namespace

void save_archive(const std::map<std::string, Tensor>& tensors,
                  const std::string& filename,
                  const SerializationOptions& options) {
  if (tensors.empty()) {
    throw SerializationError("Cannot save empty tensor archive");
  }

  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw SerializationError("Cannot open archive file for writing: " +
                             filename);
  }

  // Write archive header placeholder
  ArchiveHeader archive_header = {};
  archive_header.magic = ARCHIVE_MAGIC;
  archive_header.version = AXIOM_FILE_VERSION;
  archive_header.num_tensors = static_cast<uint32_t>(tensors.size());

  file.write(reinterpret_cast<const char*>(&archive_header),
             sizeof(archive_header));

  // Prepare index entries
  std::vector<TensorIndexEntry> index_entries(tensors.size());
  std::vector<std::streampos> tensor_positions;

  size_t entry_idx = 0;
  for (const auto& [name, tensor] : tensors) {
    if (name.length() >= 63) {
      throw SerializationError("Tensor name too long (max 63 chars): " + name);
    }

    // Record current position for tensor data
    tensor_positions.push_back(file.tellp());

    // Save tensor to stream
    save_stream(tensor, file, options);

    // Calculate size
    std::streampos end_pos = file.tellp();
    size_t tensor_size =
        static_cast<size_t>(end_pos - tensor_positions[entry_idx]);

    // Fill index entry
    std::memset(&index_entries[entry_idx], 0, sizeof(TensorIndexEntry));
    std::strncpy(index_entries[entry_idx].name, name.c_str(), 63);
    index_entries[entry_idx].offset =
        static_cast<uint64_t>(tensor_positions[entry_idx]);
    index_entries[entry_idx].size = tensor_size;

    entry_idx++;
  }

  // Record index offset and write it to header
  std::streampos index_offset = file.tellp();
  archive_header.index_offset = static_cast<uint64_t>(index_offset);

  // Write index
  for (const auto& entry : index_entries) {
    file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
  }

  // Update header with index offset
  file.seekp(0);
  file.write(reinterpret_cast<const char*>(&archive_header),
             sizeof(archive_header));

  if (!file.good()) {
    throw SerializationError("Error writing archive file: " + filename);
  }
}

std::map<std::string, Tensor> load_archive(const std::string& filename,
                                           Device device) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw SerializationError("Cannot open archive file for reading: " +
                             filename);
  }

  // Read archive header
  ArchiveHeader archive_header;
  file.read(reinterpret_cast<char*>(&archive_header), sizeof(archive_header));
  if (!file.good() || archive_header.magic != ARCHIVE_MAGIC) {
    throw FileFormatError("Invalid archive file format");
  }

  // Read index
  file.seekg(archive_header.index_offset);
  std::vector<TensorIndexEntry> index_entries(archive_header.num_tensors);
  for (auto& entry : index_entries) {
    file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    if (!file.good()) {
      throw SerializationError("Failed to read archive index");
    }
  }

  // Load all tensors
  std::map<std::string, Tensor> result;
  for (const auto& entry : index_entries) {
    file.seekg(entry.offset);
    std::string tensor_name(entry.name);
    Tensor tensor = load_stream(file, device);
    result[tensor_name] = std::move(tensor);
  }

  return result;
}

std::vector<std::string> list_archive(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw SerializationError("Cannot open archive file for reading: " +
                             filename);
  }

  // Read archive header
  ArchiveHeader archive_header;
  file.read(reinterpret_cast<char*>(&archive_header), sizeof(archive_header));
  if (!file.good() || archive_header.magic != ARCHIVE_MAGIC) {
    throw FileFormatError("Invalid archive file format");
  }

  // Read index
  file.seekg(archive_header.index_offset);
  std::vector<std::string> tensor_names;

  for (uint32_t i = 0; i < archive_header.num_tensors; ++i) {
    TensorIndexEntry entry;
    file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    if (!file.good()) {
      throw SerializationError("Failed to read archive index");
    }
    tensor_names.emplace_back(entry.name);
  }

  return tensor_names;
}

Tensor load_from_archive(const std::string& filename,
                         const std::string& tensor_name, Device device) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw SerializationError("Cannot open archive file for reading: " +
                             filename);
  }

  // Read archive header
  ArchiveHeader archive_header;
  file.read(reinterpret_cast<char*>(&archive_header), sizeof(archive_header));
  if (!file.good() || archive_header.magic != ARCHIVE_MAGIC) {
    throw FileFormatError("Invalid archive file format");
  }

  // Find tensor in index
  file.seekg(archive_header.index_offset);
  for (uint32_t i = 0; i < archive_header.num_tensors; ++i) {
    TensorIndexEntry entry;
    file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    if (!file.good()) {
      throw SerializationError("Failed to read archive index");
    }

    if (std::string(entry.name) == tensor_name) {
      file.seekg(entry.offset);
      return load_stream(file, device);
    }
  }

  throw SerializationError("Tensor not found in archive: " + tensor_name);
}

// ============================================================================
// Utility function implementations
// ============================================================================

AxiomFileHeader get_file_info(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw SerializationError("Cannot open file for reading: " + filename);
  }

  AxiomFileHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!file.good()) {
    throw SerializationError("Failed to read file header");
  }

  validate_header(header);
  return header;
}

bool is_axiom_file(const std::string& filename) {
  try {
    AxiomFileHeader header = get_file_info(filename);
    return header.magic == AXIOM_MAGIC_NUMBER;
  } catch (...) {
    return false;
  }
}

bool validate_tensor(const Tensor& tensor) {
  try {
    // Basic validation checks
    if (tensor.empty() && tensor.ndim() > 0) {
      return false;
    }

    if (tensor.size() != ShapeUtils::size(tensor.shape())) {
      return false;
    }

    if (tensor.nbytes() != tensor.size() * tensor.itemsize()) {
      return false;
    }

    return true;
  } catch (...) {
    return false;
  }
}

std::string version_string(uint32_t version) {
  switch (version) {
    case 1:
      return "1.0";
    default:
      return "Unknown (" + std::to_string(version) + ")";
  }
}

std::string to_string(const Tensor& tensor) {
    auto t_cpu = tensor.cpu();
    
    if (t_cpu.size() == 0) return "[]";
    if (t_cpu.ndim() == 0) return dispatch_element_to_string(t_cpu, 0);

    std::stringstream ss;
    std::vector<size_t> coords(t_cpu.ndim(), 0);
    print_recursive(ss, t_cpu, coords, 0, 3);
    
    return ss.str();
}

}  // namespace io
}  // namespace axiom