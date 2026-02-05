#include "axiom/io/io.hpp"
#include "axiom/io/numpy.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "axiom/error.hpp"
#include "axiom/numeric.hpp"

namespace axiom {
namespace io {

// ============================================================================
// Format Detection
// ============================================================================

FileFormat detect_format(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return FileFormat::Unknown;
    }

    // Read first 8 bytes for format detection
    char header[8];
    if (!file.read(header, 8)) {
        return FileFormat::Unknown;
    }

    // Check for NumPy magic: \x93NUMPY
    if (header[0] == '\x93' && header[1] == 'N' && header[2] == 'U' &&
        header[3] == 'M' && header[4] == 'P' && header[5] == 'Y') {
        return FileFormat::NumPy;
    }

    // Check for FlatBuffers AXFB identifier at offset 4
    if (header[4] == 'A' && header[5] == 'X' && header[6] == 'F' &&
        header[7] == 'B') {
        return FileFormat::Axiom;
    }

    return FileFormat::Unknown;
}

std::string format_name(FileFormat format) {
    switch (format) {
    case FileFormat::Axiom:
        return "Axiom FlatBuffers";
    case FileFormat::NumPy:
        return "NumPy";
    case FileFormat::Unknown:
    default:
        return "Unknown";
    }
}

// ============================================================================
// Universal Load Functions
// ============================================================================

Tensor load(const std::string &filename, Device device) {
    FileFormat format = detect_format(filename);

    switch (format) {
    case FileFormat::Axiom:
        return flatbuffers::load(filename, device);
    case FileFormat::NumPy:
        return numpy::load(filename, device);
    case FileFormat::Unknown:
    default:
        throw FileFormatError("Unknown file format. Supported formats: .axfb "
                              "(FlatBuffers), .npy (NumPy)");
    }
}

std::map<std::string, Tensor> load_archive(const std::string &filename,
                                           Device device) {
    FileFormat format = detect_format(filename);

    switch (format) {
    case FileFormat::Axiom:
        return flatbuffers::load_archive(filename, device);
    case FileFormat::NumPy: {
        // NumPy single file: wrap in map with filename as key
        std::map<std::string, Tensor> result;
        // Extract base filename without path
        size_t pos = filename.find_last_of("/\\");
        std::string name =
            (pos == std::string::npos) ? filename : filename.substr(pos + 1);
        // Remove .npy extension
        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
            name = name.substr(0, name.size() - 4);
        }
        result[name] = numpy::load(filename, device);
        return result;
    }
    case FileFormat::Unknown:
    default:
        throw FileFormatError("Unknown file format. Supported formats: .axfb "
                              "(FlatBuffers), .npy (NumPy)");
    }
}

// ============================================================================
// Save Functions (delegate to flatbuffers namespace)
// ============================================================================

void save(const Tensor &tensor, const std::string &filename) {
    flatbuffers::save(tensor, filename);
}

void save_archive(const std::map<std::string, Tensor> &tensors,
                  const std::string &filename) {
    flatbuffers::save_archive(tensors, filename);
}

// ============================================================================
// String Formatting (preserved from original implementation)
// ============================================================================

namespace {

// Helper to convert a single element to string based on dtype
template <typename T>
std::string element_to_string(const void *data, size_t index) {
    T value = static_cast<const T *>(data)[index];

    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(value)) {
            return "nan";
        }
        if (std::isinf(value)) {
            return value > 0 ? "inf" : "-inf";
        }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4) << value;
        return ss.str();
    } else if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
}

// Special handling for half-precision since it's not a native C++ type
std::string half_element_to_string(const void *data, size_t index) {
    float value =
        static_cast<float>(static_cast<const float16_t *>(data)[index]);
    if (std::isnan(value))
        return "nan";
    if (std::isinf(value))
        return value > 0 ? "inf" : "-inf";
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << value;
    return ss.str();
}

std::string dispatch_element_to_string(const Tensor &t, size_t index) {
    const void *data = t.data();
    return std::visit(
        [&]<typename T>(T &&) {
            return element_to_string<typename std::decay_t<T>::value_type>(
                data, index);
        },
        t.dtype());
}

void print_recursive(std::stringstream &ss, const Tensor &t,
                     std::vector<size_t> &coords, size_t dim,
                     size_t edge_items) {
    ss << "[";
    size_t dim_size = t.shape()[dim];

    if (dim == t.ndim() - 1) {
        if (dim_size > 2 * edge_items) {
            for (size_t i = 0; i < edge_items; ++i) {
                coords[dim] = i;
                size_t offset = ShapeUtils::linear_index(coords, t.strides()) /
                                t.itemsize();
                ss << dispatch_element_to_string(t, offset) << " ";
            }
            ss << "... ";
            for (size_t i = dim_size - edge_items; i < dim_size; ++i) {
                coords[dim] = i;
                size_t offset = ShapeUtils::linear_index(coords, t.strides()) /
                                t.itemsize();
                ss << dispatch_element_to_string(t, offset);
                if (i < dim_size - 1)
                    ss << " ";
            }
        } else {
            for (size_t i = 0; i < dim_size; ++i) {
                coords[dim] = i;
                size_t offset = ShapeUtils::linear_index(coords, t.strides()) /
                                t.itemsize();
                ss << dispatch_element_to_string(t, offset);
                if (i < dim_size - 1)
                    ss << " ";
            }
        }
    } else {
        if (dim_size > 2 * edge_items) {
            for (size_t i = 0; i < edge_items; ++i) {
                if (i > 0) {
                    ss << "\n";
                    for (size_t j = 0; j <= dim; ++j)
                        ss << " ";
                }
                coords[dim] = i;
                print_recursive(ss, t, coords, dim + 1, edge_items);
            }
            ss << "\n";
            for (size_t j = 0; j <= dim; ++j)
                ss << " ";
            ss << "...";
            for (size_t i = dim_size - edge_items; i < dim_size; ++i) {
                ss << "\n";
                for (size_t j = 0; j <= dim; ++j)
                    ss << " ";
                coords[dim] = i;
                print_recursive(ss, t, coords, dim + 1, edge_items);
            }
        } else {
            for (size_t i = 0; i < dim_size; ++i) {
                if (i > 0) {
                    ss << "\n";
                    for (size_t j = 0; j <= dim; ++j) {
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

} // anonymous namespace

std::string to_string(const Tensor &tensor) {
    auto t_cpu = tensor.cpu();

    if (t_cpu.size() == 0)
        return "[]";
    if (t_cpu.ndim() == 0)
        return dispatch_element_to_string(t_cpu, 0);

    std::stringstream ss;
    std::vector<size_t> coords(t_cpu.ndim(), 0);
    print_recursive(ss, t_cpu, coords, 0, 3);

    ss << " " << t_cpu.repr();

    return ss.str();
}

} // namespace io
} // namespace axiom
