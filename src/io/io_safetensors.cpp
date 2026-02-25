#include "axiom/io/io.hpp"
#include "axiom/io/safetensors.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "nlohmann/json.hpp"

namespace axiom {
namespace io {
namespace safetensors {

namespace {

DType parse_dtype(const std::string &dtype_str) {
    if (dtype_str == "F16")
        return DType::Float16;
    if (dtype_str == "BF16")
        return DType::BFloat16;
    if (dtype_str == "F32")
        return DType::Float32;
    if (dtype_str == "F64")
        return DType::Float64;
    if (dtype_str == "I8")
        return DType::Int8;
    if (dtype_str == "I16")
        return DType::Int16;
    if (dtype_str == "I32")
        return DType::Int32;
    if (dtype_str == "I64")
        return DType::Int64;
    if (dtype_str == "U8")
        return DType::UInt8;
    if (dtype_str == "U16")
        return DType::UInt16;
    if (dtype_str == "U32")
        return DType::UInt32;
    if (dtype_str == "U64")
        return DType::UInt64;
    if (dtype_str == "BOOL")
        return DType::Bool;
    throw FileFormatError("SafeTensors: unsupported dtype: " + dtype_str);
}

struct TensorMetadata {
    std::string name;
    DType dtype;
    Shape shape;
    size_t data_start;
    size_t data_end;
};

// Parse the header and return metadata for all tensors
std::pair<std::vector<TensorMetadata>, size_t>
parse_header(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw SerializationError("SafeTensors: cannot open file: " + filename);
    }

    // Read 8-byte LE uint64 header length
    uint64_t header_length = 0;
    file.read(reinterpret_cast<char *>(&header_length), 8);
    if (!file) {
        throw FileFormatError("SafeTensors: failed to read header length");
    }

    // Sanity check header size (max 100MB to prevent OOM)
    if (header_length > 100 * 1024 * 1024) {
        throw FileFormatError("SafeTensors: header too large (" +
                              std::to_string(header_length) + " bytes)");
    }

    // Read JSON header
    std::string header_str(header_length, '\0');
    file.read(header_str.data(), static_cast<std::streamsize>(header_length));
    if (!file) {
        throw FileFormatError("SafeTensors: failed to read header JSON");
    }

    size_t data_offset = 8 + header_length;

    nlohmann::json header = nlohmann::json::parse(header_str);

    std::vector<TensorMetadata> tensors;
    for (auto &[key, value] : header.items()) {
        // Skip __metadata__ key
        if (key == "__metadata__") {
            continue;
        }

        TensorMetadata meta;
        meta.name = key;
        meta.dtype = parse_dtype(value["dtype"].get<std::string>());

        auto &shape_arr = value["shape"];
        for (auto &dim : shape_arr) {
            meta.shape.push_back(dim.get<size_t>());
        }

        auto &offsets = value["data_offsets"];
        meta.data_start = offsets[0].get<size_t>();
        meta.data_end = offsets[1].get<size_t>();

        tensors.push_back(std::move(meta));
    }

    return {tensors, data_offset};
}

} // namespace

bool is_safetensors_file(const std::string &filename) {
    // Check file extension
    const std::string ext = ".safetensors";
    if (filename.size() < ext.size()) {
        return false;
    }
    return filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0;
}

std::map<std::string, Tensor> load(const std::string &filename, Device device) {
    auto [tensor_metas, data_offset] = parse_header(filename);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw SerializationError("SafeTensors: cannot open file: " + filename);
    }

    std::map<std::string, Tensor> result;

    for (const auto &meta : tensor_metas) {
        size_t byte_count = meta.data_end - meta.data_start;

        // Create tensor and read raw data directly into it
        Tensor tensor(meta.shape, meta.dtype, Device::CPU);

        // Verify expected size matches
        size_t expected_bytes = tensor.nbytes();
        if (byte_count != expected_bytes) {
            throw FileFormatError(
                "SafeTensors: tensor '" + meta.name +
                "' size mismatch: file=" + std::to_string(byte_count) +
                " expected=" + std::to_string(expected_bytes));
        }

        file.seekg(static_cast<std::streamoff>(data_offset + meta.data_start));
        file.read(static_cast<char *>(tensor.data()),
                  static_cast<std::streamsize>(byte_count));
        if (!file) {
            throw SerializationError("SafeTensors: failed to read tensor '" +
                                     meta.name + "' data");
        }

        result[meta.name] = (device == Device::GPU) ? tensor.gpu() : tensor;
    }

    return result;
}

std::vector<std::string> list_tensors(const std::string &filename) {
    auto [tensor_metas, data_offset] = parse_header(filename);
    (void)data_offset;

    std::vector<std::string> names;
    names.reserve(tensor_metas.size());
    for (const auto &meta : tensor_metas) {
        names.push_back(meta.name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

Tensor load_tensor(const std::string &filename, const std::string &tensor_name,
                   Device device) {
    auto [tensor_metas, data_offset] = parse_header(filename);

    const TensorMetadata *found = nullptr;
    for (const auto &meta : tensor_metas) {
        if (meta.name == tensor_name) {
            found = &meta;
            break;
        }
    }
    if (!found) {
        throw SerializationError("SafeTensors: tensor '" + tensor_name +
                                 "' not found in " + filename);
    }

    size_t byte_count = found->data_end - found->data_start;

    Tensor tensor(found->shape, found->dtype, Device::CPU);
    size_t expected_bytes = tensor.nbytes();
    if (byte_count != expected_bytes) {
        throw FileFormatError("SafeTensors: tensor '" + tensor_name +
                              "' size mismatch");
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw SerializationError("SafeTensors: cannot open file: " + filename);
    }

    file.seekg(static_cast<std::streamoff>(data_offset + found->data_start));
    file.read(static_cast<char *>(tensor.data()),
              static_cast<std::streamsize>(byte_count));
    if (!file) {
        throw SerializationError("SafeTensors: failed to read tensor data");
    }

    return (device == Device::GPU) ? tensor.gpu() : tensor;
}

} // namespace safetensors
} // namespace io
} // namespace axiom
