// #include "axiom/io/axiom_tensor_generated.h"
// #include "axiom/io/io.hpp"

// #include <cstring>
// #include <fstream>

// namespace axiom {
// namespace io {
// namespace flatbuffers {

// namespace {

// // Convert axiom::DType to FlatBuffers DType
// fb::DType to_fb_dtype(DType dtype) {
//     switch (dtype) {
//     case DType::Bool:
//         return fb::DType_Bool;
//     case DType::Int8:
//         return fb::DType_Int8;
//     case DType::Int16:
//         return fb::DType_Int16;
//     case DType::Int32:
//         return fb::DType_Int32;
//     case DType::Int64:
//         return fb::DType_Int64;
//     case DType::UInt8:
//         return fb::DType_UInt8;
//     case DType::UInt16:
//         return fb::DType_UInt16;
//     case DType::UInt32:
//         return fb::DType_UInt32;
//     case DType::UInt64:
//         return fb::DType_UInt64;
//     case DType::Float16:
//         return fb::DType_Float16;
//     case DType::Float32:
//         return fb::DType_Float32;
//     case DType::Float64:
//         return fb::DType_Float64;
//     case DType::Complex64:
//         return fb::DType_Complex64;
//     case DType::Complex128:
//         return fb::DType_Complex128;
//     }
//     return fb::DType_Float32;
// }

// // Convert FlatBuffers DType to axiom::DType
// DType from_fb_dtype(fb::DType dtype) {
//     switch (dtype) {
//     case fb::DType_Bool:
//         return DType::Bool;
//     case fb::DType_Int8:
//         return DType::Int8;
//     case fb::DType_Int16:
//         return DType::Int16;
//     case fb::DType_Int32:
//         return DType::Int32;
//     case fb::DType_Int64:
//         return DType::Int64;
//     case fb::DType_UInt8:
//         return DType::UInt8;
//     case fb::DType_UInt16:
//         return DType::UInt16;
//     case fb::DType_UInt32:
//         return DType::UInt32;
//     case fb::DType_UInt64:
//         return DType::UInt64;
//     case fb::DType_Float16:
//         return DType::Float16;
//     case fb::DType_Float32:
//         return DType::Float32;
//     case fb::DType_Float64:
//         return DType::Float64;
//     case fb::DType_Complex64:
//         return DType::Complex64;
//     case fb::DType_Complex128:
//         return DType::Complex128;
//     }
//     return DType::Float32;
// }

// // Convert axiom::MemoryOrder to FlatBuffers MemoryOrder
// fb::MemoryOrder to_fb_memory_order(MemoryOrder order) {
//     return order == MemoryOrder::RowMajor ? fb::MemoryOrder_RowMajor
//                                           : fb::MemoryOrder_ColMajor;
// }

// // Convert FlatBuffers MemoryOrder to axiom::MemoryOrder
// MemoryOrder from_fb_memory_order(fb::MemoryOrder order) {
//     return order == fb::MemoryOrder_RowMajor ? MemoryOrder::RowMajor
//                                              : MemoryOrder::ColMajor;
// }

// // Read entire file into buffer
// std::vector<uint8_t> read_file(const std::string &filename) {
//     std::ifstream file(filename, std::ios::binary | std::ios::ate);
//     if (!file.is_open()) {
//         throw SerializationError("Cannot open file for reading: " +
//         filename);
//     }

//     std::streamsize size = file.tellg();
//     file.seekg(0, std::ios::beg);

//     std::vector<uint8_t> buffer(size);
//     if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
//         throw SerializationError("Failed to read file: " + filename);
//     }

//     return buffer;
// }

// // Write buffer to file
// void write_file(const std::string &filename, const uint8_t *data, size_t
// size) {
//     std::ofstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         throw SerializationError("Cannot open file for writing: " +
//         filename);
//     }

//     file.write(reinterpret_cast<const char *>(data), size);
//     if (!file.good()) {
//         throw SerializationError("Failed to write file: " + filename);
//     }
// }

// // Build TensorData from Tensor
// ::flatbuffers::Offset<fb::TensorData>
// build_tensor_data(::flatbuffers::FlatBufferBuilder &builder,
//                   const Tensor &tensor, const std::string &name = "") {
//     // Ensure tensor is on CPU and contiguous (preserving memory order)
//     Tensor cpu_tensor = tensor.device() == Device::CPU ? tensor :
//     tensor.cpu(); if (!cpu_tensor.is_c_contiguous() &&
//     !cpu_tensor.is_f_contiguous()) {
//         // Non-contiguous tensor: preserve original memory order preference
//         if (cpu_tensor.memory_order() == MemoryOrder::ColMajor) {
//             cpu_tensor = cpu_tensor.asfortranarray();
//         } else {
//             cpu_tensor = cpu_tensor.ascontiguousarray();
//         }
//     }

//     // Create name
//     auto name_offset = name.empty() ? 0 : builder.CreateString(name);

//     // Create shape vector
//     std::vector<uint64_t> shape_vec;
//     for (size_t dim : cpu_tensor.shape()) {
//         shape_vec.push_back(static_cast<uint64_t>(dim));
//     }
//     auto shape_offset = builder.CreateVector(shape_vec);

//     // Create data vector
//     const uint8_t *data_ptr = static_cast<const uint8_t
//     *>(cpu_tensor.data()); auto data_offset = builder.CreateVector(data_ptr,
//     cpu_tensor.nbytes());

//     // Build TensorData
//     fb::TensorDataBuilder tensor_builder(builder);
//     if (!name.empty()) {
//         tensor_builder.add_name(name_offset);
//     }
//     tensor_builder.add_dtype(to_fb_dtype(cpu_tensor.dtype()));
//     tensor_builder.add_shape(shape_offset);
//     tensor_builder.add_memory_order(
//         to_fb_memory_order(cpu_tensor.memory_order()));
//     tensor_builder.add_data(data_offset);

//     return tensor_builder.Finish();
// }

// // Load Tensor from TensorData
// Tensor load_tensor_data(const fb::TensorData *tensor_data, Device device) {
//     if (!tensor_data) {
//         throw FileFormatError("Null tensor data");
//     }

//     // Extract dtype and memory order
//     DType dtype = from_fb_dtype(tensor_data->dtype());
//     MemoryOrder memory_order =
//         from_fb_memory_order(tensor_data->memory_order());

//     // Extract shape
//     Shape shape;
//     if (tensor_data->shape()) {
//         for (uint64_t dim : *tensor_data->shape()) {
//             shape.push_back(static_cast<size_t>(dim));
//         }
//     }

//     // Create tensor
//     Tensor tensor(shape, dtype, Device::CPU, memory_order);

//     // Copy data
//     if (tensor_data->data() && tensor.size() > 0) {
//         const uint8_t *src = tensor_data->data()->data();
//         size_t expected_size = tensor.nbytes();
//         size_t actual_size = tensor_data->data()->size();

//         if (actual_size != expected_size) {
//             throw FileFormatError("Data size mismatch: expected " +
//                                   std::to_string(expected_size) +
//                                   " bytes, got " +
//                                   std::to_string(actual_size));
//         }

//         std::memcpy(tensor.data(), src, expected_size);
//     }

//     // Transfer to target device if needed
//     if (device != Device::CPU) {
//         return tensor.to(device);
//     }

//     return tensor;
// }

// } // anonymous namespace

// bool is_axfb_file(const std::string &filename) {
//     try {
//         std::ifstream file(filename, std::ios::binary);
//         if (!file.is_open()) {
//             return false;
//         }

//         // Read first 8 bytes - FlatBuffers identifier is at offset 4
//         char header[8];
//         if (!file.read(header, 8)) {
//             return false;
//         }

//         // Check for "AXFB" identifier at offset 4
//         return header[4] == 'A' && header[5] == 'X' && header[6] == 'F' &&
//                header[7] == 'B';
//     } catch (...) {
//         return false;
//     }
// }

// Tensor load(const std::string &filename, Device device) {
//     auto buffer = read_file(filename);

//     // Verify buffer
//     ::flatbuffers::Verifier verifier(buffer.data(), buffer.size());
//     if (!fb::VerifyTensorArchiveBuffer(verifier)) {
//         throw FileFormatError("Invalid or corrupted FlatBuffers file");
//     }

//     // Get archive
//     const auto *archive = fb::GetTensorArchive(buffer.data());
//     if (!archive || !archive->tensors() || archive->tensors()->size() == 0) {
//         throw FileFormatError("Empty tensor archive");
//     }

//     // Load first tensor
//     return load_tensor_data(archive->tensors()->Get(0), device);
// }

// std::map<std::string, Tensor> load_archive(const std::string &filename,
//                                            Device device) {
//     auto buffer = read_file(filename);

//     // Verify buffer
//     ::flatbuffers::Verifier verifier(buffer.data(), buffer.size());
//     if (!fb::VerifyTensorArchiveBuffer(verifier)) {
//         throw FileFormatError("Invalid or corrupted FlatBuffers file");
//     }

//     // Get archive
//     const auto *archive = fb::GetTensorArchive(buffer.data());
//     if (!archive || !archive->tensors()) {
//         throw FileFormatError("Invalid tensor archive");
//     }

//     std::map<std::string, Tensor> result;
//     for (const auto *tensor_data : *archive->tensors()) {
//         std::string name =
//             tensor_data->name() ? tensor_data->name()->str() : "";

//         // Use index-based name if no name provided
//         if (name.empty()) {
//             name = "tensor_" + std::to_string(result.size());
//         }

//         result[name] = load_tensor_data(tensor_data, device);
//     }

//     return result;
// }

// void save(const Tensor &tensor, const std::string &filename) {
//     ::flatbuffers::FlatBufferBuilder builder(1024 + tensor.nbytes());

//     // Build single tensor
//     auto tensor_offset = build_tensor_data(builder, tensor, "tensor");

//     // Create tensors vector
//     std::vector<::flatbuffers::Offset<fb::TensorData>> tensors_vec;
//     tensors_vec.push_back(tensor_offset);
//     auto tensors_offset = builder.CreateVector(tensors_vec);

//     // Build archive
//     auto archive = fb::CreateTensorArchive(builder, 2, tensors_offset);
//     fb::FinishTensorArchiveBuffer(builder, archive);

//     // Write to file
//     write_file(filename, builder.GetBufferPointer(), builder.GetSize());
// }

// void save_archive(const std::map<std::string, Tensor> &tensors,
//                   const std::string &filename) {
//     if (tensors.empty()) {
//         throw SerializationError("Cannot save empty tensor archive");
//     }

//     // Calculate approximate buffer size
//     size_t total_bytes = 1024;
//     for (const auto &[name, tensor] : tensors) {
//         total_bytes += tensor.nbytes() + name.size() + 256;
//     }

//     ::flatbuffers::FlatBufferBuilder builder(total_bytes);

//     // Build all tensors
//     std::vector<::flatbuffers::Offset<fb::TensorData>> tensors_vec;
//     for (const auto &[name, tensor] : tensors) {
//         tensors_vec.push_back(build_tensor_data(builder, tensor, name));
//     }
//     auto tensors_offset = builder.CreateVector(tensors_vec);

//     // Build archive
//     auto archive = fb::CreateTensorArchive(builder, 2, tensors_offset);
//     fb::FinishTensorArchiveBuffer(builder, archive);

//     // Write to file
//     write_file(filename, builder.GetBufferPointer(), builder.GetSize());
// }

// std::vector<std::string> list_archive(const std::string &filename) {
//     auto buffer = read_file(filename);

//     // Verify buffer
//     ::flatbuffers::Verifier verifier(buffer.data(), buffer.size());
//     if (!fb::VerifyTensorArchiveBuffer(verifier)) {
//         throw FileFormatError("Invalid or corrupted FlatBuffers file");
//     }

//     // Get archive
//     const auto *archive = fb::GetTensorArchive(buffer.data());
//     if (!archive || !archive->tensors()) {
//         throw FileFormatError("Invalid tensor archive");
//     }

//     std::vector<std::string> names;
//     size_t index = 0;
//     for (const auto *tensor_data : *archive->tensors()) {
//         if (tensor_data->name()) {
//             names.push_back(tensor_data->name()->str());
//         } else {
//             names.push_back("tensor_" + std::to_string(index));
//         }
//         index++;
//     }

//     return names;
// }

// Tensor load_from_archive(const std::string &filename,
//                          const std::string &tensor_name, Device device) {
//     auto buffer = read_file(filename);

//     // Verify buffer
//     ::flatbuffers::Verifier verifier(buffer.data(), buffer.size());
//     if (!fb::VerifyTensorArchiveBuffer(verifier)) {
//         throw FileFormatError("Invalid or corrupted FlatBuffers file");
//     }

//     // Get archive
//     const auto *archive = fb::GetTensorArchive(buffer.data());
//     if (!archive || !archive->tensors()) {
//         throw FileFormatError("Invalid tensor archive");
//     }

//     // Find tensor by name
//     size_t index = 0;
//     for (const auto *tensor_data : *archive->tensors()) {
//         std::string name =
//             tensor_data->name() ? tensor_data->name()->str() : "";

//         // Check by name or by index-based name
//         if (name == tensor_name ||
//             (name.empty() &&
//              tensor_name == "tensor_" + std::to_string(index))) {
//             return load_tensor_data(tensor_data, device);
//         }
//         index++;
//     }

//     throw SerializationError("Tensor not found in archive: " + tensor_name);
// }

// } // namespace flatbuffers
// } // namespace io
// } // namespace axiom
