// #include "axiom/io/io.hpp"
// #include "axiom/io/numpy.hpp"

// #include <algorithm>
// #include <cstring>
// #include <fstream>
// #include <regex>
// #include <sstream>

// namespace axiom {
// namespace io {
// namespace numpy {

// namespace {

// // Check system endianness
// bool is_system_little_endian() {
//     uint16_t value = 1;
//     return *reinterpret_cast<uint8_t *>(&value) == 1;
// }

// // Byte swap for different sizes
// template <typename T> void byte_swap(T *data, size_t count) {
//     constexpr size_t size = sizeof(T);
//     if constexpr (size == 1) {
//         return; // No swap needed for single bytes
//     } else if constexpr (size == 2) {
//         for (size_t i = 0; i < count; ++i) {
//             uint16_t val = *reinterpret_cast<uint16_t *>(&data[i]);
//             val = ((val & 0xFF00) >> 8) | ((val & 0x00FF) << 8);
//             *reinterpret_cast<uint16_t *>(&data[i]) = val;
//         }
//     } else if constexpr (size == 4) {
//         for (size_t i = 0; i < count; ++i) {
//             uint32_t val = *reinterpret_cast<uint32_t *>(&data[i]);
//             val = ((val & 0xFF000000) >> 24) | ((val & 0x00FF0000) >> 8) |
//                   ((val & 0x0000FF00) << 8) | ((val & 0x000000FF) << 24);
//             *reinterpret_cast<uint32_t *>(&data[i]) = val;
//         }
//     } else if constexpr (size == 8) {
//         for (size_t i = 0; i < count; ++i) {
//             uint64_t val = *reinterpret_cast<uint64_t *>(&data[i]);
//             val = ((val & 0xFF00000000000000ULL) >> 56) |
//                   ((val & 0x00FF000000000000ULL) >> 40) |
//                   ((val & 0x0000FF0000000000ULL) >> 24) |
//                   ((val & 0x000000FF00000000ULL) >> 8) |
//                   ((val & 0x00000000FF000000ULL) << 8) |
//                   ((val & 0x0000000000FF0000ULL) << 24) |
//                   ((val & 0x000000000000FF00ULL) << 40) |
//                   ((val & 0x00000000000000FFULL) << 56);
//             *reinterpret_cast<uint64_t *>(&data[i]) = val;
//         }
//     } else if constexpr (size == 16) {
//         // Complex128: swap each 8-byte component
//         for (size_t i = 0; i < count; ++i) {
//             uint64_t *parts = reinterpret_cast<uint64_t *>(&data[i]);
//             for (int j = 0; j < 2; ++j) {
//                 uint64_t val = parts[j];
//                 val = ((val & 0xFF00000000000000ULL) >> 56) |
//                       ((val & 0x00FF000000000000ULL) >> 40) |
//                       ((val & 0x0000FF0000000000ULL) >> 24) |
//                       ((val & 0x000000FF00000000ULL) >> 8) |
//                       ((val & 0x00000000FF000000ULL) << 8) |
//                       ((val & 0x0000000000FF0000ULL) << 24) |
//                       ((val & 0x000000000000FF00ULL) << 40) |
//                       ((val & 0x00000000000000FFULL) << 56);
//                 parts[j] = val;
//             }
//         }
//     }
// }

// // Perform byte swap on raw data based on itemsize
// void byte_swap_data(void *data, size_t count, size_t itemsize) {
//     switch (itemsize) {
//     case 1:
//         break;
//     case 2:
//         byte_swap(static_cast<uint16_t *>(data), count);
//         break;
//     case 4:
//         byte_swap(static_cast<uint32_t *>(data), count);
//         break;
//     case 8:
//         byte_swap(static_cast<uint64_t *>(data), count);
//         break;
//     case 16:
//         // Complex128: 2 x 8-byte doubles
//         byte_swap(static_cast<uint64_t *>(data), count * 2);
//         break;
//     }
// }

// // Parse NumPy dtype string to Axiom DType
// // Returns {dtype, needs_byte_swap}
// struct DTypeInfo {
//     DType dtype;
//     bool needs_swap;
//     size_t itemsize;
// };

// DTypeInfo parse_numpy_dtype(const std::string &descr) {
//     if (descr.empty()) {
//         throw FileFormatError("Empty dtype descriptor");
//     }

//     // Parse endianness
//     char endian = descr[0];
//     bool is_little = false;
//     bool is_native = false;

//     std::string type_str;
//     if (endian == '<') {
//         is_little = true;
//         type_str = descr.substr(1);
//     } else if (endian == '>') {
//         is_little = false;
//         type_str = descr.substr(1);
//     } else if (endian == '|' || endian == '=') {
//         is_native = true;
//         type_str = descr.substr(1);
//     } else {
//         // No endian prefix (single char dtypes like 'b', '?')
//         type_str = descr;
//     }

//     bool system_little = is_system_little_endian();
//     bool needs_swap = !is_native && (is_little != system_little);

//     // Parse type character and size
//     if (type_str.empty()) {
//         throw FileFormatError("Invalid dtype descriptor: " + descr);
//     }

//     char type_char = type_str[0];
//     int type_size = 0;
//     if (type_str.size() > 1) {
//         type_size = std::stoi(type_str.substr(1));
//     }

//     DType dtype;
//     size_t itemsize;

//     switch (type_char) {
//     case 'b': // signed byte or boolean
//         // |b1 is boolean, plain 'b' without size is signed byte
//         if (is_native && type_size == 1) {
//             // |b1 is NumPy's boolean type
//             dtype = DType::Bool;
//             itemsize = 1;
//         } else {
//             dtype = DType::Int8;
//             itemsize = 1;
//         }
//         break;
//     case 'B': // unsigned byte
//         dtype = DType::UInt8;
//         itemsize = 1;
//         break;
//     case 'i': // signed integer
//         switch (type_size) {
//         case 1:
//             dtype = DType::Int8;
//             itemsize = 1;
//             break;
//         case 2:
//             dtype = DType::Int16;
//             itemsize = 2;
//             break;
//         case 4:
//             dtype = DType::Int32;
//             itemsize = 4;
//             break;
//         case 8:
//             dtype = DType::Int64;
//             itemsize = 8;
//             break;
//         default:
//             throw FileFormatError("Unsupported int size: " +
//                                   std::to_string(type_size));
//         }
//         break;
//     case 'u': // unsigned integer
//         switch (type_size) {
//         case 1:
//             dtype = DType::UInt8;
//             itemsize = 1;
//             break;
//         case 2:
//             dtype = DType::UInt16;
//             itemsize = 2;
//             break;
//         case 4:
//             dtype = DType::UInt32;
//             itemsize = 4;
//             break;
//         case 8:
//             dtype = DType::UInt64;
//             itemsize = 8;
//             break;
//         default:
//             throw FileFormatError("Unsupported uint size: " +
//                                   std::to_string(type_size));
//         }
//         break;
//     case 'f': // floating point
//         switch (type_size) {
//         case 2:
//             dtype = DType::Float16;
//             itemsize = 2;
//             break;
//         case 4:
//             dtype = DType::Float32;
//             itemsize = 4;
//             break;
//         case 8:
//             dtype = DType::Float64;
//             itemsize = 8;
//             break;
//         default:
//             throw FileFormatError("Unsupported float size: " +
//                                   std::to_string(type_size));
//         }
//         break;
//     case 'c': // complex
//         switch (type_size) {
//         case 8:
//             dtype = DType::Complex64;
//             itemsize = 8;
//             break;
//         case 16:
//             dtype = DType::Complex128;
//             itemsize = 16;
//             break;
//         default:
//             throw FileFormatError("Unsupported complex size: " +
//                                   std::to_string(type_size));
//         }
//         break;
//     case '?': // boolean
//         dtype = DType::Bool;
//         itemsize = 1;
//         break;
//     case 'e': // half float (IEEE 754 half precision)
//         dtype = DType::Float16;
//         itemsize = 2;
//         break;
//     case 'd': // double (alias)
//         dtype = DType::Float64;
//         itemsize = 8;
//         break;
//     case 'g': // long double - map to float64 (may lose precision)
//         dtype = DType::Float64;
//         itemsize = type_size > 0 ? type_size : 8;
//         break;
//     default:
//         throw FileFormatError("Unsupported dtype: " + descr);
//     }

//     return {dtype, needs_swap, itemsize};
// }

// // Parse Python tuple from string like "(3, 4, 5)"
// Shape parse_shape_tuple(const std::string &str) {
//     Shape shape;

//     // Remove parentheses and whitespace
//     std::string clean = str;
//     clean.erase(std::remove(clean.begin(), clean.end(), '('), clean.end());
//     clean.erase(std::remove(clean.begin(), clean.end(), ')'), clean.end());
//     clean.erase(std::remove(clean.begin(), clean.end(), ' '), clean.end());

//     if (clean.empty()) {
//         return shape; // Scalar (empty shape)
//     }

//     // Handle trailing comma for 1-element tuples like "(5,)"
//     if (!clean.empty() && clean.back() == ',') {
//         clean.pop_back();
//     }

//     // Split by comma
//     std::stringstream ss(clean);
//     std::string item;
//     while (std::getline(ss, item, ',')) {
//         if (!item.empty()) {
//             shape.push_back(static_cast<size_t>(std::stoull(item)));
//         }
//     }

//     return shape;
// }

// // Parse NumPy header dictionary
// struct NpyHeader {
//     std::string descr;
//     bool fortran_order;
//     Shape shape;
// };

// NpyHeader parse_header_dict(const std::string &header) {
//     NpyHeader result;
//     result.fortran_order = false;

//     // Extract 'descr' using regex
//     std::regex descr_regex(R"('descr'\s*:\s*'([^']+)')");
//     std::smatch descr_match;
//     if (std::regex_search(header, descr_match, descr_regex)) {
//         result.descr = descr_match[1].str();
//     } else {
//         throw FileFormatError("Missing 'descr' in NumPy header");
//     }

//     // Extract 'fortran_order'
//     std::regex fortran_regex(R"('fortran_order'\s*:\s*(True|False))");
//     std::smatch fortran_match;
//     if (std::regex_search(header, fortran_match, fortran_regex)) {
//         result.fortran_order = (fortran_match[1].str() == "True");
//     }

//     // Extract 'shape'
//     std::regex shape_regex(R"('shape'\s*:\s*\(([^)]*)\))");
//     std::smatch shape_match;
//     if (std::regex_search(header, shape_match, shape_regex)) {
//         result.shape = parse_shape_tuple("(" + shape_match[1].str() + ")");
//     } else {
//         throw FileFormatError("Missing 'shape' in NumPy header");
//     }

//     return result;
// }

// } // anonymous namespace

// bool is_npy_file(const std::string &filename) {
//     try {
//         std::ifstream file(filename, std::ios::binary);
//         if (!file.is_open()) {
//             return false;
//         }

//         char magic[NPY_MAGIC_SIZE];
//         if (!file.read(magic, NPY_MAGIC_SIZE)) {
//             return false;
//         }

//         return std::memcmp(magic, NPY_MAGIC, NPY_MAGIC_SIZE) == 0;
//     } catch (...) {
//         return false;
//     }
// }

// Tensor load(const std::string &filename, Device device) {
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         throw SerializationError("Cannot open file for reading: " +
//         filename);
//     }

//     // Read and verify magic
//     char magic[NPY_MAGIC_SIZE];
//     if (!file.read(magic, NPY_MAGIC_SIZE)) {
//         throw FileFormatError("Failed to read NumPy magic bytes");
//     }
//     if (std::memcmp(magic, NPY_MAGIC, NPY_MAGIC_SIZE) != 0) {
//         throw FileFormatError("Invalid NumPy magic bytes");
//     }

//     // Read version
//     uint8_t major_version, minor_version;
//     file.read(reinterpret_cast<char *>(&major_version), 1);
//     file.read(reinterpret_cast<char *>(&minor_version), 1);

//     if (major_version < 1 || major_version > 3) {
//         throw FileFormatError("Unsupported NumPy format version: " +
//                               std::to_string(major_version) + "." +
//                               std::to_string(minor_version));
//     }

//     // Read header length
//     uint32_t header_len;
//     if (major_version == 1) {
//         uint16_t len16;
//         file.read(reinterpret_cast<char *>(&len16), 2);
//         header_len = len16;
//     } else {
//         // Version 2.0 and 3.0 use 4-byte header length
//         file.read(reinterpret_cast<char *>(&header_len), 4);
//     }

//     // Read header dictionary
//     std::string header(header_len, '\0');
//     if (!file.read(&header[0], header_len)) {
//         throw FileFormatError("Failed to read NumPy header");
//     }

//     // Parse header
//     NpyHeader npy_header = parse_header_dict(header);

//     // Parse dtype
//     DTypeInfo dtype_info = parse_numpy_dtype(npy_header.descr);

//     // Determine memory order
//     MemoryOrder memory_order = npy_header.fortran_order ?
//     MemoryOrder::ColMajor
//                                                         :
//                                                         MemoryOrder::RowMajor;

//     // Create tensor
//     Tensor tensor(npy_header.shape, dtype_info.dtype, Device::CPU,
//                   memory_order);

//     // Read data
//     if (tensor.size() > 0) {
//         size_t data_size = tensor.nbytes();
//         if (!file.read(static_cast<char *>(tensor.data()), data_size)) {
//             throw FileFormatError("Failed to read NumPy data: expected " +
//                                   std::to_string(data_size) + " bytes");
//         }

//         // Byte swap if needed
//         if (dtype_info.needs_swap) {
//             byte_swap_data(tensor.data(), tensor.size(),
//             dtype_info.itemsize);
//         }
//     }

//     // Transfer to target device if needed
//     if (device != Device::CPU) {
//         return tensor.to(device);
//     }

//     return tensor;
// }

// } // namespace numpy
// } // namespace io
// } // namespace axiom
