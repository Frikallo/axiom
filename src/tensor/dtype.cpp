#include "axiom/dtype.hpp"

namespace axiom {

std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Bool:       return "bool";
        case DType::Int8:       return "int8";
        case DType::Int16:      return "int16";
        case DType::Int32:      return "int32";
        case DType::Int64:      return "int64";
        case DType::UInt8:      return "uint8";
        case DType::UInt16:     return "uint16";
        case DType::UInt32:     return "uint32";
        case DType::UInt64:     return "uint64";
        case DType::Float16:    return "float16";
        case DType::Float32:    return "float32";
        case DType::Float64:    return "float64";
        case DType::Complex64:  return "complex64";
        case DType::Complex128: return "complex128";
    }
    return "unknown";
}

} // namespace axiom