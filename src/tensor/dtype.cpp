#include "axiom/dtype.hpp"

namespace axiom {

Bool::value_type Bool::one() { return true; }
Bool::value_type Bool::zeros() { return false; }

Int8::value_type Int8::one() { return 1; }
Int8::value_type Int8::zeros() { return 0; }

Int16::value_type Int16::one() { return 1; }
Int16::value_type Int16::zeros() { return 0; }

Int32::value_type Int32::one() { return 1; }
Int32::value_type Int32::zeros() { return 0; }

Int64::value_type Int64::one() { return 1; }
Int64::value_type Int64::zeros() { return 0; }

UInt8::value_type UInt8::one() { return 1; }
UInt8::value_type UInt8::zeros() { return 0; }

UInt16::value_type UInt16::one() { return 1; }
UInt16::value_type UInt16::zeros() { return 0; }

UInt32::value_type UInt32::one() { return 1; }
UInt32::value_type UInt32::zeros() { return 0; }

UInt64::value_type UInt64::one() { return 1; }
UInt64::value_type UInt64::zeros() { return 0; }

Float16::value_type Float16::one() { return float16_t{1.0f}; }
Float16::value_type Float16::zeros() { return float16_t{0.0f}; }

BFloat16::value_type BFloat16::one() { return bfloat16_t{1.0f}; }
BFloat16::value_type BFloat16::zeros() { return bfloat16_t{0.0f}; }

Float32::value_type Float32::one() { return 1.0f; }
Float32::value_type Float32::zeros() { return 0.0f; }

Float64::value_type Float64::one() { return 1.0; }
Float64::value_type Float64::zeros() { return 0.0; }

Complex64::value_type Complex64::one() {
    return std::complex<float>{1.0f, 0.0f};
}
Complex64::value_type Complex64::zeros() {
    return std::complex<float>{0.0f, 0.0f};
}

Complex128::value_type Complex128::one() {
    return std::complex<double>{1.0, 0.0};
}
Complex128::value_type Complex128::zeros() {
    return std::complex<double>{0.0, 0.0};
}

std::string dtype_name(DType dtype) {
    switch (dtype) {
    case DType::Bool:
        return "bool";
    case DType::Int8:
        return "int8";
    case DType::Int16:
        return "int16";
    case DType::Int32:
        return "int32";
    case DType::Int64:
        return "int64";
    case DType::UInt8:
        return "uint8";
    case DType::UInt16:
        return "uint16";
    case DType::UInt32:
        return "uint32";
    case DType::UInt64:
        return "uint64";
    case DType::Float16:
        return "float16";
    case DType::BFloat16:
        return "bfloat16";
    case DType::Float32:
        return "float32";
    case DType::Float64:
        return "float64";
    case DType::Complex64:
        return "complex64";
    case DType::Complex128:
        return "complex128";
    }
    return "unknown";
}

TypeVariant variant_to_dtype(DType dtype) {
    switch (dtype) {
    case DType::Bool:
        return Bool();
    case DType::Int8:
        return Int8();
    case DType::Int16:
        return Int16();
    case DType::Int32:
        return Int32();
    case DType::Int64:
        return Int64();
    case DType::UInt8:
        return UInt8();
    case DType::UInt16:
        return UInt16();
    case DType::UInt32:
        return UInt32();
    case DType::UInt64:
        return UInt64();
    case DType::Float16:
        return Float16();
    case DType::BFloat16:
        return BFloat16();
    case DType::Float32:
        return Float32();
    case DType::Float64:
        return Float64();
    case DType::Complex64:
        return Complex64();
    case DType::Complex128:
        return Complex128();
    }
    return Bool();
}

} // namespace axiom
