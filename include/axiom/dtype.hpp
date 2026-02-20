#pragma once

#include <complex>
#include <cstdint>
#include <string>

#include "axiom/bfloat16.hpp"
#include "axiom/float16.hpp"

namespace axiom {

// float16_t is defined in axiom/float16.hpp
using complex64_t = std::complex<float>;
using complex128_t = std::complex<double>;

enum class DType : uint8_t {
    Bool,

    Int8,
    Int16,
    Int32,
    Int64,

    UInt8,
    UInt16,
    UInt32,
    UInt64,

    Float16,
    BFloat16,
    Float32,
    Float64,

    Complex64,
    Complex128
};

constexpr size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::Bool:
        return 1;
    case DType::Int8:
        return 1;
    case DType::Int16:
        return 2;
    case DType::Int32:
        return 4;
    case DType::Int64:
        return 8;
    case DType::UInt8:
        return 1;
    case DType::UInt16:
        return 2;
    case DType::UInt32:
        return 4;
    case DType::UInt64:
        return 8;
    case DType::Float16:
        return 2;
    case DType::BFloat16:
        return 2;
    case DType::Float32:
        return 4;
    case DType::Float64:
        return 8;
    case DType::Complex64:
        return 8;
    case DType::Complex128:
        return 16;
    }
    return 0;
}

std::string dtype_name(DType dtype);

template <typename T> struct dtype_of {
    static constexpr DType value = DType::Float32; // Default fallback
};

template <> struct dtype_of<bool> {
    static constexpr DType value = DType::Bool;
};
template <> struct dtype_of<int8_t> {
    static constexpr DType value = DType::Int8;
};
template <> struct dtype_of<int16_t> {
    static constexpr DType value = DType::Int16;
};
template <> struct dtype_of<int32_t> {
    static constexpr DType value = DType::Int32;
};
template <> struct dtype_of<int64_t> {
    static constexpr DType value = DType::Int64;
};
template <> struct dtype_of<uint8_t> {
    static constexpr DType value = DType::UInt8;
};
template <> struct dtype_of<uint16_t> {
    static constexpr DType value = DType::UInt16;
};
template <> struct dtype_of<uint32_t> {
    static constexpr DType value = DType::UInt32;
};
template <> struct dtype_of<uint64_t> {
    static constexpr DType value = DType::UInt64;
};
template <> struct dtype_of<float16_t> {
    static constexpr DType value = DType::Float16;
};
template <> struct dtype_of<bfloat16_t> {
    static constexpr DType value = DType::BFloat16;
};
template <> struct dtype_of<float> {
    static constexpr DType value = DType::Float32;
};
template <> struct dtype_of<double> {
    static constexpr DType value = DType::Float64;
};
template <> struct dtype_of<complex64_t> {
    static constexpr DType value = DType::Complex64;
};
template <> struct dtype_of<complex128_t> {
    static constexpr DType value = DType::Complex128;
};

template <typename T> constexpr DType dtype_of_v = dtype_of<T>::value;

// ============================================================================
// Type category queries
// ============================================================================

constexpr bool is_integer_dtype(DType dtype) {
    switch (dtype) {
    case DType::Bool:
    case DType::Int8:
    case DType::Int16:
    case DType::Int32:
    case DType::Int64:
    case DType::UInt8:
    case DType::UInt16:
    case DType::UInt32:
    case DType::UInt64:
        return true;
    default:
        return false;
    }
}

constexpr bool is_floating_dtype(DType dtype) {
    switch (dtype) {
    case DType::Float16:
    case DType::BFloat16:
    case DType::Float32:
    case DType::Float64:
        return true;
    default:
        return false;
    }
}

constexpr bool is_complex_dtype(DType dtype) {
    switch (dtype) {
    case DType::Complex64:
    case DType::Complex128:
        return true;
    default:
        return false;
    }
}

constexpr bool is_signed_integer_dtype(DType dtype) {
    switch (dtype) {
    case DType::Int8:
    case DType::Int16:
    case DType::Int32:
    case DType::Int64:
        return true;
    default:
        return false;
    }
}

constexpr bool is_unsigned_integer_dtype(DType dtype) {
    switch (dtype) {
    case DType::UInt8:
    case DType::UInt16:
    case DType::UInt32:
    case DType::UInt64:
        return true;
    default:
        return false;
    }
}

// ============================================================================
// Default values for each dtype
// ============================================================================

template <DType dtype> constexpr auto dtype_zero() {
    if constexpr (dtype == DType::Bool)
        return false;
    else if constexpr (dtype == DType::Int8)
        return int8_t(0);
    else if constexpr (dtype == DType::Int16)
        return int16_t(0);
    else if constexpr (dtype == DType::Int32)
        return int32_t(0);
    else if constexpr (dtype == DType::Int64)
        return int64_t(0);
    else if constexpr (dtype == DType::UInt8)
        return uint8_t(0);
    else if constexpr (dtype == DType::UInt16)
        return uint16_t(0);
    else if constexpr (dtype == DType::UInt32)
        return uint32_t(0);
    else if constexpr (dtype == DType::UInt64)
        return uint64_t(0);
    else if constexpr (dtype == DType::Float16)
        return float16_t(0.0f);
    else if constexpr (dtype == DType::BFloat16)
        return bfloat16_t(0.0f);
    else if constexpr (dtype == DType::Float32)
        return 0.0f;
    else if constexpr (dtype == DType::Float64)
        return 0.0;
    else if constexpr (dtype == DType::Complex64)
        return complex64_t(0.0f, 0.0f);
    else if constexpr (dtype == DType::Complex128)
        return complex128_t(0.0, 0.0);
}

template <DType dtype> constexpr auto dtype_one() {
    if constexpr (dtype == DType::Bool)
        return true;
    else if constexpr (dtype == DType::Int8)
        return int8_t(1);
    else if constexpr (dtype == DType::Int16)
        return int16_t(1);
    else if constexpr (dtype == DType::Int32)
        return int32_t(1);
    else if constexpr (dtype == DType::Int64)
        return int64_t(1);
    else if constexpr (dtype == DType::UInt8)
        return uint8_t(1);
    else if constexpr (dtype == DType::UInt16)
        return uint16_t(1);
    else if constexpr (dtype == DType::UInt32)
        return uint32_t(1);
    else if constexpr (dtype == DType::UInt64)
        return uint64_t(1);
    else if constexpr (dtype == DType::Float16)
        return float16_t(1.0f);
    else if constexpr (dtype == DType::BFloat16)
        return bfloat16_t(1.0f);
    else if constexpr (dtype == DType::Float32)
        return 1.0f;
    else if constexpr (dtype == DType::Float64)
        return 1.0;
    else if constexpr (dtype == DType::Complex64)
        return complex64_t(1.0f, 0.0f);
    else if constexpr (dtype == DType::Complex128)
        return complex128_t(1.0, 0.0);
}

} // namespace axiom