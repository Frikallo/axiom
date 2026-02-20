#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <tuple>
#include <variant>

#include "axiom/bfloat16.hpp"
#include "axiom/float16.hpp"

namespace axiom {

template <class T> class BaseType {
  public:
    using value_type = T;

    template <typename E> struct is_complex_t : public std::false_type {};

    template <typename E>
    struct is_complex_t<std::complex<E>> : public std::true_type {};

    static constexpr size_t dtype_size() { return sizeof(T); }
    static constexpr bool is_complex() {
        return is_complex_t<value_type>::value;
    }
    static constexpr bool is_float() {
        // capture 16 bit floats
        return std::is_floating_point_v<value_type> ||
               std::is_same_v<float16_t, value_type> ||
               std::is_same_v<bfloat16_t, value_type>;
    }
    static constexpr bool is_pod_float() {
        return std::is_floating_point_v<value_type>;
    }
    static constexpr bool is_int() { return std::is_integral_v<value_type>; }
    static constexpr bool is_unsigned() {
        return std::is_unsigned_v<value_type>;
    }
    static constexpr bool is_signed() { return std::is_signed_v<value_type>; }
};

class Bool : public BaseType<bool> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Bool"; }
};

class Int8 : public BaseType<int8_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Int8"; }
};

class Int16 : public BaseType<int16_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Int16"; }
};

class Int32 : public BaseType<int32_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Int32"; }
};

class Int64 : public BaseType<int64_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Int64"; }
};

class UInt8 : public BaseType<uint8_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "UInt8"; }
};

class UInt16 : public BaseType<uint16_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "UInt16"; }
};

class UInt32 : public BaseType<uint32_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "UInt32"; }
};

class UInt64 : public BaseType<uint64_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "UInt64"; }
};

class Float16 : public BaseType<float16_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Float16"; }
};

class BFloat16 : public BaseType<bfloat16_t> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "BFloat16"; }
};

class Float32 : public BaseType<float> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Float32"; }
};

class Float64 : public BaseType<double> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Float64"; }
};

class Complex64 : public BaseType<std::complex<float>> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Complex64"; }
};

class Complex128 : public BaseType<std::complex<double>> {
  public:
    using value_type = typename BaseType::value_type;
    static value_type one();
    static value_type zeros();
    static std::string name() { return "Complex128"; }
};

using TypeVariant = ::std::variant<Bool, Int8, Int16, Int32, Int64, UInt8,
                                   UInt16, UInt32, UInt64, Float16, BFloat16,
                                   Float32, Float64, Complex64, Complex128>;

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
