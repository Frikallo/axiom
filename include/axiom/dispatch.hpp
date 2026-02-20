#pragma once

#include <string>
#include <variant>

#include "axiom/dtype.hpp"
#include "axiom/error.hpp"

namespace axiom {

// Category-restricted variant types
using FloatVariant = std::variant<Float16, BFloat16, Float32, Float64>;
using IntVariant = std::variant<Bool, Int8, Int16, Int32, Int64, UInt8, UInt16,
                                UInt32, UInt64>;
using NumericVariant =
    std::variant<Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                 Float16, BFloat16, Float32, Float64>;
using ComplexVariant = std::variant<Complex64, Complex128>;

namespace detail {

inline TypeVariant dtype_to_variant(DType dtype) {
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

inline FloatVariant dtype_to_float_variant(DType dtype) {
    switch (dtype) {
    case DType::Float16:
        return Float16();
    case DType::BFloat16:
        return BFloat16();
    case DType::Float32:
        return Float32();
    case DType::Float64:
        return Float64();
    default:
        return Float32();
    }
}

inline IntVariant dtype_to_int_variant(DType dtype) {
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
    default:
        return Int32();
    }
}

inline NumericVariant dtype_to_numeric_variant(DType dtype) {
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
    default:
        return Float32();
    }
}

inline ComplexVariant dtype_to_complex_variant(DType dtype) {
    switch (dtype) {
    case DType::Complex64:
        return Complex64();
    case DType::Complex128:
        return Complex128();
    default:
        return Complex64();
    }
}

} // namespace detail

// Universal dispatch — invokes fn with the type class matching dtype
template <typename Fn> decltype(auto) dispatch(DType dtype, Fn &&fn) {
    return std::visit(std::forward<Fn>(fn), detail::dtype_to_variant(dtype));
}

// Float-only dispatch — throws TypeError for non-float dtypes
template <typename Fn>
decltype(auto) dispatch_float(DType dtype, const std::string &op, Fn &&fn) {
    if (!is_floating_dtype(dtype))
        throw TypeError::unsupported_dtype(dtype_name(dtype), op);
    return std::visit(std::forward<Fn>(fn),
                      detail::dtype_to_float_variant(dtype));
}

// Integer-only dispatch — throws TypeError for non-integer dtypes
template <typename Fn>
decltype(auto) dispatch_int(DType dtype, const std::string &op, Fn &&fn) {
    if (!is_integer_dtype(dtype))
        throw TypeError::unsupported_dtype(dtype_name(dtype), op);
    return std::visit(std::forward<Fn>(fn),
                      detail::dtype_to_int_variant(dtype));
}

// Numeric dispatch (int + float) — throws TypeError for complex dtypes
template <typename Fn>
decltype(auto) dispatch_numeric(DType dtype, const std::string &op, Fn &&fn) {
    if (is_complex_dtype(dtype))
        throw TypeError::unsupported_dtype(dtype_name(dtype), op);
    return std::visit(std::forward<Fn>(fn),
                      detail::dtype_to_numeric_variant(dtype));
}

// Complex-only dispatch — throws TypeError for non-complex dtypes
template <typename Fn>
decltype(auto) dispatch_complex(DType dtype, const std::string &op, Fn &&fn) {
    if (!is_complex_dtype(dtype))
        throw TypeError::unsupported_dtype(dtype_name(dtype), op);
    return std::visit(std::forward<Fn>(fn),
                      detail::dtype_to_complex_variant(dtype));
}

} // namespace axiom
