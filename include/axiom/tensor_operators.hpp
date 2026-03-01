#pragma once

#include <type_traits>

#include "operations.hpp"
#include "tensor.hpp"

namespace axiom {

// ============================================================================
// Arithmetic operators
// ============================================================================

inline Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
    return ops::add(lhs, rhs);
}

inline Tensor operator-(const Tensor &lhs, const Tensor &rhs) {
    return ops::subtract(lhs, rhs);
}

inline Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
    return ops::multiply(lhs, rhs);
}

inline Tensor operator/(const Tensor &lhs, const Tensor &rhs) {
    return ops::divide(lhs, rhs);
}

inline Tensor operator%(const Tensor &lhs, const Tensor &rhs) {
    return ops::modulo(lhs, rhs);
}

// ============================================================================
// Comparison operators
// ============================================================================

inline Tensor operator==(const Tensor &lhs, const Tensor &rhs) {
    return ops::equal(lhs, rhs);
}

inline Tensor operator!=(const Tensor &lhs, const Tensor &rhs) {
    return ops::not_equal(lhs, rhs);
}

inline Tensor operator<(const Tensor &lhs, const Tensor &rhs) {
    return ops::less(lhs, rhs);
}

inline Tensor operator<=(const Tensor &lhs, const Tensor &rhs) {
    return ops::less_equal(lhs, rhs);
}

inline Tensor operator>(const Tensor &lhs, const Tensor &rhs) {
    return ops::greater(lhs, rhs);
}

inline Tensor operator>=(const Tensor &lhs, const Tensor &rhs) {
    return ops::greater_equal(lhs, rhs);
}

// ============================================================================
// Logical operators
// ============================================================================

inline Tensor operator&&(const Tensor &lhs, const Tensor &rhs) {
    return ops::logical_and(lhs, rhs);
}

inline Tensor operator||(const Tensor &lhs, const Tensor &rhs) {
    return ops::logical_or(lhs, rhs);
}

// ============================================================================
// Bitwise operators
// ============================================================================

inline Tensor operator&(const Tensor &lhs, const Tensor &rhs) {
    return ops::bitwise_and(lhs, rhs);
}

inline Tensor operator|(const Tensor &lhs, const Tensor &rhs) {
    return ops::bitwise_or(lhs, rhs);
}

inline Tensor operator^(const Tensor &lhs, const Tensor &rhs) {
    return ops::bitwise_xor(lhs, rhs);
}

inline Tensor operator<<(const Tensor &lhs, const Tensor &rhs) {
    return ops::left_shift(lhs, rhs);
}

inline Tensor operator>>(const Tensor &lhs, const Tensor &rhs) {
    return ops::right_shift(lhs, rhs);
}

// ============================================================================
// Scalar comparison operators (for cleaner masking syntax)
// ============================================================================

// tensor > scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator>(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::greater(tensor, scalar_tensor);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator>(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::greater(scalar_tensor, tensor);
}

// tensor >= scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator>=(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::greater_equal(tensor, scalar_tensor);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator>=(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::greater_equal(scalar_tensor, tensor);
}

// tensor < scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator<(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::less(tensor, scalar_tensor);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator<(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::less(scalar_tensor, tensor);
}

// tensor <= scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator<=(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::less_equal(tensor, scalar_tensor);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator<=(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::less_equal(scalar_tensor, tensor);
}

// tensor == scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator==(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::equal(tensor, scalar_tensor);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator==(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::equal(scalar_tensor, tensor);
}

// tensor != scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator!=(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::not_equal(tensor, scalar_tensor);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor operator!=(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::not_equal(scalar_tensor, tensor);
}

// Logical NOT for boolean tensors
inline Tensor operator!(const Tensor &tensor) {
    // Create a tensor of ones and XOR with it
    Tensor ones = Tensor::ones(tensor.shape(), DType::Bool, tensor.device());
    return ops::logical_xor(tensor.astype(DType::Bool), ones);
}

// ============================================================================
// Scalar arithmetic operations (convenience overloads)
// ============================================================================

namespace detail {
// Create a scalar tensor matching the target tensor's dtype and device.
// All casting is done on CPU to avoid eager GPU operations that would
// fragment the lazy evaluation graph.
template <typename T>
inline Tensor make_scalar(const Tensor &target, T scalar) {
    Tensor s = Tensor::full({1}, scalar); // CPU, native dtype
    if (s.dtype() != target.dtype()) {
        s = s.astype(target.dtype()); // CPU cast — trivial for 1 element
    }
    if (target.device() != Device::CPU) {
        s = s.to(target.device());
    }
    return s;
}
} // namespace detail

template <typename T> inline Tensor operator+(const Tensor &tensor, T scalar) {
    return ops::add(tensor, detail::make_scalar(tensor, scalar));
}

template <typename T> inline Tensor operator+(T scalar, const Tensor &tensor) {
    return ops::add(detail::make_scalar(tensor, scalar), tensor);
}

template <typename T> inline Tensor operator-(const Tensor &tensor, T scalar) {
    return ops::subtract(tensor, detail::make_scalar(tensor, scalar));
}

template <typename T> inline Tensor operator-(T scalar, const Tensor &tensor) {
    return ops::subtract(detail::make_scalar(tensor, scalar), tensor);
}

template <typename T> inline Tensor operator*(const Tensor &tensor, T scalar) {
    return ops::multiply(tensor, detail::make_scalar(tensor, scalar));
}

template <typename T> inline Tensor operator*(T scalar, const Tensor &tensor) {
    return ops::multiply(detail::make_scalar(tensor, scalar), tensor);
}

template <typename T> inline Tensor operator/(const Tensor &tensor, T scalar) {
    return ops::divide(tensor, detail::make_scalar(tensor, scalar));
}

template <typename T> inline Tensor operator/(T scalar, const Tensor &tensor) {
    return ops::divide(detail::make_scalar(tensor, scalar), tensor);
}

// ============================================================================
// In-place operators
// ============================================================================

inline Tensor &operator+=(Tensor &lhs, const Tensor &rhs) {
    ops::add_inplace(lhs, rhs);
    return lhs;
}

inline Tensor &operator-=(Tensor &lhs, const Tensor &rhs) {
    ops::subtract_inplace(lhs, rhs);
    return lhs;
}

inline Tensor &operator*=(Tensor &lhs, const Tensor &rhs) {
    ops::multiply_inplace(lhs, rhs);
    return lhs;
}

inline Tensor &operator/=(Tensor &lhs, const Tensor &rhs) {
    ops::divide_inplace(lhs, rhs);
    return lhs;
}

// Convenience overloads for scalars
template <typename T> inline Tensor &operator+=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::add_inplace(lhs, scalar_tensor);
    return lhs;
}

template <typename T> inline Tensor &operator-=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::subtract_inplace(lhs, scalar_tensor);
    return lhs;
}

template <typename T> inline Tensor &operator*=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::multiply_inplace(lhs, scalar_tensor);
    return lhs;
}

template <typename T> inline Tensor &operator/=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::divide_inplace(lhs, scalar_tensor);
    return lhs;
}

} // namespace axiom