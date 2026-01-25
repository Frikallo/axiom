#pragma once

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
// Scalar operations (convenience overloads)
// ============================================================================

template <typename T> inline Tensor operator+(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::add(tensor, scalar_tensor);
}

template <typename T> inline Tensor operator+(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::add(scalar_tensor, tensor);
}

template <typename T> inline Tensor operator-(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::subtract(tensor, scalar_tensor);
}

template <typename T> inline Tensor operator-(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::subtract(scalar_tensor, tensor);
}

template <typename T> inline Tensor operator*(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::multiply(tensor, scalar_tensor);
}

template <typename T> inline Tensor operator*(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::multiply(scalar_tensor, tensor);
}

template <typename T> inline Tensor operator/(const Tensor &tensor, T scalar) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::divide(tensor, scalar_tensor);
}

template <typename T> inline Tensor operator/(T scalar, const Tensor &tensor) {
    Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
    return ops::divide(scalar_tensor, tensor);
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