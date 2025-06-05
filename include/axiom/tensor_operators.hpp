#pragma once

#include "tensor.hpp"
#include "operations.hpp"

namespace axiom {

// ============================================================================
// Arithmetic operators
// ============================================================================

inline Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
  return ops::add(lhs, rhs);
}

inline Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
  return ops::subtract(lhs, rhs);
}

inline Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
  return ops::multiply(lhs, rhs);
}

inline Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
  return ops::divide(lhs, rhs);
}

inline Tensor operator%(const Tensor& lhs, const Tensor& rhs) {
  return ops::modulo(lhs, rhs);
}

// ============================================================================
// Comparison operators
// ============================================================================

inline Tensor operator==(const Tensor& lhs, const Tensor& rhs) {
  return ops::equal(lhs, rhs);
}

inline Tensor operator!=(const Tensor& lhs, const Tensor& rhs) {
  return ops::not_equal(lhs, rhs);
}

inline Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
  return ops::less(lhs, rhs);
}

inline Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {
  return ops::less_equal(lhs, rhs);
}

inline Tensor operator>(const Tensor& lhs, const Tensor& rhs) {
  return ops::greater(lhs, rhs);
}

inline Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {
  return ops::greater_equal(lhs, rhs);
}

// ============================================================================
// Logical operators
// ============================================================================

inline Tensor operator&&(const Tensor& lhs, const Tensor& rhs) {
  return ops::logical_and(lhs, rhs);
}

inline Tensor operator||(const Tensor& lhs, const Tensor& rhs) {
  return ops::logical_or(lhs, rhs);
}

// ============================================================================
// Bitwise operators
// ============================================================================

inline Tensor operator&(const Tensor& lhs, const Tensor& rhs) {
  return ops::bitwise_and(lhs, rhs);
}

inline Tensor operator|(const Tensor& lhs, const Tensor& rhs) {
  return ops::bitwise_or(lhs, rhs);
}

inline Tensor operator^(const Tensor& lhs, const Tensor& rhs) {
  return ops::bitwise_xor(lhs, rhs);
}

inline Tensor operator<<(const Tensor& lhs, const Tensor& rhs) {
  return ops::left_shift(lhs, rhs);
}

inline Tensor operator>>(const Tensor& lhs, const Tensor& rhs) {
  return ops::right_shift(lhs, rhs);
}

// ============================================================================
// Scalar operations (convenience overloads)
// ============================================================================

template<typename T>
inline Tensor operator+(const Tensor& tensor, T scalar) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::add(tensor, scalar_tensor);
}

template<typename T>
inline Tensor operator+(T scalar, const Tensor& tensor) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::add(scalar_tensor, tensor);
}

template<typename T>
inline Tensor operator-(const Tensor& tensor, T scalar) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::subtract(tensor, scalar_tensor);
}

template<typename T>
inline Tensor operator-(T scalar, const Tensor& tensor) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::subtract(scalar_tensor, tensor);
}

template<typename T>
inline Tensor operator*(const Tensor& tensor, T scalar) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::multiply(tensor, scalar_tensor);
}

template<typename T>
inline Tensor operator*(T scalar, const Tensor& tensor) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::multiply(scalar_tensor, tensor);
}

template<typename T>
inline Tensor operator/(const Tensor& tensor, T scalar) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::divide(tensor, scalar_tensor);
}

template<typename T>
inline Tensor operator/(T scalar, const Tensor& tensor) {
  Tensor scalar_tensor = Tensor::full({1}, scalar, tensor.device());
  return ops::divide(scalar_tensor, tensor);
}

// ============================================================================
// In-place operators (future extension)
// ============================================================================

// Note: These would require in-place operation support in the backend
// For now, they create new tensors and copy the result back

inline Tensor& operator+=(Tensor& lhs, const Tensor& rhs) {
  lhs = lhs + rhs;
  return lhs;
}

inline Tensor& operator-=(Tensor& lhs, const Tensor& rhs) {
  lhs = lhs - rhs;
  return lhs;
}

inline Tensor& operator*=(Tensor& lhs, const Tensor& rhs) {
  lhs = lhs * rhs;
  return lhs;
}

inline Tensor& operator/=(Tensor& lhs, const Tensor& rhs) {
  lhs = lhs / rhs;
  return lhs;
}

}  // namespace axiom 