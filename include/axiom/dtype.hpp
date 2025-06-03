#pragma once

#include <cstdint>
#include <string>

namespace axiom {

// Data type enumeration matching NumPy's dtype system
enum class DType : uint8_t {
  // Boolean
  Bool,

  // Signed integers
  Int8,
  Int16,
  Int32,
  Int64,

  // Unsigned integers
  UInt8,
  UInt16,
  UInt32,
  UInt64,

  // Floating point
  Float16,
  Float32,
  Float64,

  // Complex (for future support)
  Complex64,
  Complex128
};

// Get size in bytes for each dtype
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

// Get string representation
std::string dtype_name(DType dtype);

// Type traits for automatic dtype deduction
template <typename T>
struct dtype_of {
  static constexpr DType value = DType::Float32;  // Default fallback
};

template <>
struct dtype_of<bool> {
  static constexpr DType value = DType::Bool;
};
template <>
struct dtype_of<int8_t> {
  static constexpr DType value = DType::Int8;
};
template <>
struct dtype_of<int16_t> {
  static constexpr DType value = DType::Int16;
};
template <>
struct dtype_of<int32_t> {
  static constexpr DType value = DType::Int32;
};
template <>
struct dtype_of<int64_t> {
  static constexpr DType value = DType::Int64;
};
template <>
struct dtype_of<uint8_t> {
  static constexpr DType value = DType::UInt8;
};
template <>
struct dtype_of<uint16_t> {
  static constexpr DType value = DType::UInt16;
};
template <>
struct dtype_of<uint32_t> {
  static constexpr DType value = DType::UInt32;
};
template <>
struct dtype_of<uint64_t> {
  static constexpr DType value = DType::UInt64;
};
template <>
struct dtype_of<float> {
  static constexpr DType value = DType::Float32;
};
template <>
struct dtype_of<double> {
  static constexpr DType value = DType::Float64;
};

template <typename T>
constexpr DType dtype_of_v = dtype_of<T>::value;

}  // namespace axiom