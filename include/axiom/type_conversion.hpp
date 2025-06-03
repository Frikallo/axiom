//=============================================================================
// include/axiom/type_conversion.hpp - Type conversion utilities (FIXED)
//=============================================================================

#pragma once

#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "axiom/dtype.hpp"
#include "axiom/shape.hpp"

// Use half-precision floating point library
// This is a header-only library for IEEE 754 half-precision support
#include "half.hpp"

namespace axiom {
namespace type_conversion {

// Type aliases for convenience
using float16_t = half_float::half;
using complex64_t = std::complex<float>;
using complex128_t = std::complex<double>;

// ============================================================================
// Type mapping from DType enum to C++ types
// ============================================================================

template <DType dtype>
struct dtype_to_type;

template <>
struct dtype_to_type<DType::Bool> {
  using type = bool;
};
template <>
struct dtype_to_type<DType::Int8> {
  using type = int8_t;
};
template <>
struct dtype_to_type<DType::Int16> {
  using type = int16_t;
};
template <>
struct dtype_to_type<DType::Int32> {
  using type = int32_t;
};
template <>
struct dtype_to_type<DType::Int64> {
  using type = int64_t;
};
template <>
struct dtype_to_type<DType::UInt8> {
  using type = uint8_t;
};
template <>
struct dtype_to_type<DType::UInt16> {
  using type = uint16_t;
};
template <>
struct dtype_to_type<DType::UInt32> {
  using type = uint32_t;
};
template <>
struct dtype_to_type<DType::UInt64> {
  using type = uint64_t;
};
template <>
struct dtype_to_type<DType::Float16> {
  using type = float16_t;
};
template <>
struct dtype_to_type<DType::Float32> {
  using type = float;
};
template <>
struct dtype_to_type<DType::Float64> {
  using type = double;
};
template <>
struct dtype_to_type<DType::Complex64> {
  using type = complex64_t;
};
template <>
struct dtype_to_type<DType::Complex128> {
  using type = complex128_t;
};

template <DType dtype>
using dtype_to_type_t = typename dtype_to_type<dtype>::type;

// ============================================================================
// Safe numeric conversion with overflow/underflow checking
// ============================================================================

template <typename To, typename From>
struct numeric_converter {
  static To convert(const From& value) {
    if constexpr (std::is_same_v<To, From>) {
      return value;
    }
    // Boolean conversions
    else if constexpr (std::is_same_v<To, bool>) {
      // Special handling for complex to bool
      if constexpr (std::is_same_v<From, complex64_t> ||
                    std::is_same_v<From, complex128_t>) {
        // Complex number is true if either real or imaginary part is non-zero
        return value.real() != 0.0 || value.imag() != 0.0;
      } else {
        return static_cast<bool>(value);
      }
    } else if constexpr (std::is_same_v<From, bool>) {
      return static_cast<To>(value ? 1 : 0);
    }
    // Complex to real (take real part)
    else if constexpr (std::is_same_v<From, complex64_t> ||
                       std::is_same_v<From, complex128_t>) {
      if constexpr (std::is_same_v<To, complex64_t> ||
                    std::is_same_v<To, complex128_t>) {
        return static_cast<To>(value);
      } else {
        return static_cast<To>(value.real());
      }
    }
    // Real to complex
    else if constexpr (std::is_same_v<To, complex64_t> ||
                       std::is_same_v<To, complex128_t>) {
      return To(static_cast<typename To::value_type>(value), 0);
    }
    // Integer to integer with range checking
    else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>) {
      if constexpr (sizeof(To) >= sizeof(From) &&
                    std::is_signed_v<To> >= std::is_signed_v<From>) {
        // Safe conversion (no precision loss)
        return static_cast<To>(value);
      } else {
        // Check bounds with proper signedness handling
        if constexpr (std::is_signed_v<From> == std::is_signed_v<To>) {
          // Same signedness, safe to compare directly
          constexpr auto min_val = std::numeric_limits<To>::min();
          constexpr auto max_val = std::numeric_limits<To>::max();
          if (value < min_val || value > max_val) {
            return static_cast<To>(value);  // NumPy behavior: wrap around
          }
        } else if constexpr (std::is_signed_v<From> && !std::is_signed_v<To>) {
          // Signed to unsigned: check for negative values
          if (value < 0) {
            return static_cast<To>(value);  // Will wrap around
          }
          // Convert to unsigned for comparison
          using UnsignedFrom = std::make_unsigned_t<From>;
          constexpr auto max_val = std::numeric_limits<To>::max();
          if (static_cast<UnsignedFrom>(value) > max_val) {
            return static_cast<To>(value);  // Will wrap around
          }
        } else {
          // Unsigned to signed: check against signed max
          constexpr auto max_val = static_cast<std::make_unsigned_t<To>>(
              std::numeric_limits<To>::max());
          if (value > max_val) {
            return static_cast<To>(value);  // Will wrap around
          }
        }
        return static_cast<To>(value);
      }
    }
    // Float to float
    else if constexpr (std::is_floating_point_v<From> &&
                       std::is_floating_point_v<To>) {
      return static_cast<To>(value);
    }
    // Float16 conversions
    else if constexpr (std::is_same_v<From, float16_t>) {
      return static_cast<To>(static_cast<float>(value));
    } else if constexpr (std::is_same_v<To, float16_t>) {
      return float16_t(static_cast<float>(value));
    }
    // Integer to float
    else if constexpr (std::is_integral_v<From> &&
                       (std::is_floating_point_v<To> ||
                        std::is_same_v<To, float16_t>)) {
      return static_cast<To>(value);
    }
    // Float to integer (truncate)
    else if constexpr ((std::is_floating_point_v<From> ||
                        std::is_same_v<From, float16_t>) &&
                       std::is_integral_v<To>) {
      // NumPy behavior: truncate towards zero
      return static_cast<To>(value);
    } else {
      return static_cast<To>(value);
    }
  }
};

// ============================================================================
// Forward declarations
// ============================================================================

template <typename To, typename From>
void convert_array_typed(void* dst, const void* src, size_t count);

template <typename To>
void convert_array(void* dst, const void* src, size_t count, DType src_dtype);

// ============================================================================
// Type-specific conversion implementation
// ============================================================================

template <typename To, typename From>
void convert_array_typed(void* dst, const void* src, size_t count) {
  const From* src_ptr = static_cast<const From*>(src);
  To* dst_ptr = static_cast<To*>(dst);

  for (size_t i = 0; i < count; ++i) {
    dst_ptr[i] = numeric_converter<To, From>::convert(src_ptr[i]);
  }
}

// ============================================================================
// Generic conversion function dispatch
// ============================================================================

template <typename To>
void convert_array(void* dst, const void* src, size_t count, DType src_dtype) {
  switch (src_dtype) {
    case DType::Bool:
      convert_array_typed<To, bool>(dst, src, count);
      break;
    case DType::Int8:
      convert_array_typed<To, int8_t>(dst, src, count);
      break;
    case DType::Int16:
      convert_array_typed<To, int16_t>(dst, src, count);
      break;
    case DType::Int32:
      convert_array_typed<To, int32_t>(dst, src, count);
      break;
    case DType::Int64:
      convert_array_typed<To, int64_t>(dst, src, count);
      break;
    case DType::UInt8:
      convert_array_typed<To, uint8_t>(dst, src, count);
      break;
    case DType::UInt16:
      convert_array_typed<To, uint16_t>(dst, src, count);
      break;
    case DType::UInt32:
      convert_array_typed<To, uint32_t>(dst, src, count);
      break;
    case DType::UInt64:
      convert_array_typed<To, uint64_t>(dst, src, count);
      break;
    case DType::Float16:
      convert_array_typed<To, float16_t>(dst, src, count);
      break;
    case DType::Float32:
      convert_array_typed<To, float>(dst, src, count);
      break;
    case DType::Float64:
      convert_array_typed<To, double>(dst, src, count);
      break;
    case DType::Complex64:
      convert_array_typed<To, complex64_t>(dst, src, count);
      break;
    case DType::Complex128:
      convert_array_typed<To, complex128_t>(dst, src, count);
      break;
  }
}

// ============================================================================
// Main conversion dispatch function
// ============================================================================

inline void convert_dtype(void* dst, const void* src, size_t count,
                          DType dst_dtype, DType src_dtype) {
  if (dst_dtype == src_dtype) {
    // No conversion needed, just copy
    std::memcpy(dst, src, count * dtype_size(src_dtype));
    return;
  }

  switch (dst_dtype) {
    case DType::Bool:
      convert_array<bool>(dst, src, count, src_dtype);
      break;
    case DType::Int8:
      convert_array<int8_t>(dst, src, count, src_dtype);
      break;
    case DType::Int16:
      convert_array<int16_t>(dst, src, count, src_dtype);
      break;
    case DType::Int32:
      convert_array<int32_t>(dst, src, count, src_dtype);
      break;
    case DType::Int64:
      convert_array<int64_t>(dst, src, count, src_dtype);
      break;
    case DType::UInt8:
      convert_array<uint8_t>(dst, src, count, src_dtype);
      break;
    case DType::UInt16:
      convert_array<uint16_t>(dst, src, count, src_dtype);
      break;
    case DType::UInt32:
      convert_array<uint32_t>(dst, src, count, src_dtype);
      break;
    case DType::UInt64:
      convert_array<uint64_t>(dst, src, count, src_dtype);
      break;
    case DType::Float16:
      convert_array<float16_t>(dst, src, count, src_dtype);
      break;
    case DType::Float32:
      convert_array<float>(dst, src, count, src_dtype);
      break;
    case DType::Float64:
      convert_array<double>(dst, src, count, src_dtype);
      break;
    case DType::Complex64:
      convert_array<complex64_t>(dst, src, count, src_dtype);
      break;
    case DType::Complex128:
      convert_array<complex128_t>(dst, src, count, src_dtype);
      break;
  }
}

// ============================================================================
// Copy with stride support (for non-contiguous tensors)
// ============================================================================

inline void convert_dtype_strided(void* dst, const void* src,
                                  const Shape& shape,
                                  const Strides& dst_strides,
                                  const Strides& src_strides, DType dst_dtype,
                                  DType src_dtype, size_t dst_offset = 0,
                                  size_t src_offset = 0) {
  if (shape.empty()) return;

  size_t total_elements = ShapeUtils::size(shape);

  for (size_t i = 0; i < total_elements; ++i) {
    auto indices = ShapeUtils::unravel_index(i, shape);
    size_t src_linear_offset =
        src_offset + ShapeUtils::linear_index(indices, src_strides);
    size_t dst_linear_offset =
        dst_offset + ShapeUtils::linear_index(indices, dst_strides);

    const void* src_element =
        static_cast<const uint8_t*>(src) + src_linear_offset;
    void* dst_element = static_cast<uint8_t*>(dst) + dst_linear_offset;

    convert_dtype(dst_element, src_element, 1, dst_dtype, src_dtype);
  }
}

// ============================================================================
// Safe casting with loss detection
// ============================================================================

bool conversion_may_lose_precision(DType from_dtype, DType to_dtype);

// ============================================================================
// NumPy-compatible dtype promotion rules
// ============================================================================

DType promote_dtypes(DType dtype1, DType dtype2);

}  // namespace type_conversion
}  // namespace axiom