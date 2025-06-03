//=============================================================================
// src/tensor/type_conversion.cpp - Type conversion implementation (FIXED)
//=============================================================================

#include "axiom/type_conversion.hpp"

#include <cstring>

namespace axiom {
namespace type_conversion {

// ============================================================================
// Safe casting with loss detection
// ============================================================================

bool conversion_may_lose_precision(DType from_dtype, DType to_dtype) {
  // Same type - no loss
  if (from_dtype == to_dtype) return false;

  // Bool is special case
  if (from_dtype == DType::Bool)
    return false;  // Bool can always convert safely
  if (to_dtype == DType::Bool)
    return true;  // Converting to bool may lose precision

  // Complex to non-complex loses imaginary part
  if (axiom::is_complex_dtype(from_dtype) && !axiom::is_complex_dtype(to_dtype))
    return true;

  // Float to integer truncates
  if (axiom::is_floating_dtype(from_dtype) && axiom::is_integer_dtype(to_dtype))
    return true;

  // Higher precision to lower precision
  if (dtype_size(from_dtype) > dtype_size(to_dtype)) return true;

  // Signed to unsigned (can lose sign)
  if (axiom::is_signed_integer_dtype(from_dtype) &&
      axiom::is_unsigned_integer_dtype(to_dtype))
    return true;

  return false;
}

// ============================================================================
// NumPy-compatible dtype promotion rules
// ============================================================================

DType promote_dtypes(DType dtype1, DType dtype2) {
  if (dtype1 == dtype2) return dtype1;

  // Complex types take precedence
  if (axiom::is_complex_dtype(dtype1) || axiom::is_complex_dtype(dtype2)) {
    if (dtype1 == DType::Complex128 || dtype2 == DType::Complex128) {
      return DType::Complex128;
    }
    return DType::Complex64;
  }

  // Floating point types
  if (axiom::is_floating_dtype(dtype1) || axiom::is_floating_dtype(dtype2)) {
    if (dtype1 == DType::Float64 || dtype2 == DType::Float64) {
      return DType::Float64;
    }
    if (dtype1 == DType::Float32 || dtype2 == DType::Float32) {
      return DType::Float32;
    }
    return DType::Float16;
  }

  // Integer types - promote to larger size and preserve signedness
  if (axiom::is_integer_dtype(dtype1) && axiom::is_integer_dtype(dtype2)) {
    size_t size1 = dtype_size(dtype1);
    size_t size2 = dtype_size(dtype2);
    size_t max_size = std::max(size1, size2);

    // If either is signed, result is signed
    bool is_signed = axiom::is_signed_integer_dtype(dtype1) ||
                     axiom::is_signed_integer_dtype(dtype2);

    if (max_size >= 8) {
      return is_signed ? DType::Int64 : DType::UInt64;
    } else if (max_size >= 4) {
      return is_signed ? DType::Int32 : DType::UInt32;
    } else if (max_size >= 2) {
      return is_signed ? DType::Int16 : DType::UInt16;
    } else {
      return is_signed ? DType::Int8 : DType::UInt8;
    }
  }

  // Bool with anything else promotes to the other type
  if (dtype1 == DType::Bool) return dtype2;
  if (dtype2 == DType::Bool) return dtype1;

  // Default to Float32 for mixed types
  return DType::Float32;
}

}  // namespace type_conversion
}  // namespace axiom