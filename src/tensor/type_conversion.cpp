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

bool conversion_may_lose_precision(DTypes from_dtype, DTypes to_dtype) {
    // Same type - no loss
    if (from_dtype.index() == to_dtype.index())
        return false;

    // Bool is special case
    if (std::holds_alternative<Bool>(from_dtype))
        return false; // Bool can always convert safely
    if (std::holds_alternative<Bool>(to_dtype))
        return true; // Converting to bool may lose precision

    // Complex to non-complex loses imaginary part
    if (axiom::is_complex(from_dtype) && !axiom::is_complex(to_dtype))
        return true;

    // Float to integer truncates
    if (axiom::is_float(from_dtype) && axiom::is_int(to_dtype))
        return true;

    // Higher precision to lower precision
    if (dtype_size(from_dtype) > dtype_size(to_dtype))
        return true;

    // Signed to unsigned (can lose sign)
    if (axiom::is_signed(from_dtype) && axiom::is_unsigned(to_dtype))
        return true;

    return false;
}

// ============================================================================
// NumPy-compatible dtype promotion rules
// ============================================================================

DTypes promote_dtypes(DTypes dtype1, DTypes dtype2) {
    if (dtype1.index() == dtype2.index())
        return dtype1;

    // Complex types take precedence
    if (axiom::is_complex(dtype1) || axiom::is_complex(dtype2)) {
        if (std::holds_alternative<Complex128>(dtype1) ||
            std::holds_alternative<Complex128>(dtype2)) {
            return DTypes{Complex128()};
        }
        return DTypes{Complex64()};
    }

    // Floating point types
    if (axiom::is_float(dtype1) || axiom::is_float(dtype2)) {
        if (std::holds_alternative<Float64>(dtype1) ||
            std::holds_alternative<Float64>(dtype2)) {
            return DTypes{Float64()};
        }
        if (std::holds_alternative<Float32>(dtype1) ||
            std::holds_alternative<Float32>(dtype2)) {
            return DTypes{Float32()};
        }
        return DTypes{Float16()};
    }

    // Integer types - promote to larger size and preserve signedness
    if (axiom::is_int(dtype1) && axiom::is_int(dtype2)) {
        size_t size1 = dtype_size(dtype1);
        size_t size2 = dtype_size(dtype2);
        size_t max_size = std::max(size1, size2);

        // If either is signed, result is signed
        bool is_signed = axiom::is_signed(dtype1) || axiom::is_signed(dtype2);

        if (max_size >= 8) {
            return is_signed ? DTypes{Int64()} : DTypes{UInt64()};
        } else if (max_size >= 4) {
            return is_signed ? DTypes{Int32()} : DTypes{UInt32()};
        } else if (max_size >= 2) {
            return is_signed ? DTypes{Int16()} : DTypes{UInt16()};
        } else {
            return is_signed ? DTypes{Int8()} : DTypes{UInt8()};
        }
    }

    // Bool with anything else promotes to the other type
    if (std::holds_alternative<Bool>(dtype1))
        return dtype2;
    if (std::holds_alternative<Bool>(dtype2))
        return dtype1;

    // Default to Float32 for mixed types
    return Float32();
}

} // namespace type_conversion
} // namespace axiom