#pragma once

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

namespace axiom {

// ============================================================================
// Axiom's NaN and Infinity constants
// ============================================================================

namespace numeric {

// Quiet NaN constants for each floating-point type
template <typename T> constexpr T nan() {
    static_assert(std::is_floating_point_v<T>,
                  "nan<T> requires floating-point type");
    return std::numeric_limits<T>::quiet_NaN();
}

// Positive infinity constants
template <typename T> constexpr T inf() {
    static_assert(std::is_floating_point_v<T>,
                  "inf<T> requires floating-point type");
    return std::numeric_limits<T>::infinity();
}

// Negative infinity
template <typename T> constexpr T neg_inf() {
    static_assert(std::is_floating_point_v<T>,
                  "neg_inf<T> requires floating-point type");
    return -std::numeric_limits<T>::infinity();
}

// Convenient aliases
constexpr float nan_f = std::numeric_limits<float>::quiet_NaN();
constexpr double nan_d = std::numeric_limits<double>::quiet_NaN();
constexpr float inf_f = std::numeric_limits<float>::infinity();
constexpr double inf_d = std::numeric_limits<double>::infinity();
constexpr float neg_inf_f = -std::numeric_limits<float>::infinity();
constexpr double neg_inf_d = -std::numeric_limits<double>::infinity();

// Machine epsilon
template <typename T> constexpr T epsilon() {
    static_assert(std::is_floating_point_v<T>,
                  "epsilon<T> requires floating-point type");
    return std::numeric_limits<T>::epsilon();
}

constexpr float epsilon_f = std::numeric_limits<float>::epsilon();
constexpr double epsilon_d = std::numeric_limits<double>::epsilon();

// ============================================================================
// Value classification
// ============================================================================

template <typename T> inline bool is_nan(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(value);
    }
    return false;
}

template <typename T> inline bool is_inf(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isinf(value);
    }
    return false;
}

template <typename T> inline bool is_pos_inf(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isinf(value) && value > 0;
    }
    return false;
}

template <typename T> inline bool is_neg_inf(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isinf(value) && value < 0;
    }
    return false;
}

template <typename T> inline bool is_finite(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isfinite(value);
    }
    return true; // Non-floating types are always "finite"
}

template <typename T> inline bool is_normal(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnormal(value);
    }
    return value != T(0);
}

// ============================================================================
// String representation for special values
// ============================================================================

// Display format for NaN/Inf values
struct NumericFormat {
    const char *nan_str = "nan";
    const char *pos_inf_str = "inf";
    const char *neg_inf_str = "-inf";
    int precision = 4;
    bool fixed = true;
};

// Global format settings (can be customized)
inline NumericFormat &default_format() {
    static NumericFormat fmt;
    return fmt;
}

// Convert a value to string with proper NaN/Inf handling
template <typename T>
std::string to_string(T value, const NumericFormat &fmt = default_format()) {
    if constexpr (std::is_floating_point_v<T>) {
        if (is_nan(value)) {
            return fmt.nan_str;
        }
        if (is_pos_inf(value)) {
            return fmt.pos_inf_str;
        }
        if (is_neg_inf(value)) {
            return fmt.neg_inf_str;
        }

        // Regular number formatting
        std::ostringstream oss;
        if (fmt.fixed) {
            oss << std::fixed;
        }
        oss << std::setprecision(fmt.precision) << value;
        return oss.str();
    } else if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

// ============================================================================
// Safe arithmetic operations (return NaN instead of throwing)
// ============================================================================

template <typename T> T safe_div(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        if (b == T(0)) {
            return (a == T(0)) ? nan<T>()
                               : (a > T(0) ? inf<T>() : neg_inf<T>());
        }
        return a / b;
    } else {
        if (b == T(0)) {
            return T(0); // Integer division by zero returns 0
        }
        return a / b;
    }
}

template <typename T> T safe_log(T value) {
    static_assert(std::is_floating_point_v<T>);
    if (value < T(0))
        return nan<T>();
    if (value == T(0))
        return neg_inf<T>();
    return std::log(value);
}

template <typename T> T safe_sqrt(T value) {
    static_assert(std::is_floating_point_v<T>);
    if (value < T(0))
        return nan<T>();
    return std::sqrt(value);
}

// ============================================================================
// Approximate comparison (for floating-point equality)
// ============================================================================

template <typename T>
bool approx_equal(T a, T b, T rel_tol = epsilon<T>() * T(100),
                  T abs_tol = epsilon<T>()) {
    static_assert(std::is_floating_point_v<T>);

    // Handle NaN
    if (is_nan(a) || is_nan(b))
        return false;

    // Handle infinity
    if (is_inf(a) || is_inf(b))
        return a == b;

    // Standard comparison
    T diff = std::abs(a - b);
    return diff <= abs_tol ||
           diff <= rel_tol * std::max(std::abs(a), std::abs(b));
}

} // namespace numeric

// Bring common functions to axiom namespace for convenience
using numeric::is_finite;
using numeric::is_inf;
using numeric::is_nan;

} // namespace axiom
