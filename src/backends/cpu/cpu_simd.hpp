#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fxdiv.h>
#include <limits>
#include <string>
#include <type_traits>
#include <xsimd/xsimd.hpp>

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// SIMD Architecture Info
// ============================================================================

struct SimdInfo {
    const char *arch_name; // Architecture name (e.g., "neon64", "avx2")
    size_t alignment;      // Required alignment in bytes
    size_t float32_width;  // Vector width for float (elements)
    size_t float64_width;  // Vector width for double (elements)
    size_t int32_width;    // Vector width for int32_t (elements)
    size_t int64_width;    // Vector width for int64_t (elements)
};

// Get compile-time SIMD architecture info
inline SimdInfo get_simd_info() {
    using arch = xsimd::default_arch;
    return SimdInfo{
        arch::name(),
        arch::alignment(),
        xsimd::batch<float>::size,
        xsimd::batch<double>::size,
        xsimd::batch<int32_t>::size,
        xsimd::batch<int64_t>::size,
    };
}

// Print SIMD architecture info to stdout
inline void print_simd_info() {
    auto info = get_simd_info();
    std::printf("SIMD Architecture: %s\n", info.arch_name);
    std::printf("  Alignment: %zu bytes\n", info.alignment);
    std::printf("  Vector widths:\n");
    std::printf("    float32: %zu elements (%zu bytes)\n", info.float32_width,
                info.float32_width * sizeof(float));
    std::printf("    float64: %zu elements (%zu bytes)\n", info.float64_width,
                info.float64_width * sizeof(double));
    std::printf("    int32:   %zu elements (%zu bytes)\n", info.int32_width,
                info.int32_width * sizeof(int32_t));
    std::printf("    int64:   %zu elements (%zu bytes)\n", info.int64_width,
                info.int64_width * sizeof(int64_t));
}

// Get SIMD info as a formatted string
inline std::string simd_info_string() {
    auto info = get_simd_info();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "SIMD: %s (align=%zu, f32x%zu, f64x%zu, i32x%zu, i64x%zu)",
                  info.arch_name, info.alignment, info.float32_width,
                  info.float64_width, info.int32_width, info.int64_width);
    return std::string(buf);
}

// ============================================================================
// SIMD Support Detection
// ============================================================================

template <typename T>
inline constexpr bool has_support = xsimd::has_simd_register<T>::value;

// ============================================================================
// SIMD Traits using XSIMD
// ============================================================================

template <typename T, typename = void> struct simd_traits {
    using scalar_type = T;
    static constexpr size_t width = 1;
};

template <typename T>
struct simd_traits<T, std::enable_if_t<xsimd::has_simd_register<T>::value>> {
    using scalar_type = T;
    using batch_type = xsimd::batch<T>;
    using batch_bool_type = xsimd::batch_bool<T>;
    using vec_type = batch_type;
    static constexpr size_t width = batch_type::size;

    static batch_type load(const T *p) { return batch_type::load_unaligned(p); }
    static void store(T *p, batch_type v) { v.store_unaligned(p); }
    static batch_type set1(T x) { return batch_type::broadcast(x); }
    static batch_type zero() { return batch_type::broadcast(T{0}); }
    static batch_type one() { return batch_type::broadcast(T{1}); }

    static batch_type add(batch_type a, batch_type b) { return a + b; }
    static batch_type sub(batch_type a, batch_type b) { return a - b; }
    static batch_type mul(batch_type a, batch_type b) { return a * b; }
    static batch_type div(batch_type a, batch_type b) { return a / b; }

    static batch_type max(batch_type a, batch_type b) {
        return xsimd::max(a, b);
    }
    static batch_type min(batch_type a, batch_type b) {
        return xsimd::min(a, b);
    }
    static batch_type abs(batch_type a) { return xsimd::abs(a); }
    static batch_type neg(batch_type a) { return -a; }

    static T hsum(batch_type v) { return xsimd::reduce_add(v); }
    static T hmax(batch_type v) { return xsimd::reduce_max(v); }
    static T hmin(batch_type v) { return xsimd::reduce_min(v); }
};

// ============================================================================
// Basic Arithmetic Functors
// ============================================================================

struct VecAdd {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a + b;
    }
    template <typename T> T scalar(T a, T b) const { return a + b; }
};

struct VecSub {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a - b;
    }
    template <typename T> T scalar(T a, T b) const { return a - b; }
};

struct VecMul {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a * b;
    }
    template <typename T> T scalar(T a, T b) const { return a * b; }
};

struct VecDiv {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a / b;
    }
    template <typename T> T scalar(T a, T b) const { return a / b; }
};

struct VecMax {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return xsimd::max(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::max(a, b); }
};

struct VecMin {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return xsimd::min(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::min(a, b); }
};

// ============================================================================
// Comparison Functors (return bool masks)
// ============================================================================

struct VecEq {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a == b;
    }
    template <typename T> bool scalar(T a, T b) const { return a == b; }
};

struct VecNe {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a != b;
    }
    template <typename T> bool scalar(T a, T b) const { return a != b; }
};

struct VecLt {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a < b;
    }
    template <typename T> bool scalar(T a, T b) const { return a < b; }
};

struct VecLe {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a <= b;
    }
    template <typename T> bool scalar(T a, T b) const { return a <= b; }
};

struct VecGt {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a > b;
    }
    template <typename T> bool scalar(T a, T b) const { return a > b; }
};

struct VecGe {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a >= b;
    }
    template <typename T> bool scalar(T a, T b) const { return a >= b; }
};

// ============================================================================
// Bitwise Operations (for integer types)
// ============================================================================

struct VecBitwiseAnd {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a & b;
    }
    template <typename T> T scalar(T a, T b) const { return a & b; }
};

struct VecBitwiseOr {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a | b;
    }
    template <typename T> T scalar(T a, T b) const { return a | b; }
};

struct VecBitwiseXor {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a ^ b;
    }
    template <typename T> T scalar(T a, T b) const { return a ^ b; }
};

struct VecLeftShift {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a << b;
    }
    template <typename T> T scalar(T a, T b) const { return a << b; }
};

struct VecRightShift {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return a >> b;
    }
    template <typename T> T scalar(T a, T b) const { return a >> b; }
};

struct VecBitwiseNot {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return ~v;
    }
    template <typename T> T scalar(T x) const { return ~x; }
};

// ============================================================================
// Basic Unary Functors
// ============================================================================

struct VecAbs {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::abs(v);
    }
    template <typename T> T scalar(T x) const {
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return std::abs(x);
        }
    }
};

struct VecNeg {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return -v;
    }
    template <typename T> T scalar(T x) const { return -x; }
};

struct VecReciprocal {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return simd_traits<T>::one() / v;
    }
    template <typename T> T scalar(T x) const { return T{1} / x; }
};

struct VecSquare {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return v * v;
    }
    template <typename T> T scalar(T x) const { return x * x; }
};

struct VecSign {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        auto zero = simd_traits<T>::zero();
        auto pos = xsimd::select(v > zero, simd_traits<T>::one(), zero);
        auto neg = xsimd::select(v < zero, -simd_traits<T>::one(), zero);
        return pos + neg;
    }
    template <typename T> T scalar(T x) const {
        if (x > T{0})
            return T{1};
        if (x < T{0})
            return T{-1};
        return T{0};
    }
};

struct VecLogicalNot {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::select(v == simd_traits<T>::zero(), simd_traits<T>::one(),
                             simd_traits<T>::zero());
    }
    template <typename T> T scalar(T x) const {
        return x == T{0} ? T{1} : T{0};
    }
};

// ============================================================================
// Activation Functions
// ============================================================================

struct VecReLU {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::max(v, simd_traits<T>::zero());
    }
    template <typename T> T scalar(T x) const { return x > T{0} ? x : T{0}; }
};

struct VecLeakyReLU {
    double alpha_;
    explicit VecLeakyReLU(double a = 0.01) : alpha_(a) {}

    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        auto zero = simd_traits<T>::zero();
        auto alpha_vec = simd_traits<T>::set1(static_cast<T>(alpha_));
        return xsimd::select(v > zero, v, alpha_vec * v);
    }
    template <typename T> T scalar(T x) const {
        return x > T{0} ? x : static_cast<T>(alpha_) * x;
    }
};

struct VecSigmoid {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return simd_traits<T>::one() / (simd_traits<T>::one() + xsimd::exp(-v));
    }
    template <typename T> T scalar(T x) const {
        return T{1} / (T{1} + std::exp(-x));
    }
};

struct VecSiLU {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        auto sigmoid =
            simd_traits<T>::one() / (simd_traits<T>::one() + xsimd::exp(-v));
        return v * sigmoid;
    }
    template <typename T> T scalar(T x) const {
        return x / (T{1} + std::exp(-x));
    }
};

struct VecGELU {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr T sqrt_2_over_pi = T{0.7978845608028654};
        constexpr T coeff = T{0.044715};
        auto half = simd_traits<T>::set1(T{0.5});
        auto one = simd_traits<T>::one();
        auto inner = simd_traits<T>::set1(sqrt_2_over_pi) *
                     (v + simd_traits<T>::set1(coeff) * v * v * v);
        return half * v * (one + xsimd::tanh(inner));
    }
    template <typename T> T scalar(T x) const {
        constexpr T sqrt_2_over_pi = T{0.7978845608028654};
        constexpr T coeff = T{0.044715};
        return T{0.5} * x *
               (T{1} + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
    }
};

// ============================================================================
// Math Functions (Unary)
// ============================================================================

struct VecExp {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::exp(v);
    }
    template <typename T> T scalar(T x) const { return std::exp(x); }
};

struct VecLog {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::log(v);
    }
    template <typename T> T scalar(T x) const { return std::log(x); }
};

struct VecSqrt {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::sqrt(v);
    }
    template <typename T> T scalar(T x) const { return std::sqrt(x); }
};

struct VecSin {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::sin(v);
    }
    template <typename T> T scalar(T x) const { return std::sin(x); }
};

struct VecCos {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::cos(v);
    }
    template <typename T> T scalar(T x) const { return std::cos(x); }
};

struct VecTan {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::tan(v);
    }
    template <typename T> T scalar(T x) const { return std::tan(x); }
};

struct VecTanh {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::tanh(v);
    }
    template <typename T> T scalar(T x) const { return std::tanh(x); }
};

struct VecErf {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::erf(v);
    }
    template <typename T> T scalar(T x) const { return std::erf(x); }
};

struct VecCbrt {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::cbrt(v);
    }
    template <typename T> T scalar(T x) const { return std::cbrt(x); }
};

// ============================================================================
// Rounding Functions
// ============================================================================

struct VecFloor {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::floor(v);
    }
    template <typename T> T scalar(T x) const { return std::floor(x); }
};

struct VecCeil {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::ceil(v);
    }
    template <typename T> T scalar(T x) const { return std::ceil(x); }
};

struct VecRound {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::round(v);
    }
    template <typename T> T scalar(T x) const { return std::round(x); }
};

struct VecTrunc {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::trunc(v);
    }
    template <typename T> T scalar(T x) const { return std::trunc(x); }
};

// ============================================================================
// Binary Math Functions
// ============================================================================

struct VecPow {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return xsimd::pow(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::pow(a, b); }
};

struct VecAtan2 {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return xsimd::atan2(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::atan2(a, b); }
};

struct VecHypot {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return xsimd::hypot(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::hypot(a, b); }
};

struct VecFmod {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type a,
                    typename simd_traits<T>::batch_type b) const {
        return xsimd::fmod(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::fmod(a, b); }
};

// ============================================================================
// Vectorized Binary Operations
// ============================================================================

template <typename T, typename Op>
inline void binary_vectorized(const T *a, const T *b, T *result, size_t n,
                              Op op) {
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = traits::load(a + i);
        auto vb = traits::load(b + i);
        traits::store(result + i, op.template operator()<T>(va, vb));
    }

    for (; i < n; ++i) {
        result[i] = op.scalar(a[i], b[i]);
    }
}

// Binary operation that outputs to a different type (e.g., comparisons -> bool)
template <typename InT, typename OutT, typename Op>
inline void binary_vectorized_compare(const InT *a, const InT *b, OutT *result,
                                      size_t n, Op op) {
    using traits = simd_traits<InT>;
    constexpr size_t width = traits::width;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = traits::load(a + i);
        auto vb = traits::load(b + i);
        auto mask = op.template operator()<InT>(va, vb);
        // Store comparison results element by element
        alignas(64) bool temp[width];
        for (size_t j = 0; j < width; ++j) {
            temp[j] = mask.get(j);
        }
        for (size_t j = 0; j < width; ++j) {
            result[i + j] = static_cast<OutT>(temp[j]);
        }
    }

    for (; i < n; ++i) {
        result[i] = static_cast<OutT>(op.scalar(a[i], b[i]));
    }
}

// ============================================================================
// Vectorized Unary Operations
// ============================================================================

template <typename T, typename Op>
inline void unary_vectorized(const T *input, T *output, size_t n, Op op) {
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        traits::store(output + i,
                      op.template operator()<T>(traits::load(input + i)));
    }

    for (; i < n; ++i) {
        output[i] = op.scalar(input[i]);
    }
}

// ============================================================================
// Vectorized Reductions
// ============================================================================

template <typename T> inline T reduce_sum(const T *input, size_t n) {
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    auto vec_sum = traits::zero();
    size_t i = 0;

    for (; i + width <= n; i += width) {
        vec_sum = vec_sum + traits::load(input + i);
    }

    T sum = traits::hsum(vec_sum);
    for (; i < n; ++i)
        sum += input[i];
    return sum;
}

template <typename T> inline T reduce_max(const T *input, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::lowest();
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    T max_val = input[0];
    size_t i = 1;

    if (n >= width) {
        auto vec_max = traits::load(input);
        for (i = width; i + width <= n; i += width) {
            vec_max = xsimd::max(vec_max, traits::load(input + i));
        }
        max_val = traits::hmax(vec_max);
    }

    for (; i < n; ++i)
        max_val = std::max(max_val, input[i]);
    return max_val;
}

template <typename T> inline T reduce_min(const T *input, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::max();
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    T min_val = input[0];
    size_t i = 1;

    if (n >= width) {
        auto vec_min = traits::load(input);
        for (i = width; i + width <= n; i += width) {
            vec_min = xsimd::min(vec_min, traits::load(input + i));
        }
        min_val = traits::hmin(vec_min);
    }

    for (; i < n; ++i)
        min_val = std::min(min_val, input[i]);
    return min_val;
}

template <typename T> inline T reduce_prod(const T *input, size_t n) {
    if (n == 0)
        return T{1};
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    auto vec_prod = traits::one();
    size_t i = 0;

    for (; i + width <= n; i += width) {
        vec_prod = vec_prod * traits::load(input + i);
    }

    // Horizontal product via sequential multiplication
    T prod = T{1};
    alignas(64) T temp[width];
    vec_prod.store_aligned(temp);
    for (size_t j = 0; j < width; ++j) {
        prod *= temp[j];
    }

    for (; i < n; ++i)
        prod *= input[i];
    return prod;
}

// ============================================================================
// FXdiv Utilities for Fast Integer Division
// ============================================================================

namespace fxdiv_utils {

// Precomputed divisor wrapper for cleaner API
struct Divisor32 {
    fxdiv_divisor_uint32_t div;

    explicit Divisor32(uint32_t d) : div(fxdiv_init_uint32_t(d)) {}

    uint32_t quotient(uint32_t n) const {
        return fxdiv_quotient_uint32_t(n, div);
    }

    uint32_t remainder(uint32_t n) const {
        return fxdiv_remainder_uint32_t(n, div);
    }

    std::pair<uint32_t, uint32_t> divmod(uint32_t n) const {
        return {fxdiv_quotient_uint32_t(n, div),
                fxdiv_remainder_uint32_t(n, div)};
    }
};

struct Divisor64 {
    fxdiv_divisor_uint64_t div;

    explicit Divisor64(uint64_t d) : div(fxdiv_init_uint64_t(d)) {}

    uint64_t quotient(uint64_t n) const {
        return fxdiv_quotient_uint64_t(n, div);
    }

    uint64_t remainder(uint64_t n) const {
        return fxdiv_remainder_uint64_t(n, div);
    }

    std::pair<uint64_t, uint64_t> divmod(uint64_t n) const {
        return {fxdiv_quotient_uint64_t(n, div),
                fxdiv_remainder_uint64_t(n, div)};
    }
};

struct DivisorSize {
    fxdiv_divisor_size_t div;

    explicit DivisorSize(size_t d) : div(fxdiv_init_size_t(d)) {}

    size_t quotient(size_t n) const { return fxdiv_quotient_size_t(n, div); }

    size_t remainder(size_t n) const { return fxdiv_remainder_size_t(n, div); }

    std::pair<size_t, size_t> divmod(size_t n) const {
        return {fxdiv_quotient_size_t(n, div), fxdiv_remainder_size_t(n, div)};
    }
};

// Multi-dimensional index calculator with precomputed divisors
class IndexCalculator {
  public:
    IndexCalculator(const size_t *dims, size_t ndim) : ndim_(ndim) {
        divisors_.reserve(ndim);
        size_t stride = 1;
        for (size_t i = ndim; i > 0; --i) {
            strides_.push_back(stride);
            stride *= dims[i - 1];
        }
        std::reverse(strides_.begin(), strides_.end());

        for (size_t i = 0; i < ndim; ++i) {
            if (strides_[i] > 0) {
                divisors_.emplace_back(strides_[i]);
            }
        }
    }

    // Convert linear index to multi-dimensional indices
    void linear_to_indices(size_t linear_idx, size_t *indices) const {
        size_t remaining = linear_idx;
        for (size_t i = 0; i < ndim_; ++i) {
            if (i < divisors_.size() && strides_[i] > 0) {
                auto [q, r] = divisors_[i].divmod(remaining);
                indices[i] = q;
                remaining = r;
            } else {
                indices[i] = remaining;
                remaining = 0;
            }
        }
    }

  private:
    size_t ndim_;
    std::vector<DivisorSize> divisors_;
    std::vector<size_t> strides_;
};

// Broadcast index calculator with precomputed divisors for each dimension
class BroadcastIndexCalculator {
  public:
    BroadcastIndexCalculator(const size_t *result_shape,
                             const int64_t *lhs_strides,
                             const int64_t *rhs_strides, size_t ndim)
        : ndim_(ndim), lhs_strides_(lhs_strides, lhs_strides + ndim),
          rhs_strides_(rhs_strides, rhs_strides + ndim) {

        // Compute result strides and precompute divisors
        size_t stride = 1;
        result_strides_.resize(ndim);
        for (size_t i = ndim; i > 0; --i) {
            result_strides_[i - 1] = stride;
            stride *= result_shape[i - 1];
        }

        for (size_t i = 0; i < ndim; ++i) {
            if (result_strides_[i] > 0) {
                divisors_.emplace_back(result_strides_[i]);
            }
        }
    }

    // Get lhs and rhs linear indices from result linear index
    std::pair<size_t, size_t> get_input_indices(size_t result_idx) const {
        size_t lhs_idx = 0, rhs_idx = 0;
        size_t remaining = result_idx;

        for (size_t i = 0; i < ndim_; ++i) {
            size_t dim_idx;
            if (i < divisors_.size() && result_strides_[i] > 0) {
                dim_idx = divisors_[i].quotient(remaining);
                remaining = divisors_[i].remainder(remaining);
            } else {
                dim_idx = remaining;
                remaining = 0;
            }

            lhs_idx += dim_idx * static_cast<size_t>(
                                     std::max(int64_t{0}, lhs_strides_[i]));
            rhs_idx += dim_idx * static_cast<size_t>(
                                     std::max(int64_t{0}, rhs_strides_[i]));
        }

        return {lhs_idx, rhs_idx};
    }

  private:
    size_t ndim_;
    std::vector<DivisorSize> divisors_;
    std::vector<size_t> result_strides_;
    std::vector<int64_t> lhs_strides_;
    std::vector<int64_t> rhs_strides_;
};

// Reduction index calculator
class ReductionIndexCalculator {
  public:
    ReductionIndexCalculator(const size_t *input_shape,
                             const int64_t *input_strides,
                             const std::vector<bool> &is_reduced_axis,
                             size_t ndim)
        : ndim_(ndim), input_strides_(input_strides, input_strides + ndim),
          is_reduced_axis_(is_reduced_axis) {

        // Compute output strides for non-reduced dimensions
        size_t output_stride = 1;
        output_strides_.resize(ndim, 0);
        for (size_t i = ndim; i > 0; --i) {
            if (!is_reduced_axis[i - 1]) {
                output_strides_[i - 1] = output_stride;
                output_stride *= input_shape[i - 1];
            }
        }

        for (size_t i = 0; i < ndim; ++i) {
            if (output_strides_[i] > 0) {
                output_divisors_.emplace_back(output_strides_[i]);
            } else {
                output_divisors_.emplace_back(
                    1); // Dummy divisor for reduced dims
            }
        }
    }

    // Get input linear index from output index and reduction index
    size_t get_input_index(size_t output_idx, size_t reduction_idx) const {
        size_t input_idx = 0;
        size_t out_remaining = output_idx;
        size_t red_remaining = reduction_idx;

        for (size_t i = 0; i < ndim_; ++i) {
            size_t dim_idx;
            if (is_reduced_axis_[i]) {
                // This dimension is reduced - get index from reduction_idx
                dim_idx =
                    red_remaining; // Simplified - actual impl needs shape info
                red_remaining = 0;
            } else {
                // Non-reduced dimension - get from output_idx
                if (output_strides_[i] > 0) {
                    dim_idx = output_divisors_[i].quotient(out_remaining);
                    out_remaining =
                        output_divisors_[i].remainder(out_remaining);
                } else {
                    dim_idx = out_remaining;
                    out_remaining = 0;
                }
            }
            input_idx += dim_idx * static_cast<size_t>(input_strides_[i]);
        }

        return input_idx;
    }

  private:
    size_t ndim_;
    std::vector<DivisorSize> output_divisors_;
    std::vector<size_t> output_strides_;
    std::vector<int64_t> input_strides_;
    std::vector<bool> is_reduced_axis_;
};

} // namespace fxdiv_utils

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
