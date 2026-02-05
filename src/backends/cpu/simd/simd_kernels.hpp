#pragma once

// SIMD Kernel Functors for Runtime Dispatch
//
// These kernels are written ONCE and compiled multiple times with different
// architecture flags (-msse2, -mavx2, etc.). The compiler generates optimal
// code for each target architecture from the same source.
//
// Pattern from xsimd docs:
// 1. Define functor with template<Arch, ...> operator()
// 2. Do NOT use in-class definitions (bypasses extern template)
// 3. Use extern template declarations to prevent inline instantiation
// 4. Explicit instantiation in separate .cpp files compiled with arch flags

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
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
// Binary Operation Kernels
// ============================================================================

struct BinaryAdd {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinarySub {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryMul {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryDiv {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryMax {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryMin {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryPow {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryAtan2 {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryHypot {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryFmod {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

// ============================================================================
// Unary Operation Kernels
// ============================================================================

struct UnaryNeg {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryAbs {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnarySqrt {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryExp {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryLog {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnarySin {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryCos {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryTanh {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryTan {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryErf {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryCbrt {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnarySquare {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryReciprocal {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnarySign {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryFloor {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryCeil {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryRound {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryTrunc {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

// ============================================================================
// Reduction Kernels
// ============================================================================

struct ReduceSum {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

struct ReduceMax {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

struct ReduceMin {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

struct ReduceProd {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

// ============================================================================
// Activation Kernels
// ============================================================================

struct ActivationReLU {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct ActivationSigmoid {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct ActivationGELU {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct ActivationSiLU {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct ActivationLeakyReLU {
    double alpha;
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n) const;
};

// ============================================================================
// Kernel Implementations
// ============================================================================

// --- Binary Operations ---

template <class Arch, typename T>
void BinaryAdd::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va + vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

template <class Arch, typename T>
void BinarySub::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va - vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}

template <class Arch, typename T>
void BinaryMul::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va * vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
}

template <class Arch, typename T>
void BinaryDiv::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va / vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] / b[i];
    }
}

template <class Arch, typename T>
void BinaryMax::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::max(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::max(a[i], b[i]);
    }
}

template <class Arch, typename T>
void BinaryMin::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::min(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::min(a[i], b[i]);
    }
}

template <class Arch, typename T>
void BinaryPow::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::pow(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::pow(a[i], b[i]);
    }
}

template <class Arch, typename T>
void BinaryAtan2::operator()(Arch, const T *a, const T *b, T *result,
                             size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::atan2(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::atan2(a[i], b[i]);
    }
}

template <class Arch, typename T>
void BinaryHypot::operator()(Arch, const T *a, const T *b, T *result,
                             size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::hypot(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::hypot(a[i], b[i]);
    }
}

template <class Arch, typename T>
void BinaryFmod::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::fmod(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::fmod(a[i], b[i]);
    }
}

// --- Unary Operations ---

template <class Arch, typename T>
void UnaryNeg::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        (-v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = -input[i];
    }
}

template <class Arch, typename T>
void UnaryAbs::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::abs(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::abs(input[i]);
    }
}

template <class Arch, typename T>
void UnarySqrt::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::sqrt(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::sqrt(input[i]);
    }
}

template <class Arch, typename T>
void UnaryExp::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::exp(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::exp(input[i]);
    }
}

template <class Arch, typename T>
void UnaryLog::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::log(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::log(input[i]);
    }
}

template <class Arch, typename T>
void UnarySin::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::sin(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::sin(input[i]);
    }
}

template <class Arch, typename T>
void UnaryCos::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::cos(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::cos(input[i]);
    }
}

template <class Arch, typename T>
void UnaryTanh::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::tanh(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

template <class Arch, typename T>
void UnaryTan::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::tan(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::tan(input[i]);
    }
}

template <class Arch, typename T>
void UnaryErf::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::erf(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::erf(input[i]);
    }
}

template <class Arch, typename T>
void UnaryCbrt::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::cbrt(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::cbrt(input[i]);
    }
}

template <class Arch, typename T>
void UnarySquare::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        (v * v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = input[i] * input[i];
    }
}

template <class Arch, typename T>
void UnaryReciprocal::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch one(static_cast<T>(1));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        (one / v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = T{1} / input[i];
    }
}

template <class Arch, typename T>
void UnarySign::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch zero(static_cast<T>(0));
    batch one(static_cast<T>(1));
    batch neg_one(static_cast<T>(-1));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        auto pos = xsimd::select(v > zero, one, zero);
        auto neg = xsimd::select(v < zero, neg_one, zero);
        (pos + neg).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        T x = input[i];
        output[i] = (x > T{0}) ? T{1} : ((x < T{0}) ? T{-1} : T{0});
    }
}

template <class Arch, typename T>
void UnaryFloor::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::floor(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::floor(input[i]);
    }
}

template <class Arch, typename T>
void UnaryCeil::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::ceil(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::ceil(input[i]);
    }
}

template <class Arch, typename T>
void UnaryRound::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::round(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::round(input[i]);
    }
}

template <class Arch, typename T>
void UnaryTrunc::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::trunc(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::trunc(input[i]);
    }
}

// --- Reductions ---

template <class Arch, typename T>
T ReduceSum::operator()(Arch, const T *data, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch acc(static_cast<T>(0));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        acc += batch::load_unaligned(data + i);
    }

    T result = xsimd::reduce_add(acc);
    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}

template <class Arch, typename T>
T ReduceMax::operator()(Arch, const T *data, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::lowest();

    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;
    T result;

    if (n >= width) {
        batch acc = batch::load_unaligned(data);
        for (i = width; i + width <= n; i += width) {
            acc = xsimd::max(acc, batch::load_unaligned(data + i));
        }
        result = xsimd::reduce_max(acc);
    } else {
        result = data[0];
        i = 1;
    }

    for (; i < n; ++i) {
        result = std::max(result, data[i]);
    }
    return result;
}

template <class Arch, typename T>
T ReduceMin::operator()(Arch, const T *data, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::max();

    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;
    T result;

    if (n >= width) {
        batch acc = batch::load_unaligned(data);
        for (i = width; i + width <= n; i += width) {
            acc = xsimd::min(acc, batch::load_unaligned(data + i));
        }
        result = xsimd::reduce_min(acc);
    } else {
        result = data[0];
        i = 1;
    }

    for (; i < n; ++i) {
        result = std::min(result, data[i]);
    }
    return result;
}

template <class Arch, typename T>
T ReduceProd::operator()(Arch, const T *data, size_t n) {
    if (n == 0)
        return static_cast<T>(1);

    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch acc(static_cast<T>(1));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        acc *= batch::load_unaligned(data + i);
    }

    // Horizontal product
    alignas(Arch::alignment()) T temp[width];
    acc.store_aligned(temp);
    T result = static_cast<T>(1);
    for (size_t j = 0; j < width; ++j) {
        result *= temp[j];
    }

    for (; i < n; ++i) {
        result *= data[i];
    }
    return result;
}

// --- Activations ---

template <class Arch, typename T>
void ActivationReLU::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch zero(static_cast<T>(0));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::max(v, zero).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > T{0} ? input[i] : T{0};
    }
}

template <class Arch, typename T>
void ActivationSigmoid::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch one(static_cast<T>(1));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        (one / (one + xsimd::exp(-v))).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = T{1} / (T{1} + std::exp(-input[i]));
    }
}

template <class Arch, typename T>
void ActivationGELU::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    constexpr T sqrt_2_over_pi = T{0.7978845608028654};
    constexpr T coeff = T{0.044715};
    batch half(static_cast<T>(0.5));
    batch one(static_cast<T>(1));
    batch sqrt2pi(sqrt_2_over_pi);
    batch c(coeff);
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto x = batch::load_unaligned(input + i);
        auto inner = sqrt2pi * (x + c * x * x * x);
        (half * x * (one + xsimd::tanh(inner))).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        T x = input[i];
        output[i] =
            T{0.5} * x *
            (T{1} + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
    }
}

template <class Arch, typename T>
void ActivationSiLU::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch one(static_cast<T>(1));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        auto sigmoid = one / (one + xsimd::exp(-v));
        (v * sigmoid).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        T x = input[i];
        output[i] = x / (T{1} + std::exp(-x));
    }
}

template <class Arch, typename T>
void ActivationLeakyReLU::operator()(Arch, const T *input, T *output,
                                     size_t n) const {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch zero(static_cast<T>(0));
    batch alpha_vec(static_cast<T>(alpha));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::select(v > zero, v, alpha_vec * v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        T x = input[i];
        output[i] = x > T{0} ? x : static_cast<T>(alpha) * x;
    }
}

// ============================================================================
// Extern Template Declarations
// ============================================================================
// These prevent the compiler from instantiating templates in every TU.
// Actual instantiations are in simd_kernels_*.cpp files compiled with
// architecture-specific flags.

#ifdef AXIOM_SIMD_MULTI_ARCH

// --- x86-64 architectures ---

// SSE2 (baseline)
extern template void BinaryAdd::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryAdd::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void BinarySub::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinarySub::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryMul::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryMul::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryDiv::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryDiv::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void UnaryExp::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryExp::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnaryLog::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryLog::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnarySqrt::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               float *, size_t);
extern template void UnarySqrt::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                                const double *,
                                                                double *,
                                                                size_t);
extern template float
ReduceSum::operator()<xsimd::sse2, float>(xsimd::sse2, const float *, size_t);
extern template double
ReduceSum::operator()<xsimd::sse2, double>(xsimd::sse2, const double *, size_t);
extern template float
ReduceMax::operator()<xsimd::sse2, float>(xsimd::sse2, const float *, size_t);
extern template double
ReduceMax::operator()<xsimd::sse2, double>(xsimd::sse2, const double *, size_t);
extern template void
ActivationReLU::operator()<xsimd::sse2, float>(xsimd::sse2, const float *,
                                               float *, size_t);
extern template void
ActivationReLU::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                                double *, size_t);

// AVX2
extern template void BinaryAdd::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryAdd::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void BinarySub::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinarySub::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryMul::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryMul::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryDiv::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryDiv::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void UnaryExp::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryExp::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnaryLog::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryLog::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnarySqrt::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               float *, size_t);
extern template void UnarySqrt::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                                const double *,
                                                                double *,
                                                                size_t);
extern template float
ReduceSum::operator()<xsimd::avx2, float>(xsimd::avx2, const float *, size_t);
extern template double
ReduceSum::operator()<xsimd::avx2, double>(xsimd::avx2, const double *, size_t);
extern template float
ReduceMax::operator()<xsimd::avx2, float>(xsimd::avx2, const float *, size_t);
extern template double
ReduceMax::operator()<xsimd::avx2, double>(xsimd::avx2, const double *, size_t);
extern template void
ActivationReLU::operator()<xsimd::avx2, float>(xsimd::avx2, const float *,
                                               float *, size_t);
extern template void
ActivationReLU::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                                double *, size_t);

// AVX512BW
extern template void
BinaryAdd::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinaryAdd::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
BinarySub::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinarySub::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
BinaryMul::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinaryMul::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
BinaryDiv::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinaryDiv::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
UnaryExp::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                             float *, size_t);
extern template void
UnaryExp::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                              double *, size_t);
extern template void
UnaryLog::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                             float *, size_t);
extern template void
UnaryLog::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                              double *, size_t);
extern template void
UnarySqrt::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              float *, size_t);
extern template void
UnarySqrt::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                               double *, size_t);
extern template float
ReduceSum::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              size_t);
extern template double
ReduceSum::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                               size_t);
extern template float
ReduceMax::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              size_t);
extern template double
ReduceMax::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                               size_t);
extern template void ActivationReLU::operator()<xsimd::avx512bw, float>(
    xsimd::avx512bw, const float *, float *, size_t);
extern template void ActivationReLU::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, double *, size_t);

#endif // AXIOM_SIMD_MULTI_ARCH

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
