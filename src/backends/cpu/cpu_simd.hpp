#pragma once

#include <xsimd/xsimd.hpp>
#include <fxdiv.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// SIMD Support Detection
// ============================================================================

template <typename T>
inline constexpr bool has_support = xsimd::has_simd_register<T>::value;

// ============================================================================
// SIMD Traits using XSIMD
// ============================================================================

template <typename T, typename = void>
struct simd_traits {
    using scalar_type = T;
    static constexpr size_t width = 1;
};

template <typename T>
struct simd_traits<T, std::enable_if_t<xsimd::has_simd_register<T>::value>> {
    using scalar_type = T;
    using batch_type = xsimd::batch<T>;
    using vec_type = batch_type;
    static constexpr size_t width = batch_type::size;

    static batch_type load(const T* p) { return batch_type::load_unaligned(p); }
    static void store(T* p, batch_type v) { v.store_unaligned(p); }
    static batch_type set1(T x) { return batch_type::broadcast(x); }
    static batch_type zero() { return batch_type::broadcast(T{0}); }

    static batch_type add(batch_type a, batch_type b) { return a + b; }
    static batch_type sub(batch_type a, batch_type b) { return a - b; }
    static batch_type mul(batch_type a, batch_type b) { return a * b; }
    static batch_type div(batch_type a, batch_type b) { return a / b; }

    static batch_type max(batch_type a, batch_type b) { return xsimd::max(a, b); }
    static batch_type min(batch_type a, batch_type b) { return xsimd::min(a, b); }
    static batch_type abs(batch_type a) { return xsimd::abs(a); }
    static batch_type neg(batch_type a) { return -a; }

    static T hsum(batch_type v) { return xsimd::reduce_add(v); }
    static T hmax(batch_type v) { return xsimd::reduce_max(v); }
    static T hmin(batch_type v) { return xsimd::reduce_min(v); }
};

// ============================================================================
// Operation Functors
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

struct VecAbs {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::abs(v);
    }
    template <typename T> T scalar(T x) const {
        if constexpr (std::is_unsigned_v<T>) {
            return x;  // abs is identity for unsigned types
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

struct VecReLU {
    template <typename T>
    auto operator()(typename simd_traits<T>::batch_type v) const {
        return xsimd::max(v, simd_traits<T>::zero());
    }
    template <typename T> T scalar(T x) const { return x > T{0} ? x : T{0}; }
};

// ============================================================================
// XSIMD Math Functions (available on all platforms)
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

// ============================================================================
// Vectorized Binary Operations
// ============================================================================

template <typename T, typename Op>
inline void binary_vectorized(const T* a, const T* b, T* result, size_t n, Op op) {
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

// ============================================================================
// Vectorized Unary Operations
// ============================================================================

template <typename T, typename Op>
inline void unary_vectorized(const T* input, T* output, size_t n, Op op) {
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        traits::store(output + i, op.template operator()<T>(traits::load(input + i)));
    }

    for (; i < n; ++i) {
        output[i] = op.scalar(input[i]);
    }
}

// ============================================================================
// Vectorized Reductions
// ============================================================================

template <typename T>
inline T reduce_sum(const T* input, size_t n) {
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;
    auto vec_sum = traits::zero();
    size_t i = 0;

    for (; i + width <= n; i += width) {
        vec_sum = vec_sum + traits::load(input + i);
    }

    T sum = traits::hsum(vec_sum);
    for (; i < n; ++i) sum += input[i];
    return sum;
}

template <typename T>
inline T reduce_max(const T* input, size_t n) {
    if (n == 0) return std::numeric_limits<T>::lowest();
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

    for (; i < n; ++i) max_val = std::max(max_val, input[i]);
    return max_val;
}

template <typename T>
inline T reduce_min(const T* input, size_t n) {
    if (n == 0) return std::numeric_limits<T>::max();
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

    for (; i < n; ++i) min_val = std::min(min_val, input[i]);
    return min_val;
}

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
