#pragma once

// ARM NEON SIMD traits and helpers for vectorized operations
// Used as Tier 2 fallback when Apple Accelerate doesn't apply

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// SIMD Support Detection
// ============================================================================

template <typename T> struct has_simd_support : std::false_type {};

#ifdef __ARM_NEON

template <> struct has_simd_support<float> : std::true_type {};

template <> struct has_simd_support<int32_t> : std::true_type {};

template <> struct has_simd_support<uint32_t> : std::true_type {};

template <> struct has_simd_support<int64_t> : std::true_type {};

template <> struct has_simd_support<uint64_t> : std::true_type {};

template <> struct has_simd_support<double> : std::true_type {};

template <typename T>
inline constexpr bool has_support = has_simd_support<T>::value;

// ============================================================================
// SIMD Traits Base Template
// ============================================================================

template <typename T> struct simd_traits {};

// ============================================================================
// SIMD Traits Specializations
// ============================================================================

// Float32 (4-wide SIMD)
template <> struct simd_traits<float> {
    using scalar_type = float;
    using vec_type = float32x4_t;
    static constexpr size_t width = 4;

    static vec_type load(const float *p) { return vld1q_f32(p); }
    static void store(float *p, vec_type v) { vst1q_f32(p, v); }
    static vec_type set1(float x) { return vdupq_n_f32(x); }
    static vec_type zero() { return vdupq_n_f32(0.0f); }

    static vec_type add(vec_type a, vec_type b) { return vaddq_f32(a, b); }
    static vec_type sub(vec_type a, vec_type b) { return vsubq_f32(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return vmulq_f32(a, b); }
    static vec_type div(vec_type a, vec_type b) { return vdivq_f32(a, b); }

    static vec_type max(vec_type a, vec_type b) { return vmaxq_f32(a, b); }
    static vec_type min(vec_type a, vec_type b) { return vminq_f32(a, b); }
    static vec_type abs(vec_type a) { return vabsq_f32(a); }
    static vec_type neg(vec_type a) { return vnegq_f32(a); }

    // Horizontal sum
    static float hsum(vec_type v) {
        float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
        return vget_lane_f32(vpadd_f32(sum, sum), 0);
    }

    // Horizontal max
    static float hmax(vec_type v) {
        float32x2_t max2 = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
        return vget_lane_f32(vpmax_f32(max2, max2), 0);
    }

    // Horizontal min
    static float hmin(vec_type v) {
        float32x2_t min2 = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
        return vget_lane_f32(vpmin_f32(min2, min2), 0);
    }
};

// Float64 (2-wide SIMD)
template <> struct simd_traits<double> {
    using scalar_type = double;
    using vec_type = float64x2_t;
    static constexpr size_t width = 2;

    static vec_type load(const double *p) { return vld1q_f64(p); }
    static void store(double *p, vec_type v) { vst1q_f64(p, v); }
    static vec_type set1(double x) { return vdupq_n_f64(x); }
    static vec_type zero() { return vdupq_n_f64(0.0); }

    static vec_type add(vec_type a, vec_type b) { return vaddq_f64(a, b); }
    static vec_type sub(vec_type a, vec_type b) { return vsubq_f64(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return vmulq_f64(a, b); }
    static vec_type div(vec_type a, vec_type b) { return vdivq_f64(a, b); }

    static vec_type max(vec_type a, vec_type b) { return vmaxq_f64(a, b); }
    static vec_type min(vec_type a, vec_type b) { return vminq_f64(a, b); }
    static vec_type abs(vec_type a) { return vabsq_f64(a); }
    static vec_type neg(vec_type a) { return vnegq_f64(a); }

    static double hsum(vec_type v) {
        return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1);
    }

    static double hmax(vec_type v) {
        return std::max(vgetq_lane_f64(v, 0), vgetq_lane_f64(v, 1));
    }

    static double hmin(vec_type v) {
        return std::min(vgetq_lane_f64(v, 0), vgetq_lane_f64(v, 1));
    }
};

// Int32 (4-wide SIMD)
template <> struct simd_traits<int32_t> {
    using scalar_type = int32_t;
    using vec_type = int32x4_t;
    static constexpr size_t width = 4;

    static vec_type load(const int32_t *p) { return vld1q_s32(p); }
    static void store(int32_t *p, vec_type v) { vst1q_s32(p, v); }
    static vec_type set1(int32_t x) { return vdupq_n_s32(x); }
    static vec_type zero() { return vdupq_n_s32(0); }

    static vec_type add(vec_type a, vec_type b) { return vaddq_s32(a, b); }
    static vec_type sub(vec_type a, vec_type b) { return vsubq_s32(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return vmulq_s32(a, b); }
    // No direct division for int32, use scalar fallback

    static vec_type max(vec_type a, vec_type b) { return vmaxq_s32(a, b); }
    static vec_type min(vec_type a, vec_type b) { return vminq_s32(a, b); }
    static vec_type abs(vec_type a) { return vabsq_s32(a); }
    static vec_type neg(vec_type a) { return vnegq_s32(a); }

    static int32_t hsum(vec_type v) {
        int32x2_t sum = vadd_s32(vget_low_s32(v), vget_high_s32(v));
        return vget_lane_s32(vpadd_s32(sum, sum), 0);
    }

    static int32_t hmax(vec_type v) {
        int32x2_t max2 = vpmax_s32(vget_low_s32(v), vget_high_s32(v));
        return vget_lane_s32(vpmax_s32(max2, max2), 0);
    }

    static int32_t hmin(vec_type v) {
        int32x2_t min2 = vpmin_s32(vget_low_s32(v), vget_high_s32(v));
        return vget_lane_s32(vpmin_s32(min2, min2), 0);
    }
};

// UInt32 (4-wide SIMD)
template <> struct simd_traits<uint32_t> {
    using scalar_type = uint32_t;
    using vec_type = uint32x4_t;
    static constexpr size_t width = 4;

    static vec_type load(const uint32_t *p) { return vld1q_u32(p); }
    static void store(uint32_t *p, vec_type v) { vst1q_u32(p, v); }
    static vec_type set1(uint32_t x) { return vdupq_n_u32(x); }
    static vec_type zero() { return vdupq_n_u32(0); }

    static vec_type add(vec_type a, vec_type b) { return vaddq_u32(a, b); }
    static vec_type sub(vec_type a, vec_type b) { return vsubq_u32(a, b); }
    static vec_type mul(vec_type a, vec_type b) { return vmulq_u32(a, b); }

    static vec_type max(vec_type a, vec_type b) { return vmaxq_u32(a, b); }
    static vec_type min(vec_type a, vec_type b) { return vminq_u32(a, b); }

    static uint32_t hsum(vec_type v) {
        uint32x2_t sum = vadd_u32(vget_low_u32(v), vget_high_u32(v));
        return vget_lane_u32(vpadd_u32(sum, sum), 0);
    }

    static uint32_t hmax(vec_type v) {
        uint32x2_t max2 = vpmax_u32(vget_low_u32(v), vget_high_u32(v));
        return vget_lane_u32(vpmax_u32(max2, max2), 0);
    }

    static uint32_t hmin(vec_type v) {
        uint32x2_t min2 = vpmin_u32(vget_low_u32(v), vget_high_u32(v));
        return vget_lane_u32(vpmin_u32(min2, min2), 0);
    }
};

// Int64 (2-wide SIMD)
template <> struct simd_traits<int64_t> {
    using scalar_type = int64_t;
    using vec_type = int64x2_t;
    static constexpr size_t width = 2;

    static vec_type load(const int64_t *p) { return vld1q_s64(p); }
    static void store(int64_t *p, vec_type v) { vst1q_s64(p, v); }
    static vec_type set1(int64_t x) { return vdupq_n_s64(x); }
    static vec_type zero() { return vdupq_n_s64(0); }

    static vec_type add(vec_type a, vec_type b) { return vaddq_s64(a, b); }
    static vec_type sub(vec_type a, vec_type b) { return vsubq_s64(a, b); }
    // No direct multiplication or div for int64

    static int64_t hsum(vec_type v) {
        return vgetq_lane_s64(v, 0) + vgetq_lane_s64(v, 1);
    }
};

// UInt64 (2-wide SIMD)
template <> struct simd_traits<uint64_t> {
    using scalar_type = uint64_t;
    using vec_type = uint64x2_t;
    static constexpr size_t width = 2;

    static vec_type load(const uint64_t *p) { return vld1q_u64(p); }
    static void store(uint64_t *p, vec_type v) { vst1q_u64(p, v); }
    static vec_type set1(uint64_t x) { return vdupq_n_u64(x); }
    static vec_type zero() { return vdupq_n_u64(0); }

    static vec_type add(vec_type a, vec_type b) { return vaddq_u64(a, b); }
    static vec_type sub(vec_type a, vec_type b) { return vsubq_u64(a, b); }

    static uint64_t hsum(vec_type v) {
        return vgetq_lane_u64(v, 0) + vgetq_lane_u64(v, 1);
    }
};

// ============================================================================
// Vectorized Binary Operations
// ============================================================================

// Operation structs for vectorized binary ops
struct VecAdd {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type a,
               typename simd_traits<T>::vec_type b) const {
        return simd_traits<T>::add(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return a + b; }
};

struct VecSub {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type a,
               typename simd_traits<T>::vec_type b) const {
        return simd_traits<T>::sub(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return a - b; }
};

struct VecMul {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type a,
               typename simd_traits<T>::vec_type b) const {
        return simd_traits<T>::mul(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return a * b; }
};

struct VecDiv {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type a,
               typename simd_traits<T>::vec_type b) const {
        return simd_traits<T>::div(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return a / b; }
};

struct VecMax {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type a,
               typename simd_traits<T>::vec_type b) const {
        return simd_traits<T>::max(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::max(a, b); }
};

struct VecMin {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type a,
               typename simd_traits<T>::vec_type b) const {
        return simd_traits<T>::min(a, b);
    }
    template <typename T> T scalar(T a, T b) const { return std::min(a, b); }
};

// Generic vectorized binary operation for types with SIMD support
template <typename T, typename Op>
inline void binary_vectorized(const T *a, const T *b, T *result, size_t n,
                              Op op) {
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;

    size_t i = 0;

    // Main vectorized loop
    for (; i + width <= n; i += width) {
        auto va = traits::load(a + i);
        auto vb = traits::load(b + i);
        auto vr = op.template operator()<T>(va, vb);
        traits::store(result + i, vr);
    }

    // Scalar tail
    for (; i < n; ++i) {
        result[i] = op.scalar(a[i], b[i]);
    }
}

// ============================================================================
// Vectorized Unary Operations
// ============================================================================

struct VecAbs {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type v) const {
        return simd_traits<T>::abs(v);
    }
    template <typename T> T scalar(T x) const { return std::abs(x); }
};

struct VecNeg {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type v) const {
        return simd_traits<T>::neg(v);
    }
    template <typename T> T scalar(T x) const { return -x; }
};

// ReLU: max(0, x)
struct VecReLU {
    template <typename T>
    typename simd_traits<T>::vec_type
    operator()(typename simd_traits<T>::vec_type v) const {
        return simd_traits<T>::max(v, simd_traits<T>::zero());
    }
    template <typename T> T scalar(T x) const { return x > T{0} ? x : T{0}; }
};

template <typename T, typename Op>
inline void unary_vectorized(const T *input, T *output, size_t n, Op op) {
    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;

    size_t i = 0;

    // Main vectorized loop
    for (; i + width <= n; i += width) {
        auto v = traits::load(input + i);
        auto vr = op.template operator()<T>(v);
        traits::store(output + i, vr);
    }

    // Scalar tail
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

    // Main vectorized loop
    for (; i + width <= n; i += width) {
        vec_sum = traits::add(vec_sum, traits::load(input + i));
    }

    // Horizontal sum of vector
    T sum = traits::hsum(vec_sum);

    // Scalar tail
    for (; i < n; ++i) {
        sum += input[i];
    }

    return sum;
}

template <typename T> inline T reduce_max(const T *input, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::lowest();

    using traits = simd_traits<T>;
    constexpr size_t width = traits::width;

    // Initialize with first element
    T max_val = input[0];
    size_t i = 1;

    if (n >= width) {
        auto vec_max = traits::load(input);
        i = width;

        for (; i + width <= n; i += width) {
            vec_max = traits::max(vec_max, traits::load(input + i));
        }

        max_val = traits::hmax(vec_max);
    }

    // Scalar tail
    for (; i < n; ++i) {
        max_val = std::max(max_val, input[i]);
    }

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
        i = width;

        for (; i + width <= n; i += width) {
            vec_min = traits::min(vec_min, traits::load(input + i));
        }

        min_val = traits::hmin(vec_min);
    }

    // Scalar tail
    for (; i < n; ++i) {
        min_val = std::min(min_val, input[i]);
    }

    return min_val;
}

#else // !__ARM_NEON

template <typename T> inline constexpr bool has_support = false;

// Provide empty simd_traits to avoid compilation errors
template <typename T> struct simd_traits {};

#endif // __ARM_NEON

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
