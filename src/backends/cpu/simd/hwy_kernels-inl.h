// hwy_kernels-inl.h - Highway SIMD kernel implementations
//
// All kernel implementations for binary, unary, reduction, and activation
// operations. This file is compiled multiple times via HWY_FOREACH_TARGET
// to generate optimal code for each supported architecture.
//
// Highway handles:
// - x86: SSE2, SSSE3, SSE4, AVX, AVX2, AVX3, AVX3_DL, AVX3_ZEN4
// - ARM: NEON
// - RISC-V: RVV
// - WebAssembly: WASM, WASM_EMU256
// - PowerPC: PPC8, PPC9, PPC10
//
// SVE/SVE2 are disabled on Apple Silicon (not supported by M-series chips)

// Disable SVE targets on Apple Silicon before including any Highway headers
#if defined(__APPLE__) && defined(__aarch64__)
#define HWY_DISABLED_TARGETS (HWY_SVE | HWY_SVE2 | HWY_SVE_256 | HWY_SVE2_128)
#endif

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "backends/cpu/simd/hwy_kernels-inl.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// clang-format on

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <algorithm>
#include <cmath>
#include <limits>

HWY_BEFORE_NAMESPACE();
namespace axiom {
namespace simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Include custom math extensions (Erf, Cbrt) - must be after namespace opens
#include "backends/cpu/simd/hwy_math_ext-inl.h"

// ============================================================================
// Binary Operations (templated implementations)
// ============================================================================

template <typename T>
HWY_NOINLINE void BinaryAddImplT(const T *HWY_RESTRICT a,
                                 const T *HWY_RESTRICT b,
                                 T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Add(va, vb), d, result + i);
    }
    // Scalar tail
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

template <typename T>
HWY_NOINLINE void BinarySubImplT(const T *HWY_RESTRICT a,
                                 const T *HWY_RESTRICT b,
                                 T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Sub(va, vb), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}

template <typename T>
HWY_NOINLINE void BinaryMulImplT(const T *HWY_RESTRICT a,
                                 const T *HWY_RESTRICT b,
                                 T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Mul(va, vb), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
}

template <typename T>
HWY_NOINLINE void BinaryDivImplT(const T *HWY_RESTRICT a,
                                 const T *HWY_RESTRICT b,
                                 T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Div(va, vb), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] / b[i];
    }
}

template <typename T>
HWY_NOINLINE void BinaryMaxImplT(const T *HWY_RESTRICT a,
                                 const T *HWY_RESTRICT b,
                                 T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Max(va, vb), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::max(a[i], b[i]);
    }
}

template <typename T>
HWY_NOINLINE void BinaryMinImplT(const T *HWY_RESTRICT a,
                                 const T *HWY_RESTRICT b,
                                 T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Min(va, vb), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::min(a[i], b[i]);
    }
}

template <typename T>
HWY_NOINLINE void BinaryPowImplT(const T *HWY_RESTRICT a,
                                 const T *HWY_RESTRICT b,
                                 T *HWY_RESTRICT result, size_t n) {
    // pow(a, b) = exp(b * log(a))
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto log_a = hn::Log(d, va);
        const auto result_vec = hn::Exp(d, hn::Mul(vb, log_a));
        hn::StoreU(result_vec, d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::pow(a[i], b[i]);
    }
}

template <typename T>
HWY_NOINLINE void BinaryAtan2ImplT(const T *HWY_RESTRICT a,
                                   const T *HWY_RESTRICT b,
                                   T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        hn::StoreU(hn::Atan2(d, va, vb), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::atan2(a[i], b[i]);
    }
}

template <typename T>
HWY_NOINLINE void BinaryHypotImplT(const T *HWY_RESTRICT a,
                                   const T *HWY_RESTRICT b,
                                   T *HWY_RESTRICT result, size_t n) {
    // hypot(a, b) = sqrt(a*a + b*b) with overflow protection
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto a2 = hn::Mul(va, va);
        const auto b2 = hn::Mul(vb, vb);
        hn::StoreU(hn::Sqrt(hn::Add(a2, b2)), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::hypot(a[i], b[i]);
    }
}

template <typename T>
HWY_NOINLINE void BinaryFmodImplT(const T *HWY_RESTRICT a,
                                  const T *HWY_RESTRICT b,
                                  T *HWY_RESTRICT result, size_t n) {
    // fmod(a, b) = a - trunc(a/b) * b
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto quot = hn::Div(va, vb);
        const auto trunc_quot = hn::Trunc(quot);
        hn::StoreU(hn::NegMulAdd(trunc_quot, vb, va), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::fmod(a[i], b[i]);
    }
}

// ============================================================================
// Unary Operations (templated implementations)
// ============================================================================

template <typename T>
HWY_NOINLINE void UnaryNegImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Neg(v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = -input[i];
    }
}

template <typename T>
HWY_NOINLINE void UnaryAbsImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Abs(v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::abs(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnarySqrtImplT(const T *HWY_RESTRICT input,
                                 T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Sqrt(v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::sqrt(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryExpImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Exp(d, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::exp(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryLogImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Log(d, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::log(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnarySinImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Sin(d, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::sin(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryCosImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Cos(d, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::cos(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryTanhImplT(const T *HWY_RESTRICT input,
                                 T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Tanh(d, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryTanImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    // tan(x) = sin(x) / cos(x)
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        const auto sin_v = hn::Sin(d, v);
        const auto cos_v = hn::Cos(d, v);
        hn::StoreU(hn::Div(sin_v, cos_v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::tan(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryErfImplT(const T *HWY_RESTRICT input,
                                T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(axiom::simd::HWY_NAMESPACE::Erf(d, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::erf(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryCbrtImplT(const T *HWY_RESTRICT input,
                                 T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(axiom::simd::HWY_NAMESPACE::Cbrt(d, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::cbrt(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnarySquareImplT(const T *HWY_RESTRICT input,
                                   T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Mul(v, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = input[i] * input[i];
    }
}

template <typename T>
HWY_NOINLINE void UnaryReciprocalImplT(const T *HWY_RESTRICT input,
                                       T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, T{1});
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Div(one, v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = T{1} / input[i];
    }
}

template <typename T>
HWY_NOINLINE void UnarySignImplT(const T *HWY_RESTRICT input,
                                 T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    const auto one = hn::Set(d, T{1});
    const auto neg_one = hn::Set(d, T{-1});
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        const auto pos_mask = hn::Gt(v, zero);
        const auto neg_mask = hn::Lt(v, zero);
        auto result = zero;
        result = hn::IfThenElse(pos_mask, one, result);
        result = hn::IfThenElse(neg_mask, neg_one, result);
        hn::StoreU(result, d, output + i);
    }
    for (; i < n; ++i) {
        output[i] =
            (input[i] > T{0}) ? T{1} : ((input[i] < T{0}) ? T{-1} : T{0});
    }
}

template <typename T>
HWY_NOINLINE void UnaryFloorImplT(const T *HWY_RESTRICT input,
                                  T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Floor(v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::floor(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryCeilImplT(const T *HWY_RESTRICT input,
                                 T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Ceil(v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::ceil(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryRoundImplT(const T *HWY_RESTRICT input,
                                  T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Round(v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::round(input[i]);
    }
}

template <typename T>
HWY_NOINLINE void UnaryTruncImplT(const T *HWY_RESTRICT input,
                                  T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Trunc(v), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::trunc(input[i]);
    }
}

// ============================================================================
// Reductions (templated implementations)
// ============================================================================

template <typename T>
HWY_NOINLINE T ReduceSumImplT(const T *HWY_RESTRICT data, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    auto acc = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        acc = hn::Add(acc, hn::LoadU(d, data + i));
    }

    T result = hn::ReduceSum(d, acc);

    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}

template <typename T>
HWY_NOINLINE T ReduceMaxImplT(const T *HWY_RESTRICT data, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::lowest();

    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    auto acc = hn::Set(d, std::numeric_limits<T>::lowest());
    size_t i = 0;

    for (; i + N <= n; i += N) {
        acc = hn::Max(acc, hn::LoadU(d, data + i));
    }

    T result = hn::ReduceMax(d, acc);

    for (; i < n; ++i) {
        result = std::max(result, data[i]);
    }
    return result;
}

template <typename T>
HWY_NOINLINE T ReduceMinImplT(const T *HWY_RESTRICT data, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::max();

    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    auto acc = hn::Set(d, std::numeric_limits<T>::max());
    size_t i = 0;

    for (; i + N <= n; i += N) {
        acc = hn::Min(acc, hn::LoadU(d, data + i));
    }

    T result = hn::ReduceMin(d, acc);

    for (; i < n; ++i) {
        result = std::min(result, data[i]);
    }
    return result;
}

template <typename T>
HWY_NOINLINE T ReduceProdImplT(const T *HWY_RESTRICT data, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    auto acc = hn::Set(d, T{1});
    size_t i = 0;

    for (; i + N <= n; i += N) {
        acc = hn::Mul(acc, hn::LoadU(d, data + i));
    }

    // Horizontal product via log/exp or manual extraction
    alignas(HWY_MAX_BYTES) T temp[HWY_MAX_LANES_D(hn::ScalableTag<T>)];
    hn::Store(acc, d, temp);
    T result = T{1};
    for (size_t j = 0; j < N; ++j) {
        result *= temp[j];
    }

    for (; i < n; ++i) {
        result *= data[i];
    }
    return result;
}

// ============================================================================
// Activation Functions (templated implementations)
// ============================================================================

template <typename T>
HWY_NOINLINE void ActivationReLUImplT(const T *HWY_RESTRICT input,
                                      T *HWY_RESTRICT output, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        hn::StoreU(hn::Max(v, zero), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::max(input[i], T{0});
    }
}

template <typename T>
HWY_NOINLINE void ActivationSigmoidImplT(const T *HWY_RESTRICT input,
                                         T *HWY_RESTRICT output, size_t n) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, T{1});
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        const auto neg_v = hn::Neg(v);
        const auto exp_neg = hn::Exp(d, neg_v);
        const auto denom = hn::Add(one, exp_neg);
        hn::StoreU(hn::Div(one, denom), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = T{1} / (T{1} + std::exp(-input[i]));
    }
}

template <typename T>
HWY_NOINLINE void ActivationGELUImplT(const T *HWY_RESTRICT input,
                                      T *HWY_RESTRICT output, size_t n) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto half = hn::Set(d, T{0.5});
    const auto one = hn::Set(d, T{1});
    const auto inv_sqrt2 = hn::Set(d, T{0.7071067811865476}); // 1/sqrt(2)
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        const auto scaled = hn::Mul(v, inv_sqrt2);
        const auto erf_val = axiom::simd::HWY_NAMESPACE::Erf(d, scaled);
        const auto inner = hn::Add(one, erf_val);
        hn::StoreU(hn::Mul(half, hn::Mul(v, inner)), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = T{0.5} * input[i] *
                    (T{1} + std::erf(input[i] * T{0.7071067811865476}));
    }
}

template <typename T>
HWY_NOINLINE void ActivationSiLUImplT(const T *HWY_RESTRICT input,
                                      T *HWY_RESTRICT output, size_t n) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, T{1});
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        const auto neg_v = hn::Neg(v);
        const auto exp_neg = hn::Exp(d, neg_v);
        const auto denom = hn::Add(one, exp_neg);
        hn::StoreU(hn::Div(v, denom), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = input[i] / (T{1} + std::exp(-input[i]));
    }
}

template <typename T>
HWY_NOINLINE void ActivationLeakyReLUImplT(const T *HWY_RESTRICT input,
                                           T *HWY_RESTRICT output, size_t n,
                                           T alpha) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    const auto alpha_vec = hn::Set(d, alpha);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, input + i);
        const auto neg_part = hn::Mul(alpha_vec, v);
        const auto pos_mask = hn::Gt(v, zero);
        hn::StoreU(hn::IfThenElse(pos_mask, v, neg_part), d, output + i);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > T{0} ? input[i] : alpha * input[i];
    }
}

// ============================================================================
// Non-template wrapper functions for HWY_EXPORT (float)
// ============================================================================

// Binary ops - float
HWY_NOINLINE void BinaryAddF(const float *a, const float *b, float *r,
                             size_t n) {
    BinaryAddImplT(a, b, r, n);
}
HWY_NOINLINE void BinarySubF(const float *a, const float *b, float *r,
                             size_t n) {
    BinarySubImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryMulF(const float *a, const float *b, float *r,
                             size_t n) {
    BinaryMulImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryDivF(const float *a, const float *b, float *r,
                             size_t n) {
    BinaryDivImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryMaxF(const float *a, const float *b, float *r,
                             size_t n) {
    BinaryMaxImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryMinF(const float *a, const float *b, float *r,
                             size_t n) {
    BinaryMinImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryPowF(const float *a, const float *b, float *r,
                             size_t n) {
    BinaryPowImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryAtan2F(const float *a, const float *b, float *r,
                               size_t n) {
    BinaryAtan2ImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryHypotF(const float *a, const float *b, float *r,
                               size_t n) {
    BinaryHypotImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryFmodF(const float *a, const float *b, float *r,
                              size_t n) {
    BinaryFmodImplT(a, b, r, n);
}

// Binary ops - double
HWY_NOINLINE void BinaryAddD(const double *a, const double *b, double *r,
                             size_t n) {
    BinaryAddImplT(a, b, r, n);
}
HWY_NOINLINE void BinarySubD(const double *a, const double *b, double *r,
                             size_t n) {
    BinarySubImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryMulD(const double *a, const double *b, double *r,
                             size_t n) {
    BinaryMulImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryDivD(const double *a, const double *b, double *r,
                             size_t n) {
    BinaryDivImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryMaxD(const double *a, const double *b, double *r,
                             size_t n) {
    BinaryMaxImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryMinD(const double *a, const double *b, double *r,
                             size_t n) {
    BinaryMinImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryPowD(const double *a, const double *b, double *r,
                             size_t n) {
    BinaryPowImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryAtan2D(const double *a, const double *b, double *r,
                               size_t n) {
    BinaryAtan2ImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryHypotD(const double *a, const double *b, double *r,
                               size_t n) {
    BinaryHypotImplT(a, b, r, n);
}
HWY_NOINLINE void BinaryFmodD(const double *a, const double *b, double *r,
                              size_t n) {
    BinaryFmodImplT(a, b, r, n);
}

// Unary ops - float
HWY_NOINLINE void UnaryNegF(const float *i, float *o, size_t n) {
    UnaryNegImplT(i, o, n);
}
HWY_NOINLINE void UnaryAbsF(const float *i, float *o, size_t n) {
    UnaryAbsImplT(i, o, n);
}
HWY_NOINLINE void UnarySqrtF(const float *i, float *o, size_t n) {
    UnarySqrtImplT(i, o, n);
}
HWY_NOINLINE void UnaryExpF(const float *i, float *o, size_t n) {
    UnaryExpImplT(i, o, n);
}
HWY_NOINLINE void UnaryLogF(const float *i, float *o, size_t n) {
    UnaryLogImplT(i, o, n);
}
HWY_NOINLINE void UnarySinF(const float *i, float *o, size_t n) {
    UnarySinImplT(i, o, n);
}
HWY_NOINLINE void UnaryCosF(const float *i, float *o, size_t n) {
    UnaryCosImplT(i, o, n);
}
HWY_NOINLINE void UnaryTanhF(const float *i, float *o, size_t n) {
    UnaryTanhImplT(i, o, n);
}
HWY_NOINLINE void UnaryTanF(const float *i, float *o, size_t n) {
    UnaryTanImplT(i, o, n);
}
HWY_NOINLINE void UnaryErfF(const float *i, float *o, size_t n) {
    UnaryErfImplT(i, o, n);
}
HWY_NOINLINE void UnaryCbrtF(const float *i, float *o, size_t n) {
    UnaryCbrtImplT(i, o, n);
}
HWY_NOINLINE void UnarySquareF(const float *i, float *o, size_t n) {
    UnarySquareImplT(i, o, n);
}
HWY_NOINLINE void UnaryReciprocalF(const float *i, float *o, size_t n) {
    UnaryReciprocalImplT(i, o, n);
}
HWY_NOINLINE void UnarySignF(const float *i, float *o, size_t n) {
    UnarySignImplT(i, o, n);
}
HWY_NOINLINE void UnaryFloorF(const float *i, float *o, size_t n) {
    UnaryFloorImplT(i, o, n);
}
HWY_NOINLINE void UnaryCeilF(const float *i, float *o, size_t n) {
    UnaryCeilImplT(i, o, n);
}
HWY_NOINLINE void UnaryRoundF(const float *i, float *o, size_t n) {
    UnaryRoundImplT(i, o, n);
}
HWY_NOINLINE void UnaryTruncF(const float *i, float *o, size_t n) {
    UnaryTruncImplT(i, o, n);
}

// Unary ops - double
HWY_NOINLINE void UnaryNegD(const double *i, double *o, size_t n) {
    UnaryNegImplT(i, o, n);
}
HWY_NOINLINE void UnaryAbsD(const double *i, double *o, size_t n) {
    UnaryAbsImplT(i, o, n);
}
HWY_NOINLINE void UnarySqrtD(const double *i, double *o, size_t n) {
    UnarySqrtImplT(i, o, n);
}
HWY_NOINLINE void UnaryExpD(const double *i, double *o, size_t n) {
    UnaryExpImplT(i, o, n);
}
HWY_NOINLINE void UnaryLogD(const double *i, double *o, size_t n) {
    UnaryLogImplT(i, o, n);
}
HWY_NOINLINE void UnarySinD(const double *i, double *o, size_t n) {
    UnarySinImplT(i, o, n);
}
HWY_NOINLINE void UnaryCosD(const double *i, double *o, size_t n) {
    UnaryCosImplT(i, o, n);
}
HWY_NOINLINE void UnaryTanhD(const double *i, double *o, size_t n) {
    UnaryTanhImplT(i, o, n);
}
HWY_NOINLINE void UnaryTanD(const double *i, double *o, size_t n) {
    UnaryTanImplT(i, o, n);
}
HWY_NOINLINE void UnaryErfD(const double *i, double *o, size_t n) {
    UnaryErfImplT(i, o, n);
}
HWY_NOINLINE void UnaryCbrtD(const double *i, double *o, size_t n) {
    UnaryCbrtImplT(i, o, n);
}
HWY_NOINLINE void UnarySquareD(const double *i, double *o, size_t n) {
    UnarySquareImplT(i, o, n);
}
HWY_NOINLINE void UnaryReciprocalD(const double *i, double *o, size_t n) {
    UnaryReciprocalImplT(i, o, n);
}
HWY_NOINLINE void UnarySignD(const double *i, double *o, size_t n) {
    UnarySignImplT(i, o, n);
}
HWY_NOINLINE void UnaryFloorD(const double *i, double *o, size_t n) {
    UnaryFloorImplT(i, o, n);
}
HWY_NOINLINE void UnaryCeilD(const double *i, double *o, size_t n) {
    UnaryCeilImplT(i, o, n);
}
HWY_NOINLINE void UnaryRoundD(const double *i, double *o, size_t n) {
    UnaryRoundImplT(i, o, n);
}
HWY_NOINLINE void UnaryTruncD(const double *i, double *o, size_t n) {
    UnaryTruncImplT(i, o, n);
}

// Reductions - float
HWY_NOINLINE float ReduceSumF(const float *d, size_t n) {
    return ReduceSumImplT(d, n);
}
HWY_NOINLINE float ReduceMaxF(const float *d, size_t n) {
    return ReduceMaxImplT(d, n);
}
HWY_NOINLINE float ReduceMinF(const float *d, size_t n) {
    return ReduceMinImplT(d, n);
}
HWY_NOINLINE float ReduceProdF(const float *d, size_t n) {
    return ReduceProdImplT(d, n);
}

// Reductions - double
HWY_NOINLINE double ReduceSumD(const double *d, size_t n) {
    return ReduceSumImplT(d, n);
}
HWY_NOINLINE double ReduceMaxD(const double *d, size_t n) {
    return ReduceMaxImplT(d, n);
}
HWY_NOINLINE double ReduceMinD(const double *d, size_t n) {
    return ReduceMinImplT(d, n);
}
HWY_NOINLINE double ReduceProdD(const double *d, size_t n) {
    return ReduceProdImplT(d, n);
}

// Activations - float
HWY_NOINLINE void ActivationReLUF(const float *i, float *o, size_t n) {
    ActivationReLUImplT(i, o, n);
}
HWY_NOINLINE void ActivationSigmoidF(const float *i, float *o, size_t n) {
    ActivationSigmoidImplT(i, o, n);
}
HWY_NOINLINE void ActivationGELUF(const float *i, float *o, size_t n) {
    ActivationGELUImplT(i, o, n);
}
HWY_NOINLINE void ActivationSiLUF(const float *i, float *o, size_t n) {
    ActivationSiLUImplT(i, o, n);
}
HWY_NOINLINE void ActivationLeakyReLUF(const float *i, float *o, size_t n,
                                       float a) {
    ActivationLeakyReLUImplT(i, o, n, a);
}

// Activations - double
HWY_NOINLINE void ActivationReLUD(const double *i, double *o, size_t n) {
    ActivationReLUImplT(i, o, n);
}
HWY_NOINLINE void ActivationSigmoidD(const double *i, double *o, size_t n) {
    ActivationSigmoidImplT(i, o, n);
}
HWY_NOINLINE void ActivationGELUD(const double *i, double *o, size_t n) {
    ActivationGELUImplT(i, o, n);
}
HWY_NOINLINE void ActivationSiLUD(const double *i, double *o, size_t n) {
    ActivationSiLUImplT(i, o, n);
}
HWY_NOINLINE void ActivationLeakyReLUD(const double *i, double *o, size_t n,
                                       double a) {
    ActivationLeakyReLUImplT(i, o, n, a);
}

} // namespace HWY_NAMESPACE
} // namespace simd
} // namespace axiom
HWY_AFTER_NAMESPACE();
