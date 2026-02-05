// Copyright 2024 Axiom Authors
// SPDX-License-Identifier: MIT
//
// Fused SIMD kernels for common operation patterns
// This file is included multiple times with different SIMD targets

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <cmath>

HWY_BEFORE_NAMESPACE();
namespace axiom {
namespace simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// ============================================================================
// Binary + Unary Fused Patterns - Float Templates
// ============================================================================

// AddReLU: relu(a + b) - common in neural networks
template <typename T>
HWY_NOINLINE void FusedAddReLUImpl(const T *HWY_RESTRICT a,
                                   const T *HWY_RESTRICT b,
                                   T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto sum = hn::Add(va, vb);
        hn::StoreU(hn::Max(sum, zero), d, result + i);
    }
    for (; i < n; ++i) {
        T sum = a[i] + b[i];
        result[i] = sum > T(0) ? sum : T(0);
    }
}

// MulReLU: relu(a * b)
template <typename T>
HWY_NOINLINE void FusedMulReLUImpl(const T *HWY_RESTRICT a,
                                   const T *HWY_RESTRICT b,
                                   T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto prod = hn::Mul(va, vb);
        hn::StoreU(hn::Max(prod, zero), d, result + i);
    }
    for (; i < n; ++i) {
        T prod = a[i] * b[i];
        result[i] = prod > T(0) ? prod : T(0);
    }
}

// SubAbs: abs(a - b) - common distance metric
template <typename T>
HWY_NOINLINE void FusedSubAbsImpl(const T *HWY_RESTRICT a,
                                  const T *HWY_RESTRICT b,
                                  T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto diff = hn::Sub(va, vb);
        hn::StoreU(hn::Abs(diff), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::abs(a[i] - b[i]);
    }
}

// AddSquare: (a + b)^2
template <typename T>
HWY_NOINLINE void FusedAddSquareImpl(const T *HWY_RESTRICT a,
                                     const T *HWY_RESTRICT b,
                                     T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto sum = hn::Add(va, vb);
        hn::StoreU(hn::Mul(sum, sum), d, result + i);
    }
    for (; i < n; ++i) {
        T sum = a[i] + b[i];
        result[i] = sum * sum;
    }
}

// SubSquare: (a - b)^2 - common in loss functions
template <typename T>
HWY_NOINLINE void FusedSubSquareImpl(const T *HWY_RESTRICT a,
                                     const T *HWY_RESTRICT b,
                                     T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto diff = hn::Sub(va, vb);
        hn::StoreU(hn::Mul(diff, diff), d, result + i);
    }
    for (; i < n; ++i) {
        T diff = a[i] - b[i];
        result[i] = diff * diff;
    }
}

// AddSigmoid: sigmoid(a + b)
template <typename T>
HWY_NOINLINE void FusedAddSigmoidImpl(const T *HWY_RESTRICT a,
                                      const T *HWY_RESTRICT b,
                                      T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, T(1));
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto sum = hn::Add(va, vb);
        const auto neg_sum = hn::Neg(sum);
        const auto exp_neg = hn::Exp(d, neg_sum);
        const auto sigmoid = hn::Div(one, hn::Add(one, exp_neg));
        hn::StoreU(sigmoid, d, result + i);
    }
    for (; i < n; ++i) {
        T sum = a[i] + b[i];
        result[i] = T(1) / (T(1) + std::exp(-sum));
    }
}

// MulSigmoid: sigmoid(a * b)
template <typename T>
HWY_NOINLINE void FusedMulSigmoidImpl(const T *HWY_RESTRICT a,
                                      const T *HWY_RESTRICT b,
                                      T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto one = hn::Set(d, T(1));
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto prod = hn::Mul(va, vb);
        const auto neg_prod = hn::Neg(prod);
        const auto exp_neg = hn::Exp(d, neg_prod);
        const auto sigmoid = hn::Div(one, hn::Add(one, exp_neg));
        hn::StoreU(sigmoid, d, result + i);
    }
    for (; i < n; ++i) {
        T prod = a[i] * b[i];
        result[i] = T(1) / (T(1) + std::exp(-prod));
    }
}

// MulAdd: a * b + c (FMA)
template <typename T>
HWY_NOINLINE void
FusedMulAddImpl(const T *HWY_RESTRICT a, const T *HWY_RESTRICT b,
                const T *HWY_RESTRICT c, T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        hn::StoreU(hn::MulAdd(va, vb, vc), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// MulSub: a * b - c
template <typename T>
HWY_NOINLINE void
FusedMulSubImpl(const T *HWY_RESTRICT a, const T *HWY_RESTRICT b,
                const T *HWY_RESTRICT c, T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        hn::StoreU(hn::MulSub(va, vb, vc), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * b[i] - c[i];
    }
}

// ScaleShiftReLU: relu(a * scale + bias)
template <typename T>
HWY_NOINLINE void FusedScaleShiftReLUImpl(const T *HWY_RESTRICT a,
                                          const T *HWY_RESTRICT scale,
                                          const T *HWY_RESTRICT bias,
                                          T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vs = hn::LoadU(d, scale + i);
        const auto vb = hn::LoadU(d, bias + i);
        const auto scaled = hn::MulAdd(va, vs, vb);
        hn::StoreU(hn::Max(scaled, zero), d, result + i);
    }
    for (; i < n; ++i) {
        T val = a[i] * scale[i] + bias[i];
        result[i] = val > T(0) ? val : T(0);
    }
}

// AddMulReLU: relu((a + b) * c)
template <typename T>
HWY_NOINLINE void
FusedAddMulReLUImpl(const T *HWY_RESTRICT a, const T *HWY_RESTRICT b,
                    const T *HWY_RESTRICT c, T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto sum = hn::Add(va, vb);
        const auto prod = hn::Mul(sum, vc);
        hn::StoreU(hn::Max(prod, zero), d, result + i);
    }
    for (; i < n; ++i) {
        T val = (a[i] + b[i]) * c[i];
        result[i] = val > T(0) ? val : T(0);
    }
}

// SubMulAbs: |((a - b) * c)|
template <typename T>
HWY_NOINLINE void
FusedSubMulAbsImpl(const T *HWY_RESTRICT a, const T *HWY_RESTRICT b,
                   const T *HWY_RESTRICT c, T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto diff = hn::Sub(va, vb);
        const auto prod = hn::Mul(diff, vc);
        hn::StoreU(hn::Abs(prod), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::abs((a[i] - b[i]) * c[i]);
    }
}

// ============================================================================
// Integer SIMD Kernels
// ============================================================================

// Integer AddReLU: max(a + b, 0) - uses saturating add where available
template <typename T>
HWY_NOINLINE void FusedAddReLUIntImpl(const T *HWY_RESTRICT a,
                                      const T *HWY_RESTRICT b,
                                      T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto sum = hn::Add(va, vb);
        hn::StoreU(hn::Max(sum, zero), d, result + i);
    }
    for (; i < n; ++i) {
        T sum = a[i] + b[i];
        result[i] = sum > T(0) ? sum : T(0);
    }
}

// Integer SubAbs: abs(a - b)
template <typename T>
HWY_NOINLINE void FusedSubAbsIntImpl(const T *HWY_RESTRICT a,
                                     const T *HWY_RESTRICT b,
                                     T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto diff = hn::Sub(va, vb);
        hn::StoreU(hn::Abs(diff), d, result + i);
    }
    for (; i < n; ++i) {
        T diff = a[i] - b[i];
        result[i] = diff >= 0 ? diff : -diff;
    }
}

// Integer MulAdd: a * b + c
template <typename T>
HWY_NOINLINE void
FusedMulAddIntImpl(const T *HWY_RESTRICT a, const T *HWY_RESTRICT b,
                   const T *HWY_RESTRICT c, T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto vc = hn::LoadU(d, c + i);
        const auto prod = hn::Mul(va, vb);
        hn::StoreU(hn::Add(prod, vc), d, result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// Integer AddSquare: (a + b)^2
template <typename T>
HWY_NOINLINE void FusedAddSquareIntImpl(const T *HWY_RESTRICT a,
                                        const T *HWY_RESTRICT b,
                                        T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto sum = hn::Add(va, vb);
        hn::StoreU(hn::Mul(sum, sum), d, result + i);
    }
    for (; i < n; ++i) {
        T sum = a[i] + b[i];
        result[i] = sum * sum;
    }
}

// Integer SubSquare: (a - b)^2
template <typename T>
HWY_NOINLINE void FusedSubSquareIntImpl(const T *HWY_RESTRICT a,
                                        const T *HWY_RESTRICT b,
                                        T *HWY_RESTRICT result, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t N = hn::Lanes(d);
    size_t i = 0;

    for (; i + N <= n; i += N) {
        const auto va = hn::LoadU(d, a + i);
        const auto vb = hn::LoadU(d, b + i);
        const auto diff = hn::Sub(va, vb);
        hn::StoreU(hn::Mul(diff, diff), d, result + i);
    }
    for (; i < n; ++i) {
        T diff = a[i] - b[i];
        result[i] = diff * diff;
    }
}

// ============================================================================
// Non-template wrappers for HWY_EXPORT
// ============================================================================

// Float32 wrappers
HWY_NOINLINE void FusedAddReLUF(const float *a, const float *b, float *result,
                                size_t n) {
    FusedAddReLUImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubAbsF(const float *a, const float *b, float *result,
                               size_t n) {
    FusedSubAbsImpl(a, b, result, n);
}

HWY_NOINLINE void FusedAddSquareF(const float *a, const float *b, float *result,
                                  size_t n) {
    FusedAddSquareImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulAddF(const float *a, const float *b, const float *c,
                               float *result, size_t n) {
    FusedMulAddImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedScaleShiftReLUF(const float *a, const float *scale,
                                       const float *bias, float *result,
                                       size_t n) {
    FusedScaleShiftReLUImpl(a, scale, bias, result, n);
}

// Float64 wrappers
HWY_NOINLINE void FusedAddReLUD(const double *a, const double *b,
                                double *result, size_t n) {
    FusedAddReLUImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubAbsD(const double *a, const double *b, double *result,
                               size_t n) {
    FusedSubAbsImpl(a, b, result, n);
}

HWY_NOINLINE void FusedAddSquareD(const double *a, const double *b,
                                  double *result, size_t n) {
    FusedAddSquareImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulAddD(const double *a, const double *b,
                               const double *c, double *result, size_t n) {
    FusedMulAddImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedScaleShiftReLUD(const double *a, const double *scale,
                                       const double *bias, double *result,
                                       size_t n) {
    FusedScaleShiftReLUImpl(a, scale, bias, result, n);
}

// New Float32 wrappers
HWY_NOINLINE void FusedMulReLUF(const float *a, const float *b, float *result,
                                size_t n) {
    FusedMulReLUImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubSquareF(const float *a, const float *b, float *result,
                                  size_t n) {
    FusedSubSquareImpl(a, b, result, n);
}

HWY_NOINLINE void FusedAddSigmoidF(const float *a, const float *b,
                                   float *result, size_t n) {
    FusedAddSigmoidImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulSigmoidF(const float *a, const float *b,
                                   float *result, size_t n) {
    FusedMulSigmoidImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulSubF(const float *a, const float *b, const float *c,
                               float *result, size_t n) {
    FusedMulSubImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedAddMulReLUF(const float *a, const float *b,
                                   const float *c, float *result, size_t n) {
    FusedAddMulReLUImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedSubMulAbsF(const float *a, const float *b,
                                  const float *c, float *result, size_t n) {
    FusedSubMulAbsImpl(a, b, c, result, n);
}

// New Float64 wrappers
HWY_NOINLINE void FusedMulReLUD(const double *a, const double *b,
                                double *result, size_t n) {
    FusedMulReLUImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubSquareD(const double *a, const double *b,
                                  double *result, size_t n) {
    FusedSubSquareImpl(a, b, result, n);
}

HWY_NOINLINE void FusedAddSigmoidD(const double *a, const double *b,
                                   double *result, size_t n) {
    FusedAddSigmoidImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulSigmoidD(const double *a, const double *b,
                                   double *result, size_t n) {
    FusedMulSigmoidImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulSubD(const double *a, const double *b,
                               const double *c, double *result, size_t n) {
    FusedMulSubImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedAddMulReLUD(const double *a, const double *b,
                                   const double *c, double *result, size_t n) {
    FusedAddMulReLUImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedSubMulAbsD(const double *a, const double *b,
                                  const double *c, double *result, size_t n) {
    FusedSubMulAbsImpl(a, b, c, result, n);
}

// Int32 wrappers
HWY_NOINLINE void FusedAddReLUI32(const int32_t *a, const int32_t *b,
                                  int32_t *result, size_t n) {
    FusedAddReLUIntImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubAbsI32(const int32_t *a, const int32_t *b,
                                 int32_t *result, size_t n) {
    FusedSubAbsIntImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulAddI32(const int32_t *a, const int32_t *b,
                                 const int32_t *c, int32_t *result, size_t n) {
    FusedMulAddIntImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedAddSquareI32(const int32_t *a, const int32_t *b,
                                    int32_t *result, size_t n) {
    FusedAddSquareIntImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubSquareI32(const int32_t *a, const int32_t *b,
                                    int32_t *result, size_t n) {
    FusedSubSquareIntImpl(a, b, result, n);
}

// Int64 wrappers
HWY_NOINLINE void FusedAddReLUI64(const int64_t *a, const int64_t *b,
                                  int64_t *result, size_t n) {
    FusedAddReLUIntImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubAbsI64(const int64_t *a, const int64_t *b,
                                 int64_t *result, size_t n) {
    FusedSubAbsIntImpl(a, b, result, n);
}

HWY_NOINLINE void FusedMulAddI64(const int64_t *a, const int64_t *b,
                                 const int64_t *c, int64_t *result, size_t n) {
    FusedMulAddIntImpl(a, b, c, result, n);
}

HWY_NOINLINE void FusedAddSquareI64(const int64_t *a, const int64_t *b,
                                    int64_t *result, size_t n) {
    FusedAddSquareIntImpl(a, b, result, n);
}

HWY_NOINLINE void FusedSubSquareI64(const int64_t *a, const int64_t *b,
                                    int64_t *result, size_t n) {
    FusedSubSquareIntImpl(a, b, result, n);
}

} // namespace HWY_NAMESPACE
} // namespace simd
} // namespace axiom
HWY_AFTER_NAMESPACE();
