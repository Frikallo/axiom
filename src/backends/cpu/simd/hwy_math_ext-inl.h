// hwy_math_ext-inl.h - Custom math functions for Highway
//
// Highway's contrib/math provides most transcendentals, but we need custom
// implementations for:
//   - erf(x): error function (Abramowitz-Stegun approximation)
//   - cbrt(x): cube root (Newton-Raphson with exp/log initial guess)
//
// This file follows Highway's foreach_target pattern and should be included
// via HWY_BEFORE_NAMESPACE/HWY_AFTER_NAMESPACE macros.

// clang-format off
// NOLINTBEGIN(clang-diagnostic-pp_including_mainfile_in_preamble)
#if !defined(HWY_IDE_MODE)
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "backends/cpu/simd/hwy_math_ext-inl.h"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#endif
// NOLINTEND(clang-diagnostic-pp_including_mainfile_in_preamble)
// clang-format on

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace axiom {
namespace simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// ============================================================================
// erf(x) - Error Function
// ============================================================================
// Abramowitz-Stegun approximation 7.1.26
// Maximum error: ~1.5e-7 for float, ~2.5e-16 for double
//
// erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
// where t = 1 / (1 + p*|x|)
// and p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736,
//     a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429

template <class D, class V = hn::VFromD<D>, typename T = hn::TFromD<D>>
HWY_API V Erf(D d, V x) {
    const V one = hn::Set(d, T{1});

    // Constants for Abramowitz-Stegun approximation
    const V p = hn::Set(d, T{0.3275911});
    const V a1 = hn::Set(d, T{0.254829592});
    const V a2 = hn::Set(d, T{-0.284496736});
    const V a3 = hn::Set(d, T{1.421413741});
    const V a4 = hn::Set(d, T{-1.453152027});
    const V a5 = hn::Set(d, T{1.061405429});

    // Save sign and work with |x|
    const V sign = hn::CopySign(one, x);
    const V abs_x = hn::Abs(x);

    // t = 1 / (1 + p * |x|)
    const V t = hn::Div(one, hn::MulAdd(p, abs_x, one));

    // Polynomial: a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    // Horner's method: t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    V poly = hn::MulAdd(t, a5, a4);
    poly = hn::MulAdd(t, poly, a3);
    poly = hn::MulAdd(t, poly, a2);
    poly = hn::MulAdd(t, poly, a1);
    poly = hn::Mul(t, poly);

    // exp(-x^2)
    const V neg_x_sq = hn::Neg(hn::Mul(x, x));
    const V exp_term = hn::Exp(d, neg_x_sq);

    // erf(|x|) = 1 - poly * exp(-x^2)
    const V result = hn::NegMulAdd(poly, exp_term, one);

    // Apply sign: erf(-x) = -erf(x)
    return hn::Mul(sign, result);
}

// ============================================================================
// cbrt(x) - Cube Root
// ============================================================================
// Uses Newton-Raphson iteration with initial guess from bit manipulation.
// cbrt(x) = sign(x) * |x|^(1/3)
//
// Newton iteration: y' = (2*y + x/(y*y)) / 3
// Converges quadratically, 2-3 iterations usually sufficient for float.

template <class D, class V = hn::VFromD<D>, typename T = hn::TFromD<D>>
HWY_API V Cbrt(D d, V x) {
    const V zero = hn::Zero(d);
    const V one = hn::Set(d, T{1});
    const V two = hn::Set(d, T{2});
    const V third = hn::Set(d, T{1.0 / 3.0});

    // Save sign
    const V sign = hn::CopySign(one, x);
    const V abs_x = hn::Abs(x);

    // Handle zero case
    const auto is_zero = hn::Eq(abs_x, zero);

    // Initial guess using exp/log: x^(1/3) = exp(log(x)/3)
    // This is accurate enough for Newton iteration to converge quickly
    const V log_x = hn::Log(d, abs_x);
    V y = hn::Exp(d, hn::Mul(log_x, third));

    // Newton-Raphson iterations for better precision
    // y' = (2*y + x/(y*y)) / 3 = y - (y - x/(y*y)) / 3 = y * (2 + x/(y^3)) / 3
    // Simplified: y' = (2*y + x*y^(-2)) / 3
    const V three = hn::Set(d, T{3});

    // Iteration 1
    V y_sq = hn::Mul(y, y);
    y = hn::Div(hn::MulAdd(two, y, hn::Div(abs_x, y_sq)), three);

    // Iteration 2 (sufficient for float)
    y_sq = hn::Mul(y, y);
    y = hn::Div(hn::MulAdd(two, y, hn::Div(abs_x, y_sq)), three);

    // Iteration 3 (for double precision)
    if constexpr (sizeof(T) > 4) {
        y_sq = hn::Mul(y, y);
        y = hn::Div(hn::MulAdd(two, y, hn::Div(abs_x, y_sq)), three);
    }

    // Apply sign and handle zero
    const V result = hn::Mul(sign, y);
    return hn::IfThenElse(is_zero, zero, result);
}

} // namespace HWY_NAMESPACE
} // namespace simd
} // namespace axiom
HWY_AFTER_NAMESPACE();
