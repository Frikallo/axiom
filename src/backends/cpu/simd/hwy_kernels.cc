// hwy_kernels.cc - Highway SIMD kernel exports
//
// This file includes hwy_kernels-inl.h which gets compiled for each target
// architecture via HWY_FOREACH_TARGET. Then we export the dispatch functions
// that select the best implementation at runtime.

#include "backends/cpu/simd/hwy_kernels-inl.h"

// Include once after foreach_target for HWY_EXPORT macros
#if HWY_ONCE

// Include dispatch header to get template declarations
#include "backends/cpu/simd/simd_dispatch.hpp"

namespace axiom {
namespace simd {

// ============================================================================
// HWY_EXPORT declarations for dynamic dispatch
// ============================================================================

// Binary operations - float
HWY_EXPORT(BinaryAddF);
HWY_EXPORT(BinarySubF);
HWY_EXPORT(BinaryMulF);
HWY_EXPORT(BinaryDivF);
HWY_EXPORT(BinaryMaxF);
HWY_EXPORT(BinaryMinF);
HWY_EXPORT(BinaryPowF);
HWY_EXPORT(BinaryAtan2F);
HWY_EXPORT(BinaryHypotF);
HWY_EXPORT(BinaryFmodF);

// Binary operations - double
HWY_EXPORT(BinaryAddD);
HWY_EXPORT(BinarySubD);
HWY_EXPORT(BinaryMulD);
HWY_EXPORT(BinaryDivD);
HWY_EXPORT(BinaryMaxD);
HWY_EXPORT(BinaryMinD);
HWY_EXPORT(BinaryPowD);
HWY_EXPORT(BinaryAtan2D);
HWY_EXPORT(BinaryHypotD);
HWY_EXPORT(BinaryFmodD);

// Unary operations - float
HWY_EXPORT(UnaryNegF);
HWY_EXPORT(UnaryAbsF);
HWY_EXPORT(UnarySqrtF);
HWY_EXPORT(UnaryExpF);
HWY_EXPORT(UnaryLogF);
HWY_EXPORT(UnarySinF);
HWY_EXPORT(UnaryCosF);
HWY_EXPORT(UnaryTanhF);
HWY_EXPORT(UnaryTanF);
HWY_EXPORT(UnaryErfF);
HWY_EXPORT(UnaryCbrtF);
HWY_EXPORT(UnarySquareF);
HWY_EXPORT(UnaryReciprocalF);
HWY_EXPORT(UnarySignF);
HWY_EXPORT(UnaryFloorF);
HWY_EXPORT(UnaryCeilF);
HWY_EXPORT(UnaryRoundF);
HWY_EXPORT(UnaryTruncF);

// Unary operations - double
HWY_EXPORT(UnaryNegD);
HWY_EXPORT(UnaryAbsD);
HWY_EXPORT(UnarySqrtD);
HWY_EXPORT(UnaryExpD);
HWY_EXPORT(UnaryLogD);
HWY_EXPORT(UnarySinD);
HWY_EXPORT(UnaryCosD);
HWY_EXPORT(UnaryTanhD);
HWY_EXPORT(UnaryTanD);
HWY_EXPORT(UnaryErfD);
HWY_EXPORT(UnaryCbrtD);
HWY_EXPORT(UnarySquareD);
HWY_EXPORT(UnaryReciprocalD);
HWY_EXPORT(UnarySignD);
HWY_EXPORT(UnaryFloorD);
HWY_EXPORT(UnaryCeilD);
HWY_EXPORT(UnaryRoundD);
HWY_EXPORT(UnaryTruncD);

// Reductions - float
HWY_EXPORT(ReduceSumF);
HWY_EXPORT(ReduceMaxF);
HWY_EXPORT(ReduceMinF);
HWY_EXPORT(ReduceProdF);

// Reductions - double
HWY_EXPORT(ReduceSumD);
HWY_EXPORT(ReduceMaxD);
HWY_EXPORT(ReduceMinD);
HWY_EXPORT(ReduceProdD);

// Activations - float
HWY_EXPORT(ActivationReLUF);
HWY_EXPORT(ActivationSigmoidF);
HWY_EXPORT(ActivationGELUF);
HWY_EXPORT(ActivationSiLUF);
HWY_EXPORT(ActivationLeakyReLUF);

// Activations - double
HWY_EXPORT(ActivationReLUD);
HWY_EXPORT(ActivationSigmoidD);
HWY_EXPORT(ActivationGELUD);
HWY_EXPORT(ActivationSiLUD);
HWY_EXPORT(ActivationLeakyReLUD);

// ============================================================================
// Public dispatch functions (non-template, with explicit type suffixes)
// ============================================================================

void dispatch_binary_add_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryAddF)(a, b, result, n);
}

void dispatch_binary_add_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryAddD)(a, b, result, n);
}

void dispatch_binary_sub_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinarySubF)(a, b, result, n);
}

void dispatch_binary_sub_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinarySubD)(a, b, result, n);
}

void dispatch_binary_mul_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryMulF)(a, b, result, n);
}

void dispatch_binary_mul_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryMulD)(a, b, result, n);
}

void dispatch_binary_div_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryDivF)(a, b, result, n);
}

void dispatch_binary_div_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryDivD)(a, b, result, n);
}

void dispatch_binary_max_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryMaxF)(a, b, result, n);
}

void dispatch_binary_max_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryMaxD)(a, b, result, n);
}

void dispatch_binary_min_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryMinF)(a, b, result, n);
}

void dispatch_binary_min_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryMinD)(a, b, result, n);
}

void dispatch_binary_pow_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryPowF)(a, b, result, n);
}

void dispatch_binary_pow_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryPowD)(a, b, result, n);
}

void dispatch_binary_atan2_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryAtan2F)(a, b, result, n);
}

void dispatch_binary_atan2_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryAtan2D)(a, b, result, n);
}

void dispatch_binary_hypot_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryHypotF)(a, b, result, n);
}

void dispatch_binary_hypot_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryHypotD)(a, b, result, n);
}

void dispatch_binary_fmod_f(const float* a, const float* b, float* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryFmodF)(a, b, result, n);
}

void dispatch_binary_fmod_d(const double* a, const double* b, double* result, size_t n) {
    HWY_DYNAMIC_DISPATCH(BinaryFmodD)(a, b, result, n);
}

// Unary operations
void dispatch_unary_neg_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryNegF)(input, output, n);
}

void dispatch_unary_neg_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryNegD)(input, output, n);
}

void dispatch_unary_abs_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryAbsF)(input, output, n);
}

void dispatch_unary_abs_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryAbsD)(input, output, n);
}

void dispatch_unary_sqrt_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySqrtF)(input, output, n);
}

void dispatch_unary_sqrt_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySqrtD)(input, output, n);
}

void dispatch_unary_exp_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryExpF)(input, output, n);
}

void dispatch_unary_exp_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryExpD)(input, output, n);
}

void dispatch_unary_log_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryLogF)(input, output, n);
}

void dispatch_unary_log_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryLogD)(input, output, n);
}

void dispatch_unary_sin_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySinF)(input, output, n);
}

void dispatch_unary_sin_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySinD)(input, output, n);
}

void dispatch_unary_cos_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryCosF)(input, output, n);
}

void dispatch_unary_cos_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryCosD)(input, output, n);
}

void dispatch_unary_tanh_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryTanhF)(input, output, n);
}

void dispatch_unary_tanh_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryTanhD)(input, output, n);
}

void dispatch_unary_tan_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryTanF)(input, output, n);
}

void dispatch_unary_tan_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryTanD)(input, output, n);
}

void dispatch_unary_erf_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryErfF)(input, output, n);
}

void dispatch_unary_erf_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryErfD)(input, output, n);
}

void dispatch_unary_cbrt_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryCbrtF)(input, output, n);
}

void dispatch_unary_cbrt_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryCbrtD)(input, output, n);
}

void dispatch_unary_square_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySquareF)(input, output, n);
}

void dispatch_unary_square_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySquareD)(input, output, n);
}

void dispatch_unary_reciprocal_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryReciprocalF)(input, output, n);
}

void dispatch_unary_reciprocal_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryReciprocalD)(input, output, n);
}

void dispatch_unary_sign_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySignF)(input, output, n);
}

void dispatch_unary_sign_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnarySignD)(input, output, n);
}

void dispatch_unary_floor_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryFloorF)(input, output, n);
}

void dispatch_unary_floor_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryFloorD)(input, output, n);
}

void dispatch_unary_ceil_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryCeilF)(input, output, n);
}

void dispatch_unary_ceil_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryCeilD)(input, output, n);
}

void dispatch_unary_round_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryRoundF)(input, output, n);
}

void dispatch_unary_round_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryRoundD)(input, output, n);
}

void dispatch_unary_trunc_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryTruncF)(input, output, n);
}

void dispatch_unary_trunc_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(UnaryTruncD)(input, output, n);
}

// Reductions
float dispatch_reduce_sum_f(const float* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceSumF)(data, n);
}

double dispatch_reduce_sum_d(const double* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceSumD)(data, n);
}

float dispatch_reduce_max_f(const float* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceMaxF)(data, n);
}

double dispatch_reduce_max_d(const double* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceMaxD)(data, n);
}

float dispatch_reduce_min_f(const float* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceMinF)(data, n);
}

double dispatch_reduce_min_d(const double* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceMinD)(data, n);
}

float dispatch_reduce_prod_f(const float* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceProdF)(data, n);
}

double dispatch_reduce_prod_d(const double* data, size_t n) {
    return HWY_DYNAMIC_DISPATCH(ReduceProdD)(data, n);
}

// Activations
void dispatch_activation_relu_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationReLUF)(input, output, n);
}

void dispatch_activation_relu_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationReLUD)(input, output, n);
}

void dispatch_activation_sigmoid_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationSigmoidF)(input, output, n);
}

void dispatch_activation_sigmoid_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationSigmoidD)(input, output, n);
}

void dispatch_activation_gelu_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationGELUF)(input, output, n);
}

void dispatch_activation_gelu_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationGELUD)(input, output, n);
}

void dispatch_activation_silu_f(const float* input, float* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationSiLUF)(input, output, n);
}

void dispatch_activation_silu_d(const double* input, double* output, size_t n) {
    HWY_DYNAMIC_DISPATCH(ActivationSiLUD)(input, output, n);
}

void dispatch_activation_leaky_relu_f(const float* input, float* output, size_t n, float alpha) {
    HWY_DYNAMIC_DISPATCH(ActivationLeakyReLUF)(input, output, n, alpha);
}

void dispatch_activation_leaky_relu_d(const double* input, double* output, size_t n, double alpha) {
    HWY_DYNAMIC_DISPATCH(ActivationLeakyReLUD)(input, output, n, alpha);
}

// ============================================================================
// Template dispatch wrappers (call the type-specific functions)
// ============================================================================

template <>
void dispatch_binary_add<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_add_f(a, b, result, n);
}

template <>
void dispatch_binary_add<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_add_d(a, b, result, n);
}

template <>
void dispatch_binary_sub<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_sub_f(a, b, result, n);
}

template <>
void dispatch_binary_sub<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_sub_d(a, b, result, n);
}

template <>
void dispatch_binary_mul<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_mul_f(a, b, result, n);
}

template <>
void dispatch_binary_mul<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_mul_d(a, b, result, n);
}

template <>
void dispatch_binary_div<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_div_f(a, b, result, n);
}

template <>
void dispatch_binary_div<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_div_d(a, b, result, n);
}

template <>
void dispatch_binary_max<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_max_f(a, b, result, n);
}

template <>
void dispatch_binary_max<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_max_d(a, b, result, n);
}

template <>
void dispatch_binary_min<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_min_f(a, b, result, n);
}

template <>
void dispatch_binary_min<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_min_d(a, b, result, n);
}

template <>
void dispatch_binary_pow<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_pow_f(a, b, result, n);
}

template <>
void dispatch_binary_pow<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_pow_d(a, b, result, n);
}

template <>
void dispatch_binary_atan2<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_atan2_f(a, b, result, n);
}

template <>
void dispatch_binary_atan2<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_atan2_d(a, b, result, n);
}

template <>
void dispatch_binary_hypot<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_hypot_f(a, b, result, n);
}

template <>
void dispatch_binary_hypot<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_hypot_d(a, b, result, n);
}

template <>
void dispatch_binary_fmod<float>(const float* a, const float* b, float* result, size_t n) {
    dispatch_binary_fmod_f(a, b, result, n);
}

template <>
void dispatch_binary_fmod<double>(const double* a, const double* b, double* result, size_t n) {
    dispatch_binary_fmod_d(a, b, result, n);
}

// Unary operations
template <>
void dispatch_unary_neg<float>(const float* input, float* output, size_t n) {
    dispatch_unary_neg_f(input, output, n);
}

template <>
void dispatch_unary_neg<double>(const double* input, double* output, size_t n) {
    dispatch_unary_neg_d(input, output, n);
}

template <>
void dispatch_unary_abs<float>(const float* input, float* output, size_t n) {
    dispatch_unary_abs_f(input, output, n);
}

template <>
void dispatch_unary_abs<double>(const double* input, double* output, size_t n) {
    dispatch_unary_abs_d(input, output, n);
}

template <>
void dispatch_unary_sqrt<float>(const float* input, float* output, size_t n) {
    dispatch_unary_sqrt_f(input, output, n);
}

template <>
void dispatch_unary_sqrt<double>(const double* input, double* output, size_t n) {
    dispatch_unary_sqrt_d(input, output, n);
}

template <>
void dispatch_unary_exp<float>(const float* input, float* output, size_t n) {
    dispatch_unary_exp_f(input, output, n);
}

template <>
void dispatch_unary_exp<double>(const double* input, double* output, size_t n) {
    dispatch_unary_exp_d(input, output, n);
}

template <>
void dispatch_unary_log<float>(const float* input, float* output, size_t n) {
    dispatch_unary_log_f(input, output, n);
}

template <>
void dispatch_unary_log<double>(const double* input, double* output, size_t n) {
    dispatch_unary_log_d(input, output, n);
}

template <>
void dispatch_unary_sin<float>(const float* input, float* output, size_t n) {
    dispatch_unary_sin_f(input, output, n);
}

template <>
void dispatch_unary_sin<double>(const double* input, double* output, size_t n) {
    dispatch_unary_sin_d(input, output, n);
}

template <>
void dispatch_unary_cos<float>(const float* input, float* output, size_t n) {
    dispatch_unary_cos_f(input, output, n);
}

template <>
void dispatch_unary_cos<double>(const double* input, double* output, size_t n) {
    dispatch_unary_cos_d(input, output, n);
}

template <>
void dispatch_unary_tanh<float>(const float* input, float* output, size_t n) {
    dispatch_unary_tanh_f(input, output, n);
}

template <>
void dispatch_unary_tanh<double>(const double* input, double* output, size_t n) {
    dispatch_unary_tanh_d(input, output, n);
}

template <>
void dispatch_unary_tan<float>(const float* input, float* output, size_t n) {
    dispatch_unary_tan_f(input, output, n);
}

template <>
void dispatch_unary_tan<double>(const double* input, double* output, size_t n) {
    dispatch_unary_tan_d(input, output, n);
}

template <>
void dispatch_unary_erf<float>(const float* input, float* output, size_t n) {
    dispatch_unary_erf_f(input, output, n);
}

template <>
void dispatch_unary_erf<double>(const double* input, double* output, size_t n) {
    dispatch_unary_erf_d(input, output, n);
}

template <>
void dispatch_unary_cbrt<float>(const float* input, float* output, size_t n) {
    dispatch_unary_cbrt_f(input, output, n);
}

template <>
void dispatch_unary_cbrt<double>(const double* input, double* output, size_t n) {
    dispatch_unary_cbrt_d(input, output, n);
}

template <>
void dispatch_unary_square<float>(const float* input, float* output, size_t n) {
    dispatch_unary_square_f(input, output, n);
}

template <>
void dispatch_unary_square<double>(const double* input, double* output, size_t n) {
    dispatch_unary_square_d(input, output, n);
}

template <>
void dispatch_unary_reciprocal<float>(const float* input, float* output, size_t n) {
    dispatch_unary_reciprocal_f(input, output, n);
}

template <>
void dispatch_unary_reciprocal<double>(const double* input, double* output, size_t n) {
    dispatch_unary_reciprocal_d(input, output, n);
}

template <>
void dispatch_unary_sign<float>(const float* input, float* output, size_t n) {
    dispatch_unary_sign_f(input, output, n);
}

template <>
void dispatch_unary_sign<double>(const double* input, double* output, size_t n) {
    dispatch_unary_sign_d(input, output, n);
}

template <>
void dispatch_unary_floor<float>(const float* input, float* output, size_t n) {
    dispatch_unary_floor_f(input, output, n);
}

template <>
void dispatch_unary_floor<double>(const double* input, double* output, size_t n) {
    dispatch_unary_floor_d(input, output, n);
}

template <>
void dispatch_unary_ceil<float>(const float* input, float* output, size_t n) {
    dispatch_unary_ceil_f(input, output, n);
}

template <>
void dispatch_unary_ceil<double>(const double* input, double* output, size_t n) {
    dispatch_unary_ceil_d(input, output, n);
}

template <>
void dispatch_unary_round<float>(const float* input, float* output, size_t n) {
    dispatch_unary_round_f(input, output, n);
}

template <>
void dispatch_unary_round<double>(const double* input, double* output, size_t n) {
    dispatch_unary_round_d(input, output, n);
}

template <>
void dispatch_unary_trunc<float>(const float* input, float* output, size_t n) {
    dispatch_unary_trunc_f(input, output, n);
}

template <>
void dispatch_unary_trunc<double>(const double* input, double* output, size_t n) {
    dispatch_unary_trunc_d(input, output, n);
}

// Reductions
template <>
float dispatch_reduce_sum<float>(const float* data, size_t n) {
    return dispatch_reduce_sum_f(data, n);
}

template <>
double dispatch_reduce_sum<double>(const double* data, size_t n) {
    return dispatch_reduce_sum_d(data, n);
}

template <>
float dispatch_reduce_max<float>(const float* data, size_t n) {
    return dispatch_reduce_max_f(data, n);
}

template <>
double dispatch_reduce_max<double>(const double* data, size_t n) {
    return dispatch_reduce_max_d(data, n);
}

template <>
float dispatch_reduce_min<float>(const float* data, size_t n) {
    return dispatch_reduce_min_f(data, n);
}

template <>
double dispatch_reduce_min<double>(const double* data, size_t n) {
    return dispatch_reduce_min_d(data, n);
}

template <>
float dispatch_reduce_prod<float>(const float* data, size_t n) {
    return dispatch_reduce_prod_f(data, n);
}

template <>
double dispatch_reduce_prod<double>(const double* data, size_t n) {
    return dispatch_reduce_prod_d(data, n);
}

// Activations
template <>
void dispatch_activation_relu<float>(const float* input, float* output, size_t n) {
    dispatch_activation_relu_f(input, output, n);
}

template <>
void dispatch_activation_relu<double>(const double* input, double* output, size_t n) {
    dispatch_activation_relu_d(input, output, n);
}

template <>
void dispatch_activation_sigmoid<float>(const float* input, float* output, size_t n) {
    dispatch_activation_sigmoid_f(input, output, n);
}

template <>
void dispatch_activation_sigmoid<double>(const double* input, double* output, size_t n) {
    dispatch_activation_sigmoid_d(input, output, n);
}

template <>
void dispatch_activation_gelu<float>(const float* input, float* output, size_t n) {
    dispatch_activation_gelu_f(input, output, n);
}

template <>
void dispatch_activation_gelu<double>(const double* input, double* output, size_t n) {
    dispatch_activation_gelu_d(input, output, n);
}

template <>
void dispatch_activation_silu<float>(const float* input, float* output, size_t n) {
    dispatch_activation_silu_f(input, output, n);
}

template <>
void dispatch_activation_silu<double>(const double* input, double* output, size_t n) {
    dispatch_activation_silu_d(input, output, n);
}

template <>
void dispatch_activation_leaky_relu<float>(const float* input, float* output, size_t n, double alpha) {
    dispatch_activation_leaky_relu_f(input, output, n, static_cast<float>(alpha));
}

template <>
void dispatch_activation_leaky_relu<double>(const double* input, double* output, size_t n, double alpha) {
    dispatch_activation_leaky_relu_d(input, output, n, alpha);
}

// ============================================================================
// Integer type support (scalar fallback - SIMD dispatch uses float/double)
// ============================================================================

// Helper macros for scalar integer loops
#define SCALAR_BINARY_OP(name, op) \
    template <> \
    void dispatch_binary_##name<int32_t>(const int32_t* a, const int32_t* b, int32_t* result, size_t n) { \
        for (size_t i = 0; i < n; ++i) result[i] = op; \
    } \
    template <> \
    void dispatch_binary_##name<int64_t>(const int64_t* a, const int64_t* b, int64_t* result, size_t n) { \
        for (size_t i = 0; i < n; ++i) result[i] = op; \
    }

SCALAR_BINARY_OP(add, a[i] + b[i])
SCALAR_BINARY_OP(sub, a[i] - b[i])
SCALAR_BINARY_OP(mul, a[i] * b[i])
SCALAR_BINARY_OP(div, a[i] / b[i])
SCALAR_BINARY_OP(max, (a[i] > b[i] ? a[i] : b[i]))
SCALAR_BINARY_OP(min, (a[i] < b[i] ? a[i] : b[i]))

#undef SCALAR_BINARY_OP

// Sign for integers
template <>
void dispatch_unary_sign<int32_t>(const int32_t* input, int32_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = (input[i] > 0) ? 1 : ((input[i] < 0) ? -1 : 0);
    }
}

template <>
void dispatch_unary_sign<int64_t>(const int64_t* input, int64_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = (input[i] > 0) ? 1 : ((input[i] < 0) ? -1 : 0);
    }
}

// Square for integers
template <>
void dispatch_unary_square<int32_t>(const int32_t* input, int32_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] * input[i];
    }
}

template <>
void dispatch_unary_square<int64_t>(const int64_t* input, int64_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] * input[i];
    }
}

// ReLU for integers
template <>
void dispatch_activation_relu<int32_t>(const int32_t* input, int32_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

template <>
void dispatch_activation_relu<int64_t>(const int64_t* input, int64_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

// Abs for integers
template <>
void dispatch_unary_abs<int32_t>(const int32_t* input, int32_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] < 0 ? -input[i] : input[i];
    }
}

template <>
void dispatch_unary_abs<int64_t>(const int64_t* input, int64_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] < 0 ? -input[i] : input[i];
    }
}

// Neg for integers
template <>
void dispatch_unary_neg<int32_t>(const int32_t* input, int32_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = -input[i];
    }
}

template <>
void dispatch_unary_neg<int64_t>(const int64_t* input, int64_t* output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = -input[i];
    }
}

}  // namespace simd
}  // namespace axiom

#endif  // HWY_ONCE
