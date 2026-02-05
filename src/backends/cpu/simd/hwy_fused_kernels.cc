// Copyright 2024 Axiom Authors
// SPDX-License-Identifier: MIT
//
// Dynamic dispatch wrappers for fused SIMD kernels

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "backends/cpu/simd/hwy_fused_kernels.cc"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"

// Include the implementation for each target
#include "hwy_fused_kernels-inl.h"

#if HWY_ONCE

// Include dispatch header to get template declarations
#include "backends/cpu/simd/simd_dispatch.hpp"

namespace axiom {
namespace simd {

// ============================================================================
// HWY_EXPORT declarations for dynamic dispatch
// ============================================================================

// Binary + Unary patterns - float
HWY_EXPORT(FusedAddReLUF);
HWY_EXPORT(FusedSubAbsF);
HWY_EXPORT(FusedAddSquareF);
HWY_EXPORT(FusedMulReLUF);
HWY_EXPORT(FusedSubSquareF);
HWY_EXPORT(FusedAddSigmoidF);
HWY_EXPORT(FusedMulSigmoidF);

// Binary + Unary patterns - double
HWY_EXPORT(FusedAddReLUD);
HWY_EXPORT(FusedSubAbsD);
HWY_EXPORT(FusedAddSquareD);
HWY_EXPORT(FusedMulReLUD);
HWY_EXPORT(FusedSubSquareD);
HWY_EXPORT(FusedAddSigmoidD);
HWY_EXPORT(FusedMulSigmoidD);

// Ternary patterns - float
HWY_EXPORT(FusedMulAddF);
HWY_EXPORT(FusedMulSubF);
HWY_EXPORT(FusedScaleShiftReLUF);
HWY_EXPORT(FusedAddMulReLUF);
HWY_EXPORT(FusedSubMulAbsF);

// Ternary patterns - double
HWY_EXPORT(FusedMulAddD);
HWY_EXPORT(FusedMulSubD);
HWY_EXPORT(FusedScaleShiftReLUD);
HWY_EXPORT(FusedAddMulReLUD);
HWY_EXPORT(FusedSubMulAbsD);

// Integer patterns - int32
HWY_EXPORT(FusedAddReLUI32);
HWY_EXPORT(FusedSubAbsI32);
HWY_EXPORT(FusedMulAddI32);
HWY_EXPORT(FusedAddSquareI32);
HWY_EXPORT(FusedSubSquareI32);

// Integer patterns - int64
HWY_EXPORT(FusedAddReLUI64);
HWY_EXPORT(FusedSubAbsI64);
HWY_EXPORT(FusedMulAddI64);
HWY_EXPORT(FusedAddSquareI64);
HWY_EXPORT(FusedSubSquareI64);

// ============================================================================
// Dispatch Functions
// ============================================================================

void dispatch_fused_add_relu_f32(const float *a, const float *b, float *result,
                                  size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddReLUF)(a, b, result, n);
}

void dispatch_fused_add_relu_f64(const double *a, const double *b,
                                  double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddReLUD)(a, b, result, n);
}

void dispatch_fused_sub_abs_f32(const float *a, const float *b, float *result,
                                 size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubAbsF)(a, b, result, n);
}

void dispatch_fused_sub_abs_f64(const double *a, const double *b,
                                 double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubAbsD)(a, b, result, n);
}

void dispatch_fused_add_square_f32(const float *a, const float *b,
                                    float *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddSquareF)(a, b, result, n);
}

void dispatch_fused_add_square_f64(const double *a, const double *b,
                                    double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddSquareD)(a, b, result, n);
}

void dispatch_fused_mul_add_f32(const float *a, const float *b, const float *c,
                                 float *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulAddF)(a, b, c, result, n);
}

void dispatch_fused_mul_add_f64(const double *a, const double *b,
                                 const double *c, double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulAddD)(a, b, c, result, n);
}

void dispatch_fused_scale_shift_relu_f32(const float *a, const float *scale,
                                          const float *bias, float *result,
                                          size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedScaleShiftReLUF)(a, scale, bias, result, n);
}

void dispatch_fused_scale_shift_relu_f64(const double *a, const double *scale,
                                          const double *bias, double *result,
                                          size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedScaleShiftReLUD)(a, scale, bias, result, n);
}

// New Binary + Unary patterns
void dispatch_fused_mul_relu_f32(const float *a, const float *b, float *result,
                                  size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulReLUF)(a, b, result, n);
}

void dispatch_fused_mul_relu_f64(const double *a, const double *b,
                                  double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulReLUD)(a, b, result, n);
}

void dispatch_fused_sub_square_f32(const float *a, const float *b, float *result,
                                    size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubSquareF)(a, b, result, n);
}

void dispatch_fused_sub_square_f64(const double *a, const double *b,
                                    double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubSquareD)(a, b, result, n);
}

void dispatch_fused_add_sigmoid_f32(const float *a, const float *b, float *result,
                                     size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddSigmoidF)(a, b, result, n);
}

void dispatch_fused_add_sigmoid_f64(const double *a, const double *b,
                                     double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddSigmoidD)(a, b, result, n);
}

void dispatch_fused_mul_sigmoid_f32(const float *a, const float *b, float *result,
                                     size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulSigmoidF)(a, b, result, n);
}

void dispatch_fused_mul_sigmoid_f64(const double *a, const double *b,
                                     double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulSigmoidD)(a, b, result, n);
}

// New Ternary patterns
void dispatch_fused_mul_sub_f32(const float *a, const float *b, const float *c,
                                 float *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulSubF)(a, b, c, result, n);
}

void dispatch_fused_mul_sub_f64(const double *a, const double *b,
                                 const double *c, double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulSubD)(a, b, c, result, n);
}

void dispatch_fused_add_mul_relu_f32(const float *a, const float *b,
                                      const float *c, float *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddMulReLUF)(a, b, c, result, n);
}

void dispatch_fused_add_mul_relu_f64(const double *a, const double *b,
                                      const double *c, double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddMulReLUD)(a, b, c, result, n);
}

void dispatch_fused_sub_mul_abs_f32(const float *a, const float *b,
                                     const float *c, float *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubMulAbsF)(a, b, c, result, n);
}

void dispatch_fused_sub_mul_abs_f64(const double *a, const double *b,
                                     const double *c, double *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubMulAbsD)(a, b, c, result, n);
}

// Integer dispatch functions
void dispatch_fused_add_relu_i32(const int32_t *a, const int32_t *b,
                                  int32_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddReLUI32)(a, b, result, n);
}

void dispatch_fused_add_relu_i64(const int64_t *a, const int64_t *b,
                                  int64_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddReLUI64)(a, b, result, n);
}

void dispatch_fused_sub_abs_i32(const int32_t *a, const int32_t *b,
                                 int32_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubAbsI32)(a, b, result, n);
}

void dispatch_fused_sub_abs_i64(const int64_t *a, const int64_t *b,
                                 int64_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubAbsI64)(a, b, result, n);
}

void dispatch_fused_mul_add_i32(const int32_t *a, const int32_t *b,
                                 const int32_t *c, int32_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulAddI32)(a, b, c, result, n);
}

void dispatch_fused_mul_add_i64(const int64_t *a, const int64_t *b,
                                 const int64_t *c, int64_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedMulAddI64)(a, b, c, result, n);
}

void dispatch_fused_add_square_i32(const int32_t *a, const int32_t *b,
                                    int32_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddSquareI32)(a, b, result, n);
}

void dispatch_fused_add_square_i64(const int64_t *a, const int64_t *b,
                                    int64_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedAddSquareI64)(a, b, result, n);
}

void dispatch_fused_sub_square_i32(const int32_t *a, const int32_t *b,
                                    int32_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubSquareI32)(a, b, result, n);
}

void dispatch_fused_sub_square_i64(const int64_t *a, const int64_t *b,
                                    int64_t *result, size_t n) {
    HWY_DYNAMIC_DISPATCH(FusedSubSquareI64)(a, b, result, n);
}

// ============================================================================
// Template Dispatch Wrappers
// ============================================================================

template <>
void dispatch_fused_add_relu<float>(const float *a, const float *b,
                                     float *result, size_t n) {
    dispatch_fused_add_relu_f32(a, b, result, n);
}

template <>
void dispatch_fused_add_relu<double>(const double *a, const double *b,
                                      double *result, size_t n) {
    dispatch_fused_add_relu_f64(a, b, result, n);
}

template <>
void dispatch_fused_sub_abs<float>(const float *a, const float *b,
                                    float *result, size_t n) {
    dispatch_fused_sub_abs_f32(a, b, result, n);
}

template <>
void dispatch_fused_sub_abs<double>(const double *a, const double *b,
                                     double *result, size_t n) {
    dispatch_fused_sub_abs_f64(a, b, result, n);
}

template <>
void dispatch_fused_add_square<float>(const float *a, const float *b,
                                       float *result, size_t n) {
    dispatch_fused_add_square_f32(a, b, result, n);
}

template <>
void dispatch_fused_add_square<double>(const double *a, const double *b,
                                        double *result, size_t n) {
    dispatch_fused_add_square_f64(a, b, result, n);
}

template <>
void dispatch_fused_mul_add<float>(const float *a, const float *b,
                                    const float *c, float *result, size_t n) {
    dispatch_fused_mul_add_f32(a, b, c, result, n);
}

template <>
void dispatch_fused_mul_add<double>(const double *a, const double *b,
                                     const double *c, double *result,
                                     size_t n) {
    dispatch_fused_mul_add_f64(a, b, c, result, n);
}

template <>
void dispatch_fused_scale_shift_relu<float>(const float *a, const float *scale,
                                             const float *bias, float *result,
                                             size_t n) {
    dispatch_fused_scale_shift_relu_f32(a, scale, bias, result, n);
}

template <>
void dispatch_fused_scale_shift_relu<double>(const double *a,
                                              const double *scale,
                                              const double *bias, double *result,
                                              size_t n) {
    dispatch_fused_scale_shift_relu_f64(a, scale, bias, result, n);
}

// New binary + unary pattern specializations
template <>
void dispatch_fused_mul_relu<float>(const float *a, const float *b,
                                     float *result, size_t n) {
    dispatch_fused_mul_relu_f32(a, b, result, n);
}

template <>
void dispatch_fused_mul_relu<double>(const double *a, const double *b,
                                      double *result, size_t n) {
    dispatch_fused_mul_relu_f64(a, b, result, n);
}

template <>
void dispatch_fused_sub_square<float>(const float *a, const float *b,
                                       float *result, size_t n) {
    dispatch_fused_sub_square_f32(a, b, result, n);
}

template <>
void dispatch_fused_sub_square<double>(const double *a, const double *b,
                                        double *result, size_t n) {
    dispatch_fused_sub_square_f64(a, b, result, n);
}

template <>
void dispatch_fused_add_sigmoid<float>(const float *a, const float *b,
                                        float *result, size_t n) {
    dispatch_fused_add_sigmoid_f32(a, b, result, n);
}

template <>
void dispatch_fused_add_sigmoid<double>(const double *a, const double *b,
                                         double *result, size_t n) {
    dispatch_fused_add_sigmoid_f64(a, b, result, n);
}

template <>
void dispatch_fused_mul_sigmoid<float>(const float *a, const float *b,
                                        float *result, size_t n) {
    dispatch_fused_mul_sigmoid_f32(a, b, result, n);
}

template <>
void dispatch_fused_mul_sigmoid<double>(const double *a, const double *b,
                                         double *result, size_t n) {
    dispatch_fused_mul_sigmoid_f64(a, b, result, n);
}

// New ternary pattern specializations
template <>
void dispatch_fused_mul_sub<float>(const float *a, const float *b,
                                    const float *c, float *result, size_t n) {
    dispatch_fused_mul_sub_f32(a, b, c, result, n);
}

template <>
void dispatch_fused_mul_sub<double>(const double *a, const double *b,
                                     const double *c, double *result, size_t n) {
    dispatch_fused_mul_sub_f64(a, b, c, result, n);
}

template <>
void dispatch_fused_add_mul_relu<float>(const float *a, const float *b,
                                         const float *c, float *result, size_t n) {
    dispatch_fused_add_mul_relu_f32(a, b, c, result, n);
}

template <>
void dispatch_fused_add_mul_relu<double>(const double *a, const double *b,
                                          const double *c, double *result, size_t n) {
    dispatch_fused_add_mul_relu_f64(a, b, c, result, n);
}

template <>
void dispatch_fused_sub_mul_abs<float>(const float *a, const float *b,
                                        const float *c, float *result, size_t n) {
    dispatch_fused_sub_mul_abs_f32(a, b, c, result, n);
}

template <>
void dispatch_fused_sub_mul_abs<double>(const double *a, const double *b,
                                         const double *c, double *result, size_t n) {
    dispatch_fused_sub_mul_abs_f64(a, b, c, result, n);
}

// Integer specializations
template <>
void dispatch_fused_add_relu<int32_t>(const int32_t *a, const int32_t *b,
                                       int32_t *result, size_t n) {
    dispatch_fused_add_relu_i32(a, b, result, n);
}

template <>
void dispatch_fused_add_relu<int64_t>(const int64_t *a, const int64_t *b,
                                       int64_t *result, size_t n) {
    dispatch_fused_add_relu_i64(a, b, result, n);
}

template <>
void dispatch_fused_sub_abs<int32_t>(const int32_t *a, const int32_t *b,
                                      int32_t *result, size_t n) {
    dispatch_fused_sub_abs_i32(a, b, result, n);
}

template <>
void dispatch_fused_sub_abs<int64_t>(const int64_t *a, const int64_t *b,
                                      int64_t *result, size_t n) {
    dispatch_fused_sub_abs_i64(a, b, result, n);
}

template <>
void dispatch_fused_mul_add<int32_t>(const int32_t *a, const int32_t *b,
                                      const int32_t *c, int32_t *result, size_t n) {
    dispatch_fused_mul_add_i32(a, b, c, result, n);
}

template <>
void dispatch_fused_mul_add<int64_t>(const int64_t *a, const int64_t *b,
                                      const int64_t *c, int64_t *result, size_t n) {
    dispatch_fused_mul_add_i64(a, b, c, result, n);
}

template <>
void dispatch_fused_add_square<int32_t>(const int32_t *a, const int32_t *b,
                                         int32_t *result, size_t n) {
    dispatch_fused_add_square_i32(a, b, result, n);
}

template <>
void dispatch_fused_add_square<int64_t>(const int64_t *a, const int64_t *b,
                                         int64_t *result, size_t n) {
    dispatch_fused_add_square_i64(a, b, result, n);
}

template <>
void dispatch_fused_sub_square<int32_t>(const int32_t *a, const int32_t *b,
                                         int32_t *result, size_t n) {
    dispatch_fused_sub_square_i32(a, b, result, n);
}

template <>
void dispatch_fused_sub_square<int64_t>(const int64_t *a, const int64_t *b,
                                         int64_t *result, size_t n) {
    dispatch_fused_sub_square_i64(a, b, result, n);
}

} // namespace simd
} // namespace axiom

#endif // HWY_ONCE
