#pragma once

// Apple Accelerate framework wrappers for high-performance BLAS/vDSP operations
// This provides optimized implementations that take advantage of Apple Silicon

#ifdef AXIOM_USE_ACCELERATE

#include <Accelerate/Accelerate.h>
#include <cstddef>

namespace axiom {
namespace backends {
namespace cpu {
namespace accelerate {

// ============================================================================
// BLAS Matrix Multiplication Wrappers
// ============================================================================

// SGEMM: Single-precision general matrix multiply
// C = alpha * op(A) * op(B) + beta * C
// For our use: C = A @ B (alpha=1, beta=0)
void gemm_f32(const float *A, const float *B, float *C, size_t M, size_t N,
              size_t K, size_t lda, size_t ldb, size_t ldc, bool transpose_a,
              bool transpose_b);

// DGEMM: Double-precision general matrix multiply
void gemm_f64(const double *A, const double *B, double *C, size_t M, size_t N,
              size_t K, size_t lda, size_t ldb, size_t ldc, bool transpose_a,
              bool transpose_b);

// Check if matrices are suitable for BLAS (contiguous, no negative strides)
bool can_use_blas(const void *data, size_t row_stride, size_t col_stride,
                  size_t itemsize, size_t rows, size_t cols);

// ============================================================================
// vDSP Binary Operations (for contiguous float32/float64 arrays)
// ============================================================================

// Element-wise addition
void vadd_f32(const float *a, const float *b, float *result, size_t n);
void vadd_f64(const double *a, const double *b, double *result, size_t n);

// Element-wise subtraction
void vsub_f32(const float *a, const float *b, float *result, size_t n);
void vsub_f64(const double *a, const double *b, double *result, size_t n);

// Element-wise multiplication
void vmul_f32(const float *a, const float *b, float *result, size_t n);
void vmul_f64(const double *a, const double *b, double *result, size_t n);

// Element-wise division
void vdiv_f32(const float *a, const float *b, float *result, size_t n);
void vdiv_f64(const double *a, const double *b, double *result, size_t n);

// Scalar operations
void vsmul_f32(const float *a, float scalar, float *result, size_t n);
void vsmul_f64(const double *a, double scalar, double *result, size_t n);

void vsadd_f32(const float *a, float scalar, float *result, size_t n);
void vsadd_f64(const double *a, double scalar, double *result, size_t n);

// ============================================================================
// vForce Unary Operations (vectorized math functions)
// ============================================================================

// Exponential and logarithm
void vexp_f32(const float *input, float *output, size_t n);
void vexp_f64(const double *input, double *output, size_t n);

void vlog_f32(const float *input, float *output, size_t n);
void vlog_f64(const double *input, double *output, size_t n);

// Square root
void vsqrt_f32(const float *input, float *output, size_t n);
void vsqrt_f64(const double *input, double *output, size_t n);

// Trigonometric functions
void vsin_f32(const float *input, float *output, size_t n);
void vsin_f64(const double *input, double *output, size_t n);

void vcos_f32(const float *input, float *output, size_t n);
void vcos_f64(const double *input, double *output, size_t n);

void vtan_f32(const float *input, float *output, size_t n);
void vtan_f64(const double *input, double *output, size_t n);

// Hyperbolic functions
void vtanh_f32(const float *input, float *output, size_t n);
void vtanh_f64(const double *input, double *output, size_t n);

// Absolute value
void vabs_f32(const float *input, float *output, size_t n);
void vabs_f64(const double *input, double *output, size_t n);

// Negation
void vneg_f32(const float *input, float *output, size_t n);
void vneg_f64(const double *input, double *output, size_t n);

// Floor, ceil, round
void vfloor_f32(const float *input, float *output, size_t n);
void vfloor_f64(const double *input, double *output, size_t n);

void vceil_f32(const float *input, float *output, size_t n);
void vceil_f64(const double *input, double *output, size_t n);

// ============================================================================
// vDSP Reduction Operations
// ============================================================================

// Sum reduction
float vsum_f32(const float *input, size_t n);
double vsum_f64(const double *input, size_t n);

// Max reduction
float vmax_f32(const float *input, size_t n);
double vmax_f64(const double *input, size_t n);

// Min reduction
float vmin_f32(const float *input, size_t n);
double vmin_f64(const double *input, size_t n);

// Mean (sum / n)
float vmean_f32(const float *input, size_t n);
double vmean_f64(const double *input, size_t n);

// ============================================================================
// Softmax Optimization Helpers
// ============================================================================

// Compute softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
void vsoftmax_f32(const float *input, float *output, size_t n);
void vsoftmax_f64(const double *input, double *output, size_t n);

// Log softmax
void vlog_softmax_f32(const float *input, float *output, size_t n);
void vlog_softmax_f64(const double *input, double *output, size_t n);

} // namespace accelerate
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_ACCELERATE
