#include "cpu_accelerate.hpp"

#ifdef AXIOM_USE_ACCELERATE

// Use the new Accelerate LAPACK interface
#define ACCELERATE_NEW_LAPACK

#include <algorithm>
#include <cmath>

namespace axiom {
namespace backends {
namespace cpu {
namespace accelerate {

// ============================================================================
// BLAS Matrix Multiplication Wrappers
// ============================================================================

void gemm_f32(const float *A, const float *B, float *C, size_t M, size_t N,
              size_t K, size_t lda, size_t ldb, size_t ldc, bool transpose_a,
              bool transpose_b) {
    // cblas_sgemm parameters:
    // CblasRowMajor: row-major storage
    // transA, transB: transpose flags
    // M: rows of op(A) and C
    // N: columns of op(B) and C
    // K: columns of op(A), rows of op(B)
    // alpha, beta: C = alpha*op(A)*op(B) + beta*C
    // lda, ldb, ldc: leading dimensions

    cblas_sgemm(CblasRowMajor, transpose_a ? CblasTrans : CblasNoTrans,
                transpose_b ? CblasTrans : CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K),
                1.0f, // alpha
                A, static_cast<int>(lda), B, static_cast<int>(ldb),
                0.0f, // beta
                C, static_cast<int>(ldc));
}

void gemm_f64(const double *A, const double *B, double *C, size_t M, size_t N,
              size_t K, size_t lda, size_t ldb, size_t ldc, bool transpose_a,
              bool transpose_b) {
    cblas_dgemm(CblasRowMajor, transpose_a ? CblasTrans : CblasNoTrans,
                transpose_b ? CblasTrans : CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K),
                1.0, // alpha
                A, static_cast<int>(lda), B, static_cast<int>(ldb),
                0.0, // beta
                C, static_cast<int>(ldc));
}

bool can_use_blas(const void *data, size_t row_stride, size_t col_stride,
                  size_t /*itemsize*/, size_t /*rows*/, size_t /*cols*/) {
    // BLAS requires:
    // 1. Non-null data pointer
    // 2. Positive strides (no negative strides from flip)
    // 3. Either row-major (col_stride == itemsize) or column-major (row_stride
    // == itemsize)
    // 4. Proper leading dimension alignment

    if (data == nullptr)
        return false;

    // Check for negative strides (from flip operations)
    // row_stride and col_stride are in elements, need to check if underlying
    // byte strides are negative Actually, in our case these are element strides
    // passed in, so we check if they'd be reasonable

    // For row-major: col_stride should be 1 (contiguous along row)
    bool is_row_major = (col_stride == 1);

    // For column-major: row_stride should be 1 (contiguous along column)
    bool is_col_major = (row_stride == 1);

    // Must be either row-major or column-major for BLAS
    return is_row_major || is_col_major;
}

// ============================================================================
// vDSP Binary Operations
// ============================================================================

void vadd_f32(const float *a, const float *b, float *result, size_t n) {
    vDSP_vadd(a, 1, b, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vadd_f64(const double *a, const double *b, double *result, size_t n) {
    vDSP_vaddD(a, 1, b, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vsub_f32(const float *a, const float *b, float *result, size_t n) {
    // vDSP_vsub computes B - A, so we swap arguments
    vDSP_vsub(b, 1, a, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vsub_f64(const double *a, const double *b, double *result, size_t n) {
    vDSP_vsubD(b, 1, a, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vmul_f32(const float *a, const float *b, float *result, size_t n) {
    vDSP_vmul(a, 1, b, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vmul_f64(const double *a, const double *b, double *result, size_t n) {
    vDSP_vmulD(a, 1, b, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vdiv_f32(const float *a, const float *b, float *result, size_t n) {
    // vDSP_vdiv computes B / A, so we swap arguments
    vDSP_vdiv(b, 1, a, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vdiv_f64(const double *a, const double *b, double *result, size_t n) {
    vDSP_vdivD(b, 1, a, 1, result, 1, static_cast<vDSP_Length>(n));
}

void vsmul_f32(const float *a, float scalar, float *result, size_t n) {
    vDSP_vsmul(a, 1, &scalar, result, 1, static_cast<vDSP_Length>(n));
}

void vsmul_f64(const double *a, double scalar, double *result, size_t n) {
    vDSP_vsmulD(a, 1, &scalar, result, 1, static_cast<vDSP_Length>(n));
}

void vsadd_f32(const float *a, float scalar, float *result, size_t n) {
    vDSP_vsadd(a, 1, &scalar, result, 1, static_cast<vDSP_Length>(n));
}

void vsadd_f64(const double *a, double scalar, double *result, size_t n) {
    vDSP_vsaddD(a, 1, &scalar, result, 1, static_cast<vDSP_Length>(n));
}

// ============================================================================
// vForce Unary Operations
// ============================================================================

void vexp_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvexpf(output, input, &count);
}

void vexp_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvexp(output, input, &count);
}

void vlog_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvlogf(output, input, &count);
}

void vlog_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvlog(output, input, &count);
}

void vsqrt_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvsqrtf(output, input, &count);
}

void vsqrt_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvsqrt(output, input, &count);
}

void vsin_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvsinf(output, input, &count);
}

void vsin_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvsin(output, input, &count);
}

void vcos_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvcosf(output, input, &count);
}

void vcos_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvcos(output, input, &count);
}

void vtan_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvtanf(output, input, &count);
}

void vtan_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvtan(output, input, &count);
}

void vtanh_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvtanhf(output, input, &count);
}

void vtanh_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvtanh(output, input, &count);
}

void vabs_f32(const float *input, float *output, size_t n) {
    vDSP_vabs(input, 1, output, 1, static_cast<vDSP_Length>(n));
}

void vabs_f64(const double *input, double *output, size_t n) {
    vDSP_vabsD(input, 1, output, 1, static_cast<vDSP_Length>(n));
}

void vneg_f32(const float *input, float *output, size_t n) {
    vDSP_vneg(input, 1, output, 1, static_cast<vDSP_Length>(n));
}

void vneg_f64(const double *input, double *output, size_t n) {
    vDSP_vnegD(input, 1, output, 1, static_cast<vDSP_Length>(n));
}

void vfloor_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvfloorf(output, input, &count);
}

void vfloor_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvfloor(output, input, &count);
}

void vceil_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvceilf(output, input, &count);
}

void vceil_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvceil(output, input, &count);
}

// ============================================================================
// vDSP Reduction Operations
// ============================================================================

float vsum_f32(const float *input, size_t n) {
    float result = 0.0f;
    vDSP_sve(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

double vsum_f64(const double *input, size_t n) {
    double result = 0.0;
    vDSP_sveD(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

float vmax_f32(const float *input, size_t n) {
    float result = 0.0f;
    vDSP_maxv(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

double vmax_f64(const double *input, size_t n) {
    double result = 0.0;
    vDSP_maxvD(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

float vmin_f32(const float *input, size_t n) {
    float result = 0.0f;
    vDSP_minv(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

double vmin_f64(const double *input, size_t n) {
    double result = 0.0;
    vDSP_minvD(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

float vmean_f32(const float *input, size_t n) {
    float result = 0.0f;
    vDSP_meanv(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

double vmean_f64(const double *input, size_t n) {
    double result = 0.0;
    vDSP_meanvD(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

// ============================================================================
// Softmax Optimization Helpers
// ============================================================================

void vsoftmax_f32(const float *input, float *output, size_t n) {
    if (n == 0)
        return;

    // Step 1: Find max for numerical stability
    float max_val;
    vDSP_maxv(input, 1, &max_val, static_cast<vDSP_Length>(n));

    // Step 2: Subtract max (output = input - max)
    float neg_max = -max_val;
    vDSP_vsadd(input, 1, &neg_max, output, 1, static_cast<vDSP_Length>(n));

    // Step 3: Compute exp(output)
    int count = static_cast<int>(n);
    vvexpf(output, output, &count);

    // Step 4: Sum the exponentials
    float sum;
    vDSP_sve(output, 1, &sum, static_cast<vDSP_Length>(n));

    // Step 5: Divide by sum (output = output / sum)
    vDSP_vsdiv(output, 1, &sum, output, 1, static_cast<vDSP_Length>(n));
}

void vsoftmax_f64(const double *input, double *output, size_t n) {
    if (n == 0)
        return;

    // Step 1: Find max for numerical stability
    double max_val;
    vDSP_maxvD(input, 1, &max_val, static_cast<vDSP_Length>(n));

    // Step 2: Subtract max
    double neg_max = -max_val;
    vDSP_vsaddD(input, 1, &neg_max, output, 1, static_cast<vDSP_Length>(n));

    // Step 3: Compute exp
    int count = static_cast<int>(n);
    vvexp(output, output, &count);

    // Step 4: Sum
    double sum;
    vDSP_sveD(output, 1, &sum, static_cast<vDSP_Length>(n));

    // Step 5: Divide
    vDSP_vsdivD(output, 1, &sum, output, 1, static_cast<vDSP_Length>(n));
}

void vlog_softmax_f32(const float *input, float *output, size_t n) {
    if (n == 0)
        return;

    // Step 1: Find max for numerical stability
    float max_val;
    vDSP_maxv(input, 1, &max_val, static_cast<vDSP_Length>(n));

    // Step 2: Compute exp(input - max) into temp
    float neg_max = -max_val;
    vDSP_vsadd(input, 1, &neg_max, output, 1, static_cast<vDSP_Length>(n));

    // Step 3: exp(output)
    int count = static_cast<int>(n);
    vvexpf(output, output, &count);

    // Step 4: Sum
    float sum;
    vDSP_sve(output, 1, &sum, static_cast<vDSP_Length>(n));

    // Step 5: log_softmax = (input - max) - log(sum)
    // output = input - max - log(sum)
    float log_sum_plus_max = std::log(sum) + max_val;
    float neg_log_sum_plus_max = -log_sum_plus_max;
    vDSP_vsadd(input, 1, &neg_log_sum_plus_max, output, 1,
               static_cast<vDSP_Length>(n));
}

void vlog_softmax_f64(const double *input, double *output, size_t n) {
    if (n == 0)
        return;

    double max_val;
    vDSP_maxvD(input, 1, &max_val, static_cast<vDSP_Length>(n));

    double neg_max = -max_val;
    vDSP_vsaddD(input, 1, &neg_max, output, 1, static_cast<vDSP_Length>(n));

    int count = static_cast<int>(n);
    vvexp(output, output, &count);

    double sum;
    vDSP_sveD(output, 1, &sum, static_cast<vDSP_Length>(n));

    double log_sum_plus_max = std::log(sum) + max_val;
    double neg_log_sum_plus_max = -log_sum_plus_max;
    vDSP_vsaddD(input, 1, &neg_log_sum_plus_max, output, 1,
                static_cast<vDSP_Length>(n));
}

} // namespace accelerate
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_ACCELERATE
