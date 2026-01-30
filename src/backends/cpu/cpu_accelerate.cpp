#include "cpu_accelerate.hpp"

#ifdef AXIOM_USE_ACCELERATE

// Use the new Accelerate LAPACK interface
#define ACCELERATE_NEW_LAPACK

#include <algorithm>
#include <cmath>
#include <vector>

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

void vround_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvnintf(output, input, &count);
}

void vround_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvnint(output, input, &count);
}

void vtrunc_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvintf(output, input, &count);
}

void vtrunc_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvint(output, input, &count);
}

void vrecip_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvrecf(output, input, &count);
}

void vrecip_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvrec(output, input, &count);
}

void vrsqrt_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvrsqrtf(output, input, &count);
}

void vrsqrt_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvrsqrt(output, input, &count);
}

void vsquare_f32(const float *input, float *output, size_t n) {
    vDSP_vsq(input, 1, output, 1, static_cast<vDSP_Length>(n));
}

void vsquare_f64(const double *input, double *output, size_t n) {
    vDSP_vsqD(input, 1, output, 1, static_cast<vDSP_Length>(n));
}

void vsinh_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvsinhf(output, input, &count);
}

void vsinh_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvsinh(output, input, &count);
}

void vcosh_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvcoshf(output, input, &count);
}

void vcosh_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvcosh(output, input, &count);
}

void vasin_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvasinf(output, input, &count);
}

void vasin_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvasin(output, input, &count);
}

void vacos_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvacosf(output, input, &count);
}

void vacos_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvacos(output, input, &count);
}

void vatan_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvatanf(output, input, &count);
}

void vatan_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvatan(output, input, &count);
}

void verf_f32(const float *input, float *output, size_t n) {
    // vForce doesn't have erf, use scalar fallback with vectorized loop
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::erff(input[i]);
    }
}

void verf_f64(const double *input, double *output, size_t n) {
    // vForce doesn't have erf, use scalar fallback
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::erf(input[i]);
    }
}

void vcbrt_f32(const float *input, float *output, size_t n) {
    // vForce doesn't have cbrt, compute as pow(x, 1/3)
    // For negative values, we need special handling
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::cbrtf(input[i]);
    }
}

void vcbrt_f64(const double *input, double *output, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::cbrt(input[i]);
    }
}

void vlog2_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvlog2f(output, input, &count);
}

void vlog2_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvlog2(output, input, &count);
}

void vlog10_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvlog10f(output, input, &count);
}

void vlog10_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvlog10(output, input, &count);
}

void vlog1p_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvlog1pf(output, input, &count);
}

void vlog1p_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvlog1p(output, input, &count);
}

void vexpm1_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvexpm1f(output, input, &count);
}

void vexpm1_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvexpm1(output, input, &count);
}

void vexp2_f32(const float *input, float *output, size_t n) {
    int count = static_cast<int>(n);
    vvexp2f(output, input, &count);
}

void vexp2_f64(const double *input, double *output, size_t n) {
    int count = static_cast<int>(n);
    vvexp2(output, input, &count);
}

// ============================================================================
// vForce Binary Math Operations
// ============================================================================

void vpow_f32(const float *a, const float *b, float *result, size_t n) {
    int count = static_cast<int>(n);
    vvpowf(result, b, a, &count);  // Note: vvpow takes (result, exponent, base, count)
}

void vpow_f64(const double *a, const double *b, double *result, size_t n) {
    int count = static_cast<int>(n);
    vvpow(result, b, a, &count);
}

void vatan2_f32(const float *y, const float *x, float *result, size_t n) {
    int count = static_cast<int>(n);
    vvatan2f(result, y, x, &count);
}

void vatan2_f64(const double *y, const double *x, double *result, size_t n) {
    int count = static_cast<int>(n);
    vvatan2(result, y, x, &count);
}

void vhypot_f32(const float *a, const float *b, float *result, size_t n) {
    // vForce doesn't have hypot, compute sqrt(a^2 + b^2) using vDSP
    // Temporary buffers for squares
    std::vector<float> a_sq(n), b_sq(n);
    vDSP_vsq(a, 1, a_sq.data(), 1, static_cast<vDSP_Length>(n));
    vDSP_vsq(b, 1, b_sq.data(), 1, static_cast<vDSP_Length>(n));
    vDSP_vadd(a_sq.data(), 1, b_sq.data(), 1, result, 1, static_cast<vDSP_Length>(n));
    int count = static_cast<int>(n);
    vvsqrtf(result, result, &count);
}

void vhypot_f64(const double *a, const double *b, double *result, size_t n) {
    std::vector<double> a_sq(n), b_sq(n);
    vDSP_vsqD(a, 1, a_sq.data(), 1, static_cast<vDSP_Length>(n));
    vDSP_vsqD(b, 1, b_sq.data(), 1, static_cast<vDSP_Length>(n));
    vDSP_vaddD(a_sq.data(), 1, b_sq.data(), 1, result, 1, static_cast<vDSP_Length>(n));
    int count = static_cast<int>(n);
    vvsqrt(result, result, &count);
}

void vfmod_f32(const float *a, const float *b, float *result, size_t n) {
    int count = static_cast<int>(n);
    vvfmodf(result, a, b, &count);
}

void vfmod_f64(const double *a, const double *b, double *result, size_t n) {
    int count = static_cast<int>(n);
    vvfmod(result, a, b, &count);
}

void vcopysign_f32(const float *a, const float *b, float *result, size_t n) {
    int count = static_cast<int>(n);
    vvcopysignf(result, a, b, &count);
}

void vcopysign_f64(const double *a, const double *b, double *result, size_t n) {
    int count = static_cast<int>(n);
    vvcopysign(result, a, b, &count);
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

float vsumsq_f32(const float *input, size_t n) {
    float result = 0.0f;
    vDSP_svesq(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

double vsumsq_f64(const double *input, size_t n) {
    double result = 0.0;
    vDSP_svesqD(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

float vrms_f32(const float *input, size_t n) {
    float result = 0.0f;
    vDSP_rmsqv(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

double vrms_f64(const double *input, size_t n) {
    double result = 0.0;
    vDSP_rmsqvD(input, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

float vdot_f32(const float *a, const float *b, size_t n) {
    float result = 0.0f;
    vDSP_dotpr(a, 1, b, 1, &result, static_cast<vDSP_Length>(n));
    return result;
}

double vdot_f64(const double *a, const double *b, size_t n) {
    double res = 0.0;
    vDSP_dotprD(a, 1, b, 1, &res, static_cast<vDSP_Length>(n));
    return res;
}

float vnorm2_f32(const float *input, size_t n) {
    // Use BLAS for L2 norm
    return cblas_snrm2(static_cast<int>(n), input, 1);
}

double vnorm2_f64(const double *input, size_t n) {
    return cblas_dnrm2(static_cast<int>(n), input, 1);
}

float vnorm1_f32(const float *input, size_t n) {
    // Use BLAS for L1 norm (sum of absolute values)
    return cblas_sasum(static_cast<int>(n), input, 1);
}

double vnorm1_f64(const double *input, size_t n) {
    return cblas_dasum(static_cast<int>(n), input, 1);
}

size_t vargmax_f32(const float *input, size_t n) {
    if (n == 0) return 0;
    float max_val;
    vDSP_Length index;
    vDSP_maxvi(input, 1, &max_val, &index, static_cast<vDSP_Length>(n));
    return static_cast<size_t>(index);
}

size_t vargmax_f64(const double *input, size_t n) {
    if (n == 0) return 0;
    double max_val;
    vDSP_Length index;
    vDSP_maxviD(input, 1, &max_val, &index, static_cast<vDSP_Length>(n));
    return static_cast<size_t>(index);
}

size_t vargmin_f32(const float *input, size_t n) {
    if (n == 0) return 0;
    float min_val;
    vDSP_Length index;
    vDSP_minvi(input, 1, &min_val, &index, static_cast<vDSP_Length>(n));
    return static_cast<size_t>(index);
}

size_t vargmin_f64(const double *input, size_t n) {
    if (n == 0) return 0;
    double min_val;
    vDSP_Length index;
    vDSP_minviD(input, 1, &min_val, &index, static_cast<vDSP_Length>(n));
    return static_cast<size_t>(index);
}

size_t vargmax_abs_f32(const float *input, size_t n) {
    if (n == 0) return 0;
    // BLAS isamax returns 1-based index
    int idx = cblas_isamax(static_cast<int>(n), input, 1);
    return static_cast<size_t>(idx);
}

size_t vargmax_abs_f64(const double *input, size_t n) {
    if (n == 0) return 0;
    int idx = cblas_idamax(static_cast<int>(n), input, 1);
    return static_cast<size_t>(idx);
}

// ============================================================================
// vDSP Vector Operations
// ============================================================================

void vclip_f32(const float *input, float low, float high, float *output, size_t n) {
    vDSP_vclip(input, 1, &low, &high, output, 1, static_cast<vDSP_Length>(n));
}

void vclip_f64(const double *input, double low, double high, double *output, size_t n) {
    vDSP_vclipD(input, 1, &low, &high, output, 1, static_cast<vDSP_Length>(n));
}

void vthreshold_f32(const float *input, float threshold, float *output, size_t n) {
    // vDSP_vthr: if input[i] >= threshold, output[i] = input[i], else output[i] = threshold
    // We want: if input[i] >= threshold, output[i] = input[i], else output[i] = 0
    // Use vDSP_vthres for this behavior
    vDSP_vthres(input, 1, &threshold, output, 1, static_cast<vDSP_Length>(n));
}

void vthreshold_f64(const double *input, double threshold, double *output, size_t n) {
    vDSP_vthresD(input, 1, &threshold, output, 1, static_cast<vDSP_Length>(n));
}

void vmtrans_f32(const float *input, float *output, size_t rows, size_t cols) {
    // vDSP_mtrans transposes an MxN matrix to NxM
    vDSP_mtrans(input, 1, output, 1,
                static_cast<vDSP_Length>(cols),   // Output rows = input cols
                static_cast<vDSP_Length>(rows));  // Output cols = input rows
}

void vmtrans_f64(const double *input, double *output, size_t rows, size_t cols) {
    vDSP_mtransD(input, 1, output, 1,
                 static_cast<vDSP_Length>(cols),
                 static_cast<vDSP_Length>(rows));
}

void vnormalize_f32(const float *input, float *output, size_t n) {
    if (n == 0) return;

    // Compute mean
    float mean;
    vDSP_meanv(input, 1, &mean, static_cast<vDSP_Length>(n));

    // Subtract mean: output = input - mean
    float neg_mean = -mean;
    vDSP_vsadd(input, 1, &neg_mean, output, 1, static_cast<vDSP_Length>(n));

    // Compute variance (mean of squared deviations)
    float var;
    vDSP_measqv(output, 1, &var, static_cast<vDSP_Length>(n));

    // Divide by std (sqrt of variance)
    float std_dev = std::sqrt(var);
    if (std_dev > 0.0f) {
        vDSP_vsdiv(output, 1, &std_dev, output, 1, static_cast<vDSP_Length>(n));
    }
}

void vnormalize_f64(const double *input, double *output, size_t n) {
    if (n == 0) return;

    double mean;
    vDSP_meanvD(input, 1, &mean, static_cast<vDSP_Length>(n));

    double neg_mean = -mean;
    vDSP_vsaddD(input, 1, &neg_mean, output, 1, static_cast<vDSP_Length>(n));

    double var;
    vDSP_measqvD(output, 1, &var, static_cast<vDSP_Length>(n));

    double std_dev = std::sqrt(var);
    if (std_dev > 0.0) {
        vDSP_vsdivD(output, 1, &std_dev, output, 1, static_cast<vDSP_Length>(n));
    }
}

void vlerp_f32(const float *a, const float *b, float t, float *output, size_t n) {
    // Linear interpolation: output = a + t * (b - a) = (1-t)*a + t*b
    // Use vDSP_vintb which does exactly this
    vDSP_vintb(a, 1, b, 1, &t, output, 1, static_cast<vDSP_Length>(n));
}

void vlerp_f64(const double *a, const double *b, double t, double *output, size_t n) {
    vDSP_vintbD(a, 1, b, 1, &t, output, 1, static_cast<vDSP_Length>(n));
}

void vpoly_f32(const float *input, const float *coeffs, size_t num_coeffs, float *output, size_t n) {
    // vDSP_vpoly evaluates polynomial: output[i] = sum(coeffs[j] * input[i]^j)
    // coeffs are in order: coeffs[0] = highest degree coefficient
    vDSP_vpoly(input, 1, coeffs, 1, output, 1,
               static_cast<vDSP_Length>(n),
               static_cast<vDSP_Length>(num_coeffs - 1));
}

void vpoly_f64(const double *input, const double *coeffs, size_t num_coeffs, double *output, size_t n) {
    vDSP_vpolyD(input, 1, coeffs, 1, output, 1,
                static_cast<vDSP_Length>(n),
                static_cast<vDSP_Length>(num_coeffs - 1));
}

// ============================================================================
// BLAS Level 1 Operations
// ============================================================================

void vaxpy_f32(float alpha, const float *x, float *y, size_t n) {
    cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
}

void vaxpy_f64(double alpha, const double *x, double *y, size_t n) {
    cblas_daxpy(static_cast<int>(n), alpha, x, 1, y, 1);
}

void vscale_f32(float alpha, float *x, size_t n) {
    cblas_sscal(static_cast<int>(n), alpha, x, 1);
}

void vscale_f64(double alpha, double *x, size_t n) {
    cblas_dscal(static_cast<int>(n), alpha, x, 1);
}

void vcopy_f32(const float *x, float *y, size_t n) {
    cblas_scopy(static_cast<int>(n), x, 1, y, 1);
}

void vcopy_f64(const double *x, double *y, size_t n) {
    cblas_dcopy(static_cast<int>(n), x, 1, y, 1);
}

void vswap_f32(float *x, float *y, size_t n) {
    cblas_sswap(static_cast<int>(n), x, 1, y, 1);
}

void vswap_f64(double *x, double *y, size_t n) {
    cblas_dswap(static_cast<int>(n), x, 1, y, 1);
}

// ============================================================================
// BLAS Level 2 Operations (Matrix-Vector)
// ============================================================================

void gemv_f32(const float *A, const float *x, float *y,
              size_t M, size_t N, size_t lda,
              bool transpose_a) {
    // y = op(A) * x
    // If not transposed: A is MxN, x is Nx1, y is Mx1
    // If transposed: A is MxN but op(A) is NxM, x is Mx1, y is Nx1
    cblas_sgemv(CblasRowMajor,
                transpose_a ? CblasTrans : CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N),
                1.0f,  // alpha
                A, static_cast<int>(lda),
                x, 1,
                0.0f,  // beta
                y, 1);
}

void gemv_f64(const double *A, const double *x, double *y,
              size_t M, size_t N, size_t lda,
              bool transpose_a) {
    cblas_dgemv(CblasRowMajor,
                transpose_a ? CblasTrans : CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N),
                1.0,
                A, static_cast<int>(lda),
                x, 1,
                0.0,
                y, 1);
}

// ============================================================================
// Activation Functions (Optimized)
// ============================================================================

void vrelu_f32(const float *input, float *output, size_t n) {
    // ReLU: max(0, x)
    // Use vDSP_vmax with a zero vector for proper ReLU behavior
    // Or use vDSP_vthr which sets values < threshold to threshold
    float zero = 0.0f;
    vDSP_vthr(input, 1, &zero, output, 1, static_cast<vDSP_Length>(n));
}

void vrelu_f64(const double *input, double *output, size_t n) {
    double zero = 0.0;
    vDSP_vthrD(input, 1, &zero, output, 1, static_cast<vDSP_Length>(n));
}

void vrelu_clipped_f32(const float *input, float cap, float *output, size_t n) {
    // Clipped ReLU: min(max(0, x), cap)
    float low = 0.0f;
    vDSP_vclip(input, 1, &low, &cap, output, 1, static_cast<vDSP_Length>(n));
}

void vrelu_clipped_f64(const double *input, double cap, double *output, size_t n) {
    double low = 0.0;
    vDSP_vclipD(input, 1, &low, &cap, output, 1, static_cast<vDSP_Length>(n));
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
