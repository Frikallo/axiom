#pragma once

// Apple Accelerate framework wrappers for high-performance vDSP/vForce
// operations This provides optimized implementations that take advantage of
// Apple Silicon

#ifdef AXIOM_USE_ACCELERATE

#include <Accelerate/Accelerate.h>
#include <cstddef>

namespace axiom {
namespace backends {
namespace cpu {
namespace accelerate {

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

// Floor, ceil, round, trunc
void vfloor_f32(const float *input, float *output, size_t n);
void vfloor_f64(const double *input, double *output, size_t n);

void vceil_f32(const float *input, float *output, size_t n);
void vceil_f64(const double *input, double *output, size_t n);

void vround_f32(const float *input, float *output, size_t n);
void vround_f64(const double *input, double *output, size_t n);

void vtrunc_f32(const float *input, float *output, size_t n);
void vtrunc_f64(const double *input, double *output, size_t n);

// Reciprocal (1/x)
void vrecip_f32(const float *input, float *output, size_t n);
void vrecip_f64(const double *input, double *output, size_t n);

// Reciprocal square root (1/sqrt(x))
void vrsqrt_f32(const float *input, float *output, size_t n);
void vrsqrt_f64(const double *input, double *output, size_t n);

// Square (x*x)
void vsquare_f32(const float *input, float *output, size_t n);
void vsquare_f64(const double *input, double *output, size_t n);

// Hyperbolic functions
void vsinh_f32(const float *input, float *output, size_t n);
void vsinh_f64(const double *input, double *output, size_t n);

void vcosh_f32(const float *input, float *output, size_t n);
void vcosh_f64(const double *input, double *output, size_t n);

// Inverse trigonometric functions
void vasin_f32(const float *input, float *output, size_t n);
void vasin_f64(const double *input, double *output, size_t n);

void vacos_f32(const float *input, float *output, size_t n);
void vacos_f64(const double *input, double *output, size_t n);

void vatan_f32(const float *input, float *output, size_t n);
void vatan_f64(const double *input, double *output, size_t n);

// Error function
void verf_f32(const float *input, float *output, size_t n);
void verf_f64(const double *input, double *output, size_t n);

// Cube root
void vcbrt_f32(const float *input, float *output, size_t n);
void vcbrt_f64(const double *input, double *output, size_t n);

// Alternative logarithm bases
void vlog2_f32(const float *input, float *output, size_t n);
void vlog2_f64(const double *input, double *output, size_t n);

void vlog10_f32(const float *input, float *output, size_t n);
void vlog10_f64(const double *input, double *output, size_t n);

// Numerically stable variants
void vlog1p_f32(const float *input, float *output, size_t n);
void vlog1p_f64(const double *input, double *output, size_t n);

void vexpm1_f32(const float *input, float *output, size_t n);
void vexpm1_f64(const double *input, double *output, size_t n);

// 2^x
void vexp2_f32(const float *input, float *output, size_t n);
void vexp2_f64(const double *input, double *output, size_t n);

// ============================================================================
// vForce Binary Math Operations
// ============================================================================

// Power (a^b)
void vpow_f32(const float *a, const float *b, float *result, size_t n);
void vpow_f64(const double *a, const double *b, double *result, size_t n);

// atan2(y, x)
void vatan2_f32(const float *y, const float *x, float *result, size_t n);
void vatan2_f64(const double *y, const double *x, double *result, size_t n);

// Hypotenuse sqrt(a^2 + b^2)
void vhypot_f32(const float *a, const float *b, float *result, size_t n);
void vhypot_f64(const double *a, const double *b, double *result, size_t n);

// Floating-point remainder
void vfmod_f32(const float *a, const float *b, float *result, size_t n);
void vfmod_f64(const double *a, const double *b, double *result, size_t n);

// Copysign (magnitude of a, sign of b)
void vcopysign_f32(const float *a, const float *b, float *result, size_t n);
void vcopysign_f64(const double *a, const double *b, double *result, size_t n);

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

// Sum of squares
float vsumsq_f32(const float *input, size_t n);
double vsumsq_f64(const double *input, size_t n);

// Root mean square
float vrms_f32(const float *input, size_t n);
double vrms_f64(const double *input, size_t n);

// Dot product
float vdot_f32(const float *a, const float *b, size_t n);
double vdot_f64(const double *a, const double *b, size_t n);

// L2 norm (Euclidean norm)
float vnorm2_f32(const float *input, size_t n);
double vnorm2_f64(const double *input, size_t n);

// L1 norm (sum of absolute values)
float vnorm1_f32(const float *input, size_t n);
double vnorm1_f64(const double *input, size_t n);

// Index of maximum value (argmax)
size_t vargmax_f32(const float *input, size_t n);
size_t vargmax_f64(const double *input, size_t n);

// Index of minimum value (argmin)
size_t vargmin_f32(const float *input, size_t n);
size_t vargmin_f64(const double *input, size_t n);

// Index of maximum absolute value
size_t vargmax_abs_f32(const float *input, size_t n);
size_t vargmax_abs_f64(const double *input, size_t n);

// ============================================================================
// vDSP Vector Operations
// ============================================================================

// Clip values to range [low, high]
void vclip_f32(const float *input, float low, float high, float *output,
               size_t n);
void vclip_f64(const double *input, double low, double high, double *output,
               size_t n);

// Threshold: output[i] = input[i] >= threshold ? input[i] : 0
// (Useful for ReLU implementation)
void vthreshold_f32(const float *input, float threshold, float *output,
                    size_t n);
void vthreshold_f64(const double *input, double threshold, double *output,
                    size_t n);

// Matrix transpose
void vmtrans_f32(const float *input, float *output, size_t rows, size_t cols);
void vmtrans_f64(const double *input, double *output, size_t rows, size_t cols);

// Normalize vector (output = (input - mean) / std)
void vnormalize_f32(const float *input, float *output, size_t n);
void vnormalize_f64(const double *input, double *output, size_t n);

// Linear interpolation: output = a + t * (b - a)
void vlerp_f32(const float *a, const float *b, float t, float *output,
               size_t n);
void vlerp_f64(const double *a, const double *b, double t, double *output,
               size_t n);

// Polynomial evaluation
void vpoly_f32(const float *input, const float *coeffs, size_t num_coeffs,
               float *output, size_t n);
void vpoly_f64(const double *input, const double *coeffs, size_t num_coeffs,
               double *output, size_t n);

// ============================================================================
// Activation Functions (Optimized)
// ============================================================================

// ReLU: max(0, x) - uses vDSP_vthr
void vrelu_f32(const float *input, float *output, size_t n);
void vrelu_f64(const double *input, double *output, size_t n);

// Clipped ReLU: min(max(0, x), cap)
void vrelu_clipped_f32(const float *input, float cap, float *output, size_t n);
void vrelu_clipped_f64(const double *input, double cap, double *output,
                       size_t n);

// ============================================================================
// Softmax Optimization Helpers
// ============================================================================

// Compute softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
void vsoftmax_f32(const float *input, float *output, size_t n);
void vsoftmax_f64(const double *input, double *output, size_t n);

// Log softmax
void vlog_softmax_f32(const float *input, float *output, size_t n);
void vlog_softmax_f64(const double *input, double *output, size_t n);

// ============================================================================
// FFT Operations (using vDSP FFT)
// ============================================================================

// Complex-to-complex FFT (in-place)
// Input/output are interleaved complex arrays: [re0, im0, re1, im1, ...]
// n must be a power of 2
// direction: 1 for forward FFT, -1 for inverse FFT
void vfft_c2c_f32(float *data, size_t n, int direction);
void vfft_c2c_f64(double *data, size_t n, int direction);

// Real-to-complex FFT
// Input: real array of size n
// Output: interleaved complex array of size n (full spectrum)
// n must be a power of 2
void vfft_r2c_f32(const float *input, float *output, size_t n);
void vfft_r2c_f64(const double *input, double *output, size_t n);

// Complex-to-real inverse FFT
// Input: interleaved complex array of size n
// Output: real array of size n
// n must be a power of 2
void vfft_c2r_f32(const float *input, float *output, size_t n);
void vfft_c2r_f64(const double *input, double *output, size_t n);

} // namespace accelerate
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_ACCELERATE
