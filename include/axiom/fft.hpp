#pragma once

#include <string>
#include <vector>

#include "tensor.hpp"

namespace axiom {
namespace fft {

// ============================================================================
// 1D FFT Operations
// ============================================================================

/**
 * Compute the 1-dimensional discrete Fourier Transform.
 * @param input Input tensor (can be real or complex)
 * @param n Length of the transformed axis (default: input size along axis)
 * @param axis Axis over which to compute the FFT (default: -1, last axis)
 * @param norm Normalization mode: "backward" (default), "forward", "ortho"
 * @return Complex tensor with FFT result
 */
Tensor fft(const Tensor &input, int64_t n = -1, int axis = -1,
           const std::string &norm = "backward");

/**
 * Compute the 1-dimensional inverse discrete Fourier Transform.
 * @param input Input tensor (typically complex)
 * @param n Length of the transformed axis
 * @param axis Axis over which to compute the inverse FFT
 * @param norm Normalization mode
 * @return Complex tensor with inverse FFT result
 */
Tensor ifft(const Tensor &input, int64_t n = -1, int axis = -1,
            const std::string &norm = "backward");

/**
 * Compute the 1-dimensional FFT of real input.
 * For real input of length n, output has length n//2+1 (Hermitian symmetry).
 * @param input Real input tensor
 * @param n Length of the FFT
 * @param axis Axis over which to compute the FFT
 * @param norm Normalization mode
 * @return Complex tensor with rfft result
 */
Tensor rfft(const Tensor &input, int64_t n = -1, int axis = -1,
            const std::string &norm = "backward");

/**
 * Compute the inverse FFT for real output.
 * @param input Complex input tensor (typically from rfft)
 * @param n Length of the output (required if input was zero-padded)
 * @param axis Axis over which to compute the inverse FFT
 * @param norm Normalization mode
 * @return Real tensor with irfft result
 */
Tensor irfft(const Tensor &input, int64_t n = -1, int axis = -1,
             const std::string &norm = "backward");

// ============================================================================
// 2D FFT Operations
// ============================================================================

/**
 * Compute the 2-dimensional discrete Fourier Transform.
 * @param input Input tensor
 * @param s Shape of the output along transformed axes
 * @param axes Axes over which to compute the FFT (default: last two)
 * @param norm Normalization mode
 * @return Complex tensor with 2D FFT result
 */
Tensor fft2(const Tensor &input, const std::vector<int64_t> &s = {},
            const std::vector<int> &axes = {-2, -1},
            const std::string &norm = "backward");

/**
 * Compute the 2-dimensional inverse discrete Fourier Transform.
 */
Tensor ifft2(const Tensor &input, const std::vector<int64_t> &s = {},
             const std::vector<int> &axes = {-2, -1},
             const std::string &norm = "backward");

/**
 * Compute the 2-dimensional FFT of real input.
 */
Tensor rfft2(const Tensor &input, const std::vector<int64_t> &s = {},
             const std::vector<int> &axes = {-2, -1},
             const std::string &norm = "backward");

/**
 * Compute the 2-dimensional inverse FFT for real output.
 */
Tensor irfft2(const Tensor &input, const std::vector<int64_t> &s = {},
              const std::vector<int> &axes = {-2, -1},
              const std::string &norm = "backward");

// ============================================================================
// N-dimensional FFT Operations
// ============================================================================

/**
 * Compute the N-dimensional discrete Fourier Transform.
 */
Tensor fftn(const Tensor &input, const std::vector<int64_t> &s = {},
            const std::vector<int> &axes = {},
            const std::string &norm = "backward");

/**
 * Compute the N-dimensional inverse discrete Fourier Transform.
 */
Tensor ifftn(const Tensor &input, const std::vector<int64_t> &s = {},
             const std::vector<int> &axes = {},
             const std::string &norm = "backward");

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Shift the zero-frequency component to the center of the spectrum.
 * @param input Input tensor
 * @param axes Axes over which to shift (default: all axes)
 * @return Shifted tensor
 */
Tensor fftshift(const Tensor &input, const std::vector<int> &axes = {});

/**
 * Inverse of fftshift.
 */
Tensor ifftshift(const Tensor &input, const std::vector<int> &axes = {});

/**
 * Return the Discrete Fourier Transform sample frequencies.
 * @param n Window length
 * @param d Sample spacing (default: 1.0)
 * @param dtype Output dtype
 * @param device Output device
 * @return 1D tensor of length n containing sample frequencies
 */
Tensor fftfreq(int64_t n, double d = 1.0, DType dtype = DType::Float64,
               Device device = Device::CPU);

/**
 * Return the Discrete Fourier Transform sample frequencies for rfft.
 * @param n Window length
 * @param d Sample spacing
 * @param dtype Output dtype
 * @param device Output device
 * @return 1D tensor of length n//2+1 containing sample frequencies
 */
Tensor rfftfreq(int64_t n, double d = 1.0, DType dtype = DType::Float64,
                Device device = Device::CPU);

} // namespace fft
} // namespace axiom
