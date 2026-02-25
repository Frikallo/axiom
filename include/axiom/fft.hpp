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

// ============================================================================
// Window Functions
// ============================================================================

/**
 * Hann (Hanning) window.
 * w[n] = 0.5 - 0.5 * cos(2*pi*n / (M-1))
 * @param M Number of points in the output window
 * @param periodic If true, returns a periodic window (for spectral analysis)
 * @param dtype Output dtype
 * @param device Output device
 * @return 1D tensor of length M containing the window
 */
Tensor hann_window(int64_t M, bool periodic = true,
                   DType dtype = DType::Float32, Device device = Device::CPU);

/**
 * Hamming window.
 * w[n] = 0.54 - 0.46 * cos(2*pi*n / (M-1))
 * @param M Number of points in the output window
 * @param periodic If true, returns a periodic window
 * @param dtype Output dtype
 * @param device Output device
 * @return 1D tensor of length M containing the window
 */
Tensor hamming_window(int64_t M, bool periodic = true,
                      DType dtype = DType::Float32,
                      Device device = Device::CPU);

/**
 * Blackman window.
 * w[n] = 0.42 - 0.5*cos(2*pi*n/(M-1)) + 0.08*cos(4*pi*n/(M-1))
 * @param M Number of points in the output window
 * @param periodic If true, returns a periodic window
 * @param dtype Output dtype
 * @param device Output device
 * @return 1D tensor of length M containing the window
 */
Tensor blackman_window(int64_t M, bool periodic = true,
                       DType dtype = DType::Float32,
                       Device device = Device::CPU);

/**
 * Bartlett (triangular) window.
 * @param M Number of points in the output window
 * @param periodic If true, returns a periodic window
 * @param dtype Output dtype
 * @param device Output device
 * @return 1D tensor of length M containing the window
 */
Tensor bartlett_window(int64_t M, bool periodic = true,
                       DType dtype = DType::Float32,
                       Device device = Device::CPU);

/**
 * Kaiser window.
 * @param M Number of points in the output window
 * @param beta Shape parameter for the window (default: 12.0)
 * @param periodic If true, returns a periodic window
 * @param dtype Output dtype
 * @param device Output device
 * @return 1D tensor of length M containing the window
 */
Tensor kaiser_window(int64_t M, double beta = 12.0, bool periodic = true,
                     DType dtype = DType::Float32, Device device = Device::CPU);

// ============================================================================
// Short-Time Fourier Transform
// ============================================================================

/**
 * Compute the Short-Time Fourier Transform.
 * @param input 1D or 2D real-valued signal (..., signal_length)
 * @param n_fft FFT window size
 * @param hop_length Number of samples between frames (default: n_fft/4)
 * @param win_length Window length (default: n_fft)
 * @param window Optional window tensor of length win_length
 * @param center If true, pad signal on both sides with reflect padding
 * @param pad_mode Padding mode when center=true ("reflect")
 * @param normalized If true, normalize output by 1/sqrt(n_fft)
 * @return Complex tensor of shape (..., n_fft/2+1, n_frames)
 */
Tensor stft(const Tensor &input, int64_t n_fft, int64_t hop_length = -1,
            int64_t win_length = -1, const Tensor &window = Tensor(),
            bool center = true, const std::string &pad_mode = "reflect",
            bool normalized = false);

} // namespace fft
} // namespace axiom
