#pragma once

#include "axiom/tensor.hpp"

#include <string>

namespace axiom {
namespace backends {
namespace cuda {

// 1D complex-to-complex FFT on GPU (used by fft / ifft).
Tensor cufft_c2c_1d(const Tensor &input, size_t fft_size, int axis,
                    bool inverse, const std::string &norm);

// 1D real-to-complex FFT on GPU (used by rfft).
// Output has fft_size/2+1 complex elements along the FFT axis.
Tensor cufft_r2c_1d(const Tensor &input, size_t fft_size, int axis,
                    const std::string &norm);

// 1D complex-to-real inverse FFT on GPU (used by irfft).
// Output has output_size real elements along the FFT axis.
Tensor cufft_c2r_1d(const Tensor &input, size_t output_size, int axis,
                    const std::string &norm);

// 2D complex-to-complex FFT on GPU (used by fft2 / ifft2).
// Operates on the last two dimensions. Input must have ndim >= 2.
Tensor cufft_c2c_2d(const Tensor &input, size_t rows, size_t cols, bool inverse,
                    const std::string &norm);

} // namespace cuda
} // namespace backends
} // namespace axiom
