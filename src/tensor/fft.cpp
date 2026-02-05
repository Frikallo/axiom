#include "axiom/fft.hpp"
#include "axiom/error.hpp"
#include "axiom/operations.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#ifdef AXIOM_USE_ACCELERATE
#include "backends/cpu/vdsp.hpp"
#endif

namespace axiom {
namespace fft {

namespace {

#ifndef AXIOM_USE_ACCELERATE
// Cooley-Tukey FFT implementation for power-of-2 sizes (fallback for non-Apple)
void fft_radix2_inplace(std::complex<double> *data, size_t n, bool inverse) {
    if (n <= 1)
        return;

    // Bit-reversal permutation
    size_t j = 0;
    for (size_t i = 1; i < n - 1; ++i) {
        size_t bit = n >> 1;
        while (j >= bit) {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // Cooley-Tukey iterative FFT
    for (size_t len = 2; len <= n; len *= 2) {
        double angle = (inverse ? 2.0 : -2.0) * M_PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                std::complex<double> u = data[i + k];
                std::complex<double> v = data[i + k + len / 2] * w;
                data[i + k] = u + v;
                data[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (size_t i = 0; i < n; ++i) {
            data[i] /= static_cast<double>(n);
        }
    }
}
#endif // AXIOM_USE_ACCELERATE

// Check if n is a power of 2
bool is_power_of_2(size_t n) { return n > 0 && (n & (n - 1)) == 0; }

// Get normalization factor
// Note: fft_radix2_inplace already applies 1/n for inverse FFT
// These factors are ADDITIONAL to what fft_radix2_inplace does
double get_norm_factor(size_t n, const std::string &norm, bool inverse) {
    if (norm == "backward") {
        // backward: no norm on FFT, 1/n on IFFT (already done in
        // fft_radix2_inplace)
        return 1.0;
    } else if (norm == "forward") {
        // forward: 1/n on FFT, no norm on IFFT
        // For inverse: fft_radix2_inplace already did 1/n, so multiply by n to
        // cancel
        return inverse ? static_cast<double>(n) : 1.0 / n;
    } else if (norm == "ortho") {
        // ortho: 1/sqrt(n) on both
        // For inverse: fft_radix2_inplace did 1/n, we want 1/sqrt(n), so
        // multiply by sqrt(n)
        return inverse ? std::sqrt(static_cast<double>(n))
                       : 1.0 / std::sqrt(static_cast<double>(n));
    }
    throw ValueError("fft: norm must be 'backward', 'forward', or 'ortho'");
}

// Perform 1D FFT along a single axis
Tensor fft_1d_impl(const Tensor &input, int64_t n, int axis, bool inverse,
                   const std::string &norm) {
    // Normalize axis
    int ndim = static_cast<int>(input.ndim());
    if (axis < 0)
        axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // Determine output length
    size_t input_size = input.shape()[axis];
    size_t fft_size = (n > 0) ? static_cast<size_t>(n) : input_size;

    // Move to CPU for computation
    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();

    // Convert to complex if needed
    DType result_dtype = input.dtype() == DType::Complex128 ? DType::Complex128
                                                            : DType::Complex64;
    Tensor complex_input = input_cpu;
    if (!is_complex_dtype(input.dtype())) {
        complex_input = input_cpu.to_complex();
    }
    if (complex_input.dtype() != result_dtype) {
        complex_input = complex_input.astype(result_dtype);
    }

    // Build output shape
    Shape output_shape = input.shape();
    output_shape[axis] = fft_size;

    Tensor result(output_shape, result_dtype, Device::CPU);

    // Process each 1D slice along the axis
    // Calculate total number of slices
    size_t n_slices = 1;
    for (size_t d = 0; d < input.ndim(); ++d) {
        if (static_cast<int>(d) != axis) {
            n_slices *= output_shape[d];
        }
    }

    // Compute strides for iteration
    std::vector<size_t> outer_shape;
    for (size_t d = 0; d < input.ndim(); ++d) {
        if (static_cast<int>(d) != axis) {
            outer_shape.push_back(output_shape[d]);
        }
    }

    double norm_factor = get_norm_factor(fft_size, norm, inverse);

    // Iterate over all slices
    std::vector<size_t> outer_idx(outer_shape.size(), 0);
    std::vector<std::complex<double>> buffer(fft_size);

    for (size_t slice = 0; slice < n_slices; ++slice) {
        // Build indices for this slice
        std::vector<size_t> in_idx(ndim), out_idx(ndim);
        size_t outer_pos = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d != axis) {
                in_idx[d] = outer_idx[outer_pos];
                out_idx[d] = outer_idx[outer_pos];
                outer_pos++;
            }
        }

        // Copy input data to buffer (with zero padding if needed)
        std::fill(buffer.begin(), buffer.end(), std::complex<double>(0, 0));
        for (size_t i = 0; i < std::min(input_size, fft_size); ++i) {
            in_idx[axis] = i;
            if (result_dtype == DType::Complex64) {
                auto val = complex_input.item<complex64_t>(in_idx);
                buffer[i] = std::complex<double>(val.real(), val.imag());
            } else {
                buffer[i] = complex_input.item<complex128_t>(in_idx);
            }
        }

        // Perform FFT
        if (is_power_of_2(fft_size)) {
#ifdef AXIOM_USE_ACCELERATE
            // Use vDSP accelerated FFT
            // Convert buffer to interleaved format for vDSP
            std::vector<double> interleaved(fft_size * 2);
            for (size_t i = 0; i < fft_size; ++i) {
                interleaved[2 * i] = buffer[i].real();
                interleaved[2 * i + 1] = buffer[i].imag();
            }

            backends::cpu::accelerate::vfft_c2c_f64(interleaved.data(),
                                                    fft_size, inverse ? -1 : 1);

            // Convert back to complex buffer
            for (size_t i = 0; i < fft_size; ++i) {
                buffer[i] = std::complex<double>(interleaved[2 * i],
                                                 interleaved[2 * i + 1]);
            }
#else
            fft_radix2_inplace(buffer.data(), fft_size, inverse);
#endif
        } else {
            // For non-power-of-2, use DFT (slower but correct)
            std::vector<std::complex<double>> temp(fft_size);
            double sign = inverse ? 1.0 : -1.0;
            for (size_t k = 0; k < fft_size; ++k) {
                std::complex<double> sum(0, 0);
                for (size_t j = 0; j < fft_size; ++j) {
                    double angle = sign * 2.0 * M_PI * k * j / fft_size;
                    sum += buffer[j] * std::complex<double>(std::cos(angle),
                                                            std::sin(angle));
                }
                temp[k] = sum;
            }
            buffer = temp;
            if (inverse) {
                for (auto &v : buffer) {
                    v /= static_cast<double>(fft_size);
                }
            }
        }

        // Apply normalization and copy to output
        for (size_t i = 0; i < fft_size; ++i) {
            out_idx[axis] = i;
            std::complex<double> val = buffer[i] * norm_factor;
            if (result_dtype == DType::Complex64) {
                result.set_item<complex64_t>(
                    out_idx, complex64_t(static_cast<float>(val.real()),
                                         static_cast<float>(val.imag())));
            } else {
                result.set_item<complex128_t>(out_idx, val);
            }
        }

        // Increment outer indices
        for (int d = static_cast<int>(outer_shape.size()) - 1; d >= 0; --d) {
            if (++outer_idx[d] < outer_shape[d])
                break;
            outer_idx[d] = 0;
        }
    }

    if (input.device() == Device::GPU) {
        return result.gpu();
    }
    return result;
}

} // namespace

// ============================================================================
// 1D FFT Operations
// ============================================================================

Tensor fft(const Tensor &input, int64_t n, int axis, const std::string &norm) {
    return fft_1d_impl(input, n, axis, false, norm);
}

Tensor ifft(const Tensor &input, int64_t n, int axis, const std::string &norm) {
    return fft_1d_impl(input, n, axis, true, norm);
}

Tensor rfft(const Tensor &input, int64_t n, int axis, const std::string &norm) {
    if (is_complex_dtype(input.dtype())) {
        throw TypeError("rfft: input must be real, got " + input.dtype_name());
    }

    // Normalize axis
    int ndim = static_cast<int>(input.ndim());
    if (axis < 0)
        axis += ndim;

    // Compute full FFT
    Tensor full_fft = fft(input, n, axis, norm);

    // Take only positive frequencies (n//2 + 1)
    size_t fft_size = full_fft.shape()[axis];
    size_t rfft_size = fft_size / 2 + 1;

    // Slice to keep only non-redundant frequencies
    std::vector<Slice> slices(ndim);
    for (int d = 0; d < ndim; ++d) {
        if (d == axis) {
            slices[d] = Slice(0, static_cast<int64_t>(rfft_size));
        } else {
            slices[d] = Slice();
        }
    }

    return full_fft.slice(slices);
}

Tensor irfft(const Tensor &input, int64_t n, int axis,
             const std::string &norm) {
    // Normalize axis
    int ndim = static_cast<int>(input.ndim());
    if (axis < 0)
        axis += ndim;

    size_t input_size = input.shape()[axis];
    size_t output_size =
        (n > 0) ? static_cast<size_t>(n) : 2 * (input_size - 1);

    // Reconstruct full spectrum using Hermitian symmetry
    Shape full_shape = input.shape();
    full_shape[axis] = output_size;

    Tensor full_spectrum(full_shape, input.dtype(), input.device());

    // Copy first half
    // For simplicity, use full ifft and take real part
    Tensor ifft_result =
        ifft(input, static_cast<int64_t>(output_size), axis, norm);

    // Take real part
    return ops::real(ifft_result);
}

// ============================================================================
// 2D FFT Operations
// ============================================================================

Tensor fft2(const Tensor &input, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
    std::vector<int> actual_axes =
        axes.empty() ? std::vector<int>{-2, -1} : axes;
    if (actual_axes.size() != 2) {
        throw ValueError("fft2: exactly 2 axes required");
    }

    std::vector<int64_t> actual_s = s;
    if (actual_s.empty()) {
        actual_s = {-1, -1};
    }

    Tensor result = input;
    result = fft(result, actual_s[0], actual_axes[0], norm);
    result = fft(result, actual_s[1], actual_axes[1], norm);
    return result;
}

Tensor ifft2(const Tensor &input, const std::vector<int64_t> &s,
             const std::vector<int> &axes, const std::string &norm) {
    std::vector<int> actual_axes =
        axes.empty() ? std::vector<int>{-2, -1} : axes;
    if (actual_axes.size() != 2) {
        throw ValueError("ifft2: exactly 2 axes required");
    }

    std::vector<int64_t> actual_s = s;
    if (actual_s.empty()) {
        actual_s = {-1, -1};
    }

    Tensor result = input;
    result = ifft(result, actual_s[0], actual_axes[0], norm);
    result = ifft(result, actual_s[1], actual_axes[1], norm);
    return result;
}

Tensor rfft2(const Tensor &input, const std::vector<int64_t> &s,
             const std::vector<int> &axes, const std::string &norm) {
    std::vector<int> actual_axes =
        axes.empty() ? std::vector<int>{-2, -1} : axes;
    if (actual_axes.size() != 2) {
        throw ValueError("rfft2: exactly 2 axes required");
    }

    std::vector<int64_t> actual_s = s;
    if (actual_s.empty()) {
        actual_s = {-1, -1};
    }

    // rfft along last axis first (real->complex), then fft along first axis
    Tensor result = rfft(input, actual_s[1], actual_axes[1], norm);
    result = fft(result, actual_s[0], actual_axes[0], norm);
    return result;
}

Tensor irfft2(const Tensor &input, const std::vector<int64_t> &s,
              const std::vector<int> &axes, const std::string &norm) {
    std::vector<int> actual_axes =
        axes.empty() ? std::vector<int>{-2, -1} : axes;
    if (actual_axes.size() != 2) {
        throw ValueError("irfft2: exactly 2 axes required");
    }

    std::vector<int64_t> actual_s = s;
    if (actual_s.empty()) {
        actual_s = {-1, -1};
    }

    // ifft along first axis first (complex->complex), then irfft along last
    // axis
    Tensor result = ifft(input, actual_s[0], actual_axes[0], norm);
    result = irfft(result, actual_s[1], actual_axes[1], norm);
    return result;
}

// ============================================================================
// N-dimensional FFT Operations
// ============================================================================

Tensor fftn(const Tensor &input, const std::vector<int64_t> &s,
            const std::vector<int> &axes, const std::string &norm) {
    std::vector<int> actual_axes = axes;
    if (actual_axes.empty()) {
        for (int i = 0; i < static_cast<int>(input.ndim()); ++i) {
            actual_axes.push_back(i);
        }
    }

    std::vector<int64_t> actual_s = s;
    if (actual_s.empty()) {
        actual_s.resize(actual_axes.size(), -1);
    }

    if (actual_s.size() != actual_axes.size()) {
        throw ValueError("fftn: s and axes must have same length");
    }

    Tensor result = input;
    for (size_t i = 0; i < actual_axes.size(); ++i) {
        result = fft(result, actual_s[i], actual_axes[i], norm);
    }
    return result;
}

Tensor ifftn(const Tensor &input, const std::vector<int64_t> &s,
             const std::vector<int> &axes, const std::string &norm) {
    std::vector<int> actual_axes = axes;
    if (actual_axes.empty()) {
        for (int i = 0; i < static_cast<int>(input.ndim()); ++i) {
            actual_axes.push_back(i);
        }
    }

    std::vector<int64_t> actual_s = s;
    if (actual_s.empty()) {
        actual_s.resize(actual_axes.size(), -1);
    }

    if (actual_s.size() != actual_axes.size()) {
        throw ValueError("ifftn: s and axes must have same length");
    }

    Tensor result = input;
    for (size_t i = 0; i < actual_axes.size(); ++i) {
        result = ifft(result, actual_s[i], actual_axes[i], norm);
    }
    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

Tensor fftshift(const Tensor &input, const std::vector<int> &axes) {
    std::vector<int> actual_axes = axes;
    if (actual_axes.empty()) {
        for (int i = 0; i < static_cast<int>(input.ndim()); ++i) {
            actual_axes.push_back(i);
        }
    }

    Tensor result = input;
    for (int axis : actual_axes) {
        int ndim = static_cast<int>(result.ndim());
        if (axis < 0)
            axis += ndim;

        int64_t n = static_cast<int64_t>(result.shape()[axis]);
        int64_t shift = n / 2;

        result = result.roll(shift, axis);
    }
    return result;
}

Tensor ifftshift(const Tensor &input, const std::vector<int> &axes) {
    std::vector<int> actual_axes = axes;
    if (actual_axes.empty()) {
        for (int i = 0; i < static_cast<int>(input.ndim()); ++i) {
            actual_axes.push_back(i);
        }
    }

    Tensor result = input;
    for (int axis : actual_axes) {
        int ndim = static_cast<int>(result.ndim());
        if (axis < 0)
            axis += ndim;

        int64_t n = static_cast<int64_t>(result.shape()[axis]);
        int64_t shift = (n + 1) / 2; // Ceiling division

        result = result.roll(-shift, axis);
    }
    return result;
}

Tensor fftfreq(int64_t n, double d, DType dtype, Device device) {
    if (n <= 0) {
        throw ValueError("fftfreq: n must be positive");
    }

    Tensor result({static_cast<size_t>(n)}, dtype, Device::CPU);

    // Frequencies: [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)
    double factor = 1.0 / (d * n);

    switch (dtype) {
    case DType::Float32: {
        float *data = result.typed_data<float>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t k = (i <= n / 2) ? i : i - n;
            data[i] = static_cast<float>(k * factor);
        }
        break;
    }
    case DType::Float64: {
        double *data = result.typed_data<double>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t k = (i <= n / 2) ? i : i - n;
            data[i] = k * factor;
        }
        break;
    }
    default:
        throw TypeError("fftfreq: unsupported dtype");
    }

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor rfftfreq(int64_t n, double d, DType dtype, Device device) {
    if (n <= 0) {
        throw ValueError("rfftfreq: n must be positive");
    }

    int64_t output_size = n / 2 + 1;
    Tensor result({static_cast<size_t>(output_size)}, dtype, Device::CPU);

    double factor = 1.0 / (d * n);

    switch (dtype) {
    case DType::Float32: {
        float *data = result.typed_data<float>();
        for (int64_t i = 0; i < output_size; ++i) {
            data[i] = static_cast<float>(i * factor);
        }
        break;
    }
    case DType::Float64: {
        double *data = result.typed_data<double>();
        for (int64_t i = 0; i < output_size; ++i) {
            data[i] = i * factor;
        }
        break;
    }
    default:
        throw TypeError("rfftfreq: unsupported dtype");
    }

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

// ============================================================================
// Window Functions
// ============================================================================

Tensor hann_window(int64_t M, bool periodic, DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("hann_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    // For periodic window, use M+1 points but return only first M
    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    const double pi = M_PI;

    switch (dtype) {
    case DType::Float32: {
        float *data = result.typed_data<float>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] =
                static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * i / N));
        }
        break;
    }
    case DType::Float64: {
        double *data = result.typed_data<double>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = 0.5 - 0.5 * std::cos(2.0 * pi * i / N);
        }
        break;
    }
    default:
        throw TypeError("hann_window: unsupported dtype");
    }

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor hamming_window(int64_t M, bool periodic, DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("hamming_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    const double pi = M_PI;
    const double alpha = 0.54;
    const double beta = 0.46;

    switch (dtype) {
    case DType::Float32: {
        float *data = result.typed_data<float>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] =
                static_cast<float>(alpha - beta * std::cos(2.0 * pi * i / N));
        }
        break;
    }
    case DType::Float64: {
        double *data = result.typed_data<double>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = alpha - beta * std::cos(2.0 * pi * i / N);
        }
        break;
    }
    default:
        throw TypeError("hamming_window: unsupported dtype");
    }

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor blackman_window(int64_t M, bool periodic, DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("blackman_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    const double pi = M_PI;
    const double a0 = 0.42;
    const double a1 = 0.5;
    const double a2 = 0.08;

    switch (dtype) {
    case DType::Float32: {
        float *data = result.typed_data<float>();
        for (int64_t i = 0; i < M; ++i) {
            double x = 2.0 * pi * i / N;
            data[i] = static_cast<float>(a0 - a1 * std::cos(x) +
                                         a2 * std::cos(2.0 * x));
        }
        break;
    }
    case DType::Float64: {
        double *data = result.typed_data<double>();
        for (int64_t i = 0; i < M; ++i) {
            double x = 2.0 * pi * i / N;
            data[i] = a0 - a1 * std::cos(x) + a2 * std::cos(2.0 * x);
        }
        break;
    }
    default:
        throw TypeError("blackman_window: unsupported dtype");
    }

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor bartlett_window(int64_t M, bool periodic, DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("bartlett_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    switch (dtype) {
    case DType::Float32: {
        float *data = result.typed_data<float>();
        for (int64_t i = 0; i < M; ++i) {
            // Triangular window: 1 - |2i/N - 1|
            double val = 1.0 - std::abs(2.0 * i / N - 1.0);
            data[i] = static_cast<float>(val);
        }
        break;
    }
    case DType::Float64: {
        double *data = result.typed_data<double>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = 1.0 - std::abs(2.0 * i / N - 1.0);
        }
        break;
    }
    default:
        throw TypeError("bartlett_window: unsupported dtype");
    }

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

// Modified Bessel function of the first kind, order 0
// Used for Kaiser window
namespace {
double bessel_i0(double x) {
    // Polynomial approximation
    double ax = std::abs(x);
    if (ax < 3.75) {
        double y = x / 3.75;
        y = y * y;
        return 1.0 + y * (3.5156229 +
                          y * (3.0899424 +
                               y * (1.2067492 +
                                    y * (0.2659732 +
                                         y * (0.0360768 + y * 0.0045813)))));
    } else {
        double y = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) *
               (0.39894228 +
                y * (0.01328592 +
                     y * (0.00225319 +
                          y * (-0.00157565 +
                               y * (0.00916281 +
                                    y * (-0.02057706 +
                                         y * (0.02635537 +
                                              y * (-0.01647633 +
                                                   y * 0.00392377))))))));
    }
}
} // namespace

Tensor kaiser_window(int64_t M, double beta, bool periodic, DType dtype,
                     Device device) {
    if (M <= 0) {
        throw ValueError("kaiser_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    double alpha = static_cast<double>(N) / 2.0;
    double i0_beta = bessel_i0(beta);

    switch (dtype) {
    case DType::Float32: {
        float *data = result.typed_data<float>();
        for (int64_t i = 0; i < M; ++i) {
            double ratio = (i - alpha) / alpha;
            double arg = beta * std::sqrt(1.0 - ratio * ratio);
            data[i] = static_cast<float>(bessel_i0(arg) / i0_beta);
        }
        break;
    }
    case DType::Float64: {
        double *data = result.typed_data<double>();
        for (int64_t i = 0; i < M; ++i) {
            double ratio = (i - alpha) / alpha;
            double arg = beta * std::sqrt(1.0 - ratio * ratio);
            data[i] = bessel_i0(arg) / i0_beta;
        }
        break;
    }
    default:
        throw TypeError("kaiser_window: unsupported dtype");
    }

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

} // namespace fft
} // namespace axiom
