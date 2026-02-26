#include "axiom/fft.hpp"
#include "axiom/dispatch.hpp"
#include "axiom/error.hpp"
#include "axiom/operations.hpp"

#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef AXIOM_USE_ACCELERATE
#include "backends/cpu/vdsp.hpp"
#include <Accelerate/Accelerate.h>
#endif

#ifdef AXIOM_CUDA_SUPPORT
#include "backends/cuda/cufft_operations.hpp"
#endif

namespace axiom {
namespace fft {

namespace {

// Check if n is a power of 2
bool is_power_of_2(size_t n) { return n > 0 && (n & (n - 1)) == 0; }

#ifdef AXIOM_USE_ACCELERATE
static inline int log2_int(size_t n) {
    int log2n = 0;
    while ((1UL << log2n) < n) {
        log2n++;
    }
    return log2n;
}

// Cached FFTSetup objects — twiddle factor tables are reusable and expensive
// to create. Keyed by log2n, never freed during runtime.
FFTSetup get_cached_fft_setup(int log2n) {
    // Max supported: 2^24 = 16M points
    static constexpr int kMaxLog2n = 24;
    static FFTSetup cache[kMaxLog2n + 1] = {};
    static std::mutex mtx;

    if (log2n < 0 || log2n > kMaxLog2n)
        return nullptr;
    FFTSetup s = cache[log2n];
    if (s)
        return s;
    std::lock_guard<std::mutex> lock(mtx);
    if (cache[log2n])
        return cache[log2n]; // double-check
    cache[log2n] = vDSP_create_fftsetup(log2n, FFT_RADIX2);
    return cache[log2n];
}

FFTSetupD get_cached_fft_setupD(int log2n) {
    static constexpr int kMaxLog2n = 24;
    static FFTSetupD cache[kMaxLog2n + 1] = {};
    static std::mutex mtx;

    if (log2n < 0 || log2n > kMaxLog2n)
        return nullptr;
    FFTSetupD s = cache[log2n];
    if (s)
        return s;
    std::lock_guard<std::mutex> lock(mtx);
    if (cache[log2n])
        return cache[log2n];
    cache[log2n] = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
    return cache[log2n];
}
#endif

// Get normalization factor (additional multiplier beyond what get_total_scale
// computes). Used internally by get_total_scale.
double get_norm_factor(size_t n, const std::string &norm, bool inverse) {
    if (norm == "backward") {
        return 1.0;
    } else if (norm == "forward") {
        return inverse ? static_cast<double>(n) : 1.0 / n;
    } else if (norm == "ortho") {
        return inverse ? std::sqrt(static_cast<double>(n))
                       : 1.0 / std::sqrt(static_cast<double>(n));
    }
    throw ValueError("fft: norm must be 'backward', 'forward', or 'ortho'");
}

// Compute combined scale factor (including inverse 1/n and norm)
// This is the total multiplier to apply after a RAW (unnormalized) FFT.
double get_total_scale(size_t n, const std::string &norm, bool inverse) {
    double norm_factor = get_norm_factor(n, norm, inverse);
    if (inverse) {
        // Raw FFT has no normalization; we need 1/n for inverse + norm_factor
        return norm_factor / static_cast<double>(n);
    }
    return norm_factor;
}

// ============================================================================
// Real-input FFT: bypass to_complex() by feeding real data directly
// ============================================================================

#ifdef AXIOM_USE_ACCELERATE
// Process 1D FFT slices with real (float32) input, writing complex64 output.
// Avoids the expensive to_complex() allocation and conversion.
void process_slices_real_f32(const float *src, complex64_t *dst,
                             size_t input_size, size_t fft_size,
                             int64_t axis_stride_in, int64_t axis_stride_out,
                             const std::vector<size_t> &outer_shape,
                             const std::vector<int64_t> &src_outer_strides,
                             const std::vector<int64_t> &dst_outer_strides,
                             size_t n_slices, bool inverse,
                             const std::string &norm) {
    if (!is_power_of_2(fft_size) || fft_size == 0)
        return; // caller falls back to complex path
    float total_scale =
        static_cast<float>(get_total_scale(fft_size, norm, inverse));
    size_t copy_count = std::min(input_size, fft_size);

    int log2n = log2_int(fft_size);
    FFTSetup setup = get_cached_fft_setup(log2n);
    if (!setup)
        return;

    std::vector<float> real_part(fft_size), imag_part(fft_size, 0.0f);
    DSPSplitComplex split = {real_part.data(), imag_part.data()};

    std::vector<size_t> outer_idx(outer_shape.size(), 0);
    FFTDirection fft_dir =
        inverse ? kFFTDirection_Inverse : kFFTDirection_Forward;

    for (size_t slice = 0; slice < n_slices; ++slice) {
        size_t src_base = 0, dst_base = 0;
        for (size_t i = 0; i < outer_idx.size(); ++i) {
            src_base += outer_idx[i] * src_outer_strides[i];
            dst_base += outer_idx[i] * dst_outer_strides[i];
        }

        // Copy real data directly into split.realp, zero imag
        if (axis_stride_in == 1 && copy_count == fft_size) {
            std::memcpy(real_part.data(), src + src_base,
                        fft_size * sizeof(float));
        } else {
            std::fill(real_part.begin(), real_part.end(), 0.0f);
            const float *slice_src = src + src_base;
            for (size_t i = 0; i < copy_count; ++i) {
                real_part[i] = slice_src[i * axis_stride_in];
            }
        }
        std::fill(imag_part.begin(), imag_part.end(), 0.0f);

        vDSP_fft_zip(setup, &split, 1, log2n, fft_dir);

        if (total_scale != 1.0f) {
            vDSP_vsmul(real_part.data(), 1, &total_scale, real_part.data(), 1,
                       fft_size);
            vDSP_vsmul(imag_part.data(), 1, &total_scale, imag_part.data(), 1,
                       fft_size);
        }

        if (axis_stride_out == 1) {
            vDSP_ztoc(&split, 1, (DSPComplex *)(dst + dst_base), 2, fft_size);
        } else {
            complex64_t *slice_dst = dst + dst_base;
            for (size_t i = 0; i < fft_size; ++i) {
                slice_dst[i * axis_stride_out] =
                    complex64_t(real_part[i], imag_part[i]);
            }
        }

        for (int d = static_cast<int>(outer_idx.size()) - 1; d >= 0; --d) {
            if (++outer_idx[d] < outer_shape[d])
                break;
            outer_idx[d] = 0;
        }
    }
}

void process_slices_real_f64(const double *src, complex128_t *dst,
                             size_t input_size, size_t fft_size,
                             int64_t axis_stride_in, int64_t axis_stride_out,
                             const std::vector<size_t> &outer_shape,
                             const std::vector<int64_t> &src_outer_strides,
                             const std::vector<int64_t> &dst_outer_strides,
                             size_t n_slices, bool inverse,
                             const std::string &norm) {
    if (!is_power_of_2(fft_size) || fft_size == 0)
        return;
    double total_scale = get_total_scale(fft_size, norm, inverse);
    size_t copy_count = std::min(input_size, fft_size);

    int log2n = log2_int(fft_size);
    FFTSetupD setup = get_cached_fft_setupD(log2n);
    if (!setup)
        return;

    std::vector<double> real_part(fft_size), imag_part(fft_size, 0.0);
    DSPDoubleSplitComplex split = {real_part.data(), imag_part.data()};

    std::vector<size_t> outer_idx(outer_shape.size(), 0);
    FFTDirection fft_dir =
        inverse ? kFFTDirection_Inverse : kFFTDirection_Forward;

    for (size_t slice = 0; slice < n_slices; ++slice) {
        size_t src_base = 0, dst_base = 0;
        for (size_t i = 0; i < outer_idx.size(); ++i) {
            src_base += outer_idx[i] * src_outer_strides[i];
            dst_base += outer_idx[i] * dst_outer_strides[i];
        }

        if (axis_stride_in == 1 && copy_count == fft_size) {
            std::memcpy(real_part.data(), src + src_base,
                        fft_size * sizeof(double));
        } else {
            std::fill(real_part.begin(), real_part.end(), 0.0);
            const double *slice_src = src + src_base;
            for (size_t i = 0; i < copy_count; ++i) {
                real_part[i] = slice_src[i * axis_stride_in];
            }
        }
        std::fill(imag_part.begin(), imag_part.end(), 0.0);

        vDSP_fft_zipD(setup, &split, 1, log2n, fft_dir);

        if (total_scale != 1.0) {
            vDSP_vsmulD(real_part.data(), 1, &total_scale, real_part.data(), 1,
                        fft_size);
            vDSP_vsmulD(imag_part.data(), 1, &total_scale, imag_part.data(), 1,
                        fft_size);
        }

        if (axis_stride_out == 1) {
            vDSP_ztocD(&split, 1, (DSPDoubleComplex *)(dst + dst_base), 2,
                       fft_size);
        } else {
            complex128_t *slice_dst = dst + dst_base;
            for (size_t i = 0; i < fft_size; ++i) {
                slice_dst[i * axis_stride_out] =
                    complex128_t(real_part[i], imag_part[i]);
            }
        }

        for (int d = static_cast<int>(outer_idx.size()) - 1; d >= 0; --d) {
            if (++outer_idx[d] < outer_shape[d])
                break;
            outer_idx[d] = 0;
        }
    }
}
#endif // AXIOM_USE_ACCELERATE

// ============================================================================
// Optimized 1D FFT slice processing
// ============================================================================

// Process all 1D FFT slices using direct pointer access (Complex64 / float32)
void process_slices_f32(const complex64_t *src, complex64_t *dst,
                        size_t input_size, size_t fft_size,
                        int64_t axis_stride_in, int64_t axis_stride_out,
                        const std::vector<size_t> &outer_shape,
                        const std::vector<int64_t> &src_outer_strides,
                        const std::vector<int64_t> &dst_outer_strides,
                        size_t n_slices, bool inverse,
                        const std::string &norm) {
    float total_scale =
        static_cast<float>(get_total_scale(fft_size, norm, inverse));
    size_t copy_count = std::min(input_size, fft_size);

#ifdef AXIOM_USE_ACCELERATE
    if (is_power_of_2(fft_size) && fft_size > 0) {
        int log2n = log2_int(fft_size);
        FFTSetup setup = get_cached_fft_setup(log2n);
        if (!setup)
            return;

        // Pre-allocate split complex buffers (reused across slices)
        std::vector<float> real_part(fft_size), imag_part(fft_size);
        DSPSplitComplex split = {real_part.data(), imag_part.data()};

        std::vector<size_t> outer_idx(outer_shape.size(), 0);
        FFTDirection fft_dir =
            inverse ? kFFTDirection_Inverse : kFFTDirection_Forward;
        bool contiguous_full = (axis_stride_in == 1 && copy_count == fft_size);

        for (size_t slice = 0; slice < n_slices; ++slice) {
            size_t src_base = 0, dst_base = 0;
            for (size_t i = 0; i < outer_idx.size(); ++i) {
                src_base += outer_idx[i] * src_outer_strides[i];
                dst_base += outer_idx[i] * dst_outer_strides[i];
            }

            if (contiguous_full) {
                // vDSP_ctoz overwrites everything — no zeroing needed
                vDSP_ctoz((const DSPComplex *)(src + src_base), 2, &split, 1,
                          fft_size);
            } else {
                // Zero for padding, then fill what we have
                std::fill(real_part.begin(), real_part.end(), 0.0f);
                std::fill(imag_part.begin(), imag_part.end(), 0.0f);
                const complex64_t *slice_src = src + src_base;
                for (size_t i = 0; i < copy_count; ++i) {
                    auto val = slice_src[i * axis_stride_in];
                    real_part[i] = val.real();
                    imag_part[i] = val.imag();
                }
            }

            vDSP_fft_zip(setup, &split, 1, log2n, fft_dir);

            if (total_scale != 1.0f) {
                vDSP_vsmul(real_part.data(), 1, &total_scale, real_part.data(),
                           1, fft_size);
                vDSP_vsmul(imag_part.data(), 1, &total_scale, imag_part.data(),
                           1, fft_size);
            }

            if (axis_stride_out == 1) {
                vDSP_ztoc(&split, 1, (DSPComplex *)(dst + dst_base), 2,
                          fft_size);
            } else {
                complex64_t *slice_dst = dst + dst_base;
                for (size_t i = 0; i < fft_size; ++i) {
                    slice_dst[i * axis_stride_out] =
                        complex64_t(real_part[i], imag_part[i]);
                }
            }

            for (int d = static_cast<int>(outer_idx.size()) - 1; d >= 0; --d) {
                if (++outer_idx[d] < outer_shape[d])
                    break;
                outer_idx[d] = 0;
            }
        }

        return;
    }
#endif // AXIOM_USE_ACCELERATE

    // Generic path: uses pocketfft for all sizes (mixed-radix O(n log n))
    pocketfft::shape_t pf_shape = {fft_size};
    pocketfft::stride_t pf_stride = {
        static_cast<ptrdiff_t>(sizeof(std::complex<float>))};
    pocketfft::shape_t pf_axes = {0};

    std::vector<std::complex<float>> buffer(fft_size);
    std::vector<size_t> outer_idx(outer_shape.size(), 0);

    for (size_t slice = 0; slice < n_slices; ++slice) {
        size_t src_base = 0, dst_base = 0;
        for (size_t i = 0; i < outer_idx.size(); ++i) {
            src_base += outer_idx[i] * src_outer_strides[i];
            dst_base += outer_idx[i] * dst_outer_strides[i];
        }

        // Copy input to buffer with zero padding
        std::fill(buffer.begin(), buffer.end(), std::complex<float>(0, 0));
        const complex64_t *slice_src = src + src_base;
        for (size_t i = 0; i < copy_count; ++i) {
            buffer[i] = slice_src[i * axis_stride_in];
        }

        // FFT via pocketfft (unnormalized; total_scale includes all scaling)
        pocketfft::c2c(pf_shape, pf_stride, pf_stride, pf_axes, !inverse,
                       buffer.data(), buffer.data(), total_scale);

        // Copy to output
        complex64_t *slice_dst = dst + dst_base;
        for (size_t i = 0; i < fft_size; ++i) {
            slice_dst[i * axis_stride_out] = buffer[i];
        }

        for (int d = static_cast<int>(outer_idx.size()) - 1; d >= 0; --d) {
            if (++outer_idx[d] < outer_shape[d])
                break;
            outer_idx[d] = 0;
        }
    }
}

// Process all 1D FFT slices using direct pointer access (Complex128 / float64)
void process_slices_f64(const complex128_t *src, complex128_t *dst,
                        size_t input_size, size_t fft_size,
                        int64_t axis_stride_in, int64_t axis_stride_out,
                        const std::vector<size_t> &outer_shape,
                        const std::vector<int64_t> &src_outer_strides,
                        const std::vector<int64_t> &dst_outer_strides,
                        size_t n_slices, bool inverse,
                        const std::string &norm) {
    double total_scale = get_total_scale(fft_size, norm, inverse);
    size_t copy_count = std::min(input_size, fft_size);

#ifdef AXIOM_USE_ACCELERATE
    if (is_power_of_2(fft_size) && fft_size > 0) {
        int log2n = log2_int(fft_size);
        FFTSetupD setup = get_cached_fft_setupD(log2n);
        if (!setup)
            return;

        std::vector<double> real_part(fft_size), imag_part(fft_size);
        DSPDoubleSplitComplex split = {real_part.data(), imag_part.data()};

        std::vector<size_t> outer_idx(outer_shape.size(), 0);
        FFTDirection fft_dir =
            inverse ? kFFTDirection_Inverse : kFFTDirection_Forward;
        bool contiguous_full = (axis_stride_in == 1 && copy_count == fft_size);

        for (size_t slice = 0; slice < n_slices; ++slice) {
            size_t src_base = 0, dst_base = 0;
            for (size_t i = 0; i < outer_idx.size(); ++i) {
                src_base += outer_idx[i] * src_outer_strides[i];
                dst_base += outer_idx[i] * dst_outer_strides[i];
            }

            if (contiguous_full) {
                vDSP_ctozD((const DSPDoubleComplex *)(src + src_base), 2,
                           &split, 1, fft_size);
            } else {
                std::fill(real_part.begin(), real_part.end(), 0.0);
                std::fill(imag_part.begin(), imag_part.end(), 0.0);
                const complex128_t *slice_src = src + src_base;
                for (size_t i = 0; i < copy_count; ++i) {
                    auto val = slice_src[i * axis_stride_in];
                    real_part[i] = val.real();
                    imag_part[i] = val.imag();
                }
            }

            vDSP_fft_zipD(setup, &split, 1, log2n, fft_dir);

            if (total_scale != 1.0) {
                vDSP_vsmulD(real_part.data(), 1, &total_scale, real_part.data(),
                            1, fft_size);
                vDSP_vsmulD(imag_part.data(), 1, &total_scale, imag_part.data(),
                            1, fft_size);
            }

            if (axis_stride_out == 1) {
                vDSP_ztocD(&split, 1, (DSPDoubleComplex *)(dst + dst_base), 2,
                           fft_size);
            } else {
                complex128_t *slice_dst = dst + dst_base;
                for (size_t i = 0; i < fft_size; ++i) {
                    slice_dst[i * axis_stride_out] =
                        complex128_t(real_part[i], imag_part[i]);
                }
            }

            for (int d = static_cast<int>(outer_idx.size()) - 1; d >= 0; --d) {
                if (++outer_idx[d] < outer_shape[d])
                    break;
                outer_idx[d] = 0;
            }
        }

        return;
    }
#endif // AXIOM_USE_ACCELERATE

    // Generic path: uses pocketfft for all sizes (mixed-radix O(n log n))
    pocketfft::shape_t pf_shape = {fft_size};
    pocketfft::stride_t pf_stride = {
        static_cast<ptrdiff_t>(sizeof(std::complex<double>))};
    pocketfft::shape_t pf_axes = {0};

    std::vector<std::complex<double>> buffer(fft_size);
    std::vector<size_t> outer_idx(outer_shape.size(), 0);

    for (size_t slice = 0; slice < n_slices; ++slice) {
        size_t src_base = 0, dst_base = 0;
        for (size_t i = 0; i < outer_idx.size(); ++i) {
            src_base += outer_idx[i] * src_outer_strides[i];
            dst_base += outer_idx[i] * dst_outer_strides[i];
        }

        std::fill(buffer.begin(), buffer.end(), std::complex<double>(0, 0));
        const complex128_t *slice_src = src + src_base;
        for (size_t i = 0; i < copy_count; ++i) {
            buffer[i] = slice_src[i * axis_stride_in];
        }

        // FFT via pocketfft (unnormalized; total_scale includes all scaling)
        pocketfft::c2c(pf_shape, pf_stride, pf_stride, pf_axes, !inverse,
                       buffer.data(), buffer.data(), total_scale);

        complex128_t *slice_dst = dst + dst_base;
        for (size_t i = 0; i < fft_size; ++i) {
            slice_dst[i * axis_stride_out] = buffer[i];
        }

        for (int d = static_cast<int>(outer_idx.size()) - 1; d >= 0; --d) {
            if (++outer_idx[d] < outer_shape[d])
                break;
            outer_idx[d] = 0;
        }
    }
}

// ============================================================================
// Main 1D FFT implementation
// ============================================================================

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

#ifdef AXIOM_CUDA_SUPPORT
    if (input.device() == Device::GPU) {
        return backends::cuda::cufft_c2c_1d(input, fft_size, axis, inverse,
                                            norm);
    }
#endif

    // Move to CPU for computation
    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();

    DType result_dtype = input.dtype() == DType::Complex128 ? DType::Complex128
                                                            : DType::Complex64;

#ifdef AXIOM_USE_ACCELERATE
    // Fast path: real input with power-of-2 size — bypass to_complex()
    // by feeding real floats directly into the split-complex FFT buffers.
    if (!is_complex_dtype(input.dtype()) && is_power_of_2(fft_size) &&
        fft_size > 0) {
        Tensor real_input = input_cpu;
        // Promote to float32 if needed (e.g., int types)
        if (result_dtype == DType::Complex64 &&
            real_input.dtype() != DType::Float32) {
            real_input = real_input.astype(DType::Float32);
        } else if (result_dtype == DType::Complex128 &&
                   real_input.dtype() != DType::Float64) {
            real_input = real_input.astype(DType::Float64);
        }
        if (!real_input.is_contiguous()) {
            real_input = real_input.ascontiguousarray();
        }

        Shape output_shape = input.shape();
        output_shape[axis] = fft_size;
        Tensor result(output_shape, result_dtype, Device::CPU);

        size_t n_slices = 1;
        std::vector<size_t> outer_shape;
        std::vector<int64_t> src_outer_strides;
        std::vector<int64_t> dst_outer_strides;

        int64_t real_elem_size = static_cast<int64_t>(
            result_dtype == DType::Complex64 ? sizeof(float) : sizeof(double));
        int64_t cplx_elem_size = static_cast<int64_t>(
            result_dtype == DType::Complex64 ? sizeof(complex64_t)
                                             : sizeof(complex128_t));

        for (size_t d = 0; d < static_cast<size_t>(ndim); ++d) {
            if (static_cast<int>(d) != axis) {
                outer_shape.push_back(output_shape[d]);
                src_outer_strides.push_back(real_input.strides()[d] /
                                            real_elem_size);
                dst_outer_strides.push_back(result.strides()[d] /
                                            cplx_elem_size);
                n_slices *= output_shape[d];
            }
        }

        int64_t axis_stride_in = real_input.strides()[axis] / real_elem_size;
        int64_t axis_stride_out = result.strides()[axis] / cplx_elem_size;

        if (result_dtype == DType::Complex64) {
            process_slices_real_f32(
                real_input.typed_data<float>(),
                result.typed_data<complex64_t>(), input_size, fft_size,
                axis_stride_in, axis_stride_out, outer_shape, src_outer_strides,
                dst_outer_strides, n_slices, inverse, norm);
        } else {
            process_slices_real_f64(
                real_input.typed_data<double>(),
                result.typed_data<complex128_t>(), input_size, fft_size,
                axis_stride_in, axis_stride_out, outer_shape, src_outer_strides,
                dst_outer_strides, n_slices, inverse, norm);
        }

        if (input.device() == Device::GPU)
            return result.gpu();
        return result;
    }
#endif // AXIOM_USE_ACCELERATE

    // Standard path: convert to complex, then process
    Tensor complex_input = input_cpu;
    if (!is_complex_dtype(input.dtype())) {
        complex_input = input_cpu.to_complex();
    }
    if (complex_input.dtype() != result_dtype) {
        complex_input = complex_input.astype(result_dtype);
    }
    if (!complex_input.is_contiguous()) {
        complex_input = complex_input.ascontiguousarray();
    }

    Shape output_shape = input.shape();
    output_shape[axis] = fft_size;

    Tensor result(output_shape, result_dtype, Device::CPU);

    int64_t elem_size = static_cast<int64_t>(result_dtype == DType::Complex64
                                                 ? sizeof(complex64_t)
                                                 : sizeof(complex128_t));

    size_t n_slices = 1;
    std::vector<size_t> outer_shape;
    std::vector<int64_t> src_outer_strides;
    std::vector<int64_t> dst_outer_strides;

    for (size_t d = 0; d < static_cast<size_t>(ndim); ++d) {
        if (static_cast<int>(d) != axis) {
            outer_shape.push_back(output_shape[d]);
            src_outer_strides.push_back(complex_input.strides()[d] / elem_size);
            dst_outer_strides.push_back(result.strides()[d] / elem_size);
            n_slices *= output_shape[d];
        }
    }

    int64_t axis_stride_in = complex_input.strides()[axis] / elem_size;
    int64_t axis_stride_out = result.strides()[axis] / elem_size;

    if (result_dtype == DType::Complex64) {
        const complex64_t *src = complex_input.typed_data<complex64_t>();
        complex64_t *dst = result.typed_data<complex64_t>();
        process_slices_f32(src, dst, input_size, fft_size, axis_stride_in,
                           axis_stride_out, outer_shape, src_outer_strides,
                           dst_outer_strides, n_slices, inverse, norm);
    } else {
        const complex128_t *src = complex_input.typed_data<complex128_t>();
        complex128_t *dst = result.typed_data<complex128_t>();
        process_slices_f64(src, dst, input_size, fft_size, axis_stride_in,
                           axis_stride_out, outer_shape, src_outer_strides,
                           dst_outer_strides, n_slices, inverse, norm);
    }

    if (input.device() == Device::GPU) {
        return result.gpu();
    }
    return result;
}

// ============================================================================
// Native 2D FFT using vDSP (Accelerate only)
// ============================================================================

#ifdef AXIOM_USE_ACCELERATE
// Check if we can use the native 2D FFT path
bool can_use_native_2d_fft(const Tensor &input, const std::vector<int> &axes,
                           const std::vector<int64_t> &s, int ndim) {
    // Need exactly 2 axes
    if (axes.size() != 2)
        return false;

    // Both dimensions must be power of 2
    int ax0 = axes[0] < 0 ? axes[0] + ndim : axes[0];
    int ax1 = axes[1] < 0 ? axes[1] + ndim : axes[1];

    size_t rows = input.shape()[ax0];
    size_t cols = input.shape()[ax1];

    // No custom sizes (no padding/truncation)
    if (s.size() >= 1 && s[0] > 0 && static_cast<size_t>(s[0]) != rows)
        return false;
    if (s.size() >= 2 && s[1] > 0 && static_cast<size_t>(s[1]) != cols)
        return false;

    if (!is_power_of_2(rows) || !is_power_of_2(cols))
        return false;

    // Must be 2D (or we handle only the last 2 dims of ND)
    // Axes must be the last 2 dims
    if (ax0 != ndim - 2 || ax1 != ndim - 1)
        return false;

    return true;
}

// Perform native 2D FFT using vDSP_fft2d_zip
Tensor fft2d_native(const Tensor &input, const std::vector<int> &axes,
                    const std::string &norm, bool inverse) {
    int ndim = static_cast<int>(input.ndim());
    int ax0 = axes[0] < 0 ? axes[0] + ndim : axes[0];
    int ax1 = axes[1] < 0 ? axes[1] + ndim : axes[1];

    size_t rows = input.shape()[ax0];
    size_t cols = input.shape()[ax1];

    // Move to CPU and convert to complex
    Tensor input_cpu = input.device() == Device::CPU ? input : input.cpu();
    DType result_dtype = input.dtype() == DType::Complex128 ? DType::Complex128
                                                            : DType::Complex64;
    Tensor complex_input = input_cpu;
    if (!is_complex_dtype(input.dtype())) {
        complex_input = input_cpu.to_complex();
    }
    if (complex_input.dtype() != result_dtype) {
        complex_input = complex_input.astype(result_dtype);
    }
    if (!complex_input.is_contiguous()) {
        complex_input = complex_input.ascontiguousarray();
    }

    Tensor result(input.shape(), result_dtype, Device::CPU);

    // Compute combined scale for 2D: product of scale for each dimension
    double scale0 = get_total_scale(rows, norm, inverse);
    double scale1 = get_total_scale(cols, norm, inverse);
    double total_scale = scale0 * scale1;

    size_t matrix_size = rows * cols;
    size_t batch_size = 1;
    for (int d = 0; d < ndim - 2; ++d) {
        batch_size *= input.shape()[d];
    }

    int log2_rows = log2_int(rows);
    int log2_cols = log2_int(cols);

    if (result_dtype == DType::Complex64) {
        const complex64_t *src = complex_input.typed_data<complex64_t>();
        complex64_t *dst = result.typed_data<complex64_t>();

        FFTSetup setup = get_cached_fft_setup(std::max(log2_rows, log2_cols));
        if (!setup)
            return result;

        std::vector<float> real_part(matrix_size), imag_part(matrix_size);
        DSPSplitComplex split = {real_part.data(), imag_part.data()};
        FFTDirection fft_dir =
            inverse ? kFFTDirection_Inverse : kFFTDirection_Forward;

        for (size_t b = 0; b < batch_size; ++b) {
            const complex64_t *batch_src = src + b * matrix_size;
            complex64_t *batch_dst = dst + b * matrix_size;

            vDSP_ctoz((const DSPComplex *)batch_src, 2, &split, 1, matrix_size);

            vDSP_fft2d_zip(setup, &split, 1, 0, log2_cols, log2_rows, fft_dir);

            if (static_cast<float>(total_scale) != 1.0f) {
                float scale_f = static_cast<float>(total_scale);
                vDSP_vsmul(real_part.data(), 1, &scale_f, real_part.data(), 1,
                           matrix_size);
                vDSP_vsmul(imag_part.data(), 1, &scale_f, imag_part.data(), 1,
                           matrix_size);
            }

            vDSP_ztoc(&split, 1, (DSPComplex *)batch_dst, 2, matrix_size);
        }
    } else {
        const complex128_t *src = complex_input.typed_data<complex128_t>();
        complex128_t *dst = result.typed_data<complex128_t>();

        FFTSetupD setup = get_cached_fft_setupD(std::max(log2_rows, log2_cols));
        if (!setup)
            return result;

        std::vector<double> real_part(matrix_size), imag_part(matrix_size);
        DSPDoubleSplitComplex split = {real_part.data(), imag_part.data()};
        FFTDirection fft_dir =
            inverse ? kFFTDirection_Inverse : kFFTDirection_Forward;

        for (size_t b = 0; b < batch_size; ++b) {
            const complex128_t *batch_src = src + b * matrix_size;
            complex128_t *batch_dst = dst + b * matrix_size;

            vDSP_ctozD((const DSPDoubleComplex *)batch_src, 2, &split, 1,
                       matrix_size);

            vDSP_fft2d_zipD(setup, &split, 1, 0, log2_cols, log2_rows, fft_dir);

            if (total_scale != 1.0) {
                vDSP_vsmulD(real_part.data(), 1, &total_scale, real_part.data(),
                            1, matrix_size);
                vDSP_vsmulD(imag_part.data(), 1, &total_scale, imag_part.data(),
                            1, matrix_size);
            }

            vDSP_ztocD(&split, 1, (DSPDoubleComplex *)batch_dst, 2,
                       matrix_size);
        }
    }

    if (input.device() == Device::GPU) {
        return result.gpu();
    }
    return result;
}
#endif // AXIOM_USE_ACCELERATE

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

#ifdef AXIOM_CUDA_SUPPORT
    if (input.device() == Device::GPU) {
        size_t fft_size =
            (n > 0) ? static_cast<size_t>(n) : input.shape()[axis];
        return backends::cuda::cufft_r2c_1d(input, fft_size, axis, norm);
    }
#endif

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

#ifdef AXIOM_CUDA_SUPPORT
    if (input.device() == Device::GPU) {
        return backends::cuda::cufft_c2r_1d(input, output_size, axis, norm);
    }
#endif

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

#ifdef AXIOM_CUDA_SUPPORT
    if (input.device() == Device::GPU) {
        int ndim_gpu = static_cast<int>(input.ndim());
        int ax0 =
            actual_axes[0] < 0 ? actual_axes[0] + ndim_gpu : actual_axes[0];
        int ax1 =
            actual_axes[1] < 0 ? actual_axes[1] + ndim_gpu : actual_axes[1];
        bool no_custom = (actual_s[0] <= 0 && actual_s[1] <= 0);
        bool last_two = (ax0 == ndim_gpu - 2 && ax1 == ndim_gpu - 1);
        if (no_custom && last_two) {
            return backends::cuda::cufft_c2c_2d(
                input, input.shape()[ax0], input.shape()[ax1], false, norm);
        }
        // Fall through to composed 1D (each will use cuFFT).
    }
#endif

#ifdef AXIOM_USE_ACCELERATE
    int ndim = static_cast<int>(input.ndim());
    if (can_use_native_2d_fft(input, actual_axes, actual_s, ndim)) {
        return fft2d_native(input, actual_axes, norm, false);
    }
#endif

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

#ifdef AXIOM_CUDA_SUPPORT
    if (input.device() == Device::GPU) {
        int ndim_gpu = static_cast<int>(input.ndim());
        int ax0 =
            actual_axes[0] < 0 ? actual_axes[0] + ndim_gpu : actual_axes[0];
        int ax1 =
            actual_axes[1] < 0 ? actual_axes[1] + ndim_gpu : actual_axes[1];
        bool no_custom = (actual_s[0] <= 0 && actual_s[1] <= 0);
        bool last_two = (ax0 == ndim_gpu - 2 && ax1 == ndim_gpu - 1);
        if (no_custom && last_two) {
            return backends::cuda::cufft_c2c_2d(input, input.shape()[ax0],
                                                input.shape()[ax1], true, norm);
        }
    }
#endif

#ifdef AXIOM_USE_ACCELERATE
    int ndim = static_cast<int>(input.ndim());
    if (can_use_native_2d_fft(input, actual_axes, actual_s, ndim)) {
        return fft2d_native(input, actual_axes, norm, true);
    }
#endif

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

    dispatch_float(dtype, "fftfreq", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < n; ++i) {
            int64_t k = (i <= n / 2) ? i : i - n;
            data[i] = static_cast<T>(k * factor);
        }
    });

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

    dispatch_float(dtype, "rfftfreq", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < output_size; ++i) {
            data[i] = static_cast<T>(i * factor);
        }
    });

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

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    const double pi = M_PI;

    dispatch_float(dtype, "hann_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = static_cast<T>(0.5 - 0.5 * std::cos(2.0 * pi * i / N));
        }
    });

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

    dispatch_float(dtype, "hamming_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = static_cast<T>(alpha - beta * std::cos(2.0 * pi * i / N));
        }
    });

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

    dispatch_float(dtype, "blackman_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            double x = 2.0 * pi * i / N;
            data[i] =
                static_cast<T>(a0 - a1 * std::cos(x) + a2 * std::cos(2.0 * x));
        }
    });

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

    dispatch_float(dtype, "bartlett_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = static_cast<T>(1.0 - std::abs(2.0 * i / N - 1.0));
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

// Modified Bessel function of the first kind, order 0
// Used for Kaiser window
namespace {
double bessel_i0(double x) {
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

    dispatch_float(dtype, "kaiser_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            double ratio = (i - alpha) / alpha;
            double arg = beta * std::sqrt(1.0 - ratio * ratio);
            data[i] = static_cast<T>(bessel_i0(arg) / i0_beta);
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor cosine_window(int64_t M, bool periodic, DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("cosine_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    dispatch_float(dtype, "cosine_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = static_cast<T>(std::sin(M_PI * i / N));
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor exponential_window(int64_t M, double tau, bool periodic, DType dtype,
                          Device device) {
    if (M <= 0) {
        throw ValueError("exponential_window: M must be positive");
    }
    if (tau <= 0.0) {
        throw ValueError("exponential_window: tau must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    double center = static_cast<double>(N) / 2.0;

    dispatch_float(dtype, "exponential_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = static_cast<T>(std::exp(-std::abs(i - center) / tau));
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor gaussian_window(int64_t M, double std, bool periodic, DType dtype,
                       Device device) {
    if (M <= 0) {
        throw ValueError("gaussian_window: M must be positive");
    }
    if (std <= 0.0) {
        throw ValueError("gaussian_window: std must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    double center = static_cast<double>(N) / 2.0;

    dispatch_float(dtype, "gaussian_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            double ratio = (i - center) / std;
            data[i] = static_cast<T>(std::exp(-0.5 * ratio * ratio));
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor nuttall_window(int64_t M, bool periodic, DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("nuttall_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    constexpr double a0 = 0.3635819;
    constexpr double a1 = 0.4891775;
    constexpr double a2 = 0.1365995;
    constexpr double a3 = 0.0106411;

    dispatch_float(dtype, "nuttall_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            double x = 2.0 * M_PI * i / N;
            data[i] =
                static_cast<T>(a0 - a1 * std::cos(x) + a2 * std::cos(2.0 * x) -
                               a3 * std::cos(3.0 * x));
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor general_cosine_window(int64_t M, const std::vector<double> &a,
                             bool periodic, DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("general_cosine_window: M must be positive");
    }
    if (a.empty()) {
        throw ValueError(
            "general_cosine_window: coefficients must not be empty");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    dispatch_float(dtype, "general_cosine_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            double val = 0.0;
            double sign = 1.0;
            for (size_t k = 0; k < a.size(); ++k) {
                val += sign * a[k] * std::cos(2.0 * M_PI * k * i / N);
                sign = -sign;
            }
            data[i] = static_cast<T>(val);
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

Tensor general_hamming_window(int64_t M, double alpha, bool periodic,
                              DType dtype, Device device) {
    if (M <= 0) {
        throw ValueError("general_hamming_window: M must be positive");
    }

    Tensor result({static_cast<size_t>(M)}, dtype, Device::CPU);

    int64_t N = periodic ? M : M - 1;
    if (N == 0)
        N = 1;

    dispatch_float(dtype, "general_hamming_window", [&]<typename DT>(DT) {
        using T = typename DT::value_type;
        T *data = result.typed_data<T>();
        for (int64_t i = 0; i < M; ++i) {
            data[i] = static_cast<T>(alpha - (1.0 - alpha) *
                                                 std::cos(2.0 * M_PI * i / N));
        }
    });

    if (device == Device::GPU) {
        return result.gpu();
    }
    return result;
}

// ============================================================================
// Short-Time Fourier Transform
// ============================================================================

Tensor stft(const Tensor &input, int64_t n_fft, int64_t hop_length,
            int64_t win_length, const Tensor &window, bool center,
            const std::string &pad_mode, bool normalized) {
    if (input.ndim() < 1 || input.ndim() > 2) {
        throw ShapeError("stft: expected 1D or 2D input, got " +
                         std::to_string(input.ndim()) + "D");
    }
    if (n_fft <= 0) {
        throw ValueError("stft: n_fft must be > 0");
    }
    if (hop_length <= 0) {
        hop_length = n_fft / 4;
    }
    if (win_length <= 0) {
        win_length = n_fft;
    }
    if (win_length > n_fft) {
        throw ValueError("stft: win_length (" + std::to_string(win_length) +
                         ") must be <= n_fft (" + std::to_string(n_fft) + ")");
    }

    // Handle batched input
    bool has_batch = (input.ndim() == 2);
    Tensor signal = has_batch ? input : input.unsqueeze(0);
    size_t batch_size = signal.shape()[0];

    // Center padding (reflect mode)
    if (center) {
        int64_t pad_amount = n_fft / 2;
        signal = ops::pad(signal,
                          {{0, 0},
                           {static_cast<size_t>(pad_amount),
                            static_cast<size_t>(pad_amount)}},
                          pad_mode);
    }

    size_t signal_length = signal.shape()[1];
    if (static_cast<int64_t>(signal_length) < n_fft) {
        throw ShapeError(
            "stft: signal length (" + std::to_string(signal_length) +
            ") is less than n_fft (" + std::to_string(n_fft) + ")");
    }

    int64_t n_frames =
        (static_cast<int64_t>(signal_length) - n_fft) / hop_length + 1;
    int64_t freq_bins = n_fft / 2 + 1;

    // Prepare window
    Tensor win;
    if (window.ndim() > 0) {
        win = window.cpu();
        if (static_cast<int64_t>(win.size()) != win_length) {
            throw ShapeError(
                "stft: window size (" + std::to_string(win.size()) +
                ") must equal win_length (" + std::to_string(win_length) + ")");
        }
    } else {
        win = hann_window(win_length, true, DType::Float32, Device::CPU);
    }

    // Pad window to n_fft if needed
    if (win_length < n_fft) {
        int64_t left_pad = (n_fft - win_length) / 2;
        int64_t right_pad = n_fft - win_length - left_pad;
        win = ops::pad(
            win,
            {{static_cast<size_t>(left_pad), static_cast<size_t>(right_pad)}},
            "constant");
    }

    // Work on CPU
    Tensor cpu_signal = signal.device() == Device::CPU ? signal : signal.cpu();
    if (!cpu_signal.is_contiguous()) {
        cpu_signal = cpu_signal.ascontiguousarray();
    }

    // Output shape: (batch, freq_bins, n_frames)
    Shape out_shape = {batch_size, static_cast<size_t>(freq_bins),
                       static_cast<size_t>(n_frames)};
    Tensor result(out_shape, DType::Complex64, Device::CPU);

    // Extract frames, apply window, compute rfft for each frame
    for (size_t b = 0; b < batch_size; ++b) {
        for (int64_t f = 0; f < n_frames; ++f) {
            int64_t start_pos = f * hop_length;

            // Extract frame: signal[b, start:start+n_fft]
            auto frame = cpu_signal.slice(
                {Slice(static_cast<int64_t>(b), static_cast<int64_t>(b + 1)),
                 Slice(start_pos, start_pos + n_fft)});
            frame = frame.squeeze(0); // Remove batch dim

            // Apply window
            frame = ops::multiply(frame.astype(DType::Float32), win);

            // rfft
            auto spectrum = rfft(frame, n_fft);

            // Normalize if requested
            if (normalized) {
                auto norm_factor = Tensor::full(
                    {1}, 1.0f / std::sqrt(static_cast<float>(n_fft)));
                spectrum = ops::multiply(spectrum, norm_factor);
            }

            // Copy into result[:, :, f]
            if (!spectrum.is_contiguous()) {
                spectrum = spectrum.ascontiguousarray();
            }
            auto spec_data = spectrum.typed_data<complex64_t>();
            for (int64_t i = 0; i < freq_bins; ++i) {
                result.set_item<complex64_t>(
                    {b, static_cast<size_t>(i), static_cast<size_t>(f)},
                    spec_data[i]);
            }
        }
    }

    // Remove batch dim if input was 1D
    if (!has_batch) {
        result = result.squeeze(0);
    }

    if (input.device() == Device::GPU) {
        return result.gpu();
    }
    return result;
}

} // namespace fft
} // namespace axiom
