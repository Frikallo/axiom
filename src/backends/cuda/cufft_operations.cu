#include "cufft_operations.hpp"

#ifdef AXIOM_CUDA_SUPPORT

#include "cuda_context.hpp"

#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace axiom {
namespace backends {
namespace cuda {

namespace {

static constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Error checking
// ============================================================================

void check_cufft(cufftResult result, const char *msg) {
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT error in ") + msg +
                                 " (code " + std::to_string(result) + ")");
    }
}

// ============================================================================
// Normalization scaling kernels
// ============================================================================

__global__ void scale_complex_f32_kernel(cufftComplex *data, size_t n,
                                         float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void scale_complex_f64_kernel(cufftDoubleComplex *data, size_t n,
                                         double scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void scale_real_f32_kernel(float *data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void scale_real_f64_kernel(double *data, size_t n, double scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// ============================================================================
// Helpers
// ============================================================================

// cuFFT produces unnormalized output.
// backward: forward=1, inverse=1/n
// forward:  forward=1/n, inverse=1
// ortho:    forward=1/√n, inverse=1/√n
double compute_scale(size_t n, const std::string &norm, bool inverse) {
    if (norm == "backward") {
        return inverse ? 1.0 / static_cast<double>(n) : 1.0;
    } else if (norm == "forward") {
        return inverse ? 1.0 : 1.0 / static_cast<double>(n);
    } else if (norm == "ortho") {
        return 1.0 / std::sqrt(static_cast<double>(n));
    }
    return 1.0;
}

// Move the FFT axis to the last position so cuFFT can process contiguous data.
// Returns {tensor_with_axis_last, needed_transpose}.
std::pair<Tensor, bool> move_axis_to_last(const Tensor &input, int axis) {
    int ndim = static_cast<int>(input.ndim());
    if (axis < 0)
        axis += ndim;
    if (axis == ndim - 1)
        return {input, false};

    std::vector<int> perm;
    for (int i = 0; i < ndim; ++i) {
        if (i != axis)
            perm.push_back(i);
    }
    perm.push_back(axis);
    return {input.transpose(perm).ascontiguousarray(), true};
}

// Inverse of move_axis_to_last: move last axis back to its original position.
Tensor move_last_to_axis(const Tensor &result, int axis, int orig_ndim) {
    if (axis < 0)
        axis += orig_ndim;
    int ndim = static_cast<int>(result.ndim());
    if (axis == ndim - 1)
        return result;

    std::vector<int> perm;
    for (int i = 0; i < axis; ++i)
        perm.push_back(i);
    perm.push_back(ndim - 1);
    for (int i = axis; i < ndim - 1; ++i)
        perm.push_back(i);
    return result.transpose(perm).ascontiguousarray();
}

// Pad or truncate a GPU tensor along the last axis using device-to-device copy.
Tensor pad_or_truncate_last(const Tensor &input, size_t target) {
    size_t current = input.shape().back();
    if (current == target)
        return input;

    Shape out_shape = input.shape();
    out_shape.back() = target;
    size_t batch = input.numel() / current;
    size_t copy_count = std::min(current, target);
    size_t elem = dtype_size(input.dtype());

    Tensor result = Tensor::zeros(out_shape, input.dtype(), Device::GPU);
    auto *stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    cudaMemcpy2DAsync(result.data(), target * elem, input.data(), current * elem,
                      copy_count * elem, batch, cudaMemcpyDeviceToDevice,
                      stream);
    return result;
}

// Apply complex-float scaling on the stream.
void apply_scale_cf32(cufftComplex *data, size_t n, double scale,
                      cudaStream_t stream) {
    if (scale == 1.0)
        return;
    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    scale_complex_f32_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        data, n, static_cast<float>(scale));
}

// Apply complex-double scaling on the stream.
void apply_scale_cf64(cufftDoubleComplex *data, size_t n, double scale,
                      cudaStream_t stream) {
    if (scale == 1.0)
        return;
    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    scale_complex_f64_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data, n, scale);
}

// Apply real-float scaling on the stream.
void apply_scale_rf32(float *data, size_t n, double scale,
                      cudaStream_t stream) {
    if (scale == 1.0)
        return;
    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    scale_real_f32_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        data, n, static_cast<float>(scale));
}

// Apply real-double scaling on the stream.
void apply_scale_rf64(double *data, size_t n, double scale,
                      cudaStream_t stream) {
    if (scale == 1.0)
        return;
    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    scale_real_f64_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data, n, scale);
}

// Ensure tensor is complex, contiguous, and on GPU.
Tensor ensure_complex_contiguous(const Tensor &input, DType target_dtype) {
    Tensor work = input;
    if (!is_complex_dtype(work.dtype()))
        work = work.astype(target_dtype);
    else if (work.dtype() != target_dtype)
        work = work.astype(target_dtype);
    if (!work.is_contiguous())
        work = work.ascontiguousarray();
    return work;
}

} // anonymous namespace

// ============================================================================
// 1D Complex-to-Complex FFT  (fft / ifft)
// ============================================================================

Tensor cufft_c2c_1d(const Tensor &input, size_t fft_size, int axis,
                     bool inverse, const std::string &norm) {
    int ndim = static_cast<int>(input.ndim());
    auto [transposed, needs_untranspose] = move_axis_to_last(input, axis);

    DType result_dtype = input.dtype() == DType::Complex128 ? DType::Complex128
                                                            : DType::Complex64;
    Tensor work = ensure_complex_contiguous(transposed, result_dtype);
    work = pad_or_truncate_last(work, fft_size);

    size_t batch = work.numel() / fft_size;
    Tensor result(work.shape(), result_dtype, Device::GPU);
    auto *stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    int dir = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    double scale = compute_scale(fft_size, norm, inverse);

    if (result_dtype == DType::Complex64) {
        cufftHandle plan;
        check_cufft(cufftPlan1d(&plan, static_cast<int>(fft_size), CUFFT_C2C,
                                static_cast<int>(batch)),
                    "cufftPlan1d C2C");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr =
            reinterpret_cast<cufftComplex *>(work.typed_data<complex64_t>());
        auto *out_ptr =
            reinterpret_cast<cufftComplex *>(result.typed_data<complex64_t>());

        check_cufft(cufftExecC2C(plan, in_ptr, out_ptr, dir), "cufftExecC2C");
        apply_scale_cf32(out_ptr, result.numel(), scale, stream);
        cufftDestroy(plan);
    } else {
        cufftHandle plan;
        check_cufft(cufftPlan1d(&plan, static_cast<int>(fft_size), CUFFT_Z2Z,
                                static_cast<int>(batch)),
                    "cufftPlan1d Z2Z");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr = reinterpret_cast<cufftDoubleComplex *>(
            work.typed_data<complex128_t>());
        auto *out_ptr = reinterpret_cast<cufftDoubleComplex *>(
            result.typed_data<complex128_t>());

        check_cufft(cufftExecZ2Z(plan, in_ptr, out_ptr, dir), "cufftExecZ2Z");
        apply_scale_cf64(out_ptr, result.numel(), scale, stream);
        cufftDestroy(plan);
    }

    CudaExecutionStream::instance().increment_batch();

    if (needs_untranspose)
        return move_last_to_axis(result, axis, ndim);
    return result;
}

// ============================================================================
// 1D Real-to-Complex FFT  (rfft)
// ============================================================================

Tensor cufft_r2c_1d(const Tensor &input, size_t fft_size, int axis,
                     const std::string &norm) {
    int ndim = static_cast<int>(input.ndim());
    auto [transposed, needs_untranspose] = move_axis_to_last(input, axis);

    bool is_double = (input.dtype() == DType::Float64);
    DType real_dtype = is_double ? DType::Float64 : DType::Float32;
    DType complex_dtype = is_double ? DType::Complex128 : DType::Complex64;

    Tensor work = transposed;
    if (work.dtype() != real_dtype)
        work = work.astype(real_dtype);
    if (!work.is_contiguous())
        work = work.ascontiguousarray();
    work = pad_or_truncate_last(work, fft_size);

    size_t batch = work.numel() / fft_size;
    size_t out_axis_size = fft_size / 2 + 1;

    Shape out_shape = work.shape();
    out_shape.back() = out_axis_size;
    Tensor result(out_shape, complex_dtype, Device::GPU);

    auto *stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    double scale = compute_scale(fft_size, norm, /*inverse=*/false);

    if (!is_double) {
        cufftHandle plan;
        int n_arr[] = {static_cast<int>(fft_size)};
        check_cufft(
            cufftPlanMany(&plan, 1, n_arr, nullptr, 1,
                          static_cast<int>(fft_size), nullptr, 1,
                          static_cast<int>(out_axis_size), CUFFT_R2C,
                          static_cast<int>(batch)),
            "cufftPlanMany R2C");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr = work.typed_data<float>();
        auto *out_ptr =
            reinterpret_cast<cufftComplex *>(result.typed_data<complex64_t>());

        check_cufft(cufftExecR2C(plan, in_ptr, out_ptr), "cufftExecR2C");
        apply_scale_cf32(out_ptr, result.numel(), scale, stream);
        cufftDestroy(plan);
    } else {
        cufftHandle plan;
        int n_arr[] = {static_cast<int>(fft_size)};
        check_cufft(
            cufftPlanMany(&plan, 1, n_arr, nullptr, 1,
                          static_cast<int>(fft_size), nullptr, 1,
                          static_cast<int>(out_axis_size), CUFFT_D2Z,
                          static_cast<int>(batch)),
            "cufftPlanMany D2Z");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr = work.typed_data<double>();
        auto *out_ptr = reinterpret_cast<cufftDoubleComplex *>(
            result.typed_data<complex128_t>());

        check_cufft(cufftExecD2Z(plan, in_ptr, out_ptr), "cufftExecD2Z");
        apply_scale_cf64(out_ptr, result.numel(), scale, stream);
        cufftDestroy(plan);
    }

    CudaExecutionStream::instance().increment_batch();

    if (needs_untranspose)
        return move_last_to_axis(result, axis, ndim);
    return result;
}

// ============================================================================
// 1D Complex-to-Real inverse FFT  (irfft)
// ============================================================================

Tensor cufft_c2r_1d(const Tensor &input, size_t output_size, int axis,
                     const std::string &norm) {
    int ndim = static_cast<int>(input.ndim());
    auto [transposed, needs_untranspose] = move_axis_to_last(input, axis);

    bool is_double = (input.dtype() == DType::Complex128);
    DType real_dtype = is_double ? DType::Float64 : DType::Float32;
    DType complex_dtype = is_double ? DType::Complex128 : DType::Complex64;

    Tensor work = ensure_complex_contiguous(transposed, complex_dtype);

    // C2R expects input of size output_size/2+1
    size_t expected_input = output_size / 2 + 1;
    work = pad_or_truncate_last(work, expected_input);

    size_t batch = work.numel() / expected_input;

    Shape out_shape = work.shape();
    out_shape.back() = output_size;
    Tensor result(out_shape, real_dtype, Device::GPU);

    auto *stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    double scale = compute_scale(output_size, norm, /*inverse=*/true);

    if (!is_double) {
        cufftHandle plan;
        int n_arr[] = {static_cast<int>(output_size)};
        check_cufft(
            cufftPlanMany(&plan, 1, n_arr, nullptr, 1,
                          static_cast<int>(expected_input), nullptr, 1,
                          static_cast<int>(output_size), CUFFT_C2R,
                          static_cast<int>(batch)),
            "cufftPlanMany C2R");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr =
            reinterpret_cast<cufftComplex *>(work.typed_data<complex64_t>());
        auto *out_ptr = result.typed_data<float>();

        check_cufft(cufftExecC2R(plan, in_ptr, out_ptr), "cufftExecC2R");
        apply_scale_rf32(out_ptr, result.numel(), scale, stream);
        cufftDestroy(plan);
    } else {
        cufftHandle plan;
        int n_arr[] = {static_cast<int>(output_size)};
        check_cufft(
            cufftPlanMany(&plan, 1, n_arr, nullptr, 1,
                          static_cast<int>(expected_input), nullptr, 1,
                          static_cast<int>(output_size), CUFFT_Z2D,
                          static_cast<int>(batch)),
            "cufftPlanMany Z2D");
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr = reinterpret_cast<cufftDoubleComplex *>(
            work.typed_data<complex128_t>());
        auto *out_ptr = result.typed_data<double>();

        check_cufft(cufftExecZ2D(plan, in_ptr, out_ptr), "cufftExecZ2D");
        apply_scale_rf64(out_ptr, result.numel(), scale, stream);
        cufftDestroy(plan);
    }

    CudaExecutionStream::instance().increment_batch();

    if (needs_untranspose)
        return move_last_to_axis(result, axis, ndim);
    return result;
}

// ============================================================================
// 2D Complex-to-Complex FFT  (fft2 / ifft2)
// ============================================================================

Tensor cufft_c2c_2d(const Tensor &input, size_t rows, size_t cols,
                     bool inverse, const std::string &norm) {
    int ndim = static_cast<int>(input.ndim());
    if (ndim < 2) {
        throw std::runtime_error("cufft_c2c_2d requires ndim >= 2");
    }

    DType result_dtype = input.dtype() == DType::Complex128 ? DType::Complex128
                                                            : DType::Complex64;
    Tensor work = ensure_complex_contiguous(input, result_dtype);

    size_t batch = 1;
    for (int d = 0; d < ndim - 2; ++d)
        batch *= work.shape()[d];

    Tensor result(work.shape(), result_dtype, Device::GPU);
    auto *stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    int dir = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    double scale0 = compute_scale(rows, norm, inverse);
    double scale1 = compute_scale(cols, norm, inverse);
    double total_scale = scale0 * scale1;

    if (result_dtype == DType::Complex64) {
        cufftHandle plan;
        if (batch == 1) {
            check_cufft(cufftPlan2d(&plan, static_cast<int>(rows),
                                    static_cast<int>(cols), CUFFT_C2C),
                        "cufftPlan2d C2C");
        } else {
            int n_arr[] = {static_cast<int>(rows), static_cast<int>(cols)};
            int dist = static_cast<int>(rows * cols);
            check_cufft(cufftPlanMany(&plan, 2, n_arr, nullptr, 1, dist,
                                      nullptr, 1, dist, CUFFT_C2C,
                                      static_cast<int>(batch)),
                        "cufftPlanMany 2D C2C");
        }
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr =
            reinterpret_cast<cufftComplex *>(work.typed_data<complex64_t>());
        auto *out_ptr =
            reinterpret_cast<cufftComplex *>(result.typed_data<complex64_t>());

        check_cufft(cufftExecC2C(plan, in_ptr, out_ptr, dir),
                    "cufftExecC2C 2D");
        apply_scale_cf32(out_ptr, result.numel(), total_scale, stream);
        cufftDestroy(plan);
    } else {
        cufftHandle plan;
        if (batch == 1) {
            check_cufft(cufftPlan2d(&plan, static_cast<int>(rows),
                                    static_cast<int>(cols), CUFFT_Z2Z),
                        "cufftPlan2d Z2Z");
        } else {
            int n_arr[] = {static_cast<int>(rows), static_cast<int>(cols)};
            int dist = static_cast<int>(rows * cols);
            check_cufft(cufftPlanMany(&plan, 2, n_arr, nullptr, 1, dist,
                                      nullptr, 1, dist, CUFFT_Z2Z,
                                      static_cast<int>(batch)),
                        "cufftPlanMany 2D Z2Z");
        }
        check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

        auto *in_ptr = reinterpret_cast<cufftDoubleComplex *>(
            work.typed_data<complex128_t>());
        auto *out_ptr = reinterpret_cast<cufftDoubleComplex *>(
            result.typed_data<complex128_t>());

        check_cufft(cufftExecZ2Z(plan, in_ptr, out_ptr, dir),
                    "cufftExecZ2Z 2D");
        apply_scale_cf64(out_ptr, result.numel(), total_scale, stream);
        cufftDestroy(plan);
    }

    CudaExecutionStream::instance().increment_batch();
    return result;
}

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif // AXIOM_CUDA_SUPPORT
