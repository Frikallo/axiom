#include "cublas_operations.hpp"
#include "cuda_buffer_provider.hpp"
#include "cuda_context.hpp"
#include "cuda_operations.hpp"

#include "axiom/dtype.hpp"
#include "axiom/error.hpp"
#include "axiom/operations.hpp"
#include "axiom/shape.hpp"
#include "axiom/tensor.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <string>

namespace axiom {
namespace backends {
namespace cuda {

#ifdef AXIOM_CUDA_SUPPORT

// ============================================================================
// Helper: extract M, N, K dimensions from tensors + transpose flags
// ============================================================================

static void get_matmul_dims(const Tensor &a, const Tensor &b,
                            bool transpose_a, bool transpose_b,
                            size_t &M, size_t &N, size_t &K, size_t &K_b) {
    size_t a_ndim = a.ndim();
    size_t b_ndim = b.ndim();

    size_t a_rows, a_cols, b_rows, b_cols;

    if (a_ndim == 1) {
        a_rows = 1;
        a_cols = a.shape()[0];
    } else {
        a_rows = a.shape()[a_ndim - 2];
        a_cols = a.shape()[a_ndim - 1];
    }

    if (b_ndim == 1) {
        b_rows = b.shape()[0];
        b_cols = 1;
    } else {
        b_rows = b.shape()[b_ndim - 2];
        b_cols = b.shape()[b_ndim - 1];
    }

    if (transpose_a) std::swap(a_rows, a_cols);
    if (transpose_b) std::swap(b_rows, b_cols);

    M = a_rows;
    K = a_cols;
    K_b = b_rows;
    N = b_cols;
}

// ============================================================================
// Helper: compute broadcasted batch shape
// ============================================================================

static Shape compute_batch_shape(const Tensor &a, const Tensor &b) {
    size_t a_batch_dims = a.ndim() > 2 ? a.ndim() - 2 : 0;
    size_t b_batch_dims = b.ndim() > 2 ? b.ndim() - 2 : 0;

    Shape a_batch, b_batch;
    for (size_t i = 0; i < a_batch_dims; ++i)
        a_batch.push_back(a.shape()[i]);
    for (size_t i = 0; i < b_batch_dims; ++i)
        b_batch.push_back(b.shape()[i]);

    return ShapeUtils::broadcast_shape(a_batch, b_batch);
}

// ============================================================================
// cuBLAS GEMM wrappers
// ============================================================================
// cuBLAS uses column-major layout.  Our tensors are row-major.
// For row-major C = A @ B we compute column-major C^T = B^T @ A^T.
// This means: swap A/B pointers and swap the transpose flags.
// ============================================================================

static void cublas_gemm_f32(cublasHandle_t handle,
                            bool trans_a, bool trans_b,
                            int M, int N, int K,
                            const float *a_ptr, int lda,
                            const float *b_ptr, int ldb,
                            float *c_ptr, int ldc) {
    // Row-major → column-major: C = A*B  becomes  C^T = B^T * A^T
    // So swap A↔B and swap their trans flags
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(
        handle, op_b, op_a,
        N, M, K,
        &alpha,
        b_ptr, ldb,
        a_ptr, lda,
        &beta,
        c_ptr, ldc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw RuntimeError::internal(
            "cublasSgemm failed with status " + std::to_string(status));
    }
}

static void cublas_gemm_f64(cublasHandle_t handle,
                            bool trans_a, bool trans_b,
                            int M, int N, int K,
                            const double *a_ptr, int lda,
                            const double *b_ptr, int ldb,
                            double *c_ptr, int ldc) {
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;

    double alpha = 1.0;
    double beta = 0.0;

    cublasStatus_t status = cublasDgemm(
        handle, op_b, op_a,
        N, M, K,
        &alpha,
        b_ptr, ldb,
        a_ptr, lda,
        &beta,
        c_ptr, ldc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw RuntimeError::internal(
            "cublasDgemm failed with status " + std::to_string(status));
    }
}

static void cublas_gemm_f16(cublasHandle_t handle,
                            bool trans_a, bool trans_b,
                            int M, int N, int K,
                            const void *a_ptr, int lda,
                            const void *b_ptr, int ldb,
                            void *c_ptr, int ldc) {
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Use float32 compute for fp16 inputs for better precision
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        handle, op_b, op_a,
        N, M, K,
        &alpha,
        b_ptr, CUDA_R_16F, ldb,
        a_ptr, CUDA_R_16F, lda,
        &beta,
        c_ptr, CUDA_R_16F, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw RuntimeError::internal(
            "cublasGemmEx (fp16) failed with status " +
            std::to_string(status));
    }
}

// ============================================================================
// Batched GEMM via cublasGemmStridedBatchedEx
// ============================================================================

static void cublas_gemm_strided_batched(
    cublasHandle_t handle, DType dtype,
    bool trans_a, bool trans_b,
    int M, int N, int K,
    const void *a_ptr, int lda, long long stride_a,
    const void *b_ptr, int ldb, long long stride_b,
    void *c_ptr, int ldc, long long stride_c,
    int batch_count) {

    // Row-major → column-major swap: C = A*B  →  C^T = B^T * A^T
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;

    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    double alpha_d = 1.0;
    double beta_d = 0.0;

    cublasStatus_t status;

    switch (dtype) {
    case DType::Float32:
        status = cublasGemmStridedBatchedEx(
            handle, op_b, op_a,
            N, M, K,
            &alpha_f,
            b_ptr, CUDA_R_32F, ldb, stride_b,
            a_ptr, CUDA_R_32F, lda, stride_a,
            &beta_f,
            c_ptr, CUDA_R_32F, ldc, stride_c,
            batch_count,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        break;
    case DType::Float64:
        status = cublasGemmStridedBatchedEx(
            handle, op_b, op_a,
            N, M, K,
            &alpha_d,
            b_ptr, CUDA_R_64F, ldb, stride_b,
            a_ptr, CUDA_R_64F, lda, stride_a,
            &beta_d,
            c_ptr, CUDA_R_64F, ldc, stride_c,
            batch_count,
            CUBLAS_COMPUTE_64F,
            CUBLAS_GEMM_DEFAULT);
        break;
    case DType::Float16:
        status = cublasGemmStridedBatchedEx(
            handle, op_b, op_a,
            N, M, K,
            &alpha_f,
            b_ptr, CUDA_R_16F, ldb, stride_b,
            a_ptr, CUDA_R_16F, lda, stride_a,
            &beta_f,
            c_ptr, CUDA_R_16F, ldc, stride_c,
            batch_count,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        break;
    default:
        throw TypeError::unsupported_dtype(dtype_name(dtype),
                                           "cuBLAS batched matmul");
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw RuntimeError::internal(
            "cublasGemmStridedBatchedEx failed with status " +
            std::to_string(status));
    }
}

// ============================================================================
// CublasMatMulOperation
// ============================================================================

class CublasMatMulOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::MatMul; }
    std::string name() const override { return "matmul"; }
    Device device() const override { return Device::GPU; }

    Tensor execute_binary(const Tensor & /*lhs*/,
                          const Tensor & /*rhs*/) const override {
        throw RuntimeError::internal(
            "Use execute_matmul for MatMul operations");
    }

    Tensor execute_unary(const Tensor & /*input*/) const override {
        throw RuntimeError::internal(
            "execute_unary called on matmul operation");
    }

    Tensor execute_reduction(const Tensor & /*input*/,
                             const std::vector<int> & /*axes*/,
                             bool /*keep_dims*/) const override {
        throw RuntimeError::internal(
            "execute_reduction called on matmul operation");
    }

    Tensor execute_matmul(const Tensor &a, const Tensor &b,
                          bool transpose_a,
                          bool transpose_b) const override {
        if (a.device() != Device::GPU || b.device() != Device::GPU) {
            throw DeviceError("cuBLAS MatMul requires GPU tensors");
        }

        if (a.ndim() == 0 || b.ndim() == 0) {
            throw ShapeError("MatMul does not support 0-dimensional tensors");
        }

        // Type promotion — cuBLAS supports f16, f32, f64
        DType result_dtype = ops::promote_types(a.dtype(), b.dtype());
        if (result_dtype != DType::Float32 && result_dtype != DType::Float64 &&
            result_dtype != DType::Float16) {
            result_dtype = DType::Float32;
        }

        // Cast inputs to common dtype if needed
        Tensor a_cast = (a.dtype() == result_dtype) ? a : a.astype(result_dtype);
        Tensor b_cast = (b.dtype() == result_dtype) ? b : b.astype(result_dtype);

        // Ensure contiguous
        Tensor a_c = ensure_gpu_contiguous(a_cast);
        Tensor b_c = ensure_gpu_contiguous(b_cast);

        // Compute M, N, K
        size_t M, N, K, K_b;
        get_matmul_dims(a_c, b_c, transpose_a, transpose_b, M, N, K, K_b);

        if (K != K_b) {
            throw ShapeError("MatMul dimension mismatch: A has " +
                             std::to_string(K) + " columns but B has " +
                             std::to_string(K_b) + " rows");
        }

        size_t a_ndim = a_c.ndim();
        size_t b_ndim = b_c.ndim();

        // Compute output shape
        Shape result_shape;
        bool is_batched = (a_ndim > 2 || b_ndim > 2);
        Shape batch_shape;

        if (is_batched) {
            batch_shape = compute_batch_shape(a_c, b_c);
            result_shape = batch_shape;
        }

        if (a_ndim == 1 && b_ndim == 1) {
            // Vector dot product → scalar
            result_shape = {};
        } else if (a_ndim == 1) {
            result_shape.push_back(N);
        } else if (b_ndim == 1) {
            result_shape.push_back(M);
        } else {
            result_shape.push_back(M);
            result_shape.push_back(N);
        }

        Tensor result(result_shape, result_dtype, Device::GPU);
        if (result.size() == 0) return result;

        auto *a_buf = as_cuda_buffer_provider(a_c.storage().get());
        auto *b_buf = as_cuda_buffer_provider(b_c.storage().get());
        auto *c_buf = as_cuda_buffer_provider(result.storage().get());
        if (!a_buf || !b_buf || !c_buf) {
            throw DeviceError("cuBLAS MatMul: storage is not CUDA-backed");
        }

        const void *a_ptr = a_buf->device_ptr();
        const void *b_ptr = b_buf->device_ptr();
        void *c_ptr = c_buf->device_ptr();

        auto handle = static_cast<cublasHandle_t>(
            CudaContext::instance().cublas_handle());

        // Leading dimensions for row-major contiguous layout.
        // For a row-major (M, K) matrix, the leading dimension when
        // interpreted as column-major by cuBLAS is K (the number of columns).
        // After the row↔col swap, cuBLAS sees B^T as (N, K) and A^T as (K, M).
        int lda = static_cast<int>(transpose_a ? M : K); // A's "width"
        int ldb = static_cast<int>(transpose_b ? K : N); // B's "width"
        int ldc = static_cast<int>(N);                    // C's "width"

        if (!is_batched) {
            // Simple 2D matmul
            switch (result_dtype) {
            case DType::Float32:
                cublas_gemm_f32(handle, transpose_a, transpose_b,
                                static_cast<int>(M), static_cast<int>(N),
                                static_cast<int>(K),
                                static_cast<const float *>(a_ptr), lda,
                                static_cast<const float *>(b_ptr), ldb,
                                static_cast<float *>(c_ptr), ldc);
                break;
            case DType::Float64:
                cublas_gemm_f64(handle, transpose_a, transpose_b,
                                static_cast<int>(M), static_cast<int>(N),
                                static_cast<int>(K),
                                static_cast<const double *>(a_ptr), lda,
                                static_cast<const double *>(b_ptr), ldb,
                                static_cast<double *>(c_ptr), ldc);
                break;
            case DType::Float16:
                cublas_gemm_f16(handle, transpose_a, transpose_b,
                                static_cast<int>(M), static_cast<int>(N),
                                static_cast<int>(K),
                                a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
                break;
            default:
                throw TypeError::unsupported_dtype(dtype_name(result_dtype),
                                                   "cuBLAS matmul");
            }
        } else {
            // Batched matmul
            size_t batch_size = ShapeUtils::size(batch_shape);
            size_t batch_ndim = batch_shape.size();

            // Check if we can use strided batched GEMM.
            // This requires uniform strides across all batch dimensions,
            // which is the case when both tensors have the same batch
            // shape (no broadcasting) and are contiguous.
            bool a_has_batch = (a_ndim > 2);
            bool b_has_batch = (b_ndim > 2);

            // Compute per-matrix strides
            size_t a_mat_size = (a_ndim >= 2)
                                    ? a_c.shape()[a_ndim - 2] * a_c.shape()[a_ndim - 1]
                                    : a_c.shape()[0];
            size_t b_mat_size = (b_ndim >= 2)
                                    ? b_c.shape()[b_ndim - 2] * b_c.shape()[b_ndim - 1]
                                    : b_c.shape()[0];
            size_t c_mat_size = M * N;

            // Simple case: no broadcast needed (batch shapes identical
            // or one side has no batch dims)
            bool can_use_strided = true;

            // Check for broadcast: if both have batch dims, shapes must match
            if (a_has_batch && b_has_batch) {
                Shape a_batch_shape(a_c.shape().begin(),
                                    a_c.shape().begin() + (a_ndim - 2));
                Shape b_batch_shape(b_c.shape().begin(),
                                    b_c.shape().begin() + (b_ndim - 2));
                if (a_batch_shape != b_batch_shape) {
                    can_use_strided = false;
                }
            }

            if (can_use_strided) {
                long long stride_a = a_has_batch
                                         ? static_cast<long long>(a_mat_size)
                                         : 0;
                long long stride_b = b_has_batch
                                         ? static_cast<long long>(b_mat_size)
                                         : 0;
                long long stride_c = static_cast<long long>(c_mat_size);

                cublas_gemm_strided_batched(
                    handle, result_dtype, transpose_a, transpose_b,
                    static_cast<int>(M), static_cast<int>(N),
                    static_cast<int>(K),
                    a_ptr, lda, stride_a,
                    b_ptr, ldb, stride_b,
                    c_ptr, ldc, stride_c,
                    static_cast<int>(batch_size));
            } else {
                // Fallback: loop over batch elements with broadcast.
                // Compute batch strides for A and B.
                size_t elem_size = dtype_size(result_dtype);
                Strides a_batch_strides(batch_ndim, 0);
                Strides b_batch_strides(batch_ndim, 0);

                size_t a_batch_offset =
                    batch_ndim - (a_ndim > 2 ? a_ndim - 2 : 0);
                size_t b_batch_offset =
                    batch_ndim - (b_ndim > 2 ? b_ndim - 2 : 0);

                for (size_t i = 0; i < batch_ndim; ++i) {
                    if (i >= a_batch_offset && a_ndim > 2) {
                        size_t dim_idx = i - a_batch_offset;
                        if (a_c.shape()[dim_idx] != 1) {
                            a_batch_strides[i] =
                                a_c.strides()[dim_idx] /
                                static_cast<int64_t>(elem_size);
                        }
                    }
                    if (i >= b_batch_offset && b_ndim > 2) {
                        size_t dim_idx = i - b_batch_offset;
                        if (b_c.shape()[dim_idx] != 1) {
                            b_batch_strides[i] =
                                b_c.strides()[dim_idx] /
                                static_cast<int64_t>(elem_size);
                        }
                    }
                }

                for (size_t batch_idx = 0; batch_idx < batch_size;
                     ++batch_idx) {
                    // Decompose flat batch_idx into coordinates
                    size_t remaining = batch_idx;
                    int64_t a_off = 0;
                    int64_t b_off = 0;

                    for (int d = static_cast<int>(batch_ndim) - 1; d >= 0;
                         --d) {
                        size_t coord = remaining % batch_shape[d];
                        remaining /= batch_shape[d];
                        a_off += static_cast<int64_t>(coord) *
                                 a_batch_strides[d];
                        b_off += static_cast<int64_t>(coord) *
                                 b_batch_strides[d];
                    }

                    int64_t c_off =
                        static_cast<int64_t>(batch_idx * c_mat_size);

                    const void *a_batch =
                        static_cast<const uint8_t *>(a_ptr) +
                        a_off * static_cast<int64_t>(elem_size);
                    const void *b_batch =
                        static_cast<const uint8_t *>(b_ptr) +
                        b_off * static_cast<int64_t>(elem_size);
                    void *c_batch =
                        static_cast<uint8_t *>(c_ptr) +
                        c_off * static_cast<int64_t>(elem_size);

                    switch (result_dtype) {
                    case DType::Float32:
                        cublas_gemm_f32(
                            handle, transpose_a, transpose_b,
                            static_cast<int>(M), static_cast<int>(N),
                            static_cast<int>(K),
                            static_cast<const float *>(a_batch), lda,
                            static_cast<const float *>(b_batch), ldb,
                            static_cast<float *>(c_batch), ldc);
                        break;
                    case DType::Float64:
                        cublas_gemm_f64(
                            handle, transpose_a, transpose_b,
                            static_cast<int>(M), static_cast<int>(N),
                            static_cast<int>(K),
                            static_cast<const double *>(a_batch), lda,
                            static_cast<const double *>(b_batch), ldb,
                            static_cast<double *>(c_batch), ldc);
                        break;
                    case DType::Float16:
                        cublas_gemm_f16(
                            handle, transpose_a, transpose_b,
                            static_cast<int>(M), static_cast<int>(N),
                            static_cast<int>(K),
                            a_batch, lda, b_batch, ldb, c_batch, ldc);
                        break;
                    default:
                        break;
                    }
                }
            }
        }

        CudaExecutionStream::instance().increment_batch();
        return result;
    }
};

#endif // AXIOM_CUDA_SUPPORT

// ============================================================================
// Registration
// ============================================================================

void register_cublas_operations() {
#ifdef AXIOM_CUDA_SUPPORT
    if (!is_cuda_available()) return;

    ops::OperationRegistry::register_operation(
        ops::OpType::MatMul, Device::GPU,
        std::make_unique<CublasMatMulOperation>());
#endif
}

} // namespace cuda
} // namespace backends
} // namespace axiom
