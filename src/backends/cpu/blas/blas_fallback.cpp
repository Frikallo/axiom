#include "blas_fallback.hpp"

#include <algorithm>
#include <cmath>

#ifdef AXIOM_USE_XSIMD
#include <xsimd/xsimd.hpp>
#endif

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

// ============================================================================
// Cache-Blocked GEMM Implementation
// ============================================================================

// Tile sizes optimized for L1/L2 cache
// 64x64 tiles fit well in L1 cache on most modern CPUs
constexpr size_t TILE_M = 64;
constexpr size_t TILE_N = 64;
constexpr size_t TILE_K = 64;

template <typename T>
void FallbackBlasBackend::gemm_impl(bool transA, bool transB, size_t M,
                                    size_t N, size_t K, T alpha, const T *A,
                                    size_t lda, const T *B, size_t ldb, T beta,
                                    T *C, size_t ldc) {
    // Handle beta scaling of C
    if (beta == T(0)) {
        // Zero out C
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] = T(0);
            }
        }
    } else if (beta != T(1)) {
        // Scale C by beta
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Skip computation if alpha is zero
    if (alpha == T(0)) {
        return;
    }

    // Cache-blocked matrix multiplication
    // C += alpha * op(A) * op(B)
    for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
        size_t i_end = std::min(i0 + TILE_M, M);

        for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
            size_t k_end = std::min(k0 + TILE_K, K);

            for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
                size_t j_end = std::min(j0 + TILE_N, N);

                // Micro-kernel for this tile
                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        // Get A[i,k] with transpose handling
                        T a_val;
                        if (transA) {
                            a_val = A[k * lda + i]; // A is KxM, access [k,i]
                        } else {
                            a_val = A[i * lda + k]; // A is MxK, access [i,k]
                        }
                        a_val *= alpha;

                        // Inner loop with potential SIMD
#ifdef AXIOM_USE_XSIMD
                        if constexpr (xsimd::has_simd_register<T>::value) {
                            using batch_type = xsimd::batch<T>;
                            constexpr size_t simd_width = batch_type::size;
                            size_t j = j0;

                            // SIMD loop
                            for (; j + simd_width <= j_end; j += simd_width) {
                                // Get B[k,j:j+simd_width]
                                batch_type b_vec;
                                if (transB) {
                                    // B is NxK, need B[j:j+simd_width, k]
                                    // Non-contiguous access, load element by
                                    // element
                                    alignas(64) T temp[simd_width];
                                    for (size_t s = 0; s < simd_width; ++s) {
                                        temp[s] = B[(j + s) * ldb + k];
                                    }
                                    b_vec = batch_type::load_aligned(temp);
                                } else {
                                    // B is KxN, access [k, j:j+simd_width]
                                    // Contiguous access
                                    b_vec = batch_type::load_unaligned(
                                        &B[k * ldb + j]);
                                }

                                // Load C[i,j:j+simd_width]
                                batch_type c_vec =
                                    batch_type::load_unaligned(&C[i * ldc + j]);

                                // C += a_val * B
                                c_vec = xsimd::fma(batch_type::broadcast(a_val),
                                                   b_vec, c_vec);

                                // Store result
                                c_vec.store_unaligned(&C[i * ldc + j]);
                            }

                            // Scalar remainder
                            for (; j < j_end; ++j) {
                                T b_val;
                                if (transB) {
                                    b_val = B[j * ldb + k];
                                } else {
                                    b_val = B[k * ldb + j];
                                }
                                C[i * ldc + j] += a_val * b_val;
                            }
                        } else
#endif
                        {
                            // Scalar fallback
                            for (size_t j = j0; j < j_end; ++j) {
                                T b_val;
                                if (transB) {
                                    b_val = B[j * ldb + k];
                                } else {
                                    b_val = B[k * ldb + j];
                                }
                                C[i * ldc + j] += a_val * b_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// GEMV Implementation
// ============================================================================

template <typename T>
void FallbackBlasBackend::gemv_impl(bool transA, size_t M, size_t N, T alpha,
                                    const T *A, size_t lda, const T *x,
                                    size_t incx, T beta, T *y, size_t incy) {
    // Dimensions of result
    size_t result_len = transA ? N : M;

    // Handle beta scaling
    if (beta == T(0)) {
        for (size_t i = 0; i < result_len; ++i) {
            y[i * incy] = T(0);
        }
    } else if (beta != T(1)) {
        for (size_t i = 0; i < result_len; ++i) {
            y[i * incy] *= beta;
        }
    }

    if (alpha == T(0)) {
        return;
    }

    if (!transA) {
        // y = alpha * A * x + beta * y
        // A is MxN, x is N, y is M
        for (size_t i = 0; i < M; ++i) {
            T sum = T(0);
#ifdef AXIOM_USE_XSIMD
            if constexpr (xsimd::has_simd_register<T>::value) {
                using batch_type = xsimd::batch<T>;
                constexpr size_t simd_width = batch_type::size;

                if (incx == 1) {
                    // x is contiguous
                    size_t j = 0;
                    batch_type sum_vec = batch_type::broadcast(T(0));

                    for (; j + simd_width <= N; j += simd_width) {
                        batch_type a_vec =
                            batch_type::load_unaligned(&A[i * lda + j]);
                        batch_type x_vec = batch_type::load_unaligned(&x[j]);
                        sum_vec = xsimd::fma(a_vec, x_vec, sum_vec);
                    }

                    sum = xsimd::reduce_add(sum_vec);

                    // Scalar remainder
                    for (; j < N; ++j) {
                        sum += A[i * lda + j] * x[j];
                    }
                } else {
                    // Strided x
                    for (size_t j = 0; j < N; ++j) {
                        sum += A[i * lda + j] * x[j * incx];
                    }
                }
            } else
#endif
            {
                for (size_t j = 0; j < N; ++j) {
                    sum += A[i * lda + j] * x[j * incx];
                }
            }
            y[i * incy] += alpha * sum;
        }
    } else {
        // y = alpha * A^T * x + beta * y
        // A is MxN (stored), A^T is NxM, x is M, y is N
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            for (size_t i = 0; i < M; ++i) {
                sum += A[i * lda + j] * x[i * incx];
            }
            y[j * incy] += alpha * sum;
        }
    }
}

// ============================================================================
// Dot Product Implementation
// ============================================================================

template <typename T>
T FallbackBlasBackend::dot_impl(size_t n, const T *x, size_t incx, const T *y,
                                size_t incy) {
    T result = T(0);

#ifdef AXIOM_USE_XSIMD
    if constexpr (xsimd::has_simd_register<T>::value) {
        using batch_type = xsimd::batch<T>;
        constexpr size_t simd_width = batch_type::size;

        if (incx == 1 && incy == 1) {
            // Both vectors are contiguous - use SIMD
            size_t i = 0;
            batch_type sum_vec = batch_type::broadcast(T(0));

            for (; i + simd_width <= n; i += simd_width) {
                batch_type x_vec = batch_type::load_unaligned(&x[i]);
                batch_type y_vec = batch_type::load_unaligned(&y[i]);
                sum_vec = xsimd::fma(x_vec, y_vec, sum_vec);
            }

            result = xsimd::reduce_add(sum_vec);

            // Handle remainder
            for (; i < n; ++i) {
                result += x[i] * y[i];
            }
        } else {
            // Strided access - use scalar
            for (size_t i = 0; i < n; ++i) {
                result += x[i * incx] * y[i * incy];
            }
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; ++i) {
            result += x[i * incx] * y[i * incy];
        }
    }

    return result;
}

// ============================================================================
// Norm Implementation
// ============================================================================

template <typename T>
T FallbackBlasBackend::nrm2_impl(size_t n, const T *x, size_t incx) {
    T sum_sq = T(0);

#ifdef AXIOM_USE_XSIMD
    if constexpr (xsimd::has_simd_register<T>::value) {
        using batch_type = xsimd::batch<T>;
        constexpr size_t simd_width = batch_type::size;

        if (incx == 1) {
            // Contiguous vector - use SIMD
            size_t i = 0;
            batch_type sum_vec = batch_type::broadcast(T(0));

            for (; i + simd_width <= n; i += simd_width) {
                batch_type x_vec = batch_type::load_unaligned(&x[i]);
                sum_vec = xsimd::fma(x_vec, x_vec, sum_vec);
            }

            sum_sq = xsimd::reduce_add(sum_vec);

            // Handle remainder
            for (; i < n; ++i) {
                sum_sq += x[i] * x[i];
            }
        } else {
            // Strided access
            for (size_t i = 0; i < n; ++i) {
                T val = x[i * incx];
                sum_sq += val * val;
            }
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; ++i) {
            T val = x[i * incx];
            sum_sq += val * val;
        }
    }

    return std::sqrt(sum_sq);
}

// ============================================================================
// Public Interface - Single Precision
// ============================================================================

void FallbackBlasBackend::sgemm(bool transA, bool transB, size_t M, size_t N,
                                size_t K, float alpha, const float *A,
                                size_t lda, const float *B, size_t ldb,
                                float beta, float *C, size_t ldc) {
    gemm_impl<float>(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}

void FallbackBlasBackend::sgemv(bool transA, size_t M, size_t N, float alpha,
                                const float *A, size_t lda, const float *x,
                                size_t incx, float beta, float *y,
                                size_t incy) {
    gemv_impl<float>(transA, M, N, alpha, A, lda, x, incx, beta, y, incy);
}

float FallbackBlasBackend::sdot(size_t n, const float *x, size_t incx,
                                const float *y, size_t incy) {
    return dot_impl<float>(n, x, incx, y, incy);
}

void FallbackBlasBackend::saxpy(size_t n, float alpha, const float *x,
                                size_t incx, float *y, size_t incy) {
    if (alpha == 0.0f)
        return;

#ifdef AXIOM_USE_XSIMD
    if constexpr (xsimd::has_simd_register<float>::value) {
        using batch_type = xsimd::batch<float>;
        constexpr size_t simd_width = batch_type::size;

        if (incx == 1 && incy == 1) {
            size_t i = 0;
            batch_type alpha_vec = batch_type::broadcast(alpha);

            for (; i + simd_width <= n; i += simd_width) {
                batch_type x_vec = batch_type::load_unaligned(&x[i]);
                batch_type y_vec = batch_type::load_unaligned(&y[i]);
                y_vec = xsimd::fma(alpha_vec, x_vec, y_vec);
                y_vec.store_unaligned(&y[i]);
            }

            // Remainder
            for (; i < n; ++i) {
                y[i] += alpha * x[i];
            }
            return;
        }
    }
#endif

    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

float FallbackBlasBackend::snrm2(size_t n, const float *x, size_t incx) {
    return nrm2_impl<float>(n, x, incx);
}

void FallbackBlasBackend::sscal(size_t n, float alpha, float *x, size_t incx) {
    if (alpha == 1.0f)
        return;

#ifdef AXIOM_USE_XSIMD
    if constexpr (xsimd::has_simd_register<float>::value) {
        using batch_type = xsimd::batch<float>;
        constexpr size_t simd_width = batch_type::size;

        if (incx == 1) {
            size_t i = 0;
            batch_type alpha_vec = batch_type::broadcast(alpha);

            for (; i + simd_width <= n; i += simd_width) {
                batch_type x_vec = batch_type::load_unaligned(&x[i]);
                x_vec = x_vec * alpha_vec;
                x_vec.store_unaligned(&x[i]);
            }

            // Remainder
            for (; i < n; ++i) {
                x[i] *= alpha;
            }
            return;
        }
    }
#endif

    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        x[i * incx] *= alpha;
    }
}

// ============================================================================
// Public Interface - Double Precision
// ============================================================================

void FallbackBlasBackend::dgemm(bool transA, bool transB, size_t M, size_t N,
                                size_t K, double alpha, const double *A,
                                size_t lda, const double *B, size_t ldb,
                                double beta, double *C, size_t ldc) {
    gemm_impl<double>(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                      ldc);
}

void FallbackBlasBackend::dgemv(bool transA, size_t M, size_t N, double alpha,
                                const double *A, size_t lda, const double *x,
                                size_t incx, double beta, double *y,
                                size_t incy) {
    gemv_impl<double>(transA, M, N, alpha, A, lda, x, incx, beta, y, incy);
}

double FallbackBlasBackend::ddot(size_t n, const double *x, size_t incx,
                                 const double *y, size_t incy) {
    return dot_impl<double>(n, x, incx, y, incy);
}

void FallbackBlasBackend::daxpy(size_t n, double alpha, const double *x,
                                size_t incx, double *y, size_t incy) {
    if (alpha == 0.0)
        return;

#ifdef AXIOM_USE_XSIMD
    if constexpr (xsimd::has_simd_register<double>::value) {
        using batch_type = xsimd::batch<double>;
        constexpr size_t simd_width = batch_type::size;

        if (incx == 1 && incy == 1) {
            size_t i = 0;
            batch_type alpha_vec = batch_type::broadcast(alpha);

            for (; i + simd_width <= n; i += simd_width) {
                batch_type x_vec = batch_type::load_unaligned(&x[i]);
                batch_type y_vec = batch_type::load_unaligned(&y[i]);
                y_vec = xsimd::fma(alpha_vec, x_vec, y_vec);
                y_vec.store_unaligned(&y[i]);
            }

            // Remainder
            for (; i < n; ++i) {
                y[i] += alpha * x[i];
            }
            return;
        }
    }
#endif

    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

double FallbackBlasBackend::dnrm2(size_t n, const double *x, size_t incx) {
    return nrm2_impl<double>(n, x, incx);
}

void FallbackBlasBackend::dscal(size_t n, double alpha, double *x,
                                size_t incx) {
    if (alpha == 1.0)
        return;

#ifdef AXIOM_USE_XSIMD
    if constexpr (xsimd::has_simd_register<double>::value) {
        using batch_type = xsimd::batch<double>;
        constexpr size_t simd_width = batch_type::size;

        if (incx == 1) {
            size_t i = 0;
            batch_type alpha_vec = batch_type::broadcast(alpha);

            for (; i + simd_width <= n; i += simd_width) {
                batch_type x_vec = batch_type::load_unaligned(&x[i]);
                x_vec = x_vec * alpha_vec;
                x_vec.store_unaligned(&x[i]);
            }

            // Remainder
            for (; i < n; ++i) {
                x[i] *= alpha;
            }
            return;
        }
    }
#endif

    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        x[i * incx] *= alpha;
    }
}

} // namespace blas
} // namespace cpu
} // namespace backends
} // namespace axiom
