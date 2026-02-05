#include "blas_native.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#ifdef AXIOM_USE_HIGHWAY
#include "hwy/highway.h"
#endif

#include "axiom/parallel.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

// ============================================================================
// Cache-Blocked GEMM Implementation with Proper Memory Access Pattern
// ============================================================================

// Tile sizes optimized for L1/L2 cache
// Tiles should fit in L1 cache to avoid repeated memory traffic
constexpr size_t TILE_M = 64;
constexpr size_t TILE_N = 64;
constexpr size_t TILE_K = 256; // Larger K tile since we don't reload C each K

template <typename T>
void NativeBlasBackend::gemm_impl(bool transA, bool transB, size_t M, size_t N,
                                  size_t K, T alpha, const T *A, size_t lda,
                                  const T *B, size_t ldb, T beta, T *C,
                                  size_t ldc) {
    // Skip computation if alpha is zero - just scale C by beta
    if (alpha == T(0)) {
        if (beta == T(0)) {
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C[i * ldc + j] = T(0);
                }
            }
        } else if (beta != T(1)) {
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C[i * ldc + j] *= beta;
                }
            }
        }
        return;
    }

    // Lambda to get A element with transpose handling
    auto get_A = [&](size_t i, size_t k) -> T {
        if (transA) {
            return A[k * lda + i]; // A stored as KxM
        } else {
            return A[i * lda + k]; // A stored as MxK
        }
    };

    // Lambda to get B element with transpose handling
    auto get_B = [&](size_t k, size_t j) -> T {
        if (transB) {
            return B[j * ldb + k]; // B stored as NxK
        } else {
            return B[k * ldb + j]; // B stored as KxN
        }
    };

    // Process tiles of C
    // OpenMP parallelization of outer tile loops
#ifdef AXIOM_USE_OPENMP
    bool should_parallel = parallel::should_parallelize_matmul(M, N, K);
    // MSVC OpenMP requires signed loop indices; collapse(2) may be ignored on
    // MSVC
#pragma omp parallel for collapse(2) schedule(dynamic) if (should_parallel)
#endif
    for (ptrdiff_t i0 = 0; i0 < static_cast<ptrdiff_t>(M); i0 += TILE_M) {
        for (ptrdiff_t j0 = 0; j0 < static_cast<ptrdiff_t>(N); j0 += TILE_N) {
            // Local tile buffer for C accumulation - each thread gets its own
            // This is the key optimization: load C once, accumulate, store once
            alignas(64) T c_tile[TILE_M][TILE_N];

            size_t tile_m = std::min(TILE_M, M - static_cast<size_t>(i0));
            size_t tile_n = std::min(TILE_N, N - static_cast<size_t>(j0));

            // Initialize c_tile: load from C and apply beta, or zero
            if (beta == T(0)) {
                for (size_t i = 0; i < tile_m; ++i) {
                    for (size_t j = 0; j < tile_n; ++j) {
                        c_tile[i][j] = T(0);
                    }
                }
            } else if (beta == T(1)) {
                for (size_t i = 0; i < tile_m; ++i) {
                    for (size_t j = 0; j < tile_n; ++j) {
                        c_tile[i][j] = C[(i0 + i) * ldc + (j0 + j)];
                    }
                }
            } else {
                for (size_t i = 0; i < tile_m; ++i) {
                    for (size_t j = 0; j < tile_n; ++j) {
                        c_tile[i][j] = C[(i0 + i) * ldc + (j0 + j)] * beta;
                    }
                }
            }

            // Accumulate A*B into c_tile - NO C memory access in this loop!
            for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                size_t tile_k = std::min(TILE_K, K - k0);

                // Micro-kernel: accumulate into c_tile
                for (size_t i = 0; i < tile_m; ++i) {
                    for (size_t k = 0; k < tile_k; ++k) {
                        T a_val = get_A(i0 + i, k0 + k) * alpha;

#ifdef AXIOM_USE_HIGHWAY
                        {
                            namespace hn = hwy::HWY_NAMESPACE;
                            const hn::ScalableTag<T> d;
                            const size_t simd_width = hn::Lanes(d);
                            const auto a_broadcast = hn::Set(d, a_val);

                            size_t j = 0;

                            if (!transB) {
                                // B is KxN - contiguous row access
                                const T *b_row = &B[(k0 + k) * ldb + j0];

                                // SIMD loop - accumulate in c_tile, not main C
                                for (; j + simd_width <= tile_n;
                                     j += simd_width) {
                                    const auto b_vec = hn::LoadU(d, &b_row[j]);
                                    const auto c_vec =
                                        hn::Load(d, &c_tile[i][j]);
                                    const auto result =
                                        hn::MulAdd(a_broadcast, b_vec, c_vec);
                                    hn::Store(result, d, &c_tile[i][j]);
                                }
                            } else {
                                // B is NxK - non-contiguous, need gather
                                // Process in smaller chunks to use temp buffer
                                for (; j + simd_width <= tile_n;
                                     j += simd_width) {
                                    alignas(HWY_MAX_BYTES)
                                        T temp[HWY_MAX_LANES_D(
                                            hn::ScalableTag<T>)];
                                    for (size_t s = 0; s < simd_width; ++s) {
                                        temp[s] =
                                            B[(j0 + j + s) * ldb + (k0 + k)];
                                    }
                                    const auto b_vec = hn::Load(d, temp);
                                    const auto c_vec =
                                        hn::Load(d, &c_tile[i][j]);
                                    const auto result =
                                        hn::MulAdd(a_broadcast, b_vec, c_vec);
                                    hn::Store(result, d, &c_tile[i][j]);
                                }
                            }

                            // Scalar remainder
                            for (; j < tile_n; ++j) {
                                c_tile[i][j] += a_val * get_B(k0 + k, j0 + j);
                            }
                        }
#else
                        {
                            // Scalar fallback
                            for (size_t j = 0; j < tile_n; ++j) {
                                c_tile[i][j] += a_val * get_B(k0 + k, j0 + j);
                            }
                        }
#endif
                    }
                }
            }

            // Store c_tile back to C - ONE store per element
            for (size_t i = 0; i < tile_m; ++i) {
                for (size_t j = 0; j < tile_n; ++j) {
                    C[(i0 + i) * ldc + (j0 + j)] = c_tile[i][j];
                }
            }
        }
    }
}

// ============================================================================
// GEMV Implementation
// ============================================================================

template <typename T>
void NativeBlasBackend::gemv_impl(bool transA, size_t M, size_t N, T alpha,
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
#ifdef AXIOM_USE_HIGHWAY
            {
                namespace hn = hwy::HWY_NAMESPACE;
                const hn::ScalableTag<T> d;
                const size_t simd_width = hn::Lanes(d);

                if (incx == 1) {
                    // x is contiguous
                    size_t j = 0;
                    auto sum_vec = hn::Zero(d);

                    for (; j + simd_width <= N; j += simd_width) {
                        const auto a_vec = hn::LoadU(d, &A[i * lda + j]);
                        const auto x_vec = hn::LoadU(d, &x[j]);
                        sum_vec = hn::MulAdd(a_vec, x_vec, sum_vec);
                    }

                    sum = hn::ReduceSum(d, sum_vec);

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
            }
#else
            {
                for (size_t j = 0; j < N; ++j) {
                    sum += A[i * lda + j] * x[j * incx];
                }
            }
#endif
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
T NativeBlasBackend::dot_impl(size_t n, const T *x, size_t incx, const T *y,
                              size_t incy) {
    T result = T(0);

#ifdef AXIOM_USE_HIGHWAY
    {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<T> d;
        const size_t simd_width = hn::Lanes(d);

        if (incx == 1 && incy == 1) {
            // Both vectors are contiguous - use SIMD
            size_t i = 0;
            auto sum_vec = hn::Zero(d);

            for (; i + simd_width <= n; i += simd_width) {
                const auto x_vec = hn::LoadU(d, &x[i]);
                const auto y_vec = hn::LoadU(d, &y[i]);
                sum_vec = hn::MulAdd(x_vec, y_vec, sum_vec);
            }

            result = hn::ReduceSum(d, sum_vec);

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
    }
#else
    {
        for (size_t i = 0; i < n; ++i) {
            result += x[i * incx] * y[i * incy];
        }
    }
#endif

    return result;
}

// ============================================================================
// Norm Implementation
// ============================================================================

template <typename T>
T NativeBlasBackend::nrm2_impl(size_t n, const T *x, size_t incx) {
    T sum_sq = T(0);

#ifdef AXIOM_USE_HIGHWAY
    {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<T> d;
        const size_t simd_width = hn::Lanes(d);

        if (incx == 1) {
            // Contiguous vector - use SIMD
            size_t i = 0;
            auto sum_vec = hn::Zero(d);

            for (; i + simd_width <= n; i += simd_width) {
                const auto x_vec = hn::LoadU(d, &x[i]);
                sum_vec = hn::MulAdd(x_vec, x_vec, sum_vec);
            }

            sum_sq = hn::ReduceSum(d, sum_vec);

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
    }
#else
    {
        for (size_t i = 0; i < n; ++i) {
            T val = x[i * incx];
            sum_sq += val * val;
        }
    }
#endif

    return std::sqrt(sum_sq);
}

// ============================================================================
// Public Interface - Single Precision
// ============================================================================

void NativeBlasBackend::sgemm(bool transA, bool transB, size_t M, size_t N,
                              size_t K, float alpha, const float *A, size_t lda,
                              const float *B, size_t ldb, float beta, float *C,
                              size_t ldc) {
    gemm_impl<float>(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}

void NativeBlasBackend::sgemv(bool transA, size_t M, size_t N, float alpha,
                              const float *A, size_t lda, const float *x,
                              size_t incx, float beta, float *y, size_t incy) {
    gemv_impl<float>(transA, M, N, alpha, A, lda, x, incx, beta, y, incy);
}

float NativeBlasBackend::sdot(size_t n, const float *x, size_t incx,
                              const float *y, size_t incy) {
    return dot_impl<float>(n, x, incx, y, incy);
}

void NativeBlasBackend::saxpy(size_t n, float alpha, const float *x,
                              size_t incx, float *y, size_t incy) {
    if (alpha == 0.0f)
        return;

#ifdef AXIOM_USE_HIGHWAY
    {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<float> d;
        const size_t simd_width = hn::Lanes(d);

        if (incx == 1 && incy == 1) {
            size_t i = 0;
            const auto alpha_vec = hn::Set(d, alpha);

            for (; i + simd_width <= n; i += simd_width) {
                const auto x_vec = hn::LoadU(d, &x[i]);
                const auto y_vec = hn::LoadU(d, &y[i]);
                const auto result = hn::MulAdd(alpha_vec, x_vec, y_vec);
                hn::StoreU(result, d, &y[i]);
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

float NativeBlasBackend::snrm2(size_t n, const float *x, size_t incx) {
    return nrm2_impl<float>(n, x, incx);
}

void NativeBlasBackend::sscal(size_t n, float alpha, float *x, size_t incx) {
    if (alpha == 1.0f)
        return;

#ifdef AXIOM_USE_HIGHWAY
    {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<float> d;
        const size_t simd_width = hn::Lanes(d);

        if (incx == 1) {
            size_t i = 0;
            const auto alpha_vec = hn::Set(d, alpha);

            for (; i + simd_width <= n; i += simd_width) {
                const auto x_vec = hn::LoadU(d, &x[i]);
                const auto result = hn::Mul(x_vec, alpha_vec);
                hn::StoreU(result, d, &x[i]);
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

void NativeBlasBackend::dgemm(bool transA, bool transB, size_t M, size_t N,
                              size_t K, double alpha, const double *A,
                              size_t lda, const double *B, size_t ldb,
                              double beta, double *C, size_t ldc) {
    gemm_impl<double>(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C,
                      ldc);
}

void NativeBlasBackend::dgemv(bool transA, size_t M, size_t N, double alpha,
                              const double *A, size_t lda, const double *x,
                              size_t incx, double beta, double *y,
                              size_t incy) {
    gemv_impl<double>(transA, M, N, alpha, A, lda, x, incx, beta, y, incy);
}

double NativeBlasBackend::ddot(size_t n, const double *x, size_t incx,
                               const double *y, size_t incy) {
    return dot_impl<double>(n, x, incx, y, incy);
}

void NativeBlasBackend::daxpy(size_t n, double alpha, const double *x,
                              size_t incx, double *y, size_t incy) {
    if (alpha == 0.0)
        return;

#ifdef AXIOM_USE_HIGHWAY
    {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<double> d;
        const size_t simd_width = hn::Lanes(d);

        if (incx == 1 && incy == 1) {
            size_t i = 0;
            const auto alpha_vec = hn::Set(d, alpha);

            for (; i + simd_width <= n; i += simd_width) {
                const auto x_vec = hn::LoadU(d, &x[i]);
                const auto y_vec = hn::LoadU(d, &y[i]);
                const auto result = hn::MulAdd(alpha_vec, x_vec, y_vec);
                hn::StoreU(result, d, &y[i]);
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

double NativeBlasBackend::dnrm2(size_t n, const double *x, size_t incx) {
    return nrm2_impl<double>(n, x, incx);
}

void NativeBlasBackend::dscal(size_t n, double alpha, double *x, size_t incx) {
    if (alpha == 1.0)
        return;

#ifdef AXIOM_USE_HIGHWAY
    {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<double> d;
        const size_t simd_width = hn::Lanes(d);

        if (incx == 1) {
            size_t i = 0;
            const auto alpha_vec = hn::Set(d, alpha);

            for (; i + simd_width <= n; i += simd_width) {
                const auto x_vec = hn::LoadU(d, &x[i]);
                const auto result = hn::Mul(x_vec, alpha_vec);
                hn::StoreU(result, d, &x[i]);
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
