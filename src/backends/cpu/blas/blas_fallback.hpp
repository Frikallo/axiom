#pragma once

// Pure C++ fallback BLAS backend implementation
// Uses cache-blocked algorithms with SIMD via xsimd for vectorization

#include "blas_backend.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

class FallbackBlasBackend : public BlasBackend {
  public:
    FallbackBlasBackend() = default;
    ~FallbackBlasBackend() override = default;

    // BLAS Level 3
    void sgemm(bool transA, bool transB, size_t M, size_t N, size_t K,
               float alpha, const float *A, size_t lda, const float *B,
               size_t ldb, float beta, float *C, size_t ldc) override;

    void dgemm(bool transA, bool transB, size_t M, size_t N, size_t K,
               double alpha, const double *A, size_t lda, const double *B,
               size_t ldb, double beta, double *C, size_t ldc) override;

    // BLAS Level 2
    void sgemv(bool transA, size_t M, size_t N, float alpha, const float *A,
               size_t lda, const float *x, size_t incx, float beta, float *y,
               size_t incy) override;

    void dgemv(bool transA, size_t M, size_t N, double alpha, const double *A,
               size_t lda, const double *x, size_t incx, double beta, double *y,
               size_t incy) override;

    // BLAS Level 1
    float sdot(size_t n, const float *x, size_t incx, const float *y,
               size_t incy) override;

    double ddot(size_t n, const double *x, size_t incx, const double *y,
                size_t incy) override;

    void saxpy(size_t n, float alpha, const float *x, size_t incx, float *y,
               size_t incy) override;

    void daxpy(size_t n, double alpha, const double *x, size_t incx, double *y,
               size_t incy) override;

    float snrm2(size_t n, const float *x, size_t incx) override;

    double dnrm2(size_t n, const double *x, size_t incx) override;

    void sscal(size_t n, float alpha, float *x, size_t incx) override;

    void dscal(size_t n, double alpha, double *x, size_t incx) override;

    // Backend info
    const char *name() const override { return "Fallback"; }
    BlasType type() const override { return BlasType::Fallback; }

  private:
    // Cache-blocked GEMM implementation
    template <typename T>
    void gemm_impl(bool transA, bool transB, size_t M, size_t N, size_t K,
                   T alpha, const T *A, size_t lda, const T *B, size_t ldb,
                   T beta, T *C, size_t ldc);

    // GEMV implementation
    template <typename T>
    void gemv_impl(bool transA, size_t M, size_t N, T alpha, const T *A,
                   size_t lda, const T *x, size_t incx, T beta, T *y,
                   size_t incy);

    // Dot product implementation
    template <typename T>
    T dot_impl(size_t n, const T *x, size_t incx, const T *y, size_t incy);

    // Norm implementation
    template <typename T> T nrm2_impl(size_t n, const T *x, size_t incx);
};

} // namespace blas
} // namespace cpu
} // namespace backends
} // namespace axiom
