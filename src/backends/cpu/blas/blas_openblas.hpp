#pragma once

// OpenBLAS backend implementation
// Uses cblas_* functions from OpenBLAS library

#ifdef AXIOM_USE_OPENBLAS

#include "blas_backend.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

class OpenBlasBackend : public BlasBackend {
  public:
    OpenBlasBackend() = default;
    ~OpenBlasBackend() override = default;

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
    const char *name() const override { return "OpenBLAS"; }
    BlasType type() const override { return BlasType::OpenBLAS; }

    // OpenBLAS-specific diagnostic and control functions
    static const char *get_config();
    static int get_num_threads();
    static void set_num_threads(int num_threads);
    static int get_num_procs();
    static const char *get_corename();
};

} // namespace blas
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_OPENBLAS
