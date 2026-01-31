#include "blas_openblas.hpp"

#ifdef AXIOM_USE_OPENBLAS

#include <cblas.h>

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

// ============================================================================
// BLAS Level 3 - Matrix-Matrix Operations
// ============================================================================

void OpenBlasBackend::sgemm(bool transA, bool transB, size_t M, size_t N,
                            size_t K, float alpha, const float *A, size_t lda,
                            const float *B, size_t ldb, float beta, float *C,
                            size_t ldc) {
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), alpha, A,
                static_cast<int>(lda), B, static_cast<int>(ldb), beta, C,
                static_cast<int>(ldc));
}

void OpenBlasBackend::dgemm(bool transA, bool transB, size_t M, size_t N,
                            size_t K, double alpha, const double *A, size_t lda,
                            const double *B, size_t ldb, double beta, double *C,
                            size_t ldc) {
    cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), alpha, A,
                static_cast<int>(lda), B, static_cast<int>(ldb), beta, C,
                static_cast<int>(ldc));
}

// ============================================================================
// BLAS Level 2 - Matrix-Vector Operations
// ============================================================================

void OpenBlasBackend::sgemv(bool transA, size_t M, size_t N, float alpha,
                            const float *A, size_t lda, const float *x,
                            size_t incx, float beta, float *y, size_t incy) {
    cblas_sgemv(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), alpha, A,
                static_cast<int>(lda), x, static_cast<int>(incx), beta, y,
                static_cast<int>(incy));
}

void OpenBlasBackend::dgemv(bool transA, size_t M, size_t N, double alpha,
                            const double *A, size_t lda, const double *x,
                            size_t incx, double beta, double *y, size_t incy) {
    cblas_dgemv(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), alpha, A,
                static_cast<int>(lda), x, static_cast<int>(incx), beta, y,
                static_cast<int>(incy));
}

// ============================================================================
// BLAS Level 1 - Vector Operations
// ============================================================================

float OpenBlasBackend::sdot(size_t n, const float *x, size_t incx,
                            const float *y, size_t incy) {
    return cblas_sdot(static_cast<int>(n), x, static_cast<int>(incx), y,
                      static_cast<int>(incy));
}

double OpenBlasBackend::ddot(size_t n, const double *x, size_t incx,
                             const double *y, size_t incy) {
    return cblas_ddot(static_cast<int>(n), x, static_cast<int>(incx), y,
                      static_cast<int>(incy));
}

void OpenBlasBackend::saxpy(size_t n, float alpha, const float *x, size_t incx,
                            float *y, size_t incy) {
    cblas_saxpy(static_cast<int>(n), alpha, x, static_cast<int>(incx), y,
                static_cast<int>(incy));
}

void OpenBlasBackend::daxpy(size_t n, double alpha, const double *x,
                            size_t incx, double *y, size_t incy) {
    cblas_daxpy(static_cast<int>(n), alpha, x, static_cast<int>(incx), y,
                static_cast<int>(incy));
}

float OpenBlasBackend::snrm2(size_t n, const float *x, size_t incx) {
    return cblas_snrm2(static_cast<int>(n), x, static_cast<int>(incx));
}

double OpenBlasBackend::dnrm2(size_t n, const double *x, size_t incx) {
    return cblas_dnrm2(static_cast<int>(n), x, static_cast<int>(incx));
}

void OpenBlasBackend::sscal(size_t n, float alpha, float *x, size_t incx) {
    cblas_sscal(static_cast<int>(n), alpha, x, static_cast<int>(incx));
}

void OpenBlasBackend::dscal(size_t n, double alpha, double *x, size_t incx) {
    cblas_dscal(static_cast<int>(n), alpha, x, static_cast<int>(incx));
}

} // namespace blas
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_OPENBLAS
