#include "blas_openblas.hpp"

#ifdef AXIOM_USE_OPENBLAS

#include <cblas.h>

// OpenBLAS-specific functions for diagnostics
// These are declared in openblas_config.h but we declare them here
// to avoid header conflicts
extern "C" {
char *openblas_get_config(void);
int openblas_get_num_threads(void);
void openblas_set_num_threads(int num_threads);
int openblas_get_num_procs(void);
char *openblas_get_corename(void);
}

#include <cstdio>

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

// ============================================================================
// OpenBLAS Diagnostics (called once on first use)
// ============================================================================

namespace {
bool g_diagnostics_printed = false;

void print_openblas_diagnostics() {
    if (g_diagnostics_printed)
        return;
    g_diagnostics_printed = true;

#ifndef NDEBUG
    // Only print in debug builds or when AXIOM_BLAS_DEBUG is set
    const char *debug_env = std::getenv("AXIOM_BLAS_DEBUG");
    if (debug_env && debug_env[0] != '0') {
        std::fprintf(stderr, "[AXIOM BLAS] OpenBLAS config: %s\n",
                     openblas_get_config());
        std::fprintf(stderr, "[AXIOM BLAS] OpenBLAS threads: %d\n",
                     openblas_get_num_threads());
        std::fprintf(stderr, "[AXIOM BLAS] OpenBLAS procs: %d\n",
                     openblas_get_num_procs());
        std::fprintf(stderr, "[AXIOM BLAS] OpenBLAS core: %s\n",
                     openblas_get_corename());
    }
#endif
}
} // namespace

// ============================================================================
// BLAS Level 3 - Matrix-Matrix Operations
// ============================================================================

void OpenBlasBackend::sgemm(bool transA, bool transB, size_t M, size_t N,
                            size_t K, float alpha, const float *A, size_t lda,
                            const float *B, size_t ldb, float beta, float *C,
                            size_t ldc) {
    print_openblas_diagnostics();
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
    print_openblas_diagnostics();
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

// ============================================================================
// OpenBLAS-specific diagnostic and control functions
// ============================================================================

const char *OpenBlasBackend::get_config() { return openblas_get_config(); }

int OpenBlasBackend::get_num_threads() { return openblas_get_num_threads(); }

void OpenBlasBackend::set_num_threads(int num_threads) {
    openblas_set_num_threads(num_threads);
}

int OpenBlasBackend::get_num_procs() { return openblas_get_num_procs(); }

const char *OpenBlasBackend::get_corename() { return openblas_get_corename(); }

} // namespace blas
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_OPENBLAS
