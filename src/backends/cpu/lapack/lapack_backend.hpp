#pragma once

// Abstract LAPACK backend interface for linear algebra operations
// Supports: Apple Accelerate, OpenBLAS, and a native fallback (errors only)

#include <complex>
#include <cstddef>
#include <memory>
#include <string>

namespace axiom {
namespace backends {
namespace cpu {
namespace lapack {

// LAPACK backend type enumeration
enum class LapackType {
    Auto,       // Auto-detect best available backend
    Accelerate, // Apple Accelerate framework (macOS only)
    OpenBLAS,   // OpenBLAS library (Linux/Windows)
    Native      // Fallback - throws errors for all operations
};

// Complex number types matching LAPACK conventions
using complex64_t = std::complex<float>;
using complex128_t = std::complex<double>;

// Abstract LAPACK backend interface
// Note: All functions return LAPACK info code (0 = success, < 0 = bad arg,
// > 0 = operation-specific)
class LapackBackend {
  public:
    virtual ~LapackBackend() = default;

    // ========================================================================
    // Backend Information
    // ========================================================================

    virtual const char *name() const = 0;
    virtual LapackType type() const = 0;
    virtual bool has_lapack() const = 0;

    // ========================================================================
    // LU Decomposition (getrf) - used for det, inv, solve, lu
    // Computes P * L * U = A
    // ========================================================================

    // Float
    virtual int sgetrf(int m, int n, float *a, int lda, int *ipiv) = 0;
    // Double
    virtual int dgetrf(int m, int n, double *a, int lda, int *ipiv) = 0;
    // Complex64
    virtual int cgetrf(int m, int n, complex64_t *a, int lda, int *ipiv) = 0;
    // Complex128
    virtual int zgetrf(int m, int n, complex128_t *a, int lda, int *ipiv) = 0;

    // ========================================================================
    // Matrix Inverse using LU factorization (getri)
    // Requires prior getrf call
    // ========================================================================

    // Float
    virtual int sgetri(int n, float *a, int lda, const int *ipiv, float *work,
                       int lwork) = 0;
    // Double
    virtual int dgetri(int n, double *a, int lda, const int *ipiv, double *work,
                       int lwork) = 0;
    // Complex64
    virtual int cgetri(int n, complex64_t *a, int lda, const int *ipiv,
                       complex64_t *work, int lwork) = 0;
    // Complex128
    virtual int zgetri(int n, complex128_t *a, int lda, const int *ipiv,
                       complex128_t *work, int lwork) = 0;

    // ========================================================================
    // Linear System Solve (gesv) - solves A * X = B
    // ========================================================================

    // Float
    virtual int sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b,
                      int ldb) = 0;
    // Double
    virtual int dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b,
                      int ldb) = 0;
    // Complex64
    virtual int cgesv(int n, int nrhs, complex64_t *a, int lda, int *ipiv,
                      complex64_t *b, int ldb) = 0;
    // Complex128
    virtual int zgesv(int n, int nrhs, complex128_t *a, int lda, int *ipiv,
                      complex128_t *b, int ldb) = 0;

    // ========================================================================
    // SVD - Singular Value Decomposition (gesdd)
    // Computes A = U * S * Vh using divide-and-conquer
    // ========================================================================

    // Float
    virtual int sgesdd(char jobz, int m, int n, float *a, int lda, float *s,
                       float *u, int ldu, float *vt, int ldvt, float *work,
                       int lwork, int *iwork) = 0;
    // Double
    virtual int dgesdd(char jobz, int m, int n, double *a, int lda, double *s,
                       double *u, int ldu, double *vt, int ldvt, double *work,
                       int lwork, int *iwork) = 0;
    // Complex64 (rwork needed for complex)
    virtual int cgesdd(char jobz, int m, int n, complex64_t *a, int lda,
                       float *s, complex64_t *u, int ldu, complex64_t *vt,
                       int ldvt, complex64_t *work, int lwork, float *rwork,
                       int *iwork) = 0;
    // Complex128
    virtual int zgesdd(char jobz, int m, int n, complex128_t *a, int lda,
                       double *s, complex128_t *u, int ldu, complex128_t *vt,
                       int ldvt, complex128_t *work, int lwork, double *rwork,
                       int *iwork) = 0;

    // ========================================================================
    // QR Decomposition (geqrf + orgqr/ungqr)
    // ========================================================================

    // Float - compute QR factorization
    virtual int sgeqrf(int m, int n, float *a, int lda, float *tau, float *work,
                       int lwork) = 0;
    // Float - generate Q from geqrf result
    virtual int sorgqr(int m, int n, int k, float *a, int lda, const float *tau,
                       float *work, int lwork) = 0;

    // Double
    virtual int dgeqrf(int m, int n, double *a, int lda, double *tau,
                       double *work, int lwork) = 0;
    virtual int dorgqr(int m, int n, int k, double *a, int lda,
                       const double *tau, double *work, int lwork) = 0;

    // Complex64
    virtual int cgeqrf(int m, int n, complex64_t *a, int lda, complex64_t *tau,
                       complex64_t *work, int lwork) = 0;
    virtual int cungqr(int m, int n, int k, complex64_t *a, int lda,
                       const complex64_t *tau, complex64_t *work,
                       int lwork) = 0;

    // Complex128
    virtual int zgeqrf(int m, int n, complex128_t *a, int lda,
                       complex128_t *tau, complex128_t *work, int lwork) = 0;
    virtual int zungqr(int m, int n, int k, complex128_t *a, int lda,
                       const complex128_t *tau, complex128_t *work,
                       int lwork) = 0;

    // ========================================================================
    // Cholesky Decomposition (potrf)
    // Computes L such that A = L * L^H (for positive definite matrices)
    // ========================================================================

    // Float
    virtual int spotrf(char uplo, int n, float *a, int lda) = 0;
    // Double
    virtual int dpotrf(char uplo, int n, double *a, int lda) = 0;
    // Complex64
    virtual int cpotrf(char uplo, int n, complex64_t *a, int lda) = 0;
    // Complex128
    virtual int zpotrf(char uplo, int n, complex128_t *a, int lda) = 0;

    // ========================================================================
    // Eigenvalue Decomposition - General (geev)
    // Computes eigenvalues and optionally eigenvectors of general matrix
    // ========================================================================

    // Float - eigenvalues are complex (wr + i*wi)
    virtual int sgeev(char jobvl, char jobvr, int n, float *a, int lda,
                      float *wr, float *wi, float *vl, int ldvl, float *vr,
                      int ldvr, float *work, int lwork) = 0;
    // Double
    virtual int dgeev(char jobvl, char jobvr, int n, double *a, int lda,
                      double *wr, double *wi, double *vl, int ldvl, double *vr,
                      int ldvr, double *work, int lwork) = 0;
    // Complex64
    virtual int cgeev(char jobvl, char jobvr, int n, complex64_t *a, int lda,
                      complex64_t *w, complex64_t *vl, int ldvl,
                      complex64_t *vr, int ldvr, complex64_t *work, int lwork,
                      float *rwork) = 0;
    // Complex128
    virtual int zgeev(char jobvl, char jobvr, int n, complex128_t *a, int lda,
                      complex128_t *w, complex128_t *vl, int ldvl,
                      complex128_t *vr, int ldvr, complex128_t *work, int lwork,
                      double *rwork) = 0;

    // ========================================================================
    // Eigenvalue Decomposition - Symmetric/Hermitian (syev/heev)
    // Computes real eigenvalues and eigenvectors
    // ========================================================================

    // Float symmetric
    virtual int ssyev(char jobz, char uplo, int n, float *a, int lda, float *w,
                      float *work, int lwork) = 0;
    // Double symmetric
    virtual int dsyev(char jobz, char uplo, int n, double *a, int lda,
                      double *w, double *work, int lwork) = 0;
    // Complex64 Hermitian
    virtual int cheev(char jobz, char uplo, int n, complex64_t *a, int lda,
                      float *w, complex64_t *work, int lwork, float *rwork) = 0;
    // Complex128 Hermitian
    virtual int zheev(char jobz, char uplo, int n, complex128_t *a, int lda,
                      double *w, complex128_t *work, int lwork,
                      double *rwork) = 0;

    // ========================================================================
    // Least Squares (gelsd) - SVD-based
    // Solves min ||A*X - B||_2 using SVD
    // ========================================================================

    // Float
    virtual int sgelsd(int m, int n, int nrhs, float *a, int lda, float *b,
                       int ldb, float *s, float rcond, int *rank, float *work,
                       int lwork, int *iwork) = 0;
    // Double
    virtual int dgelsd(int m, int n, int nrhs, double *a, int lda, double *b,
                       int ldb, double *s, double rcond, int *rank,
                       double *work, int lwork, int *iwork) = 0;
    // Complex64
    virtual int cgelsd(int m, int n, int nrhs, complex64_t *a, int lda,
                       complex64_t *b, int ldb, float *s, float rcond,
                       int *rank, complex64_t *work, int lwork, float *rwork,
                       int *iwork) = 0;
    // Complex128
    virtual int zgelsd(int m, int n, int nrhs, complex128_t *a, int lda,
                       complex128_t *b, int ldb, double *s, double rcond,
                       int *rank, complex128_t *work, int lwork, double *rwork,
                       int *iwork) = 0;
};

// ============================================================================
// Backend Factory Functions
// ============================================================================

// Get the current LAPACK backend (singleton)
// Thread-safe: backend is initialized on first call
LapackBackend &get_lapack_backend();

// Set the LAPACK backend type
// Must be called before any LAPACK operations (ideally at program start)
// If not called, auto-detection is used
void set_lapack_backend(LapackType type);

// Get the currently selected backend type
LapackType get_lapack_backend_type();

// Check if a specific backend type is available on this platform
bool is_lapack_backend_available(LapackType type);

// Get the default backend type for this platform (used in auto-detection)
LapackType get_default_lapack_backend_type();

// Check if LAPACK is available (any backend)
bool has_lapack();

} // namespace lapack
} // namespace cpu
} // namespace backends
} // namespace axiom
