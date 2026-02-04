#include "lapack_openblas.hpp"

#ifdef AXIOM_USE_OPENBLAS

// OpenBLAS provides LAPACKE interface
#include <lapacke.h>

namespace axiom {
namespace backends {
namespace cpu {
namespace lapack {

// ============================================================================
// LU Decomposition
// ============================================================================

int OpenBlasLapackBackend::sgetrf(int m, int n, float *a, int lda, int *ipiv) {
    return LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ipiv);
}

int OpenBlasLapackBackend::dgetrf(int m, int n, double *a, int lda, int *ipiv) {
    return LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ipiv);
}

int OpenBlasLapackBackend::cgetrf(int m, int n, complex64_t *a, int lda,
                                  int *ipiv) {
    return LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, n,
                          reinterpret_cast<lapack_complex_float *>(a), lda,
                          ipiv);
}

int OpenBlasLapackBackend::zgetrf(int m, int n, complex128_t *a, int lda,
                                  int *ipiv) {
    return LAPACKE_zgetrf(LAPACK_COL_MAJOR, m, n,
                          reinterpret_cast<lapack_complex_double *>(a), lda,
                          ipiv);
}

// ============================================================================
// Matrix Inverse
// ============================================================================

int OpenBlasLapackBackend::sgetri(int n, float *a, int lda, const int *ipiv,
                                  float *, int) {
    // LAPACKE handles workspace automatically
    return LAPACKE_sgetri(LAPACK_COL_MAJOR, n, a, lda, const_cast<int *>(ipiv));
}

int OpenBlasLapackBackend::dgetri(int n, double *a, int lda, const int *ipiv,
                                  double *, int) {
    return LAPACKE_dgetri(LAPACK_COL_MAJOR, n, a, lda, const_cast<int *>(ipiv));
}

int OpenBlasLapackBackend::cgetri(int n, complex64_t *a, int lda,
                                  const int *ipiv, complex64_t *, int) {
    return LAPACKE_cgetri(LAPACK_COL_MAJOR, n,
                          reinterpret_cast<lapack_complex_float *>(a), lda,
                          const_cast<int *>(ipiv));
}

int OpenBlasLapackBackend::zgetri(int n, complex128_t *a, int lda,
                                  const int *ipiv, complex128_t *, int) {
    return LAPACKE_zgetri(LAPACK_COL_MAJOR, n,
                          reinterpret_cast<lapack_complex_double *>(a), lda,
                          const_cast<int *>(ipiv));
}

// ============================================================================
// Linear System Solve
// ============================================================================

int OpenBlasLapackBackend::sgesv(int n, int nrhs, float *a, int lda, int *ipiv,
                                 float *b, int ldb) {
    return LAPACKE_sgesv(LAPACK_COL_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
}

int OpenBlasLapackBackend::dgesv(int n, int nrhs, double *a, int lda, int *ipiv,
                                 double *b, int ldb) {
    return LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
}

int OpenBlasLapackBackend::cgesv(int n, int nrhs, complex64_t *a, int lda,
                                 int *ipiv, complex64_t *b, int ldb) {
    return LAPACKE_cgesv(LAPACK_COL_MAJOR, n, nrhs,
                         reinterpret_cast<lapack_complex_float *>(a), lda, ipiv,
                         reinterpret_cast<lapack_complex_float *>(b), ldb);
}

int OpenBlasLapackBackend::zgesv(int n, int nrhs, complex128_t *a, int lda,
                                 int *ipiv, complex128_t *b, int ldb) {
    return LAPACKE_zgesv(
        LAPACK_COL_MAJOR, n, nrhs, reinterpret_cast<lapack_complex_double *>(a),
        lda, ipiv, reinterpret_cast<lapack_complex_double *>(b), ldb);
}

// ============================================================================
// SVD - Singular Value Decomposition
// ============================================================================

int OpenBlasLapackBackend::sgesdd(char jobz, int m, int n, float *a, int lda,
                                  float *s, float *u, int ldu, float *vt,
                                  int ldvt, float *work, int lwork, int *) {
    // Handle workspace query: LAPACKE handles workspace automatically,
    // but callers may do a query with lwork=-1 expecting work[0] to be set
    if (lwork == -1 && work != nullptr) {
        // Return a dummy workspace size (LAPACKE doesn't need it)
        *work = 1.0f;
        return 0;
    }
    return LAPACKE_sgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt,
                          ldvt);
}

int OpenBlasLapackBackend::dgesdd(char jobz, int m, int n, double *a, int lda,
                                  double *s, double *u, int ldu, double *vt,
                                  int ldvt, double *work, int lwork, int *) {
    if (lwork == -1 && work != nullptr) {
        *work = 1.0;
        return 0;
    }
    return LAPACKE_dgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u, ldu, vt,
                          ldvt);
}

int OpenBlasLapackBackend::cgesdd(char jobz, int m, int n, complex64_t *a,
                                  int lda, float *s, complex64_t *u, int ldu,
                                  complex64_t *vt, int ldvt, complex64_t *, int,
                                  float *, int *) {
    return LAPACKE_cgesdd(LAPACK_COL_MAJOR, jobz, m, n,
                          reinterpret_cast<lapack_complex_float *>(a), lda, s,
                          reinterpret_cast<lapack_complex_float *>(u), ldu,
                          reinterpret_cast<lapack_complex_float *>(vt), ldvt);
}

int OpenBlasLapackBackend::zgesdd(char jobz, int m, int n, complex128_t *a,
                                  int lda, double *s, complex128_t *u, int ldu,
                                  complex128_t *vt, int ldvt, complex128_t *,
                                  int, double *, int *) {
    return LAPACKE_zgesdd(LAPACK_COL_MAJOR, jobz, m, n,
                          reinterpret_cast<lapack_complex_double *>(a), lda, s,
                          reinterpret_cast<lapack_complex_double *>(u), ldu,
                          reinterpret_cast<lapack_complex_double *>(vt), ldvt);
}

// ============================================================================
// QR Decomposition
// ============================================================================

int OpenBlasLapackBackend::sgeqrf(int m, int n, float *a, int lda, float *tau,
                                  float *, int) {
    return LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}

int OpenBlasLapackBackend::sorgqr(int m, int n, int k, float *a, int lda,
                                  const float *tau, float *, int) {
    return LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, n, k, a, lda,
                          const_cast<float *>(tau));
}

int OpenBlasLapackBackend::dgeqrf(int m, int n, double *a, int lda, double *tau,
                                  double *, int) {
    return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}

int OpenBlasLapackBackend::dorgqr(int m, int n, int k, double *a, int lda,
                                  const double *tau, double *, int) {
    return LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, k, a, lda,
                          const_cast<double *>(tau));
}

int OpenBlasLapackBackend::cgeqrf(int m, int n, complex64_t *a, int lda,
                                  complex64_t *tau, complex64_t *, int) {
    return LAPACKE_cgeqrf(LAPACK_COL_MAJOR, m, n,
                          reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<lapack_complex_float *>(tau));
}

int OpenBlasLapackBackend::cungqr(int m, int n, int k, complex64_t *a, int lda,
                                  const complex64_t *tau, complex64_t *, int) {
    return LAPACKE_cungqr(LAPACK_COL_MAJOR, m, n, k,
                          reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<lapack_complex_float *>(
                              const_cast<complex64_t *>(tau)));
}

int OpenBlasLapackBackend::zgeqrf(int m, int n, complex128_t *a, int lda,
                                  complex128_t *tau, complex128_t *, int) {
    return LAPACKE_zgeqrf(LAPACK_COL_MAJOR, m, n,
                          reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<lapack_complex_double *>(tau));
}

int OpenBlasLapackBackend::zungqr(int m, int n, int k, complex128_t *a, int lda,
                                  const complex128_t *tau, complex128_t *,
                                  int) {
    return LAPACKE_zungqr(LAPACK_COL_MAJOR, m, n, k,
                          reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<lapack_complex_double *>(
                              const_cast<complex128_t *>(tau)));
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

int OpenBlasLapackBackend::spotrf(char uplo, int n, float *a, int lda) {
    return LAPACKE_spotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
}

int OpenBlasLapackBackend::dpotrf(char uplo, int n, double *a, int lda) {
    return LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, n, a, lda);
}

int OpenBlasLapackBackend::cpotrf(char uplo, int n, complex64_t *a, int lda) {
    return LAPACKE_cpotrf(LAPACK_COL_MAJOR, uplo, n,
                          reinterpret_cast<lapack_complex_float *>(a), lda);
}

int OpenBlasLapackBackend::zpotrf(char uplo, int n, complex128_t *a, int lda) {
    return LAPACKE_zpotrf(LAPACK_COL_MAJOR, uplo, n,
                          reinterpret_cast<lapack_complex_double *>(a), lda);
}

// ============================================================================
// General Eigenvalue Decomposition
// ============================================================================

int OpenBlasLapackBackend::sgeev(char jobvl, char jobvr, int n, float *a,
                                 int lda, float *wr, float *wi, float *vl,
                                 int ldvl, float *vr, int ldvr, float *, int) {
    return LAPACKE_sgeev(LAPACK_COL_MAJOR, jobvl, jobvr, n, a, lda, wr, wi, vl,
                         ldvl, vr, ldvr);
}

int OpenBlasLapackBackend::dgeev(char jobvl, char jobvr, int n, double *a,
                                 int lda, double *wr, double *wi, double *vl,
                                 int ldvl, double *vr, int ldvr, double *,
                                 int) {
    return LAPACKE_dgeev(LAPACK_COL_MAJOR, jobvl, jobvr, n, a, lda, wr, wi, vl,
                         ldvl, vr, ldvr);
}

int OpenBlasLapackBackend::cgeev(char jobvl, char jobvr, int n, complex64_t *a,
                                 int lda, complex64_t *w, complex64_t *vl,
                                 int ldvl, complex64_t *vr, int ldvr,
                                 complex64_t *, int, float *) {
    return LAPACKE_cgeev(LAPACK_COL_MAJOR, jobvl, jobvr, n,
                         reinterpret_cast<lapack_complex_float *>(a), lda,
                         reinterpret_cast<lapack_complex_float *>(w),
                         reinterpret_cast<lapack_complex_float *>(vl), ldvl,
                         reinterpret_cast<lapack_complex_float *>(vr), ldvr);
}

int OpenBlasLapackBackend::zgeev(char jobvl, char jobvr, int n, complex128_t *a,
                                 int lda, complex128_t *w, complex128_t *vl,
                                 int ldvl, complex128_t *vr, int ldvr,
                                 complex128_t *, int, double *) {
    return LAPACKE_zgeev(LAPACK_COL_MAJOR, jobvl, jobvr, n,
                         reinterpret_cast<lapack_complex_double *>(a), lda,
                         reinterpret_cast<lapack_complex_double *>(w),
                         reinterpret_cast<lapack_complex_double *>(vl), ldvl,
                         reinterpret_cast<lapack_complex_double *>(vr), ldvr);
}

// ============================================================================
// Symmetric/Hermitian Eigenvalue Decomposition
// ============================================================================

int OpenBlasLapackBackend::ssyev(char jobz, char uplo, int n, float *a, int lda,
                                 float *w, float *, int) {
    return LAPACKE_ssyev(LAPACK_COL_MAJOR, jobz, uplo, n, a, lda, w);
}

int OpenBlasLapackBackend::dsyev(char jobz, char uplo, int n, double *a,
                                 int lda, double *w, double *, int) {
    return LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo, n, a, lda, w);
}

int OpenBlasLapackBackend::cheev(char jobz, char uplo, int n, complex64_t *a,
                                 int lda, float *w, complex64_t *, int,
                                 float *) {
    return LAPACKE_cheev(LAPACK_COL_MAJOR, jobz, uplo, n,
                         reinterpret_cast<lapack_complex_float *>(a), lda, w);
}

int OpenBlasLapackBackend::zheev(char jobz, char uplo, int n, complex128_t *a,
                                 int lda, double *w, complex128_t *, int,
                                 double *) {
    return LAPACKE_zheev(LAPACK_COL_MAJOR, jobz, uplo, n,
                         reinterpret_cast<lapack_complex_double *>(a), lda, w);
}

// ============================================================================
// Least Squares
// ============================================================================

int OpenBlasLapackBackend::sgelsd(int m, int n, int nrhs, float *a, int lda,
                                  float *b, int ldb, float *s, float rcond,
                                  int *rank, float *, int, int *) {
    return LAPACKE_sgelsd(LAPACK_COL_MAJOR, m, n, nrhs, a, lda, b, ldb, s,
                          rcond, rank);
}

int OpenBlasLapackBackend::dgelsd(int m, int n, int nrhs, double *a, int lda,
                                  double *b, int ldb, double *s, double rcond,
                                  int *rank, double *, int, int *) {
    return LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, n, nrhs, a, lda, b, ldb, s,
                          rcond, rank);
}

int OpenBlasLapackBackend::cgelsd(int m, int n, int nrhs, complex64_t *a,
                                  int lda, complex64_t *b, int ldb, float *s,
                                  float rcond, int *rank, complex64_t *, int,
                                  float *, int *) {
    return LAPACKE_cgelsd(LAPACK_COL_MAJOR, m, n, nrhs,
                          reinterpret_cast<lapack_complex_float *>(a), lda,
                          reinterpret_cast<lapack_complex_float *>(b), ldb, s,
                          rcond, rank);
}

int OpenBlasLapackBackend::zgelsd(int m, int n, int nrhs, complex128_t *a,
                                  int lda, complex128_t *b, int ldb, double *s,
                                  double rcond, int *rank, complex128_t *, int,
                                  double *, int *) {
    return LAPACKE_zgelsd(LAPACK_COL_MAJOR, m, n, nrhs,
                          reinterpret_cast<lapack_complex_double *>(a), lda,
                          reinterpret_cast<lapack_complex_double *>(b), ldb, s,
                          rcond, rank);
}

} // namespace lapack
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_OPENBLAS
