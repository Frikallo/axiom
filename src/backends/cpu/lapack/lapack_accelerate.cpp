#include "lapack_accelerate.hpp"

#ifdef AXIOM_USE_ACCELERATE

#include <Accelerate/Accelerate.h>

namespace axiom {
namespace backends {
namespace cpu {
namespace lapack {

// ============================================================================
// LU Decomposition
// ============================================================================

int AccelerateLapackBackend::sgetrf(int m, int n, float *a, int lda,
                                    int *ipiv) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda;
    sgetrf_(&m_, &n_, a, &lda_, ipiv, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dgetrf(int m, int n, double *a, int lda,
                                    int *ipiv) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda;
    dgetrf_(&m_, &n_, a, &lda_, ipiv, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cgetrf(int m, int n, complex64_t *a, int lda,
                                    int *ipiv) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda;
    cgetrf_(&m_, &n_, reinterpret_cast<__LAPACK_float_complex *>(a), &lda_,
            ipiv, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zgetrf(int m, int n, complex128_t *a, int lda,
                                    int *ipiv) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda;
    zgetrf_(&m_, &n_, reinterpret_cast<__LAPACK_double_complex *>(a), &lda_,
            ipiv, &info);
    return static_cast<int>(info);
}

// ============================================================================
// Matrix Inverse
// ============================================================================

int AccelerateLapackBackend::sgetri(int n, float *a, int lda, const int *ipiv,
                                    float *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    sgetri_(&n_, a, &lda_, const_cast<int *>(ipiv), work, &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dgetri(int n, double *a, int lda, const int *ipiv,
                                    double *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    dgetri_(&n_, a, &lda_, const_cast<int *>(ipiv), work, &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cgetri(int n, complex64_t *a, int lda,
                                    const int *ipiv, complex64_t *work,
                                    int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    cgetri_(&n_, reinterpret_cast<__LAPACK_float_complex *>(a), &lda_,
            const_cast<int *>(ipiv),
            reinterpret_cast<__LAPACK_float_complex *>(work), &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zgetri(int n, complex128_t *a, int lda,
                                    const int *ipiv, complex128_t *work,
                                    int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    zgetri_(&n_, reinterpret_cast<__LAPACK_double_complex *>(a), &lda_,
            const_cast<int *>(ipiv),
            reinterpret_cast<__LAPACK_double_complex *>(work), &lwork_, &info);
    return static_cast<int>(info);
}

// ============================================================================
// Linear System Solve
// ============================================================================

int AccelerateLapackBackend::sgesv(int n, int nrhs, float *a, int lda,
                                   int *ipiv, float *b, int ldb) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
    sgesv_(&n_, &nrhs_, a, &lda_, ipiv, b, &ldb_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dgesv(int n, int nrhs, double *a, int lda,
                                   int *ipiv, double *b, int ldb) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
    dgesv_(&n_, &nrhs_, a, &lda_, ipiv, b, &ldb_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cgesv(int n, int nrhs, complex64_t *a, int lda,
                                   int *ipiv, complex64_t *b, int ldb) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
    cgesv_(&n_, &nrhs_, reinterpret_cast<__LAPACK_float_complex *>(a), &lda_,
           ipiv, reinterpret_cast<__LAPACK_float_complex *>(b), &ldb_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zgesv(int n, int nrhs, complex128_t *a, int lda,
                                   int *ipiv, complex128_t *b, int ldb) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;
    zgesv_(&n_, &nrhs_, reinterpret_cast<__LAPACK_double_complex *>(a), &lda_,
           ipiv, reinterpret_cast<__LAPACK_double_complex *>(b), &ldb_, &info);
    return static_cast<int>(info);
}

// ============================================================================
// SVD - Singular Value Decomposition
// ============================================================================

int AccelerateLapackBackend::sgesdd(char jobz, int m, int n, float *a, int lda,
                                    float *s, float *u, int ldu, float *vt,
                                    int ldvt, float *work, int lwork,
                                    int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, ldu_ = ldu, ldvt_ = ldvt,
                 lwork_ = lwork;
    sgesdd_(&jobz, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_, work, &lwork_,
            iwork, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dgesdd(char jobz, int m, int n, double *a, int lda,
                                    double *s, double *u, int ldu, double *vt,
                                    int ldvt, double *work, int lwork,
                                    int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, ldu_ = ldu, ldvt_ = ldvt,
                 lwork_ = lwork;
    dgesdd_(&jobz, &m_, &n_, a, &lda_, s, u, &ldu_, vt, &ldvt_, work, &lwork_,
            iwork, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cgesdd(char jobz, int m, int n, complex64_t *a,
                                    int lda, float *s, complex64_t *u, int ldu,
                                    complex64_t *vt, int ldvt,
                                    complex64_t *work, int lwork, float *rwork,
                                    int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, ldu_ = ldu, ldvt_ = ldvt,
                 lwork_ = lwork;
    cgesdd_(&jobz, &m_, &n_, reinterpret_cast<__LAPACK_float_complex *>(a),
            &lda_, s, reinterpret_cast<__LAPACK_float_complex *>(u), &ldu_,
            reinterpret_cast<__LAPACK_float_complex *>(vt), &ldvt_,
            reinterpret_cast<__LAPACK_float_complex *>(work), &lwork_, rwork,
            iwork, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zgesdd(char jobz, int m, int n, complex128_t *a,
                                    int lda, double *s, complex128_t *u,
                                    int ldu, complex128_t *vt, int ldvt,
                                    complex128_t *work, int lwork,
                                    double *rwork, int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, ldu_ = ldu, ldvt_ = ldvt,
                 lwork_ = lwork;
    zgesdd_(&jobz, &m_, &n_, reinterpret_cast<__LAPACK_double_complex *>(a),
            &lda_, s, reinterpret_cast<__LAPACK_double_complex *>(u), &ldu_,
            reinterpret_cast<__LAPACK_double_complex *>(vt), &ldvt_,
            reinterpret_cast<__LAPACK_double_complex *>(work), &lwork_, rwork,
            iwork, &info);
    return static_cast<int>(info);
}

// ============================================================================
// QR Decomposition
// ============================================================================

int AccelerateLapackBackend::sgeqrf(int m, int n, float *a, int lda, float *tau,
                                    float *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
    sgeqrf_(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::sorgqr(int m, int n, int k, float *a, int lda,
                                    const float *tau, float *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
    sorgqr_(&m_, &n_, &k_, a, &lda_, const_cast<float *>(tau), work, &lwork_,
            &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dgeqrf(int m, int n, double *a, int lda,
                                    double *tau, double *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
    dgeqrf_(&m_, &n_, a, &lda_, tau, work, &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dorgqr(int m, int n, int k, double *a, int lda,
                                    const double *tau, double *work,
                                    int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
    dorgqr_(&m_, &n_, &k_, a, &lda_, const_cast<double *>(tau), work, &lwork_,
            &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cgeqrf(int m, int n, complex64_t *a, int lda,
                                    complex64_t *tau, complex64_t *work,
                                    int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
    cgeqrf_(&m_, &n_, reinterpret_cast<__LAPACK_float_complex *>(a), &lda_,
            reinterpret_cast<__LAPACK_float_complex *>(tau),
            reinterpret_cast<__LAPACK_float_complex *>(work), &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cungqr(int m, int n, int k, complex64_t *a,
                                    int lda, const complex64_t *tau,
                                    complex64_t *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
    cungqr_(&m_, &n_, &k_, reinterpret_cast<__LAPACK_float_complex *>(a), &lda_,
            reinterpret_cast<__LAPACK_float_complex *>(
                const_cast<complex64_t *>(tau)),
            reinterpret_cast<__LAPACK_float_complex *>(work), &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zgeqrf(int m, int n, complex128_t *a, int lda,
                                    complex128_t *tau, complex128_t *work,
                                    int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, lda_ = lda, lwork_ = lwork;
    zgeqrf_(&m_, &n_, reinterpret_cast<__LAPACK_double_complex *>(a), &lda_,
            reinterpret_cast<__LAPACK_double_complex *>(tau),
            reinterpret_cast<__LAPACK_double_complex *>(work), &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zungqr(int m, int n, int k, complex128_t *a,
                                    int lda, const complex128_t *tau,
                                    complex128_t *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, k_ = k, lda_ = lda, lwork_ = lwork;
    zungqr_(&m_, &n_, &k_, reinterpret_cast<__LAPACK_double_complex *>(a),
            &lda_,
            reinterpret_cast<__LAPACK_double_complex *>(
                const_cast<complex128_t *>(tau)),
            reinterpret_cast<__LAPACK_double_complex *>(work), &lwork_, &info);
    return static_cast<int>(info);
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

int AccelerateLapackBackend::spotrf(char uplo, int n, float *a, int lda) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda;
    spotrf_(&uplo, &n_, a, &lda_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dpotrf(char uplo, int n, double *a, int lda) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda;
    dpotrf_(&uplo, &n_, a, &lda_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cpotrf(char uplo, int n, complex64_t *a, int lda) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda;
    cpotrf_(&uplo, &n_, reinterpret_cast<__LAPACK_float_complex *>(a), &lda_,
            &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zpotrf(char uplo, int n, complex128_t *a,
                                    int lda) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda;
    zpotrf_(&uplo, &n_, reinterpret_cast<__LAPACK_double_complex *>(a), &lda_,
            &info);
    return static_cast<int>(info);
}

// ============================================================================
// General Eigenvalue Decomposition
// ============================================================================

int AccelerateLapackBackend::sgeev(char jobvl, char jobvr, int n, float *a,
                                   int lda, float *wr, float *wi, float *vl,
                                   int ldvl, float *vr, int ldvr, float *work,
                                   int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
    sgeev_(&jobvl, &jobvr, &n_, a, &lda_, wr, wi, vl, &ldvl_, vr, &ldvr_, work,
           &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dgeev(char jobvl, char jobvr, int n, double *a,
                                   int lda, double *wr, double *wi, double *vl,
                                   int ldvl, double *vr, int ldvr, double *work,
                                   int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
    dgeev_(&jobvl, &jobvr, &n_, a, &lda_, wr, wi, vl, &ldvl_, vr, &ldvr_, work,
           &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cgeev(char jobvl, char jobvr, int n,
                                   complex64_t *a, int lda, complex64_t *w,
                                   complex64_t *vl, int ldvl, complex64_t *vr,
                                   int ldvr, complex64_t *work, int lwork,
                                   float *rwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
    cgeev_(&jobvl, &jobvr, &n_, reinterpret_cast<__LAPACK_float_complex *>(a),
           &lda_, reinterpret_cast<__LAPACK_float_complex *>(w),
           reinterpret_cast<__LAPACK_float_complex *>(vl), &ldvl_,
           reinterpret_cast<__LAPACK_float_complex *>(vr), &ldvr_,
           reinterpret_cast<__LAPACK_float_complex *>(work), &lwork_, rwork,
           &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zgeev(char jobvl, char jobvr, int n,
                                   complex128_t *a, int lda, complex128_t *w,
                                   complex128_t *vl, int ldvl, complex128_t *vr,
                                   int ldvr, complex128_t *work, int lwork,
                                   double *rwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
    zgeev_(&jobvl, &jobvr, &n_, reinterpret_cast<__LAPACK_double_complex *>(a),
           &lda_, reinterpret_cast<__LAPACK_double_complex *>(w),
           reinterpret_cast<__LAPACK_double_complex *>(vl), &ldvl_,
           reinterpret_cast<__LAPACK_double_complex *>(vr), &ldvr_,
           reinterpret_cast<__LAPACK_double_complex *>(work), &lwork_, rwork,
           &info);
    return static_cast<int>(info);
}

// ============================================================================
// Symmetric/Hermitian Eigenvalue Decomposition
// ============================================================================

int AccelerateLapackBackend::ssyev(char jobz, char uplo, int n, float *a,
                                   int lda, float *w, float *work, int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    ssyev_(&jobz, &uplo, &n_, a, &lda_, w, work, &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dsyev(char jobz, char uplo, int n, double *a,
                                   int lda, double *w, double *work,
                                   int lwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    dsyev_(&jobz, &uplo, &n_, a, &lda_, w, work, &lwork_, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cheev(char jobz, char uplo, int n, complex64_t *a,
                                   int lda, float *w, complex64_t *work,
                                   int lwork, float *rwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    cheev_(&jobz, &uplo, &n_, reinterpret_cast<__LAPACK_float_complex *>(a),
           &lda_, w, reinterpret_cast<__LAPACK_float_complex *>(work), &lwork_,
           rwork, &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zheev(char jobz, char uplo, int n, complex128_t *a,
                                   int lda, double *w, complex128_t *work,
                                   int lwork, double *rwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork;
    zheev_(&jobz, &uplo, &n_, reinterpret_cast<__LAPACK_double_complex *>(a),
           &lda_, w, reinterpret_cast<__LAPACK_double_complex *>(work), &lwork_,
           rwork, &info);
    return static_cast<int>(info);
}

// ============================================================================
// Symmetric Eigenvalue Decomposition (Divide-and-Conquer)
// ============================================================================

int AccelerateLapackBackend::ssyevd(char jobz, char uplo, int n, float *a,
                                    int lda, float *w, float *work, int lwork,
                                    int *iwork, int liwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork, liwork_ = liwork;
    ssyevd_(&jobz, &uplo, &n_, a, &lda_, w, work, &lwork_, iwork, &liwork_,
            &info);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dsyevd(char jobz, char uplo, int n, double *a,
                                    int lda, double *w, double *work, int lwork,
                                    int *iwork, int liwork) {
    __LAPACK_int info = 0;
    __LAPACK_int n_ = n, lda_ = lda, lwork_ = lwork, liwork_ = liwork;
    dsyevd_(&jobz, &uplo, &n_, a, &lda_, w, work, &lwork_, iwork, &liwork_,
            &info);
    return static_cast<int>(info);
}

// ============================================================================
// Least Squares
// ============================================================================

int AccelerateLapackBackend::sgelsd(int m, int n, int nrhs, float *a, int lda,
                                    float *b, int ldb, float *s, float rcond,
                                    int *rank, float *work, int lwork,
                                    int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb,
                 lwork_ = lwork;
    __LAPACK_int rank_ = 0;
    sgelsd_(&m_, &n_, &nrhs_, a, &lda_, b, &ldb_, s, &rcond, &rank_, work,
            &lwork_, iwork, &info);
    *rank = static_cast<int>(rank_);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::dgelsd(int m, int n, int nrhs, double *a, int lda,
                                    double *b, int ldb, double *s, double rcond,
                                    int *rank, double *work, int lwork,
                                    int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb,
                 lwork_ = lwork;
    __LAPACK_int rank_ = 0;
    dgelsd_(&m_, &n_, &nrhs_, a, &lda_, b, &ldb_, s, &rcond, &rank_, work,
            &lwork_, iwork, &info);
    *rank = static_cast<int>(rank_);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::cgelsd(int m, int n, int nrhs, complex64_t *a,
                                    int lda, complex64_t *b, int ldb, float *s,
                                    float rcond, int *rank, complex64_t *work,
                                    int lwork, float *rwork, int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb,
                 lwork_ = lwork;
    __LAPACK_int rank_ = 0;
    cgelsd_(&m_, &n_, &nrhs_, reinterpret_cast<__LAPACK_float_complex *>(a),
            &lda_, reinterpret_cast<__LAPACK_float_complex *>(b), &ldb_, s,
            &rcond, &rank_, reinterpret_cast<__LAPACK_float_complex *>(work),
            &lwork_, rwork, iwork, &info);
    *rank = static_cast<int>(rank_);
    return static_cast<int>(info);
}

int AccelerateLapackBackend::zgelsd(int m, int n, int nrhs, complex128_t *a,
                                    int lda, complex128_t *b, int ldb,
                                    double *s, double rcond, int *rank,
                                    complex128_t *work, int lwork,
                                    double *rwork, int *iwork) {
    __LAPACK_int info = 0;
    __LAPACK_int m_ = m, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb,
                 lwork_ = lwork;
    __LAPACK_int rank_ = 0;
    zgelsd_(&m_, &n_, &nrhs_, reinterpret_cast<__LAPACK_double_complex *>(a),
            &lda_, reinterpret_cast<__LAPACK_double_complex *>(b), &ldb_, s,
            &rcond, &rank_, reinterpret_cast<__LAPACK_double_complex *>(work),
            &lwork_, rwork, iwork, &info);
    *rank = static_cast<int>(rank_);
    return static_cast<int>(info);
}

} // namespace lapack
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_ACCELERATE
