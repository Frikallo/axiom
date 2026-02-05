#pragma once

// Apple Accelerate LAPACK backend implementation
// Uses clapack.h from the Accelerate framework

#ifdef AXIOM_USE_ACCELERATE

#include "lapack_backend.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace lapack {

class AccelerateLapackBackend : public LapackBackend {
  public:
    AccelerateLapackBackend() = default;
    ~AccelerateLapackBackend() override = default;

    // Backend info
    const char *name() const override { return "Accelerate"; }
    LapackType type() const override { return LapackType::Accelerate; }
    bool has_lapack() const override { return true; }

    // LU decomposition
    int sgetrf(int m, int n, float *a, int lda, int *ipiv) override;
    int dgetrf(int m, int n, double *a, int lda, int *ipiv) override;
    int cgetrf(int m, int n, complex64_t *a, int lda, int *ipiv) override;
    int zgetrf(int m, int n, complex128_t *a, int lda, int *ipiv) override;

    // Matrix inverse
    int sgetri(int n, float *a, int lda, const int *ipiv, float *work,
               int lwork) override;
    int dgetri(int n, double *a, int lda, const int *ipiv, double *work,
               int lwork) override;
    int cgetri(int n, complex64_t *a, int lda, const int *ipiv,
               complex64_t *work, int lwork) override;
    int zgetri(int n, complex128_t *a, int lda, const int *ipiv,
               complex128_t *work, int lwork) override;

    // Linear system solve
    int sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b,
              int ldb) override;
    int dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b,
              int ldb) override;
    int cgesv(int n, int nrhs, complex64_t *a, int lda, int *ipiv,
              complex64_t *b, int ldb) override;
    int zgesv(int n, int nrhs, complex128_t *a, int lda, int *ipiv,
              complex128_t *b, int ldb) override;

    // SVD
    int sgesdd(char jobz, int m, int n, float *a, int lda, float *s, float *u,
               int ldu, float *vt, int ldvt, float *work, int lwork,
               int *iwork) override;
    int dgesdd(char jobz, int m, int n, double *a, int lda, double *s,
               double *u, int ldu, double *vt, int ldvt, double *work,
               int lwork, int *iwork) override;
    int cgesdd(char jobz, int m, int n, complex64_t *a, int lda, float *s,
               complex64_t *u, int ldu, complex64_t *vt, int ldvt,
               complex64_t *work, int lwork, float *rwork, int *iwork) override;
    int zgesdd(char jobz, int m, int n, complex128_t *a, int lda, double *s,
               complex128_t *u, int ldu, complex128_t *vt, int ldvt,
               complex128_t *work, int lwork, double *rwork,
               int *iwork) override;

    // QR decomposition
    int sgeqrf(int m, int n, float *a, int lda, float *tau, float *work,
               int lwork) override;
    int sorgqr(int m, int n, int k, float *a, int lda, const float *tau,
               float *work, int lwork) override;
    int dgeqrf(int m, int n, double *a, int lda, double *tau, double *work,
               int lwork) override;
    int dorgqr(int m, int n, int k, double *a, int lda, const double *tau,
               double *work, int lwork) override;
    int cgeqrf(int m, int n, complex64_t *a, int lda, complex64_t *tau,
               complex64_t *work, int lwork) override;
    int cungqr(int m, int n, int k, complex64_t *a, int lda,
               const complex64_t *tau, complex64_t *work, int lwork) override;
    int zgeqrf(int m, int n, complex128_t *a, int lda, complex128_t *tau,
               complex128_t *work, int lwork) override;
    int zungqr(int m, int n, int k, complex128_t *a, int lda,
               const complex128_t *tau, complex128_t *work, int lwork) override;

    // Cholesky decomposition
    int spotrf(char uplo, int n, float *a, int lda) override;
    int dpotrf(char uplo, int n, double *a, int lda) override;
    int cpotrf(char uplo, int n, complex64_t *a, int lda) override;
    int zpotrf(char uplo, int n, complex128_t *a, int lda) override;

    // General eigenvalue decomposition
    int sgeev(char jobvl, char jobvr, int n, float *a, int lda, float *wr,
              float *wi, float *vl, int ldvl, float *vr, int ldvr, float *work,
              int lwork) override;
    int dgeev(char jobvl, char jobvr, int n, double *a, int lda, double *wr,
              double *wi, double *vl, int ldvl, double *vr, int ldvr,
              double *work, int lwork) override;
    int cgeev(char jobvl, char jobvr, int n, complex64_t *a, int lda,
              complex64_t *w, complex64_t *vl, int ldvl, complex64_t *vr,
              int ldvr, complex64_t *work, int lwork, float *rwork) override;
    int zgeev(char jobvl, char jobvr, int n, complex128_t *a, int lda,
              complex128_t *w, complex128_t *vl, int ldvl, complex128_t *vr,
              int ldvr, complex128_t *work, int lwork, double *rwork) override;

    // Symmetric/Hermitian eigenvalue decomposition
    int ssyev(char jobz, char uplo, int n, float *a, int lda, float *w,
              float *work, int lwork) override;
    int dsyev(char jobz, char uplo, int n, double *a, int lda, double *w,
              double *work, int lwork) override;
    int cheev(char jobz, char uplo, int n, complex64_t *a, int lda, float *w,
              complex64_t *work, int lwork, float *rwork) override;
    int zheev(char jobz, char uplo, int n, complex128_t *a, int lda, double *w,
              complex128_t *work, int lwork, double *rwork) override;

    // Least squares
    int sgelsd(int m, int n, int nrhs, float *a, int lda, float *b, int ldb,
               float *s, float rcond, int *rank, float *work, int lwork,
               int *iwork) override;
    int dgelsd(int m, int n, int nrhs, double *a, int lda, double *b, int ldb,
               double *s, double rcond, int *rank, double *work, int lwork,
               int *iwork) override;
    int cgelsd(int m, int n, int nrhs, complex64_t *a, int lda, complex64_t *b,
               int ldb, float *s, float rcond, int *rank, complex64_t *work,
               int lwork, float *rwork, int *iwork) override;
    int zgelsd(int m, int n, int nrhs, complex128_t *a, int lda,
               complex128_t *b, int ldb, double *s, double rcond, int *rank,
               complex128_t *work, int lwork, double *rwork,
               int *iwork) override;
};

} // namespace lapack
} // namespace cpu
} // namespace backends
} // namespace axiom

#endif // AXIOM_USE_ACCELERATE
