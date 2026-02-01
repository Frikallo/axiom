#include "lapack_native.hpp"

#include <stdexcept>
#include <string>

namespace axiom {
namespace backends {
namespace cpu {
namespace lapack {

[[noreturn]] void NativeLapackBackend::throw_no_lapack(const char *op) const {
    throw std::runtime_error(
        std::string(op) +
        " requires LAPACK. "
        "On macOS, Accelerate framework is used automatically. "
        "On Linux/Windows, install OpenBLAS: apt install libopenblas-dev");
}

// ============================================================================
// LU Decomposition
// ============================================================================

int NativeLapackBackend::sgetrf(int, int, float *, int, int *) {
    throw_no_lapack("LU decomposition (sgetrf)");
}

int NativeLapackBackend::dgetrf(int, int, double *, int, int *) {
    throw_no_lapack("LU decomposition (dgetrf)");
}

int NativeLapackBackend::cgetrf(int, int, complex64_t *, int, int *) {
    throw_no_lapack("LU decomposition (cgetrf)");
}

int NativeLapackBackend::zgetrf(int, int, complex128_t *, int, int *) {
    throw_no_lapack("LU decomposition (zgetrf)");
}

// ============================================================================
// Matrix Inverse
// ============================================================================

int NativeLapackBackend::sgetri(int, float *, int, const int *, float *, int) {
    throw_no_lapack("matrix inverse (sgetri)");
}

int NativeLapackBackend::dgetri(int, double *, int, const int *, double *,
                                int) {
    throw_no_lapack("matrix inverse (dgetri)");
}

int NativeLapackBackend::cgetri(int, complex64_t *, int, const int *,
                                complex64_t *, int) {
    throw_no_lapack("matrix inverse (cgetri)");
}

int NativeLapackBackend::zgetri(int, complex128_t *, int, const int *,
                                complex128_t *, int) {
    throw_no_lapack("matrix inverse (zgetri)");
}

// ============================================================================
// Linear System Solve
// ============================================================================

int NativeLapackBackend::sgesv(int, int, float *, int, int *, float *, int) {
    throw_no_lapack("linear solve (sgesv)");
}

int NativeLapackBackend::dgesv(int, int, double *, int, int *, double *, int) {
    throw_no_lapack("linear solve (dgesv)");
}

int NativeLapackBackend::cgesv(int, int, complex64_t *, int, int *,
                               complex64_t *, int) {
    throw_no_lapack("linear solve (cgesv)");
}

int NativeLapackBackend::zgesv(int, int, complex128_t *, int, int *,
                               complex128_t *, int) {
    throw_no_lapack("linear solve (zgesv)");
}

// ============================================================================
// SVD
// ============================================================================

int NativeLapackBackend::sgesdd(char, int, int, float *, int, float *, float *,
                                int, float *, int, float *, int, int *) {
    throw_no_lapack("SVD (sgesdd)");
}

int NativeLapackBackend::dgesdd(char, int, int, double *, int, double *,
                                double *, int, double *, int, double *, int,
                                int *) {
    throw_no_lapack("SVD (dgesdd)");
}

int NativeLapackBackend::cgesdd(char, int, int, complex64_t *, int, float *,
                                complex64_t *, int, complex64_t *, int,
                                complex64_t *, int, float *, int *) {
    throw_no_lapack("SVD (cgesdd)");
}

int NativeLapackBackend::zgesdd(char, int, int, complex128_t *, int, double *,
                                complex128_t *, int, complex128_t *, int,
                                complex128_t *, int, double *, int *) {
    throw_no_lapack("SVD (zgesdd)");
}

// ============================================================================
// QR Decomposition
// ============================================================================

int NativeLapackBackend::sgeqrf(int, int, float *, int, float *, float *, int) {
    throw_no_lapack("QR decomposition (sgeqrf)");
}

int NativeLapackBackend::sorgqr(int, int, int, float *, int, const float *,
                                float *, int) {
    throw_no_lapack("QR decomposition (sorgqr)");
}

int NativeLapackBackend::dgeqrf(int, int, double *, int, double *, double *,
                                int) {
    throw_no_lapack("QR decomposition (dgeqrf)");
}

int NativeLapackBackend::dorgqr(int, int, int, double *, int, const double *,
                                double *, int) {
    throw_no_lapack("QR decomposition (dorgqr)");
}

int NativeLapackBackend::cgeqrf(int, int, complex64_t *, int, complex64_t *,
                                complex64_t *, int) {
    throw_no_lapack("QR decomposition (cgeqrf)");
}

int NativeLapackBackend::cungqr(int, int, int, complex64_t *, int,
                                const complex64_t *, complex64_t *, int) {
    throw_no_lapack("QR decomposition (cungqr)");
}

int NativeLapackBackend::zgeqrf(int, int, complex128_t *, int, complex128_t *,
                                complex128_t *, int) {
    throw_no_lapack("QR decomposition (zgeqrf)");
}

int NativeLapackBackend::zungqr(int, int, int, complex128_t *, int,
                                const complex128_t *, complex128_t *, int) {
    throw_no_lapack("QR decomposition (zungqr)");
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

int NativeLapackBackend::spotrf(char, int, float *, int) {
    throw_no_lapack("Cholesky decomposition (spotrf)");
}

int NativeLapackBackend::dpotrf(char, int, double *, int) {
    throw_no_lapack("Cholesky decomposition (dpotrf)");
}

int NativeLapackBackend::cpotrf(char, int, complex64_t *, int) {
    throw_no_lapack("Cholesky decomposition (cpotrf)");
}

int NativeLapackBackend::zpotrf(char, int, complex128_t *, int) {
    throw_no_lapack("Cholesky decomposition (zpotrf)");
}

// ============================================================================
// General Eigenvalue Decomposition
// ============================================================================

int NativeLapackBackend::sgeev(char, char, int, float *, int, float *, float *,
                               float *, int, float *, int, float *, int) {
    throw_no_lapack("eigenvalue decomposition (sgeev)");
}

int NativeLapackBackend::dgeev(char, char, int, double *, int, double *,
                               double *, double *, int, double *, int, double *,
                               int) {
    throw_no_lapack("eigenvalue decomposition (dgeev)");
}

int NativeLapackBackend::cgeev(char, char, int, complex64_t *, int,
                               complex64_t *, complex64_t *, int, complex64_t *,
                               int, complex64_t *, int, float *) {
    throw_no_lapack("eigenvalue decomposition (cgeev)");
}

int NativeLapackBackend::zgeev(char, char, int, complex128_t *, int,
                               complex128_t *, complex128_t *, int,
                               complex128_t *, int, complex128_t *, int,
                               double *) {
    throw_no_lapack("eigenvalue decomposition (zgeev)");
}

// ============================================================================
// Symmetric/Hermitian Eigenvalue Decomposition
// ============================================================================

int NativeLapackBackend::ssyev(char, char, int, float *, int, float *, float *,
                               int) {
    throw_no_lapack("symmetric eigenvalue decomposition (ssyev)");
}

int NativeLapackBackend::dsyev(char, char, int, double *, int, double *,
                               double *, int) {
    throw_no_lapack("symmetric eigenvalue decomposition (dsyev)");
}

int NativeLapackBackend::cheev(char, char, int, complex64_t *, int, float *,
                               complex64_t *, int, float *) {
    throw_no_lapack("Hermitian eigenvalue decomposition (cheev)");
}

int NativeLapackBackend::zheev(char, char, int, complex128_t *, int, double *,
                               complex128_t *, int, double *) {
    throw_no_lapack("Hermitian eigenvalue decomposition (zheev)");
}

// ============================================================================
// Least Squares
// ============================================================================

int NativeLapackBackend::sgelsd(int, int, int, float *, int, float *, int,
                                float *, float, int *, float *, int, int *) {
    throw_no_lapack("least squares (sgelsd)");
}

int NativeLapackBackend::dgelsd(int, int, int, double *, int, double *, int,
                                double *, double, int *, double *, int, int *) {
    throw_no_lapack("least squares (dgelsd)");
}

int NativeLapackBackend::cgelsd(int, int, int, complex64_t *, int,
                                complex64_t *, int, float *, float, int *,
                                complex64_t *, int, float *, int *) {
    throw_no_lapack("least squares (cgelsd)");
}

int NativeLapackBackend::zgelsd(int, int, int, complex128_t *, int,
                                complex128_t *, int, double *, double, int *,
                                complex128_t *, int, double *, int *) {
    throw_no_lapack("least squares (zgelsd)");
}

} // namespace lapack
} // namespace cpu
} // namespace backends
} // namespace axiom
