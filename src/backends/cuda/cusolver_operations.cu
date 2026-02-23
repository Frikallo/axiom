#include "cusolver_operations.hpp"
#include "cuda_context.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#endif

namespace axiom {
namespace backends {
namespace cuda {

#ifdef AXIOM_CUDA_SUPPORT

// ============================================================================
// Helper: check cuSOLVER status and translate to LAPACK info convention
// ============================================================================

static void check_cusolver(cusolverStatus_t status, const char *routine) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuSOLVER ") + routine + " failed with status " +
            std::to_string(static_cast<int>(status)));
    }
}

// Helper to allocate device workspace of `bytes` bytes, returning a raw ptr.
// Caller is responsible for cudaFree.
static void *device_alloc(size_t bytes) {
    void *ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

// Helper to allocate a device int for info output.
static int *device_info_alloc() {
    int *d_info = nullptr;
    cudaMalloc(&d_info, sizeof(int));
    return d_info;
}

// Copy device info int back to host and free.
static int device_info_get(int *d_info, cudaStream_t stream) {
    int h_info = 0;
    cudaMemcpyAsync(&h_info, d_info, sizeof(int),
                     cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_info);
    return h_info;
}

// ============================================================================
// CusolverLapackBackend
// ============================================================================
// Implements the LapackBackend interface by calling cuSOLVER dense routines.
// Data must already be in device memory in column-major layout.
//
// Workspace queries: cuSOLVER provides its own workspace sizing via
// cusolverDn*_bufferSize calls.  The LAPACK lwork=-1 convention is handled
// by detecting lwork==-1 in the wrapper and calling the bufferSize variant.
// ============================================================================

class CusolverLapackBackend
    : public cpu::lapack::LapackBackend {
  public:
    CusolverLapackBackend() = default;
    ~CusolverLapackBackend() override = default;

    const char *name() const override { return "cuSOLVER"; }
    cpu::lapack::LapackType type() const override {
        return cpu::lapack::LapackType::Auto;
    }
    bool has_lapack() const override { return true; }

  private:
    cusolverDnHandle_t handle() const {
        return static_cast<cusolverDnHandle_t>(
            CudaContext::instance().cusolver_handle());
    }
    cudaStream_t stream() const {
        return static_cast<cudaStream_t>(
            CudaContext::instance().stream());
    }

  public:
    // ====================================================================
    // LU Decomposition (getrf)
    // ====================================================================

    int sgetrf(int m, int n, float *a, int lda, int *ipiv) override {
        int lwork = 0;
        check_cusolver(
            cusolverDnSgetrf_bufferSize(handle(), m, n, a, lda, &lwork),
            "Sgetrf_bufferSize");

        float *d_work = static_cast<float *>(
            device_alloc(lwork * sizeof(float)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnSgetrf(handle(), m, n, a, lda, d_work, ipiv, d_info),
            "Sgetrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int dgetrf(int m, int n, double *a, int lda, int *ipiv) override {
        int lwork = 0;
        check_cusolver(
            cusolverDnDgetrf_bufferSize(handle(), m, n, a, lda, &lwork),
            "Dgetrf_bufferSize");

        double *d_work = static_cast<double *>(
            device_alloc(lwork * sizeof(double)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnDgetrf(handle(), m, n, a, lda, d_work, ipiv, d_info),
            "Dgetrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int cgetrf(int m, int n, cpu::lapack::complex64_t *a, int lda,
               int *ipiv) override {
        auto *ca = reinterpret_cast<cuComplex *>(a);
        int lwork = 0;
        check_cusolver(
            cusolverDnCgetrf_bufferSize(handle(), m, n, ca, lda, &lwork),
            "Cgetrf_bufferSize");

        cuComplex *d_work = static_cast<cuComplex *>(
            device_alloc(lwork * sizeof(cuComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnCgetrf(handle(), m, n, ca, lda, d_work, ipiv, d_info),
            "Cgetrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int zgetrf(int m, int n, cpu::lapack::complex128_t *a, int lda,
               int *ipiv) override {
        auto *za = reinterpret_cast<cuDoubleComplex *>(a);
        int lwork = 0;
        check_cusolver(
            cusolverDnZgetrf_bufferSize(handle(), m, n, za, lda, &lwork),
            "Zgetrf_bufferSize");

        cuDoubleComplex *d_work = static_cast<cuDoubleComplex *>(
            device_alloc(lwork * sizeof(cuDoubleComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnZgetrf(handle(), m, n, za, lda, d_work, ipiv, d_info),
            "Zgetrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    // ====================================================================
    // Matrix Inverse (getri) — cuSOLVER doesn't have getri directly.
    // Use getrs with identity RHS as a workaround, or throw.
    // ====================================================================

    int sgetri(int n, float *a, int lda, const int *ipiv, float *work,
               int lwork) override {
        (void)a; (void)lda; (void)ipiv; (void)work; (void)lwork;
        throw std::runtime_error(
            "cuSOLVER does not provide getri — use CPU fallback for inv()");
    }

    int dgetri(int n, double *a, int lda, const int *ipiv, double *work,
               int lwork) override {
        (void)a; (void)lda; (void)ipiv; (void)work; (void)lwork;
        throw std::runtime_error(
            "cuSOLVER does not provide getri — use CPU fallback for inv()");
    }

    int cgetri(int n, cpu::lapack::complex64_t *a, int lda, const int *ipiv,
               cpu::lapack::complex64_t *work, int lwork) override {
        (void)a; (void)lda; (void)ipiv; (void)work; (void)lwork;
        throw std::runtime_error(
            "cuSOLVER does not provide getri — use CPU fallback for inv()");
    }

    int zgetri(int n, cpu::lapack::complex128_t *a, int lda, const int *ipiv,
               cpu::lapack::complex128_t *work, int lwork) override {
        (void)a; (void)lda; (void)ipiv; (void)work; (void)lwork;
        throw std::runtime_error(
            "cuSOLVER does not provide getri — use CPU fallback for inv()");
    }

    // ====================================================================
    // Linear System Solve (gesv) — LU + getrs
    // cuSOLVER doesn't have gesv directly; do getrf + getrs.
    // ====================================================================

    int sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b,
              int ldb) override {
        int info = sgetrf(n, n, a, lda, ipiv);
        if (info != 0) return info;

        int *d_info = device_info_alloc();
        check_cusolver(
            cusolverDnSgetrs(handle(), CUBLAS_OP_N, n, nrhs, a, lda,
                             ipiv, b, ldb, d_info),
            "Sgetrs");
        return device_info_get(d_info, stream());
    }

    int dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b,
              int ldb) override {
        int info = dgetrf(n, n, a, lda, ipiv);
        if (info != 0) return info;

        int *d_info = device_info_alloc();
        check_cusolver(
            cusolverDnDgetrs(handle(), CUBLAS_OP_N, n, nrhs, a, lda,
                             ipiv, b, ldb, d_info),
            "Dgetrs");
        return device_info_get(d_info, stream());
    }

    int cgesv(int n, int nrhs, cpu::lapack::complex64_t *a, int lda,
              int *ipiv, cpu::lapack::complex64_t *b, int ldb) override {
        int info = cgetrf(n, n, a, lda, ipiv);
        if (info != 0) return info;

        auto *ca = reinterpret_cast<cuComplex *>(a);
        auto *cb = reinterpret_cast<cuComplex *>(b);
        int *d_info = device_info_alloc();
        check_cusolver(
            cusolverDnCgetrs(handle(), CUBLAS_OP_N, n, nrhs, ca, lda,
                             ipiv, cb, ldb, d_info),
            "Cgetrs");
        return device_info_get(d_info, stream());
    }

    int zgesv(int n, int nrhs, cpu::lapack::complex128_t *a, int lda,
              int *ipiv, cpu::lapack::complex128_t *b, int ldb) override {
        int info = zgetrf(n, n, a, lda, ipiv);
        if (info != 0) return info;

        auto *za = reinterpret_cast<cuDoubleComplex *>(a);
        auto *zb = reinterpret_cast<cuDoubleComplex *>(b);
        int *d_info = device_info_alloc();
        check_cusolver(
            cusolverDnZgetrs(handle(), CUBLAS_OP_N, n, nrhs, za, lda,
                             ipiv, zb, ldb, d_info),
            "Zgetrs");
        return device_info_get(d_info, stream());
    }

    // ====================================================================
    // SVD (gesdd) — cusolverDnSgesvd / cusolverDnDgesvd
    // cuSOLVER uses gesvd (not gesdd), but the interface is compatible.
    // ====================================================================

    int sgesdd(char jobz, int m, int n, float *a, int lda, float *s,
               float *u, int ldu, float *vt, int ldvt, float *work,
               int lwork, int *iwork) override {
        (void)iwork;
        // Workspace query
        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnSgesvd_bufferSize(handle(), m, n, &required),
                "Sgesvd_bufferSize");
            *work = static_cast<float>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnSgesvd_bufferSize(handle(), m, n, &buf_size),
            "Sgesvd_bufferSize");

        float *d_work = static_cast<float *>(
            device_alloc(buf_size * sizeof(float)));
        // cuSOLVER gesvd has rwork param for convergence info (can be null)
        int *d_info = device_info_alloc();

        // Map jobz to cuSOLVER jobu/jobvt
        signed char jobu = (jobz == 'N') ? 'N' : ((jobz == 'S') ? 'S' : 'A');
        signed char jobvt = jobu;

        check_cusolver(
            cusolverDnSgesvd(handle(), jobu, jobvt, m, n, a, lda,
                             s, u, ldu, vt, ldvt, d_work, buf_size,
                             nullptr, d_info),
            "Sgesvd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int dgesdd(char jobz, int m, int n, double *a, int lda, double *s,
               double *u, int ldu, double *vt, int ldvt, double *work,
               int lwork, int *iwork) override {
        (void)iwork;
        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnDgesvd_bufferSize(handle(), m, n, &required),
                "Dgesvd_bufferSize");
            *work = static_cast<double>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnDgesvd_bufferSize(handle(), m, n, &buf_size),
            "Dgesvd_bufferSize");

        double *d_work = static_cast<double *>(
            device_alloc(buf_size * sizeof(double)));
        int *d_info = device_info_alloc();

        signed char jobu = (jobz == 'N') ? 'N' : ((jobz == 'S') ? 'S' : 'A');
        signed char jobvt = jobu;

        check_cusolver(
            cusolverDnDgesvd(handle(), jobu, jobvt, m, n, a, lda,
                             s, u, ldu, vt, ldvt, d_work, buf_size,
                             nullptr, d_info),
            "Dgesvd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int cgesdd(char jobz, int m, int n, cpu::lapack::complex64_t *a,
               int lda, float *s, cpu::lapack::complex64_t *u, int ldu,
               cpu::lapack::complex64_t *vt, int ldvt,
               cpu::lapack::complex64_t *work, int lwork,
               float *rwork, int *iwork) override {
        (void)iwork; (void)rwork;
        auto *ca = reinterpret_cast<cuComplex *>(a);
        auto *cu = reinterpret_cast<cuComplex *>(u);
        auto *cvt = reinterpret_cast<cuComplex *>(vt);

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnCgesvd_bufferSize(handle(), m, n, &required),
                "Cgesvd_bufferSize");
            // Store as real part of work[0]
            work[0] = cpu::lapack::complex64_t(
                static_cast<float>(required), 0.0f);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnCgesvd_bufferSize(handle(), m, n, &buf_size),
            "Cgesvd_bufferSize");

        cuComplex *d_work = static_cast<cuComplex *>(
            device_alloc(buf_size * sizeof(cuComplex)));
        int *d_info = device_info_alloc();

        signed char jobu = (jobz == 'N') ? 'N' : ((jobz == 'S') ? 'S' : 'A');
        signed char jobvt = jobu;

        check_cusolver(
            cusolverDnCgesvd(handle(), jobu, jobvt, m, n, ca, lda,
                             s, cu, ldu, cvt, ldvt, d_work, buf_size,
                             nullptr, d_info),
            "Cgesvd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int zgesdd(char jobz, int m, int n, cpu::lapack::complex128_t *a,
               int lda, double *s, cpu::lapack::complex128_t *u, int ldu,
               cpu::lapack::complex128_t *vt, int ldvt,
               cpu::lapack::complex128_t *work, int lwork,
               double *rwork, int *iwork) override {
        (void)iwork; (void)rwork;
        auto *za = reinterpret_cast<cuDoubleComplex *>(a);
        auto *zu = reinterpret_cast<cuDoubleComplex *>(u);
        auto *zvt = reinterpret_cast<cuDoubleComplex *>(vt);

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnZgesvd_bufferSize(handle(), m, n, &required),
                "Zgesvd_bufferSize");
            work[0] = cpu::lapack::complex128_t(
                static_cast<double>(required), 0.0);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnZgesvd_bufferSize(handle(), m, n, &buf_size),
            "Zgesvd_bufferSize");

        cuDoubleComplex *d_work = static_cast<cuDoubleComplex *>(
            device_alloc(buf_size * sizeof(cuDoubleComplex)));
        int *d_info = device_info_alloc();

        signed char jobu = (jobz == 'N') ? 'N' : ((jobz == 'S') ? 'S' : 'A');
        signed char jobvt = jobu;

        check_cusolver(
            cusolverDnZgesvd(handle(), jobu, jobvt, m, n, za, lda,
                             s, zu, ldu, zvt, ldvt, d_work, buf_size,
                             nullptr, d_info),
            "Zgesvd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    // ====================================================================
    // QR Decomposition (geqrf + orgqr/ungqr)
    // ====================================================================

    int sgeqrf(int m, int n, float *a, int lda, float *tau, float *work,
               int lwork) override {
        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnSgeqrf_bufferSize(handle(), m, n, a, lda, &required),
                "Sgeqrf_bufferSize");
            *work = static_cast<float>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnSgeqrf_bufferSize(handle(), m, n, a, lda, &buf_size),
            "Sgeqrf_bufferSize");

        float *d_work = static_cast<float *>(
            device_alloc(buf_size * sizeof(float)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnSgeqrf(handle(), m, n, a, lda, tau,
                             d_work, buf_size, d_info),
            "Sgeqrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int sorgqr(int m, int n, int k, float *a, int lda, const float *tau,
               float *work, int lwork) override {
        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnSorgqr_bufferSize(handle(), m, n, k, a, lda,
                                            tau, &required),
                "Sorgqr_bufferSize");
            *work = static_cast<float>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnSorgqr_bufferSize(handle(), m, n, k, a, lda,
                                        tau, &buf_size),
            "Sorgqr_bufferSize");

        float *d_work = static_cast<float *>(
            device_alloc(buf_size * sizeof(float)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnSorgqr(handle(), m, n, k, a, lda, tau,
                             d_work, buf_size, d_info),
            "Sorgqr");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int dgeqrf(int m, int n, double *a, int lda, double *tau, double *work,
               int lwork) override {
        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnDgeqrf_bufferSize(handle(), m, n, a, lda, &required),
                "Dgeqrf_bufferSize");
            *work = static_cast<double>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnDgeqrf_bufferSize(handle(), m, n, a, lda, &buf_size),
            "Dgeqrf_bufferSize");

        double *d_work = static_cast<double *>(
            device_alloc(buf_size * sizeof(double)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnDgeqrf(handle(), m, n, a, lda, tau,
                             d_work, buf_size, d_info),
            "Dgeqrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int dorgqr(int m, int n, int k, double *a, int lda, const double *tau,
               double *work, int lwork) override {
        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnDorgqr_bufferSize(handle(), m, n, k, a, lda,
                                            tau, &required),
                "Dorgqr_bufferSize");
            *work = static_cast<double>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnDorgqr_bufferSize(handle(), m, n, k, a, lda,
                                        tau, &buf_size),
            "Dorgqr_bufferSize");

        double *d_work = static_cast<double *>(
            device_alloc(buf_size * sizeof(double)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnDorgqr(handle(), m, n, k, a, lda, tau,
                             d_work, buf_size, d_info),
            "Dorgqr");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int cgeqrf(int m, int n, cpu::lapack::complex64_t *a, int lda,
               cpu::lapack::complex64_t *tau, cpu::lapack::complex64_t *work,
               int lwork) override {
        auto *ca = reinterpret_cast<cuComplex *>(a);
        auto *ctau = reinterpret_cast<cuComplex *>(tau);

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnCgeqrf_bufferSize(handle(), m, n, ca, lda,
                                            &required),
                "Cgeqrf_bufferSize");
            work[0] = cpu::lapack::complex64_t(
                static_cast<float>(required), 0.0f);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnCgeqrf_bufferSize(handle(), m, n, ca, lda, &buf_size),
            "Cgeqrf_bufferSize");

        cuComplex *d_work = static_cast<cuComplex *>(
            device_alloc(buf_size * sizeof(cuComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnCgeqrf(handle(), m, n, ca, lda, ctau,
                             d_work, buf_size, d_info),
            "Cgeqrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int cungqr(int m, int n, int k, cpu::lapack::complex64_t *a, int lda,
               const cpu::lapack::complex64_t *tau,
               cpu::lapack::complex64_t *work, int lwork) override {
        auto *ca = reinterpret_cast<cuComplex *>(a);
        auto *ctau = const_cast<cuComplex *>(
            reinterpret_cast<const cuComplex *>(tau));

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnCungqr_bufferSize(handle(), m, n, k, ca, lda,
                                            ctau, &required),
                "Cungqr_bufferSize");
            work[0] = cpu::lapack::complex64_t(
                static_cast<float>(required), 0.0f);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnCungqr_bufferSize(handle(), m, n, k, ca, lda,
                                        ctau, &buf_size),
            "Cungqr_bufferSize");

        cuComplex *d_work = static_cast<cuComplex *>(
            device_alloc(buf_size * sizeof(cuComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnCungqr(handle(), m, n, k, ca, lda, ctau,
                             d_work, buf_size, d_info),
            "Cungqr");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int zgeqrf(int m, int n, cpu::lapack::complex128_t *a, int lda,
               cpu::lapack::complex128_t *tau,
               cpu::lapack::complex128_t *work, int lwork) override {
        auto *za = reinterpret_cast<cuDoubleComplex *>(a);
        auto *ztau = reinterpret_cast<cuDoubleComplex *>(tau);

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnZgeqrf_bufferSize(handle(), m, n, za, lda,
                                            &required),
                "Zgeqrf_bufferSize");
            work[0] = cpu::lapack::complex128_t(
                static_cast<double>(required), 0.0);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnZgeqrf_bufferSize(handle(), m, n, za, lda, &buf_size),
            "Zgeqrf_bufferSize");

        cuDoubleComplex *d_work = static_cast<cuDoubleComplex *>(
            device_alloc(buf_size * sizeof(cuDoubleComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnZgeqrf(handle(), m, n, za, lda, ztau,
                             d_work, buf_size, d_info),
            "Zgeqrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int zungqr(int m, int n, int k, cpu::lapack::complex128_t *a, int lda,
               const cpu::lapack::complex128_t *tau,
               cpu::lapack::complex128_t *work, int lwork) override {
        auto *za = reinterpret_cast<cuDoubleComplex *>(a);
        auto *ztau = const_cast<cuDoubleComplex *>(
            reinterpret_cast<const cuDoubleComplex *>(tau));

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnZungqr_bufferSize(handle(), m, n, k, za, lda,
                                            ztau, &required),
                "Zungqr_bufferSize");
            work[0] = cpu::lapack::complex128_t(
                static_cast<double>(required), 0.0);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnZungqr_bufferSize(handle(), m, n, k, za, lda,
                                        ztau, &buf_size),
            "Zungqr_bufferSize");

        cuDoubleComplex *d_work = static_cast<cuDoubleComplex *>(
            device_alloc(buf_size * sizeof(cuDoubleComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnZungqr(handle(), m, n, k, za, lda, ztau,
                             d_work, buf_size, d_info),
            "Zungqr");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    // ====================================================================
    // Cholesky Decomposition (potrf)
    // ====================================================================

    int spotrf(char uplo, int n, float *a, int lda) override {
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        int buf_size = 0;
        check_cusolver(
            cusolverDnSpotrf_bufferSize(handle(), fill, n, a, lda, &buf_size),
            "Spotrf_bufferSize");

        float *d_work = static_cast<float *>(
            device_alloc(buf_size * sizeof(float)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnSpotrf(handle(), fill, n, a, lda,
                             d_work, buf_size, d_info),
            "Spotrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int dpotrf(char uplo, int n, double *a, int lda) override {
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        int buf_size = 0;
        check_cusolver(
            cusolverDnDpotrf_bufferSize(handle(), fill, n, a, lda, &buf_size),
            "Dpotrf_bufferSize");

        double *d_work = static_cast<double *>(
            device_alloc(buf_size * sizeof(double)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnDpotrf(handle(), fill, n, a, lda,
                             d_work, buf_size, d_info),
            "Dpotrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int cpotrf(char uplo, int n, cpu::lapack::complex64_t *a,
               int lda) override {
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        auto *ca = reinterpret_cast<cuComplex *>(a);
        int buf_size = 0;
        check_cusolver(
            cusolverDnCpotrf_bufferSize(handle(), fill, n, ca, lda, &buf_size),
            "Cpotrf_bufferSize");

        cuComplex *d_work = static_cast<cuComplex *>(
            device_alloc(buf_size * sizeof(cuComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnCpotrf(handle(), fill, n, ca, lda,
                             d_work, buf_size, d_info),
            "Cpotrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int zpotrf(char uplo, int n, cpu::lapack::complex128_t *a,
               int lda) override {
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        auto *za = reinterpret_cast<cuDoubleComplex *>(a);
        int buf_size = 0;
        check_cusolver(
            cusolverDnZpotrf_bufferSize(handle(), fill, n, za, lda, &buf_size),
            "Zpotrf_bufferSize");

        cuDoubleComplex *d_work = static_cast<cuDoubleComplex *>(
            device_alloc(buf_size * sizeof(cuDoubleComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnZpotrf(handle(), fill, n, za, lda,
                             d_work, buf_size, d_info),
            "Zpotrf");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    // ====================================================================
    // Eigenvalue Decomposition — General (geev)
    // cuSOLVER does not provide geev (non-symmetric eigenvalues).
    // Fall back to CPU.
    // ====================================================================

    int sgeev(char jobvl, char jobvr, int n, float *a, int lda,
              float *wr, float *wi, float *vl, int ldvl,
              float *vr, int ldvr, float *work, int lwork) override {
        (void)jobvl; (void)jobvr; (void)a; (void)lda;
        (void)wr; (void)wi; (void)vl; (void)ldvl;
        (void)vr; (void)ldvr; (void)work; (void)lwork;
        throw std::runtime_error(
            "cuSOLVER does not provide geev — use CPU fallback");
    }

    int dgeev(char jobvl, char jobvr, int n, double *a, int lda,
              double *wr, double *wi, double *vl, int ldvl,
              double *vr, int ldvr, double *work, int lwork) override {
        (void)jobvl; (void)jobvr; (void)a; (void)lda;
        (void)wr; (void)wi; (void)vl; (void)ldvl;
        (void)vr; (void)ldvr; (void)work; (void)lwork;
        throw std::runtime_error(
            "cuSOLVER does not provide geev — use CPU fallback");
    }

    int cgeev(char jobvl, char jobvr, int n, cpu::lapack::complex64_t *a,
              int lda, cpu::lapack::complex64_t *w,
              cpu::lapack::complex64_t *vl, int ldvl,
              cpu::lapack::complex64_t *vr, int ldvr,
              cpu::lapack::complex64_t *work, int lwork,
              float *rwork) override {
        (void)jobvl; (void)jobvr; (void)a; (void)lda;
        (void)w; (void)vl; (void)ldvl; (void)vr; (void)ldvr;
        (void)work; (void)lwork; (void)rwork;
        throw std::runtime_error(
            "cuSOLVER does not provide geev — use CPU fallback");
    }

    int zgeev(char jobvl, char jobvr, int n, cpu::lapack::complex128_t *a,
              int lda, cpu::lapack::complex128_t *w,
              cpu::lapack::complex128_t *vl, int ldvl,
              cpu::lapack::complex128_t *vr, int ldvr,
              cpu::lapack::complex128_t *work, int lwork,
              double *rwork) override {
        (void)jobvl; (void)jobvr; (void)a; (void)lda;
        (void)w; (void)vl; (void)ldvl; (void)vr; (void)ldvr;
        (void)work; (void)lwork; (void)rwork;
        throw std::runtime_error(
            "cuSOLVER does not provide geev — use CPU fallback");
    }

    // ====================================================================
    // Symmetric/Hermitian Eigenvalue Decomposition (syev/heev)
    // cusolverDnSsyevd / cusolverDnDsyevd (divide-and-conquer)
    // ====================================================================

    int ssyev(char jobz, char uplo, int n, float *a, int lda, float *w,
              float *work, int lwork) override {
        // Delegate to syevd (faster divide-and-conquer)
        int dummy_iwork = 0;
        return ssyevd(jobz, uplo, n, a, lda, w, work, lwork,
                       &dummy_iwork, 0);
    }

    int dsyev(char jobz, char uplo, int n, double *a, int lda, double *w,
              double *work, int lwork) override {
        int dummy_iwork = 0;
        return dsyevd(jobz, uplo, n, a, lda, w, work, lwork,
                       &dummy_iwork, 0);
    }

    int cheev(char jobz, char uplo, int n, cpu::lapack::complex64_t *a,
              int lda, float *w, cpu::lapack::complex64_t *work, int lwork,
              float *rwork) override {
        (void)rwork;
        auto *ca = reinterpret_cast<cuComplex *>(a);
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        cusolverEigMode_t mode = (jobz == 'V' || jobz == 'v')
                                     ? CUSOLVER_EIG_MODE_VECTOR
                                     : CUSOLVER_EIG_MODE_NOVECTOR;

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnCheevd_bufferSize(handle(), mode, fill, n, ca,
                                            lda, w, &required),
                "Cheevd_bufferSize");
            work[0] = cpu::lapack::complex64_t(
                static_cast<float>(required), 0.0f);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnCheevd_bufferSize(handle(), mode, fill, n, ca,
                                        lda, w, &buf_size),
            "Cheevd_bufferSize");

        cuComplex *d_work = static_cast<cuComplex *>(
            device_alloc(buf_size * sizeof(cuComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnCheevd(handle(), mode, fill, n, ca, lda, w,
                             d_work, buf_size, d_info),
            "Cheevd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int zheev(char jobz, char uplo, int n, cpu::lapack::complex128_t *a,
              int lda, double *w, cpu::lapack::complex128_t *work, int lwork,
              double *rwork) override {
        (void)rwork;
        auto *za = reinterpret_cast<cuDoubleComplex *>(a);
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        cusolverEigMode_t mode = (jobz == 'V' || jobz == 'v')
                                     ? CUSOLVER_EIG_MODE_VECTOR
                                     : CUSOLVER_EIG_MODE_NOVECTOR;

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnZheevd_bufferSize(handle(), mode, fill, n, za,
                                            lda, w, &required),
                "Zheevd_bufferSize");
            work[0] = cpu::lapack::complex128_t(
                static_cast<double>(required), 0.0);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnZheevd_bufferSize(handle(), mode, fill, n, za,
                                        lda, w, &buf_size),
            "Zheevd_bufferSize");

        cuDoubleComplex *d_work = static_cast<cuDoubleComplex *>(
            device_alloc(buf_size * sizeof(cuDoubleComplex)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnZheevd(handle(), mode, fill, n, za, lda, w,
                             d_work, buf_size, d_info),
            "Zheevd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    // ====================================================================
    // Symmetric Eigenvalue Divide-and-Conquer (syevd)
    // cusolverDnSsyevd / cusolverDnDsyevd
    // ====================================================================

    int ssyevd(char jobz, char uplo, int n, float *a, int lda, float *w,
               float *work, int lwork, int *iwork, int liwork) override {
        (void)iwork; (void)liwork;
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        cusolverEigMode_t mode = (jobz == 'V' || jobz == 'v')
                                     ? CUSOLVER_EIG_MODE_VECTOR
                                     : CUSOLVER_EIG_MODE_NOVECTOR;

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnSsyevd_bufferSize(handle(), mode, fill, n, a,
                                            lda, w, &required),
                "Ssyevd_bufferSize");
            *work = static_cast<float>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnSsyevd_bufferSize(handle(), mode, fill, n, a,
                                        lda, w, &buf_size),
            "Ssyevd_bufferSize");

        float *d_work = static_cast<float *>(
            device_alloc(buf_size * sizeof(float)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnSsyevd(handle(), mode, fill, n, a, lda, w,
                             d_work, buf_size, d_info),
            "Ssyevd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    int dsyevd(char jobz, char uplo, int n, double *a, int lda, double *w,
               double *work, int lwork, int *iwork, int liwork) override {
        (void)iwork; (void)liwork;
        cublasFillMode_t fill = (uplo == 'U' || uplo == 'u')
                                    ? CUBLAS_FILL_MODE_UPPER
                                    : CUBLAS_FILL_MODE_LOWER;
        cusolverEigMode_t mode = (jobz == 'V' || jobz == 'v')
                                     ? CUSOLVER_EIG_MODE_VECTOR
                                     : CUSOLVER_EIG_MODE_NOVECTOR;

        if (lwork == -1 && work != nullptr) {
            int required = 0;
            check_cusolver(
                cusolverDnDsyevd_bufferSize(handle(), mode, fill, n, a,
                                            lda, w, &required),
                "Dsyevd_bufferSize");
            *work = static_cast<double>(required);
            return 0;
        }

        int buf_size = 0;
        check_cusolver(
            cusolverDnDsyevd_bufferSize(handle(), mode, fill, n, a,
                                        lda, w, &buf_size),
            "Dsyevd_bufferSize");

        double *d_work = static_cast<double *>(
            device_alloc(buf_size * sizeof(double)));
        int *d_info = device_info_alloc();

        check_cusolver(
            cusolverDnDsyevd(handle(), mode, fill, n, a, lda, w,
                             d_work, buf_size, d_info),
            "Dsyevd");

        int info = device_info_get(d_info, stream());
        cudaFree(d_work);
        return info;
    }

    // ====================================================================
    // Least Squares (gelsd) — not available in cuSOLVER
    // ====================================================================

    int sgelsd(int m, int n, int nrhs, float *a, int lda, float *b,
               int ldb, float *s, float rcond, int *rank, float *work,
               int lwork, int *iwork) override {
        (void)m; (void)n; (void)nrhs; (void)a; (void)lda;
        (void)b; (void)ldb; (void)s; (void)rcond; (void)rank;
        (void)work; (void)lwork; (void)iwork;
        throw std::runtime_error(
            "cuSOLVER does not provide gelsd — use CPU fallback");
    }

    int dgelsd(int m, int n, int nrhs, double *a, int lda, double *b,
               int ldb, double *s, double rcond, int *rank, double *work,
               int lwork, int *iwork) override {
        (void)m; (void)n; (void)nrhs; (void)a; (void)lda;
        (void)b; (void)ldb; (void)s; (void)rcond; (void)rank;
        (void)work; (void)lwork; (void)iwork;
        throw std::runtime_error(
            "cuSOLVER does not provide gelsd — use CPU fallback");
    }

    int cgelsd(int m, int n, int nrhs, cpu::lapack::complex64_t *a, int lda,
               cpu::lapack::complex64_t *b, int ldb, float *s, float rcond,
               int *rank, cpu::lapack::complex64_t *work, int lwork,
               float *rwork, int *iwork) override {
        (void)m; (void)n; (void)nrhs; (void)a; (void)lda;
        (void)b; (void)ldb; (void)s; (void)rcond; (void)rank;
        (void)work; (void)lwork; (void)rwork; (void)iwork;
        throw std::runtime_error(
            "cuSOLVER does not provide gelsd — use CPU fallback");
    }

    int zgelsd(int m, int n, int nrhs, cpu::lapack::complex128_t *a,
               int lda, cpu::lapack::complex128_t *b, int ldb, double *s,
               double rcond, int *rank, cpu::lapack::complex128_t *work,
               int lwork, double *rwork, int *iwork) override {
        (void)m; (void)n; (void)nrhs; (void)a; (void)lda;
        (void)b; (void)ldb; (void)s; (void)rcond; (void)rank;
        (void)work; (void)lwork; (void)rwork; (void)iwork;
        throw std::runtime_error(
            "cuSOLVER does not provide gelsd — use CPU fallback");
    }
};

// Singleton accessor
static CusolverLapackBackend g_cusolver_backend;

cpu::lapack::LapackBackend *get_cusolver_backend() {
    if (!is_cuda_available()) return nullptr;
    return &g_cusolver_backend;
}

#else // !AXIOM_CUDA_SUPPORT

cpu::lapack::LapackBackend *get_cusolver_backend() {
    return nullptr;
}

#endif // AXIOM_CUDA_SUPPORT

} // namespace cuda
} // namespace backends
} // namespace axiom
