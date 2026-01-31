#pragma once

// Abstract BLAS backend interface for cross-platform matrix operations
// Supports: Apple Accelerate, OpenBLAS, and native C++ with SIMD

#include <cstddef>
#include <memory>
#include <string>

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

// BLAS backend type enumeration
enum class BlasType {
    Auto,       // Auto-detect best available backend
    Accelerate, // Apple Accelerate framework (macOS only)
    OpenBLAS,   // OpenBLAS library (Linux/Windows)
    Native    // Pure C++ implementation with SIMD
};

// Abstract BLAS backend interface
class BlasBackend {
  public:
    virtual ~BlasBackend() = default;

    // ========================================================================
    // BLAS Level 3 - Matrix-Matrix Operations
    // ========================================================================

    // SGEMM: C = alpha * op(A) * op(B) + beta * C (single precision)
    // transA/transB: if true, use transpose of the matrix
    // M: rows of op(A) and C
    // N: columns of op(B) and C
    // K: columns of op(A), rows of op(B)
    // lda, ldb, ldc: leading dimensions
    virtual void sgemm(bool transA, bool transB, size_t M, size_t N, size_t K,
                       float alpha, const float *A, size_t lda, const float *B,
                       size_t ldb, float beta, float *C, size_t ldc) = 0;

    // DGEMM: C = alpha * op(A) * op(B) + beta * C (double precision)
    virtual void dgemm(bool transA, bool transB, size_t M, size_t N, size_t K,
                       double alpha, const double *A, size_t lda,
                       const double *B, size_t ldb, double beta, double *C,
                       size_t ldc) = 0;

    // ========================================================================
    // BLAS Level 2 - Matrix-Vector Operations
    // ========================================================================

    // SGEMV: y = alpha * op(A) * x + beta * y (single precision)
    // transA: if true, use transpose of A
    // M, N: dimensions of A (M rows, N cols)
    virtual void sgemv(bool transA, size_t M, size_t N, float alpha,
                       const float *A, size_t lda, const float *x, size_t incx,
                       float beta, float *y, size_t incy) = 0;

    // DGEMV: y = alpha * op(A) * x + beta * y (double precision)
    virtual void dgemv(bool transA, size_t M, size_t N, double alpha,
                       const double *A, size_t lda, const double *x,
                       size_t incx, double beta, double *y, size_t incy) = 0;

    // ========================================================================
    // BLAS Level 1 - Vector Operations
    // ========================================================================

    // SDOT: dot product of two vectors (single precision)
    virtual float sdot(size_t n, const float *x, size_t incx, const float *y,
                       size_t incy) = 0;

    // DDOT: dot product of two vectors (double precision)
    virtual double ddot(size_t n, const double *x, size_t incx, const double *y,
                        size_t incy) = 0;

    // SAXPY: y = alpha * x + y (single precision)
    virtual void saxpy(size_t n, float alpha, const float *x, size_t incx,
                       float *y, size_t incy) = 0;

    // DAXPY: y = alpha * x + y (double precision)
    virtual void daxpy(size_t n, double alpha, const double *x, size_t incx,
                       double *y, size_t incy) = 0;

    // SNRM2: Euclidean norm of vector (single precision)
    virtual float snrm2(size_t n, const float *x, size_t incx) = 0;

    // DNRM2: Euclidean norm of vector (double precision)
    virtual double dnrm2(size_t n, const double *x, size_t incx) = 0;

    // SSCAL: x = alpha * x (single precision)
    virtual void sscal(size_t n, float alpha, float *x, size_t incx) = 0;

    // DSCAL: x = alpha * x (double precision)
    virtual void dscal(size_t n, double alpha, double *x, size_t incx) = 0;

    // ========================================================================
    // Backend Information
    // ========================================================================

    // Return the name of this backend (e.g., "Accelerate", "OpenBLAS",
    // "Native")
    virtual const char *name() const = 0;

    // Return the type of this backend
    virtual BlasType type() const = 0;
};

// ============================================================================
// Backend Factory Functions
// ============================================================================

// Get the current BLAS backend (singleton)
// Thread-safe: backend is initialized on first call
BlasBackend &get_blas_backend();

// Set the BLAS backend type
// Must be called before any BLAS operations (ideally at program start)
// If not called, auto-detection is used
void set_blas_backend(BlasType type);

// Get the currently selected backend type
BlasType get_blas_backend_type();

// Check if a specific backend type is available on this platform
bool is_backend_available(BlasType type);

// Get the default backend type for this platform (used in auto-detection)
BlasType get_default_backend_type();

} // namespace blas
} // namespace cpu
} // namespace backends
} // namespace axiom
