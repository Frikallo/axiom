#pragma once

// Linear algebra operations for Axiom tensor library
// Provides NumPy/PyTorch-compatible linear algebra functions
// Backed by LAPACK (Accelerate on macOS, OpenBLAS on Linux/Windows)

#include "tensor.hpp"

#include <string>
#include <tuple>

namespace axiom {
namespace linalg {

// ============================================================================
// Result Structures
// ============================================================================

// SVD decomposition result: A = U @ diag(S) @ Vh
struct SVDResult {
    Tensor U;  // Left singular vectors (M, K) or (M, M) for full_matrices
    Tensor S;  // Singular values (K,) where K = min(M, N)
    Tensor Vh; // Right singular vectors (K, N) or (N, N) for full_matrices
};

// QR decomposition result: A = Q @ R
struct QRResult {
    Tensor Q; // Orthogonal matrix (M, K) where K = min(M, N)
    Tensor R; // Upper triangular matrix (K, N)
};

// LU decomposition result: P @ A = L @ U
struct LUResult {
    Tensor L;   // Lower triangular matrix with unit diagonal
    Tensor U;   // Upper triangular matrix
    Tensor P;   // Permutation matrix (or tensor of permutation indices)
    Tensor piv; // Pivot indices (for efficiency)
};

// Eigenvalue decomposition result
struct EigResult {
    Tensor eigenvalues;  // Eigenvalues (may be complex for non-symmetric)
    Tensor eigenvectors; // Eigenvectors as columns
};

// ============================================================================
// Phase 1: Core Operations (det, inv, solve)
// ============================================================================

// Matrix determinant
// Computes det(A) for square matrix A (..., N, N)
// Returns tensor of shape (...,)
Tensor det(const Tensor &a);

// Matrix inverse
// Computes inv(A) for square matrix A (..., N, N)
// Returns tensor of shape (..., N, N)
Tensor inv(const Tensor &a);

// Solve linear system A @ X = B
// a: (..., N, N) square matrix
// b: (..., N) or (..., N, K) right-hand side
// Returns X with same shape as b
Tensor solve(const Tensor &a, const Tensor &b);

// ============================================================================
// Phase 2: Matrix Decompositions
// ============================================================================

// Singular Value Decomposition
// Computes A = U @ diag(S) @ Vh
// a: (..., M, N) input matrix
// full_matrices: if true, U is (M, M) and Vh is (N, N); otherwise economy size
// Returns SVDResult with U, S, Vh
SVDResult svd(const Tensor &a, bool full_matrices = true);

// QR decomposition
// Computes A = Q @ R for matrix A (..., M, N)
// Returns QRResult with Q (M, K) and R (K, N) where K = min(M, N)
QRResult qr(const Tensor &a);

// Cholesky decomposition
// Computes L such that A = L @ L^H for positive definite A (..., N, N)
// upper: if true, returns U such that A = U^H @ U
// Returns lower (or upper) triangular matrix
Tensor cholesky(const Tensor &a, bool upper = false);

// LU decomposition with partial pivoting
// Computes P, L, U such that P @ A = L @ U
// Returns LUResult with L, U, P (permutation matrix), and piv (pivot indices)
LUResult lu(const Tensor &a);

// ============================================================================
// Phase 3: Eigendecomposition & Advanced Operations
// ============================================================================

// General eigenvalue decomposition
// Computes eigenvalues and eigenvectors of general square matrix
// a: (..., N, N) input matrix
// Returns EigResult; eigenvalues may be complex even for real input
EigResult eig(const Tensor &a);

// Symmetric/Hermitian eigenvalue decomposition
// For real symmetric or complex Hermitian matrices
// a: (..., N, N) symmetric/Hermitian matrix
// Returns EigResult with real eigenvalues
EigResult eigh(const Tensor &a);

// Least squares solution
// Solves min ||A @ X - B||_2 for X
// a: (..., M, N) matrix
// b: (..., M) or (..., M, K) right-hand side
// rcond: cutoff ratio for small singular values
// Returns solution X
Tensor lstsq(const Tensor &a, const Tensor &b, double rcond = -1.0);

// Moore-Penrose pseudoinverse
// Computes A^+ using SVD
// a: (..., M, N) input matrix
// rcond: cutoff ratio for small singular values
// Returns (..., N, M) pseudoinverse
Tensor pinv(const Tensor &a, double rcond = 1e-15);

// Matrix norms
// Computes various matrix norms
// ord: "fro" (Frobenius), "nuc" (nuclear), 1, -1, 2, -2, inf, -inf
// For vector inputs: ord can be 0, 1, 2, inf, -inf, or any p > 0
// Returns scalar or tensor depending on input dimensions
Tensor norm(const Tensor &a, const std::string &ord = "fro");
Tensor norm(const Tensor &a, int ord);
Tensor norm(const Tensor &a, double ord);

// Matrix rank
// Computes rank using SVD with tolerance threshold
// tol: tolerance for singular values (negative = auto)
// Returns rank as Int64 tensor
Tensor matrix_rank(const Tensor &a, double tol = -1.0);

// Condition number
// Computes condition number (ratio of largest to smallest singular value)
// p: norm type (1, 2, or inf; default is 2)
// Returns condition number as scalar tensor
Tensor cond(const Tensor &a, int p = 2);

// ============================================================================
// Convenience Functions
// ============================================================================

// Trace of matrix (sum of diagonal elements)
// a: (..., M, N) input matrix
// offset: diagonal offset (0 = main diagonal)
// Returns (...,) tensor of traces
Tensor trace(const Tensor &a, int offset = 0);

// Matrix power
// Computes a^n for integer n (can be negative for invertible matrices)
// a: (..., N, N) square matrix
// n: integer power
// Returns (..., N, N) result
Tensor matrix_power(const Tensor &a, int n);

// Multi-dot product (chained matrix multiplication)
// Computes a[0] @ a[1] @ ... @ a[n-1] efficiently
std::pair<Tensor, Tensor> multi_dot(const std::vector<Tensor> &tensors);

// ============================================================================
// Utility Functions
// ============================================================================

// Check if LAPACK is available
bool has_lapack();

// Get LAPACK backend name
const char *lapack_backend_name();

} // namespace linalg
} // namespace axiom
