# Linear Algebra

The `axiom::linalg` namespace provides a full suite of linear algebra operations backed by LAPACK. From matrix decompositions and linear solvers to vector products and batch operations, the API follows NumPy conventions while delivering native C++ performance through hardware-accelerated BLAS/LAPACK backends.

## BLAS/LAPACK Backends

Axiom automatically selects the best available BLAS/LAPACK implementation at compile time:

| Platform | Backend | Notes |
|----------|---------|-------|
| macOS | Apple Accelerate | Ships with Xcode; no extra setup |
| Linux | OpenBLAS | Install via package manager (`libopenblas-dev`) |
| Windows / fallback | Native | Pure C++ implementation; functional but slower |

You can query the active backend at runtime:

```cpp
using namespace axiom;

if (linalg::has_lapack()) {
    std::cout << "LAPACK backend: "
              << linalg::lapack_backend_name() << std::endl;
    // e.g. "Accelerate", "OpenBLAS", or "None"
}
```

All `linalg` operations work regardless of backend availability -- when no LAPACK library is found, Axiom falls back to its native implementations. GPU tensors are automatically transferred to CPU for LAPACK operations, so you never need to manage device placement manually.

## Matrix Multiplication

The `matmul` method is the workhorse for matrix multiplication. It dispatches to BLAS `sgemm`/`dgemm` under the hood and supports 2D, batched, and broadcast semantics.

```cpp
using namespace axiom;

auto A = Tensor::randn({64, 128});
auto B = Tensor::randn({128, 32});
auto C = A.matmul(B);  // (64, 32)
```

### Zero-Copy Transposed Matmul

Instead of materializing a transposed copy, pass the `transpose_a` or `transpose_b` flags directly. The BLAS kernel reads the original memory in transposed order -- no allocation, no copy:

```cpp
using namespace axiom;

auto W = Tensor::randn({256, 512});
auto x = Tensor::randn({256, 64});

// Compute W^T @ x without creating a transposed tensor
auto result = x.matmul(W, /*transpose_self=*/true);
// Equivalent to: x.T().matmul(W), but zero-copy

// Or using the free function
auto result2 = ops::matmul(W, W, /*transpose_a=*/false,
                           /*transpose_b=*/true);  // W @ W^T
```

### Batch Matmul with Broadcasting

When inputs have more than two dimensions, `matmul` treats all leading dimensions as batch dimensions and broadcasts them following standard rules:

```cpp
using namespace axiom;

// Batched: same batch size
auto A = Tensor::randn({10, 64, 128});
auto B = Tensor::randn({10, 128, 32});
auto C = A.matmul(B);  // (10, 64, 32)

// Broadcast: single matrix applied to a batch
auto weight = Tensor::randn({128, 32});      // (128, 32)
auto batch  = Tensor::randn({10, 64, 128});  // (10, 64, 128)
auto out    = batch.matmul(weight);           // (10, 64, 32)
```

## Matrix Decompositions

All decompositions support batch dimensions. Pass a tensor of shape `(..., M, N)` and receive batched results.

### SVD

Singular Value Decomposition factors a matrix as `A = U @ diag(S) @ Vh`:

```cpp
using namespace axiom;

auto A = Tensor::randn({6, 4});

// Full decomposition
auto [U, S, Vh] = linalg::svd(A);
// U: (6, 6), S: (4,), Vh: (4, 4)

// Economy (reduced) decomposition
auto [U2, S2, Vh2] = linalg::svd(A, /*full_matrices=*/false);
// U2: (6, 4), S2: (4,), Vh2: (4, 4)

// Singular values only -- faster when you don't need U or Vh
auto sigma = linalg::svdvals(A);  // (4,)
```

### QR

QR decomposition factors a matrix as `A = Q @ R`, where `Q` is orthogonal and `R` is upper triangular:

```cpp
using namespace axiom;

auto A = Tensor::randn({6, 4});
auto [Q, R] = linalg::qr(A);
// Q: (6, 4), R: (4, 4)  -- economy size, K = min(M, N)
```

### Cholesky

Cholesky decomposition factors a symmetric positive definite matrix as `A = L @ L^H`:

```cpp
using namespace axiom;

// Create a symmetric positive definite matrix
auto X = Tensor::randn({4, 4});
auto A = X.matmul(X, /*transpose_self=*/false,
                  /*transpose_other=*/true) +
         Tensor::eye(4) * 0.1f;  // A = X @ X^T + 0.1*I

// Lower triangular (default)
auto L = linalg::cholesky(A);

// Upper triangular: A = U^H @ U
auto U = linalg::cholesky(A, /*upper=*/true);
```

### LU

LU decomposition with partial pivoting factors a matrix as `P @ A = L @ U`:

```cpp
using namespace axiom;

auto A = Tensor::randn({4, 4});
auto [L, U, P, piv] = linalg::lu(A);
// L: lower triangular with unit diagonal
// U: upper triangular
// P: permutation matrix
// piv: pivot indices
```

### Eigendecomposition

Axiom provides four eigenvalue functions, covering both general and symmetric/Hermitian matrices:

```cpp
using namespace axiom;

auto A = Tensor::randn({4, 4});

// General eigendecomposition (eigenvalues may be complex)
auto [vals, vecs] = linalg::eig(A);

// Eigenvalues only (more efficient)
auto eigenvalues = linalg::eigvals(A);

// Symmetric/Hermitian -- returns guaranteed-real eigenvalues
auto S = A + A.T();  // make symmetric
auto [real_vals, real_vecs] = linalg::eigh(S);
auto sym_eigenvalues = linalg::eigvalsh(S);
```

## Linear Systems

### Exact Solve

`linalg::solve` solves the system `A @ x = b` for `x`, where `A` is a square matrix:

```cpp
using namespace axiom;

auto A = Tensor::randn({4, 4});
auto b = Tensor::randn({4});

// Solve for vector b
auto x = linalg::solve(A, b);  // x: (4,)

// Solve for multiple right-hand sides
auto B = Tensor::randn({4, 3});
auto X = linalg::solve(A, B);  // X: (4, 3)
```

### Least-Squares Solve

When the system is overdetermined (more equations than unknowns) or underdetermined, `linalg::lstsq` finds the solution that minimizes `||A @ x - b||_2`:

```cpp
using namespace axiom;

// Overdetermined system: 6 equations, 3 unknowns
auto A = Tensor::randn({6, 3});
auto b = Tensor::randn({6});

auto x = linalg::lstsq(A, b);  // least-squares solution, x: (3,)
```

## Matrix Properties

These functions extract scalar properties from matrices. All support batch dimensions.

```cpp
using namespace axiom;

auto A = Tensor::randn({4, 4});

// Determinant
auto d = linalg::det(A);         // scalar

// Sign and log-determinant (numerically stable for large matrices)
auto [sign, logabsdet] = linalg::slogdet(A);
// det(A) == sign * exp(logabsdet)

// Inverse
auto A_inv = linalg::inv(A);     // (4, 4)

// Pseudoinverse (via SVD, works for non-square matrices)
auto B = Tensor::randn({6, 4});
auto B_pinv = linalg::pinv(B);   // (4, 6)

// Rank
auto r = linalg::matrix_rank(A); // Int64 scalar

// Condition number
auto kappa = linalg::cond(A);    // scalar (2-norm by default)

// Norms
auto fro = linalg::norm(A, "fro");  // Frobenius norm
auto nuc = linalg::norm(A, "nuc");  // nuclear norm
auto l2  = linalg::norm(A, 2);      // spectral norm (largest singular value)

// Trace
auto tr = linalg::trace(A);      // sum of diagonal elements

// Matrix power
auto A3 = linalg::matrix_power(A, 3);    // A @ A @ A
auto A_neg = linalg::matrix_power(A, -1); // equivalent to inv(A)
```

## Vector and Tensor Products

The `linalg` namespace includes a complete set of vector and tensor product operations with NumPy-compatible semantics.

```cpp
using namespace axiom;

auto u = Tensor::randn({5});
auto v = Tensor::randn({5});

// Dot product (1D inner product)
auto d = linalg::dot(u, v);    // scalar

// vdot: conjugates the first argument for complex types
auto vd = linalg::vdot(u, v);  // scalar

// Inner product (generalized to ND)
auto ip = linalg::inner(u, v); // scalar for 1D inputs

// Outer product
auto op = linalg::outer(u, v); // (5, 5)

// Cross product (3D vectors)
auto a = Tensor::randn({3});
auto b = Tensor::randn({3});
auto c = linalg::cross(a, b);  // (3,)

// Kronecker product
auto A = Tensor::randn({2, 3});
auto B = Tensor::randn({4, 5});
auto K = linalg::kron(A, B);   // (8, 15)

// Tensor contraction
auto X = Tensor::randn({3, 4, 5});
auto Y = Tensor::randn({4, 5, 6});
auto Z = linalg::tensordot(X, Y, 2);  // contract last 2 of X with first 2 of Y -> (3, 6)

// Explicit axis contraction
auto Z2 = linalg::tensordot(X, Y, {{1, 2}, {0, 1}});  // same result

// Chained matrix multiplication (optimal ordering)
auto M1 = Tensor::randn({10, 100});
auto M2 = Tensor::randn({100, 5});
auto M3 = Tensor::randn({5, 50});
auto [result, _] = linalg::multi_dot({M1, M2, M3});  // (10, 50)
```

## Batch Operations

Most `linalg` operations support batch dimensions. When you pass a tensor with leading dimensions beyond the expected matrix shape, Axiom loops over the batch automatically:

```cpp
using namespace axiom;

// 20 independent 4x4 matrices
auto A = Tensor::randn({20, 4, 4});

// All of these operate on each matrix in the batch
auto dets   = linalg::det(A);         // (20,)
auto invs   = linalg::inv(A);         // (20, 4, 4)
auto traces = linalg::trace(A);       // (20,)

// Batched SVD
auto [U, S, Vh] = linalg::svd(A);
// U: (20, 4, 4), S: (20, 4), Vh: (20, 4, 4)

// Batched solve: 20 systems, each 4x4
auto b = Tensor::randn({20, 4});
auto x = linalg::solve(A, b);         // (20, 4)

// Nested batches work too
auto big = Tensor::randn({2, 5, 3, 3});
auto big_inv = linalg::inv(big);       // (2, 5, 3, 3)
```

The batch dimensions follow standard broadcasting rules, so you can apply a single operation across different batch shapes:

```cpp
using namespace axiom;

// Single matrix, batch of vectors
auto A = Tensor::randn({4, 4});
auto b = Tensor::randn({10, 4});
auto x = linalg::solve(A, b);  // broadcasts A to each row of b
```

## Complex-Valued Matrices

Axiom supports complex linear algebra through the `Complex64` and `Complex128` dtypes. Most decompositions and solvers work directly with complex tensors:

```cpp
using namespace axiom;

// Create a complex matrix
auto real_part = Tensor::randn({4, 4});
auto imag_part = Tensor::randn({4, 4});
auto Z = Tensor::complex(real_part, imag_part);  // Complex64

// Standard operations work with complex types
auto d = linalg::det(Z);
auto Z_inv = linalg::inv(Z);
auto x = linalg::solve(Z, Tensor::complex(
    Tensor::randn({4}), Tensor::randn({4})));
auto [U, S, Vh] = linalg::svd(Z);
```

For Hermitian matrices (complex matrices equal to their conjugate transpose), use the dedicated `eigh` and `eigvalsh` functions. These exploit the Hermitian structure for better performance and guarantee real eigenvalues:

```cpp
using namespace axiom;

// Build a Hermitian matrix: H = Z @ Z^H
auto Z = Tensor::complex(
    Tensor::randn({4, 4}), Tensor::randn({4, 4}));
auto H = Z.matmul(Z.conj().T()) +
         Tensor::eye(4).to(DType::Complex64) * 0.1f;

// Hermitian eigendecomposition -- eigenvalues are always real
auto [vals, vecs] = linalg::eigh(H);
// vals: Float32 (4,), vecs: Complex64 (4, 4)

auto eigenvalues_only = linalg::eigvalsh(H);  // Float32 (4,)
```

The `vdot` function is particularly useful with complex data -- it conjugates the first argument before computing the dot product, matching the standard physics convention for inner products in complex vector spaces.

For complete function signatures, see [API Reference: Linear Algebra](../api/linalg).
