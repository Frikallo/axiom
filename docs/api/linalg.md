# Linear Algebra

*For a tutorial introduction, see [User Guide: Linear Algebra](../user-guide/linear-algebra).*

Full LAPACK-backed linear algebra in the `axiom::linalg` namespace. All operations support batch dimensions `(..., M, N)`.

**Backends:** Apple Accelerate (macOS), OpenBLAS (Linux/Windows).

GPU tensors automatically fall back to CPU for LAPACK operations.

## Core Operations

### linalg::det

```cpp
Tensor linalg::det(const Tensor &a);
Tensor Tensor::det() const;  // member function
```

Matrix determinant. Input: `(..., N, N)`. Returns: `(...)`.

---

### linalg::inv

```cpp
Tensor linalg::inv(const Tensor &a);
Tensor Tensor::inv() const;  // member function
```

Matrix inverse. Input: `(..., N, N)`. Returns: `(..., N, N)`.

---

### linalg::solve

```cpp
Tensor linalg::solve(const Tensor &a, const Tensor &b);
```

Solve `A @ X = B` for X.

**Parameters:**
- `a` -- `(..., N, N)` square coefficient matrix.
- `b` -- `(..., N)` or `(..., N, K)` right-hand side.

**Returns:** Solution X with same shape as `b`.

---

## Decompositions

### linalg::svd

```cpp
SVDResult linalg::svd(const Tensor &a, bool full_matrices = true);
```

Singular Value Decomposition: `A = U @ diag(S) @ Vh`.

**Parameters:**
- `a` -- `(..., M, N)` input matrix.
- `full_matrices` (*bool*) -- If true, U is (M, M) and Vh is (N, N). Default: `true`.

**Returns:** `SVDResult` with fields `U`, `S`, `Vh`.

```cpp
auto [U, S, Vh] = linalg::svd(A);
```

---

### linalg::svdvals

```cpp
Tensor linalg::svdvals(const Tensor &x);
```

Singular values only (more efficient than full SVD). Returns 1D tensor of singular values.

---

### linalg::qr

```cpp
QRResult linalg::qr(const Tensor &a);
```

QR decomposition: `A = Q @ R`.

**Returns:** `QRResult` with fields `Q` (M, K) and `R` (K, N) where K = min(M, N).

---

### linalg::cholesky

```cpp
Tensor linalg::cholesky(const Tensor &a, bool upper = false);
```

Cholesky decomposition for positive definite matrices: `A = L @ L^H`.

**Parameters:**
- `upper` (*bool*) -- If true, returns upper triangular U where `A = U^H @ U`. Default: `false`.

---

### linalg::lu

```cpp
LUResult linalg::lu(const Tensor &a);
```

LU decomposition with partial pivoting: `P @ A = L @ U`.

**Returns:** `LUResult` with fields `L`, `U`, `P` (permutation matrix), and `piv` (pivot indices).

---

### linalg::eig

```cpp
EigResult linalg::eig(const Tensor &a);
```

General eigenvalue decomposition. Eigenvalues may be complex even for real input.

**Returns:** `EigResult` with `eigenvalues` and `eigenvectors`.

---

### linalg::eigh

```cpp
EigResult linalg::eigh(const Tensor &a);
```

Symmetric/Hermitian eigenvalue decomposition. Returns real eigenvalues.

---

### linalg::eigvals

```cpp
Tensor linalg::eigvals(const Tensor &a);
```

Eigenvalues only (more efficient than full eigendecomposition).

---

### linalg::eigvalsh

```cpp
Tensor linalg::eigvalsh(const Tensor &a);
```

Eigenvalues of symmetric/Hermitian matrix only.

---

## Advanced Operations

### linalg::lstsq

```cpp
Tensor linalg::lstsq(const Tensor &a, const Tensor &b, double rcond = -1.0);
```

Least squares solution: minimizes `||A @ X - B||_2`.

---

### linalg::pinv

```cpp
Tensor linalg::pinv(const Tensor &a, double rcond = 1e-15);
```

Moore-Penrose pseudoinverse via SVD. Input: `(..., M, N)`. Returns: `(..., N, M)`.

---

### linalg::norm

```cpp
Tensor linalg::norm(const Tensor &a, const std::string &ord = "fro");
Tensor linalg::norm(const Tensor &a, int ord);
Tensor linalg::norm(const Tensor &a, double ord);
```

Matrix or vector norm.

**String norms:** `"fro"` (Frobenius), `"nuc"` (nuclear).

**Integer norms:** `1` (L1 / max column sum), `2` (L2 / spectral), `0` (L0, vector only).

**Float norms (vector only):** `INFINITY` (L-infinity), `-INFINITY` (min absolute value).

---

### linalg::matrix_rank

```cpp
Tensor linalg::matrix_rank(const Tensor &a, double tol = -1.0);
```

Matrix rank via SVD. Returns Int64 tensor.

---

### linalg::cond

```cpp
Tensor linalg::cond(const Tensor &a, int p = 2);
```

Condition number (ratio of largest to smallest singular value).

---

### linalg::matrix_power

```cpp
Tensor linalg::matrix_power(const Tensor &a, int n);
```

Matrix power A^n. Negative `n` requires an invertible matrix.

---

### linalg::slogdet

```cpp
std::pair<Tensor, Tensor> linalg::slogdet(const Tensor &a);
```

Sign and log of determinant. More numerically stable for large matrices. Returns `(sign, logabsdet)` where `det(a) = sign * exp(logabsdet)`.

---

## Vector & Matrix Products

### linalg::dot

```cpp
Tensor linalg::dot(const Tensor &a, const Tensor &b);
```

Dot product with NumPy semantics: 1D inner product, 2D matrix multiply, ND sum over last/second-to-last axes.

---

### linalg::vdot

```cpp
Tensor linalg::vdot(const Tensor &a, const Tensor &b);
```

Vector dot product. Conjugates the first argument for complex types.

---

### linalg::inner

```cpp
Tensor linalg::inner(const Tensor &a, const Tensor &b);
```

Inner product. For 1D: dot product. For ND: sum product over last axes.

---

### linalg::outer

```cpp
Tensor linalg::outer(const Tensor &a, const Tensor &b);
```

Outer product of two vectors. `(M,) x (N,) -> (M, N)`.

---

### linalg::matvec

```cpp
Tensor linalg::matvec(const Tensor &x1, const Tensor &x2);
```

Matrix-vector product. Uses BLAS sgemv/dgemv. `(..., M, N) x (..., N) -> (..., M)`.

---

### linalg::vecmat

```cpp
Tensor linalg::vecmat(const Tensor &x1, const Tensor &x2);
```

Vector-matrix product. `(..., M) x (..., M, N) -> (..., N)`.

---

### linalg::vecdot

```cpp
Tensor linalg::vecdot(const Tensor &x1, const Tensor &x2, int axis = -1);
```

Vector dot product along a specified axis.

---

### linalg::tensordot

```cpp
Tensor linalg::tensordot(const Tensor &a, const Tensor &b, int axes = 2);
Tensor linalg::tensordot(const Tensor &a, const Tensor &b,
                         const std::pair<std::vector<int>, std::vector<int>> &axes);
```

Tensor contraction. `axes=N` contracts last N axes of a with first N axes of b.

---

### linalg::kron

```cpp
Tensor linalg::kron(const Tensor &a, const Tensor &b);
```

Kronecker product.

---

### linalg::cross

```cpp
Tensor linalg::cross(const Tensor &x1, const Tensor &x2, int axis = -1);
```

Cross product of 3-element vectors.

---

## Convenience Functions

### linalg::trace

```cpp
Tensor linalg::trace(const Tensor &a, int offset = 0);
```

Sum of diagonal elements.

---

### linalg::multi_dot

```cpp
std::pair<Tensor, Tensor> linalg::multi_dot(const std::vector<Tensor> &tensors);
```

Chained matrix multiplication: `a[0] @ a[1] @ ... @ a[n-1]`.

---

## Utility Functions

### linalg::has_lapack

```cpp
bool linalg::has_lapack();
```

Returns `true` if LAPACK backend is available.

---

### linalg::lapack_backend_name

```cpp
const char *linalg::lapack_backend_name();
```

Returns backend name: `"Accelerate"`, `"OpenBLAS"`, or `"None"`.

---

## Complex Number Support

- **Full support:** `det`, `inv`, `solve`, `svd`, `pinv`, `norm`, `matrix_rank`, `cond`
- **Real only:** `qr`, `cholesky`, `eig`, `eigh`

## Batch Support

All operations support batch dimensions:

```cpp
auto A = Tensor::randn({10, 4, 4});  // Batch of 10 matrices
auto dets = linalg::det(A);          // Shape: (10,)
auto invs = linalg::inv(A);          // Shape: (10, 4, 4)
```

| CPU | GPU |
|-----|-----|
| ✓   | (CPU fallback) |

**See Also:** [Arithmetic](arithmetic), [Tensor Class](tensor-class)
