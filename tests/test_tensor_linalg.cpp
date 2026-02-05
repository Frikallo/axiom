#include <axiom/axiom.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Test harness
static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test_func) run_test([&]() { test_func(); }, #test_func)

void run_test(const std::function<void()> &test_func,
              const std::string &test_name) {
    tests_run++;
    std::cout << "--- Running: " << test_name << " ---" << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << " ---" << std::endl;
        tests_passed++;
    } catch (const std::exception &e) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

#define ASSERT(condition, msg)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + \
                                     std::string(msg));                        \
        }                                                                      \
    } while (0)

// ============================================================================
// Test: det (determinant)
// ============================================================================

void test_det_2x2() {
    // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto det = axiom::linalg::det(a);

    ASSERT(det.ndim() == 0, "det should be scalar");
    float det_val = det.item<float>();
    ASSERT(std::abs(det_val - (-2.0f)) < 1e-5f, "det([[1,2],[3,4]]) = -2");
}

void test_det_3x3() {
    // det([[1, 2, 3], [4, 5, 6], [7, 8, 10]]) = 1*(50-48) - 2*(40-42) +
    // 3*(32-35) = 2 + 4 - 9 = -3
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f};
    auto a = axiom::Tensor::from_data(data, {3, 3});
    auto det = axiom::linalg::det(a);

    float det_val = det.item<float>();
    ASSERT(std::abs(det_val - (-3.0f)) < 1e-4f, "3x3 determinant");
}

void test_det_identity() {
    // det(I) = 1
    auto eye = axiom::Tensor::eye(4);
    auto det = axiom::linalg::det(eye);

    float det_val = det.item<float>();
    ASSERT(std::abs(det_val - 1.0f) < 1e-5f, "det(I) = 1");
}

void test_det_singular() {
    // Singular matrix has det = 0
    float data[] = {1.0f, 2.0f, 2.0f, 4.0f}; // Rows are proportional
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto det = axiom::linalg::det(a);

    float det_val = det.item<float>();
    ASSERT(std::abs(det_val) < 1e-5f, "det of singular matrix = 0");
}

void test_det_member_function() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto det = a.det();

    float det_val = det.item<float>();
    ASSERT(std::abs(det_val - (-2.0f)) < 1e-5f, "member function det()");
}

// ============================================================================
// Test: inv (matrix inverse)
// ============================================================================

void test_inv_2x2() {
    // inv([[1, 2], [3, 4]]) = 1/(-2) * [[4, -2], [-3, 1]]
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto a_inv = axiom::linalg::inv(a);

    ASSERT(a_inv.shape() == axiom::Shape({2, 2}), "inv shape matches");

    // Check A @ inv(A) = I
    auto product = a.matmul(a_inv);
    auto eye = axiom::Tensor::eye(2);

    // Use both rtol and atol for numerical stability
    ASSERT(product.allclose(eye, 1e-4, 1e-6), "A @ inv(A) = I");
}

void test_inv_identity() {
    // inv(I) = I
    auto eye = axiom::Tensor::eye(4);
    auto eye_inv = axiom::linalg::inv(eye);

    ASSERT(eye_inv.allclose(eye, 1e-5f), "inv(I) = I");
}

void test_inv_double() {
    double data[] = {4.0, 7.0, 2.0, 6.0};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto a_inv = axiom::linalg::inv(a);

    auto product = a.matmul(a_inv);
    auto eye = axiom::Tensor::eye(2, axiom::DType::Float64);

    ASSERT(product.allclose(eye, 1e-10), "double precision inv");
}

void test_inv_member_function() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto a_inv = a.inv();

    auto product = a.matmul(a_inv);
    auto eye = axiom::Tensor::eye(2);

    ASSERT(product.allclose(eye, 1e-4, 1e-6), "member function inv()");
}

// ============================================================================
// Test: solve (linear system)
// ============================================================================

void test_solve_simple() {
    // Solve A @ x = b where A = [[1, 2], [3, 4]], b = [5, 11]
    // Expected: x = [1, 2] because 1*1 + 2*2 = 5, 3*1 + 4*2 = 11
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 11.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2});

    auto x = axiom::linalg::solve(a, b);

    ASSERT(x.shape() == axiom::Shape({2}), "solve shape matches b");

    // Verify A @ x = b
    auto result = a.matmul(x);
    ASSERT(result.allclose(b, 1e-4f), "A @ x = b");
}

void test_solve_multiple_rhs() {
    // Solve A @ X = B where B has multiple columns
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 11.0f, 14.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto x = axiom::linalg::solve(a, b);

    ASSERT(x.shape() == axiom::Shape({2, 2}), "solve shape matches B");

    auto result = a.matmul(x);
    ASSERT(result.allclose(b, 1e-4f), "A @ X = B for multiple RHS");
}

// ============================================================================
// Test: svd (singular value decomposition)
// ============================================================================

void test_svd_reconstruction() {
    // Create a random matrix and verify U @ diag(S) @ Vh = A
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [U, S, Vh] = axiom::linalg::svd(a, false);

    ASSERT(S.ndim() == 1, "S is 1D");
    ASSERT(S.shape()[0] == 2, "S has min(m,n) = 2 values");

    // Reconstruct: A = U @ diag(S) @ Vh
    // U is (2, 2), S is (2,), Vh is (2, 3)
    // Create diagonal matrix from S
    auto S_diag = axiom::Tensor::zeros({2, 2});
    S_diag.set_item<float>({0, 0}, S.item<float>({0}));
    S_diag.set_item<float>({1, 1}, S.item<float>({1}));

    auto reconstructed = U.matmul(S_diag).matmul(Vh);
    ASSERT(reconstructed.allclose(a, 1e-4f), "SVD reconstruction");
}

void test_svd_full_matrices() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [U, S, Vh] = axiom::linalg::svd(a, true);

    ASSERT(U.shape() == axiom::Shape({2, 2}), "Full U is (m, m)");
    ASSERT(Vh.shape() == axiom::Shape({3, 3}), "Full Vh is (n, n)");
}

void test_svd_economy() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [U, S, Vh] = axiom::linalg::svd(a, false);

    ASSERT(U.shape() == axiom::Shape({2, 2}), "Economy U is (m, k)");
    ASSERT(Vh.shape() == axiom::Shape({2, 3}), "Economy Vh is (k, n)");
}

// ============================================================================
// Test: qr (QR decomposition)
// ============================================================================

void test_qr_reconstruction() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [Q, R] = axiom::linalg::qr(a);

    // Q @ R should equal A
    auto reconstructed = Q.matmul(R);
    ASSERT(reconstructed.allclose(a, 1e-4f), "QR reconstruction");
}

void test_qr_orthogonal() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto a = axiom::Tensor::from_data(data, {3, 3});

    auto [Q, R] = axiom::linalg::qr(a);

    // Q should be orthogonal: Q^T @ Q = I
    auto QtQ = Q.transpose().matmul(Q);
    auto eye = axiom::Tensor::eye(3);
    ASSERT(QtQ.allclose(eye, 1e-4, 1e-6), "Q is orthogonal");
}

// ============================================================================
// Test: cholesky
// ============================================================================

void test_cholesky_positive_definite() {
    // Create a positive definite matrix: A = B @ B^T + I
    float data[] = {4.0f, 2.0f, 2.0f, 5.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto L = axiom::linalg::cholesky(a, false);

    // L @ L^T should equal A
    auto reconstructed = L.matmul(L.transpose());
    ASSERT(reconstructed.allclose(a, 1e-4f), "Cholesky reconstruction");
}

void test_cholesky_lower() {
    float data[] = {4.0f, 2.0f, 2.0f, 5.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto L = axiom::linalg::cholesky(a, false);

    // L should be lower triangular
    float upper_val = L.item<float>({0, 1});
    ASSERT(std::abs(upper_val) < 1e-6f, "Cholesky L is lower triangular");
}

// ============================================================================
// Test: lu (LU decomposition)
// ============================================================================

void test_lu_reconstruction() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [L, U, P, piv] = axiom::linalg::lu(a);

    // P @ A = L @ U (note: our P is defined this way)
    auto LU = L.matmul(U);
    auto PA = P.matmul(a);
    ASSERT(LU.allclose(PA, 1e-4f), "LU reconstruction with permutation");
}

// ============================================================================
// Test: eigh (symmetric eigenvalue decomposition)
// ============================================================================

void test_eigh_symmetric() {
    // Symmetric matrix
    float data[] = {2.0f, 1.0f, 1.0f, 2.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [eigenvalues, eigenvectors] = axiom::linalg::eigh(a);

    ASSERT(eigenvalues.ndim() == 1, "eigenvalues is 1D");
    ASSERT(eigenvalues.shape()[0] == 2, "2 eigenvalues");

    // For symmetric matrix, eigenvalues are real
    // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    float e0 = eigenvalues.item<float>({0});
    float e1 = eigenvalues.item<float>({1});

    // Sort eigenvalues
    if (e0 > e1)
        std::swap(e0, e1);

    ASSERT(std::abs(e0 - 1.0f) < 1e-4f, "First eigenvalue is 1");
    ASSERT(std::abs(e1 - 3.0f) < 1e-4f, "Second eigenvalue is 3");
}

void test_eigh_reconstruction() {
    float data[] = {2.0f, 1.0f, 1.0f, 2.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [eigenvalues, V] = axiom::linalg::eigh(a);

    // A @ V = V @ diag(eigenvalues)
    auto AV = a.matmul(V);
    auto Lambda = axiom::Tensor::zeros({2, 2}, axiom::DType::Float32);
    Lambda.set_item<float>({0, 0}, eigenvalues.item<float>({0}));
    Lambda.set_item<float>({1, 1}, eigenvalues.item<float>({1}));
    auto V_Lambda = V.matmul(Lambda);

    ASSERT(AV.allclose(V_Lambda, 1e-4f), "A @ V = V @ Lambda");
}

// ============================================================================
// Test: lstsq (least squares)
// ============================================================================

void test_lstsq_overdetermined() {
    // Overdetermined system (more equations than unknowns)
    // Find x that minimizes ||A @ x - b||_2
    float a_data[] = {1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 3.0f};
    float b_data[] = {1.0f, 2.0f, 2.0f};
    auto a = axiom::Tensor::from_data(a_data, {3, 2});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto x = axiom::linalg::lstsq(a, b);

    ASSERT(x.shape() == axiom::Shape({2}), "lstsq result has correct shape");
}

// ============================================================================
// Test: norm
// ============================================================================

void test_norm_frobenius() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto n = axiom::linalg::norm(a, "fro");

    // Frobenius norm = sqrt(1 + 4 + 9 + 16) = sqrt(30)
    float expected = std::sqrt(30.0f);
    ASSERT(std::abs(n.item<float>() - expected) < 1e-4f, "Frobenius norm");
}

void test_norm_vector_2() {
    float data[] = {3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2});

    auto n = axiom::linalg::norm(a, 2);

    // L2 norm = sqrt(9 + 16) = 5
    ASSERT(std::abs(n.item<float>() - 5.0f) < 1e-5f, "L2 norm of vector");
}

// ============================================================================
// Test: matrix_rank
// ============================================================================

void test_matrix_rank_full() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto r = axiom::linalg::matrix_rank(a);

    ASSERT(r.item<int64_t>() == 2, "Full rank 2x2 matrix has rank 2");
}

void test_matrix_rank_deficient() {
    // Rank-deficient matrix (rows are proportional)
    // Use a clearer rank-1 matrix
    double data[] = {1.0, 2.0, 2.0, 4.0};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    // Use a tolerance that handles numerical precision
    auto r = axiom::linalg::matrix_rank(a, 1e-10);

    ASSERT(r.item<int64_t>() == 1, "Rank-deficient 2x2 matrix has rank 1");
}

// ============================================================================
// Test: cond (condition number)
// ============================================================================

void test_cond_identity() {
    auto eye = axiom::Tensor::eye(4);
    auto c = axiom::linalg::cond(eye, 2);

    // Condition number of identity is 1
    ASSERT(std::abs(c.item<float>() - 1.0f) < 1e-4f, "cond(I) = 1");
}

// ============================================================================
// Test: matrix_power
// ============================================================================

void test_matrix_power_square() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto a2 = axiom::linalg::matrix_power(a, 2);

    // A^2 = A @ A
    auto expected = a.matmul(a);
    ASSERT(a2.allclose(expected, 1e-5f), "A^2 = A @ A");
}

void test_matrix_power_negative() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto a_neg1 = axiom::linalg::matrix_power(a, -1);

    // A^-1 = inv(A)
    auto expected = axiom::linalg::inv(a);
    ASSERT(a_neg1.allclose(expected, 1e-4f), "A^-1 = inv(A)");
}

void test_matrix_power_zero() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto a0 = axiom::linalg::matrix_power(a, 0);

    // A^0 = I
    auto eye = axiom::Tensor::eye(2);
    ASSERT(a0.allclose(eye, 1e-5f), "A^0 = I");
}

// ============================================================================
// Test: NumPy Parity - Phase 1: Core Products
// ============================================================================

void test_dot_1d_1d() {
    // 1D @ 1D = scalar (inner product)
    // [1, 2, 3] . [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::dot(a, b);

    ASSERT(result.ndim() == 0, "dot 1D@1D should be scalar");
    ASSERT(std::abs(result.item<float>() - 32.0f) < 1e-5f, "1*4+2*5+3*6=32");
}

void test_dot_2d_2d() {
    // 2D @ 2D = matmul
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::dot(a, b);
    auto expected = a.matmul(b);

    ASSERT(result.shape() == axiom::Shape({2, 2}), "dot 2D@2D shape");
    ASSERT(result.allclose(expected, 1e-5f), "dot 2D@2D equals matmul");
}

void test_dot_1d_2d() {
    // 1D @ 2D: (3,) @ (3, 2) -> (2,)
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3, 2});

    auto result = axiom::linalg::dot(a, b);

    ASSERT(result.shape() == axiom::Shape({2}), "dot 1D@2D shape");
    // [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    ASSERT(std::abs(result.item<float>({0}) - 22.0f) < 1e-5f, "result[0]=22");
    ASSERT(std::abs(result.item<float>({1}) - 28.0f) < 1e-5f, "result[1]=28");
}

void test_dot_2d_1d() {
    // 2D @ 1D: (2, 3) @ (3,) -> (2,)
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::dot(a, b);

    ASSERT(result.shape() == axiom::Shape({2}), "dot 2D@1D shape");
    // [[1,2,3],[4,5,6]] @ [1,2,3] = [1+4+9, 4+10+18] = [14, 32]
    ASSERT(std::abs(result.item<float>({0}) - 14.0f) < 1e-5f, "result[0]=14");
    ASSERT(std::abs(result.item<float>({1}) - 32.0f) < 1e-5f, "result[1]=32");
}

void test_vdot_real() {
    // vdot flattens and computes dot product
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::vdot(a, b);

    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    ASSERT(result.ndim() == 0, "vdot should be scalar");
    ASSERT(std::abs(result.item<float>() - 70.0f) < 1e-5f, "vdot = 70");
}

void test_inner_1d() {
    // inner for 1D is same as dot
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::inner(a, b);
    auto expected = axiom::linalg::dot(a, b);

    ASSERT(result.ndim() == 0, "inner 1D should be scalar");
    ASSERT(result.allclose(expected, 1e-5f), "inner 1D equals dot");
}

void test_inner_2d() {
    // inner for 2D: a @ b.T
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::inner(a, b);
    auto expected = a.matmul(b.transpose());

    ASSERT(result.shape() == axiom::Shape({2, 2}), "inner 2D shape");
    ASSERT(result.allclose(expected, 1e-5f), "inner 2D equals a @ b.T");
}

void test_outer() {
    // outer product: (m,) x (n,) -> (m, n)
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {2});

    auto result = axiom::linalg::outer(a, b);

    ASSERT(result.shape() == axiom::Shape({3, 2}), "outer shape (3,2)");
    // [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]] = [[4,5],[8,10],[12,15]]
    ASSERT(std::abs(result.item<float>({0, 0}) - 4.0f) < 1e-5f, "[0,0]=4");
    ASSERT(std::abs(result.item<float>({1, 1}) - 10.0f) < 1e-5f, "[1,1]=10");
    ASSERT(std::abs(result.item<float>({2, 0}) - 12.0f) < 1e-5f, "[2,0]=12");
}

void test_matvec() {
    // Matrix-vector product: (2,3) @ (3,) -> (2,)
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::matvec(a, b);

    ASSERT(result.shape() == axiom::Shape({2}), "matvec shape");
    // [1+4+9, 4+10+18] = [14, 32]
    ASSERT(std::abs(result.item<float>({0}) - 14.0f) < 1e-5f, "matvec[0]=14");
    ASSERT(std::abs(result.item<float>({1}) - 32.0f) < 1e-5f, "matvec[1]=32");
}

void test_vecmat() {
    // Vector-matrix product: (2,) @ (2,3) -> (3,)
    float a_data[] = {1.0f, 2.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {2});
    auto b = axiom::Tensor::from_data(b_data, {2, 3});

    auto result = axiom::linalg::vecmat(a, b);

    ASSERT(result.shape() == axiom::Shape({3}), "vecmat shape");
    // [1,2] @ [[1,2,3],[4,5,6]] = [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
    ASSERT(std::abs(result.item<float>({0}) - 9.0f) < 1e-5f, "vecmat[0]=9");
    ASSERT(std::abs(result.item<float>({1}) - 12.0f) < 1e-5f, "vecmat[1]=12");
    ASSERT(std::abs(result.item<float>({2}) - 15.0f) < 1e-5f, "vecmat[2]=15");
}

void test_vecdot() {
    // Vector dot product along axis
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {2, 3});

    auto result = axiom::linalg::vecdot(a, b, -1); // contract last axis

    ASSERT(result.shape() == axiom::Shape({2}), "vecdot shape");
    // Row 0: 1*1+2*1+3*1 = 6, Row 1: 4*2+5*2+6*2 = 30
    ASSERT(std::abs(result.item<float>({0}) - 6.0f) < 1e-5f, "vecdot[0]=6");
    ASSERT(std::abs(result.item<float>({1}) - 30.0f) < 1e-5f, "vecdot[1]=30");
}

// ============================================================================
// Test: NumPy Parity - Phase 2: Complex Products
// ============================================================================

void test_tensordot_simple() {
    // tensordot with axes=1: contract last axis of a with first axis of b
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {3, 2});

    auto result = axiom::linalg::tensordot(a, b, 1);

    // This is just matmul when axes=1
    auto expected = a.matmul(b);
    ASSERT(result.shape() == expected.shape(), "tensordot axes=1 shape");
    ASSERT(result.allclose(expected, 1e-5f), "tensordot axes=1 equals matmul");
}

void test_tensordot_axes() {
    // tensordot with explicit axes
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {2, 3});

    // Contract axis 1 of a with axis 1 of b
    auto result =
        axiom::linalg::tensordot(a, b, {{1}, {1}}); // a @ b.T essentially

    ASSERT(result.shape() == axiom::Shape({2, 2}), "tensordot explicit shape");
    auto expected = a.matmul(b.transpose());
    ASSERT(result.allclose(expected, 1e-5f), "tensordot explicit equals a@b.T");
}

void test_kron_2d() {
    // Kronecker product
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {0.0f, 5.0f, 6.0f, 7.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::kron(a, b);

    ASSERT(result.shape() == axiom::Shape({4, 4}), "kron shape (4,4)");
    // Top-left 2x2 block should be 1*b = [[0,5],[6,7]]
    ASSERT(std::abs(result.item<float>({0, 0}) - 0.0f) < 1e-5f, "kron[0,0]=0");
    ASSERT(std::abs(result.item<float>({0, 1}) - 5.0f) < 1e-5f, "kron[0,1]=5");
    ASSERT(std::abs(result.item<float>({1, 0}) - 6.0f) < 1e-5f, "kron[1,0]=6");
    // Top-right 2x2 block should be 2*b = [[0,10],[12,14]]
    ASSERT(std::abs(result.item<float>({0, 2}) - 0.0f) < 1e-5f, "kron[0,2]=0");
    ASSERT(std::abs(result.item<float>({0, 3}) - 10.0f) < 1e-5f,
           "kron[0,3]=10");
}

void test_cross_simple() {
    // Cross product of two 3D vectors
    // [1,0,0] x [0,1,0] = [0,0,1]
    float a_data[] = {1.0f, 0.0f, 0.0f};
    float b_data[] = {0.0f, 1.0f, 0.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::cross(a, b);

    ASSERT(result.shape() == axiom::Shape({3}), "cross shape");
    ASSERT(std::abs(result.item<float>({0}) - 0.0f) < 1e-5f, "cross[0]=0");
    ASSERT(std::abs(result.item<float>({1}) - 0.0f) < 1e-5f, "cross[1]=0");
    ASSERT(std::abs(result.item<float>({2}) - 1.0f) < 1e-5f, "cross[2]=1");
}

// ============================================================================
// Test: NumPy Parity - Phase 3: Decomposition Variants
// ============================================================================

void test_svdvals() {
    // svdvals should return just singular values
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto s = axiom::linalg::svdvals(a);
    auto [U, S_full, Vh] = axiom::linalg::svd(a, false);

    ASSERT(s.shape() == S_full.shape(), "svdvals shape matches svd.S");
    ASSERT(s.allclose(S_full, 1e-5f), "svdvals equals svd.S");
}

void test_eigvals() {
    // eigvals should return just eigenvalues
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto eigs = axiom::linalg::eigvals(a);
    auto [eigenvalues, eigenvectors] = axiom::linalg::eig(a);

    ASSERT(eigs.shape() == eigenvalues.shape(), "eigvals shape matches");
    ASSERT(eigs.allclose(eigenvalues, 1e-5), "eigvals equals eig.eigenvalues");
}

void test_eigvalsh() {
    // eigvalsh for symmetric matrix
    float data[] = {2.0f, 1.0f, 1.0f, 2.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto eigs = axiom::linalg::eigvalsh(a);
    auto [eigenvalues, eigenvectors] = axiom::linalg::eigh(a);

    ASSERT(eigs.shape() == eigenvalues.shape(), "eigvalsh shape matches");
    ASSERT(eigs.allclose(eigenvalues, 1e-5f),
           "eigvalsh equals eigh.eigenvalues");
}

void test_slogdet_positive() {
    // slogdet for matrix with positive determinant
    float data[] = {1.0f, 2.0f, 3.0f, 5.0f}; // det = 1*5 - 2*3 = -1
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [sign, logabsdet] = axiom::linalg::slogdet(a);

    // det = -1, so sign = -1, logabsdet = log(1) = 0
    ASSERT(std::abs(sign.item<float>() - (-1.0f)) < 1e-5f, "slogdet sign = -1");
    ASSERT(std::abs(logabsdet.item<float>() - 0.0f) < 1e-4f,
           "slogdet logdet=0");
}

void test_slogdet_negative() {
    // slogdet consistency check: sign * exp(logabsdet) = det
    float data[] = {4.0f, 7.0f, 2.0f, 6.0f}; // det = 4*6 - 7*2 = 10
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [sign, logabsdet] = axiom::linalg::slogdet(a);
    auto det = axiom::linalg::det(a);

    float reconstructed =
        sign.item<float>() * std::exp(logabsdet.item<float>());
    ASSERT(std::abs(reconstructed - det.item<float>()) < 1e-4f,
           "sign*exp(logdet) = det");
}

// ============================================================================
// Test: has_lapack
// ============================================================================

void test_has_lapack() {
    bool has = axiom::linalg::has_lapack();
    std::cout << "  LAPACK available: " << (has ? "YES" : "NO") << std::endl;
    std::cout << "  LAPACK backend: " << axiom::linalg::lapack_backend_name()
              << std::endl;

#ifdef AXIOM_USE_ACCELERATE
    ASSERT(has, "LAPACK should be available with Accelerate");
#endif
#ifdef AXIOM_USE_OPENBLAS
    ASSERT(has, "LAPACK should be available with OpenBLAS");
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Linear Algebra Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // LAPACK availability check
    RUN_TEST(test_has_lapack);

    // Skip other tests if no LAPACK
    if (!axiom::linalg::has_lapack()) {
        std::cout << "\nSkipping linear algebra tests (LAPACK not available)"
                  << std::endl;
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Suite Summary:" << std::endl;
        std::cout << "    " << tests_passed << " / " << tests_run
                  << " tests passed." << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    }

    // Determinant tests
    RUN_TEST(test_det_2x2);
    RUN_TEST(test_det_3x3);
    RUN_TEST(test_det_identity);
    RUN_TEST(test_det_singular);
    RUN_TEST(test_det_member_function);

    // Inverse tests
    RUN_TEST(test_inv_2x2);
    RUN_TEST(test_inv_identity);
    RUN_TEST(test_inv_double);
    RUN_TEST(test_inv_member_function);

    // Solve tests
    RUN_TEST(test_solve_simple);
    RUN_TEST(test_solve_multiple_rhs);

    // SVD tests
    RUN_TEST(test_svd_reconstruction);
    RUN_TEST(test_svd_full_matrices);
    RUN_TEST(test_svd_economy);

    // QR tests
    RUN_TEST(test_qr_reconstruction);
    RUN_TEST(test_qr_orthogonal);

    // Cholesky tests
    RUN_TEST(test_cholesky_positive_definite);
    RUN_TEST(test_cholesky_lower);

    // LU tests
    RUN_TEST(test_lu_reconstruction);

    // Eigenvalue tests
    RUN_TEST(test_eigh_symmetric);
    RUN_TEST(test_eigh_reconstruction);

    // Least squares tests
    RUN_TEST(test_lstsq_overdetermined);

    // Norm tests
    RUN_TEST(test_norm_frobenius);
    RUN_TEST(test_norm_vector_2);

    // Matrix rank tests
    RUN_TEST(test_matrix_rank_full);
    RUN_TEST(test_matrix_rank_deficient);

    // Condition number tests
    RUN_TEST(test_cond_identity);

    // Matrix power tests
    RUN_TEST(test_matrix_power_square);
    RUN_TEST(test_matrix_power_negative);
    RUN_TEST(test_matrix_power_zero);

    // New NumPy parity tests - Phase 1: Core Products
    RUN_TEST(test_dot_1d_1d);
    RUN_TEST(test_dot_2d_2d);
    RUN_TEST(test_dot_1d_2d);
    RUN_TEST(test_dot_2d_1d);
    RUN_TEST(test_vdot_real);
    RUN_TEST(test_inner_1d);
    RUN_TEST(test_inner_2d);
    RUN_TEST(test_outer);
    RUN_TEST(test_matvec);
    RUN_TEST(test_vecmat);
    RUN_TEST(test_vecdot);

    // Phase 2: Complex Products
    RUN_TEST(test_tensordot_simple);
    RUN_TEST(test_tensordot_axes);
    RUN_TEST(test_kron_2d);
    RUN_TEST(test_cross_simple);

    // Phase 3: Decomposition Variants
    RUN_TEST(test_svdvals);
    RUN_TEST(test_eigvals);
    RUN_TEST(test_eigvalsh);
    RUN_TEST(test_slogdet_positive);
    RUN_TEST(test_slogdet_negative);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
