#include "axiom_test_utils.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// LinalgTest fixture: skips all tests if LAPACK is not available
// ============================================================================

class LinalgTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!axiom::linalg::has_lapack()) {
            GTEST_SKIP() << "LAPACK not available";
        }
    }
};

// ============================================================================
// Test: has_lapack (standalone, no fixture)
// ============================================================================

TEST(TensorLinalg, HasLapack) {
    bool has = axiom::linalg::has_lapack();
    std::cout << "  LAPACK available: " << (has ? "YES" : "NO") << std::endl;
    std::cout << "  LAPACK backend: " << axiom::linalg::lapack_backend_name()
              << std::endl;

#ifdef AXIOM_USE_ACCELERATE
    ASSERT_TRUE(has) << "LAPACK should be available with Accelerate";
#endif
#ifdef AXIOM_USE_OPENBLAS
    ASSERT_TRUE(has) << "LAPACK should be available with OpenBLAS";
#endif
}

// ============================================================================
// Test: det (determinant)
// ============================================================================

TEST_F(LinalgTest, DetTwoByTwo) {
    // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto det = axiom::linalg::det(a);

    ASSERT_TRUE(det.ndim() == 0) << "det should be scalar";
    float det_val = det.item<float>();
    ASSERT_NEAR(det_val, -2.0f, 1e-5f) << "det([[1,2],[3,4]]) = -2";
}

TEST_F(LinalgTest, DetThreeByThree) {
    // det([[1, 2, 3], [4, 5, 6], [7, 8, 10]]) = -3
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f};
    auto a = axiom::Tensor::from_data(data, {3, 3});
    auto det = axiom::linalg::det(a);

    float det_val = det.item<float>();
    ASSERT_NEAR(det_val, -3.0f, 1e-4f) << "3x3 determinant";
}

TEST_F(LinalgTest, DetIdentity) {
    // det(I) = 1
    auto eye = axiom::Tensor::eye(4);
    auto det = axiom::linalg::det(eye);

    float det_val = det.item<float>();
    ASSERT_NEAR(det_val, 1.0f, 1e-5f) << "det(I) = 1";
}

TEST_F(LinalgTest, DetSingular) {
    // Singular matrix has det = 0
    float data[] = {1.0f, 2.0f, 2.0f, 4.0f}; // Rows are proportional
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto det = axiom::linalg::det(a);

    float det_val = det.item<float>();
    ASSERT_NEAR(det_val, 0.0f, 1e-5f) << "det of singular matrix = 0";
}

TEST_F(LinalgTest, DetMemberFunction) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto det = a.det();

    float det_val = det.item<float>();
    ASSERT_NEAR(det_val, -2.0f, 1e-5f) << "member function det()";
}

// ============================================================================
// Test: inv (matrix inverse)
// ============================================================================

TEST_F(LinalgTest, InvTwoByTwo) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto a_inv = axiom::linalg::inv(a);

    ASSERT_TRUE(a_inv.shape() == axiom::Shape({2, 2})) << "inv shape matches";

    // Check A @ inv(A) = I
    auto product = a.matmul(a_inv);
    auto eye = axiom::Tensor::eye(2);

    ASSERT_TRUE(product.allclose(eye, 1e-4, 1e-6)) << "A @ inv(A) = I";
}

TEST_F(LinalgTest, InvIdentity) {
    auto eye = axiom::Tensor::eye(4);
    auto eye_inv = axiom::linalg::inv(eye);

    ASSERT_TRUE(eye_inv.allclose(eye, 1e-5f)) << "inv(I) = I";
}

TEST_F(LinalgTest, InvDouble) {
    double data[] = {4.0, 7.0, 2.0, 6.0};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto a_inv = axiom::linalg::inv(a);

    auto product = a.matmul(a_inv);
    auto eye = axiom::Tensor::eye(2, axiom::DType::Float64);

    ASSERT_TRUE(product.allclose(eye, 1e-10)) << "double precision inv";
}

TEST_F(LinalgTest, InvMemberFunction) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});
    auto a_inv = a.inv();

    auto product = a.matmul(a_inv);
    auto eye = axiom::Tensor::eye(2);

    ASSERT_TRUE(product.allclose(eye, 1e-4, 1e-6)) << "member function inv()";
}

// ============================================================================
// Test: solve (linear system)
// ============================================================================

TEST_F(LinalgTest, SolveSimple) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 11.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2});

    auto x = axiom::linalg::solve(a, b);

    ASSERT_TRUE(x.shape() == axiom::Shape({2})) << "solve shape matches b";

    // Verify A @ x = b
    auto result = a.matmul(x);
    ASSERT_TRUE(result.allclose(b, 1e-4f)) << "A @ x = b";
}

TEST_F(LinalgTest, SolveMultipleRhs) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 11.0f, 14.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto x = axiom::linalg::solve(a, b);

    ASSERT_TRUE(x.shape() == axiom::Shape({2, 2})) << "solve shape matches B";

    auto result = a.matmul(x);
    ASSERT_TRUE(result.allclose(b, 1e-4f)) << "A @ X = B for multiple RHS";
}

// ============================================================================
// Test: svd (singular value decomposition)
// ============================================================================

TEST_F(LinalgTest, SvdReconstruction) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [U, S, Vh] = axiom::linalg::svd(a, false);

    ASSERT_TRUE(S.ndim() == 1) << "S is 1D";
    ASSERT_EQ(S.shape()[0], 2u) << "S has min(m,n) = 2 values";

    // Reconstruct: A = U @ diag(S) @ Vh
    auto S_diag = axiom::Tensor::zeros({2, 2});
    S_diag.set_item<float>({0, 0}, S.item<float>({0}));
    S_diag.set_item<float>({1, 1}, S.item<float>({1}));

    auto reconstructed = U.matmul(S_diag).matmul(Vh);
    ASSERT_TRUE(reconstructed.allclose(a, 1e-4f)) << "SVD reconstruction";
}

TEST_F(LinalgTest, SvdFullMatrices) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [U, S, Vh] = axiom::linalg::svd(a, true);

    ASSERT_TRUE(U.shape() == axiom::Shape({2, 2})) << "Full U is (m, m)";
    ASSERT_TRUE(Vh.shape() == axiom::Shape({3, 3})) << "Full Vh is (n, n)";
}

TEST_F(LinalgTest, SvdEconomy) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [U, S, Vh] = axiom::linalg::svd(a, false);

    ASSERT_TRUE(U.shape() == axiom::Shape({2, 2})) << "Economy U is (m, k)";
    ASSERT_TRUE(Vh.shape() == axiom::Shape({2, 3})) << "Economy Vh is (k, n)";
}

// ============================================================================
// Test: qr (QR decomposition)
// ============================================================================

TEST_F(LinalgTest, QrReconstruction) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto [Q, R] = axiom::linalg::qr(a);

    // Q @ R should equal A
    auto reconstructed = Q.matmul(R);
    ASSERT_TRUE(reconstructed.allclose(a, 1e-4f)) << "QR reconstruction";
}

TEST_F(LinalgTest, QrOrthogonal) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto a = axiom::Tensor::from_data(data, {3, 3});

    auto [Q, R] = axiom::linalg::qr(a);

    // Q should be orthogonal: Q^T @ Q = I
    auto QtQ = Q.transpose().matmul(Q);
    auto eye = axiom::Tensor::eye(3);
    ASSERT_TRUE(QtQ.allclose(eye, 1e-4, 1e-6)) << "Q is orthogonal";
}

// ============================================================================
// Test: cholesky
// ============================================================================

TEST_F(LinalgTest, CholeskyPositiveDefinite) {
    float data[] = {4.0f, 2.0f, 2.0f, 5.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto L = axiom::linalg::cholesky(a, false);

    // L @ L^T should equal A
    auto reconstructed = L.matmul(L.transpose());
    ASSERT_TRUE(reconstructed.allclose(a, 1e-4f)) << "Cholesky reconstruction";
}

TEST_F(LinalgTest, CholeskyLower) {
    float data[] = {4.0f, 2.0f, 2.0f, 5.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto L = axiom::linalg::cholesky(a, false);

    // L should be lower triangular
    float upper_val = L.item<float>({0, 1});
    ASSERT_NEAR(upper_val, 0.0f, 1e-6f) << "Cholesky L is lower triangular";
}

// ============================================================================
// Test: lu (LU decomposition)
// ============================================================================

TEST_F(LinalgTest, LuReconstruction) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [L, U, P, piv] = axiom::linalg::lu(a);

    // P @ A = L @ U
    auto LU = L.matmul(U);
    auto PA = P.matmul(a);
    ASSERT_TRUE(LU.allclose(PA, 1e-4f)) << "LU reconstruction with permutation";
}

// ============================================================================
// Test: eigh (symmetric eigenvalue decomposition)
// ============================================================================

TEST_F(LinalgTest, EighSymmetric) {
    float data[] = {2.0f, 1.0f, 1.0f, 2.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [eigenvalues, eigenvectors] = axiom::linalg::eigh(a);

    ASSERT_TRUE(eigenvalues.ndim() == 1) << "eigenvalues is 1D";
    ASSERT_EQ(eigenvalues.shape()[0], 2u) << "2 eigenvalues";

    // Eigenvalues of [[2,1],[1,2]] are 1 and 3
    float e0 = eigenvalues.item<float>({0});
    float e1 = eigenvalues.item<float>({1});

    if (e0 > e1)
        std::swap(e0, e1);

    ASSERT_NEAR(e0, 1.0f, 1e-4f) << "First eigenvalue is 1";
    ASSERT_NEAR(e1, 3.0f, 1e-4f) << "Second eigenvalue is 3";
}

TEST_F(LinalgTest, EighReconstruction) {
    float data[] = {2.0f, 1.0f, 1.0f, 2.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [eigenvalues, V] = axiom::linalg::eigh(a);

    // A @ V = V @ diag(eigenvalues)
    auto AV = a.matmul(V);
    auto Lambda = axiom::Tensor::zeros({2, 2}, axiom::DType::Float32);
    Lambda.set_item<float>({0, 0}, eigenvalues.item<float>({0}));
    Lambda.set_item<float>({1, 1}, eigenvalues.item<float>({1}));
    auto V_Lambda = V.matmul(Lambda);

    ASSERT_TRUE(AV.allclose(V_Lambda, 1e-4f)) << "A @ V = V @ Lambda";
}

// ============================================================================
// Test: lstsq (least squares)
// ============================================================================

TEST_F(LinalgTest, LstsqOverdetermined) {
    float a_data[] = {1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 3.0f};
    float b_data[] = {1.0f, 2.0f, 2.0f};
    auto a = axiom::Tensor::from_data(a_data, {3, 2});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto x = axiom::linalg::lstsq(a, b);

    ASSERT_TRUE(x.shape() == axiom::Shape({2}))
        << "lstsq result has correct shape";
}

// ============================================================================
// Test: norm
// ============================================================================

TEST_F(LinalgTest, NormFrobenius) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto n = axiom::linalg::norm(a, "fro");

    // Frobenius norm = sqrt(1 + 4 + 9 + 16) = sqrt(30)
    float expected = std::sqrt(30.0f);
    ASSERT_NEAR(n.item<float>(), expected, 1e-4f) << "Frobenius norm";
}

TEST_F(LinalgTest, NormVector2) {
    float data[] = {3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2});

    auto n = axiom::linalg::norm(a, 2);

    // L2 norm = sqrt(9 + 16) = 5
    ASSERT_NEAR(n.item<float>(), 5.0f, 1e-5f) << "L2 norm of vector";
}

// ============================================================================
// Test: matrix_rank
// ============================================================================

TEST_F(LinalgTest, MatrixRankFull) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto r = axiom::linalg::matrix_rank(a);

    ASSERT_EQ(r.item<int64_t>(), 2) << "Full rank 2x2 matrix has rank 2";
}

TEST_F(LinalgTest, MatrixRankDeficient) {
    double data[] = {1.0, 2.0, 2.0, 4.0};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto r = axiom::linalg::matrix_rank(a, 1e-10);

    ASSERT_EQ(r.item<int64_t>(), 1) << "Rank-deficient 2x2 matrix has rank 1";
}

// ============================================================================
// Test: cond (condition number)
// ============================================================================

TEST_F(LinalgTest, CondIdentity) {
    auto eye = axiom::Tensor::eye(4);
    auto c = axiom::linalg::cond(eye, 2);

    ASSERT_NEAR(c.item<float>(), 1.0f, 1e-4f) << "cond(I) = 1";
}

// ============================================================================
// Test: matrix_power
// ============================================================================

TEST_F(LinalgTest, MatrixPowerSquare) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto a2 = axiom::linalg::matrix_power(a, 2);

    // A^2 = A @ A
    auto expected = a.matmul(a);
    ASSERT_TRUE(a2.allclose(expected, 1e-5f)) << "A^2 = A @ A";
}

TEST_F(LinalgTest, MatrixPowerNegative) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto a_neg1 = axiom::linalg::matrix_power(a, -1);

    // A^-1 = inv(A)
    auto expected = axiom::linalg::inv(a);
    ASSERT_TRUE(a_neg1.allclose(expected, 1e-4f)) << "A^-1 = inv(A)";
}

TEST_F(LinalgTest, MatrixPowerZero) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto a0 = axiom::linalg::matrix_power(a, 0);

    // A^0 = I
    auto eye = axiom::Tensor::eye(2);
    ASSERT_TRUE(a0.allclose(eye, 1e-5f)) << "A^0 = I";
}

// ============================================================================
// Test: Core Products
// ============================================================================

TEST_F(LinalgTest, Dot1d1d) {
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::dot(a, b);

    ASSERT_TRUE(result.ndim() == 0) << "dot 1D@1D should be scalar";
    ASSERT_NEAR(result.item<float>(), 32.0f, 1e-5f) << "1*4+2*5+3*6=32";
}

TEST_F(LinalgTest, Dot2d2d) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::dot(a, b);
    auto expected = a.matmul(b);

    ASSERT_TRUE(result.shape() == axiom::Shape({2, 2})) << "dot 2D@2D shape";
    ASSERT_TRUE(result.allclose(expected, 1e-5f)) << "dot 2D@2D equals matmul";
}

TEST_F(LinalgTest, Dot1d2d) {
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3, 2});

    auto result = axiom::linalg::dot(a, b);

    ASSERT_TRUE(result.shape() == axiom::Shape({2})) << "dot 1D@2D shape";
    ASSERT_NEAR(result.item<float>({0}), 22.0f, 1e-5f) << "result[0]=22";
    ASSERT_NEAR(result.item<float>({1}), 28.0f, 1e-5f) << "result[1]=28";
}

TEST_F(LinalgTest, Dot2d1d) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::dot(a, b);

    ASSERT_TRUE(result.shape() == axiom::Shape({2})) << "dot 2D@1D shape";
    ASSERT_NEAR(result.item<float>({0}), 14.0f, 1e-5f) << "result[0]=14";
    ASSERT_NEAR(result.item<float>({1}), 32.0f, 1e-5f) << "result[1]=32";
}

TEST_F(LinalgTest, VdotReal) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::vdot(a, b);

    ASSERT_TRUE(result.ndim() == 0) << "vdot should be scalar";
    ASSERT_NEAR(result.item<float>(), 70.0f, 1e-5f) << "vdot = 70";
}

TEST_F(LinalgTest, Inner1d) {
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::inner(a, b);
    auto expected = axiom::linalg::dot(a, b);

    ASSERT_TRUE(result.ndim() == 0) << "inner 1D should be scalar";
    ASSERT_TRUE(result.allclose(expected, 1e-5f)) << "inner 1D equals dot";
}

TEST_F(LinalgTest, Inner2d) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::inner(a, b);
    auto expected = a.matmul(b.transpose());

    ASSERT_TRUE(result.shape() == axiom::Shape({2, 2})) << "inner 2D shape";
    ASSERT_TRUE(result.allclose(expected, 1e-5f)) << "inner 2D equals a @ b.T";
}

TEST_F(LinalgTest, Outer) {
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {2});

    auto result = axiom::linalg::outer(a, b);

    ASSERT_TRUE(result.shape() == axiom::Shape({3, 2})) << "outer shape (3,2)";
    ASSERT_NEAR(result.item<float>({0, 0}), 4.0f, 1e-5f) << "[0,0]=4";
    ASSERT_NEAR(result.item<float>({1, 1}), 10.0f, 1e-5f) << "[1,1]=10";
    ASSERT_NEAR(result.item<float>({2, 0}), 12.0f, 1e-5f) << "[2,0]=12";
}

TEST_F(LinalgTest, Matvec) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::matvec(a, b);

    ASSERT_TRUE(result.shape() == axiom::Shape({2})) << "matvec shape";
    ASSERT_NEAR(result.item<float>({0}), 14.0f, 1e-5f) << "matvec[0]=14";
    ASSERT_NEAR(result.item<float>({1}), 32.0f, 1e-5f) << "matvec[1]=32";
}

TEST_F(LinalgTest, Vecmat) {
    float a_data[] = {1.0f, 2.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {2});
    auto b = axiom::Tensor::from_data(b_data, {2, 3});

    auto result = axiom::linalg::vecmat(a, b);

    ASSERT_TRUE(result.shape() == axiom::Shape({3})) << "vecmat shape";
    ASSERT_NEAR(result.item<float>({0}), 9.0f, 1e-5f) << "vecmat[0]=9";
    ASSERT_NEAR(result.item<float>({1}), 12.0f, 1e-5f) << "vecmat[1]=12";
    ASSERT_NEAR(result.item<float>({2}), 15.0f, 1e-5f) << "vecmat[2]=15";
}

TEST_F(LinalgTest, Vecdot) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {2, 3});

    auto result = axiom::linalg::vecdot(a, b, -1); // contract last axis

    ASSERT_TRUE(result.shape() == axiom::Shape({2})) << "vecdot shape";
    ASSERT_NEAR(result.item<float>({0}), 6.0f, 1e-5f) << "vecdot[0]=6";
    ASSERT_NEAR(result.item<float>({1}), 30.0f, 1e-5f) << "vecdot[1]=30";
}

// ============================================================================
// Test: Complex Products
// ============================================================================

TEST_F(LinalgTest, TensordotSimple) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {3, 2});

    auto result = axiom::linalg::tensordot(a, b, 1);

    auto expected = a.matmul(b);
    ASSERT_TRUE(result.shape() == expected.shape()) << "tensordot axes=1 shape";
    ASSERT_TRUE(result.allclose(expected, 1e-5f))
        << "tensordot axes=1 equals matmul";
}

TEST_F(LinalgTest, TensordotAxes) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 3});
    auto b = axiom::Tensor::from_data(b_data, {2, 3});

    auto result = axiom::linalg::tensordot(a, b, {{1}, {1}});

    ASSERT_TRUE(result.shape() == axiom::Shape({2, 2}))
        << "tensordot explicit shape";
    auto expected = a.matmul(b.transpose());
    ASSERT_TRUE(result.allclose(expected, 1e-5f))
        << "tensordot explicit equals a@b.T";
}

TEST_F(LinalgTest, Kron2d) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {0.0f, 5.0f, 6.0f, 7.0f};
    auto a = axiom::Tensor::from_data(a_data, {2, 2});
    auto b = axiom::Tensor::from_data(b_data, {2, 2});

    auto result = axiom::linalg::kron(a, b);

    ASSERT_TRUE(result.shape() == axiom::Shape({4, 4})) << "kron shape (4,4)";
    ASSERT_NEAR(result.item<float>({0, 0}), 0.0f, 1e-5f) << "kron[0,0]=0";
    ASSERT_NEAR(result.item<float>({0, 1}), 5.0f, 1e-5f) << "kron[0,1]=5";
    ASSERT_NEAR(result.item<float>({1, 0}), 6.0f, 1e-5f) << "kron[1,0]=6";
    ASSERT_NEAR(result.item<float>({0, 2}), 0.0f, 1e-5f) << "kron[0,2]=0";
    ASSERT_NEAR(result.item<float>({0, 3}), 10.0f, 1e-5f) << "kron[0,3]=10";
}

TEST_F(LinalgTest, CrossSimple) {
    float a_data[] = {1.0f, 0.0f, 0.0f};
    float b_data[] = {0.0f, 1.0f, 0.0f};
    auto a = axiom::Tensor::from_data(a_data, {3});
    auto b = axiom::Tensor::from_data(b_data, {3});

    auto result = axiom::linalg::cross(a, b);

    ASSERT_TRUE(result.shape() == axiom::Shape({3})) << "cross shape";
    ASSERT_NEAR(result.item<float>({0}), 0.0f, 1e-5f) << "cross[0]=0";
    ASSERT_NEAR(result.item<float>({1}), 0.0f, 1e-5f) << "cross[1]=0";
    ASSERT_NEAR(result.item<float>({2}), 1.0f, 1e-5f) << "cross[2]=1";
}

// ============================================================================
// Test: Decomposition Variants
// ============================================================================

TEST_F(LinalgTest, Svdvals) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto a = axiom::Tensor::from_data(data, {2, 3});

    auto s = axiom::linalg::svdvals(a);
    auto [U, S_full, Vh] = axiom::linalg::svd(a, false);

    ASSERT_TRUE(s.shape() == S_full.shape()) << "svdvals shape matches svd.S";
    ASSERT_TRUE(s.allclose(S_full, 1e-5f)) << "svdvals equals svd.S";
}

TEST_F(LinalgTest, Eigvals) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto eigs = axiom::linalg::eigvals(a);
    auto [eigenvalues, eigenvectors] = axiom::linalg::eig(a);

    ASSERT_TRUE(eigs.shape() == eigenvalues.shape()) << "eigvals shape matches";
    ASSERT_TRUE(eigs.allclose(eigenvalues, 1e-5))
        << "eigvals equals eig.eigenvalues";
}

TEST_F(LinalgTest, Eigvalsh) {
    float data[] = {2.0f, 1.0f, 1.0f, 2.0f};
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto eigs = axiom::linalg::eigvalsh(a);
    auto [eigenvalues, eigenvectors] = axiom::linalg::eigh(a);

    ASSERT_TRUE(eigs.shape() == eigenvalues.shape())
        << "eigvalsh shape matches";
    ASSERT_TRUE(eigs.allclose(eigenvalues, 1e-5f))
        << "eigvalsh equals eigh.eigenvalues";
}

TEST_F(LinalgTest, SlogdetPositive) {
    float data[] = {1.0f, 2.0f, 3.0f, 5.0f}; // det = 1*5 - 2*3 = -1
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [sign, logabsdet] = axiom::linalg::slogdet(a);

    ASSERT_NEAR(sign.item<float>(), -1.0f, 1e-5f) << "slogdet sign = -1";
    ASSERT_NEAR(logabsdet.item<float>(), 0.0f, 1e-4f) << "slogdet logdet=0";
}

TEST_F(LinalgTest, SlogdetNegative) {
    float data[] = {4.0f, 7.0f, 2.0f, 6.0f}; // det = 4*6 - 7*2 = 10
    auto a = axiom::Tensor::from_data(data, {2, 2});

    auto [sign, logabsdet] = axiom::linalg::slogdet(a);
    auto det = axiom::linalg::det(a);

    float reconstructed =
        sign.item<float>() * std::exp(logabsdet.item<float>());
    ASSERT_NEAR(reconstructed, det.item<float>(), 1e-4f)
        << "sign*exp(logdet) = det";
}
