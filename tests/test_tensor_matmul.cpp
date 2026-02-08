#include "axiom_test_utils.hpp"

// Test 2D x 2D matmul
TEST(TensorMatmul, Matmul2D2D) {
    // (2, 3) @ (3, 4) -> (2, 4)
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 4})) << "Shape mismatch";
    // Expected: [[20, 23, 26, 29], [56, 68, 80, 92]]
    axiom::testing::ExpectTensorEquals<float>(
        c, {20.0f, 23.0f, 26.0f, 29.0f, 56.0f, 68.0f, 80.0f, 92.0f}, 1e-5);
}

// Test 1D (vector) x 1D (vector) -> dot product
TEST(TensorMatmul, Matmul1D1D) {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.ndim() == 0) << "Should be scalar";
    ASSERT_TRUE(c.item<float>({}) == 14.0f) << "Dot product incorrect";
}

// Test 1D x 2D -> vector-matrix multiply
TEST(TensorMatmul, Matmul1D2D) {
    // (3,) @ (3, 4) -> (4,)
    auto a = axiom::Tensor::arange(3).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.shape() == axiom::Shape({4})) << "Shape mismatch";
    // Expected: [20, 23, 26, 29]
    axiom::testing::ExpectTensorEquals<float>(c, {20.0f, 23.0f, 26.0f, 29.0f},
                                              1e-5);
}

// Test 2D x 1D -> matrix-vector multiply
TEST(TensorMatmul, Matmul2D1D) {
    // (2, 3) @ (3,) -> (2,)
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(3).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2})) << "Shape mismatch";
    // Expected: [5, 14]
    axiom::testing::ExpectTensorEquals<float>(c, {5.0f, 14.0f}, 1e-5);
}

// Test batched matmul with same batch dims
TEST(TensorMatmul, MatmulBatched) {
    // (2, 3, 4) @ (2, 4, 5) -> (2, 3, 5)
    auto a = axiom::Tensor::arange(24).reshape({2, 3, 4}).astype(
        axiom::DType::Float32);
    auto b = axiom::Tensor::arange(40).reshape({2, 4, 5}).astype(
        axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 3, 5})) << "Shape mismatch";
    ASSERT_TRUE(c.size() == 30) << "Size mismatch";
}

// Test batched matmul with broadcast
TEST(TensorMatmul, MatmulBroadcast) {
    // (2, 1, 3, 4) @ (4, 5) -> (2, 1, 3, 5)
    auto a = axiom::Tensor::arange(24)
                 .reshape({2, 1, 3, 4})
                 .astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(20).reshape({4, 5}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 1, 3, 5})) << "Shape mismatch";
}

// Test batched matmul with broadcast on both sides
TEST(TensorMatmul, MatmulBroadcastBoth) {
    // (2, 1, 3, 4) @ (1, 4, 5) -> (2, 1, 3, 5)
    auto a = axiom::Tensor::arange(24)
                 .reshape({2, 1, 3, 4})
                 .astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(20).reshape({1, 4, 5}).astype(
        axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 1, 3, 5})) << "Shape mismatch";
}

// Test matmul with transpose flags
TEST(TensorMatmul, MatmulTransposeA) {
    // (3, 2)^T @ (3, 4) -> (2, 4) where (3, 2)^T is logically (2, 3)
    auto a =
        axiom::Tensor::arange(6).reshape({3, 2}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b, true, false); // transpose_a=true

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 4})) << "Shape mismatch";
}

// Test matmul with transpose_b
TEST(TensorMatmul, MatmulTransposeB) {
    // (2, 3) @ (4, 3)^T -> (2, 4) where (4, 3)^T is logically (3, 4)
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({4, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b, false, true); // transpose_b=true

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 4})) << "Shape mismatch";
}

// Test matmul with both transposes
TEST(TensorMatmul, MatmulTransposeBoth) {
    // (3, 2)^T @ (4, 3)^T -> (2, 4)
    auto a =
        axiom::Tensor::arange(6).reshape({3, 2}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({4, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b, true, true);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 4})) << "Shape mismatch";
}

// Test member function matmul
TEST(TensorMatmul, MatmulMember) {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = a.matmul(b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 4})) << "Shape mismatch";
}

// Test mm alias
TEST(TensorMatmul, MmAlias) {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = a.mm(b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 4})) << "Shape mismatch";
}

// Test dot alias for vectors
TEST(TensorMatmul, DotAlias) {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto c = a.dot(b);

    ASSERT_TRUE(c.ndim() == 0) << "Should be scalar";
    ASSERT_TRUE(c.item<float>({}) == 14.0f) << "Dot product incorrect";
}

// Test GPU matmul if available
TEST(TensorMatmul, MatmulGpu) {
    SKIP_IF_NO_GPU();

    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto b = axiom::Tensor::arange(12)
                 .reshape({3, 4})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto c = axiom::ops::matmul(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Should be on GPU";
    ASSERT_TRUE(c.shape() == axiom::Shape({2, 4})) << "Shape mismatch";

    // Verify results
    axiom::testing::ExpectTensorEquals<float>(
        c, {20.0f, 23.0f, 26.0f, 29.0f, 56.0f, 68.0f, 80.0f, 92.0f}, 1e-5);
}
