#include "axiom_test_utils.hpp"

// Test equal
TEST(TensorComparisons, Equal) {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::equal(a, b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    std::vector<bool> expected(6, true);
    axiom::testing::ExpectTensorEquals<bool>(c, expected);
}

TEST(TensorComparisons, EqualWithOperator) {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto c = (a == b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}

// Test not_equal
TEST(TensorComparisons, NotEqual) {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({4}, 2.0f);
    auto c = axiom::ops::not_equal(a, b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    // [0, 1, 2, 3] != 2 -> [true, true, false, true]
    axiom::testing::ExpectTensorEquals<bool>(c, {true, true, false, true});
}

TEST(TensorComparisons, NotEqualWithOperator) {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({4}, 2.0f);
    auto c = (a != b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}

// Test less
TEST(TensorComparisons, Less) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::less(a, b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    // [0, 1, 2, 3, 4] < 2 -> [true, true, false, false, false]
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {true, true, false, false, false});
}

TEST(TensorComparisons, LessWithOperator) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a < b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}

// Test less_equal
TEST(TensorComparisons, LessEqual) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::less_equal(a, b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    // [0, 1, 2, 3, 4] <= 2 -> [true, true, true, false, false]
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {true, true, true, false, false});
}

TEST(TensorComparisons, LessEqualWithOperator) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a <= b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}

// Test greater
TEST(TensorComparisons, Greater) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::greater(a, b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    // [0, 1, 2, 3, 4] > 2 -> [false, false, false, true, true]
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {false, false, false, true, true});
}

TEST(TensorComparisons, GreaterWithOperator) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a > b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}

// Test greater_equal
TEST(TensorComparisons, GreaterEqual) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::greater_equal(a, b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    // [0, 1, 2, 3, 4] >= 2 -> [false, false, true, true, true]
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {false, false, true, true, true});
}

TEST(TensorComparisons, GreaterEqualWithOperator) {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a >= b);

    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}

// Test comparison with broadcasting
TEST(TensorComparisons, ComparisonBroadcasting) {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({}, 2.0f); // Scalar
    auto c = axiom::ops::greater(a, b);

    ASSERT_TRUE(c.shape() == axiom::Shape({2, 3})) << "Shape should match";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}

// ============================================================================
// GPU Tests
// ============================================================================

TEST(TensorComparisons, EqualGpu) {
    SKIP_IF_NO_GPU();

    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .gpu();
    auto b = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .gpu();
    auto c = axiom::ops::equal(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    std::vector<bool> expected(6, true);
    axiom::testing::ExpectTensorEquals<bool>(c, expected);
}

TEST(TensorComparisons, NotEqualGpu) {
    SKIP_IF_NO_GPU();

    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({4}, 2.0f).gpu();
    auto c = axiom::ops::not_equal(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    axiom::testing::ExpectTensorEquals<bool>(c, {true, true, false, true});
}

TEST(TensorComparisons, LessGpu) {
    SKIP_IF_NO_GPU();

    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::less(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {true, true, false, false, false});
}

TEST(TensorComparisons, LessEqualGpu) {
    SKIP_IF_NO_GPU();

    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::less_equal(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {true, true, true, false, false});
}

TEST(TensorComparisons, GreaterGpu) {
    SKIP_IF_NO_GPU();

    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::greater(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {false, false, false, true, true});
}

TEST(TensorComparisons, GreaterEqualGpu) {
    SKIP_IF_NO_GPU();

    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::greater_equal(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
    axiom::testing::ExpectTensorEquals<bool>(c,
                                             {false, false, true, true, true});
}

TEST(TensorComparisons, ComparisonBroadcastingGpu) {
    SKIP_IF_NO_GPU();

    // Shape (3, 1) vs (4) should broadcast to (3, 4)
    auto a = axiom::Tensor::arange(3)
                 .reshape({3, 1})
                 .astype(axiom::DType::Float32)
                 .gpu();
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32).gpu();
    auto c = axiom::ops::less(a, b);

    ASSERT_TRUE(c.device() == axiom::Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(c.shape() == axiom::Shape({3, 4})) << "Broadcasting failed";
    ASSERT_TRUE(c.dtype() == axiom::DType::Bool) << "Result should be bool";
}
