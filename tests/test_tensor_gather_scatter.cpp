#include "axiom_test_utils.hpp"
#include <axiom/tensor_operators.hpp>
#include <cmath>
#include <vector>

using namespace axiom;

// ============================================================================
// Gather Tests
// ============================================================================

TEST(TensorGatherScatter, Gather1d) {
    // Simple 1D gather
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3});

    auto result = x.gather(0, indices);

    ASSERT_TRUE(result.shape() == Shape({3})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 3.0f, 5.0f});
}

TEST(TensorGatherScatter, Gather2dDim0) {
    // 2D gather along dim 0 (select rows)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {3, 2});

    // Select rows 0 and 2
    std::vector<int64_t> indices_data = {0, 0, 2, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2, 2});

    auto result = x.gather(0, indices);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    // Row 0 is [1, 2], Row 2 is [5, 6]
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 2.0f, 5.0f, 6.0f});
}

TEST(TensorGatherScatter, Gather2dDim1) {
    // 2D gather along dim 1 (select columns)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    // For each row, gather specific columns
    std::vector<int64_t> indices_data = {0, 2, 1, 0};
    auto indices = Tensor::from_data(indices_data.data(), {2, 2});

    auto result = x.gather(1, indices);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    // Row 0: [x[0,0], x[0,2]] = [1, 3]
    // Row 1: [x[1,1], x[1,0]] = [5, 4]
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 3.0f, 5.0f, 4.0f});
}

// ============================================================================
// Index Select Tests
// ============================================================================

TEST(TensorGatherScatter, IndexSelectDim0) {
    // Select rows by index
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {3, 2});

    std::vector<int64_t> indices_data = {0, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2});

    auto result = x.index_select(0, indices);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 2.0f, 5.0f, 6.0f});
}

TEST(TensorGatherScatter, IndexSelectDim1) {
    // Select columns by index
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    std::vector<int64_t> indices_data = {0, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2});

    auto result = x.index_select(1, indices);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    // Select columns 0 and 2: [[1, 3], [4, 6]]
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 3.0f, 4.0f, 6.0f});
}

// ============================================================================
// Scatter Tests
// ============================================================================

TEST(TensorGatherScatter, Scatter1d) {
    // Simple 1D scatter
    auto x = Tensor::zeros({5});

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3});

    std::vector<float> src_data = {1.0f, 2.0f, 3.0f};
    auto src = Tensor::from_data(src_data.data(), {3});

    auto result = x.scatter(0, indices, src);

    ASSERT_TRUE(result.shape() == Shape({5})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {1.0f, 0.0f, 2.0f, 0.0f, 3.0f});
}

TEST(TensorGatherScatter, Scatter2dDim0) {
    // 2D scatter along dim 0
    auto x = Tensor::zeros({3, 2});

    std::vector<int64_t> indices_data = {0, 0, 2, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2, 2});

    std::vector<float> src_data = {1.0f, 2.0f, 5.0f, 6.0f};
    auto src = Tensor::from_data(src_data.data(), {2, 2});

    auto result = x.scatter(0, indices, src);

    ASSERT_TRUE(result.shape() == Shape({3, 2})) << "Shape mismatch";
    // Row 0 gets [1, 2], Row 2 gets [5, 6]
    axiom::testing::ExpectTensorEquals<float>(
        result, {1.0f, 2.0f, 0.0f, 0.0f, 5.0f, 6.0f});
}

// ============================================================================
// GPU Tests
// ============================================================================

TEST(TensorGatherScatter, GatherGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5}).gpu();

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3}).gpu();

    auto result = x.gather(0, indices);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 3.0f, 5.0f});
}

TEST(TensorGatherScatter, IndexSelectGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {3, 2}).gpu();

    std::vector<int64_t> indices_data = {0, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2}).gpu();

    auto result = x.index_select(0, indices);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 2.0f, 5.0f, 6.0f});
}

TEST(TensorGatherScatter, ScatterGpu) {
    SKIP_IF_NO_GPU();

    auto x = Tensor::zeros({5}).gpu();

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3}).gpu();

    std::vector<float> src_data = {1.0f, 2.0f, 3.0f};
    auto src = Tensor::from_data(src_data.data(), {3}).gpu();

    auto result = x.scatter(0, indices, src);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {1.0f, 0.0f, 2.0f, 0.0f, 3.0f});
}

// ============================================================================
// Negative Index Tests (PyTorch parity)
// ============================================================================

TEST(TensorGatherScatter, GatherNegativeIndices) {
    // Test that negative indices work like PyTorch
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    // -1 should mean index 4, -2 means index 3
    std::vector<int64_t> indices_data = {-1, -2, 0};
    auto indices = Tensor::from_data(indices_data.data(), {3});

    auto result = x.gather(0, indices);

    ASSERT_TRUE(result.shape() == Shape({3})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result, {5.0f, 4.0f, 1.0f});
}

TEST(TensorGatherScatter, IndexSelectNegativeIndices) {
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {3, 2});

    // -1 should mean row 2, -3 means row 0
    std::vector<int64_t> indices_data = {-1, -3};
    auto indices = Tensor::from_data(indices_data.data(), {2});

    auto result = x.index_select(0, indices);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    // Row -1 (2) is [5, 6], Row -3 (0) is [1, 2]
    axiom::testing::ExpectTensorEquals<float>(result, {5.0f, 6.0f, 1.0f, 2.0f});
}

TEST(TensorGatherScatter, ScatterNegativeIndices) {
    auto x = Tensor::zeros({5});

    // -1 means index 4, -3 means index 2
    std::vector<int64_t> indices_data = {-1, -3, 0};
    auto indices = Tensor::from_data(indices_data.data(), {3});

    std::vector<float> src_data = {10.0f, 20.0f, 30.0f};
    auto src = Tensor::from_data(src_data.data(), {3});

    auto result = x.scatter(0, indices, src);

    ASSERT_TRUE(result.shape() == Shape({5})) << "Shape mismatch";
    // Position 0 gets 30, position 2 gets 20, position 4 gets 10
    axiom::testing::ExpectTensorEquals<float>(
        result, {30.0f, 0.0f, 20.0f, 0.0f, 10.0f});
}
