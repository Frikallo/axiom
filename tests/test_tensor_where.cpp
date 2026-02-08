#include "axiom_test_utils.hpp"
#include <axiom/tensor_operators.hpp>
#include <cmath>
#include <vector>

using namespace axiom;

// Helper to create bool tensor
Tensor make_bool_tensor(const std::vector<bool> &data, const Shape &shape,
                        Device device = Device::CPU) {
    // Convert bool vector to uint8 for storage
    std::vector<uint8_t> u8_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        u8_data[i] = data[i] ? 1 : 0;
    }
    auto t = Tensor::from_data(u8_data.data(), shape).astype(DType::Bool);
    if (device == Device::GPU) {
        return t.to(Device::GPU);
    }
    return t;
}

// ============================================================================
// CPU Tests
// ============================================================================

TEST(TensorWhere, BasicCpu) {
    auto cond = make_bool_tensor({true, false, true, false}, {4});
    auto a = Tensor::full({4}, 1.0f);
    auto b = Tensor::full({4}, 0.0f);

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.device() == Device::CPU) << "Result should be on CPU";
    ASSERT_TRUE(result.shape() == Shape{4}) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 0.0f, 1.0f, 0.0f});
}

TEST(TensorWhere, 2dCpu) {
    auto cond = make_bool_tensor({true, false, false, true}, {2, 2});
    auto a = Tensor::full({2, 2}, 10.0f);
    auto b = Tensor::full({2, 2}, -10.0f);

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {10.0f, -10.0f, -10.0f, 10.0f});
}

TEST(TensorWhere, BroadcastConditionCpu) {
    // Condition: [true, false], shape (2,)
    // a, b: shape (3, 2)
    auto cond = make_bool_tensor({true, false}, {2});
    auto a = Tensor::ones({3, 2});
    auto b = Tensor::zeros({3, 2});

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.shape() == Shape({3, 2})) << "Shape mismatch";
    // Each row should be [1, 0]
    axiom::testing::ExpectTensorEquals<float>(
        result, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
}

TEST(TensorWhere, BroadcastValuesCpu) {
    // Condition: shape (2, 2)
    // a: scalar-like shape (1,)
    // b: scalar-like shape (1,)
    auto cond = make_bool_tensor({true, false, true, true}, {2, 2});
    auto a = Tensor::full({1}, 5.0f);
    auto b = Tensor::full({1}, -5.0f);

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {5.0f, -5.0f, 5.0f, 5.0f});
}

TEST(TensorWhere, IntConditionCpu) {
    // Non-bool condition (non-zero = true)
    std::vector<int32_t> cond_data = {1, 0, 2, 0};
    auto cond = Tensor::from_data(cond_data.data(), {4});
    auto a = Tensor::full({4}, 100.0f);
    auto b = Tensor::full({4}, 0.0f);

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.shape() == Shape{4}) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {100.0f, 0.0f, 100.0f, 0.0f});
}

TEST(TensorWhere, DifferentDtypesCpu) {
    auto cond = make_bool_tensor({true, false}, {2});
    std::vector<int32_t> a_data = {10, 20};
    std::vector<float> b_data = {0.5f, 0.5f};
    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::where(cond, a, b);

    // Result should be promoted to float
    ASSERT_TRUE(result.dtype() == DType::Float32) << "Dtype should be Float32";
    axiom::testing::ExpectTensorEquals<float>(result, {10.0f, 0.5f});
}

// ============================================================================
// GPU Tests
// ============================================================================

TEST(TensorWhere, BasicGpu) {
    SKIP_IF_NO_GPU();

    auto cond = make_bool_tensor({true, false, true, false}, {4}, Device::GPU);
    auto a = Tensor::full({4}, 1.0f).gpu();
    auto b = Tensor::full({4}, 0.0f).gpu();

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(result.shape() == Shape{4}) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 0.0f, 1.0f, 0.0f});
}

TEST(TensorWhere, 2dGpu) {
    SKIP_IF_NO_GPU();

    auto cond =
        make_bool_tensor({true, false, false, true}, {2, 2}, Device::GPU);
    auto a = Tensor::full({2, 2}, 10.0f).gpu();
    auto b = Tensor::full({2, 2}, -10.0f).gpu();

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {10.0f, -10.0f, -10.0f, 10.0f});
}

TEST(TensorWhere, BroadcastGpu) {
    SKIP_IF_NO_GPU();

    // Condition: [true, false], shape (2,)
    // a, b: shape (3, 2)
    auto cond = make_bool_tensor({true, false}, {2}, Device::GPU);
    auto a = Tensor::ones({3, 2}).gpu();
    auto b = Tensor::zeros({3, 2}).gpu();

    auto result = ops::where(cond, a, b);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(result.shape() == Shape({3, 2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(
        result, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
}

TEST(TensorWhere, AttentionMaskGpu) {
    SKIP_IF_NO_GPU();

    // Attention mask pattern: where(mask, scores, -1e9)
    auto mask =
        make_bool_tensor({true, true, false, false}, {2, 2}, Device::GPU);
    std::vector<float> scores_data = {0.5f, 0.3f, 0.2f, 0.1f};
    auto scores = Tensor::from_data(scores_data.data(), {2, 2}).gpu();
    auto neg_inf = Tensor::full({2, 2}, -1e9f).gpu();

    auto result = ops::where(mask, scores, neg_inf);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {0.5f, 0.3f, -1e9f, -1e9f}, 1.0);
}

TEST(TensorWhere, WithComparisonResult) {
    // Test using comparison result as condition
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f};
    auto x = Tensor::from_data(x_data.data(), {4});
    auto zero = Tensor::zeros({4});
    auto cond =
        ops::greater(x, zero); // x > 0 - Should be {true, false, true, false}

    auto positive = Tensor::full({4}, 1.0f);
    auto negative = Tensor::full({4}, -1.0f);

    auto result = ops::where(cond, positive, negative);

    axiom::testing::ExpectTensorEquals<float>(result,
                                              {1.0f, -1.0f, 1.0f, -1.0f});
}

TEST(TensorWhere, ReluPattern) {
    // Common pattern: ReLU using where
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f, 0.0f};
    auto x = Tensor::from_data(x_data.data(), {5});
    auto zero = Tensor::zeros({5});
    auto cond = ops::greater(x, zero); // x > 0

    auto result = ops::where(cond, x, zero);

    axiom::testing::ExpectTensorEquals<float>(result,
                                              {1.0f, 0.0f, 3.0f, 0.0f, 0.0f});
}

// ============================================================================
// Fluent API Tests
// ============================================================================

TEST(TensorWhere, FluentMaskedFill) {
    // Test masked_fill with scalar comparison
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f, 0.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    // ReLU using masked_fill: zero out negative values
    auto mask = x < 0.0f;
    auto result = x.masked_fill(mask, 0.0f);

    axiom::testing::ExpectTensorEquals<float>(result,
                                              {1.0f, 0.0f, 3.0f, 0.0f, 0.0f});
}

TEST(TensorWhere, FluentMaskedSelect) {
    // Test masked_select
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    // Get all positive values
    auto mask = x > 0.0f;
    auto positives = x.masked_select(mask);

    ASSERT_TRUE(positives.ndim() == 1) << "Result should be 1D";
    ASSERT_TRUE(positives.size() == 3) << "Should have 3 positive values";
    axiom::testing::ExpectTensorEquals<float>(positives, {1.0f, 3.0f, 5.0f});
}

TEST(TensorWhere, FluentWhereMethod) {
    // Test the fluent where method
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f};
    auto x = Tensor::from_data(x_data.data(), {4});

    // ReLU using fluent where
    auto result = x.where(x > 0.0f, 0.0f);

    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 0.0f, 3.0f, 0.0f});
}

TEST(TensorWhere, ScalarComparisonOperators) {
    // Test scalar comparison operators for clean syntax
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    // Test x > 3
    auto gt3 = x > 3.0f;
    auto gt3_cpu = gt3.cpu();
    const uint8_t *gt3_data = gt3_cpu.typed_data<uint8_t>();
    ASSERT_TRUE(gt3_data[0] == 0 && gt3_data[1] == 0 && gt3_data[2] == 0)
        << "1,2,3 should NOT be > 3";
    ASSERT_TRUE(gt3_data[3] == 1 && gt3_data[4] == 1) << "4,5 should be > 3";

    // Test x <= 2
    auto le2 = x <= 2.0f;
    auto le2_cpu = le2.cpu();
    const uint8_t *le2_data = le2_cpu.typed_data<uint8_t>();
    ASSERT_TRUE(le2_data[0] == 1 && le2_data[1] == 1) << "1,2 should be <= 2";
    ASSERT_TRUE(le2_data[2] == 0 && le2_data[3] == 0 && le2_data[4] == 0)
        << "3,4,5 should NOT be <= 2";
}

TEST(TensorWhere, AttentionMaskFluent) {
    // Attention masking pattern using fluent API
    std::vector<float> scores_data = {0.5f, 0.3f, 0.2f, 0.1f};
    auto scores = Tensor::from_data(scores_data.data(), {2, 2});

    // Create causal mask (lower triangular)
    auto mask = make_bool_tensor({true, false, true, true}, {2, 2});

    // Apply attention mask using fluent API
    auto masked_scores = scores.masked_fill(!mask, -1e9f);

    axiom::testing::ExpectTensorEquals<float>(masked_scores,
                                              {0.5f, -1e9f, 0.2f, 0.1f}, 1.0);
}
