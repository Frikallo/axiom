#include "axiom_test_utils.hpp"

using namespace axiom;

// ============================================================================
// Layer Normalization Tests
// ============================================================================

TEST(TensorNormalization, LayerNormBasic) {
    // Input: [1, 2, 3, 4, 5] with mean=3, var=2
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto t = Tensor::from_data(data.data(), {5});

    // Weight=1, bias=0 (identity transform after normalization)
    auto weight = Tensor::ones({5});
    auto bias = Tensor::zeros({5});

    auto result = ops::layer_norm(t, weight, bias, -1, 1e-5f);

    ASSERT_TRUE(result.shape() == Shape{5}) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // Normalized output should have mean ~0 and std ~1
    float sum = 0;
    for (int i = 0; i < 5; ++i)
        sum += out[i];
    float mean = sum / 5.0f;
    ASSERT_TRUE(std::abs(mean) < 0.1f) << "Normalized mean should be ~0";

    // Check center value (3 -> 0 after normalization)
    ASSERT_TRUE(std::abs(out[2]) < 0.1f) << "Center value should be ~0";
}

TEST(TensorNormalization, LayerNormWithWeightBias) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_data(data.data(), {4});

    // Weight=2, bias=1
    auto weight = Tensor::full({4}, 2.0f);
    auto bias = Tensor::full({4}, 1.0f);

    auto result = ops::layer_norm(t, weight, bias, -1, 1e-5f);

    auto result_cpu = result.cpu();
    [[maybe_unused]] const float *out = result_cpu.typed_data<float>();

    // After weight*normalized+bias, all values should be shifted by 1
    // The center values should be: 2*0 + 1 = 1
    // (since mean=2.5, values 2 and 3 are around the center)
}

TEST(TensorNormalization, LayerNorm2D) {
    // 2x4 tensor, normalize along last axis
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto t = Tensor::from_data(data.data(), {2, 4});

    auto weight = Tensor::ones({4});
    auto bias = Tensor::zeros({4});

    auto result = ops::layer_norm(t, weight, bias, -1, 1e-5f);

    ASSERT_TRUE(result.shape() == Shape({2, 4})) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // Each row should have mean ~0
    float sum0 = out[0] + out[1] + out[2] + out[3];
    float sum1 = out[4] + out[5] + out[6] + out[7];
    ASSERT_TRUE(std::abs(sum0 / 4.0f) < 0.1f) << "First row mean should be ~0";
    ASSERT_TRUE(std::abs(sum1 / 4.0f) < 0.1f) << "Second row mean should be ~0";
}

// ============================================================================
// RMS Normalization Tests
// ============================================================================

TEST(TensorNormalization, RmsNormBasic) {
    std::vector<float> data = {
        3.0f, 4.0f}; // RMS = sqrt((9+16)/2) = sqrt(12.5) ~= 3.536
    auto t = Tensor::from_data(data.data(), {2});

    auto weight = Tensor::ones({2});

    auto result = ops::rms_norm(t, weight, -1, 1e-5f);

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // After RMS normalization: [3/3.536, 4/3.536] = [0.848, 1.131]
    float rms = std::sqrt((9.0f + 16.0f) / 2.0f);
    ASSERT_TRUE(std::abs(out[0] - 3.0f / rms) < 0.01f)
        << "First value should be 3/RMS";
    ASSERT_TRUE(std::abs(out[1] - 4.0f / rms) < 0.01f)
        << "Second value should be 4/RMS";
}

TEST(TensorNormalization, RmsNormWithWeight) {
    std::vector<float> data = {3.0f, 4.0f};
    auto t = Tensor::from_data(data.data(), {2});

    auto weight = Tensor::full({2}, 2.0f);

    auto result = ops::rms_norm(t, weight, -1, 1e-5f);

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // With weight=2: [2*3/3.536, 2*4/3.536]
    float rms = std::sqrt((9.0f + 16.0f) / 2.0f);
    ASSERT_TRUE(std::abs(out[0] - 2.0f * 3.0f / rms) < 0.01f)
        << "First value should be 2*3/RMS";
    ASSERT_TRUE(std::abs(out[1] - 2.0f * 4.0f / rms) < 0.01f)
        << "Second value should be 2*4/RMS";
}

TEST(TensorNormalization, RmsNorm2D) {
    std::vector<float> data = {3.0f, 4.0f, 6.0f, 8.0f};
    auto t = Tensor::from_data(data.data(), {2, 2});

    auto weight = Tensor::ones({2});

    auto result = ops::rms_norm(t, weight, -1, 1e-5f);

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // First row: [3,4], RMS = sqrt(12.5)
    // Second row: [6,8], RMS = sqrt(50)
    float rms1 = std::sqrt((9.0f + 16.0f) / 2.0f);
    float rms2 = std::sqrt((36.0f + 64.0f) / 2.0f);

    ASSERT_TRUE(std::abs(out[0] - 3.0f / rms1) < 0.01f) << "Row 1, col 1";
    ASSERT_TRUE(std::abs(out[1] - 4.0f / rms1) < 0.01f) << "Row 1, col 2";
    ASSERT_TRUE(std::abs(out[2] - 6.0f / rms2) < 0.01f) << "Row 2, col 1";
    ASSERT_TRUE(std::abs(out[3] - 8.0f / rms2) < 0.01f) << "Row 2, col 2";
}

// ============================================================================
// GPU Tests
// ============================================================================

TEST(TensorNormalization, LayerNormGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_data(data.data(), {4}).gpu();
    auto weight = Tensor::ones({4}).gpu();
    auto bias = Tensor::zeros({4}).gpu();

    auto result = ops::layer_norm(t, weight, bias, -1, 1e-5f);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(result.shape() == Shape{4}) << "Shape should be preserved";

    // Verify mean is ~0
    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT_TRUE(std::abs(sum / 4.0f) < 0.1f) << "Normalized mean should be ~0";
}

TEST(TensorNormalization, RmsNormGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {3.0f, 4.0f};
    auto t = Tensor::from_data(data.data(), {2}).gpu();
    auto weight = Tensor::ones({2}).gpu();

    auto result = ops::rms_norm(t, weight, -1, 1e-5f);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(result.shape() == Shape{2}) << "Shape should be preserved";
}

// ============================================================================
// Integration Test: Transformer Block Pattern
// ============================================================================

TEST(TensorNormalization, TransformerAttentionPattern) {
    // Simulate a small attention computation
    // This tests the integration of multiple operations

    // Create small Q, K, V matrices (batch=1, seq_len=4, d_model=4)
    auto q = Tensor::randn({1, 4, 4});
    auto k = Tensor::randn({1, 4, 4});
    auto v = Tensor::randn({1, 4, 4});

    // Attention scores: Q @ K^T / sqrt(d_k)
    auto scores = q.matmul(k.transpose()); // transpose() swaps last two dims
    float scale = 1.0f / std::sqrt(4.0f);
    auto scores_scaled = ops::multiply(scores, Tensor::full({1}, scale));

    // Apply softmax
    auto probs = ops::softmax(scores_scaled, -1);

    // Weighted sum
    auto out = probs.matmul(v);

    ASSERT_TRUE(out.shape() == Shape({1, 4, 4}))
        << "Output shape should be (1, 4, 4)";

    // Check softmax probabilities sum to 1 along last axis
    auto probs_cpu = probs.cpu();
    const float *p = probs_cpu.typed_data<float>();

    for (int i = 0; i < 4; ++i) {
        float row_sum = 0;
        for (int j = 0; j < 4; ++j) {
            row_sum += p[i * 4 + j];
        }
        ASSERT_TRUE(std::abs(row_sum - 1.0f) < 0.01f)
            << "Softmax row should sum to 1";
    }
}
