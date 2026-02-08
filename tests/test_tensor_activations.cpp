#include "axiom_test_utils.hpp"
#include <cmath>
#include <vector>

using namespace axiom;

// ============================================================================
// Softmax Tests
// ============================================================================

TEST(TensorActivations, Softmax1dCpu) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::softmax(t, -1);

    ASSERT_TRUE(result.shape() == Shape{3}) << "Shape should be preserved";
    ASSERT_TRUE(result.dtype() == DType::Float32) << "Dtype should be Float32";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // Softmax values should sum to 1
    float sum = out[0] + out[1] + out[2];
    ASSERT_TRUE(std::abs(sum - 1.0f) < 1e-5f) << "Softmax should sum to 1";

    // Values should be in ascending order (since inputs are ascending)
    ASSERT_TRUE(out[0] < out[1] && out[1] < out[2])
        << "Softmax values should preserve order";
}

TEST(TensorActivations, Softmax2dCpu) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {2, 3});

    auto result = ops::softmax(t, -1); // Softmax along last axis

    ASSERT_TRUE(result.shape() == Shape({2, 3})) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // Each row should sum to 1
    float sum0 = out[0] + out[1] + out[2];
    float sum1 = out[3] + out[4] + out[5];
    ASSERT_TRUE(std::abs(sum0 - 1.0f) < 1e-5f) << "First row should sum to 1";
    ASSERT_TRUE(std::abs(sum1 - 1.0f) < 1e-5f) << "Second row should sum to 1";

    // Second row has equal inputs, so outputs should be equal
    ASSERT_TRUE(std::abs(out[3] - out[4]) < 1e-5f)
        << "Equal inputs should give equal outputs";
}

TEST(TensorActivations, LogSoftmaxCpu) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::log_softmax(t, -1);

    ASSERT_TRUE(result.shape() == Shape{3}) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // All log_softmax values should be negative (log of probability < 1)
    ASSERT_TRUE(out[0] < 0 && out[1] < 0 && out[2] < 0)
        << "Log softmax values should be negative";

    // exp(log_softmax) should sum to 1
    float sum_exp = std::exp(out[0]) + std::exp(out[1]) + std::exp(out[2]);
    ASSERT_TRUE(std::abs(sum_exp - 1.0f) < 1e-5f)
        << "exp(log_softmax) should sum to 1";
}

TEST(TensorActivations, SoftmaxGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_data(data.data(), {4}).gpu();

    auto result = ops::softmax(t, -1);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
    ASSERT_TRUE(result.shape() == Shape{4}) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT_TRUE(std::abs(sum - 1.0f) < 1e-4f) << "Softmax should sum to 1";
}

// ============================================================================
// ReLU Tests
// ============================================================================

TEST(TensorActivations, ReluCpu) {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {5});

    auto result = ops::relu(t);

    ASSERT_TRUE(result.shape() == Shape{5}) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // ReLU: max(0, x)
    ASSERT_TRUE(std::abs(out[0]) < 1e-5f) << "ReLU(-2) should be 0";
    ASSERT_TRUE(std::abs(out[1]) < 1e-5f) << "ReLU(-1) should be 0";
    ASSERT_TRUE(std::abs(out[2]) < 1e-5f) << "ReLU(0) should be 0";
    ASSERT_TRUE(std::abs(out[3] - 1.0f) < 1e-5f) << "ReLU(1) should be 1";
    ASSERT_TRUE(std::abs(out[4] - 2.0f) < 1e-5f) << "ReLU(2) should be 2";
}

TEST(TensorActivations, ReluFluent) {
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    // Test fluent API
    auto result = t.relu();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    ASSERT_TRUE(std::abs(out[0]) < 1e-5f) << "ReLU(-1) should be 0";
    ASSERT_TRUE(std::abs(out[1]) < 1e-5f) << "ReLU(0) should be 0";
    ASSERT_TRUE(std::abs(out[2] - 1.0f) < 1e-5f) << "ReLU(1) should be 1";
}

TEST(TensorActivations, ReluGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3}).gpu();

    auto result = t.relu();

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    ASSERT_TRUE(std::abs(out[0]) < 1e-5f) << "ReLU(-1) should be 0";
    ASSERT_TRUE(std::abs(out[2] - 1.0f) < 1e-5f) << "ReLU(1) should be 1";
}

// ============================================================================
// Sigmoid Tests
// ============================================================================

TEST(TensorActivations, SigmoidCpu) {
    std::vector<float> data = {-2.0f, 0.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = t.sigmoid();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // sigmoid(0) = 0.5
    ASSERT_TRUE(std::abs(out[1] - 0.5f) < 1e-5f) << "sigmoid(0) should be 0.5";

    // sigmoid(-x) + sigmoid(x) = 1
    ASSERT_TRUE(std::abs(out[0] + out[2] - 1.0f) < 1e-4f)
        << "sigmoid(-x) + sigmoid(x) should be 1";

    // All values should be in (0, 1)
    ASSERT_TRUE(out[0] > 0 && out[0] < 1) << "sigmoid should be in (0, 1)";
    ASSERT_TRUE(out[2] > 0 && out[2] < 1) << "sigmoid should be in (0, 1)";
}

TEST(TensorActivations, SigmoidGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {0.0f};
    auto t = Tensor::from_data(data.data(), {1}).gpu();

    auto result = t.sigmoid();

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT_TRUE(std::abs(out[0] - 0.5f) < 1e-5f) << "sigmoid(0) should be 0.5";
}

// ============================================================================
// Tanh Tests
// ============================================================================

TEST(TensorActivations, TanhCpu) {
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = t.tanh();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // tanh(0) = 0
    ASSERT_TRUE(std::abs(out[1]) < 1e-5f) << "tanh(0) should be 0";

    // tanh is odd: tanh(-x) = -tanh(x)
    ASSERT_TRUE(std::abs(out[0] + out[2]) < 1e-5f)
        << "tanh should be odd function";

    // All values should be in (-1, 1)
    ASSERT_TRUE(out[0] > -1 && out[0] < 1) << "tanh should be in (-1, 1)";
    ASSERT_TRUE(out[2] > -1 && out[2] < 1) << "tanh should be in (-1, 1)";
}

TEST(TensorActivations, TanhGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {0.0f};
    auto t = Tensor::from_data(data.data(), {1}).gpu();

    auto result = t.tanh();

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT_TRUE(std::abs(out[0]) < 1e-5f) << "tanh(0) should be 0";
}

// ============================================================================
// SiLU Tests
// ============================================================================

TEST(TensorActivations, SiluCpu) {
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = t.silu();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // SiLU(0) = 0 * sigmoid(0) = 0
    ASSERT_TRUE(std::abs(out[1]) < 1e-5f) << "SiLU(0) should be 0";

    // SiLU(1) = 1 * sigmoid(1) ~= 0.731
    ASSERT_TRUE(std::abs(out[2] - 0.731f) < 0.01f)
        << "SiLU(1) should be ~0.731";
}

TEST(TensorActivations, SiluGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {2}).gpu();

    auto result = t.silu();

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT_TRUE(std::abs(out[0]) < 1e-5f) << "SiLU(0) should be 0";
}

// ============================================================================
// Fluent API Chaining Tests
// ============================================================================

TEST(TensorActivations, FluentChaining) {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {5});

    // Chain: (x * 2).relu().sigmoid()
    auto result = (t * 2.0f).relu().sigmoid();

    ASSERT_TRUE(result.shape() == Shape{5}) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // After *2: {-4, -2, 0, 2, 4}
    // After relu: {0, 0, 0, 2, 4}
    // After sigmoid: {0.5, 0.5, 0.5, ~0.88, ~0.98}
    ASSERT_TRUE(std::abs(out[0] - 0.5f) < 1e-5f)
        << "sigmoid(relu(-4)) should be 0.5";
    ASSERT_TRUE(std::abs(out[1] - 0.5f) < 1e-5f)
        << "sigmoid(relu(-2)) should be 0.5";
    ASSERT_TRUE(std::abs(out[2] - 0.5f) < 1e-5f)
        << "sigmoid(relu(0)) should be 0.5";
    ASSERT_TRUE(out[3] > 0.85f && out[3] < 0.92f)
        << "sigmoid(2) should be ~0.88";
    ASSERT_TRUE(out[4] > 0.95f) << "sigmoid(4) should be > 0.95";
}

TEST(TensorActivations, FluentChainingGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3}).gpu();

    auto result = (t + 1.0f).relu().gelu();

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should stay on GPU";
    ASSERT_TRUE(result.shape() == Shape{3}) << "Shape should be preserved";
}

// ============================================================================
// GELU Tests
// ============================================================================

TEST(TensorActivations, GeluCpu) {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {5});

    auto result = ops::gelu(t);

    ASSERT_TRUE(result.shape() == Shape{5}) << "Shape should be preserved";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // GELU(0) = 0
    ASSERT_TRUE(std::abs(out[2]) < 1e-5f) << "GELU(0) should be 0";

    // GELU(x) approaches x for large positive x
    ASSERT_TRUE(out[4] > 1.9f && out[4] < 2.1f)
        << "GELU(2) should be close to 2";

    // GELU(x) approaches 0 for large negative x
    ASSERT_TRUE(out[0] < 0 && out[0] > -0.1f)
        << "GELU(-2) should be close to 0";

    // GELU is monotonically increasing for x > 0
    ASSERT_TRUE(out[3] < out[4]) << "GELU should be increasing for positive x";
}

TEST(TensorActivations, GeluGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {3}).gpu();

    auto result = ops::gelu(t);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    ASSERT_TRUE(std::abs(out[0]) < 1e-5f) << "GELU(0) should be 0";
}

// ============================================================================
// Erf Tests
// ============================================================================

TEST(TensorActivations, ErfCpu) {
    std::vector<float> data = {0.0f, 1.0f, -1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::erf(t);

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // erf(0) = 0
    ASSERT_TRUE(std::abs(out[0]) < 1e-5f) << "erf(0) should be 0";

    // erf(1) ~= 0.8427
    ASSERT_TRUE(std::abs(out[1] - 0.8427f) < 0.01f)
        << "erf(1) should be ~0.8427";

    // erf is odd: erf(-x) = -erf(x)
    ASSERT_TRUE(std::abs(out[1] + out[2]) < 1e-5f)
        << "erf should be odd function";
}

TEST(TensorActivations, ErfGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> data = {0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {2}).gpu();

    auto result = ops::erf(t);

    ASSERT_TRUE(result.device() == Device::GPU) << "Result should be on GPU";
}

// ============================================================================
// Dropout Tests
// ============================================================================

TEST(TensorActivations, DropoutTraining) {
    auto t = Tensor::ones({100});

    auto [output, mask] = ops::dropout(t, 0.5f, true);

    ASSERT_TRUE(output.shape() == Shape{100}) << "Output shape should match";
    ASSERT_TRUE(mask.shape() == Shape{100}) << "Mask shape should match";
    ASSERT_TRUE(mask.dtype() == DType::Bool) << "Mask should be Bool";

    auto output_cpu = output.cpu();
    const float *out = output_cpu.typed_data<float>();

    // With 50% dropout, some values should be 0 and some should be ~2 (scaled)
    int zeros = 0;
    int scaled = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(out[i]) < 1e-5f) {
            zeros++;
        } else if (std::abs(out[i] - 2.0f) < 0.1f) {
            scaled++;
        }
    }

    // Should have roughly 50% zeros (allow some variance)
    ASSERT_TRUE(zeros > 20 && zeros < 80)
        << "Should have ~50% zeros with p=0.5";
    ASSERT_TRUE(scaled > 20 && scaled < 80) << "Should have ~50% scaled values";
}

TEST(TensorActivations, DropoutInference) {
    auto t = Tensor::ones({10});

    auto [output, mask] = ops::dropout(t, 0.5f, false);

    // In inference mode (training=false), dropout should be identity
    axiom::testing::ExpectTensorEquals<float>(
        output, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        1e-4);
}

TEST(TensorActivations, DropoutPZero) {
    auto t = Tensor::ones({10});

    auto [output, mask] = ops::dropout(t, 0.0f, true);

    // With p=0, no dropout occurs
    axiom::testing::ExpectTensorEquals<float>(
        output, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        1e-4);
}

// ============================================================================
// Boolean Reduction Tests
// ============================================================================

TEST(TensorActivations, AnyCpu) {
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f};
    auto t = Tensor::from_data(data.data(), {4});

    auto result = ops::any(t, {}, false);

    ASSERT_TRUE(result.size() == 1) << "Should reduce to scalar";
    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT_TRUE(out[0] != 0.0f) << "any([0,0,1,0]) should be true";
}

TEST(TensorActivations, AllCpu) {
    std::vector<float> data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {4});

    auto result = ops::all(t, {}, false);

    ASSERT_TRUE(result.size() == 1) << "Should reduce to scalar";
    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT_TRUE(out[0] != 0.0f) << "all([1,1,1,1]) should be true";
}

TEST(TensorActivations, AllWithZeroCpu) {
    std::vector<float> data = {1.0f, 0.0f, 1.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {4});

    auto result = ops::all(t, {}, false);

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT_TRUE(out[0] == 0.0f) << "all([1,0,1,1]) should be false";
}
