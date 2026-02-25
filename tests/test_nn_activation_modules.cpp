#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;
using namespace axiom::testing;

// ============================================================================
// ReLU module
// ============================================================================

TEST(NNActivationModules, ReLUForward) {
    ReLU relu;
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto input = Tensor::from_data(data, {5});
    auto output = relu(input);

    auto ptr = output.typed_data<float>();
    EXPECT_FLOAT_EQ(ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(ptr[1], 0.0f);
    EXPECT_FLOAT_EQ(ptr[2], 0.0f);
    EXPECT_FLOAT_EQ(ptr[3], 1.0f);
    EXPECT_FLOAT_EQ(ptr[4], 2.0f);
}

// ============================================================================
// GELU module
// ============================================================================

TEST(NNActivationModules, GELUForward) {
    GELU gelu;
    auto input = Tensor::randn({4, 8});
    auto output = gelu(input);
    EXPECT_EQ(output.shape(), input.shape());

    // Compare with ops::gelu
    auto expected = ops::gelu(input);
    EXPECT_TRUE(output.allclose(expected));
}

// ============================================================================
// SiLU module
// ============================================================================

TEST(NNActivationModules, SiLUForward) {
    SiLU silu;
    auto input = Tensor::randn({4, 8});
    auto output = silu(input);
    auto expected = ops::silu(input);
    EXPECT_TRUE(output.allclose(expected));
}

// ============================================================================
// Sigmoid module
// ============================================================================

TEST(NNActivationModules, SigmoidForward) {
    nn::Sigmoid sig;
    float data[] = {0.0f};
    auto input = Tensor::from_data(data, {1});
    auto output = sig(input);
    EXPECT_NEAR(output.typed_data<float>()[0], 0.5f, 1e-5);
}

// ============================================================================
// Tanh module
// ============================================================================

TEST(NNActivationModules, TanhForward) {
    nn::Tanh tanh_mod;
    float data[] = {0.0f};
    auto input = Tensor::from_data(data, {1});
    auto output = tanh_mod(input);
    EXPECT_NEAR(output.typed_data<float>()[0], 0.0f, 1e-5);
}

// ============================================================================
// LeakyReLU module
// ============================================================================

TEST(NNActivationModules, LeakyReLUForward) {
    LeakyReLU lrelu(0.1f);
    float data[] = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    auto input = Tensor::from_data(data, {5});
    auto output = lrelu(input);

    auto ptr = output.typed_data<float>();
    EXPECT_NEAR(ptr[0], -1.0f, 1e-5);
    EXPECT_NEAR(ptr[1], -0.1f, 1e-5);
    EXPECT_NEAR(ptr[2], 0.0f, 1e-5);
    EXPECT_NEAR(ptr[3], 1.0f, 1e-5);
    EXPECT_NEAR(ptr[4], 10.0f, 1e-5);
}

TEST(NNActivationModules, LeakyReLUDefaultSlope) {
    LeakyReLU lrelu; // default 0.01
    float data[] = {-100.0f};
    auto input = Tensor::from_data(data, {1});
    auto output = lrelu(input);
    EXPECT_NEAR(output.typed_data<float>()[0], -1.0f, 1e-5);
}

// ============================================================================
// Dropout module — identity in inference
// ============================================================================

TEST(NNActivationModules, DropoutIsIdentity) {
    Dropout dropout(0.5f);
    auto input = Tensor::randn({4, 8});
    auto output = dropout(input);
    EXPECT_TRUE(output.allclose(input));
}

// ============================================================================
// Flatten module
// ============================================================================

TEST(NNActivationModules, FlattenDefault) {
    Flatten flatten;
    auto input = Tensor::randn({2, 3, 4, 5});
    auto output = flatten(input);
    // Default start_dim=1: flatten dims 1..end → (2, 60)
    EXPECT_EQ(output.ndim(), 2u);
    EXPECT_EQ(output.shape()[0], 2u);
    EXPECT_EQ(output.shape()[1], 60u);
}

TEST(NNActivationModules, FlattenCustomDims) {
    Flatten flatten(2, -1);
    auto input = Tensor::randn({2, 3, 4, 5});
    auto output = flatten(input);
    // Flatten dims 2..3 → (2, 3, 20)
    EXPECT_EQ(output.ndim(), 3u);
    EXPECT_EQ(output.shape()[0], 2u);
    EXPECT_EQ(output.shape()[1], 3u);
    EXPECT_EQ(output.shape()[2], 20u);
}

// ============================================================================
// Sequential composition with activations
// ============================================================================

TEST(NNActivationModules, SequentialComposition) {
    Sequential seq;
    seq.emplace_back<ReLU>();
    seq.emplace_back<nn::Sigmoid>();

    float data[] = {-1.0f, 0.0f, 1.0f};
    auto input = Tensor::from_data(data, {3});
    auto output = seq(input);

    // -1 → relu → 0 → sigmoid → 0.5
    // 0 → relu → 0 → sigmoid → 0.5
    // 1 → relu → 1 → sigmoid → ~0.731
    auto ptr = output.typed_data<float>();
    EXPECT_NEAR(ptr[0], 0.5f, 1e-5);
    EXPECT_NEAR(ptr[1], 0.5f, 1e-5);
    EXPECT_NEAR(ptr[2], 0.7310586f, 1e-4);
}

// ============================================================================
// GPU parity
// ============================================================================

TEST(NNActivationModules, ReLUGPU) {
    SKIP_IF_NO_GPU();
    ReLU relu;
    auto cpu_input = Tensor::randn({4, 8});
    auto gpu_input = cpu_input.gpu();

    auto cpu_out = relu(cpu_input);
    auto gpu_out = relu(gpu_input);
    EXPECT_EQ(gpu_out.device(), Device::GPU);
    ExpectTensorsClose(cpu_out, gpu_out.cpu(), 1e-5, 1e-5);
}

// ============================================================================
// Float64 dtype preservation
// ============================================================================

TEST(NNActivationModules, Float64Preservation) {
    GELU gelu;
    auto input = Tensor::randn({4, 8}, DType::Float64);
    auto output = gelu(input);
    EXPECT_EQ(output.dtype(), DType::Float64);
}
