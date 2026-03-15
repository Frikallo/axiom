#include "axiom_test_utils.hpp"
#include <gtest/gtest.h>
#include <iostream>

#ifdef AXIOM_HAS_ANE
#include "axiom/nn/ane_compiled_model.hpp"
#include "axiom/nn/linear.hpp"
#include "axiom/nn/activation.hpp"
#include "axiom/nn/normalization.hpp"
#include "backends/ane/ane_bridge.h"
#endif

#define SKIP_IF_NO_ANE()                                                       \
    do {                                                                        \
        if (!ane_is_available()) {                                             \
            GTEST_SKIP() << "ANE not available";                               \
        }                                                                      \
    } while (0)

namespace axiom {
namespace testing {

#ifdef AXIOM_HAS_ANE

using backends::ane::ANECompiledModel;

// ============================================================================
// Basic compilation tests
// ============================================================================

TEST(ANECompiled, IsSupported) {
    SKIP_IF_NO_ANE();

    nn::Linear linear;
    EXPECT_TRUE(ANECompiledModel::is_supported(linear));

    nn::ReLU relu;
    EXPECT_TRUE(ANECompiledModel::is_supported(relu));

    nn::SiLU silu;
    EXPECT_TRUE(ANECompiledModel::is_supported(silu));

    nn::LayerNorm ln;
    EXPECT_TRUE(ANECompiledModel::is_supported(ln));

    nn::Dropout dropout;
    EXPECT_TRUE(ANECompiledModel::is_supported(dropout));
}

TEST(ANECompiled, CompileLinear) {
    SKIP_IF_NO_ANE();

    // Create a Linear(4, 8) layer
    nn::Linear linear;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor({8, 4}, DType::Float32);
    state["bias"] = Tensor({8}, DType::Float32);

    // Initialize with known values
    float *w = state["weight"].typed_data<float>();
    for (int i = 0; i < 32; i++)
        w[i] = static_cast<float>(i) * 0.01f;
    float *b = state["bias"].typed_data<float>();
    for (int i = 0; i < 8; i++)
        b[i] = 0.0f;

    linear.load_state_dict(state);

    // Compile for input shape [1, 4]
    auto compiled = ANECompiledModel::compile(linear, {1, 4});

    auto info = compiled.plan_info();
    EXPECT_EQ(info.ane_steps, 1);
    EXPECT_EQ(info.compile_count, 1);
    EXPECT_GT(info.weight_bytes, 0u);
    std::cout << "[INFO] " << info.summary << "\n";
}

TEST(ANECompiled, LinearForwardCorrectness) {
    SKIP_IF_NO_ANE();

    // Create Linear(4, 8) with identity-like weight
    nn::Linear linear;
    std::map<std::string, Tensor> state;
    auto weight = Tensor({8, 4}, DType::Float32);
    auto bias = Tensor({8}, DType::Float32);

    // Set weight to scaled identity: output[i] = input[i % 4] * 0.5
    float *w = weight.typed_data<float>();
    std::memset(w, 0, 32 * sizeof(float));
    for (int i = 0; i < 4; i++)
        w[i * 4 + i] = 0.5f; // First 4 outputs are 0.5x input

    float *b = bias.typed_data<float>();
    std::memset(b, 0, 8 * sizeof(float));

    state["weight"] = weight;
    state["bias"] = bias;
    linear.load_state_dict(state);

    // Compile with batch=2
    auto compiled = ANECompiledModel::compile(linear, {2, 4});

    // Run on ANE
    auto input = Tensor({2, 4}, DType::Float32);
    float *in_data = input.typed_data<float>();
    // Row 0
    in_data[0] = 2.0f; in_data[1] = 4.0f;
    in_data[2] = 6.0f; in_data[3] = 8.0f;
    // Row 1
    in_data[4] = 1.0f; in_data[5] = 0.0f;
    in_data[6] = 0.0f; in_data[7] = 0.0f;

    auto ane_output = compiled.forward(input);

    // Run on CPU for reference
    auto cpu_output = linear(input);

    std::cout << "[INFO] ANE output: ";
    float *ane_data = ane_output.typed_data<float>();
    for (size_t i = 0; i < ane_output.size(); i++)
        std::cout << ane_data[i] << " ";
    std::cout << "\n";

    std::cout << "[INFO] CPU output: ";
    float *cpu_data = cpu_output.typed_data<float>();
    for (size_t i = 0; i < cpu_output.size(); i++)
        std::cout << cpu_data[i] << " ";
    std::cout << "\n";

    // Compare (FP16 tolerance)
    ASSERT_EQ(ane_output.shape(), cpu_output.shape());
    for (size_t i = 0; i < ane_output.size(); i++) {
        EXPECT_NEAR(ane_data[i], cpu_data[i], 0.1f)
            << "Mismatch at index " << i;
    }
}

TEST(ANECompiled, ReLUForward) {
    SKIP_IF_NO_ANE();

    nn::ReLU relu;

    // Use a larger shape with spatial > 1 to avoid edge cases
    auto compiled = ANECompiledModel::compile(relu, {2, 4});

    auto input = Tensor({2, 4}, DType::Float32);
    float *data = input.typed_data<float>();
    // Row 0: [-1, 0, 1, 2]
    data[0] = -1.0f; data[1] = 0.0f; data[2] = 1.0f; data[3] = 2.0f;
    // Row 1: [-3, -2, 3, 4]
    data[4] = -3.0f; data[5] = -2.0f; data[6] = 3.0f; data[7] = 4.0f;

    // Also test CPU reference
    auto cpu_output = relu(input);
    float *cpu_out = cpu_output.typed_data<float>();

    auto output = compiled.forward(input);
    float *out = output.typed_data<float>();

    std::cout << "[INFO] ReLU input:   ";
    for (int i = 0; i < 8; i++) std::cout << data[i] << " ";
    std::cout << "\n[INFO] CPU output:   ";
    for (int i = 0; i < 8; i++) std::cout << cpu_out[i] << " ";
    std::cout << "\n[INFO] ANE output:   ";
    for (int i = 0; i < 8; i++) std::cout << out[i] << " ";
    std::cout << "\n";

    EXPECT_NEAR(out[0], 0.0f, 0.01f);  // relu(-1) = 0
    EXPECT_NEAR(out[1], 0.0f, 0.01f);  // relu(0) = 0
    EXPECT_NEAR(out[2], 1.0f, 0.01f);  // relu(1) = 1
    EXPECT_NEAR(out[3], 2.0f, 0.01f);  // relu(2) = 2
    EXPECT_NEAR(out[4], 0.0f, 0.01f);  // relu(-3) = 0
    EXPECT_NEAR(out[5], 0.0f, 0.01f);  // relu(-2) = 0
    EXPECT_NEAR(out[6], 3.0f, 0.01f);  // relu(3) = 3
    EXPECT_NEAR(out[7], 4.0f, 0.01f);  // relu(4) = 4
}

TEST(ANECompiled, RepeatedForward) {
    SKIP_IF_NO_ANE();

    nn::ReLU relu;
    auto compiled = ANECompiledModel::compile(relu, {1, 8});

    int initial_count = ane_compile_count();

    // Run 10 times — should reuse the same compiled graph
    for (int i = 0; i < 10; i++) {
        auto input = Tensor({1, 8}, DType::Float32);
        float *data = input.typed_data<float>();
        for (int j = 0; j < 8; j++)
            data[j] = static_cast<float>(j + i);

        auto output = compiled.forward(input);
        ASSERT_EQ(output.shape(), Shape({1, 8}));
    }

    // Should NOT have compiled any additional models
    EXPECT_EQ(ane_compile_count(), initial_count);
}

TEST(ANECompiled, ShapeMismatchThrows) {
    SKIP_IF_NO_ANE();

    nn::ReLU relu;
    auto compiled = ANECompiledModel::compile(relu, {1, 4});

    // Wrong shape should throw
    auto wrong_input = Tensor({1, 8}, DType::Float32);
    EXPECT_THROW(compiled.forward(wrong_input), ShapeError);
}

#else // !AXIOM_HAS_ANE

TEST(ANECompiled, Skipped) { GTEST_SKIP() << "ANE not compiled"; }

#endif

} // namespace testing
} // namespace axiom
