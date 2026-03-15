#include "axiom_test_utils.hpp"
#include <gtest/gtest.h>
#include <iostream>

#include "axiom/nn/rnn.hpp"

#ifdef AXIOM_HAS_ANE
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

// ============================================================================
// Module::to(Device::ANE) integration tests
// ============================================================================

TEST(ANEIntegration, ModuleDeviceTracking) {
    nn::Linear linear;
    EXPECT_EQ(linear.device(), Device::CPU);

#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();
    linear.to(Device::ANE);
    EXPECT_EQ(linear.device(), Device::ANE);

    // Moving back to CPU should work
    linear.to(Device::CPU);
    EXPECT_EQ(linear.device(), Device::CPU);
#endif
}

#ifdef AXIOM_HAS_ANE

TEST(ANEIntegration, LinearToANE) {
    SKIP_IF_NO_ANE();

    // Create and initialize a Linear module
    nn::Linear linear;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor({8, 4}, DType::Float32);
    state["bias"] = Tensor({8}, DType::Float32);

    float *w = state["weight"].typed_data<float>();
    for (int i = 0; i < 32; i++)
        w[i] = static_cast<float>(i) * 0.01f;
    float *b = state["bias"].typed_data<float>();
    for (int i = 0; i < 8; i++)
        b[i] = static_cast<float>(i) * 0.1f;

    linear.load_state_dict(state);

    // Get CPU reference output
    auto input = Tensor({2, 4}, DType::Float32);
    float *in_data = input.typed_data<float>();
    for (int i = 0; i < 8; i++)
        in_data[i] = static_cast<float>(i + 1);

    auto cpu_output = linear(input);

    // Move to ANE and run
    linear.to(Device::ANE);
    EXPECT_EQ(linear.device(), Device::ANE);

    auto ane_output = linear(input); // operator() routes to ANE

    // Compare outputs
    ASSERT_EQ(ane_output.shape(), cpu_output.shape());
    float *ane_data = ane_output.typed_data<float>();
    float *cpu_data = cpu_output.typed_data<float>();

    std::cout << "[INFO] CPU output:  ";
    for (size_t i = 0; i < cpu_output.size(); i++)
        std::cout << cpu_data[i] << " ";
    std::cout << "\n[INFO] ANE output:  ";
    for (size_t i = 0; i < ane_output.size(); i++)
        std::cout << ane_data[i] << " ";
    std::cout << "\n";

    for (size_t i = 0; i < ane_output.size(); i++) {
        EXPECT_NEAR(ane_data[i], cpu_data[i], 0.15f)
            << "Mismatch at index " << i;
    }
}

TEST(ANEIntegration, ReLUToANE) {
    SKIP_IF_NO_ANE();

    nn::ReLU relu;
    relu.to(Device::ANE);

    auto input = Tensor({2, 4}, DType::Float32);
    float *data = input.typed_data<float>();
    data[0] = -1.0f; data[1] = 0.0f; data[2] = 1.0f; data[3] = 2.0f;
    data[4] = -3.0f; data[5] = -0.5f; data[6] = 0.5f; data[7] = 3.0f;

    auto output = relu(input); // Should run on ANE

    float *out = output.typed_data<float>();
    EXPECT_NEAR(out[0], 0.0f, 0.01f);
    EXPECT_NEAR(out[1], 0.0f, 0.01f);
    EXPECT_NEAR(out[2], 1.0f, 0.01f);
    EXPECT_NEAR(out[3], 2.0f, 0.01f);
    EXPECT_NEAR(out[4], 0.0f, 0.01f);
    EXPECT_NEAR(out[5], 0.0f, 0.01f);
    EXPECT_NEAR(out[6], 0.5f, 0.01f);
    EXPECT_NEAR(out[7], 3.0f, 0.01f);
}

TEST(ANEIntegration, LazyCompileOnFirstCall) {
    SKIP_IF_NO_ANE();

    nn::ReLU relu;
    int count_before_to = ane_compile_count();

    relu.to(Device::ANE);

    // No compilation yet — lazily compiled on first forward
    EXPECT_EQ(ane_compile_count(), count_before_to);

    auto input = Tensor({2, 4}, DType::Float32);
    float *data = input.typed_data<float>();
    for (int i = 0; i < 8; i++)
        data[i] = 1.0f;

    auto output = relu(input); // First call triggers compilation (or fallback)
    int count_after_first = ane_compile_count();

    // Output should be correct regardless (ANE or CPU fallback)
    EXPECT_NEAR(output.typed_data<float>()[0], 1.0f, 0.01f);

    // Second call with same shape should NOT increase compile count
    auto output2 = relu(input);
    EXPECT_EQ(ane_compile_count(), count_after_first);
}

TEST(ANEIntegration, RecompileOnShapeChange) {
    SKIP_IF_NO_ANE();

    nn::ReLU relu;
    relu.to(Device::ANE);

    // First shape
    auto input1 = Tensor({2, 4}, DType::Float32);
    float *d1 = input1.typed_data<float>();
    for (int i = 0; i < 8; i++) d1[i] = 1.0f;
    auto out1 = relu(input1);
    ASSERT_EQ(out1.shape(), Shape({2, 4}));
    EXPECT_NEAR(out1.typed_data<float>()[0], 1.0f, 0.01f);

    // Same shape — cached model is reused (no recompile)
    int count_before = ane_compile_count();
    auto out1b = relu(input1);
    EXPECT_EQ(ane_compile_count(), count_before);

    // Different shape — should work (via recompile or fallback)
    auto input2 = Tensor({4, 4}, DType::Float32);
    float *d2 = input2.typed_data<float>();
    for (int i = 0; i < 16; i++) d2[i] = 2.0f;
    auto out2 = relu(input2);
    ASSERT_EQ(out2.shape(), Shape({4, 4}));
    EXPECT_NEAR(out2.typed_data<float>()[0], 2.0f, 0.01f);
}

TEST(ANEIntegration, FallbackOnUnsupported) {
    SKIP_IF_NO_ANE();

    // LSTM is not ANE-supported — should fall back to CPU gracefully
    nn::LSTM lstm;
    // Just verify the to() call doesn't crash
    lstm.to(Device::ANE);
    EXPECT_EQ(lstm.device(), Device::ANE);
    // forward() would fall back to CPU (tested implicitly)
}

TEST(ANEIntegration, WeightUpdateInvalidatesCache) {
    SKIP_IF_NO_ANE();

    nn::Linear linear;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor({4, 4}, DType::Float32);
    state["bias"] = Tensor({4}, DType::Float32);

    float *w = state["weight"].typed_data<float>();
    for (int i = 0; i < 16; i++)
        w[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    float *b = state["bias"].typed_data<float>();
    for (int i = 0; i < 4; i++)
        b[i] = 0.0f;

    linear.load_state_dict(state);
    linear.to(Device::ANE);

    auto input = Tensor({2, 4}, DType::Float32);
    float *in_data = input.typed_data<float>();
    for (int i = 0; i < 8; i++)
        in_data[i] = 1.0f;

    int initial_count = ane_compile_count();
    auto output1 = linear(input);
    EXPECT_EQ(ane_compile_count(), initial_count + 1);

    // Update weights — should invalidate cache
    state["weight"] = Tensor({4, 4}, DType::Float32);
    w = state["weight"].typed_data<float>();
    for (int i = 0; i < 16; i++)
        w[i] = 0.5f; // All 0.5 now
    linear.load_state_dict(state);

    // Next forward should recompile with new weights
    auto output2 = linear(input);
    EXPECT_EQ(ane_compile_count(), initial_count + 2);

    // Outputs should differ (different weights)
    float *o1 = output1.typed_data<float>();
    float *o2 = output2.typed_data<float>();
    bool any_differ = false;
    for (size_t i = 0; i < output1.size(); i++) {
        if (std::abs(o1[i] - o2[i]) > 0.01f) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ) << "Outputs should differ after weight update";
}

TEST(ANEIntegration, MoveBackToCPU) {
    SKIP_IF_NO_ANE();

    nn::ReLU relu;
    relu.to(Device::ANE);

    auto input = Tensor({2, 4}, DType::Float32);
    float *data = input.typed_data<float>();
    for (int i = 0; i < 8; i++)
        data[i] = static_cast<float>(i) - 3.0f;

    auto ane_output = relu(input);

    // Move back to CPU
    relu.to(Device::CPU);
    EXPECT_EQ(relu.device(), Device::CPU);

    auto cpu_output = relu(input);

    // Both should produce same results
    ASSERT_EQ(ane_output.shape(), cpu_output.shape());
    float *ane = ane_output.typed_data<float>();
    float *cpu = cpu_output.typed_data<float>();
    for (size_t i = 0; i < ane_output.size(); i++) {
        EXPECT_NEAR(ane[i], cpu[i], 0.01f);
    }
}

#else // !AXIOM_HAS_ANE

TEST(ANEIntegration, Skipped) { GTEST_SKIP() << "ANE not compiled"; }

#endif

} // namespace testing
} // namespace axiom
