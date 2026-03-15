#include "axiom_test_utils.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>

#include "axiom/nn/container.hpp"
#include "axiom/nn/rnn.hpp"

#ifdef AXIOM_HAS_ANE
#include "axiom/nn/ane_compiled_model.hpp"
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

// ============================================================================
// Sequential mega-kernel tests
// ============================================================================

TEST(ANEIntegration, SequentialMegaKernel) {
    SKIP_IF_NO_ANE();

    // Build a mini-FFN: Linear(4,8) → ReLU → Linear(8,4)
    nn::Sequential ffn;
    auto &fc1 = ffn.emplace_back<nn::Linear>();
    ffn.emplace_back<nn::ReLU>();
    auto &fc2 = ffn.emplace_back<nn::Linear>();

    // Load weights
    std::map<std::string, Tensor> state;
    state["0.weight"] = Tensor({8, 4}, DType::Float32);
    state["0.bias"] = Tensor({8}, DType::Float32);
    state["2.weight"] = Tensor({4, 8}, DType::Float32);
    state["2.bias"] = Tensor({4}, DType::Float32);

    // Initialize to small values
    for (auto &[k, v] : state) {
        float *d = v.typed_data<float>();
        for (size_t i = 0; i < v.size(); i++)
            d[i] = static_cast<float>(i % 7) * 0.02f;
    }
    ffn.load_state_dict(state);

    // CPU reference
    auto input = Tensor({2, 4}, DType::Float32);
    float *in_data = input.typed_data<float>();
    for (int i = 0; i < 8; i++)
        in_data[i] = static_cast<float>(i + 1) * 0.5f;

    auto cpu_output = ffn(input);

    // ANE
    ffn.to(Device::ANE);
    EXPECT_TRUE(backends::ane::ANECompiledModel::is_supported(ffn));

    auto ane_output = ffn(input);

    // Verify
    ASSERT_EQ(ane_output.shape(), cpu_output.shape());
    float *ane = ane_output.typed_data<float>();
    float *cpu = cpu_output.typed_data<float>();

    std::cout << "[INFO] Sequential CPU: ";
    for (size_t i = 0; i < cpu_output.size(); i++)
        std::cout << cpu[i] << " ";
    std::cout << "\n[INFO] Sequential ANE: ";
    for (size_t i = 0; i < ane_output.size(); i++)
        std::cout << ane[i] << " ";
    std::cout << "\n";

    for (size_t i = 0; i < ane_output.size(); i++) {
        EXPECT_NEAR(ane[i], cpu[i], 0.2f)
            << "Mismatch at index " << i;
    }
}

TEST(ANEIntegration, SequentialCompileCount) {
    SKIP_IF_NO_ANE();

    // A Sequential with 3 layers should compile as ONE ANE graph
    nn::Sequential net;
    net.emplace_back<nn::Linear>();
    net.emplace_back<nn::SiLU>();
    net.emplace_back<nn::Linear>();

    std::map<std::string, Tensor> state;
    state["0.weight"] = Tensor({8, 4}, DType::Float32);
    state["0.bias"] = Tensor({8}, DType::Float32);
    state["2.weight"] = Tensor({4, 8}, DType::Float32);
    state["2.bias"] = Tensor({4}, DType::Float32);
    for (auto &[k, v] : state) {
        float *d = v.typed_data<float>();
        for (size_t i = 0; i < v.size(); i++)
            d[i] = 0.01f;
    }
    net.load_state_dict(state);

    int count_before = ane_compile_count();
    auto compiled =
        backends::ane::ANECompiledModel::compile(net, {2, 4});
    // Should be exactly 1 compilation (mega-kernel)
    EXPECT_EQ(ane_compile_count(), count_before + 1);

    auto info = compiled.plan_info();
    EXPECT_EQ(info.ane_steps, 1);
    std::cout << "[INFO] " << info.summary << "\n";
}

// ============================================================================
// Benchmark: ANE vs CPU
// ============================================================================

TEST(ANEIntegration, BenchmarkLinear) {
    SKIP_IF_NO_ANE();

    // Linear(256, 512) — a realistic layer size
    nn::Linear linear;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::randn({512, 256});
    state["bias"] = Tensor::randn({512});
    linear.load_state_dict(state);

    auto input = Tensor::randn({16, 256}); // batch=16

    // Warm up CPU
    auto cpu_out = linear(input);

    // CPU benchmark
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        auto out = linear.forward(input);
        (void)out.typed_data<float>(); // Force materialization
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(
                        cpu_end - cpu_start).count() / 100.0;

    // ANE benchmark
    linear.to(Device::ANE);
    auto ane_warmup = linear(input); // First call compiles

    auto ane_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        auto out = linear(input);
        (void)out.typed_data<float>();
    }
    auto ane_end = std::chrono::high_resolution_clock::now();
    double ane_us = std::chrono::duration<double, std::micro>(
                        ane_end - ane_start).count() / 100.0;

    std::cout << "[BENCH] Linear(256→512) x16:\n"
              << "  CPU: " << cpu_us << " µs/forward\n"
              << "  ANE: " << ane_us << " µs/forward\n"
              << "  Ratio: " << cpu_us / ane_us << "x\n";

    // Verify correctness
    linear.to(Device::CPU);
    auto ref = linear(input);
    linear.to(Device::ANE);
    auto ane_check = linear(input);
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(ref.typed_data<float>()[i],
                    ane_check.typed_data<float>()[i], 0.5f);
    }
}

TEST(ANEIntegration, BenchmarkLargeLinear) {
    SKIP_IF_NO_ANE();

    // Linear(1024, 1024) — larger layer
    nn::Linear linear;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::randn({1024, 1024});
    state["bias"] = Tensor::randn({1024});
    linear.load_state_dict(state);

    auto input = Tensor::randn({64, 1024}); // batch=64

    auto cpu_warmup = linear(input);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        auto out = linear.forward(input);
        (void)out.typed_data<float>();
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(
                        cpu_end - cpu_start).count() / 50.0;

    linear.to(Device::ANE);
    auto ane_warmup = linear(input);

    auto ane_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        auto out = linear(input);
        (void)out.typed_data<float>();
    }
    auto ane_end = std::chrono::high_resolution_clock::now();
    double ane_us = std::chrono::duration<double, std::micro>(
                        ane_end - ane_start).count() / 50.0;

    std::cout << "[BENCH] Linear(1024→1024) x64:\n"
              << "  CPU: " << cpu_us << " µs/forward\n"
              << "  ANE: " << ane_us << " µs/forward\n"
              << "  Ratio: " << cpu_us / ane_us << "x\n";
}

TEST(ANEIntegration, BenchmarkSequentialFFN) {
    SKIP_IF_NO_ANE();

    // FFN: Linear(256,1024) → SiLU → Linear(1024,256)
    nn::Sequential ffn;
    ffn.emplace_back<nn::Linear>();
    ffn.emplace_back<nn::SiLU>();
    ffn.emplace_back<nn::Linear>();

    std::map<std::string, Tensor> state;
    state["0.weight"] = Tensor::randn({1024, 256});
    state["0.bias"] = Tensor::randn({1024});
    state["2.weight"] = Tensor::randn({256, 1024});
    state["2.bias"] = Tensor::randn({256});
    ffn.load_state_dict(state);

    auto input = Tensor::randn({16, 256});

    // CPU benchmark
    auto cpu_warmup = ffn(input);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        auto out = ffn.forward(input);
        (void)out.typed_data<float>();
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(
                        cpu_end - cpu_start).count() / 50.0;

    // ANE benchmark
    ffn.to(Device::ANE);
    auto ane_warmup = ffn(input);

    auto ane_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        auto out = ffn(input);
        (void)out.typed_data<float>();
    }
    auto ane_end = std::chrono::high_resolution_clock::now();
    double ane_us = std::chrono::duration<double, std::micro>(
                        ane_end - ane_start).count() / 50.0;

    std::cout << "[BENCH] FFN(256→1024→256) x16:\n"
              << "  CPU: " << cpu_us << " µs/forward\n"
              << "  ANE: " << ane_us << " µs/forward\n"
              << "  Ratio: " << cpu_us / ane_us << "x\n";
}

#else // !AXIOM_HAS_ANE

TEST(ANEIntegration, Skipped) { GTEST_SKIP() << "ANE not compiled"; }

#endif

} // namespace testing
} // namespace axiom
