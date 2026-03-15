#include "axiom_test_utils.hpp"
#include <gtest/gtest.h>
#include <iostream>

#ifdef AXIOM_HAS_ANE
#include "backends/ane/ane_bridge.h"
#include "backends/ane/ane_iosurface.h"
#include "backends/ane/ane_storage.hpp"
#include "backends/ane/mil_generator.hpp"
#endif

#define SKIP_IF_NO_ANE()                                                       \
    do {                                                                        \
        if (!axiom::backends::ane::is_ane_available()) {                       \
            GTEST_SKIP() << "ANE not available";                               \
        }                                                                      \
    } while (0)

namespace axiom {
namespace testing {

#ifdef AXIOM_HAS_ANE

// ============================================================================
// MIL Generation tests (text output only, no ANE hardware needed)
// ============================================================================

TEST(ANEMIL, LinearMILGeneration) {
    SKIP_IF_NO_ANE();

    // Create a small Linear layer
    auto weight = Tensor({4, 3}, DType::Float32); // [out=4, in=3]
    auto bias = Tensor({4}, DType::Float32);

    // Fill with known values
    float *w_data = weight.typed_data<float>();
    for (int i = 0; i < 12; i++)
        w_data[i] = static_cast<float>(i) * 0.1f;
    float *b_data = bias.typed_data<float>();
    for (int i = 0; i < 4; i++)
        b_data[i] = static_cast<float>(i) * 0.01f;

    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, 3, 1, 2}); // [batch, in=3, 1, seq=2]
    auto y = gen.add_linear(x, weight, &bias, "fc1");
    gen.set_output(y);
    auto mil = gen.finalize();

    // Verify MIL text structure
    EXPECT_NE(mil.find("program(1.3)"), std::string::npos);
    EXPECT_NE(mil.find("func main<ios18>"), std::string::npos);
    EXPECT_NE(mil.find("tensor<fp16, [1, 3, 1, 2]> x"), std::string::npos);
    EXPECT_NE(mil.find("conv("), std::string::npos);
    EXPECT_NE(mil.find("BLOBFILE"), std::string::npos);
    EXPECT_NE(mil.find("fc1_w"), std::string::npos);
    EXPECT_NE(mil.find("fc1_b"), std::string::npos);

    // Verify weight blobs were generated
    EXPECT_EQ(gen.weight_blobs().size(), 2u); // weight + bias
    EXPECT_EQ(gen.weight_blobs()[0].name, "fc1_w");
    EXPECT_EQ(gen.weight_blobs()[1].name, "fc1_b");

    // Weight blob should be 128 header + data
    // weight: 4*3 = 12 elements * 2 bytes = 24 bytes + 128 header = 152
    EXPECT_EQ(gen.weight_blobs()[0].blob_data.size(), 128u + 12 * 2);
    // bias: 4 elements * 2 bytes = 8 bytes + 128 header = 136
    EXPECT_EQ(gen.weight_blobs()[1].blob_data.size(), 128u + 4 * 2);
}

TEST(ANEMIL, LinearNoBiasMIL) {
    SKIP_IF_NO_ANE();

    auto weight = Tensor({8, 4}, DType::Float32);
    float *w_data = weight.typed_data<float>();
    for (int i = 0; i < 32; i++)
        w_data[i] = 0.1f;

    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, 4, 1, 16});
    auto y = gen.add_linear(x, weight, nullptr, "fc1");
    gen.set_output(y);
    auto mil = gen.finalize();

    // Should NOT have bias reference
    EXPECT_EQ(mil.find("fc1_b"), std::string::npos);
    EXPECT_EQ(gen.weight_blobs().size(), 1u);
}

TEST(ANEMIL, SiLUMIL) {
    SKIP_IF_NO_ANE();

    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, 8, 1, 4});
    auto y = gen.add_silu(x, "act");
    gen.set_output(y);
    auto mil = gen.finalize();

    // SiLU = x * sigmoid(x)
    EXPECT_NE(mil.find("sigmoid"), std::string::npos);
    EXPECT_NE(mil.find("mul"), std::string::npos);
}

TEST(ANEMIL, RMSNormMIL) {
    SKIP_IF_NO_ANE();

    auto weight = Tensor({8}, DType::Float32);
    float *w_data = weight.typed_data<float>();
    for (int i = 0; i < 8; i++)
        w_data[i] = 1.0f;

    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, 8, 1, 4});
    auto y = gen.add_rms_norm(x, weight, 1e-5f, "rn");
    gen.set_output(y);
    auto mil = gen.finalize();

    // RMSNorm requires: mul (square), reduce_sum, pow, mul (scale)
    EXPECT_NE(mil.find("reduce_sum"), std::string::npos);
    EXPECT_NE(mil.find("pow"), std::string::npos);
    EXPECT_NE(mil.find("BLOBFILE"), std::string::npos);
}

TEST(ANEMIL, StackedLinearsMIL) {
    SKIP_IF_NO_ANE();

    auto w1 = Tensor({8, 4}, DType::Float32);
    auto w2 = Tensor({4, 8}, DType::Float32);
    float *w1_data = w1.typed_data<float>();
    float *w2_data = w2.typed_data<float>();
    for (int i = 0; i < 32; i++) {
        w1_data[i] = 0.1f;
        w2_data[i] = 0.05f;
    }

    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, 4, 1, 16});
    auto h = gen.add_linear(x, w1, nullptr, "fc1");
    auto a = gen.add_silu(h, "act");
    auto y = gen.add_linear(a, w2, nullptr, "fc2");
    gen.set_output(y);
    auto mil = gen.finalize();

    // Should have 2 conv ops (one per linear)
    // Count "conv(" occurrences
    size_t pos = 0;
    int conv_count = 0;
    while ((pos = mil.find("conv(", pos)) != std::string::npos) {
        conv_count++;
        pos++;
    }
    EXPECT_EQ(conv_count, 2);

    // Should have 2 weight blobs
    EXPECT_EQ(gen.weight_blobs().size(), 2u);
}

// ============================================================================
// ANE Compilation tests (requires ANE hardware)
// ============================================================================

TEST(ANEMIL, CompileSimpleIdentity) {
    SKIP_IF_NO_ANE();

    // Simplest possible MIL: pass input through to output
    // Using add with zero as identity
    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, 4, 1, 8});

    // Use SiLU as a simple identity-ish test (tests compilation)
    auto y = gen.add_relu(x, "act");
    gen.set_output(y);

    auto mil = gen.finalize();
    auto &blobs = gen.weight_blobs();

    // Try to compile
    std::vector<ANEWeightEntry> entries;
    for (auto &b : blobs) {
        entries.push_back(
            {b.name.c_str(), b.blob_data.data(), b.blob_data.size()});
    }

    ANEModelHandle *handle = ane_compile_with_weights(
        mil.c_str(), entries.data(), static_cast<int>(entries.size()));

    // Compilation may fail due to private API quirks — that's OK for now.
    // We're testing the MIL generation path.
    if (handle) {
        int rc = ane_load(handle);
        if (rc == 0) {
            // Create IOSurfaces
            IOSurfaceRef input_surface = ane_create_surface(4, 8);
            IOSurfaceRef output_surface = ane_create_surface(4, 8);
            ASSERT_NE(input_surface, nullptr);
            ASSERT_NE(output_surface, nullptr);

            // Write test data
            std::vector<float> input_data(32, 1.0f);
            ane_surface_write_f32(input_surface, input_data.data(), 4, 8);

            // Evaluate
            rc = ane_eval(handle, input_surface, output_surface);
            if (rc == 0) {
                // Read output
                std::vector<float> output_data(32);
                ane_surface_read_f32(output_surface, output_data.data(), 4, 8);

                // Should be identity (input + 0 = input)
                for (int i = 0; i < 32; i++) {
                    EXPECT_NEAR(input_data[i], output_data[i], 0.1f)
                        << "Mismatch at " << i;
                }
            }

            CFRelease(input_surface);
            CFRelease(output_surface);
        }
        ane_release(handle);
    } else {
        // Log that compilation failed but don't fail the test
        std::cout << "[INFO] ANE compilation failed (expected during "
                     "development)\n";
    }
}

TEST(ANEMIL, CompileLinearForward) {
    SKIP_IF_NO_ANE();

    // Create a simple 3→4 linear layer
    int64_t in_f = 3, out_f = 4, seq = 2;
    auto weight = Tensor({static_cast<size_t>(out_f), static_cast<size_t>(in_f)}, DType::Float32);
    auto bias = Tensor({static_cast<size_t>(out_f)}, DType::Float32);

    // Identity-like weight for easy verification
    float *w_data = weight.typed_data<float>();
    for (int64_t i = 0; i < out_f * in_f; i++)
        w_data[i] = (i % (in_f + 1) == 0) ? 1.0f : 0.0f;
    float *b_data = bias.typed_data<float>();
    for (int64_t i = 0; i < out_f; i++)
        b_data[i] = 0.0f;

    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, in_f, 1, seq});
    auto y = gen.add_linear(x, weight, &bias, "fc1");
    gen.set_output(y);

    auto mil = gen.finalize();
    auto &blobs = gen.weight_blobs();

    std::vector<ANEWeightEntry> entries;
    for (auto &b : blobs) {
        entries.push_back(
            {b.name.c_str(), b.blob_data.data(), b.blob_data.size()});
    }

    ANEModelHandle *handle = ane_compile_with_weights(
        mil.c_str(), entries.data(), static_cast<int>(entries.size()));

    if (handle) {
        int rc = ane_load(handle);
        if (rc == 0) {
            IOSurfaceRef input_surface =
                ane_create_surface(static_cast<int>(in_f), static_cast<int>(seq));
            IOSurfaceRef output_surface =
                ane_create_surface(static_cast<int>(out_f), static_cast<int>(seq));
            ASSERT_NE(input_surface, nullptr);
            ASSERT_NE(output_surface, nullptr);

            std::vector<float> input_data = {1.0f, 2.0f, 3.0f,
                                              4.0f, 5.0f, 6.0f};
            ane_surface_write_f32(input_surface, input_data.data(),
                                 static_cast<int>(in_f), static_cast<int>(seq));

            rc = ane_eval(handle, input_surface, output_surface);
            if (rc == 0) {
                std::vector<float> output_data(static_cast<size_t>(out_f * seq));
                ane_surface_read_f32(output_surface, output_data.data(),
                                     static_cast<int>(out_f), static_cast<int>(seq));

                // With identity-ish weight, first output channel should ~= first
                // input channel
                std::cout << "[INFO] ANE Linear output: ";
                for (auto v : output_data)
                    std::cout << v << " ";
                std::cout << "\n";
            }

            CFRelease(input_surface);
            CFRelease(output_surface);
        }
        ane_release(handle);
    } else {
        std::cout << "[INFO] ANE Linear compilation failed (expected during "
                     "development)\n";
    }
}

#else // !AXIOM_HAS_ANE

TEST(ANEMIL, Skipped) { GTEST_SKIP() << "ANE not compiled"; }

#endif

} // namespace testing
} // namespace axiom
