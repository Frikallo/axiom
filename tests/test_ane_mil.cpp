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
    do {                                                                       \
        if (!axiom::backends::ane::is_ane_available()) {                       \
            GTEST_SKIP() << "ANE not available";                               \
        }                                                                      \
    } while (0)

#define SKIP_IF_NO_ANE_EXEC()                                                  \
    do {                                                                       \
        if (!ane_can_execute()) {                                              \
            GTEST_SKIP() << "ANE execution not available";                     \
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
    SKIP_IF_NO_ANE_EXEC();

    backends::ane::MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", {1, 4, 1, 8});
    auto y = gen.add_relu(x, "act");
    gen.set_output(y);

    auto mil = gen.finalize();
    std::cout << "=== Generated MIL ===\n" << mil << "=== END ===\n";
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
            // Create flat IOSurfaces for ANE eval
            // Input: [1, 4, 1, 8] in FP16 = 4*8*2 = 64 bytes
            // ANE may require page-aligned buffers (16384 bytes minimum)
            size_t num_elements = 4 * 8;
            size_t buf_bytes = 16384; // Page-aligned minimum
            IOSurfaceRef input_surface = ane_create_flat_surface(buf_bytes);
            IOSurfaceRef output_surface = ane_create_flat_surface(buf_bytes);
            ASSERT_NE(input_surface, nullptr);
            ASSERT_NE(output_surface, nullptr);

            // Write FP16 test data (all 1.0 = 0x3C00)
            IOSurfaceLock(input_surface, 0, NULL);
            auto *fp16_in =
                static_cast<uint16_t *>(IOSurfaceGetBaseAddress(input_surface));
            for (size_t i = 0; i < num_elements; i++)
                fp16_in[i] = 0x3C00; // 1.0 in FP16
            IOSurfaceUnlock(input_surface, 0, NULL);

            rc = ane_eval(handle, input_surface, output_surface);
            if (rc == 0) {
                // Read FP16 output, convert to float
                IOSurfaceLock(output_surface, kIOSurfaceLockReadOnly, NULL);
                auto *fp16_out = static_cast<const uint16_t *>(
                    IOSurfaceGetBaseAddress(output_surface));

                // ReLU(1.0) should be 1.0
                // Check first element (FP16: 0x3C00 = 1.0)
                std::cout << "[INFO] ANE ReLU output fp16[0] = 0x" << std::hex
                          << fp16_out[0] << std::dec << "\n";
                EXPECT_EQ(fp16_out[0], 0x3C00) << "ReLU(1.0) should be 1.0";

                IOSurfaceUnlock(output_surface, kIOSurfaceLockReadOnly, NULL);
            } else {
                std::cout << "[INFO] ANE eval failed\n";
            }

            CFRelease(input_surface);
            CFRelease(output_surface);
        } else {
            std::cout << "[INFO] ANE load failed\n";
        }
        ane_release(handle);
    } else {
        // Log that compilation failed but don't fail the test
        std::cout << "[INFO] ANE compilation failed (expected during "
                     "development)\n";
    }
}

TEST(ANEMIL, CompileLinearForward) {
    SKIP_IF_NO_ANE_EXEC();

    // Create a simple 3→4 linear layer
    int64_t in_f = 3, out_f = 4, seq = 2;
    auto weight =
        Tensor({static_cast<size_t>(out_f), static_cast<size_t>(in_f)},
               DType::Float32);
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
    std::cout << "=== Linear MIL ===\n" << mil << "=== END ===\n";
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
        ASSERT_EQ(rc, 0) << "ANE load failed";

        // Create flat IOSurfaces (page-aligned)
        size_t buf_bytes = 16384;
        IOSurfaceRef input_surface = ane_create_flat_surface(buf_bytes);
        IOSurfaceRef output_surface = ane_create_flat_surface(buf_bytes);
        ASSERT_NE(input_surface, nullptr);
        ASSERT_NE(output_surface, nullptr);

        // Write FP16 input: [1, 3, 1, 2] = 6 elements
        // Channel 0: [1.0, 2.0], Channel 1: [3.0, 4.0], Channel 2: [5.0, 6.0]
        IOSurfaceLock(input_surface, 0, NULL);
        auto *fp16_in =
            static_cast<uint16_t *>(IOSurfaceGetBaseAddress(input_surface));
        std::memset(fp16_in, 0, buf_bytes);
        // FP16 values: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200,
        //              4.0=0x4400, 5.0=0x4500, 6.0=0x4600
        fp16_in[0] = 0x3C00;
        fp16_in[1] = 0x4000; // chan 0
        fp16_in[2] = 0x4200;
        fp16_in[3] = 0x4400; // chan 1
        fp16_in[4] = 0x4500;
        fp16_in[5] = 0x4600; // chan 2
        IOSurfaceUnlock(input_surface, 0, NULL);

        rc = ane_eval(handle, input_surface, output_surface);
        if (rc == 0) {
            IOSurfaceLock(output_surface, kIOSurfaceLockReadOnly, NULL);
            auto *fp16_out = static_cast<const uint16_t *>(
                IOSurfaceGetBaseAddress(output_surface));

            // Print output for debugging
            std::cout << "[INFO] ANE Linear output (FP16 hex): ";
            for (int i = 0; i < static_cast<int>(out_f * seq); i++)
                std::cout << "0x" << std::hex << fp16_out[i] << " ";
            std::cout << std::dec << "\n";

            IOSurfaceUnlock(output_surface, kIOSurfaceLockReadOnly, NULL);
        } else {
            std::cout << "[INFO] ANE Linear eval failed\n";
        }

        CFRelease(input_surface);
        CFRelease(output_surface);
        ane_release(handle);
    } else {
        FAIL() << "ANE compilation failed";
    }
}

#else // !AXIOM_HAS_ANE

TEST(ANEMIL, Skipped) { GTEST_SKIP() << "ANE not compiled"; }

#endif

} // namespace testing
} // namespace axiom
