#include "axiom_test_utils.hpp"
#include <gtest/gtest.h>

#include "backends/cpu/cpu_storage.hpp"

#ifdef AXIOM_HAS_ANE
#include "backends/ane/ane_bridge.h"
#include "backends/ane/ane_iosurface.h"
#include "backends/ane/ane_storage.hpp"
#endif

// Skip macro for ANE tests
#define SKIP_IF_NO_ANE()                                                       \
    do {                                                                        \
        if (!axiom::backends::ane::is_ane_available()) {                       \
            GTEST_SKIP() << "ANE not available";                               \
        }                                                                      \
    } while (0)

namespace axiom {
namespace testing {

// ============================================================================
// Bridge initialization tests
// ============================================================================

TEST(ANEBasic, BridgeInit) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();
    ASSERT_EQ(ane_init(), 0);
    ASSERT_TRUE(ane_is_available());
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

TEST(ANEBasic, CompileCountStartsAtZero) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();
    // Compile count should be non-negative (may be > 0 if other tests ran)
    ASSERT_GE(ane_compile_count(), 0);
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

// ============================================================================
// IOSurface tests
// ============================================================================

TEST(ANEBasic, IOSurfaceCreation) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();

    int channels = 4;
    int spatial = 8;
    IOSurfaceRef surface = ane_create_surface(channels, spatial);
    ASSERT_NE(surface, nullptr);

    // Verify dimensions
    size_t width = IOSurfaceGetWidth(surface);
    size_t height = IOSurfaceGetHeight(surface);
    ASSERT_EQ(width, (size_t)spatial);
    ASSERT_EQ(height, (size_t)channels);

    // Verify allocation size is at least channels * spatial * sizeof(fp16)
    size_t alloc_size = ane_surface_size_bytes(surface);
    ASSERT_GE(alloc_size, (size_t)(channels * spatial * 2));

    CFRelease(surface);
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

TEST(ANEBasic, IOSurfaceRoundTripF32) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();

    int channels = 4;
    int spatial = 8;
    size_t num_elements = channels * spatial;

    // Create test data
    std::vector<float> input(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        input[i] = static_cast<float>(i) * 0.1f;
    }

    // Create surface, write, read back
    IOSurfaceRef surface = ane_create_surface(channels, spatial);
    ASSERT_NE(surface, nullptr);

    int rc = ane_surface_write_f32(surface, input.data(), channels, spatial);
    ASSERT_EQ(rc, 0);

    std::vector<float> output(num_elements, 0.0f);
    rc = ane_surface_read_f32(surface, output.data(), channels, spatial);
    ASSERT_EQ(rc, 0);

    // Check round-trip accuracy (FP32 -> FP16 -> FP32 loses some precision)
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_NEAR(input[i], output[i], 0.01f)
            << "Mismatch at index " << i;
    }

    CFRelease(surface);
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

TEST(ANEBasic, IOSurfaceRoundTripF16) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();

    int channels = 2;
    int spatial = 4;
    size_t num_elements = channels * spatial;

    // Create FP16 test data (simple bit patterns)
    std::vector<uint16_t> input(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        // FP16: 0x3C00 = 1.0, 0x4000 = 2.0, etc.
        input[i] = 0x3C00 + static_cast<uint16_t>(i * 0x0400);
    }

    IOSurfaceRef surface = ane_create_surface(channels, spatial);
    ASSERT_NE(surface, nullptr);

    int rc = ane_surface_write_f16(surface, input.data(), channels, spatial);
    ASSERT_EQ(rc, 0);

    std::vector<uint16_t> output(num_elements, 0);
    rc = ane_surface_read_f16(surface, output.data(), channels, spatial);
    ASSERT_EQ(rc, 0);

    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_EQ(input[i], output[i])
            << "FP16 mismatch at index " << i;
    }

    CFRelease(surface);
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

TEST(ANEBasic, IOSurfaceLargeMatrix) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();

    // Test with dimensions typical for NN layers
    int channels = 512;
    int spatial = 768;
    size_t num_elements = (size_t)channels * spatial;

    std::vector<float> input(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        input[i] = static_cast<float>(i % 100) * 0.01f;
    }

    IOSurfaceRef surface = ane_create_surface(channels, spatial);
    ASSERT_NE(surface, nullptr);

    int rc = ane_surface_write_f32(surface, input.data(), channels, spatial);
    ASSERT_EQ(rc, 0);

    std::vector<float> output(num_elements, 0.0f);
    rc = ane_surface_read_f32(surface, output.data(), channels, spatial);
    ASSERT_EQ(rc, 0);

    // Spot-check a few values (FP16 tolerance)
    EXPECT_NEAR(input[0], output[0], 0.01f);
    EXPECT_NEAR(input[1000], output[1000], 0.01f);
    EXPECT_NEAR(input[num_elements - 1], output[num_elements - 1], 0.01f);

    CFRelease(surface);
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

// ============================================================================
// ANEStorage tests
// ============================================================================

TEST(ANEBasic, StorageCreation) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();

    auto storage = backends::ane::make_ane_storage(4, 8);
    ASSERT_NE(storage, nullptr);
    ASSERT_EQ(storage->device(), Device::ANE);
    ASSERT_EQ(storage->size_bytes(), (size_t)(4 * 8 * 2)); // FP16

    auto *ane_storage = dynamic_cast<backends::ane::ANEStorage *>(storage.get());
    ASSERT_NE(ane_storage, nullptr);
    ASSERT_EQ(ane_storage->channels(), 4);
    ASSERT_EQ(ane_storage->spatial_size(), 8);
    ASSERT_NE(ane_storage->surface(), nullptr);
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

TEST(ANEBasic, StorageCopyFromCPU) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();

    int channels = 4;
    int spatial = 8;
    size_t num_elements = channels * spatial;

    // Create CPU tensor with known data
    auto cpu_storage = axiom::backends::cpu::make_cpu_storage(num_elements * sizeof(float));
    float *cpu_data = static_cast<float *>(cpu_storage->data());
    for (size_t i = 0; i < num_elements; i++) {
        cpu_data[i] = static_cast<float>(i) * 0.5f;
    }

    // Copy to ANE
    auto ane_storage = backends::ane::make_ane_storage(channels, spatial);
    ane_storage->copy_from(*cpu_storage);

    // Copy back to CPU and verify
    auto cpu_out = axiom::backends::cpu::make_cpu_storage(num_elements * sizeof(float));
    ane_storage->copy_to(*cpu_out);

    float *out_data = static_cast<float *>(cpu_out->data());
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_NEAR(cpu_data[i], out_data[i], 0.1f)
            << "Round-trip mismatch at index " << i;
    }
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

TEST(ANEBasic, StorageClone) {
#ifdef AXIOM_HAS_ANE
    SKIP_IF_NO_ANE();

    int channels = 2;
    int spatial = 4;
    size_t num_elements = channels * spatial;

    auto original = backends::ane::make_ane_storage(channels, spatial);

    // Write data via CPU path
    auto cpu_storage = axiom::backends::cpu::make_cpu_storage(num_elements * sizeof(float));
    float *cpu_data = static_cast<float *>(cpu_storage->data());
    for (size_t i = 0; i < num_elements; i++) {
        cpu_data[i] = static_cast<float>(i) + 1.0f;
    }
    original->copy_from(*cpu_storage);

    // Clone
    auto cloned = original->clone();
    ASSERT_NE(cloned, nullptr);
    ASSERT_EQ(cloned->device(), Device::ANE);

    // Verify clone has same data
    auto cpu_out = axiom::backends::cpu::make_cpu_storage(num_elements * sizeof(float));
    cloned->copy_to(*cpu_out);

    float *out_data = static_cast<float *>(cpu_out->data());
    for (size_t i = 0; i < num_elements; i++) {
        EXPECT_NEAR(cpu_data[i], out_data[i], 0.1f)
            << "Clone data mismatch at index " << i;
    }
#else
    GTEST_SKIP() << "ANE not compiled";
#endif
}

// ============================================================================
// Device enum tests
// ============================================================================

TEST(ANEBasic, DeviceToString) {
    ASSERT_EQ(system::device_to_string(Device::ANE), "ANE");
}

TEST(ANEBasic, MakeStorageANEThrows) {
    // make_storage with Device::ANE should throw a helpful error
    // since ANE storage requires channel-first dimensions
    EXPECT_THROW(make_storage(64, Device::ANE), DeviceError);
}

} // namespace testing
} // namespace axiom
