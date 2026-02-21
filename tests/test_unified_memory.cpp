#include "axiom_test_utils.hpp"

#include <vector>

#ifdef AXIOM_METAL_SUPPORT
#include "backends/metal/unified_storage.hpp"
#endif

using namespace axiom;

// Helper: skip if unified memory is not available
#ifdef AXIOM_METAL_SUPPORT
#define SKIP_IF_NO_UNIFIED()                                                   \
    do {                                                                       \
        SKIP_IF_NO_GPU();                                                      \
        if (!backends::metal::is_unified_memory_available()) {                 \
            GTEST_SKIP() << "Unified memory not available";                    \
        }                                                                      \
    } while (0)
#else
#define SKIP_IF_NO_UNIFIED() GTEST_SKIP() << "Metal support not compiled"
#endif

// ============================================================================
// UnifiedStorage basics
// ============================================================================

TEST(UnifiedMemory, DataReturnsWritableCpuPointer) {
    SKIP_IF_NO_UNIFIED();

    auto t = Tensor::zeros({4, 4}, DType::Float32, Device::CPU);
    void *ptr = t.storage()->data();
    ASSERT_NE(ptr, nullptr);

    // Write through the CPU pointer and read back
    auto *f = static_cast<float *>(ptr);
    f[0] = 42.0f;
    EXPECT_FLOAT_EQ(t.item<float>({0, 0}), 42.0f);
}

TEST(UnifiedMemory, ZeroCopyToGpuAndBack) {
    SKIP_IF_NO_UNIFIED();

    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
    auto cpu_tensor = Tensor::from_data(vals.data(), {4});
    auto gpu_tensor = cpu_tensor.gpu();
    auto back_cpu = gpu_tensor.cpu();

    // Verify same underlying data pointer (zero-copy)
    EXPECT_EQ(cpu_tensor.storage()->data(), back_cpu.storage()->data());

    // Verify values are correct
    axiom::testing::ExpectTensorEquals<float>(back_cpu,
                                              {1.0f, 2.0f, 3.0f, 4.0f});
}

TEST(UnifiedMemory, GpuAddThenCpu) {
    SKIP_IF_NO_UNIFIED();

    std::vector<float> va = {1.0f, 2.0f, 3.0f};
    std::vector<float> vb = {10.0f, 20.0f, 30.0f};
    auto a = Tensor::from_data(va.data(), {3}).gpu();
    auto b = Tensor::from_data(vb.data(), {3}).gpu();
    auto result = ops::add(a, b);
    auto cpu_result = result.cpu();

    axiom::testing::ExpectTensorEquals<float>(cpu_result,
                                              {11.0f, 22.0f, 33.0f});
}

TEST(UnifiedMemory, CpuBlasMatmulOnUnifiedTensor) {
    SKIP_IF_NO_UNIFIED();

    // Create tensors that are backed by unified memory but tagged CPU
    auto a = Tensor::ones({2, 3}, DType::Float32, Device::CPU);
    auto b = Tensor::ones({3, 2}, DType::Float32, Device::CPU);
    auto result = ops::matmul(a, b);

    // Each element should be 3.0 (dot product of ones vectors of length 3)
    axiom::testing::ExpectTensorEquals<float>(result, {3, 3, 3, 3});
}

TEST(UnifiedMemory, SynchronizationCorrectness) {
    SKIP_IF_NO_UNIFIED();

    auto a = Tensor::randn({32, 32}, DType::Float32, Device::GPU);
    auto b = Tensor::randn({32, 32}, DType::Float32, Device::GPU);

    // Perform GPU computation
    auto gpu_result = ops::add(a, b);

    // Immediately read on CPU — synchronization must happen automatically
    auto cpu_result = gpu_result.cpu();
    ASSERT_NE(cpu_result.storage()->data(), nullptr);

    // Verify correctness against CPU reference
    auto a_cpu = a.cpu();
    auto b_cpu = b.cpu();
    auto ref = ops::add(a_cpu, b_cpu);

    EXPECT_TRUE(cpu_result.allclose(ref, 1e-5, 1e-5))
        << "Synchronization correctness failed";
}

TEST(UnifiedMemory, CloneCreatesIndependentCopy) {
    SKIP_IF_NO_UNIFIED();

    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
    auto original = Tensor::from_data(vals.data(), {4});
    auto cloned = original.clone();

    // Should have different underlying storage
    EXPECT_NE(original.storage()->data(), cloned.storage()->data());

    // But same values
    axiom::testing::ExpectTensorEquals<float>(cloned, {1.0f, 2.0f, 3.0f, 4.0f});

    // Mutating clone should not affect original
    static_cast<float *>(cloned.storage()->data())[0] = 99.0f;
    EXPECT_FLOAT_EQ(original.item<float>({0}), 1.0f);
}

TEST(UnifiedMemory, LazyEvalMaterializesIntoUnifiedStorage) {
    SKIP_IF_NO_UNIFIED();

    auto a = Tensor::ones({4, 4}, DType::Float32, Device::GPU);
    auto b = Tensor::ones({4, 4}, DType::Float32, Device::GPU);

    // GPU add — result should be in unified storage
    auto result = ops::add(a, b);
    auto cpu_result = result.cpu();

    // All elements should be 2.0
    auto *data = static_cast<const float *>(cpu_result.storage()->data());
    ASSERT_NE(data, nullptr);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f) << "at index " << i;
    }
}

TEST(UnifiedMemory, RoundTripPreservesValues) {
    SKIP_IF_NO_UNIFIED();

    std::vector<float> values = {1.5f, -2.3f, 0.0f, 42.0f, -100.0f, 3.14f};
    auto t = Tensor::from_data(values.data(), {6});

    // CPU -> GPU -> CPU round trip
    auto gpu = t.gpu();
    auto back = gpu.cpu();

    axiom::testing::ExpectTensorEquals<float>(back, values);
}

TEST(UnifiedMemory, DeviceTagCorrectness) {
    SKIP_IF_NO_UNIFIED();

    auto cpu_tensor = Tensor::zeros({2, 2}, DType::Float32, Device::CPU);
    EXPECT_EQ(cpu_tensor.device(), Device::CPU);

    auto gpu_tensor = cpu_tensor.gpu();
    EXPECT_EQ(gpu_tensor.device(), Device::GPU);

    auto back_cpu = gpu_tensor.cpu();
    EXPECT_EQ(back_cpu.device(), Device::CPU);
}
