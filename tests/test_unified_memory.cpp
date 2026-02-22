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

    // GPU tensors use UnifiedStorage on Apple Silicon
    auto t = Tensor::zeros({4, 4}, DType::Float32, Device::GPU);
    auto cpu_view = t.cpu(); // zero-copy: still backed by UnifiedStorage
    void *ptr = cpu_view.storage()->data();
    ASSERT_NE(ptr, nullptr);

    // Write through the CPU pointer and read back
    auto *f = static_cast<float *>(ptr);
    f[0] = 42.0f;
    EXPECT_FLOAT_EQ(cpu_view.item<float>({0, 0}), 42.0f);
}

TEST(UnifiedMemory, ZeroCopyGpuToCpu) {
    SKIP_IF_NO_UNIFIED();

    // Create on GPU (uses UnifiedStorage), move to CPU — should be zero-copy
    auto gpu_tensor = Tensor::ones({4}, DType::Float32, Device::GPU);
    auto cpu_view = gpu_tensor.cpu();

    // Both should be backed by UnifiedStorage (same underlying MTLBuffer)
#ifdef AXIOM_METAL_SUPPORT
    auto *gpu_unified = dynamic_cast<backends::metal::UnifiedStorage *>(
        gpu_tensor.storage().get());
    auto *cpu_unified = dynamic_cast<backends::metal::UnifiedStorage *>(
        cpu_view.storage().get());
    ASSERT_NE(gpu_unified, nullptr);
    ASSERT_NE(cpu_unified, nullptr);
    // Same underlying data pointer confirms zero-copy
    EXPECT_EQ(gpu_unified->data(), cpu_unified->data());
#endif

    // Verify values are correct
    axiom::testing::ExpectTensorEquals<float>(cpu_view,
                                              {1.0f, 1.0f, 1.0f, 1.0f});
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

    // Create GPU tensors (uses UnifiedStorage), move to CPU (stays unified),
    // then verify that CPU BLAS matmul works on unified-backed tensors
    auto a = Tensor::ones({2, 3}, DType::Float32, Device::GPU).cpu();
    auto b = Tensor::ones({3, 2}, DType::Float32, Device::GPU).cpu();

#ifdef AXIOM_METAL_SUPPORT
    // Confirm they are still backed by UnifiedStorage
    ASSERT_NE(
        dynamic_cast<backends::metal::UnifiedStorage *>(a.storage().get()),
        nullptr);
#endif

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

    // Create a GPU tensor (UnifiedStorage), move to CPU (stays unified), clone
    auto original = Tensor::ones({4}, DType::Float32, Device::GPU).cpu();
    auto cloned = original.clone();

    // Should have different underlying storage
    EXPECT_NE(original.storage()->data(), cloned.storage()->data());

    // But same values
    axiom::testing::ExpectTensorEquals<float>(cloned, {1.0f, 1.0f, 1.0f, 1.0f});

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

    // GPU->CPU is zero-copy (unified), verify still backed by UnifiedStorage
#ifdef AXIOM_METAL_SUPPORT
    ASSERT_NE(
        dynamic_cast<backends::metal::UnifiedStorage *>(back.storage().get()),
        nullptr);
#endif

    axiom::testing::ExpectTensorEquals<float>(back, values);
}

TEST(UnifiedMemory, DeviceTagCorrectness) {
    SKIP_IF_NO_UNIFIED();

    auto gpu_tensor = Tensor::zeros({2, 2}, DType::Float32, Device::GPU);
    EXPECT_EQ(gpu_tensor.device(), Device::GPU);

    auto cpu_view = gpu_tensor.cpu();
    EXPECT_EQ(cpu_view.device(), Device::CPU);

    auto back_gpu = cpu_view.gpu();
    EXPECT_EQ(back_gpu.device(), Device::GPU);
}
