#include "axiom_test_utils.hpp"

#include "axiom/graph/graph_registry.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"
#include "axiom/tensor_operators.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

using namespace axiom;

// ============================================================================
// Dtype coverage: Float64 through lazy + fused paths
// ============================================================================

TEST(FusionDtype, Float64UnaryChain) {
    auto x = Tensor::full<double>({64, 64}, 4.0);
    auto result = ops::exp(ops::sqrt(x));
    double val = result.item<double>({0, 0});
    ASSERT_NEAR(val, std::exp(2.0), 1e-12);
}

TEST(FusionDtype, Float64BinaryFusion) {
    auto a = Tensor::full<double>({64, 64}, -1.0);
    auto b = Tensor::full<double>({64, 64}, 0.5);
    auto result = ops::relu(ops::add(a, b));
    double val = result.item<double>({0, 0});
    ASSERT_NEAR(val, 0.0, 1e-15);
}

TEST(FusionDtype, Float64EagerParity) {
    auto a = Tensor::full<double>({32, 32}, 0.7);
    auto b = Tensor::full<double>({32, 32}, 0.3);

    auto lazy_result = ops::sigmoid(ops::add(a, b));
    double lazy_val = lazy_result.item<double>({0, 0});

    {
        graph::EagerModeScope eager;
        auto ea = Tensor::full<double>({32, 32}, 0.7);
        auto eb = Tensor::full<double>({32, 32}, 0.3);
        auto eager_result = ops::sigmoid(ops::add(ea, eb));
        double eager_val = eager_result.item<double>({0, 0});
        ASSERT_NEAR(lazy_val, eager_val, 1e-12);
    }
}

// ============================================================================
// Dtype coverage: Int32 through lazy + fused paths
// ============================================================================

TEST(FusionDtype, Int32AddReLU) {
    auto a = Tensor::full<int32_t>({64, 64}, -3);
    auto b = Tensor::full<int32_t>({64, 64}, 1);
    auto result = ops::relu(ops::add(a, b));
    int32_t val = result.item<int32_t>({0, 0});
    ASSERT_EQ(val, 0); // relu(-2) = 0
}

TEST(FusionDtype, Int32MulAdd) {
    auto a = Tensor::full<int32_t>({64, 64}, 3);
    auto b = Tensor::full<int32_t>({64, 64}, 4);
    auto c = Tensor::full<int32_t>({64, 64}, 5);
    auto result = ops::add(ops::multiply(a, b), c);
    int32_t val = result.item<int32_t>({0, 0});
    ASSERT_EQ(val, 17); // 3*4 + 5 = 17
}

TEST(FusionDtype, Int32EagerParity) {
    auto a = Tensor::full<int32_t>({32, 32}, 10);
    auto b = Tensor::full<int32_t>({32, 32}, 3);

    auto lazy_result = ops::abs(ops::subtract(a, b));
    int32_t lazy_val = lazy_result.item<int32_t>({0, 0});

    {
        graph::EagerModeScope eager;
        auto ea = Tensor::full<int32_t>({32, 32}, 10);
        auto eb = Tensor::full<int32_t>({32, 32}, 3);
        auto eager_result = ops::abs(ops::subtract(ea, eb));
        int32_t eager_val = eager_result.item<int32_t>({0, 0});
        ASSERT_EQ(lazy_val, eager_val);
    }
}

// ============================================================================
// Dtype coverage: Int64 through lazy + fused paths
// ============================================================================

TEST(FusionDtype, Int64SubAbs) {
    auto a = Tensor::full<int64_t>({64, 64}, 10L);
    auto b = Tensor::full<int64_t>({64, 64}, 15L);
    auto result = ops::abs(ops::subtract(a, b));
    int64_t val = result.item<int64_t>({0, 0});
    ASSERT_EQ(val, 5L); // |10 - 15| = 5
}

TEST(FusionDtype, Int64Chain) {
    auto a = Tensor::full<int64_t>({32, 32}, 2L);
    auto b = Tensor::full<int64_t>({32, 32}, 3L);
    // (a + b) * a = (2 + 3) * 2 = 10
    auto result = ops::multiply(ops::add(a, b), a);
    int64_t val = result.item<int64_t>({0, 0});
    ASSERT_EQ(val, 10L);
}

// ============================================================================
// All 12 known fused patterns (SIMD dispatch)
// ============================================================================

// Pattern 1: AddReLU — relu(a + b)
TEST(FusionPattern, AddReLU) {
    auto a = Tensor::full<float>({100, 100}, -1.0f);
    auto b = Tensor::full<float>({100, 100}, 0.5f);
    auto result = ops::relu(ops::add(a, b));
    ASSERT_NEAR(result.item<float>({0, 0}), 0.0f, 1e-6f);

    auto a2 = Tensor::full<float>({100, 100}, 2.0f);
    auto result2 = ops::relu(ops::add(a2, b));
    ASSERT_NEAR(result2.item<float>({0, 0}), 2.5f, 1e-6f);
}

// Pattern 2: SubAbs — |a - b|
TEST(FusionPattern, SubAbs) {
    auto a = Tensor::full<float>({100, 100}, 3.0f);
    auto b = Tensor::full<float>({100, 100}, 5.0f);
    auto result = ops::abs(ops::subtract(a, b));
    ASSERT_NEAR(result.item<float>({0, 0}), 2.0f, 1e-6f);
}

// Pattern 3: AddSquare — (a + b)^2
TEST(FusionPattern, AddSquare) {
    auto a = Tensor::full<float>({100, 100}, 2.0f);
    auto b = Tensor::full<float>({100, 100}, 3.0f);
    auto result = ops::square(ops::add(a, b));
    ASSERT_NEAR(result.item<float>({0, 0}), 25.0f, 1e-5f);
}

// Pattern 4: MulReLU — relu(a * b)
TEST(FusionPattern, MulReLU) {
    auto a = Tensor::full<float>({100, 100}, -2.0f);
    auto b = Tensor::full<float>({100, 100}, 3.0f);
    auto result = ops::relu(ops::multiply(a, b));
    ASSERT_NEAR(result.item<float>({0, 0}), 0.0f, 1e-6f); // relu(-6) = 0
}

// Pattern 5: SubSquare — (a - b)^2
TEST(FusionPattern, SubSquare) {
    auto a = Tensor::full<float>({100, 100}, 7.0f);
    auto b = Tensor::full<float>({100, 100}, 4.0f);
    auto result = ops::square(ops::subtract(a, b));
    ASSERT_NEAR(result.item<float>({0, 0}), 9.0f, 1e-5f);
}

// Pattern 6: AddSigmoid — sigmoid(a + b)
TEST(FusionPattern, AddSigmoid) {
    auto a = Tensor::full<float>({100, 100}, 0.0f);
    auto b = Tensor::full<float>({100, 100}, 0.0f);
    auto result = ops::sigmoid(ops::add(a, b));
    ASSERT_NEAR(result.item<float>({0, 0}), 0.5f, 1e-5f);
}

// Pattern 7: MulSigmoid — sigmoid(a * b)
TEST(FusionPattern, MulSigmoid) {
    auto a = Tensor::full<float>({100, 100}, 1.0f);
    auto b = Tensor::full<float>({100, 100}, 0.0f);
    auto result = ops::sigmoid(ops::multiply(a, b));
    ASSERT_NEAR(result.item<float>({0, 0}), 0.5f, 1e-5f);
}

// Pattern 8: MulAdd (FMA) — a * b + c
TEST(FusionPattern, MulAdd) {
    auto a = Tensor::full<float>({100, 100}, 3.0f);
    auto b = Tensor::full<float>({100, 100}, 4.0f);
    auto c = Tensor::full<float>({100, 100}, 5.0f);
    auto result = ops::add(ops::multiply(a, b), c);
    ASSERT_NEAR(result.item<float>({0, 0}), 17.0f, 1e-5f);
}

// Pattern 9: MulSub — a * b - c
TEST(FusionPattern, MulSub) {
    auto a = Tensor::full<float>({100, 100}, 3.0f);
    auto b = Tensor::full<float>({100, 100}, 4.0f);
    auto c = Tensor::full<float>({100, 100}, 2.0f);
    auto result = ops::subtract(ops::multiply(a, b), c);
    ASSERT_NEAR(result.item<float>({0, 0}), 10.0f, 1e-5f);
}

// Pattern 10: AddMulReLU — relu((a + b) * c)
TEST(FusionPattern, AddMulReLU) {
    auto a = Tensor::full<float>({100, 100}, 1.0f);
    auto b = Tensor::full<float>({100, 100}, 2.0f);
    auto c = Tensor::full<float>({100, 100}, -1.0f);
    auto result = ops::relu(ops::multiply(ops::add(a, b), c));
    ASSERT_NEAR(result.item<float>({0, 0}), 0.0f, 1e-6f); // relu(-3) = 0
}

// Pattern 11: SubMulAbs — |(a - b) * c|
TEST(FusionPattern, SubMulAbs) {
    auto a = Tensor::full<float>({100, 100}, 3.0f);
    auto b = Tensor::full<float>({100, 100}, 5.0f);
    auto c = Tensor::full<float>({100, 100}, 4.0f);
    auto result = ops::abs(ops::multiply(ops::subtract(a, b), c));
    ASSERT_NEAR(result.item<float>({0, 0}), 8.0f, 1e-5f); // |(3-5)*4| = 8
}

// Pattern 12: ScaleShiftReLU — relu(a * scale + bias)
TEST(FusionPattern, ScaleShiftReLU) {
    auto a = Tensor::full<float>({100, 100}, 2.0f);
    auto scale = Tensor::full<float>({100, 100}, 3.0f);
    auto bias = Tensor::full<float>({100, 100}, -10.0f);
    auto result = ops::relu(ops::add(ops::multiply(a, scale), bias));
    ASSERT_NEAR(result.item<float>({0, 0}), 0.0f, 1e-6f); // relu(6-10) = 0

    auto bias2 = Tensor::full<float>({100, 100}, 1.0f);
    auto result2 = ops::relu(ops::add(ops::multiply(a, scale), bias2));
    ASSERT_NEAR(result2.item<float>({0, 0}), 7.0f, 1e-5f); // relu(6+1) = 7
}

// ============================================================================
// Known fused patterns with integer dtypes
// ============================================================================

TEST(FusionPattern, AddReLU_Int32) {
    auto a = Tensor::full<int32_t>({100, 100}, -5);
    auto b = Tensor::full<int32_t>({100, 100}, 3);
    auto result = ops::relu(ops::add(a, b));
    ASSERT_EQ(result.item<int32_t>({0, 0}), 0); // relu(-2) = 0
}

TEST(FusionPattern, SubAbs_Int64) {
    auto a = Tensor::full<int64_t>({100, 100}, 10L);
    auto b = Tensor::full<int64_t>({100, 100}, 25L);
    auto result = ops::abs(ops::subtract(a, b));
    ASSERT_EQ(result.item<int64_t>({0, 0}), 15L);
}

TEST(FusionPattern, MulAdd_Int32) {
    auto a = Tensor::full<int32_t>({100, 100}, 5);
    auto b = Tensor::full<int32_t>({100, 100}, 6);
    auto c = Tensor::full<int32_t>({100, 100}, 7);
    auto result = ops::add(ops::multiply(a, b), c);
    ASSERT_EQ(result.item<int32_t>({0, 0}), 37); // 5*6+7
}

// ============================================================================
// Shared-node materialization (ref_count > 1)
// ============================================================================

TEST(FusionSharedNode, BinaryUsesSharedInput) {
    auto x = Tensor::full<float>({64, 64}, 3.0f);

    // y = sqrt(x) creates a lazy node
    auto y = ops::sqrt(x);

    // Both z1 and z2 use y — y has ref_count > 1
    auto z1 = ops::add(y, y);      // 2 * sqrt(3)
    auto z2 = ops::multiply(y, y); // sqrt(3)^2 = 3

    float v1 = z1.item<float>({0, 0});
    float v2 = z2.item<float>({0, 0});

    ASSERT_NEAR(v1, 2.0f * std::sqrt(3.0f), 1e-5f);
    ASSERT_NEAR(v2, 3.0f, 1e-5f);
}

TEST(FusionSharedNode, ChainWithSharedIntermediate) {
    auto a = Tensor::full<float>({32, 32}, 2.0f);
    auto b = Tensor::full<float>({32, 32}, 1.0f);

    // s = a + b (shared intermediate)
    auto s = ops::add(a, b);

    // Two separate chains using s
    auto r1 = ops::exp(s);    // exp(3)
    auto r2 = ops::negate(s); // -3

    float v1 = r1.item<float>({0, 0});
    float v2 = r2.item<float>({0, 0});

    ASSERT_NEAR(v1, std::exp(3.0f), 1e-4f);
    ASSERT_NEAR(v2, -3.0f, 1e-6f);
}

TEST(FusionSharedNode, TripleUse) {
    auto x = Tensor::full<float>({32, 32}, 4.0f);

    auto y = ops::sqrt(x); // y = 2.0, ref_count will be 3

    auto a = ops::add(y, y);      // 4.0
    auto b = ops::multiply(y, y); // 4.0
    auto c = ops::subtract(y, x); // 2.0 - 4.0 = -2.0

    ASSERT_NEAR(a.item<float>({0, 0}), 4.0f, 1e-5f);
    ASSERT_NEAR(b.item<float>({0, 0}), 4.0f, 1e-5f);
    ASSERT_NEAR(c.item<float>({0, 0}), -2.0f, 1e-5f);
}

// ============================================================================
// Non-contiguous input fallback
// ============================================================================

TEST(FusionFallback, NonContiguousSlice) {
    // Create a tensor and take a non-contiguous view
    auto x = Tensor::full<float>({10, 10}, 2.0f);
    // Slice to get a view (non-contiguous in the underlying storage)
    auto view = x.slice({Slice(0, 5)}); // first 5 rows

    auto result = ops::sqrt(view);

    ASSERT_EQ(result.shape(), Shape({5, 10}));
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, std::sqrt(2.0f), 1e-5f);
}

TEST(FusionFallback, NonContiguousTranspose) {
    auto x = Tensor::full<float>({8, 12}, 9.0f);
    auto t = x.transpose(); // transpose produces non-contiguous view

    auto result = ops::sqrt(t);

    ASSERT_EQ(result.shape(), Shape({12, 8}));
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, 3.0f, 1e-5f);
}

TEST(FusionFallback, NonContiguousFusedChain) {
    auto x = Tensor::full<float>({10, 10}, 4.0f);
    auto view = x.slice({Slice(0, 5)});

    // This chain would normally be fused, but non-contiguous
    // forces fallback to op-by-op
    auto result = ops::relu(ops::add(view, view));

    ASSERT_EQ(result.shape(), Shape({5, 10}));
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, 8.0f, 1e-5f);
}

// ============================================================================
// Fused reductions
// ============================================================================

TEST(FusionReduction, SumAfterChain) {
    auto a = Tensor::full<float>({4, 4}, 2.0f);
    auto b = Tensor::full<float>({4, 4}, 3.0f);
    // sum(a + b) = sum(5.0 * 16) = 80
    auto result = ops::sum(ops::add(a, b));
    ASSERT_NEAR(result.item<float>(), 80.0f, 1e-4f);
}

TEST(FusionReduction, MeanAfterChain) {
    auto a = Tensor::full<float>({4, 4}, 2.0f);
    auto b = Tensor::full<float>({4, 4}, 6.0f);
    // mean(a + b) = mean(8.0) = 8.0
    auto result = ops::mean(ops::add(a, b));
    ASSERT_NEAR(result.item<float>(), 8.0f, 1e-5f);
}

TEST(FusionReduction, MaxAfterChain) {
    auto a = Tensor::full<float>({4, 4}, 2.0f);
    auto b = Tensor::full<float>({4, 4}, 3.0f);
    // max(a * b) = max(6.0) = 6.0
    auto result = ops::max(ops::multiply(a, b));
    ASSERT_NEAR(result.item<float>(), 6.0f, 1e-5f);
}

TEST(FusionReduction, MinAfterChain) {
    auto a = Tensor::full<float>({4, 4}, 5.0f);
    auto b = Tensor::full<float>({4, 4}, 2.0f);
    // min(a - b) = min(3.0) = 3.0
    auto result = ops::min(ops::subtract(a, b));
    ASSERT_NEAR(result.item<float>(), 3.0f, 1e-5f);
}

TEST(FusionReduction, ProdAfterUnary) {
    auto a = Tensor::full<float>({2, 2}, 4.0f);
    // prod(sqrt(4)) = prod(2.0) = 2^4 = 16
    auto result = ops::prod(ops::sqrt(a));
    ASSERT_NEAR(result.item<float>(), 16.0f, 1e-4f);
}

TEST(FusionReduction, SumFloat64) {
    auto a = Tensor::full<double>({4, 4}, 1.5);
    auto b = Tensor::full<double>({4, 4}, 2.5);
    // sum(a * b) = 16 * 3.75 = 60.0
    auto result = ops::sum(ops::multiply(a, b));
    ASSERT_NEAR(result.item<double>(), 60.0, 1e-10);
}

TEST(FusionReduction, EagerParitySum) {
    auto a = Tensor::full<float>({16, 16}, 1.5f);
    auto b = Tensor::full<float>({16, 16}, 2.0f);

    auto lazy_result = ops::sum(ops::relu(ops::subtract(a, b)));
    float lazy_val = lazy_result.item<float>();

    {
        graph::EagerModeScope eager;
        auto ea = Tensor::full<float>({16, 16}, 1.5f);
        auto eb = Tensor::full<float>({16, 16}, 2.0f);
        auto eager_result = ops::sum(ops::relu(ops::subtract(ea, eb)));
        float eager_val = eager_result.item<float>();
        ASSERT_NEAR(lazy_val, eager_val, 1e-5f);
    }
}

// ============================================================================
// Large tensor parallel paths (above FUSED_KNOWN_MIN_PARALLEL = 524288)
// ============================================================================

TEST(FusionParallel, LargeAddReLU) {
    // 1024 * 1024 = 1M elements, triggers parallel dispatch
    auto a = Tensor::full<float>({1024, 1024}, -0.5f);
    auto b = Tensor::full<float>({1024, 1024}, 1.0f);

    auto result = ops::relu(ops::add(a, b));

    ASSERT_EQ(result.shape(), Shape({1024, 1024}));
    ASSERT_NEAR(result.item<float>({0, 0}), 0.5f, 1e-6f);
    ASSERT_NEAR(result.item<float>({511, 511}), 0.5f, 1e-6f);
    ASSERT_NEAR(result.item<float>({1023, 1023}), 0.5f, 1e-6f);
}

TEST(FusionParallel, LargeGenericChain) {
    auto x = Tensor::full<float>({1024, 1024}, 1.0f);
    // tanh(sigmoid(x)) — generic fused loop path
    auto result = ops::tanh(ops::sigmoid(x));

    float expected = std::tanh(1.0f / (1.0f + std::exp(-1.0f)));
    ASSERT_NEAR(result.item<float>({512, 512}), expected, 1e-5f);
}

TEST(FusionParallel, LargeFusedReduction) {
    auto a = Tensor::full<float>({1024, 1024}, 1.0f);
    auto b = Tensor::full<float>({1024, 1024}, 1.0f);
    // sum(a + b) = 1M * 2.0 = 2097152
    auto result = ops::sum(ops::add(a, b));
    float expected = 1024.0f * 1024.0f * 2.0f;
    ASSERT_NEAR(result.item<float>(), expected, expected * 1e-5f);
}

TEST(FusionParallel, LargeFloat64) {
    auto a = Tensor::full<double>({1024, 1024}, 0.5);
    auto result = ops::exp(ops::negate(a));
    double expected = std::exp(-0.5);
    ASSERT_NEAR(result.item<double>({512, 512}), expected, 1e-12);
}

// ============================================================================
// GPU lazy path tests
// ============================================================================

class FusionGpuTest : public axiom::testing::GpuTest {};

TEST_F(FusionGpuTest, BasicAdd) {
    auto a = Tensor::full<float>({64, 64}, 2.0f, Device::GPU);
    auto b = Tensor::full<float>({64, 64}, 3.0f, Device::GPU);
    auto c = ops::add(a, b);
    auto c_cpu = c.cpu();
    ASSERT_NEAR(c_cpu.item<float>({0, 0}), 5.0f, 1e-5f);
}

TEST_F(FusionGpuTest, UnaryChain) {
    auto x = Tensor::full<float>({64, 64}, 4.0f, Device::GPU);
    auto result = ops::exp(ops::sqrt(x));
    auto result_cpu = result.cpu();
    ASSERT_NEAR(result_cpu.item<float>({0, 0}), std::exp(2.0f), 1e-4f);
}

TEST_F(FusionGpuTest, BinaryUnaryFusion) {
    auto a = Tensor::full<float>({64, 64}, -1.0f, Device::GPU);
    auto b = Tensor::full<float>({64, 64}, 0.5f, Device::GPU);
    auto result = ops::relu(ops::add(a, b));
    auto result_cpu = result.cpu();
    ASSERT_NEAR(result_cpu.item<float>({0, 0}), 0.0f, 1e-6f);
}

TEST_F(FusionGpuTest, LongerChain) {
    auto a = Tensor::full<float>({64, 64}, 0.5f, Device::GPU);
    auto b = Tensor::full<float>({64, 64}, 2.0f, Device::GPU);
    auto c = Tensor::full<float>({64, 64}, -0.5f, Device::GPU);

    auto result = ops::tanh(ops::sigmoid(ops::add(ops::multiply(a, b), c)));
    auto result_cpu = result.cpu();

    float expected = std::tanh(1.0f / (1.0f + std::exp(-0.5f)));
    ASSERT_NEAR(result_cpu.item<float>({0, 0}), expected, 1e-4f);
}

TEST_F(FusionGpuTest, Reduction) {
    auto a = Tensor::full<float>({4, 4}, 2.0f, Device::GPU);
    auto result = ops::sum(a);
    auto result_cpu = result.cpu();
    ASSERT_NEAR(result_cpu.item<float>(), 32.0f, 1e-4f);
}

TEST_F(FusionGpuTest, FusedReduction) {
    auto a = Tensor::full<float>({4, 4}, 2.0f, Device::GPU);
    auto b = Tensor::full<float>({4, 4}, 3.0f, Device::GPU);
    auto result = ops::sum(ops::add(a, b));
    auto result_cpu = result.cpu();
    ASSERT_NEAR(result_cpu.item<float>(), 80.0f, 1e-4f);
}

TEST_F(FusionGpuTest, EagerParity) {
    auto a = Tensor::full<float>({32, 32}, 1.5f, Device::GPU);
    auto b = Tensor::full<float>({32, 32}, 0.5f, Device::GPU);

    auto lazy_result = ops::sigmoid(ops::add(a, b));
    float lazy_val = lazy_result.cpu().item<float>({0, 0});

    {
        graph::EagerModeScope eager;
        auto ea = Tensor::full<float>({32, 32}, 1.5f, Device::GPU);
        auto eb = Tensor::full<float>({32, 32}, 0.5f, Device::GPU);
        auto eager_result = ops::sigmoid(ops::add(ea, eb));
        float eager_val = eager_result.cpu().item<float>({0, 0});
        ASSERT_NEAR(lazy_val, eager_val, 1e-4f);
    }
}

// ============================================================================
// Comparison/logical ops: should NOT fuse with arithmetic (dtype guard)
// ============================================================================

TEST(FusionDtypeGuard, ComparisonNotFusedWithArithmetic) {
    auto a = Tensor::full<float>({32, 32}, 2.0f);
    auto b = Tensor::full<float>({32, 32}, 3.0f);

    // add(a, b) is Float32, less(...) is Bool — should not be fused
    auto sum = ops::add(a, b);
    auto mask = ops::less(sum, b);

    ASSERT_EQ(mask.dtype(), DType::Bool);
    // sum = 5, b = 3 → 5 < 3 = false
    ASSERT_FALSE(mask.item<bool>({0, 0}));
}

TEST(FusionDtypeGuard, ArithmeticAfterComparison) {
    auto a = Tensor::full<float>({32, 32}, 1.0f);
    auto b = Tensor::full<float>({32, 32}, 2.0f);

    // These produce different dtypes, so should not be in same fused chain
    auto mask = ops::greater(a, b);
    ASSERT_EQ(mask.dtype(), DType::Bool);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(FusionEdge, ScalarTensor) {
    auto x = Tensor::full<float>({1}, 4.0f);
    auto result = ops::sqrt(x);
    ASSERT_NEAR(result.item<float>(), 2.0f, 1e-5f);
}

TEST(FusionEdge, VeryLongChain) {
    auto x = Tensor::full<float>({32, 32}, 1.0f);
    // Chain of 10 operations — should still produce correct result
    auto r = x;
    for (int i = 0; i < 10; ++i) {
        r = ops::relu(ops::add(r, x));
    }
    // After each step: relu(prev + 1) where prev starts at 1
    // Step 1: relu(1+1) = 2; Step 2: relu(2+1) = 3; ...
    // Step 10: 11
    ASSERT_NEAR(r.item<float>({0, 0}), 11.0f, 1e-4f);
}

TEST(FusionEdge, IdentityOps) {
    auto x = Tensor::full<float>({32, 32}, 5.0f);
    // negate(negate(x)) should be x
    auto result = ops::negate(ops::negate(x));
    ASSERT_NEAR(result.item<float>({0, 0}), 5.0f, 1e-6f);
}
