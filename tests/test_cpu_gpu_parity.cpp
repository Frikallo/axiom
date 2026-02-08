#include "axiom_test_utils.hpp"
#include <cmath>
#include <iomanip>
#include <vector>

using namespace axiom;

// ============================================================================
// Binary Operation Parity Tests
// ============================================================================

TEST(CpuGpuParity, AddParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::add(a, b);
    auto gpu_result = ops::add(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Add parity failed";
}

TEST(CpuGpuParity, SubtractParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::subtract(a, b);
    auto gpu_result = ops::subtract(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Subtract parity failed";
}

TEST(CpuGpuParity, MultiplyParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::multiply(a, b);
    auto gpu_result = ops::multiply(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Multiply parity failed";
}

TEST(CpuGpuParity, DivideParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    // Avoid division by very small numbers
    b = ops::add(b, Tensor::full({4, 5}, 1.0f, Device::CPU));

    auto cpu_result = ops::divide(a, b);
    auto gpu_result = ops::divide(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Divide parity failed";
}

TEST(CpuGpuParity, MaximumParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::maximum(a, b);
    auto gpu_result = ops::maximum(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Maximum parity failed";
}

TEST(CpuGpuParity, MinimumParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::minimum(a, b);
    auto gpu_result = ops::minimum(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Minimum parity failed";
}

TEST(CpuGpuParity, HypotParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::hypot(a, b);
    auto gpu_result = ops::hypot(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "Hypot parity failed";
}

// ============================================================================
// Comparison Operation Parity Tests
// ============================================================================

TEST(CpuGpuParity, EqualParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = a.copy(); // Same values

    auto cpu_result = ops::equal(a, b);
    auto gpu_result = ops::equal(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "Equal parity failed";
}

TEST(CpuGpuParity, LessParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::less(a, b);
    auto gpu_result = ops::less(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "Less parity failed";
}

TEST(CpuGpuParity, GreaterParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::greater(a, b);
    auto gpu_result = ops::greater(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "Greater parity failed";
}

// ============================================================================
// Logical Operation Parity Tests
// ============================================================================

TEST(CpuGpuParity, LogicalAndParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::logical_and(a, b);
    auto gpu_result = ops::logical_and(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "LogicalAnd parity failed";
}

TEST(CpuGpuParity, LogicalOrParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::logical_or(a, b);
    auto gpu_result = ops::logical_or(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "LogicalOr parity failed";
}

TEST(CpuGpuParity, LogicalNotParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::logical_not(a);
    auto gpu_result = ops::logical_not(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "LogicalNot parity failed";
}

// ============================================================================
// Unary Operation Parity Tests
// ============================================================================

TEST(CpuGpuParity, NegateParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::negate(a);
    auto gpu_result = ops::negate(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Negate parity failed";
}

TEST(CpuGpuParity, AbsParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::abs(a);
    auto gpu_result = ops::abs(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Abs parity failed";
}

TEST(CpuGpuParity, SqrtParity) {
    SKIP_IF_NO_GPU();

    // Use positive values for sqrt
    auto a = ops::abs(Tensor::randn({4, 5}, DType::Float32, Device::CPU));
    a = ops::add(a, Tensor::full({4, 5}, 0.1f, Device::CPU)); // Avoid sqrt(0)

    auto cpu_result = ops::sqrt(a);
    auto gpu_result = ops::sqrt(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Sqrt parity failed";
}

TEST(CpuGpuParity, ExpParity) {
    SKIP_IF_NO_GPU();

    // Use small values to avoid overflow
    auto a = ops::divide(Tensor::randn({4, 5}, DType::Float32, Device::CPU),
                         Tensor::full({4, 5}, 10.0f, Device::CPU));

    auto cpu_result = ops::exp(a);
    auto gpu_result = ops::exp(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "Exp parity failed";
}

TEST(CpuGpuParity, LogParity) {
    SKIP_IF_NO_GPU();

    // Use positive values for log
    auto a = ops::abs(Tensor::randn({4, 5}, DType::Float32, Device::CPU));
    a = ops::add(a, Tensor::full({4, 5}, 1.0f, Device::CPU));

    auto cpu_result = ops::log(a);
    auto gpu_result = ops::log(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Log parity failed";
}

TEST(CpuGpuParity, SinParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::sin(a);
    auto gpu_result = ops::sin(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Sin parity failed";
}

TEST(CpuGpuParity, CosParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::cos(a);
    auto gpu_result = ops::cos(a.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Cos parity failed";
}

// ============================================================================
// Reduction Operation Parity Tests
// ============================================================================

TEST(CpuGpuParity, SumParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    // Full reduction
    auto cpu_result = ops::sum(a, {}, false);
    auto gpu_result = ops::sum(a.gpu(), {}, false);
    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "Sum full parity failed";

    // Axis reduction
    cpu_result = ops::sum(a, {1}, false);
    gpu_result = ops::sum(a.gpu(), {1}, false);
    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "Sum axis parity failed";

    // Keep dims
    cpu_result = ops::sum(a, {1}, true);
    gpu_result = ops::sum(a.gpu(), {1}, true);
    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "Sum keep_dims parity failed";
}

TEST(CpuGpuParity, MeanParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::mean(a, {1}, false);
    auto gpu_result = ops::mean(a.gpu(), {1}, false);

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "Mean parity failed";
}

TEST(CpuGpuParity, MaxParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::max(a, {1}, false);
    auto gpu_result = ops::max(a.gpu(), {1}, false);

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Max parity failed";
}

TEST(CpuGpuParity, MinParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::min(a, {1}, false);
    auto gpu_result = ops::min(a.gpu(), {1}, false);

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Min parity failed";
}

TEST(CpuGpuParity, ArgmaxParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::argmax(a, 1, false);
    auto gpu_result = ops::argmax(a.gpu(), 1, false);

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "ArgMax parity failed";
}

TEST(CpuGpuParity, ArgminParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::argmin(a, 1, false);
    auto gpu_result = ops::argmin(a.gpu(), 1, false);

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 0, 0))
        << "ArgMin parity failed";
}

// ============================================================================
// MatMul Parity Tests
// ============================================================================

TEST(CpuGpuParity, Matmul2dParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::matmul(a, b);
    auto gpu_result = ops::matmul(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "MatMul 2D parity failed";
}

TEST(CpuGpuParity, MatmulBatchedParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({2, 4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({2, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::matmul(a, b);
    auto gpu_result = ops::matmul(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "MatMul batched parity failed";
}

// ============================================================================
// Special Operation Parity Tests
// ============================================================================

TEST(CpuGpuParity, WhereParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto cond =
        ops::greater(Tensor::randn({4, 5}, DType::Float32, Device::CPU),
                     Tensor::zeros({4, 5}, DType::Float32, Device::CPU));

    auto cpu_result = ops::where(cond, a, b);
    auto gpu_result = ops::where(cond.gpu(), a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Where parity failed";
}

TEST(CpuGpuParity, SoftmaxParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::softmax(a, -1);
    auto gpu_result = ops::softmax(a.gpu(), -1);

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "Softmax parity failed";
}

TEST(CpuGpuParity, GatherParity) {
    SKIP_IF_NO_GPU();

    auto input = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    int64_t idx_data[] = {0, 2, 1, 3};
    auto indices = Tensor::from_data(idx_data, {4});

    auto cpu_result =
        ops::gather(input, 1, indices.unsqueeze(1).expand({4, 1}));
    auto gpu_result =
        ops::gather(input.gpu(), 1, indices.unsqueeze(1).expand({4, 1}).gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Gather parity failed";
}

TEST(CpuGpuParity, IndexSelectParity) {
    SKIP_IF_NO_GPU();

    auto input = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    int64_t idx_data[] = {0, 2, 1};
    auto indices = Tensor::from_data(idx_data, {3});

    auto cpu_result = ops::index_select(input, 0, indices);
    auto gpu_result = ops::index_select(input.gpu(), 0, indices.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "IndexSelect parity failed";
}

// ============================================================================
// Broadcasting Parity Tests
// ============================================================================

TEST(CpuGpuParity, BroadcastAddParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({1, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::add(a, b);
    auto gpu_result = ops::add(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Broadcast add parity failed";
}

TEST(CpuGpuParity, BroadcastMultiplyParity) {
    SKIP_IF_NO_GPU();

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::multiply(a, b);
    auto gpu_result = ops::multiply(a.gpu(), b.gpu());

    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-5))
        << "Broadcast multiply parity failed";
}
