#include "axiom_test_utils.hpp"

#include <cmath>
#include <vector>

using namespace axiom;

// ==================================
//
//      CPU REDUCTION OP TESTS
//
// ==================================

TEST(TensorReductions, SumAll) {
    auto a = Tensor::arange(6).reshape({2, 3}).astype(DType::Float32);
    auto c = ops::sum(a);
    axiom::testing::ExpectTensorEquals<float>(c, {15.0f});
}

TEST(TensorReductions, SumAxis0) {
    auto a = Tensor::arange(6).reshape({2, 3}).astype(DType::Float32);
    auto c = ops::sum(a, {0});
    axiom::testing::ExpectTensorEquals<float>(c, {3.0f, 5.0f, 7.0f});
}

TEST(TensorReductions, SumAxis1) {
    auto a = Tensor::arange(6).reshape({2, 3}).astype(DType::Float32);
    auto c = ops::sum(a, {1});
    axiom::testing::ExpectTensorEquals<float>(c, {3.0f, 12.0f});
}

TEST(TensorReductions, SumAxisDefault) {
    auto a = Tensor::arange(6).reshape({2, 3}).astype(DType::Float32);
    auto c = ops::sum(a);
    axiom::testing::ExpectTensorEquals<float>(c, {15.0f});
}

TEST(TensorReductions, SumKeepdims) {
    auto a = Tensor::arange(6).reshape({2, 3}).astype(DType::Float32);
    auto c = ops::sum(a, {1}, true);
    ASSERT_TRUE(c.shape() == Shape({2, 1})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(c, {3.0f, 12.0f});
}

TEST(TensorReductions, MeanAll) {
    auto a = Tensor::arange(6).reshape({2, 3}).astype(DType::Float32);
    auto c = ops::mean(a);
    axiom::testing::ExpectTensorEquals<float>(c, {2.5f});
}

TEST(TensorReductions, MaxAll) {
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a = Tensor::from_data(data.data(), {2, 3});
    auto c = ops::max(a);
    axiom::testing::ExpectTensorEquals<float>(c, {9.0f});
}

TEST(TensorReductions, MinAll) {
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a = Tensor::from_data(data.data(), {2, 3});
    auto c = ops::min(a);
    axiom::testing::ExpectTensorEquals<float>(c, {1.0f});
}

// ==================================
//
//      GPU REDUCTION OP TESTS
//
// ==================================

TEST(TensorReductions, SumAllGpu) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(DType::Float32)
                 .to(Device::GPU);
    auto c = ops::sum(a);
    axiom::testing::ExpectTensorEquals<float>(c, {15.0f});
}

TEST(TensorReductions, SumAxis0Gpu) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(DType::Float32)
                 .to(Device::GPU);
    auto c = ops::sum(a, {0});
    axiom::testing::ExpectTensorEquals<float>(c, {3.0f, 5.0f, 7.0f});
}

TEST(TensorReductions, SumAxis1Gpu) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(DType::Float32)
                 .to(Device::GPU);
    auto c = ops::sum(a, {1});
    axiom::testing::ExpectTensorEquals<float>(c, {3.0f, 12.0f});
}

TEST(TensorReductions, SumKeepdimsGpu) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(DType::Float32)
                 .to(Device::GPU);
    auto c = ops::sum(a, {1}, true);
    ASSERT_TRUE(c.shape() == Shape({2, 1})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(c, {3.0f, 12.0f});
}

TEST(TensorReductions, MeanAllGpu) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(DType::Float32)
                 .to(Device::GPU);
    auto c = ops::mean(a);
    axiom::testing::ExpectTensorEquals<float>(c, {2.5f});
}

TEST(TensorReductions, MaxAllGpu) {
    SKIP_IF_NO_GPU();
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a = Tensor::from_data(data.data(), {2, 3}).to(Device::GPU);
    auto c = ops::max(a);
    axiom::testing::ExpectTensorEquals<float>(c, {9.0f});
}

TEST(TensorReductions, MinAllGpu) {
    SKIP_IF_NO_GPU();
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a = Tensor::from_data(data.data(), {2, 3}).to(Device::GPU);
    auto c = ops::min(a);
    axiom::testing::ExpectTensorEquals<float>(c, {1.0f});
}

TEST(TensorReductions, NonContiguousSumGpu) {
    SKIP_IF_NO_GPU();
    // Create a larger tensor ON THE GPU first
    auto a_gpu = Tensor::arange(24)
                     .reshape({2, 3, 4})
                     .astype(DType::Float32)
                     .to(Device::GPU);

    // Perform non-contiguous-making operations on the GPU tensor
    auto b_gpu =
        a_gpu.slice({Slice(), Slice(1, 3), Slice()}); // Shape {2, 2, 4}
    auto c_gpu = b_gpu.transpose({2, 0, 1}); // Shape {4, 2, 2}, non-contiguous

    ASSERT_TRUE(!c_gpu.is_contiguous())
        << "Tensor should be non-contiguous for this test";

    // Perform the reduction on the non-contiguous GPU tensor
    auto result_gpu = ops::sum(c_gpu, {1, 2});

    // Create the equivalent non-contiguous tensor on CPU for verification
    auto a_cpu = Tensor::arange(24).reshape({2, 3, 4}).astype(DType::Float32);
    auto b_cpu = a_cpu.slice({Slice(), Slice(1, 3), Slice()});
    auto c_cpu = b_cpu.transpose({2, 0, 1});
    auto result_cpu = ops::sum(c_cpu, {1, 2});

    // Compare the results
    axiom::testing::ExpectTensorsClose(result_gpu, result_cpu);
}
