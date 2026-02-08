#include "axiom_test_utils.hpp"

#include <cmath>
#include <vector>

using namespace axiom;

// ==================================
//
//      UNARY OP TESTS - CPU
//
// ==================================

TEST(TensorUnaryOperations, NegateCPU) {
    auto device = Device::CPU;
    auto a =
        Tensor::arange(6).reshape({2, 3}).astype(DType::Float32).to(device);
    auto b = ops::negate(a);
    axiom::testing::ExpectTensorEquals<float>(b, {0, -1, -2, -3, -4, -5});

    // Test with operator overload
    auto c = -a;
    axiom::testing::ExpectTensorEquals<float>(c, {0, -1, -2, -3, -4, -5});
}

TEST(TensorUnaryOperations, AbsCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({-1, 2, -3, 0, -5, 6});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::abs(a);
    axiom::testing::ExpectTensorEquals<float>(c, {1, 2, 3, 0, 5, 6});
}

TEST(TensorUnaryOperations, SqrtCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({0, 1, 4, 9, 16, 25});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::sqrt(a);
    axiom::testing::ExpectTensorEquals<float>(c, {0, 1, 2, 3, 4, 5});
}

TEST(TensorUnaryOperations, ExpCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::exp(a);
    axiom::testing::ExpectTensorEquals<float>(c,
                                              {std::exp(0.0f), std::exp(1.0f),
                                               std::exp(2.0f), std::exp(3.0f),
                                               std::exp(4.0f), std::exp(5.0f)},
                                              1e-5);
}

TEST(TensorUnaryOperations, LogCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1, 2, 3, 4, 5, 6});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::log(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::log(1.0f), std::log(2.0f), std::log(3.0f), std::log(4.0f),
            std::log(5.0f), std::log(6.0f)});
}

TEST(TensorUnaryOperations, SinCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::sin(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::sin(0.0f), std::sin(1.0f), std::sin(2.0f), std::sin(3.0f),
            std::sin(4.0f), std::sin(5.0f)});
}

TEST(TensorUnaryOperations, CosCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::cos(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::cos(0.0f), std::cos(1.0f), std::cos(2.0f), std::cos(3.0f),
            std::cos(4.0f), std::cos(5.0f)});
}

TEST(TensorUnaryOperations, TanCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::tan(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::tan(0.0f), std::tan(1.0f), std::tan(2.0f), std::tan(3.0f),
            std::tan(4.0f), std::tan(5.0f)});
}

// ==================================
//
//      UNARY OP TESTS - GPU
//
// ==================================

TEST(TensorUnaryOperations, NegateGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto a =
        Tensor::arange(6).reshape({2, 3}).astype(DType::Float32).to(device);
    auto b = ops::negate(a);
    axiom::testing::ExpectTensorEquals<float>(b, {0, -1, -2, -3, -4, -5});

    // Test with operator overload
    auto c = -a;
    axiom::testing::ExpectTensorEquals<float>(c, {0, -1, -2, -3, -4, -5});
}

TEST(TensorUnaryOperations, AbsGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({-1, 2, -3, 0, -5, 6});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::abs(a);
    axiom::testing::ExpectTensorEquals<float>(c, {1, 2, 3, 0, 5, 6});
}

TEST(TensorUnaryOperations, SqrtGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({0, 1, 4, 9, 16, 25});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::sqrt(a);
    axiom::testing::ExpectTensorEquals<float>(c, {0, 1, 2, 3, 4, 5});
}

TEST(TensorUnaryOperations, ExpGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::exp(a);
    axiom::testing::ExpectTensorEquals<float>(c,
                                              {std::exp(0.0f), std::exp(1.0f),
                                               std::exp(2.0f), std::exp(3.0f),
                                               std::exp(4.0f), std::exp(5.0f)},
                                              1e-5);
}

TEST(TensorUnaryOperations, LogGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({1, 2, 3, 4, 5, 6});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::log(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::log(1.0f), std::log(2.0f), std::log(3.0f), std::log(4.0f),
            std::log(5.0f), std::log(6.0f)});
}

TEST(TensorUnaryOperations, SinGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::sin(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::sin(0.0f), std::sin(1.0f), std::sin(2.0f), std::sin(3.0f),
            std::sin(4.0f), std::sin(5.0f)});
}

TEST(TensorUnaryOperations, CosGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::cos(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::cos(0.0f), std::cos(1.0f), std::cos(2.0f), std::cos(3.0f),
            std::cos(4.0f), std::cos(5.0f)});
}

TEST(TensorUnaryOperations, TanGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = ops::tan(a);
    axiom::testing::ExpectTensorEquals<float>(
        c, {std::tan(0.0f), std::tan(1.0f), std::tan(2.0f), std::tan(3.0f),
            std::tan(4.0f), std::tan(5.0f)});
}
