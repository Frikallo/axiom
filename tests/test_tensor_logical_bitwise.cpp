#include "axiom_test_utils.hpp"
#include <cmath>
#include <vector>

using namespace axiom;

// ==================================
//      LOGICAL OPERATION TESTS
// ==================================

TEST(TensorLogicalBitwise, LogicalAndBasic) {
    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = ops::logical_and(a, b);

    ASSERT_TRUE(result.dtype() == DType::Bool) << "Should return Bool";
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {true, false, false, false});
}

TEST(TensorLogicalBitwise, LogicalOrBasic) {
    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = ops::logical_or(a, b);

    ASSERT_TRUE(result.dtype() == DType::Bool) << "Should return Bool";
    axiom::testing::ExpectTensorEquals<bool>(result, {true, true, true, false});
}

TEST(TensorLogicalBitwise, LogicalXorBasic) {
    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = ops::logical_xor(a, b);

    ASSERT_TRUE(result.dtype() == DType::Bool) << "Should return Bool";
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {false, true, true, false});
}

TEST(TensorLogicalBitwise, LogicalNotBasic) {
    bool a_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});

    auto result = ops::logical_not(a);

    ASSERT_TRUE(result.dtype() == DType::Bool) << "Should return Bool";
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {false, true, false, true});
}

TEST(TensorLogicalBitwise, LogicalNotFromFloat) {
    // Non-zero values should be treated as true
    std::vector<float> a_data = {1.0f, 0.0f, -3.5f, 0.0f};
    auto a = Tensor::from_data(a_data.data(), {4});

    auto result = ops::logical_not(a);

    ASSERT_TRUE(result.dtype() == DType::Bool) << "Should return Bool";
    // 1.0 -> true -> false, 0.0 -> false -> true, etc.
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {false, true, false, true});
}

TEST(TensorLogicalBitwise, LogicalNotFromInt) {
    std::vector<int32_t> a_data = {1, 0, -5, 0};
    auto a = Tensor::from_data(a_data.data(), {4});

    auto result = ops::logical_not(a);

    ASSERT_TRUE(result.dtype() == DType::Bool) << "Should return Bool";
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {false, true, false, true});
}

TEST(TensorLogicalBitwise, LogicalAndGpu) {
    SKIP_IF_NO_GPU();

    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4}).gpu();
    auto b = Tensor::from_data(b_data, {4}).gpu();

    auto result = ops::logical_and(a, b);

    ASSERT_TRUE(result.device() == Device::GPU) << "Should be on GPU";
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {true, false, false, false});
}

TEST(TensorLogicalBitwise, LogicalNotGpu) {
    SKIP_IF_NO_GPU();

    bool a_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4}).gpu();

    auto result = ops::logical_not(a);

    ASSERT_TRUE(result.device() == Device::GPU) << "Should be on GPU";
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {false, true, false, true});
}

// ==================================
//      BITWISE OPERATION TESTS
// ==================================

TEST(TensorLogicalBitwise, BitwiseAndBasic) {
    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b1010, 0b1111, 0b0000};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_and(a, b);

    ASSERT_TRUE(result.dtype() == DType::Int32) << "Should return Int32";
    axiom::testing::ExpectTensorEquals<int32_t>(
        result, {0b1010, 0b1010, 0b0000, 0b0000});
}

TEST(TensorLogicalBitwise, BitwiseOrBasic) {
    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b0101, 0b1111, 0b0000};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_or(a, b);

    ASSERT_TRUE(result.dtype() == DType::Int32) << "Should return Int32";
    axiom::testing::ExpectTensorEquals<int32_t>(
        result, {0b1111, 0b1111, 0b1111, 0b1111});
}

TEST(TensorLogicalBitwise, BitwiseXorBasic) {
    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b1010, 0b1111, 0b1111};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_xor(a, b);

    ASSERT_TRUE(result.dtype() == DType::Int32) << "Should return Int32";
    axiom::testing::ExpectTensorEquals<int32_t>(
        result, {0b0101, 0b0000, 0b1111, 0b0000});
}

TEST(TensorLogicalBitwise, LeftShiftBasic) {
    std::vector<int32_t> a_data = {1, 2, 4, 8};
    std::vector<int32_t> b_data = {1, 2, 1, 0};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::left_shift(a, b);

    ASSERT_TRUE(result.dtype() == DType::Int32) << "Should return Int32";
    // 1 << 1 = 2, 2 << 2 = 8, 4 << 1 = 8, 8 << 0 = 8
    axiom::testing::ExpectTensorEquals<int32_t>(result, {2, 8, 8, 8});
}

TEST(TensorLogicalBitwise, RightShiftBasic) {
    std::vector<int32_t> a_data = {8, 16, 4, 1};
    std::vector<int32_t> b_data = {1, 2, 1, 0};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::right_shift(a, b);

    ASSERT_TRUE(result.dtype() == DType::Int32) << "Should return Int32";
    // 8 >> 1 = 4, 16 >> 2 = 4, 4 >> 1 = 2, 1 >> 0 = 1
    axiom::testing::ExpectTensorEquals<int32_t>(result, {4, 4, 2, 1});
}

TEST(TensorLogicalBitwise, BitwiseAndGpu) {
    SKIP_IF_NO_GPU();

    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b1010, 0b1111, 0b0000};
    auto a = Tensor::from_data(a_data.data(), {4}).gpu();
    auto b = Tensor::from_data(b_data.data(), {4}).gpu();

    auto result = ops::bitwise_and(a, b);

    ASSERT_TRUE(result.device() == Device::GPU) << "Should be on GPU";
    axiom::testing::ExpectTensorEquals<int32_t>(
        result, {0b1010, 0b1010, 0b0000, 0b0000});
}

TEST(TensorLogicalBitwise, BitwiseWithUint8) {
    std::vector<uint8_t> a_data = {0xFF, 0xAA, 0x00, 0x55};
    std::vector<uint8_t> b_data = {0xAA, 0xAA, 0xFF, 0x55};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_and(a, b);

    ASSERT_TRUE(result.dtype() == DType::UInt8) << "Should return UInt8";
    axiom::testing::ExpectTensorEquals<uint8_t>(result,
                                                {0xAA, 0xAA, 0x00, 0x55});
}

// ==================================
//      MATH OPERATION TESTS
// ==================================

TEST(TensorLogicalBitwise, MaximumBasic) {
    std::vector<float> a_data = {1.0f, 5.0f, 3.0f, 0.0f};
    std::vector<float> b_data = {2.0f, 4.0f, 3.0f, -1.0f};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::maximum(a, b);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0] - 2.0f) < 1e-5f) << "max(1,2)=2";
    ASSERT_TRUE(std::abs(data[1] - 5.0f) < 1e-5f) << "max(5,4)=5";
    ASSERT_TRUE(std::abs(data[2] - 3.0f) < 1e-5f) << "max(3,3)=3";
    ASSERT_TRUE(std::abs(data[3] - 0.0f) < 1e-5f) << "max(0,-1)=0";
}

TEST(TensorLogicalBitwise, MinimumBasic) {
    std::vector<float> a_data = {1.0f, 5.0f, 3.0f, 0.0f};
    std::vector<float> b_data = {2.0f, 4.0f, 3.0f, -1.0f};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::minimum(a, b);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0] - 1.0f) < 1e-5f) << "min(1,2)=1";
    ASSERT_TRUE(std::abs(data[1] - 4.0f) < 1e-5f) << "min(5,4)=4";
    ASSERT_TRUE(std::abs(data[2] - 3.0f) < 1e-5f) << "min(3,3)=3";
    ASSERT_TRUE(std::abs(data[3] - (-1.0f)) < 1e-5f) << "min(0,-1)=-1";
}

TEST(TensorLogicalBitwise, Atan2Basic) {
    std::vector<float> y_data = {1.0f, 1.0f, 0.0f};
    std::vector<float> x_data = {1.0f, 0.0f, 1.0f};
    auto y = Tensor::from_data(y_data.data(), {3});
    auto x = Tensor::from_data(x_data.data(), {3});

    auto result = ops::atan2(y, x);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    // atan2(1,1) = pi/4, atan2(1,0) = pi/2, atan2(0,1) = 0
    ASSERT_TRUE(std::abs(data[0] - static_cast<float>(M_PI / 4)) < 1e-5f)
        << "atan2(1,1)=pi/4";
    ASSERT_TRUE(std::abs(data[1] - static_cast<float>(M_PI / 2)) < 1e-5f)
        << "atan2(1,0)=pi/2";
    ASSERT_TRUE(std::abs(data[2] - 0.0f) < 1e-5f) << "atan2(0,1)=0";
}

TEST(TensorLogicalBitwise, HypotBasic) {
    std::vector<float> a_data = {3.0f, 0.0f, 5.0f};
    std::vector<float> b_data = {4.0f, 5.0f, 12.0f};
    auto a = Tensor::from_data(a_data.data(), {3});
    auto b = Tensor::from_data(b_data.data(), {3});

    auto result = ops::hypot(a, b);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    // hypot(3,4)=5, hypot(0,5)=5, hypot(5,12)=13
    ASSERT_TRUE(std::abs(data[0] - 5.0f) < 1e-5f) << "hypot(3,4)=5";
    ASSERT_TRUE(std::abs(data[1] - 5.0f) < 1e-5f) << "hypot(0,5)=5";
    ASSERT_TRUE(std::abs(data[2] - 13.0f) < 1e-5f) << "hypot(5,12)=13";
}

TEST(TensorLogicalBitwise, HypotGpu) {
    SKIP_IF_NO_GPU();

    std::vector<float> a_data = {3.0f, 0.0f, 5.0f};
    std::vector<float> b_data = {4.0f, 5.0f, 12.0f};
    auto a = Tensor::from_data(a_data.data(), {3}).gpu();
    auto b = Tensor::from_data(b_data.data(), {3}).gpu();

    auto result = ops::hypot(a, b);

    ASSERT_TRUE(result.device() == Device::GPU) << "Should be on GPU";
    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0] - 5.0f) < 1e-4f) << "hypot(3,4)=5";
    ASSERT_TRUE(std::abs(data[1] - 5.0f) < 1e-4f) << "hypot(0,5)=5";
    ASSERT_TRUE(std::abs(data[2] - 13.0f) < 1e-4f) << "hypot(5,12)=13";
}
