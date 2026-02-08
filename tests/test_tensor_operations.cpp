#include "axiom_test_utils.hpp"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

using namespace axiom;

// ==================================
//
//      ORIGINAL TEST BED
//
// ==================================

TEST(TensorOperations, CpuAddSuccess) {
    auto a = Tensor::full({2, 2}, 2.0f);
    auto b = Tensor::full({2, 2}, 3.0f);
    auto c = ops::add(a, b);
    ASSERT_TRUE(c.device() == Device::CPU) << "Device mismatch";
    const float *c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 5.0f, 1e-6);
}

TEST(TensorOperations, MetalAddSuccess) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::full({2, 2}, 2.0f).to(Device::GPU);
    auto b = Tensor::full({2, 2}, 3.0f).to(Device::GPU);
    auto c = ops::add(a, b);
    ASSERT_TRUE(c.device() == Device::GPU) << "Device mismatch";
    auto c_cpu = c.cpu();
    const float *c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 5.0f, 1e-6);
}

TEST(TensorOperations, CpuSubSuccess) {
    auto a = Tensor::full({2, 2}, 10.0f);
    auto b = Tensor::full({2, 2}, 3.0f);
    auto c = ops::subtract(a, b);
    ASSERT_TRUE(c.device() == Device::CPU) << "Device mismatch";
    const float *c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 7.0f, 1e-6);
}

TEST(TensorOperations, MetalSubSuccess) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::full({2, 2}, 10.0f).to(Device::GPU);
    auto b = Tensor::full({2, 2}, 3.0f).to(Device::GPU);
    auto c = ops::subtract(a, b);
    ASSERT_TRUE(c.device() == Device::GPU) << "Device mismatch";
    auto c_cpu = c.cpu();
    const float *c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 7.0f, 1e-6);
}

TEST(TensorOperations, MetalMulSuccess) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::full({2, 2}, 10.0f).to(Device::GPU);
    auto b = Tensor::full({2, 2}, 3.0f).to(Device::GPU);
    auto c = ops::multiply(a, b);
    ASSERT_TRUE(c.device() == Device::GPU) << "Device mismatch";
    auto c_cpu = c.cpu();
    const float *c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 30.0f, 1e-6);
}

TEST(TensorOperations, MetalDivSuccess) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::full({2, 2}, 30.0f).to(Device::GPU);
    auto b = Tensor::full({2, 2}, 3.0f).to(Device::GPU);
    auto c = ops::divide(a, b);
    ASSERT_TRUE(c.device() == Device::GPU) << "Device mismatch";
    auto c_cpu = c.cpu();
    const float *c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 10.0f, 1e-6);
}

TEST(TensorOperations, CpuBroadcastingSuccess) {
    auto a = Tensor::full({2, 2}, 10.0f);
    auto b = Tensor::full({}, 5.0f); // Scalar
    auto c = ops::add(a, b);
    ASSERT_TRUE(c.shape() == Shape({2, 2})) << "Shape mismatch";
    const float *c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 15.0f, 1e-6);
}

TEST(TensorOperations, CpuTypePromotionSuccess) {
    auto a = Tensor::full({2, 2}, static_cast<int32_t>(10));
    auto b = Tensor::full({2, 2}, 5.5f);
    ASSERT_TRUE(a.dtype() == DType::Int32) << "DType mismatch";
    ASSERT_TRUE(b.dtype() == DType::Float32) << "DType mismatch";

    auto c = ops::add(a, b);
    ASSERT_TRUE(c.dtype() == DType::Float32) << "Promotion failed";
    const float *c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 15.5f, 1e-6);
}

TEST(TensorOperations, UnsupportedGpuOpFallback) {
    SKIP_IF_NO_GPU();
    // Power is now implemented on Metal via MPSGraph, should run on GPU!
    auto a = Tensor::full({2, 2}, 3.0f).to(Device::GPU);
    auto b = Tensor::full({2, 2}, 4.0f).to(Device::GPU);
    auto c = ops::power(a, b);

    // NEW: Power now runs on GPU via MPSGraph!
    ASSERT_TRUE(c.device() == Device::GPU) << "Power should run on GPU";

    auto c_cpu = c.cpu();
    const float *c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 81.0f, 1e-6);
}

TEST(TensorOperations, MixedDeviceOpSuccess) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::full({2, 2}, 3.0f, Device::CPU);
    auto b = Tensor::full({2, 2}, 4.0f).to(Device::GPU);
    auto c = ops::add(a, b);

    ASSERT_TRUE(c.device() == Device::GPU) << "Device mismatch";
    auto c_cpu = c.cpu();
    const float *c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 7.0f, 1e-6);
}

TEST(TensorOperations, ShapeMismatchError) {
    auto a = Tensor::full({2, 3}, 1.0f);
    auto b = Tensor::full({2, 4}, 1.0f);
    EXPECT_THROW(ops::add(a, b), std::exception);
}

TEST(TensorOperations, MetalAddIntSuccess) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::full({2, 2}, static_cast<int32_t>(1)).to(Device::GPU);
    auto b = Tensor::full({2, 2}, static_cast<int32_t>(2)).to(Device::GPU);
    ASSERT_TRUE(a.dtype() == DType::Int32) << "DType mismatch";
    ASSERT_TRUE(b.dtype() == DType::Int32) << "DType mismatch";

    auto c = ops::add(a, b);
    ASSERT_TRUE(c.device() == Device::GPU) << "Device mismatch";
    ASSERT_TRUE(c.dtype() == DType::Int32) << "DType mismatch";

    auto c_cpu = c.cpu();
    const int32_t *c_data = c_cpu.typed_data<int32_t>();
    for (size_t i = 0; i < 4; ++i)
        ASSERT_TRUE(c_data[i] == 3) << "Data mismatch";
}

TEST(TensorOperations, MetalBroadcastingSuccess) {
    SKIP_IF_NO_GPU();
    auto a = Tensor::full({2, 2}, 10.0f).to(Device::GPU);
    auto b = Tensor::full({}, 5.0f).to(Device::GPU);

    auto c = ops::add(a, b);

    ASSERT_TRUE(c.device() == Device::GPU) << "Device mismatch, expected GPU";
    ASSERT_TRUE(c.shape() == a.shape()) << "Broadcast shape mismatch";

    auto c_cpu = c.cpu();
    const float *c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(c_data[i], 15.0f, 1e-6);
}

TEST(TensorOperations, InplaceOperations) {
    // Basic in-place addition
    auto a = Tensor::full({2, 2}, (int32_t)5);
    auto b = Tensor::full({2, 2}, (int32_t)3);
    a += b;
    axiom::testing::ExpectTensorEquals<int32_t>(a, {8, 8, 8, 8});

    // In-place subtraction
    auto c = Tensor::full({2, 2}, (int32_t)8);
    auto d = Tensor::full({2, 2}, (int32_t)2);
    c -= d;
    axiom::testing::ExpectTensorEquals<int32_t>(c, {6, 6, 6, 6});

    // In-place multiplication
    auto e = Tensor::full({2, 2}, (int32_t)6);
    auto f = Tensor::full({2, 2}, (int32_t)3);
    e *= f;
    axiom::testing::ExpectTensorEquals<int32_t>(e, {18, 18, 18, 18});

    // In-place division
    auto g = Tensor::full({2, 2}, (int32_t)18);
    auto h = Tensor::full({2, 2}, (int32_t)3);
    g /= h;
    axiom::testing::ExpectTensorEquals<int32_t>(g, {6, 6, 6, 6});

    // In-place with scalar
    auto i = Tensor::full({3, 3}, 10.f);
    i += 5.f;
    axiom::testing::ExpectTensorEquals<float>(
        i, {15.f, 15.f, 15.f, 15.f, 15.f, 15.f, 15.f, 15.f, 15.f});

    // Test unsafe type cast
    auto j = Tensor::full({2, 2}, (int32_t)5);
    auto k = Tensor::full({2, 2}, 3.f);
    EXPECT_THROW(j += k, std::exception);

    // Test shape mismatch
    auto l = Tensor::full({2, 2}, (int32_t)5);
    auto m = Tensor::full({3, 3}, (int32_t)5);
    EXPECT_THROW(l += m, std::exception);

    // Test broadcasting with in-place (broadcasts {2} to {2,2})
    auto n = Tensor::full({2, 2}, (int32_t)5);
    auto o = Tensor::full({2}, (int32_t)5);
    EXPECT_NO_THROW(n += o);
    axiom::testing::ExpectTensorEquals<int32_t>(n, {10, 10, 10, 10});
}

// ==================================
//
//      FULL COVERAGE TESTS
//
// ==================================

// DType list for iteration
const std::vector<DType> all_dtypes = {
    DType::Bool,   DType::Int8,    DType::Int16,   DType::Int32,
    DType::Int64,  DType::UInt8,   DType::UInt16,  DType::UInt32,
    DType::UInt64, DType::Float16, DType::Float32, DType::Float64,
};

template <typename T_a, typename T_b, typename T_exp>
void test_arithmetic_op(
    const std::string &op_name,
    std::function<Tensor(const Tensor &, const Tensor &)> op, T_a val_a,
    T_b val_b, T_exp val_exp) {
    auto a = Tensor::full({2, 2}, val_a);
    auto b = Tensor::full({2, 2}, val_b);

    auto result = op(a, b);

    auto expected_dtype = ops::promote_types(a.dtype(), b.dtype());
    ASSERT_TRUE(result.dtype() == expected_dtype)
        << "DType promotion failed for " << op_name;

    std::vector<T_exp> expected_data(4, val_exp);
    axiom::testing::ExpectTensorEquals<T_exp>(result, expected_data);
}

TEST(TensorOperations, AllArithmeticOps) {
    test_arithmetic_op<float, int, float>("add", ops::add, 2.5f, 3, 5.5f);
    test_arithmetic_op<int, float, float>("sub", ops::subtract, 10, 3.5f, 6.5f);
    test_arithmetic_op<double, int, double>("mul", ops::multiply, 2.5, 4, 10.0);
    test_arithmetic_op<int, int, int>("div", ops::divide, 10, 3, 3);
    test_arithmetic_op<uint8_t, int8_t, int16_t>("add_mixed_sign", ops::add, 10,
                                                 -5, 5);
}

TEST(TensorOperations, FullBroadcastingCPU) {
    auto device = Device::CPU;

    // (2, 3) + scalar
    auto a_data = std::vector<float>{1, 2, 3, 4, 5, 6};
    auto a = Tensor::from_data(a_data.data(), {2, 3}).to(device);
    auto b_scalar = Tensor::full({}, 10.0f).to(device);
    auto c = ops::add(a, b_scalar);
    ASSERT_TRUE(c.shape() == a.shape()) << "Scalar broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(c, {11, 12, 13, 14, 15, 16});

    // (2, 3) + (3,)
    auto b_vec_data = std::vector<float>{10, 20, 30};
    auto b_vec = Tensor::from_data(b_vec_data.data(), {3}).to(device);
    auto d = ops::add(a, b_vec);
    ASSERT_TRUE(d.shape() == a.shape()) << "Vector broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(d, {11, 22, 33, 14, 25, 36});

    // (2, 3) + (2, 1)
    auto b_col_data = std::vector<float>{10, 20};
    auto b_col = Tensor::from_data(b_col_data.data(), {2, 1}).to(device);
    auto e = ops::add(a, b_col);
    ASSERT_TRUE(e.shape() == a.shape()) << "Column broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(e, {11, 12, 13, 24, 25, 26});

    // (1, 3) + (2, 1) -> (2, 3)
    auto a_row_data = std::vector<float>{1, 2, 3};
    auto a_row = Tensor::from_data(a_row_data.data(), {1, 3}).to(device);
    auto f = ops::add(a_row, b_col);
    Shape expected_shape = {2, 3};
    ASSERT_TRUE(f.shape() == expected_shape)
        << "2D-2D broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(f, {11, 12, 13, 21, 22, 23});

    // Incompatible broadcast
    auto b_wrong = Tensor::full({4}, 1.0f).to(device);
    EXPECT_THROW(ops::add(a, b_wrong), std::exception);
}

TEST(TensorOperations, FullBroadcastingGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;

    // (2, 3) + scalar
    auto a_data = std::vector<float>{1, 2, 3, 4, 5, 6};
    auto a = Tensor::from_data(a_data.data(), {2, 3}).to(device);
    auto b_scalar = Tensor::full({}, 10.0f).to(device);
    auto c = ops::add(a, b_scalar);
    ASSERT_TRUE(c.shape() == a.shape()) << "Scalar broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(c, {11, 12, 13, 14, 15, 16});

    // (2, 3) + (3,)
    auto b_vec_data = std::vector<float>{10, 20, 30};
    auto b_vec = Tensor::from_data(b_vec_data.data(), {3}).to(device);
    auto d = ops::add(a, b_vec);
    ASSERT_TRUE(d.shape() == a.shape()) << "Vector broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(d, {11, 22, 33, 14, 25, 36});

    // (2, 3) + (2, 1)
    auto b_col_data = std::vector<float>{10, 20};
    auto b_col = Tensor::from_data(b_col_data.data(), {2, 1}).to(device);
    auto e = ops::add(a, b_col);
    ASSERT_TRUE(e.shape() == a.shape()) << "Column broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(e, {11, 12, 13, 24, 25, 26});

    // (1, 3) + (2, 1) -> (2, 3)
    auto a_row_data = std::vector<float>{1, 2, 3};
    auto a_row = Tensor::from_data(a_row_data.data(), {1, 3}).to(device);
    auto f = ops::add(a_row, b_col);
    Shape expected_shape = {2, 3};
    ASSERT_TRUE(f.shape() == expected_shape)
        << "2D-2D broadcast shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(f, {11, 12, 13, 21, 22, 23});

    // Incompatible broadcast
    auto b_wrong = Tensor::full({4}, 1.0f).to(device);
    EXPECT_THROW(ops::add(a, b_wrong), std::exception);
}

TEST(TensorOperations, TypePromotionGrid) {
    for (auto dtype_a : all_dtypes) {
        for (auto dtype_b : all_dtypes) {
            // Create dummy tensors of the specified types
            auto a = Tensor::empty({1}, dtype_a);
            auto b = Tensor::empty({1}, dtype_b);

            // Get expected promotion result
            auto expected_dtype = ops::promote_types(dtype_a, dtype_b);

            // Perform an operation (add is fine, we only care about the
            // resulting type)
            auto result = ops::add(a, b);

            std::string test_msg = "Promotion failed for " +
                                   dtype_name(dtype_a) + " + " +
                                   dtype_name(dtype_b);
            ASSERT_TRUE(result.dtype() == expected_dtype) << test_msg;
        }
    }
}
