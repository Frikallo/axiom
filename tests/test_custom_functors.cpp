//=============================================================================
// tests/test_custom_functors.cpp - Tests for ops::apply / ops::vectorize
//=============================================================================

#include "axiom_test_utils.hpp"
#include <cmath>
#include <vector>

using namespace axiom;

// ============================================================================
// Unary apply tests
// ============================================================================

TEST(CustomFunctors, UnaryBasicFloat32) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_data(data, {4});

    auto result = ops::apply(t, [](float x) { return x * x + 1.0f; });

    ASSERT_EQ(result.dtype(), DType::Float32);
    ASSERT_EQ(result.shape(), Shape{4});
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {2.0f, 5.0f, 10.0f, 17.0f});
}

TEST(CustomFunctors, UnaryAutocast) {
    // Int32 tensor + float lambda should auto-cast
    int32_t data[] = {1, 2, 3, 4};
    auto t = Tensor::from_data(data, {4});
    ASSERT_EQ(t.dtype(), DType::Int32);

    auto result = ops::apply(t, [](float x) { return x * 0.5f; });

    ASSERT_EQ(result.dtype(), DType::Float32);
    axiom::testing::ExpectTensorEquals<float>(result, {0.5f, 1.0f, 1.5f, 2.0f});
}

TEST(CustomFunctors, UnaryNonContiguous) {
    // Create a transposed view (non-contiguous)
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::from_data(data, {2, 3});
    auto tv = t.transpose(); // shape (3,2), non-contiguous
    ASSERT_FALSE(tv.is_contiguous());

    auto result = ops::apply(tv, [](float x) { return x * 2.0f; });

    ASSERT_EQ(result.shape(), (Shape{3, 2}));
    ASSERT_TRUE(result.is_contiguous());
    // Transposed data: row-major iteration of (3,2) transpose of
    // [[1,2,3],[4,5,6]] gives [[1,4],[2,5],[3,6]]
    axiom::testing::ExpectTensorEquals<float>(
        result, {2.0f, 8.0f, 4.0f, 10.0f, 6.0f, 12.0f});
}

TEST(CustomFunctors, UnaryEmptyTensor) {
    auto t = Tensor({0, 3}, DType::Float32);
    auto result = ops::apply(t, [](float x) { return x + 1.0f; });

    ASSERT_EQ(result.shape(), (Shape{0, 3}));
    ASSERT_EQ(result.dtype(), DType::Float32);
    ASSERT_TRUE(result.empty());
}

TEST(CustomFunctors, UnaryScalar) {
    auto t = Tensor::full<float>({}, 5.0f);
    ASSERT_EQ(t.ndim(), 0u);
    ASSERT_EQ(t.size(), 1u);

    auto result = ops::apply(t, [](float x) { return x * x; });

    ASSERT_EQ(result.ndim(), 0u);
    ASSERT_EQ(result.size(), 1u);
    ASSERT_NEAR(result.item<float>(), 25.0f, 1e-6f);
}

// ============================================================================
// Binary apply tests
// ============================================================================

TEST(CustomFunctors, BinaryBasicFloat32) {
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    auto a = Tensor::from_data(a_data, {4});
    auto b = Tensor::from_data(b_data, {4});

    auto result =
        ops::apply(a, b, [](float x, float y) { return x * y + 1.0f; });

    ASSERT_EQ(result.dtype(), DType::Float32);
    ASSERT_EQ(result.shape(), Shape{4});
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {11.0f, 41.0f, 91.0f, 161.0f});
}

TEST(CustomFunctors, BinaryBroadcast) {
    // (3,4) + (4,) should broadcast
    float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float b_data[] = {10, 20, 30, 40};
    auto a = Tensor::from_data(a_data, {3, 4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = ops::apply(a, b, [](float x, float y) { return x + y; });

    ASSERT_EQ(result.shape(), (Shape{3, 4}));
    axiom::testing::ExpectTensorEquals<float>(
        result, {11, 22, 33, 44, 15, 26, 37, 48, 19, 30, 41, 52});
}

TEST(CustomFunctors, BinaryBroadcastDifferentDtypes) {
    // Int32 lhs + Float32 rhs, lambda takes float for both
    int32_t a_data[] = {1, 2, 3};
    float b_data[] = {0.5f, 1.0f, 1.5f};
    auto a = Tensor::from_data(a_data, {3});
    auto b = Tensor::from_data(b_data, {3});

    auto result = ops::apply(a, b, [](float x, float y) { return x + y; });

    ASSERT_EQ(result.dtype(), DType::Float32);
    axiom::testing::ExpectTensorEquals<float>(result, {1.5f, 3.0f, 4.5f});
}

// ============================================================================
// vectorize tests
// ============================================================================

TEST(CustomFunctors, VectorizeUnary) {
    auto square = ops::vectorize([](float x) { return x * x; });

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_data(data, {4});

    auto result = square(t);
    auto expected = ops::apply(t, [](float x) { return x * x; });

    ASSERT_TRUE(result.allclose(expected));
}

TEST(CustomFunctors, VectorizeBinary) {
    auto my_add = ops::vectorize([](float x, float y) { return x + y; });

    float a_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float b_data[] = {10, 20, 30, 40};
    auto a = Tensor::from_data(a_data, {3, 4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = my_add(a, b);

    ASSERT_EQ(result.shape(), (Shape{3, 4}));
    axiom::testing::ExpectTensorEquals<float>(
        result, {11, 22, 33, 44, 15, 26, 37, 48, 19, 30, 41, 52});
}

// ============================================================================
// Type-related tests
// ============================================================================

TEST(CustomFunctors, BoolOutput) {
    float data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data, {4});

    auto result = ops::apply(t, [](float x) -> bool { return x > 0.0f; });

    ASSERT_EQ(result.dtype(), DType::Bool);
    ASSERT_EQ(result.shape(), Shape{4});
    axiom::testing::ExpectTensorEquals<bool>(result,
                                             {false, false, true, true});
}

TEST(CustomFunctors, DoubleTypes) {
    double data[] = {1.0, 4.0, 9.0, 16.0};
    auto t = Tensor::from_data(data, {4});

    auto result = ops::apply(t, [](double x) { return std::sqrt(x); });

    ASSERT_EQ(result.dtype(), DType::Float64);
    axiom::testing::ExpectTensorEquals<double>(result, {1.0, 2.0, 3.0, 4.0});
}

// ============================================================================
// apply_along_axis tests
// ============================================================================

TEST(CustomFunctors, ApplyAlongAxisScalarReturn) {
    // Sum each row of a (2,3) matrix -> shape (2,)
    // np.apply_along_axis(np.sum, 1, arr)
    float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_data(data, {2, 3});

    auto result = ops::apply_along_axis(
        [](const Tensor &slice) { return ops::sum(slice); }, 1, t);

    ASSERT_EQ(result.shape(), (Shape{2}));
    axiom::testing::ExpectTensorEquals<float>(result, {6.0f, 15.0f});
}

TEST(CustomFunctors, ApplyAlongAxisAxis0) {
    // Sum each column of a (2,3) matrix -> shape (3,)
    float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_data(data, {2, 3});

    auto result = ops::apply_along_axis(
        [](const Tensor &slice) { return ops::sum(slice); }, 0, t);

    ASSERT_EQ(result.shape(), (Shape{3}));
    axiom::testing::ExpectTensorEquals<float>(result, {5.0f, 7.0f, 9.0f});
}

TEST(CustomFunctors, ApplyAlongAxis1DInput) {
    // 1-D input: just calls func1d directly
    float data[] = {1, 2, 3, 4};
    auto t = Tensor::from_data(data, {4});

    auto result = ops::apply_along_axis(
        [](const Tensor &slice) { return ops::sum(slice); }, 0, t);

    ASSERT_NEAR(result.item<float>(), 10.0f, 1e-6f);
}

TEST(CustomFunctors, ApplyAlongAxisVectorReturn) {
    // func1d returns a 1-D tensor (same length) -> result keeps shape
    // e.g. sort each row
    float data[] = {3, 1, 2, 6, 4, 5};
    auto t = Tensor::from_data(data, {2, 3});

    auto result = ops::apply_along_axis(
        [](const Tensor &slice) {
            // Reverse the slice: multiply by -1, negate back after sort
            // Just double each element as a simple transform
            return ops::apply(slice, [](float x) { return x * 2.0f; });
        },
        1, t);

    ASSERT_EQ(result.shape(), (Shape{2, 3}));
    axiom::testing::ExpectTensorEquals<float>(result, {6, 2, 4, 12, 8, 10});
}

TEST(CustomFunctors, ApplyAlongAxis3D) {
    // 3-D tensor (2,3,4), apply sum along axis=2 -> (2,3)
    float data[24];
    for (int i = 0; i < 24; ++i)
        data[i] = static_cast<float>(i + 1);
    auto t = Tensor::from_data(data, {2, 3, 4});

    auto result = ops::apply_along_axis(
        [](const Tensor &slice) { return ops::sum(slice); }, 2, t);

    ASSERT_EQ(result.shape(), (Shape{2, 3}));
    // Row sums: [1+2+3+4, 5+6+7+8, 9+10+11+12, ...]
    axiom::testing::ExpectTensorEquals<float>(result, {10, 26, 42, 58, 74, 90});
}

TEST(CustomFunctors, ApplyAlongAxisNegativeAxis) {
    float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_data(data, {2, 3});

    // axis=-1 is same as axis=1 for 2-D
    auto result = ops::apply_along_axis(
        [](const Tensor &slice) { return ops::sum(slice); }, -1, t);

    ASSERT_EQ(result.shape(), (Shape{2}));
    axiom::testing::ExpectTensorEquals<float>(result, {6.0f, 15.0f});
}

// ============================================================================
// apply_over_axes tests
// ============================================================================

TEST(CustomFunctors, ApplyOverAxesSingle) {
    // Sum over axis 0 with keepdims
    float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_data(data, {2, 3});

    auto result = ops::apply_over_axes(
        [](const Tensor &t, int ax) {
            return ops::sum(t, {ax}, /*keep_dims=*/true);
        },
        t, {0});

    ASSERT_EQ(result.shape(), (Shape{1, 3}));
    axiom::testing::ExpectTensorEquals<float>(result, {5, 7, 9});
}

TEST(CustomFunctors, ApplyOverAxesMultiple) {
    // Sum over axes 0 and 1 sequentially -> scalar-like (1,1)
    float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_data(data, {2, 3});

    auto result = ops::apply_over_axes(
        [](const Tensor &t, int ax) {
            return ops::sum(t, {ax}, /*keep_dims=*/true);
        },
        t, {0, 1});

    ASSERT_EQ(result.shape(), (Shape{1, 1}));
    ASSERT_NEAR(result.item<float>(), 21.0f, 1e-6f);
}

TEST(CustomFunctors, ApplyOverAxes3D) {
    // 3-D tensor, reduce over axes 0 and 2
    float data[24];
    for (int i = 0; i < 24; ++i)
        data[i] = static_cast<float>(i + 1);
    auto t = Tensor::from_data(data, {2, 3, 4});

    auto result = ops::apply_over_axes(
        [](const Tensor &t, int ax) {
            return ops::sum(t, {ax}, /*keep_dims=*/true);
        },
        t, {0, 2});

    // After sum axis=0: (1,3,4), after sum axis=2: (1,3,1)
    ASSERT_EQ(result.shape(), (Shape{1, 3, 1}));
    // axis=0 sums: [1+13,2+14,...] = [14,16,18,20, 22,24,26,28, 30,32,34,36]
    // then axis=2 sums each group of 4: [14+16+18+20, 22+24+26+28,
    // 30+32+34+36] = [68, 100, 132]
    axiom::testing::ExpectTensorEquals<float>(result, {68, 100, 132});
}

// ============================================================================
// fromfunc tests
// ============================================================================

TEST(CustomFunctors, FromFuncUnary) {
    auto my_func = ops::fromfunc([](float x) { return x * x; });

    float data[] = {1, 2, 3, 4};
    auto t = Tensor::from_data(data, {4});
    auto result = my_func(t);

    axiom::testing::ExpectTensorEquals<float>(result, {1, 4, 9, 16});
}

TEST(CustomFunctors, FromFuncBinary) {
    auto my_func = ops::fromfunc([](float x, float y) { return x * y; });

    float a_data[] = {1, 2, 3};
    float b_data[] = {10, 20, 30};
    auto a = Tensor::from_data(a_data, {3});
    auto b = Tensor::from_data(b_data, {3});

    auto result = my_func(a, b);
    axiom::testing::ExpectTensorEquals<float>(result, {10, 40, 90});
}

// ============================================================================
// Error handling
// ============================================================================

TEST(CustomFunctors, GPURejects) {
    SKIP_IF_NO_GPU();

    auto t = Tensor({4}, DType::Float32, Device::GPU);
    EXPECT_THROW(ops::apply(t, [](float x) { return x; }), DeviceError);
}

TEST(CustomFunctors, ApplyAlongAxisGPURejects) {
    SKIP_IF_NO_GPU();

    auto t = Tensor({4}, DType::Float32, Device::GPU);
    EXPECT_THROW(ops::apply_along_axis(
                     [](const Tensor &s) { return ops::sum(s); }, 0, t),
                 DeviceError);
}

TEST(CustomFunctors, ApplyAlongAxisInvalidAxis) {
    float data[] = {1, 2, 3};
    auto t = Tensor::from_data(data, {3});
    EXPECT_THROW(ops::apply_along_axis(
                     [](const Tensor &s) { return ops::sum(s); }, 5, t),
                 ShapeError);
}
