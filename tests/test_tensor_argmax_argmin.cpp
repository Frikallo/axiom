#include "axiom_test_utils.hpp"

// Test argmax on 1D tensor
TEST(TensorArgmaxArgmin, Argmax1D) {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = axiom::ops::argmax(t);

    ASSERT_TRUE(result.dtype() == axiom::DType::Int64)
        << "Result should be Int64";
    ASSERT_TRUE(result.ndim() == 0) << "Should be scalar";
    ASSERT_TRUE(result.item<int64_t>({}) == 3) << "Max is at index 3";
}

// Test argmin on 1D tensor
TEST(TensorArgmaxArgmin, Argmin1D) {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, -2.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = axiom::ops::argmin(t);

    ASSERT_TRUE(result.dtype() == axiom::DType::Int64)
        << "Result should be Int64";
    ASSERT_TRUE(result.ndim() == 0) << "Should be scalar";
    ASSERT_TRUE(result.item<int64_t>({}) == 3) << "Min is at index 3";
}

// Test argmax along axis
TEST(TensorArgmaxArgmin, ArgmaxAxis0) {
    // [[1, 5], [3, 2]] -> argmax(axis=0) -> [1, 0]
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmax(t, 0);

    ASSERT_TRUE(result.shape() == axiom::Shape({2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<int64_t>(result, {1, 0});
}

// Test argmax along axis 1
TEST(TensorArgmaxArgmin, ArgmaxAxis1) {
    // [[1, 5], [3, 2]] -> argmax(axis=1) -> [1, 0]
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmax(t, 1);

    ASSERT_TRUE(result.shape() == axiom::Shape({2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<int64_t>(result, {1, 0});
}

// Test argmin along axis
TEST(TensorArgmaxArgmin, ArgminAxis0) {
    // [[1, 5], [3, 2]] -> argmin(axis=0) -> [0, 1]
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmin(t, 0);

    ASSERT_TRUE(result.shape() == axiom::Shape({2})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<int64_t>(result, {0, 1});
}

// Test argmax with keep_dims
TEST(TensorArgmaxArgmin, ArgmaxKeepDims) {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmax(t, 1, true);

    ASSERT_TRUE(result.shape() == axiom::Shape({2, 1}))
        << "Shape should have kept dim";
}

// Test argmin with keep_dims
TEST(TensorArgmaxArgmin, ArgminKeepDims) {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmin(t, 0, true);

    ASSERT_TRUE(result.shape() == axiom::Shape({1, 2}))
        << "Shape should have kept dim";
}

// Test argmax member function
TEST(TensorArgmaxArgmin, ArgmaxMember) {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = t.argmax();

    ASSERT_TRUE(result.item<int64_t>({}) == 3) << "Max is at index 3";
}

// Test argmin member function
TEST(TensorArgmaxArgmin, ArgminMember) {
    std::vector<float> data = {1.0f, 5.0f, -3.0f, 9.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = t.argmin();

    ASSERT_TRUE(result.item<int64_t>({}) == 2) << "Min is at index 2";
}

// Test argmax on 3D tensor
TEST(TensorArgmaxArgmin, Argmax3D) {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4}).astype(
        axiom::DType::Float32);
    auto result = axiom::ops::argmax(t, 2); // Along last axis

    ASSERT_TRUE(result.shape() == axiom::Shape({2, 3})) << "Shape mismatch";
    // Each row of 4 elements should have max at index 3
    std::vector<int64_t> expected(6, 3);
    axiom::testing::ExpectTensorEquals<int64_t>(result, expected);
}
