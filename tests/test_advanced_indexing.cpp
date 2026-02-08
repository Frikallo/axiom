// Tests for advanced indexing operations

#include "axiom_test_utils.hpp"

using namespace axiom;

TEST(AdvancedIndexing, Take1D) {
    auto t = Tensor::arange(10);
    auto indices =
        Tensor::from_data(std::vector<int64_t>{0, 2, 4}.data(), {3}, true);

    auto result = ops::take(t, indices);
    ASSERT_TRUE(result.shape() == Shape({3})) << "Take result shape wrong";
    ASSERT_TRUE(result.item<int32_t>({0}) == 0) << "Take value 0 wrong";
    ASSERT_TRUE(result.item<int32_t>({1}) == 2) << "Take value 1 wrong";
    ASSERT_TRUE(result.item<int32_t>({2}) == 4) << "Take value 2 wrong";
}

TEST(AdvancedIndexing, Take2D) {
    auto t = Tensor::from_data(std::vector<float>{1, 2, 3, 4, 5, 6}.data(),
                               {2, 3}, true);
    auto indices =
        Tensor::from_data(std::vector<int64_t>{0, 1}.data(), {2}, true);

    // Take along axis 0 (rows)
    auto result = ops::take(t, indices, 0);
    ASSERT_TRUE(result.shape() == Shape({2, 3})) << "Take 2D shape wrong";
}

TEST(AdvancedIndexing, TakeNegativeIndices) {
    auto t = Tensor::arange(5);
    auto indices =
        Tensor::from_data(std::vector<int64_t>{-1, -2}.data(), {2}, true);

    auto result = ops::take(t, indices);
    ASSERT_TRUE(result.item<int32_t>({0}) == 4) << "Negative index -1 wrong";
    ASSERT_TRUE(result.item<int32_t>({1}) == 3) << "Negative index -2 wrong";
}

TEST(AdvancedIndexing, TakeAlongAxis) {
    auto t = Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}.data(), {2, 3},
        true);
    auto indices = Tensor::from_data(
        std::vector<int64_t>{0, 2, 1, 1, 0, 2}.data(), {2, 3}, true);

    auto result = ops::take_along_axis(t, indices, 1);
    ASSERT_TRUE(result.shape() == Shape({2, 3}))
        << "take_along_axis shape wrong";

    // Row 0: indices [0, 2, 1] -> values [1, 3, 2]
    ASSERT_TRUE(result.item<float>({0, 0}) == 1.0f)
        << "take_along_axis [0,0] wrong";
    ASSERT_TRUE(result.item<float>({0, 1}) == 3.0f)
        << "take_along_axis [0,1] wrong";
    ASSERT_TRUE(result.item<float>({0, 2}) == 2.0f)
        << "take_along_axis [0,2] wrong";
}

TEST(AdvancedIndexing, TakeAlongAxis0) {
    auto t = Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}.data(), {2, 3},
        true);
    auto indices =
        Tensor::from_data(std::vector<int64_t>{1, 0, 1}.data(), {1, 3}, true);

    auto result = ops::take_along_axis(t, indices, 0);
    ASSERT_TRUE(result.shape() == Shape({1, 3}))
        << "take_along_axis axis=0 shape wrong";

    // Column 0: index 1 -> 4, Column 1: index 0 -> 2, Column 2: index 1 -> 6
    ASSERT_TRUE(result.item<float>({0, 0}) == 4.0f)
        << "take_along_axis axis=0 [0,0] wrong";
    ASSERT_TRUE(result.item<float>({0, 1}) == 2.0f)
        << "take_along_axis axis=0 [0,1] wrong";
    ASSERT_TRUE(result.item<float>({0, 2}) == 6.0f)
        << "take_along_axis axis=0 [0,2] wrong";
}

TEST(AdvancedIndexing, PutAlongAxis) {
    auto t = Tensor::zeros({3, 3});
    auto indices =
        Tensor::from_data(std::vector<int64_t>{0, 1, 2}.data(), {3, 1}, true);
    auto values = Tensor::ones({3, 1});

    auto result = ops::put_along_axis(t, indices, values, 1);
    ASSERT_TRUE(result.shape() == Shape({3, 3}))
        << "put_along_axis shape wrong";

    // Diagonal should be 1
    ASSERT_TRUE(result.item<float>({0, 0}) == 1.0f)
        << "Diagonal 0,0 should be 1";
    ASSERT_TRUE(result.item<float>({1, 1}) == 1.0f)
        << "Diagonal 1,1 should be 1";
    ASSERT_TRUE(result.item<float>({2, 2}) == 1.0f)
        << "Diagonal 2,2 should be 1";

    // Off-diagonal should be 0
    ASSERT_TRUE(result.item<float>({0, 1}) == 0.0f)
        << "Off-diagonal 0,1 should be 0";
    ASSERT_TRUE(result.item<float>({1, 0}) == 0.0f)
        << "Off-diagonal 1,0 should be 0";
}

TEST(AdvancedIndexing, PutAlongAxisMultiple) {
    auto t = Tensor::zeros({2, 4});
    auto indices = Tensor::from_data(std::vector<int64_t>{0, 2, 1, 3}.data(),
                                     {2, 2}, true);
    auto values = Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}.data(), {2, 2}, true);

    auto result = ops::put_along_axis(t, indices, values, 1);
    ASSERT_TRUE(result.shape() == Shape({2, 4}))
        << "put_along_axis multiple shape wrong";

    // Row 0: put 1 at index 0, 2 at index 2
    ASSERT_TRUE(result.item<float>({0, 0}) == 1.0f) << "Row 0 index 0 wrong";
    ASSERT_TRUE(result.item<float>({0, 2}) == 2.0f) << "Row 0 index 2 wrong";

    // Row 1: put 3 at index 1, 4 at index 3
    ASSERT_TRUE(result.item<float>({1, 1}) == 3.0f) << "Row 1 index 1 wrong";
    ASSERT_TRUE(result.item<float>({1, 3}) == 4.0f) << "Row 1 index 3 wrong";
}
