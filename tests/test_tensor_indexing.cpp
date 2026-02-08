// Tests for tensor indexing operations

#include "axiom_test_utils.hpp"

using namespace axiom;

TEST(TensorIndexing, SimpleIndexingSlice) {
    auto a = Tensor::arange(0, 10); // {0, 1, ..., 9}
    auto b = a[{Slice(2, 5)}];      // {2, 3, 4}
    ASSERT_TRUE(b.shape() == Shape({3}))
        << "Simple indexing slice shape mismatch";
    axiom::testing::ExpectTensorEquals<int>(b, {2, 3, 4});
}

TEST(TensorIndexing, SimpleIndexingInteger) {
    auto data = std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7};
    auto a = Tensor::from_data(data.data(), {2, 4});
    auto b = a[{0}]; // First row
    ASSERT_TRUE(b.shape() == Shape({4}))
        << "Simple indexing integer shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(b, {0, 1, 2, 3});
}

TEST(TensorIndexing, MixedIndexing) {
    auto a = Tensor::arange(0, 16).reshape({4, 4});
    // Corresponds to numpy's a[1, 1:3] -> {5, 6}
    auto b = a[{1, Slice(1, 3)}];
    ASSERT_TRUE(b.shape() == Shape({2})) << "Mixed indexing shape mismatch";
    axiom::testing::ExpectTensorEquals<int>(b, {5, 6});
}

TEST(TensorIndexing, NegativeIndexing) {
    auto a = Tensor::arange(0, 5); // {0, 1, 2, 3, 4}
    auto b = a[{-1}];              // Last element
    ASSERT_TRUE(b.shape() == Shape{}) << "Negative indexing shape mismatch";
    ASSERT_TRUE(b.item<int>({}) == 4) << "Negative indexing value mismatch";

    // Corresponds to numpy's a[-3:-1] -> {2, 3}
    auto c = a[{Slice(-3, -1)}];
    ASSERT_TRUE(c.shape() == Shape({2})) << "Negative slice shape mismatch";
    axiom::testing::ExpectTensorEquals<int>(c, {2, 3});
}

TEST(TensorIndexing, IndexingSharedStorage) {
    auto a = Tensor::zeros({4, 4});
    // Get a view of the second row
    auto b = a[{1}];
    // Fill the view with 1s
    b.fill(1.0f);

    // The original tensor should now be modified
    axiom::testing::ExpectTensorEquals<float>(
        a, {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
}
