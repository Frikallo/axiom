// Tests for tensor view/slice operations

#include "axiom_test_utils.hpp"

using namespace axiom;

TEST(TensorViews, SimpleSlice) {
    auto a_data = std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7};
    auto a = Tensor::from_data(a_data.data(), {8});

    // Slice from index 2 to 5: {2, 3, 4}
    auto b = a.slice({Slice(2, 5)});

    ASSERT_TRUE(b.shape() == Shape({3})) << "Simple slice shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(b, {2, 3, 4});
}

TEST(TensorViews, MultiDimSlice) {
    auto a_data = std::vector<float>{0, 1, 2,  3,  4,  5,  6,  7,
                                     8, 9, 10, 11, 12, 13, 14, 15};
    auto a = Tensor::from_data(a_data.data(), {4, 4});

    // Slice rows 1-3 and columns 2-4
    // Corresponds to numpy's a[1:3, 2:4]
    auto b = a.slice({Slice(1, 3), Slice(2, 4)});

    ASSERT_TRUE(b.shape() == Shape({2, 2})) << "Multi-dim slice shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(b, {6, 7, 10, 11});
}

TEST(TensorViews, StepSlice) {
    auto a_data = std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7};
    auto a = Tensor::from_data(a_data.data(), {8});

    // Slice from index 1 to 7 with a step of 2: {1, 3, 5}
    auto b = a.slice({Slice(1, 7, 2)});

    ASSERT_TRUE(b.shape() == Shape({3})) << "Step slice shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(b, {1, 3, 5});
}

TEST(TensorViews, SharedStorage) {
    auto a = Tensor::zeros({4, 4});

    // Create a view of the center 2x2
    auto b = a.slice({Slice(1, 3), Slice(1, 3)});

    // Fill the view with 1s
    b.fill(1.0f);

    // The original tensor should now be modified
    axiom::testing::ExpectTensorEquals<float>(
        a, {0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0});
}

TEST(TensorViews, SliceBoundsError) {
    auto a = Tensor::zeros({5, 5});
    // Out-of-bounds slice is clamped (no exception thrown)
    EXPECT_NO_THROW(a.slice({Slice(0, 10)}));

    // Too many slice arguments
    EXPECT_THROW(a.slice({Slice(0, 1), Slice(0, 1), Slice(0, 1)}),
                 std::exception);
}
