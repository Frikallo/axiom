#include "axiom_test_utils.hpp"

// Test arange
TEST(TensorShapeOps, ArangeBasic) {
    auto t = axiom::Tensor::arange(5);
    ASSERT_TRUE(t.shape() == axiom::Shape({5})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<int>(t, {0, 1, 2, 3, 4});
}

TEST(TensorShapeOps, ArangeStartEnd) {
    auto t = axiom::Tensor::arange(2, 8);
    ASSERT_TRUE(t.shape() == axiom::Shape({6})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<int>(t, {2, 3, 4, 5, 6, 7});
}

TEST(TensorShapeOps, ArangeWithStep) {
    auto t = axiom::Tensor::arange(0, 10, 2);
    ASSERT_TRUE(t.shape() == axiom::Shape({5})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<int>(t, {0, 2, 4, 6, 8});
}

// Test flatten
TEST(TensorShapeOps, FlattenDefault) {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4});
    auto flat = t.flatten();
    ASSERT_TRUE(flat.ndim() == 1) << "Should be 1D";
    ASSERT_TRUE(flat.shape()[0] == 24) << "Size mismatch";
}

TEST(TensorShapeOps, FlattenPartial) {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4});
    auto flat = t.flatten(1, 2); // Flatten dims 1 and 2
    ASSERT_TRUE(flat.shape() == axiom::Shape({2, 12})) << "Shape mismatch";
}

TEST(TensorShapeOps, FlattenNegativeIndex) {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4});
    auto flat = t.flatten(0, -1); // Flatten all dims
    ASSERT_TRUE(flat.ndim() == 1) << "Should be 1D";
    ASSERT_TRUE(flat.shape()[0] == 24) << "Size mismatch";
}

// Test expand (zero-copy broadcast)
TEST(TensorShapeOps, ExpandBasic) {
    auto t = axiom::Tensor::ones({1, 4});
    auto expanded = t.expand({3, 4});

    ASSERT_TRUE(expanded.shape() == axiom::Shape({3, 4})) << "Shape mismatch";
    ASSERT_TRUE(expanded.has_zero_stride()) << "Should have zero stride";

    // Verify data
    std::vector<float> expected(12, 1.0f);
    axiom::testing::ExpectTensorEquals<float>(expanded, expected);
}

TEST(TensorShapeOps, ExpandMultidim) {
    auto t = axiom::Tensor::ones({1, 1, 4});
    auto expanded = t.expand({2, 3, 4});

    ASSERT_TRUE(expanded.shape() == axiom::Shape({2, 3, 4}))
        << "Shape mismatch";
    ASSERT_TRUE(expanded.has_zero_stride()) << "Should have zero stride";
}

TEST(TensorShapeOps, ExpandAs) {
    auto t = axiom::Tensor::ones({1, 4});
    auto target = axiom::Tensor::zeros({3, 4});
    auto expanded = t.expand_as(target);

    ASSERT_TRUE(expanded.shape() == target.shape())
        << "Shape should match target";
}

TEST(TensorShapeOps, BroadcastTo) {
    auto t = axiom::Tensor::ones({1, 4});
    auto broadcasted = t.broadcast_to({3, 4});

    ASSERT_TRUE(broadcasted.shape() == axiom::Shape({3, 4}))
        << "Shape mismatch";
    ASSERT_TRUE(broadcasted.has_zero_stride()) << "Should have zero stride";
}

// Test repeat (copies data)
TEST(TensorShapeOps, RepeatBasic) {
    auto t = axiom::Tensor::arange(4).reshape({2, 2});
    auto repeated = t.repeat({2, 3});

    ASSERT_TRUE(repeated.shape() == axiom::Shape({4, 6})) << "Shape mismatch";
    ASSERT_TRUE(!repeated.has_zero_stride())
        << "Should NOT have zero stride (data copied)";
}

TEST(TensorShapeOps, RepeatSingleDim) {
    auto t = axiom::Tensor::arange(4);
    auto repeated = t.repeat({3});

    ASSERT_TRUE(repeated.shape() == axiom::Shape({12})) << "Shape mismatch";
    // Expected: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    axiom::testing::ExpectTensorEquals<int>(
        repeated, {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
}

TEST(TensorShapeOps, TileAlias) {
    auto t = axiom::Tensor::arange(4);
    auto tiled = t.tile({2});

    ASSERT_TRUE(tiled.shape() == axiom::Shape({8})) << "Shape mismatch";
}

// Test from_data
TEST(TensorShapeOps, FromData) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 3});

    ASSERT_TRUE(t.shape() == axiom::Shape({2, 3})) << "Shape mismatch";
    axiom::testing::ExpectTensorEquals<float>(t, data);
}

// Test rearrange (if implemented)
TEST(TensorShapeOps, RearrangeFlatten) {
    auto t =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto r = t.rearrange("h w -> (h w)");

    ASSERT_TRUE(r.ndim() == 1) << "Should be 1D";
    ASSERT_TRUE(r.shape()[0] == 6) << "Size mismatch";
}

TEST(TensorShapeOps, RearrangeTranspose) {
    auto t =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto r = t.rearrange("h w -> w h");

    ASSERT_TRUE(r.shape() == axiom::Shape({3, 2})) << "Shape mismatch";
}
