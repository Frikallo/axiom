#include "axiom_test_utils.hpp"
#include <axiom/einops.hpp>
#include <vector>

using namespace axiom;

// ============================================================================
// numel() and shape_str() Tests
// ============================================================================

TEST(TensorConvenience, Numel) {
    auto t = Tensor::zeros({2, 3, 4});
    EXPECT_EQ(t.numel(), 24u);
    EXPECT_EQ(t.numel(), t.size());
}

TEST(TensorConvenience, NumelScalar) {
    auto t = Tensor::zeros({1});
    EXPECT_EQ(t.numel(), 1u);
}

TEST(TensorConvenience, ShapeStr) {
    auto t = Tensor::zeros({2, 3, 4});
    EXPECT_EQ(t.shape_str(), "(2, 3, 4)");
}

TEST(TensorConvenience, ShapeStr1D) {
    auto t = Tensor::zeros({5});
    EXPECT_EQ(t.shape_str(), "(5)");
}

TEST(TensorConvenience, ShapeStrScalar) {
    auto t = Tensor::zeros({});
    EXPECT_EQ(t.shape_str(), "()");
}

// ============================================================================
// einops::repeat() Tests
// ============================================================================

TEST(EinopsRepeat, AddNewAxis) {
    // Add a channel dimension to a 2D image
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor::from_data(data.data(), {2, 2});

    auto result = einops::repeat(x, "h w -> h w c", {{"c", 3}});

    ASSERT_TRUE(result.shape() == Shape({2, 2, 3}));
    // Each pixel value is repeated 3 times along the new axis
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 2}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1, 1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1, 1, 2}), 4.0f);
}

TEST(EinopsRepeat, TileRows) {
    // Repeat each row r times
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor::from_data(data.data(), {2, 2});

    auto result = einops::repeat(x, "h w -> (h r) w", {{"r", 3}});

    ASSERT_TRUE(result.shape() == Shape({6, 2}));
    // Row 0 repeated 3 times, then row 1 repeated 3 times
    EXPECT_FLOAT_EQ(result.item<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({2, 0}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({3, 0}), 3.0f);
    EXPECT_FLOAT_EQ(result.item<float>({4, 0}), 3.0f);
    EXPECT_FLOAT_EQ(result.item<float>({5, 0}), 3.0f);
}

TEST(EinopsRepeat, TensorMethod) {
    // Test Tensor::repeat(string, map) method
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto x = Tensor::from_data(data.data(), {3});

    auto result = x.repeat("w -> h w", {{"h", 4}});

    ASSERT_TRUE(result.shape() == Shape({4, 3}));
    for (size_t h = 0; h < 4; ++h) {
        EXPECT_FLOAT_EQ(result.item<float>({h, 0}), 1.0f);
        EXPECT_FLOAT_EQ(result.item<float>({h, 1}), 2.0f);
        EXPECT_FLOAT_EQ(result.item<float>({h, 2}), 3.0f);
    }
}

TEST(EinopsRepeat, Interleave) {
    // Interleave: insert repeat dim adjacent to existing
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(data.data(), {2, 3});

    auto result = einops::repeat(x, "h w -> h (w r)", {{"r", 2}});

    ASSERT_TRUE(result.shape() == Shape({2, 6}));
    // w=0 repeated, then w=1 repeated, etc.
    EXPECT_FLOAT_EQ(result.item<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 2}), 2.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 3}), 2.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 4}), 3.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 5}), 3.0f);
}

// ============================================================================
// einops::pack() / unpack() Tests
// ============================================================================

TEST(EinopsPack, BasicPack) {
    auto a = Tensor::ones({3, 4});
    auto b = Tensor::ones({5, 4}) * 2.0f;

    auto [packed, ps] = einops::pack({a, b}, "* w");

    ASSERT_TRUE(packed.shape() == Shape({8, 4}));
    EXPECT_EQ(ps.size(), 2u);
    EXPECT_TRUE(ps[0] == Shape({3}));
    EXPECT_TRUE(ps[1] == Shape({5}));
}

TEST(EinopsPack, Roundtrip) {
    auto a = Tensor::ones({3, 4}) * 1.0f;
    auto b = Tensor::ones({5, 4}) * 2.0f;

    auto [packed, ps] = einops::pack({a, b}, "* w");
    auto unpacked = einops::unpack(packed, ps, "* w");

    ASSERT_EQ(unpacked.size(), 2u);
    ASSERT_TRUE(unpacked[0].shape() == Shape({3, 4}));
    ASSERT_TRUE(unpacked[1].shape() == Shape({5, 4}));

    // Verify values
    EXPECT_FLOAT_EQ(unpacked[0].item<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(unpacked[1].item<float>({0, 0}), 2.0f);
}

TEST(EinopsPack, MultiDimStar) {
    // 4D tensors with batch and channel as star dims
    auto a = Tensor::ones({2, 3, 8, 8});
    auto b = Tensor::ones({4, 8, 8});

    auto [packed, ps] = einops::pack({a, b}, "* h w");

    // a has star dims (2,3) -> flat 6, b has star dim (4) -> flat 4
    // packed star dim = 6+4 = 10
    ASSERT_TRUE(packed.shape() == Shape({10, 8, 8}));
    EXPECT_TRUE(ps[0] == Shape({2, 3}));
    EXPECT_TRUE(ps[1] == Shape({4}));

    auto unpacked = einops::unpack(packed, ps, "* h w");
    ASSERT_TRUE(unpacked[0].shape() == Shape({2, 3, 8, 8}));
    ASSERT_TRUE(unpacked[1].shape() == Shape({4, 8, 8}));
}

TEST(EinopsPack, TrailingFixed) {
    // Pattern with both leading and trailing fixed dims
    auto a = Tensor::ones({2, 3, 4});
    auto b = Tensor::ones({2, 5, 4});

    auto [packed, ps] = einops::pack({a, b}, "batch * features");

    ASSERT_TRUE(packed.shape() == Shape({2, 8, 4}));
    EXPECT_TRUE(ps[0] == Shape({3}));
    EXPECT_TRUE(ps[1] == Shape({5}));
}

// ============================================================================
// Anonymous axis _ Tests
// ============================================================================

TEST(EinopsAnonymous, ReduceWithUnderscore) {
    // Use _ for an axis we don't care to name
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(data.data(), {2, 3});

    // Sum over the second dimension (anonymous)
    auto result = einops::reduce(x, "h _ -> h", "sum");

    ASSERT_TRUE(result.shape() == Shape({2}));
    axiom::testing::ExpectTensorEquals<float>(result, {6.0f, 15.0f});
}

TEST(EinopsAnonymous, MultipleUnderscores) {
    // Multiple _ get unique names
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i)
        data[i] = static_cast<float>(i);
    auto x = Tensor::from_data(data.data(), {2, 3, 4});

    // Sum over both anonymous dims
    auto result = einops::reduce(x, "b _ _ -> b", "sum");

    ASSERT_TRUE(result.shape() == Shape({2}));
    // batch 0: sum(0..11) = 66, batch 1: sum(12..23) = 210
    EXPECT_FLOAT_EQ(result.item<float>({0}), 66.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1}), 210.0f);
}

TEST(EinopsAnonymous, RearrangeWithUnderscore) {
    // _ used as a dimension we don't want to name in a flatten pattern
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(data.data(), {2, 3});

    // Reduce anonymous axis via sum
    auto result = einops::reduce(x, "h _ -> h", "mean");

    ASSERT_TRUE(result.shape() == Shape({2}));
    // Row 0: mean(1,2,3) = 2, Row 1: mean(4,5,6) = 5
    EXPECT_FLOAT_EQ(result.item<float>({0}), 2.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1}), 5.0f);
}
