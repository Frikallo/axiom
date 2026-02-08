#include "axiom_test_utils.hpp"

TEST(Pooling, MaxPool1D) {
    auto x = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {1, 1, 6}, true);

    auto result = axiom::ops::max_pool1d(x, 2, 2, 0);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 3}))
        << "max_pool1d shape wrong";
    ASSERT_TRUE(result.item<float>({0, 0, 0}) == 2.0f)
        << "max_pool1d wrong at 0";
    ASSERT_TRUE(result.item<float>({0, 0, 1}) == 4.0f)
        << "max_pool1d wrong at 1";
    ASSERT_TRUE(result.item<float>({0, 0, 2}) == 6.0f)
        << "max_pool1d wrong at 2";
}

TEST(Pooling, MaxPool1DStride1) {
    auto x = axiom::Tensor::from_data(
        std::vector<float>{1, 3, 2, 4, 3, 5}.data(), {1, 1, 6}, true);

    auto result = axiom::ops::max_pool1d(x, 2, 1, 0);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 5}))
        << "max_pool1d stride=1 shape wrong";
    ASSERT_TRUE(result.item<float>({0, 0, 0}) == 3.0f) << "max(1,3) = 3";
    ASSERT_TRUE(result.item<float>({0, 0, 1}) == 3.0f) << "max(3,2) = 3";
    ASSERT_TRUE(result.item<float>({0, 0, 2}) == 4.0f) << "max(2,4) = 4";
}

TEST(Pooling, AvgPool1D) {
    auto x = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {1, 1, 6}, true);

    auto result = axiom::ops::avg_pool1d(x, 2, 2, 0, true);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 3}))
        << "avg_pool1d shape wrong";
    ASSERT_TRUE(result.item<float>({0, 0, 0}) == 1.5f)
        << "avg_pool1d wrong at 0";
    ASSERT_TRUE(result.item<float>({0, 0, 1}) == 3.5f)
        << "avg_pool1d wrong at 1";
    ASSERT_TRUE(result.item<float>({0, 0, 2}) == 5.5f)
        << "avg_pool1d wrong at 2";
}

TEST(Pooling, MaxPool2D) {
    // 1 batch, 1 channel, 4x4 input
    auto x =
        axiom::Tensor::from_data(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                    10, 11, 12, 13, 14, 15, 16}
                                     .data(),
                                 {1, 1, 4, 4}, true);

    auto result = axiom::ops::max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2, 2}))
        << "max_pool2d shape wrong";
    ASSERT_TRUE(result.item<float>({0, 0, 0, 0}) == 6.0f)
        << "max_pool2d wrong at 0,0";
    ASSERT_TRUE(result.item<float>({0, 0, 0, 1}) == 8.0f)
        << "max_pool2d wrong at 0,1";
    ASSERT_TRUE(result.item<float>({0, 0, 1, 0}) == 14.0f)
        << "max_pool2d wrong at 1,0";
    ASSERT_TRUE(result.item<float>({0, 0, 1, 1}) == 16.0f)
        << "max_pool2d wrong at 1,1";
}

TEST(Pooling, MaxPool2DStride1) {
    auto x = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {1, 1, 3, 3},
        true);

    auto result = axiom::ops::max_pool2d(x, {2, 2}, {1, 1}, {0, 0});
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2, 2}))
        << "max_pool2d stride=1 shape wrong";
    ASSERT_TRUE(result.item<float>({0, 0, 0, 0}) == 5.0f)
        << "max of top-left 2x2";
    ASSERT_TRUE(result.item<float>({0, 0, 0, 1}) == 6.0f)
        << "max of top-right 2x2";
    ASSERT_TRUE(result.item<float>({0, 0, 1, 0}) == 8.0f)
        << "max of bottom-left 2x2";
    ASSERT_TRUE(result.item<float>({0, 0, 1, 1}) == 9.0f)
        << "max of bottom-right 2x2";
}

TEST(Pooling, AvgPool2D) {
    auto x =
        axiom::Tensor::from_data(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                    10, 11, 12, 13, 14, 15, 16}
                                     .data(),
                                 {1, 1, 4, 4}, true);

    auto result = axiom::ops::avg_pool2d(x, {2, 2}, {2, 2}, {0, 0}, true);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2, 2}))
        << "avg_pool2d shape wrong";
    // (1+2+5+6)/4 = 3.5
    EXPECT_NEAR(result.item<float>({0, 0, 0, 0}), 3.5f, 1e-5)
        << "avg_pool2d wrong at 0,0";
    // (3+4+7+8)/4 = 5.5
    EXPECT_NEAR(result.item<float>({0, 0, 0, 1}), 5.5f, 1e-5)
        << "avg_pool2d wrong at 0,1";
    // (9+10+13+14)/4 = 11.5
    EXPECT_NEAR(result.item<float>({0, 0, 1, 0}), 11.5f, 1e-5)
        << "avg_pool2d wrong at 1,0";
    // (11+12+15+16)/4 = 13.5
    EXPECT_NEAR(result.item<float>({0, 0, 1, 1}), 13.5f, 1e-5)
        << "avg_pool2d wrong at 1,1";
}

TEST(Pooling, MaxPool2DMultichannel) {
    // 1 batch, 2 channels, 4x4 input
    std::vector<float> data(2 * 4 * 4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    auto x = axiom::Tensor::from_data(data.data(), {1, 2, 4, 4}, true);
    auto result = axiom::ops::max_pool2d(x, {2, 2}, {2, 2}, {0, 0});

    ASSERT_TRUE(result.shape() == axiom::Shape({1, 2, 2, 2}))
        << "multichannel max_pool2d shape wrong";
}

TEST(Pooling, AdaptiveAvgPool2D) {
    auto x = axiom::Tensor::randn({1, 3, 8, 8});

    auto result = axiom::ops::adaptive_avg_pool2d(x, {1, 1});
    ASSERT_TRUE(result.shape()[2] == 1 && result.shape()[3] == 1)
        << "adaptive_avg_pool2d should reduce to 1x1";
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 3, 1, 1}))
        << "adaptive_avg_pool2d output shape wrong";
}

TEST(Pooling, AdaptiveAvgPool2DLarger) {
    auto x = axiom::Tensor::randn({2, 4, 16, 16});

    auto result = axiom::ops::adaptive_avg_pool2d(x, {4, 4});
    ASSERT_TRUE(result.shape() == axiom::Shape({2, 4, 4, 4}))
        << "adaptive_avg_pool2d to 4x4 shape wrong";
}

TEST(Pooling, AdaptiveMaxPool2D) {
    auto x = axiom::Tensor::randn({1, 2, 8, 8});

    auto result = axiom::ops::adaptive_max_pool2d(x, {1, 1});
    ASSERT_TRUE(result.shape()[2] == 1 && result.shape()[3] == 1)
        << "adaptive_max_pool2d should reduce to 1x1";
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 2, 1, 1}))
        << "adaptive_max_pool2d output shape wrong";
}

TEST(Pooling, MaxPool3D) {
    // 1 batch, 1 channel, 4x4x4 input
    std::vector<float> data(64);
    for (size_t i = 0; i < 64; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto x = axiom::Tensor::from_data(data.data(), {1, 1, 4, 4, 4}, true);
    auto result = axiom::ops::max_pool3d(x, {2, 2, 2}, {2, 2, 2}, {0, 0, 0});

    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2, 2, 2}))
        << "max_pool3d shape wrong";

    // First element should be max of indices 0,1,4,5,16,17,20,21
    // = max(0,1,4,5,16,17,20,21) = 21
    ASSERT_TRUE(result.item<float>({0, 0, 0, 0, 0}) == 21.0f)
        << "max_pool3d wrong at 0,0,0";
}

TEST(Pooling, AvgPool3D) {
    std::vector<float> data(64);
    for (size_t i = 0; i < 64; ++i) {
        data[i] = static_cast<float>(i);
    }

    auto x = axiom::Tensor::from_data(data.data(), {1, 1, 4, 4, 4}, true);
    auto result =
        axiom::ops::avg_pool3d(x, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, true);

    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2, 2, 2}))
        << "avg_pool3d shape wrong";

    // First element should be mean of indices 0,1,4,5,16,17,20,21
    // = (0+1+4+5+16+17+20+21)/8 = 84/8 = 10.5
    EXPECT_NEAR(result.item<float>({0, 0, 0, 0, 0}), 10.5f, 1e-4)
        << "avg_pool3d wrong at 0,0,0";
}
