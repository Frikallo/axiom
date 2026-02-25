#include "axiom_test_utils.hpp"

TEST(Conv2D, NoPadding) {
    // Input: (1, 1, 3, 3), Weight: (1, 1, 2, 2)
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {1, 1, 3, 3},
        true);
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{1, 1, 1, 1}.data(), {1, 1, 2, 2}, true);

    auto result = axiom::ops::conv2d(input, weight);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2, 2}));
    // 1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0, 0}), 12.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0, 1}), 16.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1, 0}), 24.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1, 1}), 28.0f);
}

TEST(Conv2D, WithPadding) {
    // Input: (1, 1, 3, 3), Weight: (1, 1, 3, 3), padding=1
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {1, 1, 3, 3},
        true);
    auto weight = axiom::Tensor::ones({1, 1, 3, 3});

    auto result =
        axiom::ops::conv2d(input, weight, axiom::Tensor(), {1, 1}, {1, 1});
    // Same spatial size with padding=1 and kernel=3
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 3, 3}));
    // Center element: full 3x3 sum = 45
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1, 1}), 45.0f);
}

TEST(Conv2D, Stride2) {
    // Input: (1, 1, 4, 4), Weight: (1, 1, 2, 2), stride=2
    std::vector<float> data(16);
    for (int i = 0; i < 16; ++i)
        data[i] = static_cast<float>(i + 1);
    auto input = axiom::Tensor::from_data(data.data(), {1, 1, 4, 4}, true);
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{1, 1, 1, 1}.data(), {1, 1, 2, 2}, true);

    auto result = axiom::ops::conv2d(input, weight, axiom::Tensor(), {2, 2});
    // H_out = (4 - 2) / 2 + 1 = 2, W_out = 2
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2, 2}));
    // (1+2+5+6)=14, (3+4+7+8)=22, (9+10+13+14)=46, (11+12+15+16)=54
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0, 0}), 14.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1, 0}), 46.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1, 1}), 54.0f);
}

TEST(Conv2D, Dilation) {
    // Input: (1, 1, 5, 5), Weight: (1, 1, 2, 2), dilation=2
    std::vector<float> data(25);
    for (int i = 0; i < 25; ++i)
        data[i] = static_cast<float>(i + 1);
    auto input = axiom::Tensor::from_data(data.data(), {1, 1, 5, 5}, true);
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{1, 1, 1, 1}.data(), {1, 1, 2, 2}, true);

    auto result = axiom::ops::conv2d(input, weight, axiom::Tensor(), {1, 1},
                                     {0, 0}, {2, 2});
    // dilated kernel: effective 3x3, H_out = (5 - 3)/1 + 1 = 3
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 3, 3}));
    // pos (0,0): input[0,0]+input[0,2]+input[2,0]+input[2,2] = 1+3+11+13 = 28
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0, 0}), 28.0f);
}

TEST(Conv2D, WithBias) {
    auto input = axiom::Tensor::from_data(std::vector<float>{1, 2, 3, 4}.data(),
                                          {1, 1, 2, 2}, true);
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{1, 1, 1, 1}.data(), {1, 1, 2, 2}, true);
    auto bias =
        axiom::Tensor::from_data(std::vector<float>{10.0f}.data(), {1}, true);

    auto result = axiom::ops::conv2d(input, weight, bias);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 1, 1}));
    // 1+2+3+4+10 = 20
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0, 0}), 20.0f);
}

TEST(Conv2D, MultiChannel) {
    // 2 input channels, 3 output channels
    // Input: (1, 2, 3, 3)
    std::vector<float> in_data(18);
    for (int i = 0; i < 18; ++i)
        in_data[i] = static_cast<float>(i + 1);
    auto input = axiom::Tensor::from_data(in_data.data(), {1, 2, 3, 3}, true);
    // Weight: (3, 2, 2, 2) — all ones
    std::vector<float> w_data(24, 1.0f);
    auto weight = axiom::Tensor::from_data(w_data.data(), {3, 2, 2, 2}, true);

    auto result = axiom::ops::conv2d(input, weight);
    // H_out = (3-2)/1+1 = 2, W_out = 2
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 3, 2, 2}));
}

TEST(Conv2D, Unbatched) {
    // Input: (1, 3, 3) — no batch dim
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {1, 3, 3}, true);
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{1, 1, 1, 1}.data(), {1, 1, 2, 2}, true);

    auto result = axiom::ops::conv2d(input, weight);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 2, 2}));
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 12.0f);
}

TEST(Conv2D, Groups) {
    // groups=2: input (1, 4, 3, 3), weight (4, 2, 1, 1)
    std::vector<float> in_data(36);
    for (int i = 0; i < 36; ++i)
        in_data[i] = static_cast<float>(i + 1);
    auto input = axiom::Tensor::from_data(in_data.data(), {1, 4, 3, 3}, true);
    // Weight: (4, 2, 1, 1) — all ones
    std::vector<float> w_data(8, 1.0f);
    auto weight = axiom::Tensor::from_data(w_data.data(), {4, 2, 1, 1}, true);

    auto result = axiom::ops::conv2d(input, weight, axiom::Tensor(), {1, 1},
                                     {0, 0}, {1, 1}, 2);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 4, 3, 3}));
}
