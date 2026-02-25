#include "axiom_test_utils.hpp"

TEST(Conv1D, NoPadding) {
    // Input: (1, 1, 5), Weight: (1, 1, 3)
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5}.data(), {1, 1, 5}, true);
    auto weight = axiom::Tensor::from_data(std::vector<float>{1, 1, 1}.data(),
                                           {1, 1, 3}, true);

    auto result = axiom::ops::conv1d(input, weight);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 3}));
    // Sliding sum: 1+2+3=6, 2+3+4=9, 3+4+5=12
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1}), 9.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 2}), 12.0f);
}

TEST(Conv1D, WithPadding) {
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5}.data(), {1, 1, 5}, true);
    auto weight = axiom::Tensor::from_data(std::vector<float>{1, 1, 1}.data(),
                                           {1, 1, 3}, true);

    auto result = axiom::ops::conv1d(input, weight, axiom::Tensor(), 1, 1);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 5}));
    // pad=1: [0,1,2,3,4,5,0]
    // 0+1+2=3, 1+2+3=6, 2+3+4=9, 3+4+5=12, 4+5+0=9
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1}), 6.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 2}), 9.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 3}), 12.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 4}), 9.0f);
}

TEST(Conv1D, Stride2) {
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {1, 1, 6}, true);
    auto weight = axiom::Tensor::from_data(std::vector<float>{1, 1, 1}.data(),
                                           {1, 1, 3}, true);

    auto result = axiom::ops::conv1d(input, weight, axiom::Tensor(), 2);
    // L_out = (6 - 3) / 2 + 1 = 2
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 2}));
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 6.0f);  // 1+2+3
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1}), 12.0f); // 3+4+5
}

TEST(Conv1D, Dilation) {
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5}.data(), {1, 1, 5}, true);
    auto weight = axiom::Tensor::from_data(std::vector<float>{1, 1, 1}.data(),
                                           {1, 1, 3}, true);

    // dilation=2: effective kernel positions at 0, 2, 4
    auto result = axiom::ops::conv1d(input, weight, axiom::Tensor(), 1, 0, 2);
    // dilated_kernel = 2*(3-1)+1 = 5
    // L_out = (5 - 5)/1 + 1 = 1
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 1}));
    // input[0]+input[2]+input[4] = 1+3+5 = 9
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 9.0f);
}

TEST(Conv1D, WithBias) {
    auto input = axiom::Tensor::from_data(std::vector<float>{1, 2, 3}.data(),
                                          {1, 1, 3}, true);
    auto weight = axiom::Tensor::from_data(std::vector<float>{1, 1, 1}.data(),
                                           {1, 1, 3}, true);
    auto bias =
        axiom::Tensor::from_data(std::vector<float>{10.0f}.data(), {1}, true);

    auto result = axiom::ops::conv1d(input, weight, bias);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 1, 1}));
    // 1+2+3+10 = 16
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 16.0f);
}

TEST(Conv1D, MultiChannel) {
    // 2 input channels, 3 output channels
    // Input: (1, 2, 4)
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}.data(), {1, 2, 4}, true);
    // Weight: (3, 2, 2)
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1}.data(),
        {3, 2, 2}, true);

    auto result = axiom::ops::conv1d(input, weight);
    // L_out = (4-2)/1+1 = 3
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 3, 3}));
}

TEST(Conv1D, Unbatched) {
    // Input: (1, 5) — no batch dim
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5}.data(), {1, 5}, true);
    auto weight = axiom::Tensor::from_data(std::vector<float>{1, 1, 1}.data(),
                                           {1, 1, 3}, true);

    auto result = axiom::ops::conv1d(input, weight);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 3}));
    EXPECT_FLOAT_EQ(result.item<float>({0, 0}), 6.0f);
}

TEST(Conv1D, Groups) {
    // groups=2: input (1, 4, 3), weight (4, 2, 1)
    // Group 0: input channels [0,1] -> output channels [0,1]
    // Group 1: input channels [2,3] -> output channels [2,3]
    auto input = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data(),
        {1, 4, 3}, true);
    // Weight: (4, 2, 1) — all ones
    std::vector<float> w_data(8, 1.0f);
    auto weight = axiom::Tensor::from_data(w_data.data(), {4, 2, 1}, true);

    auto result =
        axiom::ops::conv1d(input, weight, axiom::Tensor(), 1, 0, 1, 2);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 4, 3}));
}
