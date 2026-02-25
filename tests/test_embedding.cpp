#include "axiom_test_utils.hpp"

TEST(Embedding, BasicLookup) {
    // 4 vocab, 3 embed_dim
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}.data(), {4, 3},
        true);

    auto indices =
        axiom::Tensor::from_data(std::vector<int64_t>{0, 2}.data(), {2}, true);

    auto result = axiom::ops::embedding(weight, indices);
    ASSERT_TRUE(result.shape() == axiom::Shape({2, 3}));
    EXPECT_FLOAT_EQ(result.item<float>({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 2}), 2.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1, 0}), 6.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1, 1}), 7.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1, 2}), 8.0f);
}

TEST(Embedding, BatchedIndices) {
    // 5 vocab, 2 embed_dim
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {5, 2}, true);

    // batch=2, seq_len=3
    auto indices = axiom::Tensor::from_data(
        std::vector<int64_t>{0, 1, 2, 3, 4, 0}.data(), {2, 3}, true);

    auto result = axiom::ops::embedding(weight, indices);
    ASSERT_TRUE(result.shape() == axiom::Shape({2, 3, 2}));
    // First batch, first token (index 0)
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 0, 1}), 1.0f);
    // Second batch, last token (index 0)
    EXPECT_FLOAT_EQ(result.item<float>({1, 2, 0}), 0.0f);
    EXPECT_FLOAT_EQ(result.item<float>({1, 2, 1}), 1.0f);
}

TEST(Embedding, SingleIndex) {
    auto weight = axiom::Tensor::from_data(
        std::vector<float>{10, 20, 30, 40}.data(), {2, 2}, true);

    auto indices =
        axiom::Tensor::from_data(std::vector<int64_t>{1}.data(), {1}, true);

    auto result = axiom::ops::embedding(weight, indices);
    ASSERT_TRUE(result.shape() == axiom::Shape({1, 2}));
    EXPECT_FLOAT_EQ(result.item<float>({0, 0}), 30.0f);
    EXPECT_FLOAT_EQ(result.item<float>({0, 1}), 40.0f);
}
