#include "axiom_test_utils.hpp"
#include <cmath>
#include <limits>

TEST(Sampling, TemperatureIdentity) {
    auto logits = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, true);

    auto result = axiom::sampling::temperature_scale(logits, 1.0f);
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 2.0f, 3.0f});
}

TEST(Sampling, TemperatureScaling) {
    auto logits = axiom::Tensor::from_data(
        std::vector<float>{2.0f, 4.0f, 6.0f}.data(), {3}, true);

    auto result = axiom::sampling::temperature_scale(logits, 2.0f);
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 2.0f, 3.0f});
}

TEST(Sampling, TopKKeepsMax) {
    auto logits = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 5.0f, 3.0f, 2.0f}.data(), {4}, true);

    auto result = axiom::sampling::top_k(logits, 1);
    auto cpu = result.cpu();
    const float *data = cpu.typed_data<float>();

    // Only the max (5.0) should remain, rest -inf
    EXPECT_FLOAT_EQ(data[1], 5.0f);
    EXPECT_TRUE(std::isinf(data[0]) && data[0] < 0);
    EXPECT_TRUE(std::isinf(data[2]) && data[2] < 0);
    EXPECT_TRUE(std::isinf(data[3]) && data[3] < 0);
}

TEST(Sampling, TopKKeepsMultiple) {
    auto logits = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 5.0f, 3.0f, 2.0f}.data(), {4}, true);

    auto result = axiom::sampling::top_k(logits, 2);
    auto cpu = result.cpu();
    const float *data = cpu.typed_data<float>();

    // Top 2: 5.0 and 3.0 should remain
    EXPECT_FLOAT_EQ(data[1], 5.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_TRUE(std::isinf(data[0]) && data[0] < 0);
    EXPECT_TRUE(std::isinf(data[3]) && data[3] < 0);
}

TEST(Sampling, TopKBatched) {
    // 2 rows of 4 logits each
    auto logits = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 5.0f, 3.0f, 2.0f, 10.0f, 1.0f, 8.0f, 3.0f}
            .data(),
        {2, 4}, true);

    auto result = axiom::sampling::top_k(logits, 1);
    auto cpu = result.cpu();
    const float *data = cpu.typed_data<float>();

    // Row 0: max is 5.0 at index 1
    EXPECT_FLOAT_EQ(data[1], 5.0f);
    EXPECT_TRUE(std::isinf(data[0]) && data[0] < 0);

    // Row 1: max is 10.0 at index 0
    EXPECT_FLOAT_EQ(data[4], 10.0f);
    EXPECT_TRUE(std::isinf(data[5]) && data[5] < 0);
}

TEST(Sampling, TopPKeepsAll) {
    auto logits = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f}.data(), {3}, true);

    auto result = axiom::sampling::top_p(logits, 1.0f);
    axiom::testing::ExpectTensorEquals<float>(result, {1.0f, 2.0f, 3.0f});
}

TEST(Sampling, TopPNucleus) {
    // Logits where one token dominates: softmax(10, 1, 1) ≈ (0.9998, 0.0001,
    // 0.0001)
    auto logits = axiom::Tensor::from_data(
        std::vector<float>{10.0f, 1.0f, 1.0f}.data(), {3}, true);

    auto result = axiom::sampling::top_p(logits, 0.9f);
    auto cpu = result.cpu();
    const float *data = cpu.typed_data<float>();

    // The dominant logit (10.0) should be kept
    EXPECT_FLOAT_EQ(data[0], 10.0f);
    // Others masked to -inf since cumsum of first token exceeds 0.9
    EXPECT_TRUE(std::isinf(data[1]) && data[1] < 0);
    EXPECT_TRUE(std::isinf(data[2]) && data[2] < 0);
}
