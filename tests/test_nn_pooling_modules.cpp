#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;
using namespace axiom::testing;

// ============================================================================
// MaxPool1d
// ============================================================================

TEST(NNPoolingModules, MaxPool1dBasic) {
    MaxPool1d pool(2);
    auto input = Tensor::randn({1, 3, 8});
    auto output = pool(input);
    // kernel_size=2, stride=2 (default=kernel_size), padding=0
    // L_out = (8 - 2) / 2 + 1 = 4
    EXPECT_EQ(output.shape()[0], 1u);
    EXPECT_EQ(output.shape()[1], 3u);
    EXPECT_EQ(output.shape()[2], 4u);
}

TEST(NNPoolingModules, MaxPool1dStrideDefault) {
    // stride=0 should default to kernel_size
    MaxPool1d pool(3, 0, 0);
    auto input = Tensor::randn({1, 2, 9});
    auto output = pool(input);
    // L_out = (9 - 3) / 3 + 1 = 3
    EXPECT_EQ(output.shape()[2], 3u);
}

// ============================================================================
// MaxPool2d
// ============================================================================

TEST(NNPoolingModules, MaxPool2dBasic) {
    MaxPool2d pool({2, 2});
    auto input = Tensor::randn({1, 3, 8, 8});
    auto output = pool(input);
    EXPECT_EQ(output.shape()[2], 4u);
    EXPECT_EQ(output.shape()[3], 4u);
}

// ============================================================================
// AvgPool1d
// ============================================================================

TEST(NNPoolingModules, AvgPool1dBasic) {
    AvgPool1d pool(2);
    auto input = Tensor::randn({1, 3, 8});
    auto output = pool(input);
    EXPECT_EQ(output.shape()[2], 4u);
}

// ============================================================================
// AvgPool2d
// ============================================================================

TEST(NNPoolingModules, AvgPool2dBasic) {
    AvgPool2d pool({2, 2});
    auto input = Tensor::randn({1, 3, 8, 8});
    auto output = pool(input);
    EXPECT_EQ(output.shape()[2], 4u);
    EXPECT_EQ(output.shape()[3], 4u);
}

// ============================================================================
// AdaptiveAvgPool1d
// ============================================================================

TEST(NNPoolingModules, AdaptiveAvgPool1d) {
    AdaptiveAvgPool1d pool(4);
    auto input = Tensor::randn({1, 3, 16});
    auto output = pool(input);
    EXPECT_EQ(output.shape()[2], 4u);
}

// ============================================================================
// AdaptiveAvgPool2d
// ============================================================================

TEST(NNPoolingModules, AdaptiveAvgPool2d) {
    AdaptiveAvgPool2d pool({4, 4});
    auto input = Tensor::randn({1, 3, 16, 16});
    auto output = pool(input);
    EXPECT_EQ(output.shape()[2], 4u);
    EXPECT_EQ(output.shape()[3], 4u);
}

// ============================================================================
// AdaptiveMaxPool1d
// ============================================================================

TEST(NNPoolingModules, AdaptiveMaxPool1d) {
    AdaptiveMaxPool1d pool(4);
    auto input = Tensor::randn({1, 3, 16});
    auto output = pool(input);
    EXPECT_EQ(output.shape()[2], 4u);
}

// ============================================================================
// AdaptiveMaxPool2d
// ============================================================================

TEST(NNPoolingModules, AdaptiveMaxPool2d) {
    AdaptiveMaxPool2d pool({4, 4});
    auto input = Tensor::randn({1, 3, 16, 16});
    auto output = pool(input);
    EXPECT_EQ(output.shape()[2], 4u);
    EXPECT_EQ(output.shape()[3], 4u);
}

// ============================================================================
// Known values test
// ============================================================================

TEST(NNPoolingModules, MaxPool1dKnownValues) {
    float data[] = {1, 3, 2, 4, 5, 1};
    auto input = Tensor::from_data(data, {1, 1, 6});

    MaxPool1d pool(2, 2); // kernel=2, stride=2
    auto output = pool(input);

    auto ptr = output.typed_data<float>();
    EXPECT_FLOAT_EQ(ptr[0], 3.0f); // max(1,3)
    EXPECT_FLOAT_EQ(ptr[1], 4.0f); // max(2,4)
    EXPECT_FLOAT_EQ(ptr[2], 5.0f); // max(5,1)
}
