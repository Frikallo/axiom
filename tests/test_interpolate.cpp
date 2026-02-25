#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;
using namespace axiom::testing;

// ============================================================================
// Nearest interpolation — exact 2x scale
// ============================================================================

TEST(Interpolate, Nearest2x) {
    // (1, 1, 2, 2) → (1, 1, 4, 4) via nearest
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Tensor::from_data(data, {1, 1, 2, 2});

    auto output =
        ops::interpolate(input, {4, 4}, {}, ops::InterpolateMode::Nearest);
    EXPECT_EQ(output.shape()[2], 4u);
    EXPECT_EQ(output.shape()[3], 4u);

    auto ptr = output.typed_data<float>();
    // Row 0: [1,1,2,2], Row 1: [1,1,2,2], Row 2: [3,3,4,4], Row 3: [3,3,4,4]
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 2.0f);
    EXPECT_FLOAT_EQ(ptr[3], 2.0f);
    EXPECT_FLOAT_EQ(ptr[4], 1.0f);
    EXPECT_FLOAT_EQ(ptr[5], 1.0f);
    EXPECT_FLOAT_EQ(ptr[8], 3.0f);
    EXPECT_FLOAT_EQ(ptr[10], 4.0f);
}

// ============================================================================
// Bilinear interpolation — known case
// ============================================================================

TEST(Interpolate, BilinearSmall) {
    float data[] = {0.0f, 1.0f, 2.0f, 3.0f};
    auto input = Tensor::from_data(data, {1, 1, 2, 2});

    auto output = ops::interpolate(input, {4, 4}, {},
                                   ops::InterpolateMode::Bilinear, true);
    EXPECT_EQ(output.shape()[2], 4u);
    EXPECT_EQ(output.shape()[3], 4u);

    auto ptr = output.typed_data<float>();
    // With align_corners=true:
    // corners should be preserved: (0,0)=0, (0,3)=1, (3,0)=2, (3,3)=3
    EXPECT_NEAR(ptr[0], 0.0f, 1e-4);
    EXPECT_NEAR(ptr[3], 1.0f, 1e-4);
    EXPECT_NEAR(ptr[12], 2.0f, 1e-4);
    EXPECT_NEAR(ptr[15], 3.0f, 1e-4);
}

// ============================================================================
// Scale factor mode
// ============================================================================

TEST(Interpolate, ScaleFactor) {
    auto input = Tensor::randn({1, 3, 4, 4});
    auto output = ops::interpolate(input, {}, {2.0f, 2.0f},
                                   ops::InterpolateMode::Nearest);
    EXPECT_EQ(output.shape()[2], 8u);
    EXPECT_EQ(output.shape()[3], 8u);
}

// ============================================================================
// 1D interpolation
// ============================================================================

TEST(Interpolate, Nearest1D) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Tensor::from_data(data, {1, 1, 4});

    auto output =
        ops::interpolate(input, {8}, {}, ops::InterpolateMode::Nearest);
    EXPECT_EQ(output.shape()[2], 8u);

    auto ptr = output.typed_data<float>();
    // Each element should be repeated twice
    EXPECT_FLOAT_EQ(ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr[1], 1.0f);
    EXPECT_FLOAT_EQ(ptr[2], 2.0f);
    EXPECT_FLOAT_EQ(ptr[3], 2.0f);
}

// ============================================================================
// Bicubic interpolation
// ============================================================================

TEST(Interpolate, BicubicBasic) {
    auto input = Tensor::randn({1, 1, 4, 4});
    auto output =
        ops::interpolate(input, {8, 8}, {}, ops::InterpolateMode::Bicubic);
    EXPECT_EQ(output.shape()[2], 8u);
    EXPECT_EQ(output.shape()[3], 8u);
}

// ============================================================================
// Upsample module — size-based
// ============================================================================

TEST(Interpolate, UpsampleModuleSize) {
    Upsample up(std::vector<size_t>{8, 8});
    auto input = Tensor::randn({1, 3, 4, 4});
    auto output = up(input);
    EXPECT_EQ(output.shape()[2], 8u);
    EXPECT_EQ(output.shape()[3], 8u);
}

// ============================================================================
// Upsample module — scale-based
// ============================================================================

TEST(Interpolate, UpsampleModuleScale) {
    Upsample up(std::vector<float>{2.0f, 2.0f});
    auto input = Tensor::randn({1, 3, 4, 4});
    auto output = up(input);
    EXPECT_EQ(output.shape()[2], 8u);
    EXPECT_EQ(output.shape()[3], 8u);
}

// ============================================================================
// Upsample module — bilinear with align_corners
// ============================================================================

TEST(Interpolate, UpsampleBilinearAlignCorners) {
    Upsample up(std::vector<size_t>{6, 6}, ops::InterpolateMode::Bilinear,
                true);
    auto input = Tensor::randn({1, 1, 3, 3});
    auto output = up(input);
    EXPECT_EQ(output.shape()[2], 6u);
    EXPECT_EQ(output.shape()[3], 6u);
}

// ============================================================================
// GPU parity
// ============================================================================

TEST(Interpolate, NearestGPU) {
    SKIP_IF_NO_GPU();
    auto input = Tensor::randn({1, 3, 4, 4});
    auto cpu_out =
        ops::interpolate(input, {8, 8}, {}, ops::InterpolateMode::Nearest);
    auto gpu_out = ops::interpolate(input.gpu(), {8, 8}, {},
                                    ops::InterpolateMode::Nearest);
    EXPECT_EQ(gpu_out.device(), Device::GPU);
    ExpectTensorsClose(cpu_out, gpu_out.cpu(), 1e-5, 1e-5);
}

TEST(Interpolate, BilinearGPU) {
    SKIP_IF_NO_GPU();
    auto input = Tensor::randn({1, 3, 4, 4});
    auto cpu_out = ops::interpolate(input, {8, 8}, {},
                                    ops::InterpolateMode::Bilinear, true);
    auto gpu_out = ops::interpolate(input.gpu(), {8, 8}, {},
                                    ops::InterpolateMode::Bilinear, true);
    EXPECT_EQ(gpu_out.device(), Device::GPU);
    ExpectTensorsClose(cpu_out, gpu_out.cpu(), 1e-4, 1e-4);
}

// ============================================================================
// Error cases
// ============================================================================

TEST(Interpolate, NoSizeOrScale) {
    auto input = Tensor::randn({1, 3, 4, 4});
    EXPECT_THROW(ops::interpolate(input), ValueError);
}

TEST(Interpolate, WrongSizeLength) {
    auto input = Tensor::randn({1, 3, 4, 4});
    // 2D spatial but size has 1 element
    EXPECT_THROW(ops::interpolate(input, {8}), ValueError);
}
