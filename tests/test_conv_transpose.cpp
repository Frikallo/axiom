#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;
using namespace axiom::testing;

// ============================================================================
// conv_transpose1d — functional API
// ============================================================================

TEST(ConvTranspose, Conv1dIdentityKernel) {
    // Weight: (1, 1, 1) — identity kernel with C_in=1, C_out=1, K=1
    // Output should equal input
    float w[] = {1.0f};
    auto weight = Tensor::from_data(w, {1, 1, 1});
    auto input = Tensor::randn({1, 1, 5});

    auto output = ops::conv_transpose1d(input, weight);
    ExpectTensorsClose(input, output, 1e-5, 1e-5);
}

TEST(ConvTranspose, Conv1dKnownSmall) {
    // input: (1, 1, 3) = [1, 2, 3]
    // weight: (1, 1, 2) = [1, 1]  (C_in=1, C_out=1, K=2)
    // stride=1, padding=0, output_padding=0
    // out_length = (3-1)*1 - 0 + 1*(2-1) + 0 + 1 = 4
    // Scatter:
    //   in[0]=1: out[0] += 1*1=1, out[1] += 1*1=1
    //   in[1]=2: out[1] += 2*1=2, out[2] += 2*1=2
    //   in[2]=3: out[2] += 3*1=3, out[3] += 3*1=3
    // out = [1, 3, 5, 3]
    float in[] = {1.0f, 2.0f, 3.0f};
    float w[] = {1.0f, 1.0f};
    auto input = Tensor::from_data(in, {1, 1, 3});
    auto weight = Tensor::from_data(w, {1, 1, 2});

    auto output = ops::conv_transpose1d(input, weight);
    EXPECT_EQ(output.shape()[2], 4u);

    auto ptr = output.typed_data<float>();
    EXPECT_NEAR(ptr[0], 1.0f, 1e-5);
    EXPECT_NEAR(ptr[1], 3.0f, 1e-5);
    EXPECT_NEAR(ptr[2], 5.0f, 1e-5);
    EXPECT_NEAR(ptr[3], 3.0f, 1e-5);
}

TEST(ConvTranspose, Conv1dWithStride) {
    // stride=2: out_length = (3-1)*2 - 0 + 1*(2-1) + 0 + 1 = 6
    float in[] = {1.0f, 2.0f, 3.0f};
    float w[] = {1.0f, 1.0f};
    auto input = Tensor::from_data(in, {1, 1, 3});
    auto weight = Tensor::from_data(w, {1, 1, 2});

    auto output = ops::conv_transpose1d(input, weight, Tensor(), 2);
    EXPECT_EQ(output.shape()[2], 6u);
}

TEST(ConvTranspose, Conv1dWithBias) {
    float in[] = {1.0f, 2.0f};
    float w[] = {1.0f};
    float b[] = {10.0f};
    auto input = Tensor::from_data(in, {1, 1, 2});
    auto weight = Tensor::from_data(w, {1, 1, 1});
    auto bias = Tensor::from_data(b, {1});

    auto output = ops::conv_transpose1d(input, weight, bias);
    auto ptr = output.typed_data<float>();
    EXPECT_NEAR(ptr[0], 11.0f, 1e-5);
    EXPECT_NEAR(ptr[1], 12.0f, 1e-5);
}

// ============================================================================
// conv_transpose2d — functional API
// ============================================================================

TEST(ConvTranspose, Conv2dIdentityKernel) {
    float w[] = {1.0f};
    auto weight = Tensor::from_data(w, {1, 1, 1, 1});
    auto input = Tensor::randn({1, 1, 4, 4});

    auto output = ops::conv_transpose2d(input, weight);
    ExpectTensorsClose(input, output, 1e-5, 1e-5);
}

TEST(ConvTranspose, Conv2dKnownSmall) {
    // input: (1,1,2,2) = [[1,2],[3,4]]
    // weight: (1,1,2,2) = [[1,0],[0,1]] — identity-like
    // out_h = (2-1)*1 - 0 + 1*(2-1) + 0 + 1 = 3
    // out_w = same = 3
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 0.0f, 0.0f, 1.0f};
    auto input = Tensor::from_data(in, {1, 1, 2, 2});
    auto weight = Tensor::from_data(w, {1, 1, 2, 2});

    auto output = ops::conv_transpose2d(input, weight);
    EXPECT_EQ(output.shape()[2], 3u);
    EXPECT_EQ(output.shape()[3], 3u);
}

TEST(ConvTranspose, Conv2dWithOutputPadding) {
    float w[] = {1.0f};
    auto weight = Tensor::from_data(w, {1, 1, 1, 1});
    auto input = Tensor::randn({1, 1, 3, 3});

    // stride=2, output_padding=1
    auto output =
        ops::conv_transpose2d(input, weight, Tensor(), {2, 2}, {0, 0}, {1, 1});
    // out_h = (3-1)*2 - 0 + 0 + 1 + 1 = 6
    EXPECT_EQ(output.shape()[2], 6u);
    EXPECT_EQ(output.shape()[3], 6u);
}

// ============================================================================
// ConvTranspose1d module
// ============================================================================

TEST(ConvTranspose, Module1dForward) {
    ConvTranspose1d ct;
    float w[] = {1.0f};
    float b[] = {0.0f};
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::from_data(w, {1, 1, 1});
    state["bias"] = Tensor::from_data(b, {1});
    ct.load_state_dict(state);

    auto input = Tensor::randn({1, 1, 4});
    auto output = ct(input);
    ExpectTensorsClose(input, output, 1e-5, 1e-5);
}

// ============================================================================
// ConvTranspose2d module
// ============================================================================

TEST(ConvTranspose, Module2dForward) {
    ConvTranspose2d ct;
    float w[] = {1.0f};
    float b[] = {0.0f};
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::from_data(w, {1, 1, 1, 1});
    state["bias"] = Tensor::from_data(b, {1});
    ct.load_state_dict(state);

    auto input = Tensor::randn({1, 1, 4, 4});
    auto output = ct(input);
    ExpectTensorsClose(input, output, 1e-5, 1e-5);
}

// ============================================================================
// GPU parity
// ============================================================================

TEST(ConvTranspose, Conv1dGPU) {
    SKIP_IF_NO_GPU();
    float in[] = {1.0f, 2.0f, 3.0f};
    float w[] = {1.0f, 1.0f};
    auto input = Tensor::from_data(in, {1, 1, 3});
    auto weight = Tensor::from_data(w, {1, 1, 2});

    auto cpu_out = ops::conv_transpose1d(input, weight);
    auto gpu_out = ops::conv_transpose1d(input.gpu(), weight.gpu());
    EXPECT_EQ(gpu_out.device(), Device::GPU);
    ExpectTensorsClose(cpu_out, gpu_out.cpu(), 1e-4, 1e-4);
}

TEST(ConvTranspose, Conv2dGPU) {
    SKIP_IF_NO_GPU();
    auto input = Tensor::randn({1, 2, 4, 4});
    auto weight = Tensor::randn({2, 3, 3, 3});

    auto cpu_out = ops::conv_transpose2d(input, weight);
    auto gpu_out = ops::conv_transpose2d(input.gpu(), weight.gpu());
    EXPECT_EQ(gpu_out.device(), Device::GPU);
    ExpectTensorsClose(cpu_out, gpu_out.cpu(), 1e-3, 1e-3);
}

// ============================================================================
// Groups support
// ============================================================================

TEST(ConvTranspose, Conv1dGroups) {
    // 2 input channels, 2 output channels, groups=2
    // Each group has 1 in-channel and 1 out-channel
    auto input = Tensor::ones({1, 2, 3});
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f}; // (2, 1, 2)
    auto weight = Tensor::from_data(w, {2, 1, 2});

    auto output = ops::conv_transpose1d(input, weight, Tensor(), 1, 0, 0, 1, 2);
    EXPECT_EQ(output.shape()[1], 2u); // 2 output channels
    EXPECT_EQ(output.shape()[2], 4u); // (3-1)*1 + 2 = 4
}
