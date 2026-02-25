#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;
using namespace axiom::testing;

// ============================================================================
// GroupNorm — basic
// ============================================================================

TEST(NNGroupNorm, BasicIdentity) {
    // Groups=2, C=4. With weight=1, bias=0, the output should be normalized
    // per-group.
    GroupNorm gn(2);

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    gn.load_state_dict(state);

    // Create input where each group has known mean/std
    auto input = Tensor::randn({2, 4, 8});
    auto output = gn(input);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(output.dtype(), DType::Float32);
}

TEST(NNGroupNorm, SingleGroup) {
    // num_groups=1 should normalize over entire channel dim (like LayerNorm
    // over C,spatial)
    GroupNorm gn(1);
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    gn.load_state_dict(state);

    auto input = Tensor::randn({2, 4, 8});
    auto output = gn(input);
    EXPECT_EQ(output.shape(), input.shape());
}

TEST(NNGroupNorm, ChannelEqualsGroups) {
    // num_groups=C means each channel is its own group (like InstanceNorm)
    GroupNorm gn(4);
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    gn.load_state_dict(state);

    auto input = Tensor::randn({2, 4, 8});
    auto output = gn(input);
    EXPECT_EQ(output.shape(), input.shape());
}

TEST(NNGroupNorm, BadGroupsDivisibility) {
    GroupNorm gn(3);
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    gn.load_state_dict(state);

    auto input = Tensor::randn({2, 4, 8});
    EXPECT_THROW(gn(input), ShapeError);
}

TEST(NNGroupNorm, Float64) {
    GroupNorm gn(2);
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4}, DType::Float64);
    state["bias"] = Tensor::zeros({4}, DType::Float64);
    gn.load_state_dict(state);

    auto input = Tensor::randn({2, 4, 8}, DType::Float64);
    auto output = gn(input);
    EXPECT_EQ(output.dtype(), DType::Float64);
}

TEST(NNGroupNorm, GPU) {
    SKIP_IF_NO_GPU();
    GroupNorm gn(2);
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    gn.load_state_dict(state);

    auto cpu_input = Tensor::randn({2, 4, 8});
    auto cpu_output = gn(cpu_input);
    auto gpu_output = gn(cpu_input.gpu());
    EXPECT_EQ(gpu_output.device(), Device::GPU);
    ExpectTensorsClose(cpu_output, gpu_output.cpu(), 1e-4, 1e-4);
}

// ============================================================================
// InstanceNorm1d
// ============================================================================

TEST(NNInstanceNorm, InstanceNorm1dBasic) {
    InstanceNorm1d in_norm;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    in_norm.load_state_dict(state);

    auto input = Tensor::randn({2, 4, 8});
    auto output = in_norm(input);
    EXPECT_EQ(output.shape(), input.shape());
}

TEST(NNInstanceNorm, InstanceNorm1dNoAffine) {
    InstanceNorm1d in_norm(1e-5f, false);
    auto input = Tensor::randn({2, 4, 8});
    auto output = in_norm(input);
    EXPECT_EQ(output.shape(), input.shape());

    // With no affine, should have no parameters
    EXPECT_EQ(in_norm.parameters().size(), 0u);
}

TEST(NNInstanceNorm, InstanceNorm1dShapeError) {
    InstanceNorm1d in_norm;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    in_norm.load_state_dict(state);

    auto bad_input = Tensor::randn({2, 4}); // 2D instead of 3D
    EXPECT_THROW(in_norm(bad_input), ShapeError);
}

// ============================================================================
// InstanceNorm2d
// ============================================================================

TEST(NNInstanceNorm, InstanceNorm2dBasic) {
    InstanceNorm2d in_norm;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({3});
    state["bias"] = Tensor::zeros({3});
    in_norm.load_state_dict(state);

    auto input = Tensor::randn({2, 3, 8, 8});
    auto output = in_norm(input);
    EXPECT_EQ(output.shape(), input.shape());
}

TEST(NNInstanceNorm, InstanceNorm2dShapeError) {
    InstanceNorm2d in_norm;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({3});
    state["bias"] = Tensor::zeros({3});
    in_norm.load_state_dict(state);

    auto bad_input = Tensor::randn({2, 3, 8}); // 3D instead of 4D
    EXPECT_THROW(in_norm(bad_input), ShapeError);
}

TEST(NNInstanceNorm, InstanceNorm2dGPU) {
    SKIP_IF_NO_GPU();
    InstanceNorm2d in_norm;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({3});
    state["bias"] = Tensor::zeros({3});
    in_norm.load_state_dict(state);

    auto cpu_input = Tensor::randn({2, 3, 4, 4});
    auto cpu_output = in_norm(cpu_input);
    auto gpu_output = in_norm(cpu_input.gpu());
    EXPECT_EQ(gpu_output.device(), Device::GPU);
    ExpectTensorsClose(cpu_output, gpu_output.cpu(), 1e-4, 1e-4);
}

// ============================================================================
// Numerical correctness test for GroupNorm
// ============================================================================

TEST(NNGroupNorm, NumericalCorrectness) {
    // Single sample, 4 channels, 2 groups of 2.
    // Manually compute expected values.
    GroupNorm gn(2);

    float w[] = {2.0f, 2.0f, 3.0f, 3.0f};
    float b[] = {1.0f, 1.0f, -1.0f, -1.0f};
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::from_data(w, {4});
    state["bias"] = Tensor::from_data(b, {4});
    gn.load_state_dict(state);

    // Input: (1, 4, 1) — each "spatial" is just 1 element
    // Group 0: channels 0,1 → values [1, 3] → mean=2, var=1
    // Group 1: channels 2,3 → values [5, 7] → mean=6, var=1
    float in[] = {1.0f, 3.0f, 5.0f, 7.0f};
    auto input = Tensor::from_data(in, {1, 4, 1});
    auto output = gn(input);

    auto ptr = output.typed_data<float>();
    // channel 0: (1-2)/sqrt(1+1e-5) * 2 + 1 = -2 + 1 = -1
    // channel 1: (3-2)/sqrt(1+1e-5) * 2 + 1 = 2 + 1 = 3
    // channel 2: (5-6)/sqrt(1+1e-5) * 3 + (-1) = -3 + (-1) = -4
    // channel 3: (7-6)/sqrt(1+1e-5) * 3 + (-1) = 3 + (-1) = 2
    EXPECT_NEAR(ptr[0], -1.0f, 1e-3);
    EXPECT_NEAR(ptr[1], 3.0f, 1e-3);
    EXPECT_NEAR(ptr[2], -4.0f, 1e-3);
    EXPECT_NEAR(ptr[3], 2.0f, 1e-3);
}
