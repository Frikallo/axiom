#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;

// ============================================================================
// Linear
// ============================================================================

TEST(NNLinear, ForwardBasic) {
    Linear linear(true);

    // Manually set weights: (out=2, in=3)
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({2, 3});
    state["bias"] = Tensor::zeros({2});
    linear.load_state_dict(state);

    auto input = Tensor::ones({1, 3});
    auto output = linear(input);

    ASSERT_TRUE(output.shape() == Shape({1, 2}));
    // ones @ ones^T = 3.0 for each output, + bias 0 = 3.0
    EXPECT_FLOAT_EQ(output.item<float>({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(output.item<float>({0, 1}), 3.0f);
}

TEST(NNLinear, ForwardWithBias) {
    Linear linear(true);

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({2, 3});
    state["bias"] =
        Tensor::from_data(std::vector<float>{1.0f, -1.0f}.data(), {2}, true);
    linear.load_state_dict(state);

    auto input = Tensor::ones({1, 3});
    auto output = linear(input);

    EXPECT_FLOAT_EQ(output.item<float>({0, 0}), 4.0f); // 3 + 1
    EXPECT_FLOAT_EQ(output.item<float>({0, 1}), 2.0f); // 3 - 1
}

TEST(NNLinear, ForwardNoBias) {
    Linear linear(false);

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({2, 3});
    linear.load_state_dict(state);

    auto input = Tensor::ones({4, 3});
    auto output = linear(input);

    ASSERT_TRUE(output.shape() == Shape({4, 2}));
    EXPECT_FLOAT_EQ(output.item<float>({0, 0}), 3.0f);
}

TEST(NNLinear, BatchedForward) {
    Linear linear(true);

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4, 8});
    state["bias"] = Tensor::zeros({4});
    linear.load_state_dict(state);

    auto input = Tensor::ones({2, 5, 8}); // batch=2, seq=5, features=8
    auto output = linear(input);

    ASSERT_TRUE(output.shape() == Shape({2, 5, 4}));
    EXPECT_FLOAT_EQ(output.item<float>({0, 0, 0}), 8.0f);
}

TEST(NNLinear, ParameterCount) {
    Linear with_bias(true);
    EXPECT_EQ(with_bias.parameters().size(), 2u);

    Linear no_bias(false);
    EXPECT_EQ(no_bias.parameters().size(), 1u);
}

TEST(NNLinear, ForwardBeforeLoadThrows) {
    Linear linear(true);
    auto input = Tensor::ones({1, 3});
    EXPECT_THROW(linear(input), RuntimeError);
}

TEST(NNEmbedding, ForwardBeforeLoadThrows) {
    Embedding embed;
    auto indices = Tensor::from_data(std::vector<int64_t>{0}.data(), {1}, true);
    EXPECT_THROW(embed(indices), RuntimeError);
}

TEST(NNLayerNorm, ForwardBeforeLoadThrows) {
    LayerNorm ln;
    auto input = Tensor::ones({1, 4});
    EXPECT_THROW(ln(input), RuntimeError);
}

TEST(NNRMSNorm, ForwardBeforeLoadThrows) {
    RMSNorm rn;
    auto input = Tensor::ones({1, 4});
    EXPECT_THROW(rn(input), RuntimeError);
}

TEST(NNConv1d, ForwardBeforeLoadThrows) {
    Conv1d conv;
    auto input = Tensor::ones({1, 1, 5});
    EXPECT_THROW(conv(input), RuntimeError);
}

TEST(NNConv2d, ForwardBeforeLoadThrows) {
    Conv2d conv;
    auto input = Tensor::ones({1, 1, 5, 5});
    EXPECT_THROW(conv(input), RuntimeError);
}

// ============================================================================
// Embedding
// ============================================================================

TEST(NNEmbedding, ForwardBasic) {
    Embedding embed;

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::from_data(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}.data(), {4, 3},
        true);
    embed.load_state_dict(state);

    auto indices =
        Tensor::from_data(std::vector<int64_t>{0, 2}.data(), {2}, true);
    auto output = embed(indices);

    ASSERT_TRUE(output.shape() == Shape({2, 3}));
    EXPECT_FLOAT_EQ(output.item<float>({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(output.item<float>({1, 0}), 6.0f);
}

TEST(NNEmbedding, BatchedIndices) {
    Embedding embed;

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::from_data(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.data(), {5, 2}, true);
    embed.load_state_dict(state);

    auto indices = Tensor::from_data(
        std::vector<int64_t>{0, 1, 2, 3, 4, 0}.data(), {2, 3}, true);
    auto output = embed(indices);

    ASSERT_TRUE(output.shape() == Shape({2, 3, 2}));
}

// ============================================================================
// LayerNorm
// ============================================================================

TEST(NNLayerNorm, ForwardBasic) {
    LayerNorm ln(1e-5f);

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    state["bias"] = Tensor::zeros({4});
    ln.load_state_dict(state);

    auto input =
        Tensor::from_data(std::vector<float>{1, 2, 3, 4}.data(), {1, 4}, true);
    auto output = ln(input);

    ASSERT_TRUE(output.shape() == Shape({1, 4}));
    // LayerNorm normalizes to mean=0, std=1 (approx with weight=1, bias=0)
    float mean = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        mean += output.item<float>({0, i});
    }
    mean /= 4.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);
}

TEST(NNLayerNorm, ParameterCount) {
    LayerNorm ln;
    EXPECT_EQ(ln.parameters().size(), 2u);
}

// ============================================================================
// RMSNorm
// ============================================================================

TEST(NNRMSNorm, ForwardBasic) {
    RMSNorm rn(1e-5f);

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({4});
    rn.load_state_dict(state);

    auto input =
        Tensor::from_data(std::vector<float>{1, 2, 3, 4}.data(), {1, 4}, true);
    auto output = rn(input);

    ASSERT_TRUE(output.shape() == Shape({1, 4}));
    // RMS norm with weight=1 should normalize by RMS value
    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5)
    float rms = std::sqrt(7.5f);
    EXPECT_NEAR(output.item<float>({0, 0}), 1.0f / rms, 1e-4f);
}

TEST(NNRMSNorm, ParameterCount) {
    RMSNorm rn;
    EXPECT_EQ(rn.parameters().size(), 1u);
}

// ============================================================================
// Conv1d
// ============================================================================

TEST(NNConv1d, ForwardBasic) {
    Conv1d conv(1, 0, 1, 1, false); // stride=1, padding=0, no bias

    std::map<std::string, Tensor> state;
    // Weight: (out_channels=1, in_channels=1, kernel_size=3)
    state["weight"] = Tensor::ones({1, 1, 3});
    conv.load_state_dict(state);

    // Input: (batch=1, channels=1, length=5)
    auto input = Tensor::ones({1, 1, 5});
    auto output = conv(input);

    // Output length = (5 + 2*0 - 1*(3-1) - 1) / 1 + 1 = 3
    ASSERT_TRUE(output.shape() == Shape({1, 1, 3}));
    EXPECT_FLOAT_EQ(output.item<float>({0, 0, 0}), 3.0f);
}

TEST(NNConv1d, ForwardWithBias) {
    Conv1d conv(1, 0, 1, 1, true);

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({1, 1, 3});
    state["bias"] =
        Tensor::from_data(std::vector<float>{2.0f}.data(), {1}, true);
    conv.load_state_dict(state);

    auto input = Tensor::ones({1, 1, 5});
    auto output = conv(input);

    EXPECT_FLOAT_EQ(output.item<float>({0, 0, 0}), 5.0f); // 3 + 2
}

TEST(NNConv1d, ParameterCount) {
    Conv1d with_bias(1, 0, 1, 1, true);
    EXPECT_EQ(with_bias.parameters().size(), 2u);

    Conv1d no_bias(1, 0, 1, 1, false);
    EXPECT_EQ(no_bias.parameters().size(), 1u);
}

// ============================================================================
// Conv2d
// ============================================================================

TEST(NNConv2d, ForwardBasic) {
    Conv2d conv({1, 1}, {0, 0}, {1, 1}, 1, false);

    std::map<std::string, Tensor> state;
    // Weight: (out_channels=1, in_channels=1, kH=3, kW=3)
    state["weight"] = Tensor::ones({1, 1, 3, 3});
    conv.load_state_dict(state);

    // Input: (batch=1, channels=1, H=5, W=5)
    auto input = Tensor::ones({1, 1, 5, 5});
    auto output = conv(input);

    // Output: (1, 1, 3, 3)
    ASSERT_TRUE(output.shape() == Shape({1, 1, 3, 3}));
    EXPECT_FLOAT_EQ(output.item<float>({0, 0, 0, 0}), 9.0f);
}

// ============================================================================
// ModuleList
// ============================================================================

TEST(NNModuleList, EmplaceBack) {
    ModuleList list;
    auto &l1 = list.emplace_back<Linear>(true);
    auto &l2 = list.emplace_back<Linear>(false);
    (void)l1;
    (void)l2;

    EXPECT_EQ(list.size(), 2u);
}

TEST(NNModuleList, Indexing) {
    ModuleList list;
    list.emplace_back<Linear>(true);
    list.emplace_back<Linear>(false);

    // Should not throw
    auto &mod = list[0];
    (void)mod;

    // Out of bounds
    EXPECT_THROW(list[5], IndexError);
}

TEST(NNModuleList, LoadStateDict) {
    ModuleList list;
    list.emplace_back<Linear>(true);
    list.emplace_back<Linear>(false);

    std::map<std::string, Tensor> state;
    state["0.weight"] = Tensor::ones({2, 3});
    state["0.bias"] = Tensor::zeros({2});
    state["1.weight"] = Tensor::ones({4, 2});

    list.load_state_dict(state);

    // Verify through parameters
    auto params = list.named_parameters();
    EXPECT_EQ(params.size(), 3u); // 2 from first, 1 from second
}

TEST(NNModuleList, ParametersRecursive) {
    ModuleList list;
    list.emplace_back<Linear>(true); // 2 params
    list.emplace_back<Linear>(true); // 2 params

    EXPECT_EQ(list.parameters().size(), 4u);
}

// ============================================================================
// Composite module test
// ============================================================================

struct TwoLayerNet : Module {
    Linear fc1_;
    Linear fc2_;

    TwoLayerNet() : fc1_(true), fc2_(true) {
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    Tensor forward(const Tensor &x) const { return fc2_(ops::relu(fc1_(x))); }
};

TEST(NNModule, CompositeModuleLoadAndForward) {
    TwoLayerNet net;

    std::map<std::string, Tensor> state;
    state["fc1.weight"] = Tensor::ones({4, 3});
    state["fc1.bias"] = Tensor::zeros({4});
    state["fc2.weight"] = Tensor::ones({2, 4});
    state["fc2.bias"] = Tensor::zeros({2});
    net.load_state_dict(state);

    auto input = Tensor::ones({1, 3});
    auto output = net.forward(input);

    ASSERT_TRUE(output.shape() == Shape({1, 2}));
    // fc1: ones@ones^T = 3 per output, relu(3)=3
    // fc2: ones@[3,3,3,3]^T = 12 per output
    EXPECT_FLOAT_EQ(output.item<float>({0, 0}), 12.0f);
    EXPECT_FLOAT_EQ(output.item<float>({0, 1}), 12.0f);
}

TEST(NNModule, CompositeModuleParameterCount) {
    TwoLayerNet net;
    // fc1: weight(4,3) + bias(4) = 2 params
    // fc2: weight(2,4) + bias(2) = 2 params
    EXPECT_EQ(net.parameters().size(), 4u);
}
