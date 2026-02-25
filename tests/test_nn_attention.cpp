#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;

// ============================================================================
// MultiHeadAttention
// ============================================================================

TEST(NNMultiHeadAttention, ParameterStructure) {
    MultiHeadAttention mha(2);

    auto named = mha.named_parameters();
    // 4 projections x 2 params (weight + bias) = 8
    EXPECT_EQ(named.size(), 8u);

    // Check names
    std::vector<std::string> expected_names = {
        "q_proj.weight", "q_proj.bias", "k_proj.weight",   "k_proj.bias",
        "v_proj.weight", "v_proj.bias", "out_proj.weight", "out_proj.bias"};
    for (size_t i = 0; i < expected_names.size(); ++i) {
        EXPECT_EQ(named[i].first, expected_names[i]);
    }
}

TEST(NNMultiHeadAttention, LoadStateDict) {
    MultiHeadAttention mha(2);

    size_t d_model = 4;
    std::map<std::string, Tensor> state;
    state["q_proj.weight"] = Tensor::ones({d_model, d_model});
    state["q_proj.bias"] = Tensor::zeros({d_model});
    state["k_proj.weight"] = Tensor::ones({d_model, d_model});
    state["k_proj.bias"] = Tensor::zeros({d_model});
    state["v_proj.weight"] = Tensor::ones({d_model, d_model});
    state["v_proj.bias"] = Tensor::zeros({d_model});
    state["out_proj.weight"] = Tensor::ones({d_model, d_model});
    state["out_proj.bias"] = Tensor::zeros({d_model});

    EXPECT_NO_THROW(mha.load_state_dict(state));
}

TEST(NNMultiHeadAttention, ForwardShape) {
    MultiHeadAttention mha(2);

    size_t d_model = 4;
    std::map<std::string, Tensor> state;
    // Use identity-like weights for predictable behavior
    state["q_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["q_proj.bias"] = Tensor::zeros({d_model});
    state["k_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["k_proj.bias"] = Tensor::zeros({d_model});
    state["v_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["v_proj.bias"] = Tensor::zeros({d_model});
    state["out_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["out_proj.bias"] = Tensor::zeros({d_model});
    mha.load_state_dict(state);

    // Input: (batch=1, seq=3, d_model=4)
    auto input = Tensor::ones({1, 3, 4});
    auto output = mha.forward(input, input, input);

    ASSERT_TRUE(output.shape() == Shape({1, 3, 4}));
}

TEST(NNMultiHeadAttention, SelfAttentionValues) {
    MultiHeadAttention mha(1); // Single head for simpler verification

    size_t d_model = 2;
    std::map<std::string, Tensor> state;
    // Identity projections
    state["q_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["q_proj.bias"] = Tensor::zeros({d_model});
    state["k_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["k_proj.bias"] = Tensor::zeros({d_model});
    state["v_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["v_proj.bias"] = Tensor::zeros({d_model});
    state["out_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["out_proj.bias"] = Tensor::zeros({d_model});
    mha.load_state_dict(state);

    // Uniform input -> softmax produces uniform attention -> output = mean of V
    auto input = Tensor::ones({1, 3, 2});
    auto output = mha.forward(input, input, input);

    ASSERT_TRUE(output.shape() == Shape({1, 3, 2}));
    // With uniform input and identity projections, output should be ~1.0
    EXPECT_NEAR(output.item<float>({0, 0, 0}), 1.0f, 1e-4f);
    EXPECT_NEAR(output.item<float>({0, 0, 1}), 1.0f, 1e-4f);
}

TEST(NNMultiHeadAttention, WithPrefixLoading) {
    MultiHeadAttention mha(2);

    size_t d_model = 4;
    std::map<std::string, Tensor> state;
    state["encoder.attn.q_proj.weight"] = Tensor::ones({d_model, d_model});
    state["encoder.attn.q_proj.bias"] = Tensor::zeros({d_model});
    state["encoder.attn.k_proj.weight"] = Tensor::ones({d_model, d_model});
    state["encoder.attn.k_proj.bias"] = Tensor::zeros({d_model});
    state["encoder.attn.v_proj.weight"] = Tensor::ones({d_model, d_model});
    state["encoder.attn.v_proj.bias"] = Tensor::zeros({d_model});
    state["encoder.attn.out_proj.weight"] = Tensor::ones({d_model, d_model});
    state["encoder.attn.out_proj.bias"] = Tensor::zeros({d_model});

    EXPECT_NO_THROW(mha.load_state_dict(state, "encoder.attn."));
}

TEST(NNMultiHeadAttention, InvalidInputDimThrows) {
    MultiHeadAttention mha(2);

    size_t d_model = 4;
    std::map<std::string, Tensor> state;
    state["q_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["q_proj.bias"] = Tensor::zeros({d_model});
    state["k_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["k_proj.bias"] = Tensor::zeros({d_model});
    state["v_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["v_proj.bias"] = Tensor::zeros({d_model});
    state["out_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["out_proj.bias"] = Tensor::zeros({d_model});
    mha.load_state_dict(state);

    // 2D input should throw
    auto input_2d = Tensor::ones({3, 4});
    EXPECT_THROW(mha.forward(input_2d, input_2d, input_2d), ShapeError);
}

TEST(NNMultiHeadAttention, WithMask) {
    MultiHeadAttention mha(1);

    size_t seq_len = 3;
    size_t d_model = 2;
    std::map<std::string, Tensor> state;
    state["q_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["q_proj.bias"] = Tensor::zeros({d_model});
    state["k_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["k_proj.bias"] = Tensor::zeros({d_model});
    state["v_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["v_proj.bias"] = Tensor::zeros({d_model});
    state["out_proj.weight"] = Tensor::eye(d_model).astype(DType::Float32);
    state["out_proj.bias"] = Tensor::zeros({d_model});
    mha.load_state_dict(state);

    // Causal mask: mask[i][j] = true where j > i (future positions)
    // After attention reshape, scores are (1, 1, 3, 3) and mask must match
    auto mask =
        Tensor::zeros({1, 1, seq_len, seq_len}, DType::Bool, Device::CPU);
    bool *mask_ptr = mask.typed_data<bool>();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            mask_ptr[i * seq_len + j] = true;
        }
    }

    auto input = Tensor::ones({1, seq_len, d_model});
    auto output = mha.forward(input, input, input, mask);

    ASSERT_TRUE(output.shape() == Shape({1, seq_len, d_model}));
    // With uniform input and identity projections, output should be ~1.0
    // even with causal mask (all values are the same)
    EXPECT_NEAR(output.item<float>({0, 0, 0}), 1.0f, 1e-4f);
}

// ============================================================================
// Integration: ModuleList with attention layers
// ============================================================================

struct SimpleTransformerLayer : Module {
    MultiHeadAttention attn_;
    LayerNorm ln_;

    SimpleTransformerLayer(int num_heads) : attn_(num_heads) {
        register_module("attn", attn_);
        register_module("ln", ln_);
    }
};

TEST(NNAttention, TransformerLayerStructure) {
    SimpleTransformerLayer layer(2);
    auto named = layer.named_parameters();
    // attn: 8 params (4 projections * 2)
    // ln: 2 params (weight + bias)
    EXPECT_EQ(named.size(), 10u);
}

TEST(NNAttention, ModuleListOfAttentionLayers) {
    ModuleList layers;
    layers.emplace_back<SimpleTransformerLayer>(2);
    layers.emplace_back<SimpleTransformerLayer>(2);

    auto params = layers.parameters();
    EXPECT_EQ(params.size(), 20u); // 10 per layer * 2 layers

    auto named = layers.named_parameters();
    // First layer params should start with "0."
    EXPECT_EQ(named[0].first.substr(0, 2), "0.");
    // Second layer params should start with "1."
    EXPECT_EQ(named[10].first.substr(0, 2), "1.");
}
