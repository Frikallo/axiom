#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>
#include <cmath>
#include <vector>

using namespace axiom;
using namespace axiom::ops;

// ============================================================================
// Naive reference: decomposed attention for correctness comparison
// ============================================================================

static Tensor naive_attention(const Tensor &query, const Tensor &key,
                              const Tensor &value, const Tensor &mask,
                              float scale, bool is_causal) {
    // S = Q @ K^T * scale
    auto scores = matmul(query, key, false, true);
    auto scale_t =
        Tensor::full({1}, scale, scores.device()).astype(scores.dtype());
    scores = multiply(scores, scale_t);

    auto seq_q = static_cast<int>(query.shape()[2]);
    auto seq_kv = static_cast<int>(key.shape()[2]);

    if (is_causal) {
        auto causal_mask = Tensor::zeros(
            {static_cast<size_t>(seq_q), static_cast<size_t>(seq_kv)},
            DType::Bool, Device::CPU);
        auto *m = causal_mask.typed_data<uint8_t>();
        for (int i = 0; i < seq_q; ++i) {
            for (int j = i + 1; j < seq_kv; ++j) {
                m[i * seq_kv + j] = 1;
            }
        }
        scores = masked_fill(scores.ascontiguousarray(), causal_mask, -1e9f);
    }

    if (mask.storage()) {
        scores = masked_fill(scores.ascontiguousarray(), mask, -1e9f);
    }

    auto attn_weights = softmax(scores, -1);
    return matmul(attn_weights, value);
}

// ============================================================================
// Helper: create random 4D attention tensors
// ============================================================================

struct AttentionInputs {
    Tensor q, k, v;
};

static AttentionInputs make_inputs(int batch, int heads, int seq_q, int seq_kv,
                                   int head_dim, DType dtype = DType::Float32) {
    auto q = Tensor::randn(
        {static_cast<size_t>(batch), static_cast<size_t>(heads),
         static_cast<size_t>(seq_q), static_cast<size_t>(head_dim)},
        dtype);
    auto k = Tensor::randn(
        {static_cast<size_t>(batch), static_cast<size_t>(heads),
         static_cast<size_t>(seq_kv), static_cast<size_t>(head_dim)},
        dtype);
    auto v = Tensor::randn(
        {static_cast<size_t>(batch), static_cast<size_t>(heads),
         static_cast<size_t>(seq_kv), static_cast<size_t>(head_dim)},
        dtype);
    return {q, k, v};
}

// ============================================================================
// Correctness tests: fused vs naive reference
// ============================================================================

TEST(FlashAttention, BasicSelfAttention) {
    auto [q, k, v] = make_inputs(1, 2, 8, 8, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    ASSERT_TRUE(fused.shape() == naive.shape());
    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, CrossAttention) {
    auto [q, k, v] = make_inputs(1, 2, 16, 32, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    ASSERT_TRUE(fused.shape() == Shape({1, 2, 16, 32}));
    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, CausalMask) {
    auto [q, k, v] = make_inputs(1, 2, 16, 16, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, true);
    auto naive = naive_attention(q, k, v, Tensor(), scale, true);

    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, ExplicitBoolMask) {
    auto [q, k, v] = make_inputs(1, 1, 8, 8, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    // Create a random mask — mask out ~30% of positions
    auto mask = Tensor::zeros({8, 8}, DType::Bool, Device::CPU);
    auto *m = mask.typed_data<uint8_t>();
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if ((i + j) % 3 == 0)
                m[i * 8 + j] = 1;
        }
    }

    auto fused = scaled_dot_product_attention(q, k, v, mask, scale, false);
    auto naive = naive_attention(q, k, v, mask, scale, false);

    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, NonDivisibleSequenceLength) {
    // seq=37 doesn't evenly divide into blocks of 64
    auto [q, k, v] = make_inputs(1, 1, 37, 37, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    ASSERT_TRUE(fused.shape() == Shape({1, 1, 37, 32}));
    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, LargeSequence) {
    auto [q, k, v] = make_inputs(1, 1, 256, 256, 64);
    float scale = 1.0f / std::sqrt(64.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    ASSERT_TRUE(fused.shape() == Shape({1, 1, 256, 64}));
    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, HeadDim32) {
    auto [q, k, v] = make_inputs(1, 2, 8, 8, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, HeadDim64) {
    auto [q, k, v] = make_inputs(1, 2, 8, 8, 64);
    float scale = 1.0f / std::sqrt(64.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, HeadDim128) {
    auto [q, k, v] = make_inputs(1, 2, 8, 8, 128);
    float scale = 1.0f / std::sqrt(128.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, ScaleParameter) {
    auto [q, k, v] = make_inputs(1, 1, 8, 8, 32);
    float custom_scale = 0.5f;

    auto fused =
        scaled_dot_product_attention(q, k, v, Tensor(), custom_scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), custom_scale, false);

    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, AutoScale) {
    auto [q, k, v] = make_inputs(1, 1, 8, 8, 64);

    // scale <= 0 triggers auto-computation: 1/sqrt(head_dim)
    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), -1.0f, false);

    float auto_scale = 1.0f / std::sqrt(64.0f);
    auto naive = naive_attention(q, k, v, Tensor(), auto_scale, false);

    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, BatchedMultiHead) {
    auto [q, k, v] = make_inputs(4, 8, 16, 16, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    ASSERT_TRUE(fused.shape() == Shape({4, 8, 16, 32}));
    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

// ============================================================================
// GPU tests
// ============================================================================

TEST(FlashAttentionGPU, CorrectnessFloat32) {
    SKIP_IF_NO_GPU();
    auto [q, k, v] = make_inputs(1, 2, 16, 16, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto cpu_result =
        scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto gpu_result = scaled_dot_product_attention(q.gpu(), k.gpu(), v.gpu(),
                                                   Tensor(), scale, false);

    axiom::testing::AssertTensorsClose(cpu_result, gpu_result.cpu(), 1e-3,
                                       1e-3);
}

TEST(FlashAttentionGPU, CorrectnessFloat16) {
    SKIP_IF_NO_GPU();
    auto [q, k, v] = make_inputs(1, 2, 16, 16, 64, DType::Float16);
    float scale = 1.0f / std::sqrt(64.0f);

    // Compare against Float32 reference
    auto q32 = q.astype(DType::Float32);
    auto k32 = k.astype(DType::Float32);
    auto v32 = v.astype(DType::Float32);
    auto ref = naive_attention(q32, k32, v32, Tensor(), scale, false);

    auto gpu_result = scaled_dot_product_attention(q.gpu(), k.gpu(), v.gpu(),
                                                   Tensor(), scale, false);

    axiom::testing::AssertTensorsClose(
        ref, gpu_result.cpu().astype(DType::Float32), 1e-2, 1e-2);
}

TEST(FlashAttentionGPU, CausalMask) {
    SKIP_IF_NO_GPU();
    auto [q, k, v] = make_inputs(1, 2, 16, 16, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto cpu_result =
        scaled_dot_product_attention(q, k, v, Tensor(), scale, true);
    auto gpu_result = scaled_dot_product_attention(q.gpu(), k.gpu(), v.gpu(),
                                                   Tensor(), scale, true);

    axiom::testing::AssertTensorsClose(cpu_result, gpu_result.cpu(), 1e-3,
                                       1e-3);
}

TEST(FlashAttentionGPU, CPUParity) {
    SKIP_IF_NO_GPU();
    auto [q, k, v] = make_inputs(2, 4, 32, 32, 64);
    float scale = 1.0f / std::sqrt(64.0f);

    auto cpu_result =
        scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto gpu_result = scaled_dot_product_attention(q.gpu(), k.gpu(), v.gpu(),
                                                   Tensor(), scale, false);

    axiom::testing::AssertTensorsClose(cpu_result, gpu_result.cpu(), 1e-3,
                                       1e-3);
}

// ============================================================================
// nn::MultiHeadAttention integration tests
// ============================================================================

TEST(FlashAttentionMHA, FusedOpForward) {
    nn::MultiHeadAttention mha(1);

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

    auto input = Tensor::ones({1, 3, 2});
    auto output = mha.forward(input, input, input);

    ASSERT_TRUE(output.shape() == Shape({1, 3, 2}));
    // Uniform input + identity projections -> output ≈ 1.0
    EXPECT_NEAR(output.item<float>({0, 0, 0}), 1.0f, 1e-4f);
    EXPECT_NEAR(output.item<float>({0, 0, 1}), 1.0f, 1e-4f);
}

TEST(FlashAttentionMHA, CausalForward) {
    nn::MultiHeadAttention mha(1);

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

    auto input = Tensor::ones({1, 4, 2});
    auto output = mha.forward(input, input, input, Tensor(), true);

    ASSERT_TRUE(output.shape() == Shape({1, 4, 2}));
    // Uniform input -> causal mask doesn't change result
    EXPECT_NEAR(output.item<float>({0, 0, 0}), 1.0f, 1e-4f);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(FlashAttention, SeqLen1) {
    auto [q, k, v] = make_inputs(1, 1, 1, 1, 32);
    float scale = 1.0f / std::sqrt(32.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    ASSERT_TRUE(fused.shape() == Shape({1, 1, 1, 32}));
    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, UnsupportedHeadDimFallback) {
    // head_dim=48 should still work via decomposed fallback on GPU
    auto [q, k, v] = make_inputs(1, 1, 8, 8, 48);
    float scale = 1.0f / std::sqrt(48.0f);

    auto fused = scaled_dot_product_attention(q, k, v, Tensor(), scale, false);
    auto naive = naive_attention(q, k, v, Tensor(), scale, false);

    ASSERT_TRUE(fused.shape() == Shape({1, 1, 8, 48}));
    axiom::testing::AssertTensorsClose(fused, naive, 1e-4, 1e-4);
}

TEST(FlashAttention, InputValidation4D) {
    auto q = Tensor::randn({2, 8, 32}); // 3D — invalid
    auto k = Tensor::randn({2, 8, 32});
    auto v = Tensor::randn({2, 8, 32});

    EXPECT_THROW(scaled_dot_product_attention(q, k, v), ShapeError);
}

TEST(FlashAttention, BatchMismatchThrows) {
    auto q = Tensor::randn({1, 2, 8, 32});
    auto k = Tensor::randn({2, 2, 8, 32}); // batch mismatch
    auto v = Tensor::randn({2, 2, 8, 32});

    EXPECT_THROW(scaled_dot_product_attention(q, k, v), ShapeError);
}
