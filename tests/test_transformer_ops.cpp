#include <axiom/axiom.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace axiom;

// ==================================
//      TEST HARNESS
// ==================================

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test_func) run_test([&]() { test_func(); }, #test_func)

void run_test(const std::function<void()> &test_func,
              const std::string &test_name) {
    tests_run++;
    std::cout << "--- Running: " << test_name << " ---" << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << " ---" << std::endl;
        tests_passed++;
    } catch (const std::exception &e) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
    }
    std::cout << std::endl;
}

#define ASSERT(condition, msg)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + \
                                     std::string(msg));                        \
        }                                                                      \
    } while (0)

// ==================================
//      TRANSFORMER COMPONENT TESTS
// ==================================

// Test scaled dot-product attention pattern
void test_scaled_dot_product_attention_cpu() {
    // Simulating attention: softmax(Q @ K^T / sqrt(d_k)) @ V
    // Batch=2, SeqLen=4, d_k=8
    size_t batch = 2;
    size_t seq_len = 4;
    size_t d_k = 8;

    auto Q = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::CPU);
    auto K = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::CPU);
    auto V = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::CPU);

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Q @ K^T: (batch, seq, d_k) @ (batch, d_k, seq) -> (batch, seq, seq)
    auto scores = ops::matmul(Q, K, false, true);
    ASSERT(scores.shape() == Shape({batch, seq_len, seq_len}),
           "Scores shape wrong");

    // Scale
    auto scaled_scores =
        ops::multiply(scores, Tensor::full({1}, scale, Device::CPU));

    // Softmax on last dimension
    auto attn_weights = ops::softmax(scaled_scores, -1);
    ASSERT(attn_weights.shape() == Shape({batch, seq_len, seq_len}),
           "Attn weights shape wrong");

    // Verify softmax sums to 1 along last dimension
    // Manually verify first row
    float manual_sum = 0.0f;
    for (size_t j = 0; j < seq_len; ++j) {
        manual_sum += attn_weights.item<float>({0, 0, j});
    }
    if (std::abs(manual_sum - 1.0f) >= 0.01f) {
        std::cerr << "Manual sum of row [0,0,:] = " << manual_sum << std::endl;
    }
    ASSERT(std::abs(manual_sum - 1.0f) < 0.01f, "Softmax row should sum to 1");

    // attn_weights @ V: (batch, seq, seq) @ (batch, seq, d_k) -> (batch, seq,
    // d_k)
    auto output = ops::matmul(attn_weights, V);
    ASSERT(output.shape() == Shape({batch, seq_len, d_k}),
           "Output shape wrong");
}

void test_scaled_dot_product_attention_gpu() {
    if (!system::is_metal_available()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    size_t batch = 2;
    size_t seq_len = 4;
    size_t d_k = 8;

    auto Q = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::GPU);
    auto K = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::GPU);
    auto V = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::GPU);

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Q @ K^T
    auto scores = ops::matmul(Q, K, false, true);
    ASSERT(scores.shape() == Shape({batch, seq_len, seq_len}),
           "Scores shape wrong");
    ASSERT(scores.device() == Device::GPU, "Should be on GPU");

    // Scale
    auto scale_tensor = Tensor::full({1}, scale, Device::GPU);
    auto scaled_scores = ops::multiply(scores, scale_tensor);

    // Softmax
    auto attn_weights = ops::softmax(scaled_scores, -1);
    ASSERT(attn_weights.shape() == Shape({batch, seq_len, seq_len}),
           "Attn weights shape wrong");

    // attn_weights @ V
    auto output = ops::matmul(attn_weights, V);
    ASSERT(output.shape() == Shape({batch, seq_len, d_k}),
           "Output shape wrong");
    ASSERT(output.device() == Device::GPU, "Output should be on GPU");
}

// Test causal masking for autoregressive attention
void test_causal_masked_attention_cpu() {
    size_t batch = 1;
    size_t seq_len = 4;
    size_t d_k = 8;

    auto Q = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::CPU);
    auto K = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::CPU);
    auto V = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::CPU);

    // Create causal mask (upper triangular is masked)
    // mask[i,j] = 1 if j > i (future positions)
    auto causal_mask =
        Tensor::zeros({seq_len, seq_len}, DType::Bool, Device::CPU);
    bool *mask_data = causal_mask.typed_data<bool>();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = (j > i); // mask future
        }
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Q @ K^T
    auto scores = ops::matmul(Q, K, false, true);
    auto scaled_scores =
        ops::multiply(scores, Tensor::full({1}, scale, Device::CPU));

    // Apply causal mask: fill masked positions with -inf
    auto masked_scores =
        ops::masked_fill(scaled_scores.squeeze(0), causal_mask, -1e9f)
            .unsqueeze(0);

    // Softmax
    auto attn_weights = ops::softmax(masked_scores, -1);

    // Verify causal: attention weights to future positions should be ~0
    auto attn_cpu = attn_weights.cpu();
    const float *attn_data = attn_cpu.typed_data<float>();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = i + 1; j < seq_len; ++j) {
            float weight = attn_data[i * seq_len + j];
            ASSERT(weight < 1e-6f, "Future attention should be ~0");
        }
    }

    // attn_weights @ V
    auto output = ops::matmul(attn_weights, V);
    ASSERT(output.shape() == Shape({batch, seq_len, d_k}),
           "Output shape wrong");
}

void test_causal_masked_attention_gpu() {
    if (!system::is_metal_available()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    size_t batch = 1;
    size_t seq_len = 4;
    size_t d_k = 8;

    auto Q = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::GPU);
    auto K = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::GPU);
    auto V = Tensor::randn({batch, seq_len, d_k}, DType::Float32, Device::GPU);

    // Create causal mask
    auto causal_mask =
        Tensor::zeros({seq_len, seq_len}, DType::Bool, Device::CPU);
    bool *mask_data = causal_mask.typed_data<bool>();
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = (j > i);
        }
    }
    causal_mask = causal_mask.gpu();

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Q @ K^T
    auto scores = ops::matmul(Q, K, false, true);
    auto scale_tensor = Tensor::full({1}, scale, Device::GPU);
    auto scaled_scores = ops::multiply(scores, scale_tensor);

    // Apply causal mask
    auto masked_scores =
        ops::masked_fill(scaled_scores.squeeze(0), causal_mask, -1e9f)
            .unsqueeze(0);

    // Softmax
    auto attn_weights = ops::softmax(masked_scores, -1);

    // attn_weights @ V
    auto output = ops::matmul(attn_weights, V);
    ASSERT(output.shape() == Shape({batch, seq_len, d_k}),
           "Output shape wrong");
    ASSERT(output.device() == Device::GPU, "Output should be on GPU");
}

// Test layer normalization
void test_layer_norm_cpu() {
    size_t batch = 2;
    size_t seq_len = 4;
    size_t hidden = 8;

    auto x =
        Tensor::randn({batch, seq_len, hidden}, DType::Float32, Device::CPU);
    auto weight = Tensor::ones({hidden}, DType::Float32, Device::CPU);
    auto bias = Tensor::zeros({hidden}, DType::Float32, Device::CPU);

    auto output = ops::layer_norm(x, weight, bias, -1, 1e-5f);

    ASSERT(output.shape() == x.shape(), "Shape should be preserved");

    // Check for NaN/Inf in output
    ASSERT(!output.has_nan(), "LayerNorm output should not have NaN");
    ASSERT(!output.has_inf(), "LayerNorm output should not have Inf");

    // Verify normalized: compute mean/var manually for first sample
    float manual_mean = 0.0f;
    for (size_t j = 0; j < hidden; ++j) {
        manual_mean += output.item<float>({0, 0, j});
    }
    manual_mean /= static_cast<float>(hidden);

    float manual_var = 0.0f;
    for (size_t j = 0; j < hidden; ++j) {
        float v = output.item<float>({0, 0, j});
        manual_var += v * v;
    }
    manual_var /= static_cast<float>(hidden);

    ASSERT(std::abs(manual_mean) < 0.1f, "Manual mean should be ~0");
    ASSERT(std::abs(manual_var - 1.0f) < 0.2f, "Manual var should be ~1");
}

void test_layer_norm_gpu() {
    if (!system::is_metal_available()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    size_t batch = 2;
    size_t seq_len = 4;
    size_t hidden = 8;

    auto x =
        Tensor::randn({batch, seq_len, hidden}, DType::Float32, Device::GPU);
    auto weight = Tensor::ones({hidden}, DType::Float32, Device::GPU);
    auto bias = Tensor::zeros({hidden}, DType::Float32, Device::GPU);

    auto output = ops::layer_norm(x, weight, bias, -1, 1e-5f);

    ASSERT(output.shape() == x.shape(), "Shape should be preserved");
    ASSERT(output.device() == Device::GPU, "Should be on GPU");
}

// Test RMS normalization
void test_rms_norm_cpu() {
    size_t batch = 2;
    size_t hidden = 8;

    auto x = Tensor::randn({batch, hidden}, DType::Float32, Device::CPU);
    auto weight = Tensor::ones({hidden}, DType::Float32, Device::CPU);

    auto output = ops::rms_norm(x, weight, -1, 1e-5f);

    ASSERT(output.shape() == x.shape(), "Shape should be preserved");
}

void test_rms_norm_gpu() {
    if (!system::is_metal_available()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    size_t batch = 2;
    size_t hidden = 8;

    auto x = Tensor::randn({batch, hidden}, DType::Float32, Device::GPU);
    auto weight = Tensor::ones({hidden}, DType::Float32, Device::GPU);

    auto output = ops::rms_norm(x, weight, -1, 1e-5f);

    ASSERT(output.shape() == x.shape(), "Shape should be preserved");
    ASSERT(output.device() == Device::GPU, "Should be on GPU");
}

// Test GELU activation
void test_gelu_cpu() {
    auto x = Tensor::randn({2, 4, 8}, DType::Float32, Device::CPU);
    auto output = ops::gelu(x);

    ASSERT(output.shape() == x.shape(), "Shape should be preserved");

    // GELU(0) should be 0
    auto zero = Tensor::zeros({1}, DType::Float32, Device::CPU);
    auto gelu_zero = ops::gelu(zero);
    const float *data = gelu_zero.typed_data<float>();
    ASSERT(std::abs(data[0]) < 1e-6f, "GELU(0) should be 0");
}

void test_gelu_gpu() {
    if (!system::is_metal_available()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    auto x = Tensor::randn({2, 4, 8}, DType::Float32, Device::GPU);
    auto output = ops::gelu(x);

    ASSERT(output.shape() == x.shape(), "Shape should be preserved");
    ASSERT(output.device() == Device::GPU, "Should be on GPU");
}

// Test dropout
void test_dropout_cpu() {
    auto x = Tensor::ones({100, 100}, DType::Float32, Device::CPU);

    auto [output, mask] = ops::dropout(x, 0.5f, true);

    ASSERT(output.shape() == x.shape(), "Output shape should match");
    ASSERT(mask.shape() == x.shape(), "Mask shape should match");
    ASSERT(mask.dtype() == DType::Bool, "Mask should be Bool");

    // When training, approximately half should be zero
    auto sum_mask = ops::sum(mask.astype(DType::Float32), {}, false);
    float kept_ratio = sum_mask.typed_data<float>()[0] / (100.0f * 100.0f);
    ASSERT(kept_ratio > 0.3f && kept_ratio < 0.7f,
           "Roughly half should be kept");

    // Scale factor should be applied: non-zero values should be scaled by
    // 1/(1-p) = 2
    const float *out_data = output.typed_data<float>();
    const bool *mask_data = mask.typed_data<bool>();
    for (size_t i = 0; i < output.size(); ++i) {
        if (mask_data[i]) {
            ASSERT(std::abs(out_data[i] - 2.0f) < 1e-5f,
                   "Kept values should be scaled");
        } else {
            ASSERT(std::abs(out_data[i]) < 1e-5f, "Dropped values should be 0");
        }
    }
}

// Test full transformer block pattern (simplified)
void test_transformer_block_pattern_cpu() {
    size_t batch = 2;
    size_t seq_len = 4;
    size_t hidden = 16;
    size_t num_heads = 2;
    size_t head_dim = hidden / num_heads;

    // Input
    auto x =
        Tensor::randn({batch, seq_len, hidden}, DType::Float32, Device::CPU);

    // Pre-norm (RMSNorm)
    auto norm_weight = Tensor::ones({hidden}, DType::Float32, Device::CPU);
    auto x_norm = ops::rms_norm(x, norm_weight, -1, 1e-5f);

    // QKV projection (simplified as single matmul per head)
    auto Wq = Tensor::randn({hidden, hidden}, DType::Float32, Device::CPU);
    auto Wk = Tensor::randn({hidden, hidden}, DType::Float32, Device::CPU);
    auto Wv = Tensor::randn({hidden, hidden}, DType::Float32, Device::CPU);

    // Project Q, K, V
    // x_norm: (batch, seq, hidden) @ W: (hidden, hidden) -> (batch, seq,
    // hidden)
    auto Q = ops::matmul(x_norm, Wq);
    auto K = ops::matmul(x_norm, Wk);
    auto V = ops::matmul(x_norm, Wv);

    // Reshape for multi-head attention: (batch, seq, num_heads, head_dim) ->
    // (batch, num_heads, seq, head_dim) Note: simplified version, actual
    // implementation would use reshape + transpose

    // Scaled dot-product attention per head
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto scores = ops::matmul(Q, K, false, true);
    auto scaled_scores =
        ops::multiply(scores, Tensor::full({1}, scale, Device::CPU));
    auto attn_weights = ops::softmax(scaled_scores, -1);
    auto attn_output = ops::matmul(attn_weights, V);

    // Output projection
    auto Wo = Tensor::randn({hidden, hidden}, DType::Float32, Device::CPU);
    auto output = ops::matmul(attn_output, Wo);

    // Residual connection
    auto residual = ops::add(x, output);

    ASSERT(residual.shape() == x.shape(), "Output shape should match input");

    // FFN block
    auto ffn_norm = ops::rms_norm(residual, norm_weight, -1, 1e-5f);

    auto W1 = Tensor::randn({hidden, hidden * 4}, DType::Float32, Device::CPU);
    auto W2 = Tensor::randn({hidden * 4, hidden}, DType::Float32, Device::CPU);

    auto ffn_hidden = ops::matmul(ffn_norm, W1);
    auto ffn_activated = ops::gelu(ffn_hidden);
    auto ffn_output = ops::matmul(ffn_activated, W2);

    auto final_output = ops::add(residual, ffn_output);

    ASSERT(final_output.shape() == x.shape(),
           "Final output shape should match input");
    ASSERT(!final_output.has_nan(), "Should not have NaN");
    ASSERT(!final_output.has_inf(), "Should not have Inf");
}

void test_transformer_block_pattern_gpu() {
    if (!system::is_metal_available()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    size_t batch = 2;
    size_t seq_len = 4;
    size_t hidden = 16;

    auto x =
        Tensor::randn({batch, seq_len, hidden}, DType::Float32, Device::GPU);

    auto norm_weight = Tensor::ones({hidden}, DType::Float32, Device::GPU);
    auto x_norm = ops::rms_norm(x, norm_weight, -1, 1e-5f);

    auto Wq = Tensor::randn({hidden, hidden}, DType::Float32, Device::GPU);
    auto Q = ops::matmul(x_norm, Wq);

    auto Wk = Tensor::randn({hidden, hidden}, DType::Float32, Device::GPU);
    auto K = ops::matmul(x_norm, Wk);

    auto Wv = Tensor::randn({hidden, hidden}, DType::Float32, Device::GPU);
    auto V = ops::matmul(x_norm, Wv);

    float scale = 1.0f / std::sqrt(static_cast<float>(hidden));
    auto scores = ops::matmul(Q, K, false, true);
    auto scale_tensor = Tensor::full({1}, scale, Device::GPU);
    auto scaled_scores = ops::multiply(scores, scale_tensor);
    auto attn_weights = ops::softmax(scaled_scores, -1);
    auto attn_output = ops::matmul(attn_weights, V);

    auto Wo = Tensor::randn({hidden, hidden}, DType::Float32, Device::GPU);
    auto output = ops::matmul(attn_output, Wo);
    auto residual = ops::add(x, output);

    ASSERT(residual.shape() == x.shape(), "Output shape should match input");
    ASSERT(residual.device() == Device::GPU, "Should be on GPU");
}

// ==================================
//      MAIN
// ==================================

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Transformer Operations Tests ===" << std::endl
              << std::endl;

    // Attention tests
    std::cout << "--- Attention Tests ---" << std::endl;
    RUN_TEST(test_scaled_dot_product_attention_cpu);
    RUN_TEST(test_scaled_dot_product_attention_gpu);
    RUN_TEST(test_causal_masked_attention_cpu);
    RUN_TEST(test_causal_masked_attention_gpu);

    // Normalization tests
    std::cout << "--- Normalization Tests ---" << std::endl;
    RUN_TEST(test_layer_norm_cpu);
    RUN_TEST(test_layer_norm_gpu);
    RUN_TEST(test_rms_norm_cpu);
    RUN_TEST(test_rms_norm_gpu);

    // Activation tests
    std::cout << "--- Activation Tests ---" << std::endl;
    RUN_TEST(test_gelu_cpu);
    RUN_TEST(test_gelu_gpu);
    RUN_TEST(test_dropout_cpu);

    // Full pattern tests
    std::cout << "--- Transformer Block Pattern Tests ---" << std::endl;
    RUN_TEST(test_transformer_block_pattern_cpu);
    RUN_TEST(test_transformer_block_pattern_gpu);

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
