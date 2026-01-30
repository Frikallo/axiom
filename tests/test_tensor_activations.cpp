#include <axiom/axiom.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace axiom;

// Test harness
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

template <typename T>
void assert_tensor_near(const Tensor &t, const std::vector<T> &expected,
                        T tol = T(1e-4)) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.size() == expected.size(), "Size mismatch");
    const T *data = t_cpu.typed_data<T>();
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(data[i] - expected[i]) > tol) {
            throw std::runtime_error("Value mismatch at index " +
                                     std::to_string(i) + ": got " +
                                     std::to_string(data[i]) + ", expected " +
                                     std::to_string(expected[i]));
        }
    }
}

// ============================================================================
// Softmax Tests
// ============================================================================

void test_softmax_1d_cpu() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::softmax(t, -1);

    ASSERT(result.shape() == Shape{3}, "Shape should be preserved");
    ASSERT(result.dtype() == DType::Float32, "Dtype should be Float32");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // Softmax values should sum to 1
    float sum = out[0] + out[1] + out[2];
    ASSERT(std::abs(sum - 1.0f) < 1e-5f, "Softmax should sum to 1");

    // Values should be in ascending order (since inputs are ascending)
    ASSERT(out[0] < out[1] && out[1] < out[2],
           "Softmax values should preserve order");
}

void test_softmax_2d_cpu() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {2, 3});

    auto result = ops::softmax(t, -1); // Softmax along last axis

    ASSERT(result.shape() == Shape({2, 3}), "Shape should be preserved");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // Each row should sum to 1
    float sum0 = out[0] + out[1] + out[2];
    float sum1 = out[3] + out[4] + out[5];
    ASSERT(std::abs(sum0 - 1.0f) < 1e-5f, "First row should sum to 1");
    ASSERT(std::abs(sum1 - 1.0f) < 1e-5f, "Second row should sum to 1");

    // Second row has equal inputs, so outputs should be equal
    ASSERT(std::abs(out[3] - out[4]) < 1e-5f,
           "Equal inputs should give equal outputs");
}

void test_log_softmax_cpu() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::log_softmax(t, -1);

    ASSERT(result.shape() == Shape{3}, "Shape should be preserved");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // All log_softmax values should be negative (log of probability < 1)
    ASSERT(out[0] < 0 && out[1] < 0 && out[2] < 0,
           "Log softmax values should be negative");

    // exp(log_softmax) should sum to 1
    float sum_exp = std::exp(out[0]) + std::exp(out[1]) + std::exp(out[2]);
    ASSERT(std::abs(sum_exp - 1.0f) < 1e-5f,
           "exp(log_softmax) should sum to 1");
}

void test_softmax_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t = Tensor::from_data(data.data(), {4}).gpu();

    auto result = ops::softmax(t, -1);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    ASSERT(result.shape() == Shape{4}, "Shape should be preserved");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    float sum = out[0] + out[1] + out[2] + out[3];
    ASSERT(std::abs(sum - 1.0f) < 1e-4f, "Softmax should sum to 1");
}

// ============================================================================
// ReLU Tests
// ============================================================================

void test_relu_cpu() {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {5});

    auto result = ops::relu(t);

    ASSERT(result.shape() == Shape{5}, "Shape should be preserved");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // ReLU: max(0, x)
    ASSERT(std::abs(out[0]) < 1e-5f, "ReLU(-2) should be 0");
    ASSERT(std::abs(out[1]) < 1e-5f, "ReLU(-1) should be 0");
    ASSERT(std::abs(out[2]) < 1e-5f, "ReLU(0) should be 0");
    ASSERT(std::abs(out[3] - 1.0f) < 1e-5f, "ReLU(1) should be 1");
    ASSERT(std::abs(out[4] - 2.0f) < 1e-5f, "ReLU(2) should be 2");
}

void test_relu_fluent() {
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    // Test fluent API
    auto result = t.relu();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    ASSERT(std::abs(out[0]) < 1e-5f, "ReLU(-1) should be 0");
    ASSERT(std::abs(out[1]) < 1e-5f, "ReLU(0) should be 0");
    ASSERT(std::abs(out[2] - 1.0f) < 1e-5f, "ReLU(1) should be 1");
}

void test_relu_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3}).gpu();

    auto result = t.relu();

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    ASSERT(std::abs(out[0]) < 1e-5f, "ReLU(-1) should be 0");
    ASSERT(std::abs(out[2] - 1.0f) < 1e-5f, "ReLU(1) should be 1");
}

// ============================================================================
// Sigmoid Tests
// ============================================================================

void test_sigmoid_cpu() {
    std::vector<float> data = {-2.0f, 0.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = t.sigmoid();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // sigmoid(0) = 0.5
    ASSERT(std::abs(out[1] - 0.5f) < 1e-5f, "sigmoid(0) should be 0.5");

    // sigmoid(-x) + sigmoid(x) = 1
    ASSERT(std::abs(out[0] + out[2] - 1.0f) < 1e-4f,
           "sigmoid(-x) + sigmoid(x) should be 1");

    // All values should be in (0, 1)
    ASSERT(out[0] > 0 && out[0] < 1, "sigmoid should be in (0, 1)");
    ASSERT(out[2] > 0 && out[2] < 1, "sigmoid should be in (0, 1)");
}

void test_sigmoid_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {0.0f};
    auto t = Tensor::from_data(data.data(), {1}).gpu();

    auto result = t.sigmoid();

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT(std::abs(out[0] - 0.5f) < 1e-5f, "sigmoid(0) should be 0.5");
}

// ============================================================================
// Tanh Tests
// ============================================================================

void test_tanh_cpu() {
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = t.tanh();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // tanh(0) = 0
    ASSERT(std::abs(out[1]) < 1e-5f, "tanh(0) should be 0");

    // tanh is odd: tanh(-x) = -tanh(x)
    ASSERT(std::abs(out[0] + out[2]) < 1e-5f, "tanh should be odd function");

    // All values should be in (-1, 1)
    ASSERT(out[0] > -1 && out[0] < 1, "tanh should be in (-1, 1)");
    ASSERT(out[2] > -1 && out[2] < 1, "tanh should be in (-1, 1)");
}

void test_tanh_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {0.0f};
    auto t = Tensor::from_data(data.data(), {1}).gpu();

    auto result = t.tanh();

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT(std::abs(out[0]) < 1e-5f, "tanh(0) should be 0");
}

// ============================================================================
// SiLU Tests
// ============================================================================

void test_silu_cpu() {
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = t.silu();

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // SiLU(0) = 0 * sigmoid(0) = 0
    ASSERT(std::abs(out[1]) < 1e-5f, "SiLU(0) should be 0");

    // SiLU(1) = 1 * sigmoid(1) ~= 0.731
    ASSERT(std::abs(out[2] - 0.731f) < 0.01f, "SiLU(1) should be ~0.731");
}

void test_silu_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {2}).gpu();

    auto result = t.silu();

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT(std::abs(out[0]) < 1e-5f, "SiLU(0) should be 0");
}

// ============================================================================
// Fluent API Chaining Tests
// ============================================================================

void test_fluent_chaining() {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {5});

    // Chain: (x * 2).relu().sigmoid()
    // Expression templates now support fluent chaining - still lazy!
    Tensor result = (t * 2.0f).relu().sigmoid();

    ASSERT(result.shape() == Shape{5}, "Shape should be preserved");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // After *2: {-4, -2, 0, 2, 4}
    // After relu: {0, 0, 0, 2, 4}
    // After sigmoid: {0.5, 0.5, 0.5, ~0.88, ~0.98}
    ASSERT(std::abs(out[0] - 0.5f) < 1e-5f, "sigmoid(relu(-4)) should be 0.5");
    ASSERT(std::abs(out[1] - 0.5f) < 1e-5f, "sigmoid(relu(-2)) should be 0.5");
    ASSERT(std::abs(out[2] - 0.5f) < 1e-5f, "sigmoid(relu(0)) should be 0.5");
    ASSERT(out[3] > 0.85f && out[3] < 0.92f, "sigmoid(2) should be ~0.88");
    ASSERT(out[4] > 0.95f, "sigmoid(4) should be > 0.95");
}

void test_fluent_chaining_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {3}).gpu();

    // Expression templates support fluent chaining - still lazy!
    Tensor result = (t + 1.0f).relu().gelu();

    ASSERT(result.device() == Device::GPU, "Result should stay on GPU");
    ASSERT(result.shape() == Shape{3}, "Shape should be preserved");
}

// ============================================================================
// GELU Tests
// ============================================================================

void test_gelu_cpu() {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {5});

    auto result = ops::gelu(t);

    ASSERT(result.shape() == Shape{5}, "Shape should be preserved");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // GELU(0) = 0
    ASSERT(std::abs(out[2]) < 1e-5f, "GELU(0) should be 0");

    // GELU(x) approaches x for large positive x
    ASSERT(out[4] > 1.9f && out[4] < 2.1f, "GELU(2) should be close to 2");

    // GELU(x) approaches 0 for large negative x
    ASSERT(out[0] < 0 && out[0] > -0.1f, "GELU(-2) should be close to 0");

    // GELU is monotonically increasing for x > 0
    ASSERT(out[3] < out[4], "GELU should be increasing for positive x");
}

void test_gelu_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {0.0f, 1.0f, 2.0f};
    auto t = Tensor::from_data(data.data(), {3}).gpu();

    auto result = ops::gelu(t);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    ASSERT(std::abs(out[0]) < 1e-5f, "GELU(0) should be 0");
}

// ============================================================================
// Erf Tests
// ============================================================================

void test_erf_cpu() {
    std::vector<float> data = {0.0f, 1.0f, -1.0f};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::erf(t);

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();

    // erf(0) = 0
    ASSERT(std::abs(out[0]) < 1e-5f, "erf(0) should be 0");

    // erf(1) ~= 0.8427
    ASSERT(std::abs(out[1] - 0.8427f) < 0.01f, "erf(1) should be ~0.8427");

    // erf is odd: erf(-x) = -erf(x)
    ASSERT(std::abs(out[1] + out[2]) < 1e-5f, "erf should be odd function");
}

void test_erf_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    std::vector<float> data = {0.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {2}).gpu();

    auto result = ops::erf(t);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
}

// ============================================================================
// Dropout Tests
// ============================================================================

void test_dropout_training() {
    auto t = Tensor::ones({100});

    auto [output, mask] = ops::dropout(t, 0.5f, true);

    ASSERT(output.shape() == Shape{100}, "Output shape should match");
    ASSERT(mask.shape() == Shape{100}, "Mask shape should match");
    ASSERT(mask.dtype() == DType::Bool, "Mask should be Bool");

    auto output_cpu = output.cpu();
    const float *out = output_cpu.typed_data<float>();

    // With 50% dropout, some values should be 0 and some should be ~2 (scaled)
    int zeros = 0;
    int scaled = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(out[i]) < 1e-5f) {
            zeros++;
        } else if (std::abs(out[i] - 2.0f) < 0.1f) {
            scaled++;
        }
    }

    // Should have roughly 50% zeros (allow some variance)
    ASSERT(zeros > 20 && zeros < 80, "Should have ~50% zeros with p=0.5");
    ASSERT(scaled > 20 && scaled < 80, "Should have ~50% scaled values");
}

void test_dropout_inference() {
    auto t = Tensor::ones({10});

    auto [output, mask] = ops::dropout(t, 0.5f, false);

    // In inference mode (training=false), dropout should be identity
    assert_tensor_near<float>(
        output, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
}

void test_dropout_p_zero() {
    auto t = Tensor::ones({10});

    auto [output, mask] = ops::dropout(t, 0.0f, true);

    // With p=0, no dropout occurs
    assert_tensor_near<float>(
        output, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
}

// ============================================================================
// Boolean Reduction Tests
// ============================================================================

void test_any_cpu() {
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f};
    auto t = Tensor::from_data(data.data(), {4});

    auto result = ops::any(t, {}, false);

    ASSERT(result.size() == 1, "Should reduce to scalar");
    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT(out[0] != 0.0f, "any([0,0,1,0]) should be true");
}

void test_all_cpu() {
    std::vector<float> data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {4});

    auto result = ops::all(t, {}, false);

    ASSERT(result.size() == 1, "Should reduce to scalar");
    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT(out[0] != 0.0f, "all([1,1,1,1]) should be true");
}

void test_all_with_zero_cpu() {
    std::vector<float> data = {1.0f, 0.0f, 1.0f, 1.0f};
    auto t = Tensor::from_data(data.data(), {4});

    auto result = ops::all(t, {}, false);

    auto result_cpu = result.cpu();
    const float *out = result_cpu.typed_data<float>();
    ASSERT(out[0] == 0.0f, "all([1,0,1,1]) should be false");
}

int main() {
    // Initialize operations registry
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Activation and Reduction Tests ===" << std::endl
              << std::endl;

    // ReLU tests
    std::cout << "--- ReLU Tests ---" << std::endl;
    RUN_TEST(test_relu_cpu);
    RUN_TEST(test_relu_fluent);
    RUN_TEST(test_relu_gpu);

    // Sigmoid tests
    std::cout << "--- Sigmoid Tests ---" << std::endl;
    RUN_TEST(test_sigmoid_cpu);
    RUN_TEST(test_sigmoid_gpu);

    // Tanh tests
    std::cout << "--- Tanh Tests ---" << std::endl;
    RUN_TEST(test_tanh_cpu);
    RUN_TEST(test_tanh_gpu);

    // SiLU tests
    std::cout << "--- SiLU Tests ---" << std::endl;
    RUN_TEST(test_silu_cpu);
    RUN_TEST(test_silu_gpu);

    // Fluent chaining tests
    std::cout << "--- Fluent Chaining Tests ---" << std::endl;
    RUN_TEST(test_fluent_chaining);
    RUN_TEST(test_fluent_chaining_gpu);

    // Softmax tests
    std::cout << "--- Softmax Tests ---" << std::endl;
    RUN_TEST(test_softmax_1d_cpu);
    RUN_TEST(test_softmax_2d_cpu);
    RUN_TEST(test_log_softmax_cpu);
    RUN_TEST(test_softmax_gpu);

    // GELU tests
    std::cout << "--- GELU Tests ---" << std::endl;
    RUN_TEST(test_gelu_cpu);
    RUN_TEST(test_gelu_gpu);

    // Erf tests
    std::cout << "--- Erf Tests ---" << std::endl;
    RUN_TEST(test_erf_cpu);
    RUN_TEST(test_erf_gpu);

    // Dropout tests
    std::cout << "--- Dropout Tests ---" << std::endl;
    RUN_TEST(test_dropout_training);
    RUN_TEST(test_dropout_inference);
    RUN_TEST(test_dropout_p_zero);

    // Boolean reduction tests
    std::cout << "--- Boolean Reduction Tests ---" << std::endl;
    RUN_TEST(test_any_cpu);
    RUN_TEST(test_all_cpu);
    RUN_TEST(test_all_with_zero_cpu);

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
