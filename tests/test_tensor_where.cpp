#include <axiom/axiom.hpp>
#include <axiom/tensor_operators.hpp>
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
                        T tol = T(1e-5)) {
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

// Helper to create bool tensor
Tensor make_bool_tensor(const std::vector<bool> &data, const Shape &shape,
                        Device device = Device::CPU) {
    // Convert bool vector to uint8 for storage
    std::vector<uint8_t> u8_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        u8_data[i] = data[i] ? 1 : 0;
    }
    auto t = Tensor::from_data(u8_data.data(), shape).astype(DType::Bool);
    if (device == Device::GPU) {
        return t.to(Device::GPU);
    }
    return t;
}

// ============================================================================
// CPU Tests
// ============================================================================

void test_where_basic_cpu() {
    auto cond = make_bool_tensor({true, false, true, false}, {4});
    auto a = Tensor::full({4}, 1.0f);
    auto b = Tensor::full({4}, 0.0f);

    auto result = ops::where(cond, a, b);

    ASSERT(result.device() == Device::CPU, "Result should be on CPU");
    ASSERT(result.shape() == Shape{4}, "Shape mismatch");
    assert_tensor_near<float>(result, {1.0f, 0.0f, 1.0f, 0.0f});
}

void test_where_2d_cpu() {
    auto cond = make_bool_tensor({true, false, false, true}, {2, 2});
    auto a = Tensor::full({2, 2}, 10.0f);
    auto b = Tensor::full({2, 2}, -10.0f);

    auto result = ops::where(cond, a, b);

    ASSERT(result.shape() == Shape({2, 2}), "Shape mismatch");
    assert_tensor_near<float>(result, {10.0f, -10.0f, -10.0f, 10.0f});
}

void test_where_broadcast_condition_cpu() {
    // Condition: [true, false], shape (2,)
    // a, b: shape (3, 2)
    auto cond = make_bool_tensor({true, false}, {2});
    auto a = Tensor::ones({3, 2});
    auto b = Tensor::zeros({3, 2});

    auto result = ops::where(cond, a, b);

    ASSERT(result.shape() == Shape({3, 2}), "Shape mismatch");
    // Each row should be [1, 0]
    assert_tensor_near<float>(result, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
}

void test_where_broadcast_values_cpu() {
    // Condition: shape (2, 2)
    // a: scalar-like shape (1,)
    // b: scalar-like shape (1,)
    auto cond = make_bool_tensor({true, false, true, true}, {2, 2});
    auto a = Tensor::full({1}, 5.0f);
    auto b = Tensor::full({1}, -5.0f);

    auto result = ops::where(cond, a, b);

    ASSERT(result.shape() == Shape({2, 2}), "Shape mismatch");
    assert_tensor_near<float>(result, {5.0f, -5.0f, 5.0f, 5.0f});
}

void test_where_int_condition_cpu() {
    // Non-bool condition (non-zero = true)
    std::vector<int32_t> cond_data = {1, 0, 2, 0};
    auto cond = Tensor::from_data(cond_data.data(), {4});
    auto a = Tensor::full({4}, 100.0f);
    auto b = Tensor::full({4}, 0.0f);

    auto result = ops::where(cond, a, b);

    ASSERT(result.shape() == Shape{4}, "Shape mismatch");
    assert_tensor_near<float>(result, {100.0f, 0.0f, 100.0f, 0.0f});
}

void test_where_different_dtypes_cpu() {
    auto cond = make_bool_tensor({true, false}, {2});
    std::vector<int32_t> a_data = {10, 20};
    std::vector<float> b_data = {0.5f, 0.5f};
    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::where(cond, a, b);

    // Result should be promoted to float
    ASSERT(result.dtype() == DType::Float32, "Dtype should be Float32");
    assert_tensor_near<float>(result, {10.0f, 0.5f});
}

// ============================================================================
// GPU Tests
// ============================================================================

void test_where_basic_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    auto cond = make_bool_tensor({true, false, true, false}, {4}, Device::GPU);
    auto a = Tensor::full({4}, 1.0f).gpu();
    auto b = Tensor::full({4}, 0.0f).gpu();

    auto result = ops::where(cond, a, b);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    ASSERT(result.shape() == Shape{4}, "Shape mismatch");
    assert_tensor_near<float>(result, {1.0f, 0.0f, 1.0f, 0.0f});
}

void test_where_2d_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    auto cond =
        make_bool_tensor({true, false, false, true}, {2, 2}, Device::GPU);
    auto a = Tensor::full({2, 2}, 10.0f).gpu();
    auto b = Tensor::full({2, 2}, -10.0f).gpu();

    auto result = ops::where(cond, a, b);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    ASSERT(result.shape() == Shape({2, 2}), "Shape mismatch");
    assert_tensor_near<float>(result, {10.0f, -10.0f, -10.0f, 10.0f});
}

void test_where_broadcast_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    // Condition: [true, false], shape (2,)
    // a, b: shape (3, 2)
    auto cond = make_bool_tensor({true, false}, {2}, Device::GPU);
    auto a = Tensor::ones({3, 2}).gpu();
    auto b = Tensor::zeros({3, 2}).gpu();

    auto result = ops::where(cond, a, b);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    ASSERT(result.shape() == Shape({3, 2}), "Shape mismatch");
    assert_tensor_near<float>(result, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
}

void test_where_attention_mask_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    // Attention mask pattern: where(mask, scores, -1e9)
    auto mask =
        make_bool_tensor({true, true, false, false}, {2, 2}, Device::GPU);
    std::vector<float> scores_data = {0.5f, 0.3f, 0.2f, 0.1f};
    auto scores = Tensor::from_data(scores_data.data(), {2, 2}).gpu();
    auto neg_inf = Tensor::full({2, 2}, -1e9f).gpu();

    auto result = ops::where(mask, scores, neg_inf);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    assert_tensor_near<float>(result, {0.5f, 0.3f, -1e9f, -1e9f}, 1.0f);
}

void test_where_with_comparison_result() {
    // Test using comparison result as condition
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f};
    auto x = Tensor::from_data(x_data.data(), {4});
    auto zero = Tensor::zeros({4});
    auto cond =
        ops::greater(x, zero); // x > 0 - Should be {true, false, true, false}

    auto positive = Tensor::full({4}, 1.0f);
    auto negative = Tensor::full({4}, -1.0f);

    auto result = ops::where(cond, positive, negative);

    assert_tensor_near<float>(result, {1.0f, -1.0f, 1.0f, -1.0f});
}

void test_where_relu_pattern() {
    // Common pattern: ReLU using where
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f, 0.0f};
    auto x = Tensor::from_data(x_data.data(), {5});
    auto zero = Tensor::zeros({5});
    auto cond = ops::greater(x, zero); // x > 0

    auto result = ops::where(cond, x, zero);

    assert_tensor_near<float>(result, {1.0f, 0.0f, 3.0f, 0.0f, 0.0f});
}

// ============================================================================
// Fluent API Tests
// ============================================================================

void test_fluent_masked_fill() {
    // Test masked_fill with scalar comparison
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f, 0.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    // ReLU using masked_fill: zero out negative values
    auto mask = x < 0.0f;
    auto result = x.masked_fill(mask, 0.0f);

    assert_tensor_near<float>(result, {1.0f, 0.0f, 3.0f, 0.0f, 0.0f});
}

void test_fluent_masked_select() {
    // Test masked_select
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    // Get all positive values
    auto mask = x > 0.0f;
    auto positives = x.masked_select(mask);

    ASSERT(positives.ndim() == 1, "Result should be 1D");
    ASSERT(positives.size() == 3, "Should have 3 positive values");
    assert_tensor_near<float>(positives, {1.0f, 3.0f, 5.0f});
}

void test_fluent_where_method() {
    // Test the fluent where method
    std::vector<float> x_data = {1.0f, -2.0f, 3.0f, -4.0f};
    auto x = Tensor::from_data(x_data.data(), {4});

    // ReLU using fluent where
    auto result = x.where(x > 0.0f, 0.0f);

    assert_tensor_near<float>(result, {1.0f, 0.0f, 3.0f, 0.0f});
}

void test_scalar_comparison_operators() {
    // Test scalar comparison operators for clean syntax
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    // Test x > 3
    Tensor gt3 = x > 3.0f; // Expression evaluates via implicit conversion
    auto gt3_cpu = gt3.cpu();
    const uint8_t *gt3_data = gt3_cpu.typed_data<uint8_t>();
    ASSERT(gt3_data[0] == 0 && gt3_data[1] == 0 && gt3_data[2] == 0,
           "1,2,3 should NOT be > 3");
    ASSERT(gt3_data[3] == 1 && gt3_data[4] == 1, "4,5 should be > 3");

    // Test x <= 2
    Tensor le2 = x <= 2.0f; // Expression evaluates via implicit conversion
    auto le2_cpu = le2.cpu();
    const uint8_t *le2_data = le2_cpu.typed_data<uint8_t>();
    ASSERT(le2_data[0] == 1 && le2_data[1] == 1, "1,2 should be <= 2");
    ASSERT(le2_data[2] == 0 && le2_data[3] == 0 && le2_data[4] == 0,
           "3,4,5 should NOT be <= 2");
}

void test_attention_mask_fluent() {
    // Attention masking pattern using fluent API
    std::vector<float> scores_data = {0.5f, 0.3f, 0.2f, 0.1f};
    auto scores = Tensor::from_data(scores_data.data(), {2, 2});

    // Create causal mask (lower triangular)
    auto mask = make_bool_tensor({true, false, true, true}, {2, 2});

    // Apply attention mask using fluent API
    auto masked_scores = scores.masked_fill(!mask, -1e9f);

    assert_tensor_near<float>(masked_scores, {0.5f, -1e9f, 0.2f, 0.1f}, 1.0f);
}

int main() {
    // Initialize operations registry
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Where Operation Tests ===" << std::endl << std::endl;

    // CPU tests
    std::cout << "--- CPU Tests ---" << std::endl;
    RUN_TEST(test_where_basic_cpu);
    RUN_TEST(test_where_2d_cpu);
    RUN_TEST(test_where_broadcast_condition_cpu);
    RUN_TEST(test_where_broadcast_values_cpu);
    RUN_TEST(test_where_int_condition_cpu);
    RUN_TEST(test_where_different_dtypes_cpu);

    // GPU tests
    std::cout << "--- GPU Tests ---" << std::endl;
    RUN_TEST(test_where_basic_gpu);
    RUN_TEST(test_where_2d_gpu);
    RUN_TEST(test_where_broadcast_gpu);
    RUN_TEST(test_where_attention_mask_gpu);

    // Integration tests
    std::cout << "--- Integration Tests ---" << std::endl;
    RUN_TEST(test_where_with_comparison_result);
    RUN_TEST(test_where_relu_pattern);

    // Fluent API tests
    std::cout << "--- Fluent API Tests ---" << std::endl;
    RUN_TEST(test_fluent_masked_fill);
    RUN_TEST(test_fluent_masked_select);
    RUN_TEST(test_fluent_where_method);
    RUN_TEST(test_scalar_comparison_operators);
    RUN_TEST(test_attention_mask_fluent);

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
