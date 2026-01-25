#include <axiom/axiom.hpp>
#include <axiom/einops.hpp>
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
    ASSERT(t_cpu.size() == expected.size(),
           "Size mismatch: got " + std::to_string(t_cpu.size()) + " expected " +
               std::to_string(expected.size()));
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
// Basic Reduce Tests
// ============================================================================

void test_reduce_sum_simple() {
    // Sum over one dimension
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    // Sum over dimension 1 (columns)
    auto result = einops::reduce(x, "h w -> h", "sum");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    // Row 0: 1+2+3 = 6, Row 1: 4+5+6 = 15
    assert_tensor_near<float>(result, {6.0f, 15.0f});
}

void test_reduce_mean_simple() {
    // Mean over one dimension
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    // Mean over dimension 1 (columns)
    auto result = einops::reduce(x, "h w -> h", "mean");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    // Row 0: (1+2+3)/3 = 2, Row 1: (4+5+6)/3 = 5
    assert_tensor_near<float>(result, {2.0f, 5.0f});
}

void test_reduce_max_simple() {
    // Max over one dimension
    std::vector<float> x_data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    // Max over dimension 1
    auto result = einops::reduce(x, "h w -> h", "max");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    // Row 0: max(1,5,3) = 5, Row 1: max(4,2,6) = 6
    assert_tensor_near<float>(result, {5.0f, 6.0f});
}

void test_reduce_min_simple() {
    // Min over one dimension
    std::vector<float> x_data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    // Min over dimension 1
    auto result = einops::reduce(x, "h w -> h", "min");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    // Row 0: min(1,5,3) = 1, Row 1: min(4,2,6) = 2
    assert_tensor_near<float>(result, {1.0f, 2.0f});
}

void test_reduce_multiple_dims() {
    // Reduce over multiple dimensions
    std::vector<float> x_data;
    for (int i = 0; i < 24; i++) {
        x_data.push_back(static_cast<float>(i + 1));
    }
    auto x = Tensor::from_data(x_data.data(), {2, 3, 4});

    // Sum over h and w, keep c
    auto result = einops::reduce(x, "b h w -> b", "sum");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    // Batch 0: sum(1..12) = 78, Batch 1: sum(13..24) = 222
    assert_tensor_near<float>(result, {78.0f, 222.0f});
}

void test_reduce_global() {
    // Global reduction - reduce all dimensions to a single scalar
    // Use the standard reduction API for global reduction instead of einops
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 2});

    // Sum all elements using standard reduction
    auto result = x.sum();

    // Result should have 1 element
    ASSERT(result.size() == 1, "Should have 1 element");
    assert_tensor_near<float>(result, {10.0f});
}

// ============================================================================
// Tensor Method Tests
// ============================================================================

void test_tensor_reduce_method() {
    // Test the Tensor::reduce method
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    auto result = x.reduce("h w -> h", "mean");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    assert_tensor_near<float>(result, {2.0f, 5.0f});
}

// ============================================================================
// Rearrange Tests (existing functionality)
// ============================================================================

void test_rearrange_basic() {
    // Basic rearrange (transpose)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    auto result = einops::rearrange(x, "h w -> w h");

    ASSERT(result.shape() == Shape({3, 2}), "Shape mismatch");
}

void test_rearrange_flatten() {
    // Flatten using rearrange
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    auto result = einops::rearrange(x, "h w -> (h w)");

    ASSERT(result.shape() == Shape({6}), "Shape mismatch");
    assert_tensor_near<float>(result, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Einops Reduce Tests ===" << std::endl << std::endl;

    // Basic reduce tests
    std::cout << "--- Basic Reduce Tests ---" << std::endl;
    RUN_TEST(test_reduce_sum_simple);
    RUN_TEST(test_reduce_mean_simple);
    RUN_TEST(test_reduce_max_simple);
    RUN_TEST(test_reduce_min_simple);
    RUN_TEST(test_reduce_multiple_dims);
    RUN_TEST(test_reduce_global);

    // Tensor method tests
    std::cout << "--- Tensor Method Tests ---" << std::endl;
    RUN_TEST(test_tensor_reduce_method);

    // Rearrange tests
    std::cout << "--- Rearrange Tests ---" << std::endl;
    RUN_TEST(test_rearrange_basic);
    RUN_TEST(test_rearrange_flatten);

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
