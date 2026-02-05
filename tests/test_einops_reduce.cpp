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
// Advanced Reduce Tests
// ============================================================================

void test_reduce_prod() {
    // Product reduction
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    auto result = einops::reduce(x, "h w -> h", "prod");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    // Row 0: 1*2*3 = 6, Row 1: 4*5*6 = 120
    assert_tensor_near<float>(result, {6.0f, 120.0f});
}

void test_reduce_unity_output() {
    // Unity axes in output (keepdims-like behavior)
    // Like: x - reduce(x, 'b c h w -> b c 1 1', 'mean')
    std::vector<float> x_data;
    for (int i = 0; i < 24; ++i) {
        x_data.push_back(static_cast<float>(i + 1));
    }
    auto x = Tensor::from_data(x_data.data(), {2, 3, 2, 2});

    // Mean over h and w, keeping dims as size 1
    auto result = einops::reduce(x, "b c h w -> b c 1 1", "mean");

    ASSERT(result.shape() == Shape({2, 3, 1, 1}),
           "Shape should be (2, 3, 1, 1) but got " +
               std::to_string(result.shape()[0]) + ", " +
               std::to_string(result.shape()[1]) + ", " +
               std::to_string(result.shape()[2]) + ", " +
               std::to_string(result.shape()[3]));
}

void test_reduce_empty_composition() {
    // Empty composition () for unity axes in output
    // Same as 'b c h w -> b c 1 1' but using ()
    std::vector<float> x_data;
    for (int i = 0; i < 24; ++i) {
        x_data.push_back(static_cast<float>(i + 1));
    }
    auto x = Tensor::from_data(x_data.data(), {2, 3, 2, 2});

    auto result = einops::reduce(x, "b c h w -> b c () ()", "mean");

    ASSERT(result.shape() == Shape({2, 3, 1, 1}),
           "Shape should be (2, 3, 1, 1)");
}

void test_reduce_pooling_with_axis_sizes() {
    // 2D max-pooling with kernel size 2x2
    // reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)
    std::vector<float> x_data;
    for (int i = 0; i < 64; ++i) {
        x_data.push_back(static_cast<float>(i));
    }
    // Shape: batch=2, channel=2, height=4, width=4
    auto x = Tensor::from_data(x_data.data(), {2, 2, 4, 4});

    auto result = einops::reduce(x, "b c (h1 h2) (w1 w2) -> b c h1 w1", "max",
                                 {{"h2", 2}, {"w2", 2}});

    // Output shape should be (2, 2, 2, 2) - height and width halved
    ASSERT(result.shape() == Shape({2, 2, 2, 2}),
           "Pooling output shape should be (2, 2, 2, 2)");
}

void test_reduce_anonymous_axes() {
    // Anonymous axes: reduce(x, 'b c (h1 2) (w1 2) -> b c h1 w1', 'max')
    std::vector<float> x_data;
    for (int i = 0; i < 64; ++i) {
        x_data.push_back(static_cast<float>(i));
    }
    // Shape: batch=2, channel=2, height=4, width=4
    auto x = Tensor::from_data(x_data.data(), {2, 2, 4, 4});

    auto result = einops::reduce(x, "b c (h1 2) (w1 2) -> b c h1 w1", "max");

    // Output shape should be (2, 2, 2, 2) - height and width halved
    ASSERT(result.shape() == Shape({2, 2, 2, 2}),
           "Anonymous axis pooling output shape should be (2, 2, 2, 2)");
}

void test_reduce_global_pooling() {
    // Global average pooling: reduce(x, 'b c h w -> b c', 'mean')
    std::vector<float> x_data;
    for (int i = 0; i < 48; ++i) {
        x_data.push_back(static_cast<float>(i + 1));
    }
    // Shape: batch=2, channel=3, height=2, width=4
    auto x = Tensor::from_data(x_data.data(), {2, 3, 2, 4});

    auto result = einops::reduce(x, "b c h w -> b c", "mean");

    ASSERT(result.shape() == Shape({2, 3}),
           "Global pooling output shape should be (2, 3)");
}

void test_reduce_adaptive_pooling() {
    // Adaptive pooling: reduce to specific grid size
    // reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=2, w1=2)
    std::vector<float> x_data;
    for (int i = 0; i < 72; ++i) {
        x_data.push_back(static_cast<float>(i));
    }
    // Shape: batch=2, channel=1, height=6, width=6
    auto x = Tensor::from_data(x_data.data(), {2, 1, 6, 6});

    auto result = einops::reduce(x, "b c (h1 h2) (w1 w2) -> b c h1 w1", "max",
                                 {{"h1", 2}, {"w1", 2}});

    // Output shape should be (2, 1, 2, 2)
    // h2 = 6/2 = 3, w2 = 6/2 = 3
    ASSERT(result.shape() == Shape({2, 1, 2, 2}),
           "Adaptive pooling output shape should be (2, 1, 2, 2)");
}

void test_reduce_first_dim() {
    // Reduce over the first dimension
    // reduce(x, 't b c -> b c', 'max')
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    // Shape: time=2, batch=2, channel=3
    auto x = Tensor::from_data(x_data.data(), {2, 2, 3});

    auto result = einops::reduce(x, "t b c -> b c", "max");

    ASSERT(result.shape() == Shape({2, 3}), "Shape should be (2, 3)");
    // Max over time dimension:
    // batch 0: max(1,7), max(2,8), max(3,9) = 7, 8, 9
    // batch 1: max(4,10), max(5,11), max(6,12) = 10, 11, 12
    assert_tensor_near<float>(result, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
}

void test_reduce_any() {
    // Test 'any' reduction (logical OR)
    std::vector<uint8_t> x_data = {0, 0, 1, 0, 1, 1};
    auto x = Tensor::from_data(x_data.data(), {2, 3}).astype(DType::Bool);

    auto result = einops::reduce(x, "h w -> h", "any");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    ASSERT(result.dtype() == DType::Bool, "Should return Bool tensor");
    const bool *data = result.typed_data<bool>();
    // Row 0: any(false, false, true) = true
    // Row 1: any(false, true, true) = true
    ASSERT(data[0] == true, "Row 0 should be true");
    ASSERT(data[1] == true, "Row 1 should be true");
}

void test_reduce_all() {
    // Test 'all' reduction (logical AND)
    std::vector<uint8_t> x_data = {1, 1, 1, 0, 1, 1};
    auto x = Tensor::from_data(x_data.data(), {2, 3}).astype(DType::Bool);

    auto result = einops::reduce(x, "h w -> h", "all");

    ASSERT(result.shape() == Shape({2}), "Shape mismatch");
    ASSERT(result.dtype() == DType::Bool, "Should return Bool tensor");
    const bool *data = result.typed_data<bool>();
    // Row 0: all(true, true, true) = true
    // Row 1: all(false, true, true) = false
    ASSERT(data[0] == true, "Row 0 should be true");
    ASSERT(data[1] == false, "Row 1 should be false");
}

void test_reduce_pooling_values() {
    // Verify correct pooling values with 2x2 max pooling
    std::vector<float> x_data = {
        // Channel 0: 4x4 grid with known max values per 2x2 block
        1.0f,  2.0f,
        3.0f,  4.0f, // row 0
        5.0f,  6.0f,
        7.0f,  8.0f, // row 1  -> max of top-left 2x2 = 6, top-right 2x2 = 8
        9.0f,  10.0f,
        11.0f, 12.0f, // row 2
        13.0f, 14.0f,
        15.0f, 16.0f // row 3  -> max of bot-left 2x2 = 14, bot-right 2x2 = 16
    };
    // Shape: batch=1, channel=1, height=4, width=4
    auto x = Tensor::from_data(x_data.data(), {1, 1, 4, 4});

    auto result = einops::reduce(x, "b c (h1 2) (w1 2) -> b c h1 w1", "max");

    ASSERT(result.shape() == Shape({1, 1, 2, 2}),
           "Shape should be (1, 1, 2, 2)");
    assert_tensor_near<float>(result, {6.0f, 8.0f, 14.0f, 16.0f});
}

void test_reduce_mean_pooling_values() {
    // Verify correct mean pooling values with 2x2 mean pooling
    std::vector<float> x_data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                 7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                 13.0f, 14.0f, 15.0f, 16.0f};
    // Shape: batch=1, channel=1, height=4, width=4
    auto x = Tensor::from_data(x_data.data(), {1, 1, 4, 4});

    auto result = einops::reduce(x, "b c (h1 2) (w1 2) -> b c h1 w1", "mean");

    ASSERT(result.shape() == Shape({1, 1, 2, 2}),
           "Shape should be (1, 1, 2, 2)");
    // Mean of top-left 2x2: (1+2+5+6)/4 = 3.5
    // Mean of top-right 2x2: (3+4+7+8)/4 = 5.5
    // Mean of bot-left 2x2: (9+10+13+14)/4 = 11.5
    // Mean of bot-right 2x2: (11+12+15+16)/4 = 13.5
    assert_tensor_near<float>(result, {3.5f, 5.5f, 11.5f, 13.5f});
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

    // Advanced reduce tests
    std::cout << "--- Advanced Reduce Tests ---" << std::endl;
    RUN_TEST(test_reduce_prod);
    RUN_TEST(test_reduce_unity_output);
    RUN_TEST(test_reduce_empty_composition);
    RUN_TEST(test_reduce_pooling_with_axis_sizes);
    RUN_TEST(test_reduce_anonymous_axes);
    RUN_TEST(test_reduce_global_pooling);
    RUN_TEST(test_reduce_adaptive_pooling);
    RUN_TEST(test_reduce_first_dim);
    RUN_TEST(test_reduce_any);
    RUN_TEST(test_reduce_all);
    RUN_TEST(test_reduce_pooling_values);
    RUN_TEST(test_reduce_mean_pooling_values);

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
