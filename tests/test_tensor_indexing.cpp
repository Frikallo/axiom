#include <axiom/axiom.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

// ==================================
//
//      TEST HARNESS
//
// ==================================

static int tests_run = 0;
static int tests_passed = 0;
static std::string current_test_name;

#define RUN_TEST(test_func, ...)                                               \
    run_test([&]() { test_func(__VA_ARGS__); }, #test_func)

void run_test(const std::function<void()> &test_func,
              const std::string &test_name) {
    tests_run++;
    current_test_name = test_name;
    std::cout << "--- Running: " << test_name << " ---" << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << " ---" << std::endl;
        tests_passed++;
    } catch (const std::exception &e) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: Unknown exception caught." << std::endl;
    }
    std::cout << std::endl;
}

int print_test_summary() {
    std::cout << "========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}

// ==================================
//
//      CUSTOM ASSERTIONS
//
// ==================================

#define ASSERT(condition, msg)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + \
                                     std::string(msg));                        \
        }                                                                      \
    } while (0)

#define ASSERT_THROWS(expression)                                              \
    do {                                                                       \
        try {                                                                  \
            (expression);                                                      \
            throw std::runtime_error(                                          \
                "Expected exception was not thrown for: " #expression);        \
        } catch (const std::exception &e) {                                    \
            (void)e; /* Caught expected exception */                           \
        }                                                                      \
    } while (0)

template <typename T>
void assert_tensor_equals_cpu(const axiom::Tensor &t,
                              const std::vector<T> &expected_data,
                              double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");

    if (t.is_contiguous()) {
        const T *t_data = t_cpu.template typed_data<T>();
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                if (std::abs(static_cast<double>(t_data[i]) -
                             static_cast<double>(expected_data[i])) >=
                    epsilon) {
                    throw std::runtime_error("Tensor data mismatch at index " +
                                             std::to_string(i));
                }
            } else {
                if (t_data[i] != expected_data[i]) {
                    throw std::runtime_error("Tensor data mismatch at index " +
                                             std::to_string(i));
                }
            }
        }
    } else {
        // Fallback for non-contiguous tensors
        std::vector<size_t> indices(t.ndim(), 0);
        for (size_t i = 0; i < expected_data.size(); ++i) {
            T val = t.item<T>(indices);
            if constexpr (std::is_floating_point_v<T>) {
                if (std::abs(static_cast<double>(val) -
                             static_cast<double>(expected_data[i])) >=
                    epsilon) {
                    throw std::runtime_error("Tensor data mismatch at index " +
                                             std::to_string(i));
                }
            } else {
                if (val != expected_data[i]) {
                    throw std::runtime_error("Tensor data mismatch at index " +
                                             std::to_string(i));
                }
            }

            // Increment indices
            for (int j = t.ndim() - 1; j >= 0; --j) {
                if (++indices[j] < t.shape()[j]) {
                    break;
                }
                indices[j] = 0;
            }
        }
    }
}

void test_simple_indexing_slice() {
    auto a = axiom::Tensor::arange(0, 10); // {0, 1, ..., 9}
    auto b = a[{axiom::Slice(2, 5)}];      // {2, 3, 4}
    ASSERT(b.shape() == axiom::Shape({3}),
           "Simple indexing slice shape mismatch");
    assert_tensor_equals_cpu<int>(b, {2, 3, 4});
}

void test_simple_indexing_integer() {
    auto data = std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7};
    auto a = axiom::Tensor::from_data(data.data(), {2, 4});
    auto b = a[{0}]; // First row
    ASSERT(b.shape() == axiom::Shape({4}),
           "Simple indexing integer shape mismatch");
    assert_tensor_equals_cpu<float>(b, {0, 1, 2, 3});
}

void test_mixed_indexing() {
    auto a = axiom::Tensor::arange(0, 16).reshape({4, 4});
    // Corresponds to numpy's a[1, 1:3] -> {5, 6}
    auto b = a[{1, axiom::Slice(1, 3)}];
    ASSERT(b.shape() == axiom::Shape({2}), "Mixed indexing shape mismatch");
    assert_tensor_equals_cpu<int>(b, {5, 6});
}

void test_negative_indexing() {
    auto a = axiom::Tensor::arange(0, 5); // {0, 1, 2, 3, 4}
    auto b = a[{-1}];                     // Last element
    ASSERT(b.shape() == axiom::Shape{}, "Negative indexing shape mismatch");
    ASSERT(b.item<int>({}) == 4, "Negative indexing value mismatch");

    // Corresponds to numpy's a[-3:-1] -> {2, 3}
    auto c = a[{axiom::Slice(-3, -1)}];
    ASSERT(c.shape() == axiom::Shape({2}), "Negative slice shape mismatch");
    assert_tensor_equals_cpu<int>(c, {2, 3});
}

void test_indexing_shared_storage() {
    auto a = axiom::Tensor::zeros({4, 4});
    // Get a view of the second row
    auto b = a[{1}];
    // Fill the view with 1s
    b.fill(1.0f);

    // The original tensor should now be modified
    assert_tensor_equals_cpu<float>(
        a, {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    RUN_TEST(test_simple_indexing_slice);
    RUN_TEST(test_simple_indexing_integer);
    RUN_TEST(test_mixed_indexing);
    RUN_TEST(test_negative_indexing);
    RUN_TEST(test_indexing_shared_storage);

    return print_test_summary();
}