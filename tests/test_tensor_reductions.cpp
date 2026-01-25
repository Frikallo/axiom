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

// ==================================
//
//      UTILITIES
//
// ==================================

bool is_gpu_available() {
    // Use the system function which checks both Metal availability
    // and the AXIOM_SKIP_GPU_TESTS environment variable
    return axiom::system::should_run_gpu_tests();
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

template <typename T>
void assert_tensor_equals_cpu(const axiom::Tensor &t,
                              const std::vector<T> &expected_data,
                              double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");

    const T *t_data = t_cpu.template typed_data<T>();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(static_cast<double>(t_data[i]) -
                         static_cast<double>(expected_data[i])) >= epsilon) {
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
}

template <typename T>
void assert_tensor_equals(const axiom::Tensor &a, const axiom::Tensor &b,
                          double epsilon = 1e-6) {
    auto a_cpu = a.cpu();
    auto b_cpu = b.cpu();
    ASSERT(a_cpu.shape() == b_cpu.shape(), "Tensor shape mismatch");
    ASSERT(a_cpu.size() == b_cpu.size(), "Tensor size mismatch");

    const T *a_data = a_cpu.template typed_data<T>();
    const T *b_data = b_cpu.template typed_data<T>();
    for (size_t i = 0; i < a_cpu.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(static_cast<double>(a_data[i]) -
                         static_cast<double>(b_data[i])) >= epsilon) {
                throw std::runtime_error("Tensor data mismatch at index " +
                                         std::to_string(i));
            }
        } else {
            if (a_data[i] != b_data[i]) {
                throw std::runtime_error("Tensor data mismatch at index " +
                                         std::to_string(i));
            }
        }
    }
}

// ==================================
//
//      REDUCTION OP TESTS
//
// ==================================

void test_sum_all() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a);
    assert_tensor_equals_cpu<float>(c, {15.0f});
}

void test_sum_axis0() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a, {0});
    assert_tensor_equals_cpu<float>(c, {3.0f, 5.0f, 7.0f});
}

void test_sum_axis1() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a, {1});
    assert_tensor_equals_cpu<float>(c, {3.0f, 12.0f});
}

void test_sum_axis_default() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a);
    assert_tensor_equals_cpu<float>(c, {15.0f});
}

void test_sum_keepdims() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a, {1}, true);
    ASSERT(c.shape() == axiom::Shape({2, 1}), "Shape mismatch");
    assert_tensor_equals_cpu<float>(c, {3.0f, 12.0f});
}

void test_mean_all() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::mean(a);
    assert_tensor_equals_cpu<float>(c, {2.5f});
}

void test_max_all() {
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a = axiom::Tensor::from_data(data.data(), {2, 3});
    auto c = axiom::ops::max(a);
    assert_tensor_equals_cpu<float>(c, {9.0f});
}

void test_min_all() {
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a = axiom::Tensor::from_data(data.data(), {2, 3});
    auto c = axiom::ops::min(a);
    assert_tensor_equals_cpu<float>(c, {1.0f});
}

// ==================================
//
//      GPU REDUCTION OP TESTS
//
// ==================================

void test_sum_all_gpu() {
    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto c = axiom::ops::sum(a);
    assert_tensor_equals_cpu<float>(c, {15.0f});
}

void test_sum_axis0_gpu() {
    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto c = axiom::ops::sum(a, {0});
    assert_tensor_equals_cpu<float>(c, {3.0f, 5.0f, 7.0f});
}

void test_sum_axis1_gpu() {
    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto c = axiom::ops::sum(a, {1});
    assert_tensor_equals_cpu<float>(c, {3.0f, 12.0f});
}

void test_sum_keepdims_gpu() {
    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto c = axiom::ops::sum(a, {1}, true);
    ASSERT(c.shape() == axiom::Shape({2, 1}), "Shape mismatch");
    assert_tensor_equals_cpu<float>(c, {3.0f, 12.0f});
}

void test_mean_all_gpu() {
    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto c = axiom::ops::mean(a);
    assert_tensor_equals_cpu<float>(c, {2.5f});
}

void test_max_all_gpu() {
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a =
        axiom::Tensor::from_data(data.data(), {2, 3}).to(axiom::Device::GPU);
    auto c = axiom::ops::max(a);
    assert_tensor_equals_cpu<float>(c, {9.0f});
}

void test_min_all_gpu() {
    std::vector<float> data = {1, 5, 2, 9, 3, 4};
    auto a =
        axiom::Tensor::from_data(data.data(), {2, 3}).to(axiom::Device::GPU);
    auto c = axiom::ops::min(a);
    assert_tensor_equals_cpu<float>(c, {1.0f});
}

void test_non_contiguous_sum_gpu() {
    // Create a larger tensor ON THE GPU first
    auto a_gpu = axiom::Tensor::arange(24)
                     .reshape({2, 3, 4})
                     .astype(axiom::DType::Float32)
                     .to(axiom::Device::GPU);

    // Perform non-contiguous-making operations on the GPU tensor
    auto b_gpu = a_gpu.slice({axiom::Slice(), axiom::Slice(1, 3),
                              axiom::Slice()}); // Shape {2, 2, 4}
    auto c_gpu = b_gpu.transpose({2, 0, 1}); // Shape {4, 2, 2}, non-contiguous

    ASSERT(!c_gpu.is_contiguous(),
           "Tensor should be non-contiguous for this test");

    // Perform the reduction on the non-contiguous GPU tensor
    auto result_gpu = axiom::ops::sum(c_gpu, {1, 2});

    // Create the equivalent non-contiguous tensor on CPU for verification
    auto a_cpu = axiom::Tensor::arange(24).reshape({2, 3, 4}).astype(
        axiom::DType::Float32);
    auto b_cpu =
        a_cpu.slice({axiom::Slice(), axiom::Slice(1, 3), axiom::Slice()});
    auto c_cpu = b_cpu.transpose({2, 0, 1});
    auto result_cpu = axiom::ops::sum(c_cpu, {1, 2});

    // Compare the results
    assert_tensor_equals<float>(result_gpu, result_cpu);
}

// ==================================
//
//      MAIN
//
// ==================================

int main(int argc, char **argv) {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "--- RUNNING CPU TESTS ---\n";
    RUN_TEST(test_sum_all);
    RUN_TEST(test_sum_axis0);
    RUN_TEST(test_sum_axis1);
    RUN_TEST(test_sum_axis_default);
    RUN_TEST(test_sum_keepdims);
    RUN_TEST(test_mean_all);
    RUN_TEST(test_max_all);
    RUN_TEST(test_min_all);

    if (is_gpu_available()) {
        std::cout << "\n--- RUNNING GPU TESTS ---\n";
        RUN_TEST(test_sum_all_gpu);
        RUN_TEST(test_sum_axis0_gpu);
        RUN_TEST(test_sum_axis1_gpu);
        RUN_TEST(test_sum_keepdims_gpu);
        RUN_TEST(test_mean_all_gpu);
        RUN_TEST(test_max_all_gpu);
        RUN_TEST(test_min_all_gpu);
        RUN_TEST(test_non_contiguous_sum_gpu);
    } else {
        std::cout << "\n--- SKIPPING GPU TESTS (GPU not available) ---\n";
    }

    std::cout << "\n----------------------------------\n";
    std::cout << "         TEST SUMMARY\n";
    std::cout << "----------------------------------\n";
    std::cout << "TOTAL TESTS: " << tests_run << std::endl;
    std::cout << "PASSED:      " << tests_passed << std::endl;
    std::cout << "FAILED:      " << tests_run - tests_passed << std::endl;
    std::cout << "----------------------------------\n";

    return (tests_run == tests_passed) ? 0 : 1;
}