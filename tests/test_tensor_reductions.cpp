#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <map>
#include <variant>
#include <iomanip>

#include <axiom/axiom.hpp>

// ==================================
//
//      TEST HARNESS
//
// ==================================

static int tests_run = 0;
static int tests_passed = 0;
static std::string current_test_name;

#define RUN_TEST(test_func, ...) run_test([&]() { test_func(__VA_ARGS__); }, #test_func)

void run_test(const std::function<void()>& test_func, const std::string& test_name) {
    tests_run++;
    current_test_name = test_name;
    std::cout << "--- Running: " << test_name << " ---" << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << " ---" << std::endl;
        tests_passed++;
    } catch (const std::exception& e) {
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
//      CUSTOM ASSERTIONS
//
// ==================================

#define ASSERT(condition, msg) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + std::string(msg)); \
        } \
    } while (0)

template<typename T>
void assert_tensor_equals_cpu(const axiom::Tensor& t, const std::vector<T>& expected_data, double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");
    
    const T* t_data = t_cpu.template typed_data<T>();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(static_cast<double>(t_data[i]) - static_cast<double>(expected_data[i])) >= epsilon) {
                throw std::runtime_error("Tensor data mismatch at index " + std::to_string(i));
            }
        } else {
            if (t_data[i] != expected_data[i]) {
                 throw std::runtime_error("Tensor data mismatch at index " + std::to_string(i));
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
    auto a = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a);
    assert_tensor_equals_cpu<float>(c, {15.0f});
}

void test_sum_axis0() {
    auto a = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a, {0});
    assert_tensor_equals_cpu<float>(c, {3.0f, 5.0f, 7.0f});
}

void test_sum_axis1() {
    auto a = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a, {1});
    assert_tensor_equals_cpu<float>(c, {3.0f, 12.0f});
}

void test_sum_axis_default() {
    auto a = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a);
    assert_tensor_equals_cpu<float>(c, {15.0f});
}

void test_sum_keepdims() {
    auto a = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::sum(a, {1}, true);
    ASSERT(c.shape() == axiom::Shape({2, 1}), "Shape mismatch");
    assert_tensor_equals_cpu<float>(c, {3.0f, 12.0f});
}

void test_mean_all() {
    auto a = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
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
//      MAIN
//
// ==================================

int main(int argc, char** argv) {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    RUN_TEST(test_sum_all);
    RUN_TEST(test_sum_axis0);
    RUN_TEST(test_sum_axis1);
    RUN_TEST(test_sum_axis_default);
    RUN_TEST(test_sum_keepdims);
    RUN_TEST(test_mean_all);
    RUN_TEST(test_max_all);
    RUN_TEST(test_min_all);

    std::cout << "\n----------------------------------\n";
    std::cout << "         TEST SUMMARY\n";
    std::cout << "----------------------------------\n";
    std::cout << "TOTAL TESTS: " << tests_run << std::endl;
    std::cout << "PASSED:      " << tests_passed << std::endl;
    std::cout << "FAILED:      " << tests_run - tests_passed << std::endl;
    std::cout << "----------------------------------\n";

    return (tests_run == tests_passed) ? 0 : 1;
} 