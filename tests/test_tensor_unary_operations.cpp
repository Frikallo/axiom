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

#define RUN_TEST(test_func, device)                                            \
    run_test([&]() { test_func(device); }, #test_func,                         \
             std::string(" (") + axiom::system::device_to_string(device) +     \
                 ")")

void run_test(const std::function<void()> &test_func,
              const std::string &test_name, const std::string &device_str) {
    tests_run++;
    current_test_name = test_name;
    std::cout << "--- Running: " << test_name << device_str << " ---"
              << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << device_str << " ---"
                  << std::endl;
        tests_passed++;
    } catch (const std::exception &e) {
        std::cerr << "--- FAILED: " << test_name << device_str << " ---"
                  << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "--- FAILED: " << test_name << device_str << " ---"
                  << std::endl;
        std::cerr << "    Error: Unknown exception caught." << std::endl;
    }
    std::cout << std::endl;
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

// ==================================
//
//      UNARY OP TESTS
//
// ==================================

void test_negate(axiom::Device device) {
    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(device);
    auto b = axiom::ops::negate(a);
    assert_tensor_equals_cpu<float>(b, {0, -1, -2, -3, -4, -5});

    // Test with operator overload
    auto c = -a;
    assert_tensor_equals_cpu<float>(c, {0, -1, -2, -3, -4, -5});
}

void test_abs(axiom::Device device) {
    auto data = std::vector<float>({-1, 2, -3, 0, -5, 6});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = axiom::ops::abs(a);
    assert_tensor_equals_cpu<float>(c, {1, 2, 3, 0, 5, 6});
}

void test_sqrt(axiom::Device device) {
    auto data = std::vector<float>({0, 1, 4, 9, 16, 25});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = axiom::ops::sqrt(a);
    assert_tensor_equals_cpu<float>(c, {0, 1, 2, 3, 4, 5});
}

void test_exp(axiom::Device device) {
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = axiom::ops::exp(a);
    assert_tensor_equals_cpu<float>(c,
                                    {std::exp(0.0f), std::exp(1.0f),
                                     std::exp(2.0f), std::exp(3.0f),
                                     std::exp(4.0f), std::exp(5.0f)},
                                    1e-5);
}

void test_log(axiom::Device device) {
    auto data = std::vector<float>({1, 2, 3, 4, 5, 6});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = axiom::ops::log(a);
    assert_tensor_equals_cpu<float>(c, {std::log(1.0f), std::log(2.0f),
                                        std::log(3.0f), std::log(4.0f),
                                        std::log(5.0f), std::log(6.0f)});
}

void test_sin(axiom::Device device) {
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = axiom::ops::sin(a);
    assert_tensor_equals_cpu<float>(c, {std::sin(0.0f), std::sin(1.0f),
                                        std::sin(2.0f), std::sin(3.0f),
                                        std::sin(4.0f), std::sin(5.0f)});
}

void test_cos(axiom::Device device) {
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = axiom::ops::cos(a);
    assert_tensor_equals_cpu<float>(c, {std::cos(0.0f), std::cos(1.0f),
                                        std::cos(2.0f), std::cos(3.0f),
                                        std::cos(4.0f), std::cos(5.0f)});
}

void test_tan(axiom::Device device) {
    auto data = std::vector<float>({0, 1, 2, 3, 4, 5});
    auto a = axiom::Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    auto c = axiom::ops::tan(a);
    assert_tensor_equals_cpu<float>(c, {std::tan(0.0f), std::tan(1.0f),
                                        std::tan(2.0f), std::tan(3.0f),
                                        std::tan(4.0f), std::tan(5.0f)});
}

// ==================================
//
//      MAIN
//
// ==================================

int main(int argc, char **argv) {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    RUN_TEST(test_negate, axiom::Device::CPU);
    RUN_TEST(test_abs, axiom::Device::CPU);
    RUN_TEST(test_sqrt, axiom::Device::CPU);
    RUN_TEST(test_exp, axiom::Device::CPU);
    RUN_TEST(test_log, axiom::Device::CPU);
    RUN_TEST(test_sin, axiom::Device::CPU);
    RUN_TEST(test_cos, axiom::Device::CPU);
    RUN_TEST(test_tan, axiom::Device::CPU);

    if (axiom::system::is_metal_available()) {
        std::cout << "\n--- Running tests on GPU ---\n" << std::endl;
        RUN_TEST(test_negate, axiom::Device::GPU);
        RUN_TEST(test_abs, axiom::Device::GPU);
        RUN_TEST(test_sqrt, axiom::Device::GPU);
        RUN_TEST(test_exp, axiom::Device::GPU);
        RUN_TEST(test_log, axiom::Device::GPU);
        RUN_TEST(test_sin, axiom::Device::GPU);
        RUN_TEST(test_cos, axiom::Device::GPU);
        RUN_TEST(test_tan, axiom::Device::GPU);
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