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

template <typename T>
void assert_tensor_equals(const Tensor &t, const std::vector<T> &expected) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.size() == expected.size(), "Size mismatch");
    const T *data = t_cpu.typed_data<T>();
    for (size_t i = 0; i < expected.size(); ++i) {
        if (data[i] != expected[i]) {
            throw std::runtime_error(
                "Value mismatch at index " + std::to_string(i) + ": got " +
                std::to_string(static_cast<int>(data[i])) + ", expected " +
                std::to_string(static_cast<int>(expected[i])));
        }
    }
}

// ==================================
//      LOGICAL OPERATION TESTS
// ==================================

void test_logical_and_basic() {
    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = ops::logical_and(a, b);

    ASSERT(result.dtype() == DType::Bool, "Should return Bool");
    assert_tensor_equals<bool>(result, {true, false, false, false});
}

void test_logical_or_basic() {
    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = ops::logical_or(a, b);

    ASSERT(result.dtype() == DType::Bool, "Should return Bool");
    assert_tensor_equals<bool>(result, {true, true, true, false});
}

void test_logical_xor_basic() {
    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});
    auto b = Tensor::from_data(b_data, {4});

    auto result = ops::logical_xor(a, b);

    ASSERT(result.dtype() == DType::Bool, "Should return Bool");
    assert_tensor_equals<bool>(result, {false, true, true, false});
}

void test_logical_not_basic() {
    bool a_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4});

    auto result = ops::logical_not(a);

    ASSERT(result.dtype() == DType::Bool, "Should return Bool");
    assert_tensor_equals<bool>(result, {false, true, false, true});
}

void test_logical_not_from_float() {
    // Non-zero values should be treated as true
    std::vector<float> a_data = {1.0f, 0.0f, -3.5f, 0.0f};
    auto a = Tensor::from_data(a_data.data(), {4});

    auto result = ops::logical_not(a);

    ASSERT(result.dtype() == DType::Bool, "Should return Bool");
    // 1.0 -> true -> false, 0.0 -> false -> true, etc.
    assert_tensor_equals<bool>(result, {false, true, false, true});
}

void test_logical_not_from_int() {
    std::vector<int32_t> a_data = {1, 0, -5, 0};
    auto a = Tensor::from_data(a_data.data(), {4});

    auto result = ops::logical_not(a);

    ASSERT(result.dtype() == DType::Bool, "Should return Bool");
    assert_tensor_equals<bool>(result, {false, true, false, true});
}

void test_logical_and_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    bool a_data[] = {true, true, false, false};
    bool b_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4}).gpu();
    auto b = Tensor::from_data(b_data, {4}).gpu();

    auto result = ops::logical_and(a, b);

    ASSERT(result.device() == Device::GPU, "Should be on GPU");
    assert_tensor_equals<bool>(result, {true, false, false, false});
}

void test_logical_not_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    bool a_data[] = {true, false, true, false};
    auto a = Tensor::from_data(a_data, {4}).gpu();

    auto result = ops::logical_not(a);

    ASSERT(result.device() == Device::GPU, "Should be on GPU");
    assert_tensor_equals<bool>(result, {false, true, false, true});
}

// ==================================
//      BITWISE OPERATION TESTS
// ==================================

void test_bitwise_and_basic() {
    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b1010, 0b1111, 0b0000};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_and(a, b);

    ASSERT(result.dtype() == DType::Int32, "Should return Int32");
    assert_tensor_equals<int32_t>(result, {0b1010, 0b1010, 0b0000, 0b0000});
}

void test_bitwise_or_basic() {
    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b0101, 0b1111, 0b0000};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_or(a, b);

    ASSERT(result.dtype() == DType::Int32, "Should return Int32");
    assert_tensor_equals<int32_t>(result, {0b1111, 0b1111, 0b1111, 0b1111});
}

void test_bitwise_xor_basic() {
    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b1010, 0b1111, 0b1111};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_xor(a, b);

    ASSERT(result.dtype() == DType::Int32, "Should return Int32");
    assert_tensor_equals<int32_t>(result, {0b0101, 0b0000, 0b1111, 0b0000});
}

void test_left_shift_basic() {
    std::vector<int32_t> a_data = {1, 2, 4, 8};
    std::vector<int32_t> b_data = {1, 2, 1, 0};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::left_shift(a, b);

    ASSERT(result.dtype() == DType::Int32, "Should return Int32");
    // 1 << 1 = 2, 2 << 2 = 8, 4 << 1 = 8, 8 << 0 = 8
    assert_tensor_equals<int32_t>(result, {2, 8, 8, 8});
}

void test_right_shift_basic() {
    std::vector<int32_t> a_data = {8, 16, 4, 1};
    std::vector<int32_t> b_data = {1, 2, 1, 0};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::right_shift(a, b);

    ASSERT(result.dtype() == DType::Int32, "Should return Int32");
    // 8 >> 1 = 4, 16 >> 2 = 4, 4 >> 1 = 2, 1 >> 0 = 1
    assert_tensor_equals<int32_t>(result, {4, 4, 2, 1});
}

void test_bitwise_and_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    std::vector<int32_t> a_data = {0b1111, 0b1010, 0b0000, 0b1111};
    std::vector<int32_t> b_data = {0b1010, 0b1010, 0b1111, 0b0000};
    auto a = Tensor::from_data(a_data.data(), {4}).gpu();
    auto b = Tensor::from_data(b_data.data(), {4}).gpu();

    auto result = ops::bitwise_and(a, b);

    ASSERT(result.device() == Device::GPU, "Should be on GPU");
    assert_tensor_equals<int32_t>(result, {0b1010, 0b1010, 0b0000, 0b0000});
}

void test_bitwise_with_uint8() {
    std::vector<uint8_t> a_data = {0xFF, 0xAA, 0x00, 0x55};
    std::vector<uint8_t> b_data = {0xAA, 0xAA, 0xFF, 0x55};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::bitwise_and(a, b);

    ASSERT(result.dtype() == DType::UInt8, "Should return UInt8");
    assert_tensor_equals<uint8_t>(result, {0xAA, 0xAA, 0x00, 0x55});
}

// ==================================
//      MATH OPERATION TESTS
// ==================================

void test_maximum_basic() {
    std::vector<float> a_data = {1.0f, 5.0f, 3.0f, 0.0f};
    std::vector<float> b_data = {2.0f, 4.0f, 3.0f, -1.0f};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::maximum(a, b);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    ASSERT(std::abs(data[0] - 2.0f) < 1e-5f, "max(1,2)=2");
    ASSERT(std::abs(data[1] - 5.0f) < 1e-5f, "max(5,4)=5");
    ASSERT(std::abs(data[2] - 3.0f) < 1e-5f, "max(3,3)=3");
    ASSERT(std::abs(data[3] - 0.0f) < 1e-5f, "max(0,-1)=0");
}

void test_minimum_basic() {
    std::vector<float> a_data = {1.0f, 5.0f, 3.0f, 0.0f};
    std::vector<float> b_data = {2.0f, 4.0f, 3.0f, -1.0f};
    auto a = Tensor::from_data(a_data.data(), {4});
    auto b = Tensor::from_data(b_data.data(), {4});

    auto result = ops::minimum(a, b);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    ASSERT(std::abs(data[0] - 1.0f) < 1e-5f, "min(1,2)=1");
    ASSERT(std::abs(data[1] - 4.0f) < 1e-5f, "min(5,4)=4");
    ASSERT(std::abs(data[2] - 3.0f) < 1e-5f, "min(3,3)=3");
    ASSERT(std::abs(data[3] - (-1.0f)) < 1e-5f, "min(0,-1)=-1");
}

void test_atan2_basic() {
    std::vector<float> y_data = {1.0f, 1.0f, 0.0f};
    std::vector<float> x_data = {1.0f, 0.0f, 1.0f};
    auto y = Tensor::from_data(y_data.data(), {3});
    auto x = Tensor::from_data(x_data.data(), {3});

    auto result = ops::atan2(y, x);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    // atan2(1,1) = pi/4, atan2(1,0) = pi/2, atan2(0,1) = 0
    ASSERT(std::abs(data[0] - static_cast<float>(M_PI / 4)) < 1e-5f,
           "atan2(1,1)=pi/4");
    ASSERT(std::abs(data[1] - static_cast<float>(M_PI / 2)) < 1e-5f,
           "atan2(1,0)=pi/2");
    ASSERT(std::abs(data[2] - 0.0f) < 1e-5f, "atan2(0,1)=0");
}

void test_hypot_basic() {
    std::vector<float> a_data = {3.0f, 0.0f, 5.0f};
    std::vector<float> b_data = {4.0f, 5.0f, 12.0f};
    auto a = Tensor::from_data(a_data.data(), {3});
    auto b = Tensor::from_data(b_data.data(), {3});

    auto result = ops::hypot(a, b);

    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    // hypot(3,4)=5, hypot(0,5)=5, hypot(5,12)=13
    ASSERT(std::abs(data[0] - 5.0f) < 1e-5f, "hypot(3,4)=5");
    ASSERT(std::abs(data[1] - 5.0f) < 1e-5f, "hypot(0,5)=5");
    ASSERT(std::abs(data[2] - 13.0f) < 1e-5f, "hypot(5,12)=13");
}

void test_hypot_gpu() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (Metal not available)" << std::endl;
        return;
    }

    std::vector<float> a_data = {3.0f, 0.0f, 5.0f};
    std::vector<float> b_data = {4.0f, 5.0f, 12.0f};
    auto a = Tensor::from_data(a_data.data(), {3}).gpu();
    auto b = Tensor::from_data(b_data.data(), {3}).gpu();

    auto result = ops::hypot(a, b);

    ASSERT(result.device() == Device::GPU, "Should be on GPU");
    auto result_cpu = result.cpu();
    const float *data = result_cpu.typed_data<float>();
    ASSERT(std::abs(data[0] - 5.0f) < 1e-4f, "hypot(3,4)=5");
    ASSERT(std::abs(data[1] - 5.0f) < 1e-4f, "hypot(0,5)=5");
    ASSERT(std::abs(data[2] - 13.0f) < 1e-4f, "hypot(5,12)=13");
}

// ==================================
//      MAIN
// ==================================

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Logical/Bitwise/Math Operations Tests ===" << std::endl
              << std::endl;

    // Logical operations
    std::cout << "--- Logical Operations ---" << std::endl;
    RUN_TEST(test_logical_and_basic);
    RUN_TEST(test_logical_or_basic);
    RUN_TEST(test_logical_xor_basic);
    RUN_TEST(test_logical_not_basic);
    RUN_TEST(test_logical_not_from_float);
    RUN_TEST(test_logical_not_from_int);
    RUN_TEST(test_logical_and_gpu);
    RUN_TEST(test_logical_not_gpu);

    // Bitwise operations
    std::cout << "--- Bitwise Operations ---" << std::endl;
    RUN_TEST(test_bitwise_and_basic);
    RUN_TEST(test_bitwise_or_basic);
    RUN_TEST(test_bitwise_xor_basic);
    RUN_TEST(test_left_shift_basic);
    RUN_TEST(test_right_shift_basic);
    RUN_TEST(test_bitwise_and_gpu);
    RUN_TEST(test_bitwise_with_uint8);

    // Math operations
    std::cout << "--- Math Operations ---" << std::endl;
    RUN_TEST(test_maximum_basic);
    RUN_TEST(test_minimum_basic);
    RUN_TEST(test_atan2_basic);
    RUN_TEST(test_hypot_basic);
    RUN_TEST(test_hypot_gpu);

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
