#include <axiom/axiom.hpp>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
void assert_tensor_equals_cpu(const axiom::Tensor &t,
                              const std::vector<T> &expected_data) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");

    const T *t_data = t_cpu.template typed_data<T>();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if (t_data[i] != expected_data[i]) {
            throw std::runtime_error(
                "Tensor data mismatch at index " + std::to_string(i) +
                " expected: " + std::to_string(expected_data[i]) +
                " got: " + std::to_string(t_data[i]));
        }
    }
}

// Test argmax on 1D tensor
void test_argmax_1d() {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = axiom::ops::argmax(t);

    ASSERT(result.dtype() == axiom::DType::Int64, "Result should be Int64");
    ASSERT(result.ndim() == 0, "Should be scalar");
    ASSERT(result.item<int64_t>({}) == 3, "Max is at index 3");
}

// Test argmin on 1D tensor
void test_argmin_1d() {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, -2.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = axiom::ops::argmin(t);

    ASSERT(result.dtype() == axiom::DType::Int64, "Result should be Int64");
    ASSERT(result.ndim() == 0, "Should be scalar");
    ASSERT(result.item<int64_t>({}) == 3, "Min is at index 3");
}

// Test argmax along axis
void test_argmax_axis0() {
    // [[1, 5], [3, 2]] -> argmax(axis=0) -> [1, 0]
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmax(t, 0);

    ASSERT(result.shape() == axiom::Shape({2}), "Shape mismatch");
    assert_tensor_equals_cpu<int64_t>(result, {1, 0});
}

// Test argmax along axis 1
void test_argmax_axis1() {
    // [[1, 5], [3, 2]] -> argmax(axis=1) -> [1, 0]
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmax(t, 1);

    ASSERT(result.shape() == axiom::Shape({2}), "Shape mismatch");
    assert_tensor_equals_cpu<int64_t>(result, {1, 0});
}

// Test argmin along axis
void test_argmin_axis0() {
    // [[1, 5], [3, 2]] -> argmin(axis=0) -> [0, 1]
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmin(t, 0);

    ASSERT(result.shape() == axiom::Shape({2}), "Shape mismatch");
    assert_tensor_equals_cpu<int64_t>(result, {0, 1});
}

// Test argmax with keep_dims
void test_argmax_keep_dims() {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmax(t, 1, true);

    ASSERT(result.shape() == axiom::Shape({2, 1}),
           "Shape should have kept dim");
}

// Test argmin with keep_dims
void test_argmin_keep_dims() {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 2});
    auto result = axiom::ops::argmin(t, 0, true);

    ASSERT(result.shape() == axiom::Shape({1, 2}),
           "Shape should have kept dim");
}

// Test argmax member function
void test_argmax_member() {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = t.argmax();

    ASSERT(result.item<int64_t>({}) == 3, "Max is at index 3");
}

// Test argmin member function
void test_argmin_member() {
    std::vector<float> data = {1.0f, 5.0f, -3.0f, 9.0f, 2.0f};
    auto t = axiom::Tensor::from_data(data.data(), {5});
    auto result = t.argmin();

    ASSERT(result.item<int64_t>({}) == 2, "Min is at index 2");
}

// Test argmax on 3D tensor
void test_argmax_3d() {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4}).astype(
        axiom::DType::Float32);
    auto result = axiom::ops::argmax(t, 2); // Along last axis

    ASSERT(result.shape() == axiom::Shape({2, 3}), "Shape mismatch");
    // Each row of 4 elements should have max at index 3
    std::vector<int64_t> expected(6, 3);
    assert_tensor_equals_cpu<int64_t>(result, expected);
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Argmax/Argmin Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_argmax_1d);
    RUN_TEST(test_argmin_1d);
    RUN_TEST(test_argmax_axis0);
    RUN_TEST(test_argmax_axis1);
    RUN_TEST(test_argmin_axis0);
    RUN_TEST(test_argmax_keep_dims);
    RUN_TEST(test_argmin_keep_dims);
    RUN_TEST(test_argmax_member);
    RUN_TEST(test_argmin_member);
    RUN_TEST(test_argmax_3d);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
