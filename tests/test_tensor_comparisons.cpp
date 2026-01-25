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
            throw std::runtime_error("Tensor data mismatch at index " +
                                     std::to_string(i));
        }
    }
}

// Test equal
void test_equal() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::equal(a, b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    std::vector<bool> expected(6, true);
    assert_tensor_equals_cpu<bool>(c, expected);
}

void test_equal_with_operator() {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto c = (a == b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

// Test not_equal
void test_not_equal() {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({4}, 2.0f);
    auto c = axiom::ops::not_equal(a, b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    // [0, 1, 2, 3] != 2 -> [true, true, false, true]
    assert_tensor_equals_cpu<bool>(c, {true, true, false, true});
}

void test_not_equal_with_operator() {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({4}, 2.0f);
    auto c = (a != b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

// Test less
void test_less() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::less(a, b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    // [0, 1, 2, 3, 4] < 2 -> [true, true, false, false, false]
    assert_tensor_equals_cpu<bool>(c, {true, true, false, false, false});
}

void test_less_with_operator() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a < b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

// Test less_equal
void test_less_equal() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::less_equal(a, b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    // [0, 1, 2, 3, 4] <= 2 -> [true, true, true, false, false]
    assert_tensor_equals_cpu<bool>(c, {true, true, true, false, false});
}

void test_less_equal_with_operator() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a <= b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

// Test greater
void test_greater() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::greater(a, b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    // [0, 1, 2, 3, 4] > 2 -> [false, false, false, true, true]
    assert_tensor_equals_cpu<bool>(c, {false, false, false, true, true});
}

void test_greater_with_operator() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a > b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

// Test greater_equal
void test_greater_equal() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = axiom::ops::greater_equal(a, b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    // [0, 1, 2, 3, 4] >= 2 -> [false, false, true, true, true]
    assert_tensor_equals_cpu<bool>(c, {false, false, true, true, true});
}

void test_greater_equal_with_operator() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({5}, 2.0f);
    auto c = (a >= b);

    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

// Test comparison with broadcasting
void test_comparison_broadcasting() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::full({}, 2.0f); // Scalar
    auto c = axiom::ops::greater(a, b);

    ASSERT(c.shape() == axiom::Shape({2, 3}), "Shape should match");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

// ============================================================================
// GPU Tests
// ============================================================================

void test_equal_gpu() {
    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .gpu();
    auto b = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .gpu();
    auto c = axiom::ops::equal(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Result should be on GPU");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    std::vector<bool> expected(6, true);
    assert_tensor_equals_cpu<bool>(c, expected);
}

void test_not_equal_gpu() {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({4}, 2.0f).gpu();
    auto c = axiom::ops::not_equal(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Result should be on GPU");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    assert_tensor_equals_cpu<bool>(c, {true, true, false, true});
}

void test_less_gpu() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::less(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Result should be on GPU");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    assert_tensor_equals_cpu<bool>(c, {true, true, false, false, false});
}

void test_less_equal_gpu() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::less_equal(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Result should be on GPU");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    assert_tensor_equals_cpu<bool>(c, {true, true, true, false, false});
}

void test_greater_gpu() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::greater(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Result should be on GPU");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    assert_tensor_equals_cpu<bool>(c, {false, false, false, true, true});
}

void test_greater_equal_gpu() {
    auto a = axiom::Tensor::arange(5).astype(axiom::DType::Float32).gpu();
    auto b = axiom::Tensor::full({5}, 2.0f).gpu();
    auto c = axiom::ops::greater_equal(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Result should be on GPU");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
    assert_tensor_equals_cpu<bool>(c, {false, false, true, true, true});
}

void test_comparison_broadcasting_gpu() {
    // Shape (3, 1) vs (4) should broadcast to (3, 4)
    auto a = axiom::Tensor::arange(3)
                 .reshape({3, 1})
                 .astype(axiom::DType::Float32)
                 .gpu();
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32).gpu();
    auto c = axiom::ops::less(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Result should be on GPU");
    ASSERT(c.shape() == axiom::Shape({3, 4}), "Broadcasting failed");
    ASSERT(c.dtype() == axiom::DType::Bool, "Result should be bool");
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Comparison Operations Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_equal);
    RUN_TEST(test_equal_with_operator);
    RUN_TEST(test_not_equal);
    RUN_TEST(test_not_equal_with_operator);
    RUN_TEST(test_less);
    RUN_TEST(test_less_with_operator);
    RUN_TEST(test_less_equal);
    RUN_TEST(test_less_equal_with_operator);
    RUN_TEST(test_greater);
    RUN_TEST(test_greater_with_operator);
    RUN_TEST(test_greater_equal);
    RUN_TEST(test_greater_equal_with_operator);
    RUN_TEST(test_comparison_broadcasting);

    // GPU tests (if Metal is available)
    if (axiom::system::should_run_gpu_tests()) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running GPU Tests" << std::endl;
        std::cout << "========================================\n" << std::endl;

        RUN_TEST(test_equal_gpu);
        RUN_TEST(test_not_equal_gpu);
        RUN_TEST(test_less_gpu);
        RUN_TEST(test_less_equal_gpu);
        RUN_TEST(test_greater_gpu);
        RUN_TEST(test_greater_equal_gpu);
        RUN_TEST(test_comparison_broadcasting_gpu);
    } else {
        std::cout << "\nSkipping GPU tests (Metal not available)" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
