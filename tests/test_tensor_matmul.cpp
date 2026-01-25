#include <axiom/axiom.hpp>
#include <cmath>
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
                              const std::vector<T> &expected_data,
                              double epsilon = 1e-5) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");

    const T *t_data = t_cpu.template typed_data<T>();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(static_cast<double>(t_data[i]) -
                         static_cast<double>(expected_data[i])) >= epsilon) {
                throw std::runtime_error(
                    "Tensor data mismatch at index " + std::to_string(i) +
                    " expected: " + std::to_string(expected_data[i]) +
                    " got: " + std::to_string(t_data[i]));
            }
        } else {
            if (t_data[i] != expected_data[i]) {
                throw std::runtime_error("Tensor data mismatch at index " +
                                         std::to_string(i));
            }
        }
    }
}

// Test 2D x 2D matmul
void test_matmul_2d_2d() {
    // (2, 3) @ (3, 4) -> (2, 4)
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.shape() == axiom::Shape({2, 4}), "Shape mismatch");
    // Expected: [[20, 23, 26, 29], [56, 68, 80, 92]]
    assert_tensor_equals_cpu<float>(
        c, {20.0f, 23.0f, 26.0f, 29.0f, 56.0f, 68.0f, 80.0f, 92.0f});
}

// Test 1D (vector) x 1D (vector) -> dot product
void test_matmul_1d_1d() {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.ndim() == 0, "Should be scalar");
    ASSERT(c.item<float>({}) == 14.0f, "Dot product incorrect");
}

// Test 1D x 2D -> vector-matrix multiply
void test_matmul_1d_2d() {
    // (3,) @ (3, 4) -> (4,)
    auto a = axiom::Tensor::arange(3).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.shape() == axiom::Shape({4}), "Shape mismatch");
    // Expected: [20, 23, 26, 29]
    assert_tensor_equals_cpu<float>(c, {20.0f, 23.0f, 26.0f, 29.0f});
}

// Test 2D x 1D -> matrix-vector multiply
void test_matmul_2d_1d() {
    // (2, 3) @ (3,) -> (2,)
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(3).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.shape() == axiom::Shape({2}), "Shape mismatch");
    // Expected: [5, 14]
    assert_tensor_equals_cpu<float>(c, {5.0f, 14.0f});
}

// Test batched matmul with same batch dims
void test_matmul_batched() {
    // (2, 3, 4) @ (2, 4, 5) -> (2, 3, 5)
    auto a = axiom::Tensor::arange(24).reshape({2, 3, 4}).astype(
        axiom::DType::Float32);
    auto b = axiom::Tensor::arange(40).reshape({2, 4, 5}).astype(
        axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.shape() == axiom::Shape({2, 3, 5}), "Shape mismatch");
    ASSERT(c.size() == 30, "Size mismatch");
}

// Test batched matmul with broadcast
void test_matmul_broadcast() {
    // (2, 1, 3, 4) @ (4, 5) -> (2, 1, 3, 5)
    auto a = axiom::Tensor::arange(24)
                 .reshape({2, 1, 3, 4})
                 .astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(20).reshape({4, 5}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.shape() == axiom::Shape({2, 1, 3, 5}), "Shape mismatch");
}

// Test batched matmul with broadcast on both sides
void test_matmul_broadcast_both() {
    // (2, 1, 3, 4) @ (1, 4, 5) -> (2, 1, 3, 5)
    auto a = axiom::Tensor::arange(24)
                 .reshape({2, 1, 3, 4})
                 .astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(20).reshape({1, 4, 5}).astype(
        axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.shape() == axiom::Shape({2, 1, 3, 5}), "Shape mismatch");
}

// Test matmul with transpose flags
void test_matmul_transpose_a() {
    // (3, 2)^T @ (3, 4) -> (2, 4) where (3, 2)^T is logically (2, 3)
    auto a =
        axiom::Tensor::arange(6).reshape({3, 2}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b, true, false); // transpose_a=true

    ASSERT(c.shape() == axiom::Shape({2, 4}), "Shape mismatch");
}

// Test matmul with transpose_b
void test_matmul_transpose_b() {
    // (2, 3) @ (4, 3)^T -> (2, 4) where (4, 3)^T is logically (3, 4)
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({4, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b, false, true); // transpose_b=true

    ASSERT(c.shape() == axiom::Shape({2, 4}), "Shape mismatch");
}

// Test matmul with both transposes
void test_matmul_transpose_both() {
    // (3, 2)^T @ (4, 3)^T -> (2, 4)
    auto a =
        axiom::Tensor::arange(6).reshape({3, 2}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({4, 3}).astype(axiom::DType::Float32);
    auto c = axiom::ops::matmul(a, b, true, true);

    ASSERT(c.shape() == axiom::Shape({2, 4}), "Shape mismatch");
}

// Test member function matmul
void test_matmul_member() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = a.matmul(b);

    ASSERT(c.shape() == axiom::Shape({2, 4}), "Shape mismatch");
}

// Test mm alias
void test_mm_alias() {
    auto a =
        axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto b =
        axiom::Tensor::arange(12).reshape({3, 4}).astype(axiom::DType::Float32);
    auto c = a.mm(b);

    ASSERT(c.shape() == axiom::Shape({2, 4}), "Shape mismatch");
}

// Test dot alias for vectors
void test_dot_alias() {
    auto a = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto b = axiom::Tensor::arange(4).astype(axiom::DType::Float32);
    auto c = a.dot(b);

    ASSERT(c.ndim() == 0, "Should be scalar");
    ASSERT(c.item<float>({}) == 14.0f, "Dot product incorrect");
}

// Test GPU matmul if available
void test_matmul_gpu() {
    if (!axiom::system::should_run_gpu_tests()) {
        std::cout << "Skipping GPU test (Metal not available)" << std::endl;
        return;
    }

    auto a = axiom::Tensor::arange(6)
                 .reshape({2, 3})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto b = axiom::Tensor::arange(12)
                 .reshape({3, 4})
                 .astype(axiom::DType::Float32)
                 .to(axiom::Device::GPU);
    auto c = axiom::ops::matmul(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Should be on GPU");
    ASSERT(c.shape() == axiom::Shape({2, 4}), "Shape mismatch");

    // Verify results
    assert_tensor_equals_cpu<float>(
        c, {20.0f, 23.0f, 26.0f, 29.0f, 56.0f, 68.0f, 80.0f, 92.0f});
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Matrix Multiplication Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_matmul_2d_2d);
    RUN_TEST(test_matmul_1d_1d);
    RUN_TEST(test_matmul_1d_2d);
    RUN_TEST(test_matmul_2d_1d);
    RUN_TEST(test_matmul_batched);
    RUN_TEST(test_matmul_broadcast);
    RUN_TEST(test_matmul_broadcast_both);
    RUN_TEST(test_matmul_transpose_a);
    RUN_TEST(test_matmul_transpose_b);
    RUN_TEST(test_matmul_transpose_both);
    RUN_TEST(test_matmul_member);
    RUN_TEST(test_mm_alias);
    RUN_TEST(test_dot_alias);
    RUN_TEST(test_matmul_gpu);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
