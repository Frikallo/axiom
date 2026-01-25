#include <axiom/axiom.hpp>
#include <axiom/tensor_operators.hpp>
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
                        T tol = T(1e-5)) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.size() == expected.size(), "Size mismatch");
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
// Gather Tests
// ============================================================================

void test_gather_1d() {
    // Simple 1D gather
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5});

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3});

    auto result = x.gather(0, indices);

    ASSERT(result.shape() == Shape({3}), "Shape mismatch");
    assert_tensor_near<float>(result, {1.0f, 3.0f, 5.0f});
}

void test_gather_2d_dim0() {
    // 2D gather along dim 0 (select rows)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {3, 2});

    // Select rows 0 and 2
    std::vector<int64_t> indices_data = {0, 0, 2, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2, 2});

    auto result = x.gather(0, indices);

    ASSERT(result.shape() == Shape({2, 2}), "Shape mismatch");
    // Row 0 is [1, 2], Row 2 is [5, 6]
    assert_tensor_near<float>(result, {1.0f, 2.0f, 5.0f, 6.0f});
}

void test_gather_2d_dim1() {
    // 2D gather along dim 1 (select columns)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    // For each row, gather specific columns
    std::vector<int64_t> indices_data = {0, 2, 1, 0};
    auto indices = Tensor::from_data(indices_data.data(), {2, 2});

    auto result = x.gather(1, indices);

    ASSERT(result.shape() == Shape({2, 2}), "Shape mismatch");
    // Row 0: [x[0,0], x[0,2]] = [1, 3]
    // Row 1: [x[1,1], x[1,0]] = [5, 4]
    assert_tensor_near<float>(result, {1.0f, 3.0f, 5.0f, 4.0f});
}

// ============================================================================
// Index Select Tests
// ============================================================================

void test_index_select_dim0() {
    // Select rows by index
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {3, 2});

    std::vector<int64_t> indices_data = {0, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2});

    auto result = x.index_select(0, indices);

    ASSERT(result.shape() == Shape({2, 2}), "Shape mismatch");
    assert_tensor_near<float>(result, {1.0f, 2.0f, 5.0f, 6.0f});
}

void test_index_select_dim1() {
    // Select columns by index
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {2, 3});

    std::vector<int64_t> indices_data = {0, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2});

    auto result = x.index_select(1, indices);

    ASSERT(result.shape() == Shape({2, 2}), "Shape mismatch");
    // Select columns 0 and 2: [[1, 3], [4, 6]]
    assert_tensor_near<float>(result, {1.0f, 3.0f, 4.0f, 6.0f});
}

// ============================================================================
// Scatter Tests
// ============================================================================

void test_scatter_1d() {
    // Simple 1D scatter
    auto x = Tensor::zeros({5});

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3});

    std::vector<float> src_data = {1.0f, 2.0f, 3.0f};
    auto src = Tensor::from_data(src_data.data(), {3});

    auto result = x.scatter(0, indices, src);

    ASSERT(result.shape() == Shape({5}), "Shape mismatch");
    assert_tensor_near<float>(result, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f});
}

void test_scatter_2d_dim0() {
    // 2D scatter along dim 0
    auto x = Tensor::zeros({3, 2});

    std::vector<int64_t> indices_data = {0, 0, 2, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2, 2});

    std::vector<float> src_data = {1.0f, 2.0f, 5.0f, 6.0f};
    auto src = Tensor::from_data(src_data.data(), {2, 2});

    auto result = x.scatter(0, indices, src);

    ASSERT(result.shape() == Shape({3, 2}), "Shape mismatch");
    // Row 0 gets [1, 2], Row 2 gets [5, 6]
    assert_tensor_near<float>(result, {1.0f, 2.0f, 0.0f, 0.0f, 5.0f, 6.0f});
}

// ============================================================================
// GPU Tests
// ============================================================================

void test_gather_gpu() {
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x = Tensor::from_data(x_data.data(), {5}).gpu();

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3}).gpu();

    auto result = x.gather(0, indices);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    assert_tensor_near<float>(result, {1.0f, 3.0f, 5.0f});
}

void test_index_select_gpu() {
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor::from_data(x_data.data(), {3, 2}).gpu();

    std::vector<int64_t> indices_data = {0, 2};
    auto indices = Tensor::from_data(indices_data.data(), {2}).gpu();

    auto result = x.index_select(0, indices);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    assert_tensor_near<float>(result, {1.0f, 2.0f, 5.0f, 6.0f});
}

void test_scatter_gpu() {
    auto x = Tensor::zeros({5}).gpu();

    std::vector<int64_t> indices_data = {0, 2, 4};
    auto indices = Tensor::from_data(indices_data.data(), {3}).gpu();

    std::vector<float> src_data = {1.0f, 2.0f, 3.0f};
    auto src = Tensor::from_data(src_data.data(), {3}).gpu();

    auto result = x.scatter(0, indices, src);

    ASSERT(result.device() == Device::GPU, "Result should be on GPU");
    assert_tensor_near<float>(result, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f});
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Gather/Scatter/IndexSelect Tests ===" << std::endl
              << std::endl;

    // CPU Gather tests
    std::cout << "--- CPU Gather Tests ---" << std::endl;
    RUN_TEST(test_gather_1d);
    RUN_TEST(test_gather_2d_dim0);
    RUN_TEST(test_gather_2d_dim1);

    // CPU Index Select tests
    std::cout << "--- CPU Index Select Tests ---" << std::endl;
    RUN_TEST(test_index_select_dim0);
    RUN_TEST(test_index_select_dim1);

    // CPU Scatter tests
    std::cout << "--- CPU Scatter Tests ---" << std::endl;
    RUN_TEST(test_scatter_1d);
    RUN_TEST(test_scatter_2d_dim0);

    // GPU tests
    std::cout << "--- GPU Tests ---" << std::endl;
    RUN_TEST(test_gather_gpu);
    RUN_TEST(test_index_select_gpu);
    RUN_TEST(test_scatter_gpu);

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
