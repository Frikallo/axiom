// Tests for advanced indexing operations (Phase 4)

#include <axiom/axiom.hpp>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

void test_take_1d() {
    auto t = axiom::Tensor::arange(10);
    auto indices = axiom::Tensor::from_data(
        std::vector<int64_t>{0, 2, 4}.data(), {3}, true);

    auto result = axiom::ops::take(t, indices);
    ASSERT(result.shape() == axiom::Shape({3}), "Take result shape wrong");
    ASSERT(result.item<int32_t>({0}) == 0, "Take value 0 wrong");
    ASSERT(result.item<int32_t>({1}) == 2, "Take value 1 wrong");
    ASSERT(result.item<int32_t>({2}) == 4, "Take value 2 wrong");
}

void test_take_2d() {
    auto t = axiom::Tensor::from_data(
        std::vector<float>{1, 2, 3, 4, 5, 6}.data(), {2, 3}, true);
    auto indices =
        axiom::Tensor::from_data(std::vector<int64_t>{0, 1}.data(), {2}, true);

    // Take along axis 0 (rows)
    auto result = axiom::ops::take(t, indices, 0);
    ASSERT(result.shape() == axiom::Shape({2, 3}), "Take 2D shape wrong");
}

void test_take_negative_indices() {
    auto t = axiom::Tensor::arange(5);
    auto indices = axiom::Tensor::from_data(std::vector<int64_t>{-1, -2}.data(),
                                            {2}, true);

    auto result = axiom::ops::take(t, indices);
    ASSERT(result.item<int32_t>({0}) == 4, "Negative index -1 wrong");
    ASSERT(result.item<int32_t>({1}) == 3, "Negative index -2 wrong");
}

void test_take_along_axis() {
    auto t = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}.data(), {2, 3},
        true);
    auto indices = axiom::Tensor::from_data(
        std::vector<int64_t>{0, 2, 1, 1, 0, 2}.data(), {2, 3}, true);

    auto result = axiom::ops::take_along_axis(t, indices, 1);
    ASSERT(result.shape() == axiom::Shape({2, 3}),
           "take_along_axis shape wrong");

    // Row 0: indices [0, 2, 1] -> values [1, 3, 2]
    ASSERT(result.item<float>({0, 0}) == 1.0f, "take_along_axis [0,0] wrong");
    ASSERT(result.item<float>({0, 1}) == 3.0f, "take_along_axis [0,1] wrong");
    ASSERT(result.item<float>({0, 2}) == 2.0f, "take_along_axis [0,2] wrong");
}

void test_take_along_axis_0() {
    auto t = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}.data(), {2, 3},
        true);
    auto indices = axiom::Tensor::from_data(
        std::vector<int64_t>{1, 0, 1}.data(), {1, 3}, true);

    auto result = axiom::ops::take_along_axis(t, indices, 0);
    ASSERT(result.shape() == axiom::Shape({1, 3}),
           "take_along_axis axis=0 shape wrong");

    // Column 0: index 1 -> 4, Column 1: index 0 -> 2, Column 2: index 1 -> 6
    ASSERT(result.item<float>({0, 0}) == 4.0f,
           "take_along_axis axis=0 [0,0] wrong");
    ASSERT(result.item<float>({0, 1}) == 2.0f,
           "take_along_axis axis=0 [0,1] wrong");
    ASSERT(result.item<float>({0, 2}) == 6.0f,
           "take_along_axis axis=0 [0,2] wrong");
}

void test_put_along_axis() {
    auto t = axiom::Tensor::zeros({3, 3});
    auto indices = axiom::Tensor::from_data(
        std::vector<int64_t>{0, 1, 2}.data(), {3, 1}, true);
    auto values = axiom::Tensor::ones({3, 1});

    auto result = axiom::ops::put_along_axis(t, indices, values, 1);
    ASSERT(result.shape() == axiom::Shape({3, 3}),
           "put_along_axis shape wrong");

    // Diagonal should be 1
    ASSERT(result.item<float>({0, 0}) == 1.0f, "Diagonal 0,0 should be 1");
    ASSERT(result.item<float>({1, 1}) == 1.0f, "Diagonal 1,1 should be 1");
    ASSERT(result.item<float>({2, 2}) == 1.0f, "Diagonal 2,2 should be 1");

    // Off-diagonal should be 0
    ASSERT(result.item<float>({0, 1}) == 0.0f, "Off-diagonal 0,1 should be 0");
    ASSERT(result.item<float>({1, 0}) == 0.0f, "Off-diagonal 1,0 should be 0");
}

void test_put_along_axis_multiple() {
    auto t = axiom::Tensor::zeros({2, 4});
    auto indices = axiom::Tensor::from_data(
        std::vector<int64_t>{0, 2, 1, 3}.data(), {2, 2}, true);
    auto values = axiom::Tensor::from_data(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}.data(), {2, 2}, true);

    auto result = axiom::ops::put_along_axis(t, indices, values, 1);
    ASSERT(result.shape() == axiom::Shape({2, 4}),
           "put_along_axis multiple shape wrong");

    // Row 0: put 1 at index 0, 2 at index 2
    ASSERT(result.item<float>({0, 0}) == 1.0f, "Row 0 index 0 wrong");
    ASSERT(result.item<float>({0, 2}) == 2.0f, "Row 0 index 2 wrong");

    // Row 1: put 3 at index 1, 4 at index 3
    ASSERT(result.item<float>({1, 1}) == 3.0f, "Row 1 index 1 wrong");
    ASSERT(result.item<float>({1, 3}) == 4.0f, "Row 1 index 3 wrong");
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Advanced Indexing Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_take_1d);
    RUN_TEST(test_take_2d);
    RUN_TEST(test_take_negative_indices);
    RUN_TEST(test_take_along_axis);
    RUN_TEST(test_take_along_axis_0);
    RUN_TEST(test_put_along_axis);
    RUN_TEST(test_put_along_axis_multiple);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
