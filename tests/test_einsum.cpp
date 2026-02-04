// Tests for einsum (Einstein summation) operations

#include <axiom/axiom.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

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

void test_einsum_matmul() {
    auto A = axiom::Tensor::randn({2, 3});
    auto B = axiom::Tensor::randn({3, 4});

    auto result = axiom::einops::einsum("ij,jk->ik", {A, B});
    auto expected = axiom::ops::matmul(A, B);

    ASSERT(result.shape() == expected.shape(), "einsum matmul shape mismatch");
    ASSERT(result.allclose(expected, 1e-4, 1e-4),
           "einsum matmul values mismatch");
}

void test_einsum_matmul_larger() {
    auto A = axiom::Tensor::randn({10, 20});
    auto B = axiom::Tensor::randn({20, 15});

    auto result = axiom::einops::einsum("ij,jk->ik", {A, B});
    auto expected = axiom::ops::matmul(A, B);

    ASSERT(result.shape() == expected.shape(),
           "einsum large matmul shape mismatch");
    ASSERT(result.allclose(expected, 1e-4, 1e-4),
           "einsum large matmul values mismatch");
}

void test_einsum_transpose() {
    auto A = axiom::Tensor::randn({3, 4});

    auto result = axiom::einops::einsum("ij->ji", {A});
    auto expected = A.transpose();

    ASSERT(result.shape() == expected.shape(),
           "einsum transpose shape mismatch");
    ASSERT(result.allclose(expected), "einsum transpose values mismatch");
}

void test_einsum_trace() {
    auto A = axiom::Tensor::randn({4, 4});

    auto result = axiom::einops::einsum("ii->", {A});
    auto expected = A.trace();

    ASSERT(result.shape() == expected.shape(), "einsum trace shape mismatch");
    ASSERT(result.allclose(expected, 1e-4, 1e-4),
           "einsum trace values mismatch");
}

void test_einsum_sum_all() {
    auto A = axiom::Tensor::randn({3, 4});

    auto result = axiom::einops::einsum("ij->", {A});
    auto expected = axiom::ops::sum(A);

    ASSERT(result.shape() == expected.shape(), "einsum sum all shape mismatch");
    ASSERT(result.allclose(expected, 1e-4, 1e-4),
           "einsum sum all values mismatch");
}

void test_einsum_sum_axis() {
    auto A = axiom::Tensor::randn({2, 3, 4});

    auto result = axiom::einops::einsum("ijk->j", {A});
    auto expected = axiom::ops::sum(A, {0, 2});

    ASSERT(result.shape() == expected.shape(),
           "einsum sum axis shape mismatch");
    ASSERT(result.allclose(expected, 1e-4, 1e-4),
           "einsum sum axis values mismatch");
}

void test_einsum_sum_keepdim() {
    auto A = axiom::Tensor::randn({3, 4});

    // Sum over second axis, keep first
    auto result = axiom::einops::einsum("ij->i", {A});
    auto expected = axiom::ops::sum(A, {1});

    ASSERT(result.shape() == expected.shape(),
           "einsum sum keepdim shape mismatch");
    ASSERT(result.allclose(expected, 1e-4, 1e-4),
           "einsum sum keepdim values mismatch");
}

void test_einsum_elementwise() {
    auto A = axiom::Tensor::randn({3, 4});
    auto B = axiom::Tensor::randn({3, 4});

    auto result = axiom::einops::einsum("ij,ij->ij", {A, B});
    auto expected = axiom::ops::multiply(A, B);

    ASSERT(result.shape() == expected.shape(),
           "einsum elementwise shape mismatch");
    ASSERT(result.allclose(expected), "einsum elementwise values mismatch");
}

void test_einsum_batched_matmul() {
    auto A = axiom::Tensor::randn({2, 3, 4});
    auto B = axiom::Tensor::randn({2, 4, 5});

    auto result = axiom::einops::einsum("bij,bjk->bik", {A, B});

    ASSERT(result.shape() == axiom::Shape({2, 3, 5}),
           "einsum batched matmul shape mismatch");

    // Verify by computing batch elements separately
    for (size_t b = 0; b < 2; ++b) {
        auto A_b = A.slice({{b, b + 1}}).squeeze(0);
        auto B_b = B.slice({{b, b + 1}}).squeeze(0);
        auto expected_b = axiom::ops::matmul(A_b, B_b);

        auto result_b = result.slice({{b, b + 1}}).squeeze(0);
        ASSERT(result_b.allclose(expected_b, 1e-4, 1e-4),
               "einsum batched matmul batch element mismatch");
    }
}

void test_einsum_dot_product() {
    auto a = axiom::Tensor::randn({5});
    auto b = axiom::Tensor::randn({5});

    // Dot product: i,i->
    auto result = axiom::einops::einsum("i,i->", {a, b});
    auto expected = axiom::ops::sum(axiom::ops::multiply(a, b));

    ASSERT(result.shape() == expected.shape(),
           "einsum dot product shape mismatch");
    ASSERT(result.allclose(expected, 1e-4, 1e-4),
           "einsum dot product value mismatch");
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Einsum Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_einsum_matmul);
    RUN_TEST(test_einsum_matmul_larger);
    RUN_TEST(test_einsum_transpose);
    RUN_TEST(test_einsum_trace);
    RUN_TEST(test_einsum_sum_all);
    RUN_TEST(test_einsum_sum_axis);
    RUN_TEST(test_einsum_sum_keepdim);
    RUN_TEST(test_einsum_elementwise);
    RUN_TEST(test_einsum_batched_matmul);
    RUN_TEST(test_einsum_dot_product);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
