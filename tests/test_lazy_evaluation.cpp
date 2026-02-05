#include "axiom/graph/graph_registry.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

#include <cmath>
#include <iostream>
#include <string>

static int tests_passed = 0;
static int tests_total = 0;

#define ASSERT(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            throw std::runtime_error(std::string("Assertion failed: ") + #cond \
                                     + " - " + (msg));                         \
        }                                                                      \
    } while (0)

#define RUN_TEST(test_func)                                                    \
    do {                                                                       \
        tests_total++;                                                         \
        std::cout << "--- Running: " << #test_func << " ---" << std::endl;     \
        try {                                                                  \
            test_func();                                                       \
            std::cout << "--- PASSED: " << #test_func << " ---\n" << std::endl;\
            tests_passed++;                                                    \
        } catch (const std::exception &e) {                                    \
            std::cout << "--- FAILED: " << #test_func << " ---" << std::endl;  \
            std::cout << "    Error: " << e.what() << "\n" << std::endl;       \
        }                                                                      \
    } while (0)

template <typename T>
bool approx_equal(T a, T b, T tol = static_cast<T>(1e-5)) {
    return std::abs(a - b) < tol;
}

using namespace axiom;

// Test that lazy tensors defer execution
void test_lazy_deferral() {
    auto a = Tensor::randn({100, 100});
    auto b = Tensor::randn({100, 100});

    // These operations should create lazy tensors
    auto c = ops::add(a, b);
    auto d = ops::multiply(c, c);

    // c and d should be lazy (if lazy evaluation is enabled)
    // Note: We can check shape without materializing
    ASSERT(d.shape() == Shape({100, 100}), "Shape should be inferred without execution");

    // Now access the data - this should trigger materialization
    float first_val = d.typed_data<float>()[0];
    (void)first_val;

    // After materialization, tensor should work normally
    ASSERT(d.size() == 10000, "Size should be correct after materialization");
}

// Test that lazy evaluation produces correct results for basic ops
void test_lazy_correctness_add() {
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    auto c = ops::add(a, b);

    // All elements should be 5.0
    float val = c.typed_data<float>()[0];
    ASSERT(approx_equal(val, 5.0f), "Add result should be 5.0");
}

void test_lazy_correctness_multiply() {
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    auto c = ops::multiply(a, b);

    float val = c.typed_data<float>()[0];
    ASSERT(approx_equal(val, 6.0f), "Multiply result should be 6.0");
}

void test_lazy_chain() {
    // Test a chain of operations
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    // (a + b) * a = (2 + 3) * 2 = 10
    auto c = ops::add(a, b);
    auto d = ops::multiply(c, a);

    float val = d.typed_data<float>()[0];
    ASSERT(approx_equal(val, 10.0f), "Chained result should be 10.0");
}

void test_lazy_unary() {
    auto a = Tensor::full<float>({3, 3}, 4.0f);

    auto b = ops::sqrt(a);

    float val = b.typed_data<float>()[0];
    ASSERT(approx_equal(val, 2.0f), "Sqrt of 4.0 should be 2.0");
}

void test_lazy_relu() {
    auto a = Tensor::full<float>({3, 3}, -2.0f);

    auto b = ops::relu(a);

    float val = b.typed_data<float>()[0];
    ASSERT(approx_equal(val, 0.0f), "ReLU of -2.0 should be 0.0");
}

void test_lazy_reduction() {
    // Create tensor with known values
    auto a = Tensor::full<float>({2, 3}, 2.0f);

    // Sum all elements: 2 * 3 * 2.0 = 12.0
    auto b = ops::sum(a);

    float val = b.item<float>();
    ASSERT(approx_equal(val, 12.0f), "Sum should be 12.0");
}

void test_lazy_matmul() {
    // Simple 2x2 matmul
    auto a = Tensor::full<float>({2, 2}, 1.0f);
    auto b = Tensor::full<float>({2, 2}, 2.0f);

    auto c = ops::matmul(a, b);

    // Each element should be 1*2 + 1*2 = 4.0
    ASSERT(c.shape() == Shape({2, 2}), "Matmul shape should be 2x2");
    float val = c.typed_data<float>()[0];
    ASSERT(approx_equal(val, 4.0f), "Matmul result should be 4.0");
}

void test_lazy_comparison() {
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    auto c = ops::less(a, b);

    ASSERT(c.dtype() == DType::Bool, "Comparison should return bool");
    bool val = c.typed_data<bool>()[0];
    ASSERT(val == true, "2 < 3 should be true");
}

void test_lazy_broadcast() {
    auto a = Tensor::full<float>({3, 1}, 2.0f);
    auto b = Tensor::full<float>({1, 4}, 3.0f);

    auto c = ops::add(a, b);

    ASSERT(c.shape() == Shape({3, 4}), "Broadcast shape should be 3x4");
    float val = c.typed_data<float>()[0];
    ASSERT(approx_equal(val, 5.0f), "Broadcast add should be 5.0");
}

void test_eager_mode_env_var() {
    // Test that AXIOM_EAGER_MODE environment variable is respected
    // This test just verifies the function exists
    bool eager = graph::is_eager_mode_enabled();
    (void)eager; // Just check it compiles and runs
    std::cout << "  AXIOM_EAGER_MODE enabled: " << (eager ? "yes" : "no") << std::endl;
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Lazy Evaluation Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_lazy_deferral);
    RUN_TEST(test_lazy_correctness_add);
    RUN_TEST(test_lazy_correctness_multiply);
    RUN_TEST(test_lazy_chain);
    RUN_TEST(test_lazy_unary);
    RUN_TEST(test_lazy_relu);
    RUN_TEST(test_lazy_reduction);
    RUN_TEST(test_lazy_matmul);
    RUN_TEST(test_lazy_comparison);
    RUN_TEST(test_lazy_broadcast);
    RUN_TEST(test_eager_mode_env_var);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_total << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_passed == tests_total ? 0 : 1;
}
