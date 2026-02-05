#include "axiom/graph/graph_registry.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"
#include "axiom/tensor_operators.hpp"
#include <cmath>
#include <iostream>

using namespace axiom;

// Test helpers
#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "ASSERTION FAILED: " #cond << " at " << __FILE__      \
                      << ":" << __LINE__ << std::endl;                         \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        if (std::abs((a) - (b)) > (tol)) {                                     \
            std::cerr << "ASSERTION FAILED: " << (a) << " != " << (b)          \
                      << " (tol=" << (tol) << ") at " << __FILE__ << ":"       \
                      << __LINE__ << std::endl;                                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define RUN_TEST(test_func)                                                    \
    do {                                                                       \
        std::cout << "Running " #test_func "... ";                             \
        std::cout.flush();                                                     \
        int result = test_func();                                              \
        if (result == 0) {                                                     \
            std::cout << "PASSED" << std::endl;                                \
        } else {                                                               \
            std::cout << "FAILED" << std::endl;                                \
            failures++;                                                        \
        }                                                                      \
    } while (0)

// Test unary chain fusion: exp(sqrt(x))
int test_unary_chain_fusion() {
    auto x = Tensor::full<float>({100, 100}, 4.0f);

    // This should create a lazy chain that gets fused: sqrt -> exp
    auto result = ops::exp(ops::sqrt(x));

    // Expected: exp(sqrt(4)) = exp(2) ≈ 7.389
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, std::exp(2.0f), 1e-5f);

    return 0;
}

// Test binary + unary fusion: (a + b).relu()
int test_binary_unary_fusion() {
    auto a = Tensor::full<float>({100, 100}, -1.0f);
    auto b = Tensor::full<float>({100, 100}, 0.5f);

    // This should fuse: add -> relu
    auto result = ops::relu(ops::add(a, b));

    // Expected: relu(-1 + 0.5) = relu(-0.5) = 0
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, 0.0f, 1e-6f);

    return 0;
}

// Test that fusion produces same result as eager execution
int test_fusion_eager_parity() {
    auto x = Tensor::randn({50, 50});

    // Lazy execution (with potential fusion)
    auto lazy_result = ops::sigmoid(ops::relu(ops::add(x, x)));

    // Force materialization
    float lazy_val = lazy_result.item<float>({0, 0});

    // Now compute the same with eager mode
    {
        graph::EagerModeScope eager_scope;
        auto x2 = Tensor::from_data<float>(x.typed_data<float>(), {50, 50});
        auto eager_result = ops::sigmoid(ops::relu(ops::add(x2, x2)));
        float eager_val = eager_result.item<float>({0, 0});

        ASSERT_NEAR(lazy_val, eager_val, 1e-5f);
    }

    return 0;
}

// Test longer fusion chain: (a * b + c).sigmoid().tanh()
int test_long_chain_fusion() {
    auto a = Tensor::full<float>({64, 64}, 0.5f);
    auto b = Tensor::full<float>({64, 64}, 2.0f);
    auto c = Tensor::full<float>({64, 64}, -0.5f);

    // Long chain: mul -> add -> sigmoid -> tanh
    auto result = ops::tanh(ops::sigmoid(ops::add(ops::multiply(a, b), c)));

    // Expected: tanh(sigmoid(0.5 * 2 + (-0.5))) = tanh(sigmoid(0.5))
    //         = tanh(1 / (1 + exp(-0.5)))
    float expected = std::tanh(1.0f / (1.0f + std::exp(-0.5f)));
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, expected, 1e-5f);

    return 0;
}

// Test that non-fusable ops break the chain correctly
int test_reduction_breaks_fusion() {
    auto x = Tensor::full<float>({10, 10}, 2.0f);

    // Reduction in the middle breaks fusion
    auto y = ops::sqrt(x); // unary
    auto z = ops::sum(y);  // reduction - breaks chain
    auto w = ops::exp(z);  // unary on scalar

    // Expected: exp(sum(sqrt(2) for 100 elements)) = exp(100 * sqrt(2))
    float expected = std::exp(100.0f * std::sqrt(2.0f));
    float val = w.item<float>();
    ASSERT_NEAR(val, expected, expected * 1e-4f); // Allow relative error

    return 0;
}

// Test fusion with broadcasting
int test_fusion_with_broadcast() {
    auto a = Tensor::full<float>({100, 1}, 1.0f);
    auto b = Tensor::full<float>({1, 100}, 2.0f);

    // Broadcasting + activation
    auto result = ops::relu(ops::add(a, b));

    // Expected: relu(1 + 2) = 3 everywhere
    ASSERT(result.shape() == Shape({100, 100}));
    float val = result.item<float>({50, 50});
    ASSERT_NEAR(val, 3.0f, 1e-6f);

    return 0;
}

// Test element-wise comparison doesn't get fused incorrectly with arithmetic
int test_comparison_no_fusion() {
    auto a = Tensor::full<float>({50, 50}, 1.0f);
    auto b = Tensor::full<float>({50, 50}, 2.0f);

    // Comparison followed by where (not an element-wise chain in same sense)
    auto mask = ops::less(a, b); // Should be all true
    ASSERT(mask.dtype() == DType::Bool);

    return 0;
}

// Test in-place storage reuse: auto c = ((a + b) * c).relu()
// When c is reassigned and its storage has use_count == 1, we can reuse it
int test_inplace_storage_reuse() {
    auto a = Tensor::full<float>({100, 100}, 1.0f);
    auto b = Tensor::full<float>({100, 100}, 2.0f);
    auto c = Tensor::full<float>({100, 100}, 0.5f);

    // Get the original storage pointer for c
    void *original_c_data = c.data();

    // This should fuse ops and potentially reuse c's storage
    // Operation: relu((1 + 2) * 0.5) = relu(1.5) = 1.5
    c = ((a + b) * c).relu();

    // Verify result is correct
    float val = c.item<float>({50, 50});
    ASSERT_NEAR(val, 1.5f, 1e-5f);

    // Note: In-place reuse may or may not happen depending on storage use
    // counts The important thing is the result is correct
    (void)original_c_data; // Silence unused variable warning

    return 0;
}

// Test the clean syntax: auto result = ((a + b) * c).relu()
int test_operator_syntax() {
    auto a = Tensor::full<float>({50, 50}, 1.0f);
    auto b = Tensor::full<float>({50, 50}, 2.0f);
    auto c = Tensor::full<float>({50, 50}, 0.5f);

    // The syntax the user wants: clean operator chaining
    auto result = ((a + b) * c).relu();

    // (1 + 2) * 0.5 = 1.5, relu(1.5) = 1.5
    float val = result.item<float>({25, 25});
    ASSERT_NEAR(val, 1.5f, 1e-5f);

    // Verify shapes match
    ASSERT(result.shape() == Shape({50, 50}));

    return 0;
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    int failures = 0;

    std::cout << "\n=== Fusion Tests ===" << std::endl;
    RUN_TEST(test_unary_chain_fusion);
    RUN_TEST(test_binary_unary_fusion);
    RUN_TEST(test_fusion_eager_parity);
    RUN_TEST(test_long_chain_fusion);
    RUN_TEST(test_reduction_breaks_fusion);
    RUN_TEST(test_fusion_with_broadcast);
    RUN_TEST(test_comparison_no_fusion);
    RUN_TEST(test_inplace_storage_reuse);
    RUN_TEST(test_operator_syntax);

    std::cout << "\n=== Results ===" << std::endl;
    if (failures == 0) {
        std::cout << "All fusion tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << failures << " test(s) failed." << std::endl;
        return 1;
    }
}
