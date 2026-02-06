#include "axiom/graph/compiled_graph.hpp"
#include "axiom/graph/graph_cache.hpp"
#include "axiom/graph/graph_compiler.hpp"
#include "axiom/graph/graph_executor.hpp"
#include "axiom/graph/graph_registry.hpp"
#include "axiom/graph/graph_signature.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

#include <cmath>
#include <iostream>

using namespace axiom;

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

// ============================================================================
// Signature Tests
// ============================================================================

int test_signature_determinism() {
    auto a = Tensor::randn({64, 64});
    auto b = Tensor::randn({64, 64});
    auto c = ops::add(a, b);
    auto d = ops::relu(c);

    auto *node = d.lazy_node().get();
    auto sig1 = graph::compute_signature(node);
    auto sig2 = graph::compute_signature(node);

    ASSERT(sig1 == sig2);
    return 0;
}

int test_signature_uniqueness() {
    // Different ops should produce different signatures
    auto a = Tensor::randn({64, 64});
    auto b = Tensor::randn({64, 64});

    auto c1 = ops::add(a, b);
    auto c2 = ops::multiply(a, b);

    auto sig1 = graph::compute_signature(c1.lazy_node().get());
    auto sig2 = graph::compute_signature(c2.lazy_node().get());

    ASSERT(sig1 != sig2);

    // Different shapes should produce different signatures
    auto x1 = Tensor::randn({32, 32});
    auto y1 = Tensor::randn({32, 32});
    auto x2 = Tensor::randn({64, 64});
    auto y2 = Tensor::randn({64, 64});

    auto r1 = ops::add(x1, y1);
    auto r2 = ops::add(x2, y2);

    auto s1 = graph::compute_signature(r1.lazy_node().get());
    auto s2 = graph::compute_signature(r2.lazy_node().get());

    ASSERT(s1 != s2);

    return 0;
}

int test_signature_same_structure() {
    // Two graphs with the same structure but different data
    // should have the same signature
    auto a1 = Tensor::full<float>({100, 100}, 1.0f);
    auto b1 = Tensor::full<float>({100, 100}, 2.0f);
    auto c1 = ops::add(a1, b1);
    auto d1 = ops::relu(c1);

    auto a2 = Tensor::full<float>({100, 100}, 3.0f);
    auto b2 = Tensor::full<float>({100, 100}, 4.0f);
    auto c2 = ops::add(a2, b2);
    auto d2 = ops::relu(c2);

    auto sig1 = graph::compute_signature(d1.lazy_node().get());
    auto sig2 = graph::compute_signature(d2.lazy_node().get());

    ASSERT(sig1 == sig2);

    return 0;
}

// ============================================================================
// Cache Tests
// ============================================================================

int test_cache_hit() {
    graph::GraphCache::instance().clear();

    // First materialization — cache miss
    auto a1 = Tensor::full<float>({100, 100}, 1.0f);
    auto b1 = Tensor::full<float>({100, 100}, 2.0f);
    auto c1 = ops::add(a1, b1);
    auto d1 = ops::relu(c1);
    float v1 = d1.item<float>({0, 0}); // triggers materialization
    (void)v1;

    size_t misses_after_first = graph::GraphCache::instance().misses();
    ASSERT(misses_after_first >= 1);

    // Second materialization of same-shaped graph — cache hit
    auto a2 = Tensor::full<float>({100, 100}, 3.0f);
    auto b2 = Tensor::full<float>({100, 100}, 4.0f);
    auto c2 = ops::add(a2, b2);
    auto d2 = ops::relu(c2);
    float v2 = d2.item<float>({0, 0});
    (void)v2;

    size_t hits_after_second = graph::GraphCache::instance().hits();
    ASSERT(hits_after_second >= 1);

    return 0;
}

int test_cache_miss_on_shape_change() {
    graph::GraphCache::instance().clear();

    auto a1 = Tensor::full<float>({50, 50}, 1.0f);
    auto b1 = Tensor::full<float>({50, 50}, 2.0f);
    auto c1 = ops::add(a1, b1);
    float v1 = c1.item<float>({0, 0});
    (void)v1;

    size_t misses1 = graph::GraphCache::instance().misses();

    auto a2 = Tensor::full<float>({100, 100}, 1.0f);
    auto b2 = Tensor::full<float>({100, 100}, 2.0f);
    auto c2 = ops::add(a2, b2);
    float v2 = c2.item<float>({0, 0});
    (void)v2;

    size_t misses2 = graph::GraphCache::instance().misses();
    ASSERT(misses2 > misses1);

    return 0;
}

// ============================================================================
// Compiler Tests
// ============================================================================

int test_compiler_single_op() {
    auto a = Tensor::full<float>({10}, 2.0f);
    auto b = Tensor::full<float>({10}, 3.0f);
    auto c = ops::add(a, b);

    // Verify result is correct
    float val = c.item<float>({0});
    ASSERT_NEAR(val, 5.0f, 1e-5f);

    return 0;
}

int test_compiler_reduction() {
    auto a = Tensor::full<float>({10, 10}, 1.0f);
    auto s = ops::sum(a);

    float val = s.item<float>({0});
    ASSERT_NEAR(val, 100.0f, 1e-4f);

    return 0;
}

int test_compiler_matmul() {
    auto a = Tensor::full<float>({4, 3}, 1.0f);
    auto b = Tensor::full<float>({3, 5}, 1.0f);
    auto c = ops::matmul(a, b);

    ASSERT(c.shape() == Shape({4, 5}));
    float val = c.item<float>({0, 0});
    ASSERT_NEAR(val, 3.0f, 1e-5f);

    return 0;
}

// ============================================================================
// Generic Fused Correctness
// ============================================================================

int test_generic_fused_correctness() {
    // Build a chain: sqrt(exp(x + y))
    auto x = Tensor::full<float>({100}, 0.5f);
    auto y = Tensor::full<float>({100}, 0.5f);
    auto sum = ops::add(x, y);
    auto e = ops::exp(sum);
    auto result = ops::sqrt(e);

    float val = result.item<float>({0});
    float expected = std::sqrt(std::exp(1.0f));
    ASSERT_NEAR(val, expected, 1e-5f);

    return 0;
}

int test_known_pattern_preserved() {
    // AddReLU should use the known SIMD pattern
    auto a = Tensor::full<float>({1000}, -1.0f);
    auto b = Tensor::full<float>({1000}, 0.5f);
    auto c = ops::relu(ops::add(a, b));

    // Result: relu(-1 + 0.5) = relu(-0.5) = 0
    float val = c.item<float>({0});
    ASSERT_NEAR(val, 0.0f, 1e-5f);

    // Also check a positive case
    auto a2 = Tensor::full<float>({1000}, 2.0f);
    auto b2 = Tensor::full<float>({1000}, 3.0f);
    auto c2 = ops::relu(ops::add(a2, b2));
    float val2 = c2.item<float>({0});
    ASSERT_NEAR(val2, 5.0f, 1e-5f);

    return 0;
}

// ============================================================================
// DCE Test
// ============================================================================

int test_dce() {
    // Create a graph with a dead branch
    auto a = Tensor::full<float>({100}, 1.0f);
    auto b = Tensor::full<float>({100}, 2.0f);
    auto c = Tensor::full<float>({100}, 3.0f);

    // Live path: a + b
    auto live = ops::add(a, b);

    // Dead path: c * c (never used but shares a constant)
    auto dead = ops::multiply(c, c);
    (void)dead; // intentionally unused

    float val = live.item<float>({0});
    ASSERT_NEAR(val, 3.0f, 1e-5f);

    return 0;
}

// ============================================================================
// Backward Compat: Eager Mode
// ============================================================================

int test_backward_compat_eager() {
    graph::EagerModeScope scope;

    auto a = Tensor::full<float>({10, 10}, 2.0f);
    auto b = Tensor::full<float>({10, 10}, 3.0f);
    auto c = ops::add(a, b);
    auto d = ops::relu(c);

    float val = d.item<float>({0, 0});
    ASSERT_NEAR(val, 5.0f, 1e-5f);

    return 0;
}

// ============================================================================
// Memory Plan Reuse
// ============================================================================

int test_memory_plan_reuse() {
    graph::GraphCache::instance().clear();

    auto a = Tensor::full<float>({1000}, 1.0f);
    auto b = Tensor::full<float>({1000}, 2.0f);

    // Chain: add -> exp -> sqrt (3 steps in 2 fused)
    auto c = ops::add(a, b);
    auto d = ops::exp(c);
    auto e = ops::sqrt(d);

    // Materialize
    float val = e.item<float>({0});
    float expected = std::sqrt(std::exp(3.0f));
    ASSERT_NEAR(val, expected, 1e-4f);

    // Check that cache has at least one entry
    ASSERT(graph::GraphCache::instance().size() >= 1);

    return 0;
}

// ============================================================================
// Reduction Chain
// ============================================================================

int test_reduction_chain() {
    // sum(exp(x)) — elementwise followed by reduction
    auto x = Tensor::full<float>({10}, 0.0f);
    auto e = ops::exp(x);
    auto s = ops::sum(e);

    float val = s.item<float>({0});
    // exp(0) = 1, sum of 10 ones = 10
    ASSERT_NEAR(val, 10.0f, 1e-4f);

    return 0;
}

// ============================================================================
// MatMul + Activation
// ============================================================================

int test_matmul_activation() {
    auto a = Tensor::full<float>({4, 3}, 1.0f);
    auto b = Tensor::full<float>({3, 5}, 1.0f);
    auto mm = ops::matmul(a, b);
    auto r = ops::relu(mm);

    ASSERT(r.shape() == Shape({4, 5}));
    float val = r.item<float>({0, 0});
    // matmul of ones: inner dim 3, so result is 3.0
    // relu(3.0) = 3.0
    ASSERT_NEAR(val, 3.0f, 1e-5f);

    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    int failures = 0;

    std::cout << "=== Graph Compiler Tests ===" << std::endl;

    RUN_TEST(test_signature_determinism);
    RUN_TEST(test_signature_uniqueness);
    RUN_TEST(test_signature_same_structure);
    RUN_TEST(test_cache_hit);
    RUN_TEST(test_cache_miss_on_shape_change);
    RUN_TEST(test_compiler_single_op);
    RUN_TEST(test_compiler_reduction);
    RUN_TEST(test_compiler_matmul);
    RUN_TEST(test_generic_fused_correctness);
    RUN_TEST(test_known_pattern_preserved);
    RUN_TEST(test_dce);
    RUN_TEST(test_backward_compat_eager);
    RUN_TEST(test_memory_plan_reuse);
    RUN_TEST(test_reduction_chain);
    RUN_TEST(test_matmul_activation);

    std::cout << "\n=== Results ===" << std::endl;
    if (failures == 0) {
        std::cout << "All graph compiler tests passed!" << std::endl;
    } else {
        std::cout << failures << " test(s) FAILED!" << std::endl;
    }

    return failures;
}
