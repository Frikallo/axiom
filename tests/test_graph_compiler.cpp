#include "axiom_test_utils.hpp"

#include "axiom/graph/compiled_graph.hpp"
#include "axiom/graph/graph_cache.hpp"
#include "axiom/graph/graph_compiler.hpp"
#include "axiom/graph/graph_executor.hpp"
#include "axiom/graph/graph_registry.hpp"
#include "axiom/graph/graph_signature.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

#include <cmath>

using namespace axiom;

// ============================================================================
// Signature Tests
// ============================================================================

TEST(GraphCompiler, SignatureDeterminism) {
    auto a = Tensor::randn({64, 64});
    auto b = Tensor::randn({64, 64});
    auto c = ops::add(a, b);
    auto d = ops::relu(c);

    auto *node = d.lazy_node().get();
    auto sig1 = graph::compute_signature(node);
    auto sig2 = graph::compute_signature(node);

    ASSERT_EQ(sig1, sig2);
}

TEST(GraphCompiler, SignatureUniqueness) {
    auto a = Tensor::randn({64, 64});
    auto b = Tensor::randn({64, 64});

    auto c1 = ops::add(a, b);
    auto c2 = ops::multiply(a, b);

    auto sig1 = graph::compute_signature(c1.lazy_node().get());
    auto sig2 = graph::compute_signature(c2.lazy_node().get());
    ASSERT_NE(sig1, sig2);

    // Different shapes should produce different signatures
    auto x1 = Tensor::randn({32, 32});
    auto y1 = Tensor::randn({32, 32});
    auto x2 = Tensor::randn({64, 64});
    auto y2 = Tensor::randn({64, 64});

    auto r1 = ops::add(x1, y1);
    auto r2 = ops::add(x2, y2);

    auto s1 = graph::compute_signature(r1.lazy_node().get());
    auto s2 = graph::compute_signature(r2.lazy_node().get());
    ASSERT_NE(s1, s2);
}

TEST(GraphCompiler, SignatureSameStructure) {
    // Same structure, different data → same signature
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
    ASSERT_EQ(sig1, sig2);
}

// ============================================================================
// Cache Tests
// ============================================================================

TEST(GraphCompiler, CacheHit) {
    graph::GraphCache::instance().clear();

    // First materialization — cache miss
    auto a1 = Tensor::full<float>({100, 100}, 1.0f);
    auto b1 = Tensor::full<float>({100, 100}, 2.0f);
    auto c1 = ops::add(a1, b1);
    auto d1 = ops::relu(c1);
    float v1 = d1.item<float>({0, 0});
    (void)v1;

    size_t misses_after_first = graph::GraphCache::instance().misses();
    ASSERT_GE(misses_after_first, 1u);

    // Second materialization of same-shaped graph — cache hit
    auto a2 = Tensor::full<float>({100, 100}, 3.0f);
    auto b2 = Tensor::full<float>({100, 100}, 4.0f);
    auto c2 = ops::add(a2, b2);
    auto d2 = ops::relu(c2);
    float v2 = d2.item<float>({0, 0});
    (void)v2;

    size_t hits_after_second = graph::GraphCache::instance().hits();
    ASSERT_GE(hits_after_second, 1u);
}

TEST(GraphCompiler, CacheMissOnShapeChange) {
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
    ASSERT_GT(misses2, misses1);
}

// ============================================================================
// Compiler Tests
// ============================================================================

TEST(GraphCompiler, SingleOp) {
    auto a = Tensor::full<float>({10}, 2.0f);
    auto b = Tensor::full<float>({10}, 3.0f);
    auto c = ops::add(a, b);

    float val = c.item<float>({0});
    ASSERT_NEAR(val, 5.0f, 1e-5f);
}

TEST(GraphCompiler, Reduction) {
    auto a = Tensor::full<float>({10, 10}, 1.0f);
    auto s = ops::sum(a);

    float val = s.item<float>({0});
    ASSERT_NEAR(val, 100.0f, 1e-4f);
}

TEST(GraphCompiler, Matmul) {
    auto a = Tensor::full<float>({4, 3}, 1.0f);
    auto b = Tensor::full<float>({3, 5}, 1.0f);
    auto c = ops::matmul(a, b);

    ASSERT_EQ(c.shape(), Shape({4, 5}));
    float val = c.item<float>({0, 0});
    ASSERT_NEAR(val, 3.0f, 1e-5f);
}

// ============================================================================
// Generic Fused Correctness
// ============================================================================

TEST(GraphCompiler, GenericFusedCorrectness) {
    // Chain: sqrt(exp(x + y))
    auto x = Tensor::full<float>({100}, 0.5f);
    auto y = Tensor::full<float>({100}, 0.5f);
    auto sum = ops::add(x, y);
    auto e = ops::exp(sum);
    auto result = ops::sqrt(e);

    float val = result.item<float>({0});
    float expected = std::sqrt(std::exp(1.0f));
    ASSERT_NEAR(val, expected, 1e-5f);
}

TEST(GraphCompiler, KnownPatternPreserved) {
    // AddReLU should use the known SIMD pattern
    auto a = Tensor::full<float>({1000}, -1.0f);
    auto b = Tensor::full<float>({1000}, 0.5f);
    auto c = ops::relu(ops::add(a, b));

    float val = c.item<float>({0});
    ASSERT_NEAR(val, 0.0f, 1e-5f);

    auto a2 = Tensor::full<float>({1000}, 2.0f);
    auto b2 = Tensor::full<float>({1000}, 3.0f);
    auto c2 = ops::relu(ops::add(a2, b2));
    float val2 = c2.item<float>({0});
    ASSERT_NEAR(val2, 5.0f, 1e-5f);
}

// ============================================================================
// DCE Test
// ============================================================================

TEST(GraphCompiler, DCE) {
    auto a = Tensor::full<float>({100}, 1.0f);
    auto b = Tensor::full<float>({100}, 2.0f);
    auto c = Tensor::full<float>({100}, 3.0f);

    auto live = ops::add(a, b);

    // Dead path (never used)
    auto dead = ops::multiply(c, c);
    (void)dead;

    float val = live.item<float>({0});
    ASSERT_NEAR(val, 3.0f, 1e-5f);
}

// ============================================================================
// Backward Compat: Eager Mode
// ============================================================================

TEST(GraphCompiler, BackwardCompatEager) {
    graph::EagerModeScope scope;

    auto a = Tensor::full<float>({10, 10}, 2.0f);
    auto b = Tensor::full<float>({10, 10}, 3.0f);
    auto c = ops::add(a, b);
    auto d = ops::relu(c);

    float val = d.item<float>({0, 0});
    ASSERT_NEAR(val, 5.0f, 1e-5f);
}

// ============================================================================
// Memory Plan Reuse
// ============================================================================

TEST(GraphCompiler, MemoryPlanReuse) {
    graph::GraphCache::instance().clear();

    auto a = Tensor::full<float>({1000}, 1.0f);
    auto b = Tensor::full<float>({1000}, 2.0f);

    // Chain: add -> exp -> sqrt
    auto c = ops::add(a, b);
    auto d = ops::exp(c);
    auto e = ops::sqrt(d);

    float val = e.item<float>({0});
    float expected = std::sqrt(std::exp(3.0f));
    ASSERT_NEAR(val, expected, 1e-4f);

    ASSERT_GE(graph::GraphCache::instance().size(), 1u);
}

// ============================================================================
// Reduction Chain
// ============================================================================

TEST(GraphCompiler, ReductionChain) {
    // sum(exp(x))
    auto x = Tensor::full<float>({10}, 0.0f);
    auto e = ops::exp(x);
    auto s = ops::sum(e);

    float val = s.item<float>({0});
    // exp(0) = 1, sum of 10 ones = 10
    ASSERT_NEAR(val, 10.0f, 1e-4f);
}

// ============================================================================
// MatMul + Activation
// ============================================================================

TEST(GraphCompiler, MatmulActivation) {
    auto a = Tensor::full<float>({4, 3}, 1.0f);
    auto b = Tensor::full<float>({3, 5}, 1.0f);
    auto mm = ops::matmul(a, b);
    auto r = ops::relu(mm);

    ASSERT_EQ(r.shape(), Shape({4, 5}));
    float val = r.item<float>({0, 0});
    // matmul of ones: inner dim 3, so result is 3.0
    ASSERT_NEAR(val, 3.0f, 1e-5f);
}
