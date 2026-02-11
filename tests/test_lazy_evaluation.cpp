#include "axiom_test_utils.hpp"

#include "axiom/graph/graph_registry.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

#include <cmath>

using namespace axiom;

// Test that lazy tensors defer execution
TEST(LazyEvaluation, DeferralAndMaterialization) {
    auto a = Tensor::randn({100, 100});
    auto b = Tensor::randn({100, 100});

    // These operations should create lazy tensors
    auto c = ops::add(a, b);
    auto d = ops::multiply(c, c);

    // Shape should be inferred without execution
    ASSERT_EQ(d.shape(), Shape({100, 100}));

    // Access data — triggers materialization
    float first_val = d.typed_data<float>()[0];
    (void)first_val;

    ASSERT_EQ(d.size(), 10000u);
}

TEST(LazyEvaluation, CorrectnessAdd) {
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    auto c = ops::add(a, b);

    float val = c.typed_data<float>()[0];
    ASSERT_NEAR(val, 5.0f, 1e-5f);
}

TEST(LazyEvaluation, CorrectnessMultiply) {
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    auto c = ops::multiply(a, b);

    float val = c.typed_data<float>()[0];
    ASSERT_NEAR(val, 6.0f, 1e-5f);
}

TEST(LazyEvaluation, Chain) {
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    // (a + b) * a = (2 + 3) * 2 = 10
    auto c = ops::add(a, b);
    auto d = ops::multiply(c, a);

    float val = d.typed_data<float>()[0];
    ASSERT_NEAR(val, 10.0f, 1e-5f);
}

TEST(LazyEvaluation, Unary) {
    auto a = Tensor::full<float>({3, 3}, 4.0f);

    auto b = ops::sqrt(a);

    float val = b.typed_data<float>()[0];
    ASSERT_NEAR(val, 2.0f, 1e-5f);
}

TEST(LazyEvaluation, ReLU) {
    auto a = Tensor::full<float>({3, 3}, -2.0f);

    auto b = ops::relu(a);

    float val = b.typed_data<float>()[0];
    ASSERT_NEAR(val, 0.0f, 1e-5f);
}

TEST(LazyEvaluation, Reduction) {
    auto a = Tensor::full<float>({2, 3}, 2.0f);

    // Sum all elements: 2 * 3 * 2.0 = 12.0
    auto b = ops::sum(a);

    float val = b.item<float>();
    ASSERT_NEAR(val, 12.0f, 1e-5f);
}

TEST(LazyEvaluation, Matmul) {
    auto a = Tensor::full<float>({2, 2}, 1.0f);
    auto b = Tensor::full<float>({2, 2}, 2.0f);

    auto c = ops::matmul(a, b);

    // Each element should be 1*2 + 1*2 = 4.0
    ASSERT_EQ(c.shape(), Shape({2, 2}));
    float val = c.typed_data<float>()[0];
    ASSERT_NEAR(val, 4.0f, 1e-5f);
}

TEST(LazyEvaluation, Comparison) {
    auto a = Tensor::full<float>({3, 3}, 2.0f);
    auto b = Tensor::full<float>({3, 3}, 3.0f);

    auto c = ops::less(a, b);

    ASSERT_EQ(c.dtype(), DType::Bool);
    bool val = c.typed_data<bool>()[0];
    ASSERT_TRUE(val);
}

TEST(LazyEvaluation, Broadcast) {
    auto a = Tensor::full<float>({3, 1}, 2.0f);
    auto b = Tensor::full<float>({1, 4}, 3.0f);

    auto c = ops::add(a, b);

    ASSERT_EQ(c.shape(), Shape({3, 4}));
    float val = c.typed_data<float>()[0];
    ASSERT_NEAR(val, 5.0f, 1e-5f);
}

TEST(LazyEvaluation, EagerModeEnvVar) {
    bool eager = graph::is_eager_mode_enabled();
    (void)eager; // Just check it compiles and runs
}
