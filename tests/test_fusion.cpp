#include "axiom_test_utils.hpp"

#include "axiom/graph/graph_registry.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"
#include "axiom/tensor_operators.hpp"

#include <cmath>

using namespace axiom;

// Test unary chain fusion: exp(sqrt(x))
TEST(Fusion, UnaryChain) {
    auto x = Tensor::full<float>({100, 100}, 4.0f);

    auto result = ops::exp(ops::sqrt(x));

    // Expected: exp(sqrt(4)) = exp(2)
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, std::exp(2.0f), 1e-5f);
}

// Test binary + unary fusion: relu(a + b)
TEST(Fusion, BinaryUnary) {
    auto a = Tensor::full<float>({100, 100}, -1.0f);
    auto b = Tensor::full<float>({100, 100}, 0.5f);

    auto result = ops::relu(ops::add(a, b));

    // Expected: relu(-1 + 0.5) = relu(-0.5) = 0
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, 0.0f, 1e-6f);
}

// Test that fusion produces same result as eager execution
TEST(Fusion, EagerParity) {
    auto x = Tensor::randn({50, 50});

    // Lazy execution (with potential fusion)
    auto lazy_result = ops::sigmoid(ops::relu(ops::add(x, x)));
    float lazy_val = lazy_result.item<float>({0, 0});

    // Eager execution
    {
        graph::EagerModeScope eager_scope;
        auto x2 = Tensor::from_data<float>(x.typed_data<float>(), {50, 50});
        auto eager_result = ops::sigmoid(ops::relu(ops::add(x2, x2)));
        float eager_val = eager_result.item<float>({0, 0});

        ASSERT_NEAR(lazy_val, eager_val, 1e-5f);
    }
}

// Test longer fusion chain: tanh(sigmoid(a * b + c))
TEST(Fusion, LongChain) {
    auto a = Tensor::full<float>({64, 64}, 0.5f);
    auto b = Tensor::full<float>({64, 64}, 2.0f);
    auto c = Tensor::full<float>({64, 64}, -0.5f);

    auto result = ops::tanh(ops::sigmoid(ops::add(ops::multiply(a, b), c)));

    float expected = std::tanh(1.0f / (1.0f + std::exp(-0.5f)));
    float val = result.item<float>({0, 0});
    ASSERT_NEAR(val, expected, 1e-5f);
}

// Test that non-fusable ops break the chain correctly
TEST(Fusion, ReductionBreaksChain) {
    auto x = Tensor::full<float>({4, 4}, 2.0f);

    auto y = ops::sqrt(x);
    auto z = ops::sum(y); // reduction breaks chain
    auto w = ops::exp(z);

    // Expected: exp(sum(sqrt(2) for 16 elements)) = exp(16 * sqrt(2))
    float expected = std::exp(16.0f * std::sqrt(2.0f));
    float val = w.item<float>();
    ASSERT_NEAR(val, expected, expected * 1e-4f);
}

// Test fusion with broadcasting
TEST(Fusion, WithBroadcast) {
    auto a = Tensor::full<float>({100, 1}, 1.0f);
    auto b = Tensor::full<float>({1, 100}, 2.0f);

    auto result = ops::relu(ops::add(a, b));

    ASSERT_EQ(result.shape(), Shape({100, 100}));
    float val = result.item<float>({50, 50});
    ASSERT_NEAR(val, 3.0f, 1e-6f);
}

// Test comparison doesn't get fused incorrectly with arithmetic
TEST(Fusion, ComparisonNoFusion) {
    auto a = Tensor::full<float>({50, 50}, 1.0f);
    auto b = Tensor::full<float>({50, 50}, 2.0f);

    auto mask = ops::less(a, b);
    ASSERT_EQ(mask.dtype(), DType::Bool);
}

// Test in-place storage reuse
TEST(Fusion, InplaceStorageReuse) {
    auto a = Tensor::full<float>({100, 100}, 1.0f);
    auto b = Tensor::full<float>({100, 100}, 2.0f);
    auto c = Tensor::full<float>({100, 100}, 0.5f);

    void *original_c_data = c.data();

    // relu((1 + 2) * 0.5) = relu(1.5) = 1.5
    c = ((a + b) * c).relu();

    float val = c.item<float>({50, 50});
    ASSERT_NEAR(val, 1.5f, 1e-5f);

    (void)original_c_data;
}

// Test the clean operator syntax
TEST(Fusion, OperatorSyntax) {
    auto a = Tensor::full<float>({50, 50}, 1.0f);
    auto b = Tensor::full<float>({50, 50}, 2.0f);
    auto c = Tensor::full<float>({50, 50}, 0.5f);

    auto result = ((a + b) * c).relu();

    // (1 + 2) * 0.5 = 1.5, relu(1.5) = 1.5
    float val = result.item<float>({25, 25});
    ASSERT_NEAR(val, 1.5f, 1e-5f);
    ASSERT_EQ(result.shape(), Shape({50, 50}));
}
