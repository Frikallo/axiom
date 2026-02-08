// Tests for broadcast utilities

#include "axiom_test_utils.hpp"

using namespace axiom;

TEST(Broadcast, Shapes) {
    // Test broadcasting two shapes
    auto shapes = std::vector<Shape>{{3, 1}, {1, 4}};
    auto result = ops::broadcast_shapes(shapes);
    ASSERT_TRUE(result == Shape({3, 4})) << "broadcast_shapes failed";

    // Test broadcasting three shapes
    shapes = {{2, 1, 3}, {1, 4, 1}, {1, 1, 1}};
    result = ops::broadcast_shapes(shapes);
    ASSERT_TRUE(result == Shape({2, 4, 3}))
        << "broadcast_shapes 3 shapes failed";

    // Test broadcasting with different ndim
    shapes = {{5}, {2, 3, 5}};
    result = ops::broadcast_shapes(shapes);
    ASSERT_TRUE(result == Shape({2, 3, 5}))
        << "broadcast_shapes different ndim failed";

    // Test single shape
    shapes = {{3, 4}};
    result = ops::broadcast_shapes(shapes);
    ASSERT_TRUE(result == Shape({3, 4}))
        << "broadcast_shapes single shape failed";
}

TEST(Broadcast, Tensors) {
    auto a = Tensor::ones({3, 1});
    auto b = Tensor::ones({1, 4});

    auto result = ops::broadcast_tensors({a, b});
    ASSERT_TRUE(result.size() == 2) << "Should return 2 tensors";
    ASSERT_TRUE(result[0].shape() == Shape({3, 4}))
        << "First tensor shape wrong";
    ASSERT_TRUE(result[1].shape() == Shape({3, 4}))
        << "Second tensor shape wrong";

    // Verify values are correct (broadcasting expands via strides)
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            ASSERT_TRUE(result[0].item<float>({i, j}) == 1.0f)
                << "Broadcast tensor 0 value wrong";
            ASSERT_TRUE(result[1].item<float>({i, j}) == 1.0f)
                << "Broadcast tensor 1 value wrong";
        }
    }
}

TEST(Broadcast, TensorsThree) {
    auto a = Tensor::full({2, 1, 1}, 1.0f);
    auto b = Tensor::full({1, 3, 1}, 2.0f);
    auto c = Tensor::full({1, 1, 4}, 3.0f);

    auto result = ops::broadcast_tensors({a, b, c});
    ASSERT_TRUE(result.size() == 3) << "Should return 3 tensors";
    ASSERT_TRUE(result[0].shape() == Shape({2, 3, 4}))
        << "First tensor shape wrong";
    ASSERT_TRUE(result[1].shape() == Shape({2, 3, 4}))
        << "Second tensor shape wrong";
    ASSERT_TRUE(result[2].shape() == Shape({2, 3, 4}))
        << "Third tensor shape wrong";
}
