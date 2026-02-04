// Tests for broadcast utilities

#include <axiom/axiom.hpp>
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

void test_broadcast_shapes() {
    // Test broadcasting two shapes
    auto shapes = std::vector<axiom::Shape>{{3, 1}, {1, 4}};
    auto result = axiom::ops::broadcast_shapes(shapes);
    ASSERT(result == axiom::Shape({3, 4}), "broadcast_shapes failed");

    // Test broadcasting three shapes
    shapes = {{2, 1, 3}, {1, 4, 1}, {1, 1, 1}};
    result = axiom::ops::broadcast_shapes(shapes);
    ASSERT(result == axiom::Shape({2, 4, 3}),
           "broadcast_shapes 3 shapes failed");

    // Test broadcasting with different ndim
    shapes = {{5}, {2, 3, 5}};
    result = axiom::ops::broadcast_shapes(shapes);
    ASSERT(result == axiom::Shape({2, 3, 5}),
           "broadcast_shapes different ndim failed");

    // Test single shape
    shapes = {{3, 4}};
    result = axiom::ops::broadcast_shapes(shapes);
    ASSERT(result == axiom::Shape({3, 4}),
           "broadcast_shapes single shape failed");
}

void test_broadcast_tensors() {
    auto a = axiom::Tensor::ones({3, 1});
    auto b = axiom::Tensor::ones({1, 4});

    auto result = axiom::ops::broadcast_tensors({a, b});
    ASSERT(result.size() == 2, "Should return 2 tensors");
    ASSERT(result[0].shape() == axiom::Shape({3, 4}),
           "First tensor shape wrong");
    ASSERT(result[1].shape() == axiom::Shape({3, 4}),
           "Second tensor shape wrong");

    // Verify values are correct (broadcasting expands via strides)
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            ASSERT(result[0].item<float>({i, j}) == 1.0f,
                   "Broadcast tensor 0 value wrong");
            ASSERT(result[1].item<float>({i, j}) == 1.0f,
                   "Broadcast tensor 1 value wrong");
        }
    }
}

void test_broadcast_tensors_three() {
    auto a = axiom::Tensor::full({2, 1, 1}, 1.0f);
    auto b = axiom::Tensor::full({1, 3, 1}, 2.0f);
    auto c = axiom::Tensor::full({1, 1, 4}, 3.0f);

    auto result = axiom::ops::broadcast_tensors({a, b, c});
    ASSERT(result.size() == 3, "Should return 3 tensors");
    ASSERT(result[0].shape() == axiom::Shape({2, 3, 4}),
           "First tensor shape wrong");
    ASSERT(result[1].shape() == axiom::Shape({2, 3, 4}),
           "Second tensor shape wrong");
    ASSERT(result[2].shape() == axiom::Shape({2, 3, 4}),
           "Third tensor shape wrong");
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Broadcast Utilities Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_broadcast_shapes);
    RUN_TEST(test_broadcast_tensors);
    RUN_TEST(test_broadcast_tensors_three);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
