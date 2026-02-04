// Tests for extended shape operations (Phase 3)

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

void test_meshgrid_xy() {
    auto x = axiom::Tensor::arange(3);
    auto y = axiom::Tensor::arange(4);

    auto grids = axiom::ops::meshgrid({x, y}, "xy");
    ASSERT(grids.size() == 2, "Should return 2 grids");
    ASSERT(grids[0].shape() == axiom::Shape({4, 3}),
           "X grid shape wrong for xy");
    ASSERT(grids[1].shape() == axiom::Shape({4, 3}),
           "Y grid shape wrong for xy");
}

void test_meshgrid_ij() {
    auto x = axiom::Tensor::arange(3);
    auto y = axiom::Tensor::arange(4);

    auto grids = axiom::ops::meshgrid({x, y}, "ij");
    ASSERT(grids[0].shape() == axiom::Shape({3, 4}),
           "X grid shape wrong for ij");
    ASSERT(grids[1].shape() == axiom::Shape({3, 4}),
           "Y grid shape wrong for ij");
}

void test_meshgrid_three() {
    auto x = axiom::Tensor::arange(2);
    auto y = axiom::Tensor::arange(3);
    auto z = axiom::Tensor::arange(4);

    auto grids = axiom::ops::meshgrid({x, y, z}, "ij");
    ASSERT(grids.size() == 3, "Should return 3 grids");
    ASSERT(grids[0].shape() == axiom::Shape({2, 3, 4}), "X grid shape wrong");
    ASSERT(grids[1].shape() == axiom::Shape({2, 3, 4}), "Y grid shape wrong");
    ASSERT(grids[2].shape() == axiom::Shape({2, 3, 4}), "Z grid shape wrong");
}

void test_pad_constant() {
    auto t = axiom::Tensor::ones({3, 3});
    auto padded = axiom::ops::pad(t, {{1, 1}, {1, 1}}, "constant", 0.0);

    ASSERT(padded.shape() == axiom::Shape({5, 5}), "Padded shape wrong");
    // Check corner is 0
    ASSERT(padded.item<float>({0, 0}) == 0.0f, "Padding value should be 0");
    // Check center is 1
    ASSERT(padded.item<float>({2, 2}) == 1.0f, "Original value should be 1");
}

void test_pad_asymmetric() {
    auto t = axiom::Tensor::ones({2, 2});
    auto padded = axiom::ops::pad(t, {{1, 2}, {0, 3}}, "constant", 5.0);

    ASSERT(padded.shape() == axiom::Shape({5, 5}),
           "Asymmetric padded shape wrong");
    ASSERT(padded.item<float>({0, 0}) == 5.0f, "Top padding value wrong");
    ASSERT(padded.item<float>({4, 4}) == 5.0f,
           "Bottom-right padding value wrong");
    ASSERT(padded.item<float>({1, 0}) == 1.0f, "Original value position wrong");
}

void test_pad_1d() {
    auto t = axiom::Tensor::arange(5);
    auto padded = axiom::ops::pad(t, {{2, 2}}, "constant", -1.0);

    ASSERT(padded.shape() == axiom::Shape({9}), "1D padded shape wrong");
    ASSERT(padded.item<int32_t>({0}) == -1, "Left padding wrong");
    ASSERT(padded.item<int32_t>({2}) == 0, "First original value wrong");
    ASSERT(padded.item<int32_t>({8}) == -1, "Right padding wrong");
}

void test_atleast_1d() {
    // Scalar to 1D
    auto scalar = axiom::Tensor::full({}, 1.0f);
    auto t1d = axiom::ops::atleast_1d(scalar);
    ASSERT(t1d.ndim() == 1, "atleast_1d should make scalar 1D");
    ASSERT(t1d.shape() == axiom::Shape({1}), "atleast_1d scalar shape wrong");

    // Already 1D
    auto vec = axiom::Tensor::arange(5);
    auto vec1d = axiom::ops::atleast_1d(vec);
    ASSERT(vec1d.shape() == vec.shape(),
           "atleast_1d should not change 1D tensor");

    // Higher dim unchanged
    auto mat = axiom::Tensor::ones({3, 4});
    auto mat1d = axiom::ops::atleast_1d(mat);
    ASSERT(mat1d.shape() == mat.shape(),
           "atleast_1d should not change 2D tensor");
}

void test_atleast_2d() {
    // Scalar to 2D
    auto scalar = axiom::Tensor::full({}, 2.0f);
    auto t2d = axiom::ops::atleast_2d(scalar);
    ASSERT(t2d.ndim() == 2, "atleast_2d should make scalar 2D");
    ASSERT(t2d.shape() == axiom::Shape({1, 1}),
           "atleast_2d scalar shape wrong");

    // 1D to 2D
    auto vec = axiom::Tensor::arange(5);
    auto vec2d = axiom::ops::atleast_2d(vec);
    ASSERT(vec2d.ndim() == 2, "atleast_2d should make 1D tensor 2D");
    ASSERT(vec2d.shape() == axiom::Shape({1, 5}), "atleast_2d 1D shape wrong");

    // Already 2D
    auto mat = axiom::Tensor::ones({3, 4});
    auto mat2d = axiom::ops::atleast_2d(mat);
    ASSERT(mat2d.shape() == mat.shape(),
           "atleast_2d should not change 2D tensor");
}

void test_atleast_3d() {
    // Scalar to 3D
    auto scalar = axiom::Tensor::full({}, 3.0f);
    auto t3d = axiom::ops::atleast_3d(scalar);
    ASSERT(t3d.ndim() == 3, "atleast_3d should make scalar 3D");
    ASSERT(t3d.shape() == axiom::Shape({1, 1, 1}),
           "atleast_3d scalar shape wrong");

    // 1D to 3D
    auto vec = axiom::Tensor::arange(5);
    auto vec3d = axiom::ops::atleast_3d(vec);
    ASSERT(vec3d.ndim() == 3, "atleast_3d should make 1D tensor 3D");
    ASSERT(vec3d.shape() == axiom::Shape({1, 5, 1}),
           "atleast_3d 1D shape wrong");

    // 2D to 3D
    auto mat = axiom::Tensor::ones({3, 4});
    auto mat3d = axiom::ops::atleast_3d(mat);
    ASSERT(mat3d.ndim() == 3, "atleast_3d should make 2D tensor 3D");
    ASSERT(mat3d.shape() == axiom::Shape({3, 4, 1}),
           "atleast_3d 2D shape wrong");
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Extended Shape Operations Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_meshgrid_xy);
    RUN_TEST(test_meshgrid_ij);
    RUN_TEST(test_meshgrid_three);
    RUN_TEST(test_pad_constant);
    RUN_TEST(test_pad_asymmetric);
    RUN_TEST(test_pad_1d);
    RUN_TEST(test_atleast_1d);
    RUN_TEST(test_atleast_2d);
    RUN_TEST(test_atleast_3d);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
