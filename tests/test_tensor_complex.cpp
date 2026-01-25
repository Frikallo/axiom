#include <axiom/axiom.hpp>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace axiom;

// Test harness
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

// ============================================================================
// Complex Tensor Creation Tests
// ============================================================================

void test_complex64_creation() {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    ASSERT(t.dtype() == DType::Complex64, "Should be Complex64");
    ASSERT(t.size() == 2, "Should have 2 elements");
    ASSERT(t.itemsize() == 8, "Complex64 should be 8 bytes");
}

void test_complex128_creation() {
    std::vector<complex128_t> data = {{1.0, 2.0}, {3.0, 4.0}};
    auto t = Tensor::from_data(data.data(), {2});

    ASSERT(t.dtype() == DType::Complex128, "Should be Complex128");
    ASSERT(t.size() == 2, "Should have 2 elements");
    ASSERT(t.itemsize() == 16, "Complex128 should be 16 bytes");
}

// ============================================================================
// Real/Imag View Tests
// ============================================================================

void test_real_view() {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto real_view = t.real();

    ASSERT(real_view.dtype() == DType::Float32, "Real view should be Float32");
    ASSERT(real_view.shape() == Shape{2}, "Real view shape should match");

    // Real view is strided (stride=8, itemsize=4), so use item() for proper
    // access
    ASSERT(std::abs(real_view.item<float>({0}) - 1.0f) < 1e-5f,
           "First real should be 1.0");
    ASSERT(std::abs(real_view.item<float>({1}) - 3.0f) < 1e-5f,
           "Second real should be 3.0");
}

void test_imag_view() {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto imag_view = t.imag();

    ASSERT(imag_view.dtype() == DType::Float32, "Imag view should be Float32");
    ASSERT(imag_view.shape() == Shape{2}, "Imag view shape should match");

    // Imag view is strided (stride=8, itemsize=4), so use item() for proper
    // access
    ASSERT(std::abs(imag_view.item<float>({0}) - 2.0f) < 1e-5f,
           "First imag should be 2.0");
    ASSERT(std::abs(imag_view.item<float>({1}) - 4.0f) < 1e-5f,
           "Second imag should be 4.0");
}

void test_real_imag_2d() {
    std::vector<complex64_t> data = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}};
    auto t = Tensor::from_data(data.data(), {2, 2});

    auto real_view = t.real();
    auto imag_view = t.imag();

    ASSERT(real_view.shape() == Shape({2, 2}), "Real view shape should be 2x2");
    ASSERT(imag_view.shape() == Shape({2, 2}), "Imag view shape should be 2x2");
}

// ============================================================================
// Complex Operation Legality Tests
// ============================================================================

// ============================================================================
// Complex Arithmetic Tests
// ============================================================================

void test_complex_add() {
    std::vector<complex64_t> a_data = {{1.0f, 1.0f}, {2.0f, 2.0f}};
    std::vector<complex64_t> b_data = {{1.0f, 0.0f}, {0.0f, 1.0f}};

    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::add(a, b);

    ASSERT(result.dtype() == DType::Complex64, "Result should be Complex64");
    const complex64_t *data = result.typed_data<complex64_t>();
    // (1+i) + (1+0i) = (2+i)
    ASSERT(std::abs(data[0].real() - 2.0f) < 1e-5f, "Add real part 0");
    ASSERT(std::abs(data[0].imag() - 1.0f) < 1e-5f, "Add imag part 0");
    // (2+2i) + (0+i) = (2+3i)
    ASSERT(std::abs(data[1].real() - 2.0f) < 1e-5f, "Add real part 1");
    ASSERT(std::abs(data[1].imag() - 3.0f) < 1e-5f, "Add imag part 1");
}

void test_complex_subtract() {
    std::vector<complex64_t> a_data = {{3.0f, 4.0f}, {1.0f, 2.0f}};
    std::vector<complex64_t> b_data = {{1.0f, 1.0f}, {1.0f, 2.0f}};

    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::subtract(a, b);

    const complex64_t *data = result.typed_data<complex64_t>();
    // (3+4i) - (1+i) = (2+3i)
    ASSERT(std::abs(data[0].real() - 2.0f) < 1e-5f, "Subtract real part 0");
    ASSERT(std::abs(data[0].imag() - 3.0f) < 1e-5f, "Subtract imag part 0");
    // (1+2i) - (1+2i) = (0+0i)
    ASSERT(std::abs(data[1].real()) < 1e-5f, "Subtract real part 1");
    ASSERT(std::abs(data[1].imag()) < 1e-5f, "Subtract imag part 1");
}

void test_complex_multiply() {
    std::vector<complex64_t> a_data = {{1.0f, 2.0f}, {0.0f, 1.0f}};
    std::vector<complex64_t> b_data = {{3.0f, 4.0f}, {0.0f, 1.0f}};

    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::multiply(a, b);

    const complex64_t *data = result.typed_data<complex64_t>();
    // (1+2i) * (3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
    ASSERT(std::abs(data[0].real() - (-5.0f)) < 1e-5f, "Multiply real part 0");
    ASSERT(std::abs(data[0].imag() - 10.0f) < 1e-5f, "Multiply imag part 0");
    // (0+i) * (0+i) = i² = -1
    ASSERT(std::abs(data[1].real() - (-1.0f)) < 1e-5f, "Multiply real part 1");
    ASSERT(std::abs(data[1].imag()) < 1e-5f, "Multiply imag part 1");
}

void test_complex_divide() {
    std::vector<complex64_t> a_data = {{4.0f, 2.0f}};
    std::vector<complex64_t> b_data = {{2.0f, 0.0f}};

    auto a = Tensor::from_data(a_data.data(), {1});
    auto b = Tensor::from_data(b_data.data(), {1});

    auto result = ops::divide(a, b);

    const complex64_t *data = result.typed_data<complex64_t>();
    // (4+2i) / 2 = (2+i)
    ASSERT(std::abs(data[0].real() - 2.0f) < 1e-5f, "Divide real part");
    ASSERT(std::abs(data[0].imag() - 1.0f) < 1e-5f, "Divide imag part");
}

void test_complex_negate() {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {-3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::negate(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    ASSERT(std::abs(res_data[0].real() - (-1.0f)) < 1e-5f, "Negate real 0");
    ASSERT(std::abs(res_data[0].imag() - (-2.0f)) < 1e-5f, "Negate imag 0");
    ASSERT(std::abs(res_data[1].real() - 3.0f) < 1e-5f, "Negate real 1");
    ASSERT(std::abs(res_data[1].imag() - (-4.0f)) < 1e-5f, "Negate imag 1");
}

void test_complex_abs() {
    // abs of complex returns magnitude
    std::vector<complex64_t> data = {{3.0f, 4.0f}, {0.0f, 5.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::abs(t);

    ASSERT(result.dtype() == DType::Float32, "Abs should return Float32");
    const float *res_data = result.typed_data<float>();
    // |3+4i| = 5
    ASSERT(std::abs(res_data[0] - 5.0f) < 1e-5f, "Abs of 3+4i = 5");
    // |0+5i| = 5
    ASSERT(std::abs(res_data[1] - 5.0f) < 1e-5f, "Abs of 0+5i = 5");
}

void test_complex_exp() {
    // exp(i*pi) = -1
    std::vector<complex64_t> data = {{0.0f, static_cast<float>(M_PI)}};
    auto t = Tensor::from_data(data.data(), {1});

    auto result = ops::exp(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // exp(i*pi) ≈ -1 + 0i
    ASSERT(std::abs(res_data[0].real() - (-1.0f)) < 1e-5f,
           "exp(i*pi) real ≈ -1");
    ASSERT(std::abs(res_data[0].imag()) < 1e-5f, "exp(i*pi) imag ≈ 0");
}

void test_complex_log() {
    // log(-1) = i*pi
    std::vector<complex64_t> data = {{-1.0f, 0.0f}};
    auto t = Tensor::from_data(data.data(), {1});

    auto result = ops::log(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // log(-1) = i*pi
    ASSERT(std::abs(res_data[0].real()) < 1e-5f, "log(-1) real ≈ 0");
    ASSERT(std::abs(res_data[0].imag() - static_cast<float>(M_PI)) < 1e-5f,
           "log(-1) imag ≈ pi");
}

void test_complex_sqrt() {
    // sqrt(-1) = i
    std::vector<complex64_t> data = {{-1.0f, 0.0f}};
    auto t = Tensor::from_data(data.data(), {1});

    auto result = ops::sqrt(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // sqrt(-1) = i
    ASSERT(std::abs(res_data[0].real()) < 1e-5f, "sqrt(-1) real ≈ 0");
    ASSERT(std::abs(res_data[0].imag() - 1.0f) < 1e-5f, "sqrt(-1) imag ≈ 1");
}

void test_complex_conj() {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {-3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::conj(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // conj(1+2i) = 1-2i
    ASSERT(std::abs(res_data[0].real() - 1.0f) < 1e-5f, "Conj real 0");
    ASSERT(std::abs(res_data[0].imag() - (-2.0f)) < 1e-5f, "Conj imag 0");
    // conj(-3+4i) = -3-4i
    ASSERT(std::abs(res_data[1].real() - (-3.0f)) < 1e-5f, "Conj real 1");
    ASSERT(std::abs(res_data[1].imag() - (-4.0f)) < 1e-5f, "Conj imag 1");
}

// ============================================================================
// Complex Reduction Tests
// ============================================================================

void test_complex_sum() {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::sum(t, {}, false);

    ASSERT(result.dtype() == DType::Complex64, "Sum should preserve Complex64");
    const complex64_t *res_data = result.typed_data<complex64_t>();
    // (1+2i) + (3+4i) + (5+6i) = (9+12i)
    ASSERT(std::abs(res_data[0].real() - 9.0f) < 1e-5f, "Sum real");
    ASSERT(std::abs(res_data[0].imag() - 12.0f) < 1e-5f, "Sum imag");
}

void test_complex_mean() {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::mean(t, {}, false);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // mean = (1+2i + 3+4i) / 2 = (2+3i)
    ASSERT(std::abs(res_data[0].real() - 2.0f) < 1e-5f, "Mean real");
    ASSERT(std::abs(res_data[0].imag() - 3.0f) < 1e-5f, "Mean imag");
}

// ============================================================================
// Complex MatMul Tests
// ============================================================================

void test_complex_matmul() {
    // Simple 2x2 complex matmul
    std::vector<complex64_t> a_data = {
        {1.0f, 0.0f},
        {0.0f, 1.0f}, // [[1, i],
        {0.0f, 0.0f},
        {1.0f, 0.0f} //  [0, 1]]
    };
    std::vector<complex64_t> b_data = {
        {1.0f, 0.0f},
        {0.0f, 0.0f}, // [[1, 0],
        {0.0f, 0.0f},
        {1.0f, 0.0f} //  [0, 1]]
    };
    auto a = Tensor::from_data(a_data.data(), {2, 2});
    auto b = Tensor::from_data(b_data.data(), {2, 2});

    auto result = ops::matmul(a, b);

    ASSERT(result.shape() == Shape({2, 2}), "MatMul shape");
    ASSERT(result.dtype() == DType::Complex64, "MatMul dtype");

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // [[1, i], [0, 1]] @ [[1, 0], [0, 1]] = [[1, i], [0, 1]]
    ASSERT(std::abs(res_data[0].real() - 1.0f) < 1e-5f, "(0,0) real");
    ASSERT(std::abs(res_data[0].imag()) < 1e-5f, "(0,0) imag");
    ASSERT(std::abs(res_data[1].real()) < 1e-5f, "(0,1) real");
    ASSERT(std::abs(res_data[1].imag() - 1.0f) < 1e-5f, "(0,1) imag");
}

// ============================================================================
// Complex Illegal Operations Tests
// ============================================================================

void test_complex_illegal_op_throws() {
    // Test that illegal operations on complex types throw errors
    std::vector<complex64_t> data = {{1.0f, 1.0f}};
    auto t = Tensor::from_data(data.data(), {1});

    bool threw = false;
    try {
        // Maximum is not allowed for complex types
        // This should throw TypeError
        (void)t.max();
    } catch (const TypeError &) {
        threw = true;
    } catch (const std::exception &) {
        // Other exceptions might be thrown
        threw = true;
    }

    ASSERT(threw, "max() on complex should throw");
}

void test_complex_comparison_throws() {
    std::vector<complex64_t> a_data = {{1.0f, 1.0f}};
    std::vector<complex64_t> b_data = {{2.0f, 0.0f}};
    auto a = Tensor::from_data(a_data.data(), {1});
    auto b = Tensor::from_data(b_data.data(), {1});

    bool threw = false;
    try {
        // Less comparison is not allowed for complex types
        (void)ops::less(a, b);
    } catch (const TypeError &) {
        threw = true;
    } catch (const std::exception &) {
        threw = true;
    }

    ASSERT(threw, "less() on complex should throw");
}

void test_real_on_non_complex_throws() {
    auto t = Tensor::ones({2}, DType::Float32);

    bool threw = false;
    try {
        auto real_view = t.real();
    } catch (const TypeError &) {
        threw = true;
    }

    ASSERT(threw, "real() on non-complex should throw TypeError");
}

void test_imag_on_non_complex_throws() {
    auto t = Tensor::ones({2}, DType::Float32);

    bool threw = false;
    try {
        auto imag_view = t.imag();
    } catch (const TypeError &) {
        threw = true;
    }

    ASSERT(threw, "imag() on non-complex should throw TypeError");
}

int main() {
    // Initialize operations registry
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== Complex Tensor Tests ===" << std::endl << std::endl;

    // Creation tests
    std::cout << "--- Creation Tests ---" << std::endl;
    RUN_TEST(test_complex64_creation);
    RUN_TEST(test_complex128_creation);

    // View tests
    std::cout << "--- Real/Imag View Tests ---" << std::endl;
    RUN_TEST(test_real_view);
    RUN_TEST(test_imag_view);
    RUN_TEST(test_real_imag_2d);

    // Arithmetic tests
    std::cout << "--- Arithmetic Tests ---" << std::endl;
    RUN_TEST(test_complex_add);
    RUN_TEST(test_complex_subtract);
    RUN_TEST(test_complex_multiply);
    RUN_TEST(test_complex_divide);
    RUN_TEST(test_complex_negate);
    RUN_TEST(test_complex_abs);
    RUN_TEST(test_complex_conj);

    // Special functions
    std::cout << "--- Special Function Tests ---" << std::endl;
    RUN_TEST(test_complex_exp);
    RUN_TEST(test_complex_log);
    RUN_TEST(test_complex_sqrt);

    // Reduction tests
    std::cout << "--- Reduction Tests ---" << std::endl;
    RUN_TEST(test_complex_sum);
    RUN_TEST(test_complex_mean);

    // MatMul tests
    std::cout << "--- MatMul Tests ---" << std::endl;
    RUN_TEST(test_complex_matmul);

    // Illegal operation tests
    std::cout << "--- Illegal Operation Tests ---" << std::endl;
    RUN_TEST(test_complex_illegal_op_throws);
    RUN_TEST(test_complex_comparison_throws);
    RUN_TEST(test_real_on_non_complex_throws);
    RUN_TEST(test_imag_on_non_complex_throws);

    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
