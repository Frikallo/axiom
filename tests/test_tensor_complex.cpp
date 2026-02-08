#include "axiom_test_utils.hpp"
#include <cmath>
#include <complex>
#include <vector>

using namespace axiom;

// ============================================================================
// Complex Tensor Creation Tests
// ============================================================================

TEST(TensorComplex, Complex64Creation) {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    ASSERT_TRUE(t.dtype() == DType::Complex64) << "Should be Complex64";
    ASSERT_TRUE(t.size() == 2) << "Should have 2 elements";
    ASSERT_TRUE(t.itemsize() == 8) << "Complex64 should be 8 bytes";
}

TEST(TensorComplex, Complex128Creation) {
    std::vector<complex128_t> data = {{1.0, 2.0}, {3.0, 4.0}};
    auto t = Tensor::from_data(data.data(), {2});

    ASSERT_TRUE(t.dtype() == DType::Complex128) << "Should be Complex128";
    ASSERT_TRUE(t.size() == 2) << "Should have 2 elements";
    ASSERT_TRUE(t.itemsize() == 16) << "Complex128 should be 16 bytes";
}

// ============================================================================
// Real/Imag View Tests
// ============================================================================

TEST(TensorComplex, RealView) {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto real_view = t.real();

    ASSERT_TRUE(real_view.dtype() == DType::Float32)
        << "Real view should be Float32";
    ASSERT_TRUE(real_view.shape() == Shape{2})
        << "Real view shape should match";

    // Real view is strided (stride=8, itemsize=4), so use item() for proper
    // access
    ASSERT_TRUE(std::abs(real_view.item<float>({0}) - 1.0f) < 1e-5f)
        << "First real should be 1.0";
    ASSERT_TRUE(std::abs(real_view.item<float>({1}) - 3.0f) < 1e-5f)
        << "Second real should be 3.0";
}

TEST(TensorComplex, ImagView) {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto imag_view = t.imag();

    ASSERT_TRUE(imag_view.dtype() == DType::Float32)
        << "Imag view should be Float32";
    ASSERT_TRUE(imag_view.shape() == Shape{2})
        << "Imag view shape should match";

    // Imag view is strided (stride=8, itemsize=4), so use item() for proper
    // access
    ASSERT_TRUE(std::abs(imag_view.item<float>({0}) - 2.0f) < 1e-5f)
        << "First imag should be 2.0";
    ASSERT_TRUE(std::abs(imag_view.item<float>({1}) - 4.0f) < 1e-5f)
        << "Second imag should be 4.0";
}

TEST(TensorComplex, RealImag2d) {
    std::vector<complex64_t> data = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}};
    auto t = Tensor::from_data(data.data(), {2, 2});

    auto real_view = t.real();
    auto imag_view = t.imag();

    ASSERT_TRUE(real_view.shape() == Shape({2, 2}))
        << "Real view shape should be 2x2";
    ASSERT_TRUE(imag_view.shape() == Shape({2, 2}))
        << "Imag view shape should be 2x2";
}

// ============================================================================
// Complex Arithmetic Tests
// ============================================================================

TEST(TensorComplex, Add) {
    std::vector<complex64_t> a_data = {{1.0f, 1.0f}, {2.0f, 2.0f}};
    std::vector<complex64_t> b_data = {{1.0f, 0.0f}, {0.0f, 1.0f}};

    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::add(a, b);

    ASSERT_TRUE(result.dtype() == DType::Complex64)
        << "Result should be Complex64";
    const complex64_t *data = result.typed_data<complex64_t>();
    // (1+i) + (1+0i) = (2+i)
    ASSERT_TRUE(std::abs(data[0].real() - 2.0f) < 1e-5f) << "Add real part 0";
    ASSERT_TRUE(std::abs(data[0].imag() - 1.0f) < 1e-5f) << "Add imag part 0";
    // (2+2i) + (0+i) = (2+3i)
    ASSERT_TRUE(std::abs(data[1].real() - 2.0f) < 1e-5f) << "Add real part 1";
    ASSERT_TRUE(std::abs(data[1].imag() - 3.0f) < 1e-5f) << "Add imag part 1";
}

TEST(TensorComplex, Subtract) {
    std::vector<complex64_t> a_data = {{3.0f, 4.0f}, {1.0f, 2.0f}};
    std::vector<complex64_t> b_data = {{1.0f, 1.0f}, {1.0f, 2.0f}};

    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::subtract(a, b);

    const complex64_t *data = result.typed_data<complex64_t>();
    // (3+4i) - (1+i) = (2+3i)
    ASSERT_TRUE(std::abs(data[0].real() - 2.0f) < 1e-5f)
        << "Subtract real part 0";
    ASSERT_TRUE(std::abs(data[0].imag() - 3.0f) < 1e-5f)
        << "Subtract imag part 0";
    // (1+2i) - (1+2i) = (0+0i)
    ASSERT_TRUE(std::abs(data[1].real()) < 1e-5f) << "Subtract real part 1";
    ASSERT_TRUE(std::abs(data[1].imag()) < 1e-5f) << "Subtract imag part 1";
}

TEST(TensorComplex, Multiply) {
    std::vector<complex64_t> a_data = {{1.0f, 2.0f}, {0.0f, 1.0f}};
    std::vector<complex64_t> b_data = {{3.0f, 4.0f}, {0.0f, 1.0f}};

    auto a = Tensor::from_data(a_data.data(), {2});
    auto b = Tensor::from_data(b_data.data(), {2});

    auto result = ops::multiply(a, b);

    const complex64_t *data = result.typed_data<complex64_t>();
    // (1+2i) * (3+4i) = 3 + 4i + 6i + 8i^2 = 3 + 10i - 8 = -5 + 10i
    ASSERT_TRUE(std::abs(data[0].real() - (-5.0f)) < 1e-5f)
        << "Multiply real part 0";
    ASSERT_TRUE(std::abs(data[0].imag() - 10.0f) < 1e-5f)
        << "Multiply imag part 0";
    // (0+i) * (0+i) = i^2 = -1
    ASSERT_TRUE(std::abs(data[1].real() - (-1.0f)) < 1e-5f)
        << "Multiply real part 1";
    ASSERT_TRUE(std::abs(data[1].imag()) < 1e-5f) << "Multiply imag part 1";
}

TEST(TensorComplex, Divide) {
    std::vector<complex64_t> a_data = {{4.0f, 2.0f}};
    std::vector<complex64_t> b_data = {{2.0f, 0.0f}};

    auto a = Tensor::from_data(a_data.data(), {1});
    auto b = Tensor::from_data(b_data.data(), {1});

    auto result = ops::divide(a, b);

    const complex64_t *data = result.typed_data<complex64_t>();
    // (4+2i) / 2 = (2+i)
    ASSERT_TRUE(std::abs(data[0].real() - 2.0f) < 1e-5f) << "Divide real part";
    ASSERT_TRUE(std::abs(data[0].imag() - 1.0f) < 1e-5f) << "Divide imag part";
}

TEST(TensorComplex, Negate) {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {-3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::negate(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    ASSERT_TRUE(std::abs(res_data[0].real() - (-1.0f)) < 1e-5f)
        << "Negate real 0";
    ASSERT_TRUE(std::abs(res_data[0].imag() - (-2.0f)) < 1e-5f)
        << "Negate imag 0";
    ASSERT_TRUE(std::abs(res_data[1].real() - 3.0f) < 1e-5f) << "Negate real 1";
    ASSERT_TRUE(std::abs(res_data[1].imag() - (-4.0f)) < 1e-5f)
        << "Negate imag 1";
}

TEST(TensorComplex, Abs) {
    // abs of complex returns magnitude
    std::vector<complex64_t> data = {{3.0f, 4.0f}, {0.0f, 5.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::abs(t);

    ASSERT_TRUE(result.dtype() == DType::Float32)
        << "Abs should return Float32";
    const float *res_data = result.typed_data<float>();
    // |3+4i| = 5
    ASSERT_TRUE(std::abs(res_data[0] - 5.0f) < 1e-5f) << "Abs of 3+4i = 5";
    // |0+5i| = 5
    ASSERT_TRUE(std::abs(res_data[1] - 5.0f) < 1e-5f) << "Abs of 0+5i = 5";
}

TEST(TensorComplex, Exp) {
    // exp(i*pi) = -1
    std::vector<complex64_t> data = {{0.0f, static_cast<float>(M_PI)}};
    auto t = Tensor::from_data(data.data(), {1});

    auto result = ops::exp(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // exp(i*pi) ~ -1 + 0i
    ASSERT_TRUE(std::abs(res_data[0].real() - (-1.0f)) < 1e-5f)
        << "exp(i*pi) real ~ -1";
    ASSERT_TRUE(std::abs(res_data[0].imag()) < 1e-5f) << "exp(i*pi) imag ~ 0";
}

TEST(TensorComplex, Log) {
    // log(-1) = i*pi
    std::vector<complex64_t> data = {{-1.0f, 0.0f}};
    auto t = Tensor::from_data(data.data(), {1});

    auto result = ops::log(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // log(-1) = i*pi
    ASSERT_TRUE(std::abs(res_data[0].real()) < 1e-5f) << "log(-1) real ~ 0";
    ASSERT_TRUE(std::abs(res_data[0].imag() - static_cast<float>(M_PI)) < 1e-5f)
        << "log(-1) imag ~ pi";
}

TEST(TensorComplex, Sqrt) {
    // sqrt(-1) = i
    std::vector<complex64_t> data = {{-1.0f, 0.0f}};
    auto t = Tensor::from_data(data.data(), {1});

    auto result = ops::sqrt(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // sqrt(-1) = i
    ASSERT_TRUE(std::abs(res_data[0].real()) < 1e-5f) << "sqrt(-1) real ~ 0";
    ASSERT_TRUE(std::abs(res_data[0].imag() - 1.0f) < 1e-5f)
        << "sqrt(-1) imag ~ 1";
}

TEST(TensorComplex, Conj) {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {-3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::conj(t);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // conj(1+2i) = 1-2i
    ASSERT_TRUE(std::abs(res_data[0].real() - 1.0f) < 1e-5f) << "Conj real 0";
    ASSERT_TRUE(std::abs(res_data[0].imag() - (-2.0f)) < 1e-5f)
        << "Conj imag 0";
    // conj(-3+4i) = -3-4i
    ASSERT_TRUE(std::abs(res_data[1].real() - (-3.0f)) < 1e-5f)
        << "Conj real 1";
    ASSERT_TRUE(std::abs(res_data[1].imag() - (-4.0f)) < 1e-5f)
        << "Conj imag 1";
}

// ============================================================================
// Complex Reduction Tests
// ============================================================================

TEST(TensorComplex, Sum) {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    auto t = Tensor::from_data(data.data(), {3});

    auto result = ops::sum(t, {}, false);

    ASSERT_TRUE(result.dtype() == DType::Complex64)
        << "Sum should preserve Complex64";
    const complex64_t *res_data = result.typed_data<complex64_t>();
    // (1+2i) + (3+4i) + (5+6i) = (9+12i)
    ASSERT_TRUE(std::abs(res_data[0].real() - 9.0f) < 1e-5f) << "Sum real";
    ASSERT_TRUE(std::abs(res_data[0].imag() - 12.0f) < 1e-5f) << "Sum imag";
}

TEST(TensorComplex, Mean) {
    std::vector<complex64_t> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto t = Tensor::from_data(data.data(), {2});

    auto result = ops::mean(t, {}, false);

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // mean = (1+2i + 3+4i) / 2 = (2+3i)
    ASSERT_TRUE(std::abs(res_data[0].real() - 2.0f) < 1e-5f) << "Mean real";
    ASSERT_TRUE(std::abs(res_data[0].imag() - 3.0f) < 1e-5f) << "Mean imag";
}

// ============================================================================
// Complex MatMul Tests
// ============================================================================

TEST(TensorComplex, MatMul) {
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

    ASSERT_TRUE(result.shape() == Shape({2, 2})) << "MatMul shape";
    ASSERT_TRUE(result.dtype() == DType::Complex64) << "MatMul dtype";

    const complex64_t *res_data = result.typed_data<complex64_t>();
    // [[1, i], [0, 1]] @ [[1, 0], [0, 1]] = [[1, i], [0, 1]]
    ASSERT_TRUE(std::abs(res_data[0].real() - 1.0f) < 1e-5f) << "(0,0) real";
    ASSERT_TRUE(std::abs(res_data[0].imag()) < 1e-5f) << "(0,0) imag";
    ASSERT_TRUE(std::abs(res_data[1].real()) < 1e-5f) << "(0,1) real";
    ASSERT_TRUE(std::abs(res_data[1].imag() - 1.0f) < 1e-5f) << "(0,1) imag";
}

// ============================================================================
// Complex Illegal Operations Tests
// ============================================================================

TEST(TensorComplex, IllegalOpThrows) {
    // Test that illegal operations on complex types throw errors
    std::vector<complex64_t> data = {{1.0f, 1.0f}};
    auto t = Tensor::from_data(data.data(), {1});

    // Maximum is not allowed for complex types
    EXPECT_THROW(t.max(), std::exception);
}

TEST(TensorComplex, ComparisonThrows) {
    std::vector<complex64_t> a_data = {{1.0f, 1.0f}};
    std::vector<complex64_t> b_data = {{2.0f, 0.0f}};
    auto a = Tensor::from_data(a_data.data(), {1});
    auto b = Tensor::from_data(b_data.data(), {1});

    // Less comparison is not allowed for complex types
    EXPECT_THROW(ops::less(a, b), std::exception);
}

TEST(TensorComplex, RealOnNonComplexThrows) {
    auto t = Tensor::ones({2}, DType::Float32);

    EXPECT_THROW(t.real(), TypeError);
}

TEST(TensorComplex, ImagOnNonComplexThrows) {
    auto t = Tensor::ones({2}, DType::Float32);

    EXPECT_THROW(t.imag(), TypeError);
}
