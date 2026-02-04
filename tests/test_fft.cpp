// Tests for FFT (Fast Fourier Transform) operations

#include <axiom/axiom.hpp>
#include <cmath>
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

void test_fft_roundtrip() {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::fft(x);
    auto x_back = axiom::fft::ifft(X);

    // Take real part since original was real
    auto x_real = axiom::ops::real(x_back);

    // Use looser tolerance for FFT roundtrip
    ASSERT(x_real.allclose(x, 1e-4, 1e-4),
           "FFT roundtrip should recover original");
}

void test_fft_roundtrip_larger() {
    auto x = axiom::Tensor::randn({64});

    auto X = axiom::fft::fft(x);
    auto x_back = axiom::fft::ifft(X);
    auto x_real = axiom::ops::real(x_back);

    ASSERT(x_real.allclose(x, 1e-4, 1e-4),
           "FFT roundtrip (n=64) should recover original");
}

void test_fft_output_shape() {
    auto x = axiom::Tensor::randn({16});

    auto X = axiom::fft::fft(x);
    ASSERT(X.shape() == axiom::Shape({16}),
           "FFT output shape should match input");
    ASSERT(X.dtype() == axiom::DType::Complex64,
           "FFT output should be Complex64");
}

void test_fft_norm_ortho() {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::fft(x, -1, -1, "ortho");
    auto x_back = axiom::fft::ifft(X, -1, -1, "ortho");
    auto x_real = axiom::ops::real(x_back);

    ASSERT(x_real.allclose(x, 1e-4, 1e-4),
           "FFT ortho norm roundtrip should recover original");
}

void test_rfft_shape() {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::rfft(x);
    // rfft output size is n/2 + 1 = 5
    ASSERT(X.shape() == axiom::Shape({5}), "RFFT output shape should be n/2+1");
    ASSERT(X.dtype() == axiom::DType::Complex64,
           "RFFT output should be Complex64");
}

void test_rfft_output_dtype() {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::rfft(x);
    ASSERT(X.dtype() == axiom::DType::Complex64,
           "RFFT output should be Complex64");
}

void test_fftshift_even() {
    auto x = axiom::Tensor::arange(6);
    auto shifted = axiom::fft::fftshift(x);

    // [0,1,2,3,4,5] -> [3,4,5,0,1,2]
    ASSERT(shifted.item<int32_t>({0}) == 3, "fftshift wrong at 0");
    ASSERT(shifted.item<int32_t>({1}) == 4, "fftshift wrong at 1");
    ASSERT(shifted.item<int32_t>({2}) == 5, "fftshift wrong at 2");
    ASSERT(shifted.item<int32_t>({3}) == 0, "fftshift wrong at 3");
    ASSERT(shifted.item<int32_t>({4}) == 1, "fftshift wrong at 4");
    ASSERT(shifted.item<int32_t>({5}) == 2, "fftshift wrong at 5");
}

void test_fftshift_odd() {
    auto x = axiom::Tensor::arange(5);
    auto shifted = axiom::fft::fftshift(x);

    // [0,1,2,3,4] -> [3,4,0,1,2] (shift right by n//2 = 2)
    ASSERT(shifted.item<int32_t>({0}) == 3, "fftshift odd wrong at 0");
    ASSERT(shifted.item<int32_t>({1}) == 4, "fftshift odd wrong at 1");
    ASSERT(shifted.item<int32_t>({2}) == 0, "fftshift odd wrong at 2");
}

void test_ifftshift_roundtrip() {
    auto x = axiom::Tensor::arange(8);
    auto shifted = axiom::fft::fftshift(x);
    auto unshifted = axiom::fft::ifftshift(shifted);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT(unshifted.item<int32_t>({i}) == static_cast<int32_t>(i),
               "ifftshift should reverse fftshift");
    }
}

void test_fftfreq() {
    auto freqs = axiom::fft::fftfreq(4, 1.0);

    // For n=4, d=1: frequencies at DC and positive, then negative
    ASSERT(freqs.shape() == axiom::Shape({4}), "fftfreq shape wrong");
    ASSERT(std::abs(freqs.item<double>({0}) - 0.0) < 1e-10, "fftfreq[0] wrong");
    ASSERT(std::abs(freqs.item<double>({1}) - 0.25) < 1e-10,
           "fftfreq[1] wrong");
    // Last element should be negative
    ASSERT(freqs.item<double>({3}) < 0, "fftfreq[3] should be negative");
}

void test_fftfreq_scaled() {
    // With d=0.5, frequencies should be doubled (factor = 1/(d*n) = 1/2)
    auto freqs = axiom::fft::fftfreq(4, 0.5);

    ASSERT(std::abs(freqs.item<double>({0}) - 0.0) < 1e-10,
           "fftfreq scaled [0] wrong");
    ASSERT(std::abs(freqs.item<double>({1}) - 0.5) < 1e-10,
           "fftfreq scaled [1] wrong");
    // Last element should be negative
    ASSERT(freqs.item<double>({3}) < 0,
           "fftfreq scaled [3] should be negative");
}

void test_rfftfreq() {
    auto freqs = axiom::fft::rfftfreq(8, 1.0);

    // For n=8: output has length 5, frequencies [0, 0.125, 0.25, 0.375, 0.5]
    ASSERT(freqs.shape() == axiom::Shape({5}), "rfftfreq shape wrong");
    ASSERT(std::abs(freqs.item<double>({0}) - 0.0) < 1e-10,
           "rfftfreq[0] wrong");
    ASSERT(std::abs(freqs.item<double>({1}) - 0.125) < 1e-10,
           "rfftfreq[1] wrong");
    ASSERT(std::abs(freqs.item<double>({4}) - 0.5) < 1e-10,
           "rfftfreq[4] wrong");
}

void test_fft2_shape() {
    auto x = axiom::Tensor::randn({4, 4});

    auto X = axiom::fft::fft2(x);
    ASSERT(X.shape() == axiom::Shape({4, 4}),
           "FFT2 output shape should match input");
    ASSERT(X.dtype() == axiom::DType::Complex64,
           "FFT2 output should be Complex64");
}

void test_fft2_roundtrip() {
    auto x = axiom::Tensor::randn({4, 4});

    auto X = axiom::fft::fft2(x);
    auto x_back = axiom::fft::ifft2(X);
    auto x_real = axiom::ops::real(x_back);

    ASSERT(x_real.allclose(x, 1e-4, 1e-4),
           "FFT2 roundtrip should recover original");
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   FFT Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_fft_roundtrip);
    RUN_TEST(test_fft_roundtrip_larger);
    RUN_TEST(test_fft_output_shape);
    RUN_TEST(test_fft_norm_ortho);
    RUN_TEST(test_rfft_shape);
    RUN_TEST(test_rfft_output_dtype);
    RUN_TEST(test_fftshift_even);
    RUN_TEST(test_fftshift_odd);
    RUN_TEST(test_ifftshift_roundtrip);
    RUN_TEST(test_fftfreq);
    RUN_TEST(test_fftfreq_scaled);
    RUN_TEST(test_rfftfreq);
    RUN_TEST(test_fft2_shape);
    RUN_TEST(test_fft2_roundtrip);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
