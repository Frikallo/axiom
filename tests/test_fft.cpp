#include "axiom_test_utils.hpp"

TEST(FFT, FftRoundtrip) {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::fft(x);
    auto x_back = axiom::fft::ifft(X);

    // Take real part since original was real
    auto x_real = axiom::ops::real(x_back);

    // Use looser tolerance for FFT roundtrip
    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT roundtrip should recover original";
}

TEST(FFT, FftRoundtripLarger) {
    auto x = axiom::Tensor::randn({64});

    auto X = axiom::fft::fft(x);
    auto x_back = axiom::fft::ifft(X);
    auto x_real = axiom::ops::real(x_back);

    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT roundtrip (n=64) should recover original";
}

TEST(FFT, FftOutputShape) {
    auto x = axiom::Tensor::randn({16});

    auto X = axiom::fft::fft(x);
    ASSERT_TRUE(X.shape() == axiom::Shape({16}))
        << "FFT output shape should match input";
    ASSERT_TRUE(X.dtype() == axiom::DType::Complex64)
        << "FFT output should be Complex64";
}

TEST(FFT, FftNormOrtho) {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::fft(x, -1, -1, "ortho");
    auto x_back = axiom::fft::ifft(X, -1, -1, "ortho");
    auto x_real = axiom::ops::real(x_back);

    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT ortho norm roundtrip should recover original";
}

TEST(FFT, RfftShape) {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::rfft(x);
    // rfft output size is n/2 + 1 = 5
    ASSERT_TRUE(X.shape() == axiom::Shape({5}))
        << "RFFT output shape should be n/2+1";
    ASSERT_TRUE(X.dtype() == axiom::DType::Complex64)
        << "RFFT output should be Complex64";
}

TEST(FFT, RfftOutputDtype) {
    auto x = axiom::Tensor::randn({8});

    auto X = axiom::fft::rfft(x);
    ASSERT_TRUE(X.dtype() == axiom::DType::Complex64)
        << "RFFT output should be Complex64";
}

TEST(FFT, FftshiftEven) {
    auto x = axiom::Tensor::arange(6);
    auto shifted = axiom::fft::fftshift(x);

    // [0,1,2,3,4,5] -> [3,4,5,0,1,2]
    ASSERT_TRUE(shifted.item<int32_t>({0}) == 3) << "fftshift wrong at 0";
    ASSERT_TRUE(shifted.item<int32_t>({1}) == 4) << "fftshift wrong at 1";
    ASSERT_TRUE(shifted.item<int32_t>({2}) == 5) << "fftshift wrong at 2";
    ASSERT_TRUE(shifted.item<int32_t>({3}) == 0) << "fftshift wrong at 3";
    ASSERT_TRUE(shifted.item<int32_t>({4}) == 1) << "fftshift wrong at 4";
    ASSERT_TRUE(shifted.item<int32_t>({5}) == 2) << "fftshift wrong at 5";
}

TEST(FFT, FftshiftOdd) {
    auto x = axiom::Tensor::arange(5);
    auto shifted = axiom::fft::fftshift(x);

    // [0,1,2,3,4] -> [3,4,0,1,2] (shift right by n//2 = 2)
    ASSERT_TRUE(shifted.item<int32_t>({0}) == 3) << "fftshift odd wrong at 0";
    ASSERT_TRUE(shifted.item<int32_t>({1}) == 4) << "fftshift odd wrong at 1";
    ASSERT_TRUE(shifted.item<int32_t>({2}) == 0) << "fftshift odd wrong at 2";
}

TEST(FFT, IfftshiftRoundtrip) {
    auto x = axiom::Tensor::arange(8);
    auto shifted = axiom::fft::fftshift(x);
    auto unshifted = axiom::fft::ifftshift(shifted);

    for (size_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(unshifted.item<int32_t>({i}) == static_cast<int32_t>(i))
            << "ifftshift should reverse fftshift";
    }
}

TEST(FFT, Fftfreq) {
    auto freqs = axiom::fft::fftfreq(4, 1.0);

    // For n=4, d=1: frequencies at DC and positive, then negative
    ASSERT_TRUE(freqs.shape() == axiom::Shape({4})) << "fftfreq shape wrong";
    ASSERT_TRUE(std::abs(freqs.item<double>({0}) - 0.0) < 1e-10)
        << "fftfreq[0] wrong";
    ASSERT_TRUE(std::abs(freqs.item<double>({1}) - 0.25) < 1e-10)
        << "fftfreq[1] wrong";
    // Last element should be negative
    ASSERT_TRUE(freqs.item<double>({3}) < 0) << "fftfreq[3] should be negative";
}

TEST(FFT, FftfreqScaled) {
    // With d=0.5, frequencies should be doubled (factor = 1/(d*n) = 1/2)
    auto freqs = axiom::fft::fftfreq(4, 0.5);

    ASSERT_TRUE(std::abs(freqs.item<double>({0}) - 0.0) < 1e-10)
        << "fftfreq scaled [0] wrong";
    ASSERT_TRUE(std::abs(freqs.item<double>({1}) - 0.5) < 1e-10)
        << "fftfreq scaled [1] wrong";
    // Last element should be negative
    ASSERT_TRUE(freqs.item<double>({3}) < 0)
        << "fftfreq scaled [3] should be negative";
}

TEST(FFT, Rfftfreq) {
    auto freqs = axiom::fft::rfftfreq(8, 1.0);

    // For n=8: output has length 5, frequencies [0, 0.125, 0.25, 0.375, 0.5]
    ASSERT_TRUE(freqs.shape() == axiom::Shape({5})) << "rfftfreq shape wrong";
    ASSERT_TRUE(std::abs(freqs.item<double>({0}) - 0.0) < 1e-10)
        << "rfftfreq[0] wrong";
    ASSERT_TRUE(std::abs(freqs.item<double>({1}) - 0.125) < 1e-10)
        << "rfftfreq[1] wrong";
    ASSERT_TRUE(std::abs(freqs.item<double>({4}) - 0.5) < 1e-10)
        << "rfftfreq[4] wrong";
}

TEST(FFT, Fft2Shape) {
    auto x = axiom::Tensor::randn({4, 4});

    auto X = axiom::fft::fft2(x);
    ASSERT_TRUE(X.shape() == axiom::Shape({4, 4}))
        << "FFT2 output shape should match input";
    ASSERT_TRUE(X.dtype() == axiom::DType::Complex64)
        << "FFT2 output should be Complex64";
}

TEST(FFT, Fft2Roundtrip) {
    auto x = axiom::Tensor::randn({4, 4});

    auto X = axiom::fft::fft2(x);
    auto x_back = axiom::fft::ifft2(X);
    auto x_real = axiom::ops::real(x_back);

    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT2 roundtrip should recover original";
}

// ============================================================================
// Non-power-of-2 FFT Tests (exercise pocketfft mixed-radix path)
// ============================================================================

TEST(FFT, FftNonPowerOf2Roundtrip) {
    auto x = axiom::Tensor::randn({12});

    auto X = axiom::fft::fft(x);
    auto x_back = axiom::fft::ifft(X);
    auto x_real = axiom::ops::real(x_back);

    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT roundtrip (n=12) should recover original";
}

TEST(FFT, FftPrimeSize) {
    auto x = axiom::Tensor::randn({13});

    auto X = axiom::fft::fft(x);
    auto x_back = axiom::fft::ifft(X);
    auto x_real = axiom::ops::real(x_back);

    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT roundtrip (n=13, prime) should recover original";
}

TEST(FFT, FftNonPowerOf2OrthoRoundtrip) {
    auto x = axiom::Tensor::randn({12});

    auto X = axiom::fft::fft(x, -1, -1, "ortho");
    auto x_back = axiom::fft::ifft(X, -1, -1, "ortho");
    auto x_real = axiom::ops::real(x_back);

    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT ortho roundtrip (n=12) should recover original";
}

TEST(FFT, Fft2NonPowerOf2Roundtrip) {
    auto x = axiom::Tensor::randn({3, 5});

    auto X = axiom::fft::fft2(x);
    auto x_back = axiom::fft::ifft2(X);
    auto x_real = axiom::ops::real(x_back);

    ASSERT_TRUE(x_real.allclose(x, 1e-4, 1e-4))
        << "FFT2 roundtrip (3x5) should recover original";
}

// ============================================================================
// Window Function Tests
// ============================================================================

TEST(FFT, HannWindow) {
    auto w = axiom::fft::hann_window(10);
    ASSERT_TRUE(w.shape() == axiom::Shape({10})) << "Hann window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Hann window dtype wrong";

    // Hann window should start at 0 for periodic window
    const float *data = w.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0]) < 1e-6) << "Hann window should start near 0";

    // Middle should be close to 1
    ASSERT_TRUE(data[5] > 0.9f) << "Hann window middle should be near 1";
}

TEST(FFT, HannWindowSymmetric) {
    auto w = axiom::fft::hann_window(10, false);
    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "Hann symmetric shape wrong";

    // Symmetric window: first and last should both be near 0
    const float *data = w.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0]) < 1e-6)
        << "Hann symmetric should start near 0";
    ASSERT_TRUE(std::abs(data[9]) < 1e-6) << "Hann symmetric should end near 0";
}

TEST(FFT, HammingWindow) {
    auto w = axiom::fft::hamming_window(10);
    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "Hamming window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Hamming window dtype wrong";

    // Hamming window starts at 0.08 (alpha - beta = 0.54 - 0.46)
    const float *data = w.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0] - 0.08f) < 0.01f)
        << "Hamming window should start at ~0.08";
}

TEST(FFT, BlackmanWindow) {
    auto w = axiom::fft::blackman_window(10);
    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "Blackman window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Blackman window dtype wrong";

    // Blackman window starts near 0
    const float *data = w.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0]) < 0.01f)
        << "Blackman window should start near 0";
}

TEST(FFT, BartlettWindow) {
    auto w = axiom::fft::bartlett_window(10);
    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "Bartlett window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Bartlett window dtype wrong";

    // Bartlett (triangular) window starts at 0
    const float *data = w.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0]) < 1e-6)
        << "Bartlett window should start at 0";

    // Peak should be at center
    ASSERT_TRUE(data[5] > 0.9f) << "Bartlett window middle should be near 1";
}

TEST(FFT, KaiserWindow) {
    auto w = axiom::fft::kaiser_window(10, 12.0);
    ASSERT_TRUE(w.shape() == axiom::Shape({10})) << "Kaiser window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Kaiser window dtype wrong";

    // Kaiser window should be symmetric and positive
    const float *data = w.typed_data<float>();
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(data[i] > 0.0f) << "Kaiser window values should be "
                                       "positive";
    }

    // Peak should be at center
    ASSERT_TRUE(data[4] > data[0]) << "Kaiser window should peak in middle";
    ASSERT_TRUE(data[5] > data[0]) << "Kaiser window should peak in middle";
}

TEST(FFT, WindowFloat64) {
    auto w = axiom::fft::hann_window(8, true, axiom::DType::Float64);
    ASSERT_TRUE(w.dtype() == axiom::DType::Float64)
        << "Window should support Float64";
    ASSERT_TRUE(w.shape() == axiom::Shape({8})) << "Window shape wrong";
}

TEST(FFT, WindowSize1) {
    // For M=1, Hann window is 0.5 - 0.5*cos(0) = 0 (mathematically correct)
    auto w = axiom::fft::hann_window(1);
    ASSERT_TRUE(w.shape() == axiom::Shape({1})) << "Window size 1 shape wrong";
    ASSERT_TRUE(std::abs(w.item<float>({0})) < 1e-6)
        << "Hann window size 1 should be 0";

    // Hamming has non-zero offset, so it's 0.08 for M=1
    auto w2 = axiom::fft::hamming_window(1);
    ASSERT_TRUE(w2.shape() == axiom::Shape({1}))
        << "Hamming window size 1 shape wrong";
    ASSERT_TRUE(std::abs(w2.item<float>({0}) - 0.08f) < 0.01f)
        << "Hamming window size 1 should be ~0.08";
}

TEST(FFT, CosineWindow) {
    auto w = axiom::fft::cosine_window(10);
    ASSERT_TRUE(w.shape() == axiom::Shape({10})) << "Cosine window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Cosine window dtype wrong";

    // Cosine window: sin(pi*0/N) = 0 at start
    const float *data = w.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0]) < 1e-6) << "Cosine window should start at 0";

    // Peak near middle
    ASSERT_TRUE(data[5] > 0.9f) << "Cosine window middle should be near 1";
}

TEST(FFT, CosineWindowSymmetric) {
    auto w = axiom::fft::cosine_window(10, false);
    const float *data = w.typed_data<float>();

    // Symmetric: first and last should be near 0
    ASSERT_TRUE(std::abs(data[0]) < 1e-6)
        << "Cosine symmetric should start near 0";
    ASSERT_TRUE(std::abs(data[9]) < 1e-6)
        << "Cosine symmetric should end near 0";
}

TEST(FFT, ExponentialWindow) {
    auto w = axiom::fft::exponential_window(10, 2.0);
    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "Exponential window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Exponential window dtype wrong";

    // All values should be positive
    const float *data = w.typed_data<float>();
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(data[i] > 0.0f)
            << "Exponential window values should be positive";
    }

    // Center should be the maximum (value = 1.0)
    ASSERT_TRUE(data[5] > data[0])
        << "Exponential window should peak in middle";
}

TEST(FFT, GaussianWindow) {
    auto w = axiom::fft::gaussian_window(10, 2.0);
    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "Gaussian window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Gaussian window dtype wrong";

    // All values should be positive
    const float *data = w.typed_data<float>();
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(data[i] > 0.0f)
            << "Gaussian window values should be positive";
    }

    // Peak should be at center
    ASSERT_TRUE(data[5] > data[0]) << "Gaussian window should peak in middle";
    ASSERT_TRUE(data[5] > data[9]) << "Gaussian window should peak in middle";
}

TEST(FFT, NuttallWindow) {
    auto w = axiom::fft::nuttall_window(10);
    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "Nuttall window shape wrong";
    ASSERT_TRUE(w.dtype() == axiom::DType::Float32)
        << "Nuttall window dtype wrong";

    // Nuttall window starts near 0
    const float *data = w.typed_data<float>();
    ASSERT_TRUE(std::abs(data[0]) < 0.01f)
        << "Nuttall window should start near 0";

    // Peak should be in the middle region
    ASSERT_TRUE(data[5] > data[0]) << "Nuttall window should peak in middle";
}

TEST(FFT, NuttallWindowSymmetric) {
    auto w = axiom::fft::nuttall_window(10, false);
    const float *data = w.typed_data<float>();

    // Symmetric: first and last should be near 0
    ASSERT_TRUE(std::abs(data[0]) < 0.01f)
        << "Nuttall symmetric should start near 0";
    ASSERT_TRUE(std::abs(data[9]) < 0.01f)
        << "Nuttall symmetric should end near 0";
}

TEST(FFT, GeneralCosineWindow) {
    // Hann window coefficients: should reproduce hann_window
    std::vector<double> hann_coeffs = {0.5, 0.5};
    auto w = axiom::fft::general_cosine_window(10, hann_coeffs);
    auto hann = axiom::fft::hann_window(10);

    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "General cosine window shape wrong";

    const float *wdata = w.typed_data<float>();
    const float *hdata = hann.typed_data<float>();
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(std::abs(wdata[i] - hdata[i]) < 1e-5)
            << "General cosine with Hann coeffs should match hann_window at "
            << i;
    }
}

TEST(FFT, GeneralCosineWindowNuttallCoeffs) {
    // Nuttall coefficients: should reproduce nuttall_window
    std::vector<double> nuttall_coeffs = {0.3635819, 0.4891775, 0.1365995,
                                          0.0106411};
    auto w = axiom::fft::general_cosine_window(10, nuttall_coeffs);
    auto nuttall = axiom::fft::nuttall_window(10);

    const float *wdata = w.typed_data<float>();
    const float *ndata = nuttall.typed_data<float>();
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(std::abs(wdata[i] - ndata[i]) < 1e-5)
            << "General cosine with Nuttall coeffs should match nuttall_window "
               "at "
            << i;
    }
}

TEST(FFT, GeneralHammingWindow) {
    // Default alpha=0.54 should match hamming_window
    auto w = axiom::fft::general_hamming_window(10, 0.54);
    auto hamming = axiom::fft::hamming_window(10);

    ASSERT_TRUE(w.shape() == axiom::Shape({10}))
        << "General Hamming window shape wrong";

    const float *wdata = w.typed_data<float>();
    const float *hdata = hamming.typed_data<float>();
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(std::abs(wdata[i] - hdata[i]) < 1e-4)
            << "General Hamming (0.54) should match hamming_window at " << i;
    }
}

TEST(FFT, GeneralHammingWindowHann) {
    // alpha=0.5 should produce Hann window
    auto w = axiom::fft::general_hamming_window(10, 0.5);
    auto hann = axiom::fft::hann_window(10);

    const float *wdata = w.typed_data<float>();
    const float *hdata = hann.typed_data<float>();
    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(std::abs(wdata[i] - hdata[i]) < 1e-5)
            << "General Hamming (0.5) should match hann_window at " << i;
    }
}
