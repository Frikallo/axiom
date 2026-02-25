#include "axiom_test_utils.hpp"

#include <cmath>

TEST(Audio, MelFilterbankShape) {
    auto fb = axiom::audio::mel_filterbank(201, 80, 16000.0f);
    ASSERT_TRUE(fb.shape() == axiom::Shape({201, 80}));
    EXPECT_EQ(fb.dtype(), axiom::DType::Float32);
}

TEST(Audio, MelFilterbankValues) {
    // Small filterbank for verification
    auto fb = axiom::audio::mel_filterbank(10, 4, 16000.0f);
    ASSERT_TRUE(fb.shape() == axiom::Shape({10, 4}));

    // Filterbank should be non-negative
    auto cpu = fb.cpu();
    const float *data = cpu.typed_data<float>();
    for (size_t i = 0; i < fb.size(); ++i) {
        EXPECT_GE(data[i], 0.0f) << "Filterbank has negative value at " << i;
    }

    // Each row should sum to <= 1 (triangular filters overlap)
    for (int64_t f = 0; f < 10; ++f) {
        float row_sum = 0.0f;
        for (int64_t m = 0; m < 4; ++m) {
            row_sum += data[f * 4 + m];
        }
        EXPECT_LE(row_sum, 1.01f) << "Row sum exceeds 1 at freq bin " << f;
    }
}

TEST(Audio, MelFilterbankCustomRange) {
    auto fb = axiom::audio::mel_filterbank(257, 40, 22050.0f, 80.0f, 7600.0f);
    ASSERT_TRUE(fb.shape() == axiom::Shape({257, 40}));
}

TEST(Audio, STFTBasicShape) {
    // 1D signal of 1024 samples
    auto signal = axiom::Tensor::zeros({1024}, axiom::DType::Float32);
    auto spec = axiom::fft::stft(signal, 256);

    // With center=true, padded by 128 on each side: 1280 samples
    // n_frames = (1280 - 256) / 64 + 1 = 17 (hop_length = 256/4 = 64)
    EXPECT_EQ(spec.shape()[0], 129u); // n_fft/2+1 = 129
    EXPECT_EQ(spec.dtype(), axiom::DType::Complex64);
}

TEST(Audio, STFTNoCenterPad) {
    auto signal = axiom::Tensor::zeros({512}, axiom::DType::Float32);
    auto spec = axiom::fft::stft(signal, 128, 64, -1, axiom::Tensor(), false);

    // n_frames = (512 - 128)/64 + 1 = 7
    EXPECT_EQ(spec.shape()[0], 65u); // n_fft/2+1
    EXPECT_EQ(spec.shape()[1], 7u);  // n_frames
}

TEST(Audio, STFTBatched) {
    auto signal = axiom::Tensor::zeros({2, 1024}, axiom::DType::Float32);
    auto spec = axiom::fft::stft(signal, 256);

    EXPECT_EQ(spec.ndim(), 3u);
    EXPECT_EQ(spec.shape()[0], 2u);   // batch
    EXPECT_EQ(spec.shape()[1], 129u); // freq bins
}

TEST(Audio, STFTSineWave) {
    // Generate 440Hz sine wave at 16kHz sample rate
    size_t num_samples = 4096;
    float sample_rate = 16000.0f;
    float freq = 440.0f;

    std::vector<float> samples(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = std::sin(2.0f * static_cast<float>(M_PI) * freq *
                              static_cast<float>(i) / sample_rate);
    }
    auto signal = axiom::Tensor::from_data(samples.data(), {num_samples}, true);

    auto spec = axiom::fft::stft(signal, 1024);

    // Should have freq_bins=513, and some frames
    EXPECT_EQ(spec.shape()[0], 513u);
    EXPECT_EQ(spec.dtype(), axiom::DType::Complex64);
}

TEST(Audio, MelSpectrogramShape) {
    // 1 second of silence at 16kHz
    auto waveform = axiom::Tensor::zeros({16000}, axiom::DType::Float32);

    auto mel = axiom::audio::mel_spectrogram(waveform, 16000.0f, 400, 160, 80);

    // Should produce (80, n_frames)
    EXPECT_EQ(mel.shape()[0], 80u); // n_mels
    EXPECT_EQ(mel.ndim(), 2u);
}

TEST(Audio, MelSpectrogramSineWave) {
    // 440Hz sine wave, 0.5 seconds, 16kHz
    size_t num_samples = 8000;
    float sample_rate = 16000.0f;
    float freq = 440.0f;

    std::vector<float> samples(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = std::sin(2.0f * static_cast<float>(M_PI) * freq *
                              static_cast<float>(i) / sample_rate);
    }
    auto waveform =
        axiom::Tensor::from_data(samples.data(), {num_samples}, true);

    auto mel =
        axiom::audio::mel_spectrogram(waveform, sample_rate, 512, 128, 40);

    EXPECT_EQ(mel.shape()[0], 40u); // n_mels
    EXPECT_EQ(mel.ndim(), 2u);

    // Values should be finite
    auto cpu = mel.cpu();
    auto finite = axiom::ops::isfinite(cpu);
    auto all_finite = axiom::ops::all(finite);
    EXPECT_TRUE(all_finite.item<bool>({0}))
        << "Mel spectrogram contains non-finite values";
}
