#include "axiom/audio.hpp"
#include "axiom/error.hpp"
#include "axiom/fft.hpp"
#include "axiom/operations.hpp"

#include <cmath>
#include <vector>

namespace axiom {
namespace audio {

namespace {

// Convert frequency in Hz to mel scale
inline double hz_to_mel(double freq) {
    return 2595.0 * std::log10(1.0 + freq / 700.0);
}

// Convert mel scale to frequency in Hz
inline double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

} // namespace

Tensor mel_filterbank(int64_t n_freqs, int64_t n_mels, float sample_rate,
                      float f_min, float f_max, DType dtype) {
    if (f_max <= 0.0f) {
        f_max = sample_rate / 2.0f;
    }

    // Mel scale boundaries
    double mel_min = hz_to_mel(static_cast<double>(f_min));
    double mel_max = hz_to_mel(static_cast<double>(f_max));

    // Create n_mels + 2 evenly spaced mel points
    std::vector<double> mel_points(static_cast<size_t>(n_mels + 2));
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        mel_points[static_cast<size_t>(i)] =
            mel_min + static_cast<double>(i) * (mel_max - mel_min) /
                          static_cast<double>(n_mels + 1);
    }

    // Convert back to Hz
    std::vector<double> hz_points(mel_points.size());
    for (size_t i = 0; i < mel_points.size(); ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Convert to FFT bin indices
    std::vector<double> bin_freqs(static_cast<size_t>(n_freqs));
    for (int64_t i = 0; i < n_freqs; ++i) {
        bin_freqs[static_cast<size_t>(i)] =
            static_cast<double>(i) * static_cast<double>(sample_rate) /
            (2.0 * static_cast<double>(n_freqs - 1));
    }

    // Build filterbank matrix (n_freqs, n_mels)
    std::vector<float> fb_data(static_cast<size_t>(n_freqs * n_mels), 0.0f);

    for (int64_t m = 0; m < n_mels; ++m) {
        double left = hz_points[static_cast<size_t>(m)];
        double center = hz_points[static_cast<size_t>(m + 1)];
        double right = hz_points[static_cast<size_t>(m + 2)];

        for (int64_t f = 0; f < n_freqs; ++f) {
            double freq = bin_freqs[static_cast<size_t>(f)];
            float val = 0.0f;

            if (freq >= left && freq <= center && center > left) {
                val = static_cast<float>((freq - left) / (center - left));
            } else if (freq > center && freq <= right && right > center) {
                val = static_cast<float>((right - freq) / (right - center));
            }

            fb_data[static_cast<size_t>(f * n_mels + m)] = val;
        }
    }

    auto result = Tensor::from_data(
        fb_data.data(),
        Shape{static_cast<size_t>(n_freqs), static_cast<size_t>(n_mels)}, true);

    if (dtype != DType::Float32) {
        result = result.astype(dtype);
    }

    return result;
}

Tensor mel_spectrogram(const Tensor &waveform, float sample_rate, int64_t n_fft,
                       int64_t hop_length, int64_t n_mels, float f_min,
                       float f_max, float power, bool log_mel) {
    if (hop_length <= 0) {
        hop_length = n_fft / 4;
    }
    if (f_max <= 0.0f) {
        f_max = sample_rate / 2.0f;
    }

    // STFT → complex spectrogram: (..., n_fft/2+1, n_frames)
    auto spec = fft::stft(waveform, n_fft, hop_length);

    // Magnitude: |STFT|
    auto mag = ops::abs(spec);

    // Apply power
    if (power != 1.0f) {
        auto pow_tensor = Tensor::full({1}, power);
        if (mag.device() == Device::GPU) {
            pow_tensor = pow_tensor.gpu();
        }
        mag = ops::power(mag, pow_tensor);
    }

    // Create mel filterbank: (n_fft/2+1, n_mels)
    int64_t n_freqs = n_fft / 2 + 1;
    auto fb =
        mel_filterbank(n_freqs, n_mels, sample_rate, f_min, f_max, mag.dtype());

    // Move filterbank to same device as magnitude
    if (mag.device() != fb.device()) {
        fb = fb.to(mag.device());
    }

    // Mel spectrogram = mag^T @ fb → transpose mag last 2 dims, matmul,
    // transpose back mag: (..., n_freqs, n_frames), fb: (n_freqs, n_mels) We
    // want: (..., n_mels, n_frames) = (mag.transpose(-2,-1) @
    // fb).transpose(-2,-1) but simpler: mel = fb^T @ mag → (n_mels, n_freqs) @
    // (n_freqs, n_frames) = (n_mels, n_frames)

    auto fb_t = fb.transpose();        // (n_mels, n_freqs)
    auto mel = ops::matmul(fb_t, mag); // (..., n_mels, n_frames)

    // Log mel
    if (log_mel) {
        // Clamp to minimum to avoid log(0)
        auto floor_val = Tensor::full({1}, 1e-10f, mel.device());
        mel = ops::maximum(mel, floor_val);
        mel = ops::log(mel);
    }

    return mel;
}

} // namespace audio
} // namespace axiom
