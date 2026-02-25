#pragma once

#include "dtype.hpp"
#include "tensor.hpp"

#include <cstdint>

namespace axiom {
namespace audio {

// Create a mel filterbank matrix
// Returns: (n_freqs, n_mels) matrix of triangular mel filters
Tensor mel_filterbank(int64_t n_freqs, int64_t n_mels, float sample_rate,
                      float f_min = 0.0f, float f_max = -1.0f,
                      DType dtype = DType::Float32);

// Compute mel spectrogram from waveform
// waveform: 1D or 2D tensor (..., samples)
// Returns: (..., n_mels, n_frames)
Tensor mel_spectrogram(const Tensor &waveform, float sample_rate,
                       int64_t n_fft = 400, int64_t hop_length = -1,
                       int64_t n_mels = 80, float f_min = 0.0f,
                       float f_max = -1.0f, float power = 2.0f,
                       bool log_mel = true);

} // namespace audio
} // namespace axiom
