#include "axiom/nn/positional.hpp"

#include <cmath>

namespace axiom::nn {

Tensor sinusoidal_position_embedding(int seq_len, int d_model, DType dtype,
                                     Device device) {
    // Generate embeddings for positions (seq_len-1) down to -(seq_len-1)
    // matching NeMo's RelPositionalEncoding.
    int total = 2 * seq_len - 1;
    auto pe = Tensor::zeros(
        {static_cast<size_t>(total), static_cast<size_t>(d_model)});

    float *pe_data = pe.typed_data<float>();
    for (int pos_idx = 0; pos_idx < total; ++pos_idx) {
        float position = static_cast<float>(seq_len - 1 - pos_idx);
        for (int i = 0; i < d_model; i += 2) {
            float div_term = std::exp(static_cast<float>(i) *
                                      (-std::log(10000.0f) / d_model));
            pe_data[pos_idx * d_model + i] = std::sin(position * div_term);
            if (i + 1 < d_model) {
                pe_data[pos_idx * d_model + i + 1] =
                    std::cos(position * div_term);
            }
        }
    }

    // Cast and transfer to target dtype/device
    if (dtype != DType::Float32) {
        pe = pe.astype(dtype);
    }
    if (device != Device::CPU) {
        pe = pe.to(device);
    }
    return pe; // (2*seq_len-1, d_model)
}

} // namespace axiom::nn
