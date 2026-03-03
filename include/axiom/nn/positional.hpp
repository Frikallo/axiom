#pragma once

#include "axiom/dtype.hpp"
#include "axiom/storage.hpp"
#include "axiom/tensor.hpp"

namespace axiom::nn {

// Generate sinusoidal position embeddings for relative positions.
// Returns (2*seq_len - 1, d_model) encoding relative positions
// -(seq_len-1) to +(seq_len-1).
// Computes in Float32 on CPU, then casts to target dtype and device.
Tensor sinusoidal_position_embedding(int seq_len, int d_model,
                                     DType dtype = DType::Float32,
                                     Device device = Device::CPU);

} // namespace axiom::nn
