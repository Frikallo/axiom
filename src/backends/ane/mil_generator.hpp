#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "axiom/tensor.hpp"

namespace axiom {
namespace backends {
namespace ane {

// A weight entry ready for ANE compilation.
struct WeightBlob {
    std::string name; // e.g., "wq", "w1"
    std::vector<uint8_t>
        blob_data; // Full blob with 128-byte header + FP16 data
};

// Generates MIL (Model Intermediate Language) text for Apple Neural Engine.
//
// Usage:
//   MILGenerator gen;
//   gen.begin_program();
//   auto x = gen.add_input("x", {1, 512, 1, 256});
//   auto y = gen.add_linear(x, weight, bias, "fc1");
//   auto z = gen.add_relu(y);
//   gen.set_output(z);
//   auto mil = gen.finalize();
//   auto& weights = gen.weight_blobs();
//
// All tensors use ANE's native [1, C, 1, S] layout in FP16.
// Every variable gets an explicit type annotation — ANE's MIL compiler
// does not support type inference.
class MILGenerator {
  public:
    MILGenerator() = default;

    void begin_program();

    // Declare an input tensor. shape must be 4D: [batch, channels, height,
    // width]
    std::string add_input(const std::string &name,
                          const std::vector<int64_t> &shape);

    // ================================================================
    // Core operations
    // ================================================================

    // Linear layer as 1x1 convolution (3x faster than matmul on ANE).
    // weight: [out_features, in_features], bias: [out_features] or nullptr.
    std::string add_linear(const std::string &input_var, const Tensor &weight,
                           const Tensor *bias, const std::string &name);

    std::string add_add(const std::string &a, const std::string &b,
                        const std::string &name);
    std::string add_mul(const std::string &a, const std::string &b,
                        const std::string &name);
    std::string add_sub(const std::string &a, const std::string &b,
                        const std::string &name);

    // ================================================================
    // Activations
    // ================================================================

    std::string add_relu(const std::string &input_var, const std::string &name);
    std::string add_sigmoid(const std::string &input_var,
                            const std::string &name);
    std::string add_silu(const std::string &input_var, const std::string &name);
    std::string add_gelu(const std::string &input_var, const std::string &name);
    std::string add_softmax(const std::string &input_var, int axis,
                            const std::string &name);

    // ================================================================
    // Normalization
    // ================================================================

    std::string add_rms_norm(const std::string &input_var, const Tensor &weight,
                             float eps, const std::string &name);
    std::string add_layer_norm(const std::string &input_var,
                               const Tensor &weight, const Tensor &bias,
                               float eps, const std::string &name);

    // ================================================================
    // Shape operations
    // ================================================================

    std::string add_reshape(const std::string &input_var,
                            const std::vector<int64_t> &shape,
                            const std::string &name);
    std::string add_transpose(const std::string &input_var,
                              const std::vector<int> &perm,
                              const std::string &name);

    // ================================================================
    // Dynamic weight staging
    // ================================================================

    // Linear with weights packed into the input IOSurface's spatial dimension.
    // Instead of BLOBFILE constants, weights are sliced from the input tensor.
    // This avoids recompilation when weights change.
    //
    // The input tensor is extended: [1, C, 1, S + weight_cols]
    //   spatial [0:S] = activation data
    //   spatial [S:S+in*out/C] = weight data (packed per-channel row)
    //
    // weight_offset: the spatial offset where weight data starts
    // Returns the output variable name and updates weight_offset.
    // graph_input: the original program input (where weights are packed)
    // activation: the actual activation to process (may differ from graph_input
    //             for layers after the first in a Sequential)
    std::string add_linear_dynamic(const std::string &graph_input,
                                   const std::string &activation,
                                   int64_t in_features, int64_t out_features,
                                   bool has_bias, int64_t seq_len,
                                   int64_t &weight_offset,
                                   const std::string &name);

    // Get the total spatial size needed for dynamic weight staging.
    // Call after building the graph to know how large the input IOSurface
    // needs to be.
    int64_t dynamic_spatial_total() const { return dynamic_spatial_total_; }

    // ================================================================
    // INT8 quantized operations
    // ================================================================

    // Linear with INT8 weights (dequantized to FP16 on ANE before compute).
    // Saves ~2x memory bandwidth; 1.88x throughput for bandwidth-bound models.
    // int8_data: quantized weight [out, in] as int8
    // scale: per-channel scale [out] as FP32
    // zero_point: per-channel zero point [out] as int8
    std::string add_linear_int8(const std::string &input_var,
                                const std::vector<int8_t> &int8_data,
                                const std::vector<float> &scale,
                                const std::vector<int8_t> &zero_point,
                                int64_t out_features, int64_t in_features,
                                const Tensor *bias, const std::string &name);

    // ================================================================
    // Conv2d (native convolution on ANE)
    // ================================================================

    // weight: [out_ch, in_ch/groups, kH, kW], bias: [out_ch] or nullptr
    // Input must be 4D: [1, in_ch, H, W]
    std::string add_conv2d(const std::string &input_var, const Tensor &weight,
                           const Tensor *bias, std::array<int, 2> stride,
                           std::array<int, 2> padding,
                           std::array<int, 2> dilation, int groups,
                           const std::string &name);

    // ================================================================
    // Multi-Head Attention (fused QKV + SDPA, non-causal)
    // ================================================================

    // Fused attention: QKV projections → reshape → Q@K^T → scale →
    // softmax → @V → reshape → output projection.
    // Input: [1, d_model, 1, seq_len] (ANE layout)
    // Returns: [1, d_model, 1, seq_len]
    std::string add_multihead_attention(
        const std::string &input_var, const Tensor &q_weight,
        const Tensor &k_weight, const Tensor &v_weight, const Tensor &o_weight,
        const Tensor *q_bias, const Tensor *k_bias, const Tensor *v_bias,
        const Tensor *o_bias, int num_heads, const std::string &name);

    // ================================================================
    // Matmul (for attention scores, not linear layers)
    // ================================================================

    std::string add_matmul(const std::string &a, const std::string &b,
                           bool transpose_a, bool transpose_b,
                           const std::string &name);

    // ================================================================
    // Output and finalization
    // ================================================================

    void set_output(const std::string &output_var);
    std::string finalize();

    const std::vector<WeightBlob> &weight_blobs() const {
        return weight_blobs_;
    }

    // Get the tracked shape for a variable.
    const std::vector<int64_t> &shape_of(const std::string &var) const;

    // Public helpers for trace-based compilation
    void emit_raw(const std::string &mil_line) { body_ += mil_line; }
    void track_shape(const std::string &var,
                     const std::vector<int64_t> &shape) {
        track(var, shape);
    }
    std::string emit_scalar_const_public(const std::string &name,
                                         const std::string &type, float value) {
        return emit_scalar_const(name, type, value);
    }
    void pack_weight_public(const Tensor &t, const std::vector<int64_t> &shape,
                            const std::string &name) {
        pack_weight(t, shape, name);
    }
    static std::string mil_type_public(const std::vector<int64_t> &shape) {
        return mil_type(shape);
    }
    std::string emit_int_const_public(const std::string &name, int value) {
        return emit_int_const(name, value);
    }
    std::string emit_int_tensor_const_public(const std::string &name,
                                             const std::vector<int> &values) {
        return emit_int_tensor_const(name, values);
    }

  private:
    std::string body_;
    std::string input_decl_;
    std::string output_var_;
    std::vector<WeightBlob> weight_blobs_;
    std::map<std::string, std::vector<int64_t>> shapes_; // var → shape
    int var_counter_ = 0;
    bool conv_consts_emitted_ = false;
    int64_t dynamic_spatial_total_ = 0; // Total spatial for dynamic weights

    std::string next_var(const std::string &prefix = "v");

    // Register a variable with its shape.
    void track(const std::string &var, const std::vector<int64_t> &shape);

    // Pack a tensor into weight blob format. Returns blob name.
    std::string pack_weight(const Tensor &t,
                            const std::vector<int64_t> &ane_shape,
                            const std::string &name);

    // Emit typed MIL const declarations.
    std::string emit_scalar_const(const std::string &name,
                                  const std::string &type, float value);
    std::string emit_int_const(const std::string &name, int value);
    std::string emit_bool_const(const std::string &name, bool value);
    std::string emit_int_tensor_const(const std::string &name,
                                      const std::vector<int> &values);

    void ensure_conv_consts();

    static std::string mil_type(const std::vector<int64_t> &shape);
    static std::string mil_shape(const std::vector<int64_t> &shape);
};

} // namespace ane
} // namespace backends
} // namespace axiom
