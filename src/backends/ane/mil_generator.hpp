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
    std::string name;               // e.g., "wq", "w1"
    std::vector<uint8_t> blob_data; // Full blob with 128-byte header + FP16 data
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

    // Declare an input tensor. shape must be 4D: [batch, channels, height, width]
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

    std::string add_relu(const std::string &input_var,
                         const std::string &name);
    std::string add_sigmoid(const std::string &input_var,
                            const std::string &name);
    std::string add_silu(const std::string &input_var,
                         const std::string &name);
    std::string add_gelu(const std::string &input_var,
                         const std::string &name);
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
    std::string add_multihead_attention(const std::string &input_var,
                                         const Tensor &q_weight,
                                         const Tensor &k_weight,
                                         const Tensor &v_weight,
                                         const Tensor &o_weight,
                                         const Tensor *q_bias,
                                         const Tensor *k_bias,
                                         const Tensor *v_bias,
                                         const Tensor *o_bias,
                                         int num_heads,
                                         const std::string &name);

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

  private:
    std::string body_;
    std::string input_decl_;
    std::string output_var_;
    std::vector<WeightBlob> weight_blobs_;
    std::map<std::string, std::vector<int64_t>> shapes_; // var → shape
    int var_counter_ = 0;
    bool conv_consts_emitted_ = false;

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
