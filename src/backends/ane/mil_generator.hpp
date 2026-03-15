#pragma once

#include <cstdint>
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
class MILGenerator {
  public:
    MILGenerator() = default;

    // Reset state and begin a new MIL program.
    void begin_program();

    // Declare an input tensor. Returns the MIL variable name.
    // shape must be 4D: [batch, channels, height, width]
    std::string add_input(const std::string &name,
                          const std::vector<int64_t> &shape);

    // ================================================================
    // Core operations
    // ================================================================

    // Linear layer as 1x1 convolution (3x faster than matmul on ANE).
    // weight: [out_features, in_features] (Axiom convention)
    // bias: [out_features] or nullptr
    // Input var must be [1, in_features, 1, seq_len].
    // Output: [1, out_features, 1, seq_len].
    std::string add_linear(const std::string &input_var, const Tensor &weight,
                           const Tensor *bias, const std::string &name);

    // Element-wise add.
    std::string add_add(const std::string &a, const std::string &b,
                        const std::string &name);

    // Element-wise multiply.
    std::string add_mul(const std::string &a, const std::string &b,
                        const std::string &name);

    // Element-wise subtract.
    std::string add_sub(const std::string &a, const std::string &b,
                        const std::string &name);

    // ================================================================
    // Activations
    // ================================================================

    // ReLU: max(0, x) — implemented as clip(x, alpha=0)
    std::string add_relu(const std::string &input_var,
                         const std::string &name);

    // Sigmoid: 1 / (1 + exp(-x))
    std::string add_sigmoid(const std::string &input_var,
                            const std::string &name);

    // SiLU (Swish): x * sigmoid(x)
    std::string add_silu(const std::string &input_var,
                         const std::string &name);

    // GELU (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    // On ANE, approximated as x * sigmoid(1.702 * x) for speed.
    std::string add_gelu(const std::string &input_var,
                         const std::string &name);

    // Softmax along given axis.
    std::string add_softmax(const std::string &input_var, int axis,
                            const std::string &name);

    // ================================================================
    // Normalization
    // ================================================================

    // RMS normalization (built from primitives).
    // weight: [dim] — learned scale
    // eps: epsilon value
    // Input/output: [1, dim, 1, seq_len]
    std::string add_rms_norm(const std::string &input_var, const Tensor &weight,
                             float eps, const std::string &name);

    // Layer normalization (built from primitives).
    // weight: [dim], bias: [dim]
    std::string add_layer_norm(const std::string &input_var,
                               const Tensor &weight, const Tensor &bias,
                               float eps, const std::string &name);

    // ================================================================
    // Shape operations
    // ================================================================

    // Reshape tensor to new shape.
    std::string add_reshape(const std::string &input_var,
                            const std::vector<int64_t> &shape,
                            const std::string &name);

    // Transpose with permutation.
    std::string add_transpose(const std::string &input_var,
                              const std::vector<int> &perm,
                              const std::string &name);

    // ================================================================
    // Matmul (for attention scores, not for linear layers)
    // ================================================================

    // Matrix multiply. Use for Q@K^T and scores@V in attention.
    // For linear layers, use add_linear() (1x1 conv) instead.
    std::string add_matmul(const std::string &a, const std::string &b,
                           bool transpose_a, bool transpose_b,
                           const std::string &name);

    // ================================================================
    // Output and finalization
    // ================================================================

    // Set the output variable and get output shape info.
    void set_output(const std::string &output_var);

    // Finalize and return the complete MIL program text.
    std::string finalize();

    // Get the weight blobs generated during MIL construction.
    const std::vector<WeightBlob> &weight_blobs() const {
        return weight_blobs_;
    }

  private:
    std::string body_;              // MIL operation lines
    std::string input_decl_;        // Function input declaration
    std::string output_var_;        // Final output variable name
    std::vector<WeightBlob> weight_blobs_;
    int var_counter_ = 0;
    bool conv_consts_emitted_ = false;

    // Generate a unique variable name.
    std::string next_var(const std::string &prefix = "v");

    // Pack an Axiom tensor into a weight blob (128-byte header + FP16 data).
    // ane_shape: the shape to use in the MIL const declaration.
    // Returns the weight blob name.
    std::string pack_weight(const Tensor &t,
                            const std::vector<int64_t> &ane_shape,
                            const std::string &name);

    // Emit a MIL const line for a scalar.
    std::string emit_scalar_const(const std::string &name,
                                  const std::string &type, float value);

    // Emit a MIL const line for an integer scalar.
    std::string emit_int_const(const std::string &name, int value);

    // Emit a MIL const line for a bool.
    std::string emit_bool_const(const std::string &name, bool value);

    // Emit a MIL const line for an int32 tensor.
    std::string emit_int_tensor_const(const std::string &name,
                                      const std::vector<int> &values);

    // Emit the conv boilerplate constants (pad_type, strides, etc.) once.
    void ensure_conv_consts();

    // Format a shape as MIL type string: "tensor<fp16, [1, 512, 1, 256]>"
    static std::string mil_type(const std::vector<int64_t> &shape);

    // Format a shape as value string: "[1, 512, 1, 256]"
    static std::string mil_shape(const std::vector<int64_t> &shape);
};

} // namespace ane
} // namespace backends
} // namespace axiom
