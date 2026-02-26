#pragma once

#include <memory>
#include <vector>

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"
#include "axiom/shape.hpp"
#include "graph_node.hpp"

namespace axiom {

// Forward declaration
class Tensor;

namespace graph {

// Forward declarations for graph compiler infrastructure
struct GraphSignature;
struct CompiledGraph;
class GraphCache;

// Environment variable to disable lazy evaluation globally
// Set AXIOM_EAGER_MODE=1 to disable lazy evaluation
bool is_eager_mode_enabled();

// Get/set the maximum number of pending nodes before auto-materialization
size_t get_max_pending_nodes();
void set_max_pending_nodes(size_t max_nodes);

// Shape inference functions
Shape infer_binary_shape(const Shape &lhs, const Shape &rhs);
Shape infer_unary_shape(const Shape &input);
Shape infer_reduction_shape(const Shape &input, const std::vector<int> &axes,
                            bool keep_dims);
Shape infer_matmul_shape(const Shape &a, const Shape &b, bool transpose_a,
                         bool transpose_b);

// Dtype inference functions
DType infer_binary_dtype(DType lhs, DType rhs);
DType infer_unary_dtype(ops::OpType op, DType input);
DType infer_reduction_dtype(ops::OpType op, DType input);

// The graph registry manages lazy tensor creation and materialization
class GraphRegistry {
  public:
    // Create a lazy tensor from a unary operation
    static Tensor create_lazy_unary(ops::OpType op, const Tensor &input,
                                    const OpParams &params = NoParams{});

    // Create a lazy tensor from a binary operation
    static Tensor create_lazy_binary(ops::OpType op, const Tensor &lhs,
                                     const Tensor &rhs,
                                     const OpParams &params = NoParams{});

    // Create a lazy tensor from a reduction operation
    static Tensor create_lazy_reduction(ops::OpType op, const Tensor &input,
                                        const std::vector<int> &axes,
                                        bool keep_dims);

    // Create a lazy tensor from a matmul operation
    static Tensor create_lazy_matmul(const Tensor &a, const Tensor &b,
                                     bool transpose_a, bool transpose_b);

    // Create lazy tensors for GPU full-graph compilation.
    // These capture ops that normally dispatch eagerly (softmax, layernorm,
    // conv, etc.) into the lazy DAG so the GPU graph compiler can build
    // a single MPSGraph for the entire forward pass.

    // Softmax/LogSoftmax along axis
    static Tensor create_lazy_softmax(ops::OpType op, const Tensor &input,
                                      int axis);

    // LayerNorm: input, weight, bias tensors + axis + eps
    static Tensor create_lazy_layernorm(const Tensor &input,
                                        const Tensor &weight,
                                        const Tensor &bias, int axis,
                                        float eps);

    // BatchNorm1D: input, weight, bias, running_mean, running_var + eps
    static Tensor create_lazy_batchnorm(const Tensor &input,
                                        const Tensor &weight,
                                        const Tensor &bias,
                                        const Tensor &running_mean,
                                        const Tensor &running_var, float eps);

    // Conv1D/Conv2D: input, weight, bias + conv params
    static Tensor create_lazy_conv(ops::OpType op, const Tensor &input,
                                   const Tensor &weight, const Tensor &bias,
                                   const ConvParams &params,
                                   const Shape &output_shape);

    // Reshape (GPU only — CPU uses zero-copy views)
    static Tensor create_lazy_reshape(const Tensor &input,
                                      const Shape &new_shape);

    // Transpose/Permute (GPU only)
    static Tensor create_lazy_transpose(const Tensor &input,
                                        const std::vector<int> &axes,
                                        const Shape &output_shape,
                                        const Strides &output_strides);

    // Pad
    static Tensor
    create_lazy_pad(const Tensor &input,
                    const std::vector<std::pair<size_t, size_t>> &pad_widths,
                    double value, const Shape &output_shape);

    // Slice
    static Tensor create_lazy_slice(const Tensor &input,
                                    const std::vector<int64_t> &starts,
                                    const std::vector<int64_t> &ends,
                                    const std::vector<int64_t> &strides,
                                    const Shape &output_shape);

    // MaskedFill: fill positions where mask is true with value
    static Tensor create_lazy_masked_fill(const Tensor &input,
                                          const Tensor &mask, float value);

    // GLU along dimension
    static Tensor create_lazy_glu(const Tensor &input, int dim,
                                  const Shape &output_shape);

    // Materialize a graph node and all its dependencies.
    // Uses signature → cache → compile → execute pipeline.
    static void materialize(GraphNode *node);

    // Get statistics about pending lazy tensors (for debugging)
    static size_t pending_node_count();

    // Force materialization of all pending nodes
    static void materialize_all();

    // Internal helpers for lazy node creation (need access to pending_nodes_)
    static Tensor finalize_lazy_node(std::shared_ptr<GraphNode> node);
    static Tensor materialize_and_return(std::shared_ptr<GraphNode> node);

  private:
    // Thread-local storage for pending nodes
    static thread_local std::vector<std::weak_ptr<GraphNode>> pending_nodes_;
};

// RAII helper to temporarily enable eager mode in a scope
class EagerModeScope {
  public:
    EagerModeScope();
    ~EagerModeScope();

    EagerModeScope(const EagerModeScope &) = delete;
    EagerModeScope &operator=(const EagerModeScope &) = delete;

  private:
    bool previous_mode_;
};

} // namespace graph
} // namespace axiom
