#pragma once

#include <memory>
#include <vector>

#include "../dtype.hpp"
#include "../operations.hpp"
#include "../shape.hpp"
#include "graph_node.hpp"

namespace axiom {

// Forward declaration
class Tensor;

namespace graph {

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
                                    const GraphNode::Params &params = {});

    // Create a lazy tensor from a binary operation
    static Tensor create_lazy_binary(ops::OpType op, const Tensor &lhs,
                                     const Tensor &rhs,
                                     const GraphNode::Params &params = {});

    // Create a lazy tensor from a reduction operation
    static Tensor create_lazy_reduction(ops::OpType op, const Tensor &input,
                                        const std::vector<int> &axes,
                                        bool keep_dims);

    // Create a lazy tensor from a matmul operation
    static Tensor create_lazy_matmul(const Tensor &a, const Tensor &b,
                                     bool transpose_a, bool transpose_b);

    // Materialize a graph node and all its dependencies
    static void materialize(GraphNode *node);

    // Optimize a subgraph before execution (fusion passes)
    static void optimize_subgraph(GraphNode *root);

    // Identify fusable element-wise chains in the graph
    static std::vector<FusedOpChain> find_fusable_chains(GraphNode *root);

    // Check if two adjacent nodes can be fused
    static bool can_fuse(GraphNode *producer, GraphNode *consumer);

    // Execute a fused operation chain
    static void execute_fused_chain(const FusedOpChain &chain,
                                    std::shared_ptr<Storage> &output_storage,
                                    Shape &output_shape,
                                    Strides &output_strides);

    // Get statistics about pending lazy tensors (for debugging)
    static size_t pending_node_count();

    // Force materialization of all pending nodes
    static void materialize_all();

  private:
    // Internal helper to execute a single node
    static void execute_node(GraphNode *node);

    // Topological sort of graph nodes for execution
    static std::vector<GraphNode *> topological_sort(GraphNode *root);

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
