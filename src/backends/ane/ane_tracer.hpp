#pragma once

#ifdef AXIOM_HAS_ANE

#include <memory>
#include <vector>

#include "axiom/graph/graph_node.hpp"
#include "axiom/nn/module.hpp"
#include "axiom/tensor.hpp"

namespace axiom {
namespace backends {
namespace ane {

// ============================================================================
// ANE Tracer — captures arbitrary forward() logic as a computation graph
// ============================================================================

// Thread-local flag: when true, tensor operations create lazy graph nodes
// instead of executing eagerly. This lets us trace through any Module's
// forward() method, capturing residuals, inline ops, permutations, etc.
bool is_ane_tracing();
void set_ane_tracing(bool enabled);

// RAII guard for tracing mode
class TraceScope {
  public:
    TraceScope();
    ~TraceScope();
    TraceScope(const TraceScope &) = delete;
    TraceScope &operator=(const TraceScope &) = delete;

  private:
    bool previous_;
};

// Trace a module's forward() pass and return the output graph node.
// This runs module.forward(input) with tracing enabled, so all ops
// (including residuals, inline activations, permutations) are captured
// in the lazy graph. The returned GraphNode is the root of the DAG.
//
// input_shape: the shape of the input tensor
// Returns: {output_graph_node, input_graph_node, all_constant_nodes}
struct TraceResult {
    std::shared_ptr<graph::GraphNode> output_node;
    std::shared_ptr<graph::GraphNode> input_node;
    Tensor output_tensor; // Keep alive to prevent graph GC
};

TraceResult trace_module(const nn::Module &module, const Shape &input_shape);

// Compile a traced graph to MIL text + weight blobs.
// Walks the GraphNode DAG topologically and emits MIL operations.
struct TracedMIL {
    std::string mil_text;
    std::vector<struct WeightBlob> weight_blobs;
    std::vector<int64_t> ane_input_shape;  // [1, C, 1, S]
    std::vector<int64_t> ane_output_shape; // [1, C, 1, S]
};

TracedMIL compile_trace_to_mil(const TraceResult &trace,
                               const Shape &input_shape);

} // namespace ane
} // namespace backends
} // namespace axiom

#endif // AXIOM_HAS_ANE
