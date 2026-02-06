#pragma once

#include <memory>

#include "compiled_graph.hpp"
#include "graph_node.hpp"
#include "graph_signature.hpp"

namespace axiom {
namespace graph {

class GraphCompiler {
  public:
    // Compile a graph rooted at `root` into an executable plan.
    // The signature is attached to the resulting CompiledGraph.
    static std::shared_ptr<CompiledGraph> compile(const GraphSignature &sig,
                                                  const GraphNode *root);

  private:
    // Pipeline stages
    static std::vector<const GraphNode *>
    topological_sort(const GraphNode *root);

    static std::vector<const GraphNode *>
    dead_code_elimination(const std::vector<const GraphNode *> &sorted,
                          const GraphNode *root);

    // Fusion analysis: groups consecutive element-wise ops into fused steps
    struct FusionGroup {
        std::vector<const GraphNode *> nodes;
        FusedPattern pattern;
        bool is_fused;
    };

    static std::vector<FusionGroup>
    fusion_analysis(const std::vector<const GraphNode *> &sorted);

    // Memory planning: compute liveness and buffer reuse
    static void memory_plan(CompiledGraph &plan);

    // Compute loop parameters for each step
    static void compute_loop_params(CompiledGraph &plan);
};

} // namespace graph
} // namespace axiom
