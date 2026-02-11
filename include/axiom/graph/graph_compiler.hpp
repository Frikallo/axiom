#pragma once

#include <memory>

#include "compiled_graph.hpp"
#include "graph_node.hpp"
#include "graph_signature.hpp"

namespace axiom {
namespace graph {

std::shared_ptr<CompiledGraph> compile(const GraphSignature &sig,
                                       const GraphNode *root);

} // namespace graph
} // namespace axiom
