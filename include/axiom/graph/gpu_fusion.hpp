#pragma once

#include "axiom/graph/compiled_graph.hpp"
#include "axiom/tensor.hpp"

#include <vector>

namespace axiom {
namespace graph {

// Execute a fused elementwise chain on GPU via a single MPSGraph.
// Builds one graph containing all ops in the chain, allowing MPS to
// optimize the full pipeline. Returns true on success.
bool execute_gpu_fused_chain(const StepBase &step,
                             const std::vector<ops::OpType> &op_chain,
                             std::vector<Tensor> &buffers);

// Execute a fused elementwise chain + full reduction on GPU.
// Builds a single MPSGraph containing the elementwise ops followed
// by a reduction, returning a scalar result. Returns true on success.
bool execute_gpu_fused_reduction(const FusedReductionStep &step,
                                 std::vector<Tensor> &buffers);

} // namespace graph
} // namespace axiom
