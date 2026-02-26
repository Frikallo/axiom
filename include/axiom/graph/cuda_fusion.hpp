#pragma once

#ifdef AXIOM_CUDA_SUPPORT

#include "axiom/graph/compiled_graph.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

#include <vector>

namespace axiom {
namespace graph {

// Execute a chain of elementwise ops as a single NVRTC-compiled CUDA kernel.
// Returns false if any op/dtype is unsupported (caller falls back to op-by-op).
bool execute_cuda_fused_chain(const StepBase &step,
                              const std::vector<ops::OpType> &op_chain,
                              std::vector<Tensor> &buffers);

// Execute a fused elementwise chain followed by a reduction.
// Returns false on failure (caller falls back to separate chain + reduce).
bool execute_cuda_fused_reduction(const FusedReductionStep &step,
                                  std::vector<Tensor> &buffers);

} // namespace graph
} // namespace axiom

#endif
