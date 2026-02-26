#pragma once

#include "axiom/operations.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include "axiom/graph/compiled_graph.hpp"
#include "axiom/tensor.hpp"
#include <vector>
#endif

namespace axiom {
namespace backends {
namespace cuda {

// Register fused CUDA kernel patterns (AddReLU, MulAdd, etc.).
void register_cuda_fused_operations();

} // namespace cuda
} // namespace backends

#ifdef AXIOM_CUDA_SUPPORT
namespace graph {

bool execute_cuda_fused_chain(const StepBase &step,
                              const std::vector<ops::OpType> &op_chain,
                              std::vector<Tensor> &buffers);

bool execute_cuda_fused_reduction(const FusedReductionStep &step,
                                  std::vector<Tensor> &buffers);

} // namespace graph
#endif

} // namespace axiom
