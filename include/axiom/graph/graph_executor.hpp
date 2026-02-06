#pragma once

#include <memory>
#include <vector>

#include "../tensor.hpp"
#include "compiled_graph.hpp"

namespace axiom {
namespace graph {

class GraphExecutor {
  public:
    // Execute a compiled graph plan with the given input tensors.
    // Returns the output tensor.
    static Tensor execute(const CompiledGraph &plan,
                          const std::vector<Tensor> &inputs);

  private:
    static void execute_step(const ExecutionStep &step,
                             const CompiledGraph &plan,
                             std::vector<Tensor> &buffers);

    static void execute_single_op(const ExecutionStep &step,
                                  std::vector<Tensor> &buffers);

    static void execute_fused_known(const ExecutionStep &step,
                                    const CompiledGraph &plan,
                                    std::vector<Tensor> &buffers);

    static void execute_fused_generic(const ExecutionStep &step,
                                      const CompiledGraph &plan,
                                      std::vector<Tensor> &buffers);

    static void execute_matmul_activation(const ExecutionStep &step,
                                          std::vector<Tensor> &buffers);
};

} // namespace graph
} // namespace axiom
