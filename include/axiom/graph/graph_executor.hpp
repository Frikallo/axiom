#pragma once

#include <memory>
#include <vector>

#include "axiom/tensor.hpp"
#include "compiled_graph.hpp"

namespace axiom {
namespace graph {

Tensor execute(const CompiledGraph &plan, const std::vector<Tensor> &inputs);

} // namespace graph
} // namespace axiom
