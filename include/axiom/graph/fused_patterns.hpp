#pragma once

#include "axiom/dtype.hpp"
#include "axiom/tensor.hpp"
#include "graph_node.hpp"

#include <cstddef>
#include <vector>

namespace axiom {
namespace graph {

// Detect a known fusion pattern from an op chain.
FusedPattern detect_pattern(const FusedOpChain &chain);

// Execute a known fused pattern using optimized SIMD kernels.
// off/cnt allow processing a sub-range for parallel dispatch.
// Returns true on success, false if the pattern/dtype is unsupported.
bool dispatch_fused_pattern(FusedPattern pattern,
                            const std::vector<Tensor> &inputs, Tensor &result,
                            size_t off = 0, size_t cnt = 0);

// Check if dtype is supported for SIMD fused patterns.
bool is_fused_simd_dtype(DType dtype);

// Check if an integer dtype is supported for a given pattern.
bool pattern_supports_integer(FusedPattern pattern);

} // namespace graph
} // namespace axiom
