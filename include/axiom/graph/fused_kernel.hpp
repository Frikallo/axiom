#pragma once

#include <cstddef>

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"

namespace axiom {
namespace graph {

// Devirtualized function pointer types for the generic fused loop.
// These wrap the existing HWY dispatch functions, avoiding virtual
// dispatch overhead.
using UnaryFn = void (*)(const void *in, void *out, size_t n);
using BinaryFn = void (*)(const void *a, const void *b, void *out, size_t n);

// Look up the devirtualized function pointer for a unary op + dtype.
// Returns nullptr if the op/dtype combination is not supported.
UnaryFn get_unary_fn(ops::OpType op, DType dtype);

// Look up the devirtualized function pointer for a binary op + dtype.
// Returns nullptr if the op/dtype combination is not supported.
BinaryFn get_binary_fn(ops::OpType op, DType dtype);

} // namespace graph
} // namespace axiom
