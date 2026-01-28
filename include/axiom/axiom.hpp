#pragma once

// This is the single entry-point for the Axiom library.
// Include this file to get access to all the core functionality.

#include "axiom/debug.hpp"
#include "axiom/dtype.hpp"
#include "axiom/error.hpp"
#include "axiom/io.hpp"
#include "axiom/numeric.hpp"
#include "axiom/operations.hpp"
#include "axiom/random.hpp"
#include "axiom/shape.hpp"
#include "axiom/storage.hpp"
#include "axiom/system.hpp"
#include "axiom/tensor.hpp"

// Expression templates for lazy evaluation
#include "axiom/expr/base.hpp"
#include "axiom/expr/binary.hpp"
#include "axiom/expr/cpu_fused.hpp"
#include "axiom/expr/fluent.hpp"
#include "axiom/expr/matmul.hpp"
#include "axiom/expr/traits.hpp"
#include "axiom/expr/unary.hpp"

// Tensor operators (uses expression templates for lazy evaluation)
#include "axiom/tensor_operators.hpp"