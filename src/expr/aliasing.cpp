#include "axiom/expr/base.hpp"
#include "axiom/expr/binary.hpp"
#include "axiom/expr/matmul.hpp"
#include "axiom/expr/traits.hpp"
#include "axiom/expr/unary.hpp"
#include "axiom/tensor.hpp"

#include <vector>

namespace axiom {
namespace expr {

// ============================================================================
// Aliasing Detection Utilities
// ============================================================================

namespace {

// Collect all tensor pointers from an expression tree
template <typename Expr>
void collect_tensor_ptrs_impl(const Expr &expr,
                              std::vector<const Tensor *> &ptrs) {
    if constexpr (is_tensor_ref_v<std::decay_t<Expr>>) {
        ptrs.push_back(expr.tensor_ptr());
    } else if constexpr (is_scalar_expr_v<std::decay_t<Expr>>) {
        // Scalars have no tensor storage
    } else if constexpr (is_binary_expr_v<std::decay_t<Expr>>) {
        collect_tensor_ptrs_impl(expr.lhs(), ptrs);
        collect_tensor_ptrs_impl(expr.rhs(), ptrs);
    } else if constexpr (is_unary_expr_v<std::decay_t<Expr>>) {
        collect_tensor_ptrs_impl(expr.operand(), ptrs);
    } else if constexpr (is_matmul_expr_v<std::decay_t<Expr>>) {
        collect_tensor_ptrs_impl(expr.lhs(), ptrs);
        collect_tensor_ptrs_impl(expr.rhs(), ptrs);
    }
}

// Check if two tensors share the same storage
bool shares_storage(const Tensor &a, const Tensor &b) {
    return a.storage().get() == b.storage().get();
}

// Check if two memory ranges overlap
// Returns true if [start1, start1+size1) overlaps [start2, start2+size2)
bool ranges_overlap(size_t start1, size_t size1, size_t start2, size_t size2) {
    return start1 < start2 + size2 && start2 < start1 + size1;
}

// Check if tensor 'a' memory range overlaps with tensor 'b' memory range
// This handles views with different offsets into the same storage
bool memory_overlaps(const Tensor &a, const Tensor &b) {
    if (!shares_storage(a, b)) {
        return false;
    }

    // Same storage - check if the actual data regions overlap
    // For contiguous tensors, this is straightforward
    size_t a_start = a.offset();
    size_t a_end = a_start + a.nbytes();

    size_t b_start = b.offset();
    size_t b_end = b_start + b.nbytes();

    return ranges_overlap(a_start, a_end - a_start, b_start, b_end - b_start);
}

} // namespace

// ============================================================================
// Public API for aliasing detection
// ============================================================================

// Check if output tensor aliases any input tensor in the expression
template <typename Expr>
bool aliases_any_input(const Expr &expr, const Tensor &output) {
    std::vector<const Tensor *> input_ptrs;
    collect_tensor_ptrs_impl(expr, input_ptrs);

    for (const Tensor *input : input_ptrs) {
        if (memory_overlaps(*input, output)) {
            return true;
        }
    }
    return false;
}

// Explicit instantiations for common expression types

// TensorRef
template bool aliases_any_input(const TensorRef &expr, const Tensor &output);

// Binary expressions
template bool
aliases_any_input(const BinaryExpr<AddOp, TensorRef, TensorRef> &expr,
                  const Tensor &output);
template bool
aliases_any_input(const BinaryExpr<SubOp, TensorRef, TensorRef> &expr,
                  const Tensor &output);
template bool
aliases_any_input(const BinaryExpr<MulOp, TensorRef, TensorRef> &expr,
                  const Tensor &output);
template bool
aliases_any_input(const BinaryExpr<DivOp, TensorRef, TensorRef> &expr,
                  const Tensor &output);

// MatMul expressions
template bool aliases_any_input(const MatMulExpr<TensorRef, TensorRef> &expr,
                                const Tensor &output);

// Nested expressions
using AddExpr = BinaryExpr<AddOp, TensorRef, TensorRef>;
using MulExpr = BinaryExpr<MulOp, TensorRef, TensorRef>;

template bool
aliases_any_input(const BinaryExpr<MulOp, AddExpr, TensorRef> &expr,
                  const Tensor &output);
template bool aliases_any_input(const MatMulExpr<AddExpr, TensorRef> &expr,
                                const Tensor &output);

// ============================================================================
// Safe evaluation with aliasing check
// ============================================================================

// Evaluate expression, handling aliasing if output is provided
// If output aliases any input, evaluates to a temporary first
template <typename Expr> Tensor safe_eval(const Expr &expr, Tensor *output) {
    if (output == nullptr) {
        // No output specified, just evaluate normally
        return expr.eval();
    }

    // Check if matmul expression needs aliasing check
    if constexpr (is_matmul_expr_v<std::decay_t<Expr>>) {
        if (expr.needs_alias_check() && aliases_any_input(expr, *output)) {
            // Aliasing detected - evaluate to temporary first
            Tensor temp = expr.eval();
            *output = std::move(temp);
            return *output;
        }
    }

    // For other expressions or when no aliasing, evaluate directly
    // Note: For element-wise ops, in-place aliasing is generally safe
    // (e.g., a = a + b) because we read before writing each element
    Tensor result = expr.eval();
    *output = std::move(result);
    return *output;
}

// Explicit instantiations for safe_eval
template Tensor safe_eval(const MatMulExpr<TensorRef, TensorRef> &expr,
                          Tensor *output);
template Tensor safe_eval(const BinaryExpr<AddOp, TensorRef, TensorRef> &expr,
                          Tensor *output);
template Tensor safe_eval(const BinaryExpr<MulOp, TensorRef, TensorRef> &expr,
                          Tensor *output);

} // namespace expr
} // namespace axiom
