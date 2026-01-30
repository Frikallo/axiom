#pragma once

#include <utility>

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"
#include "axiom/shape.hpp"
#include "base.hpp"
#include "traits.hpp"

namespace axiom {
namespace expr {

// Matrix multiplication expression
// Supports noalias() optimization hint for safe self-assignment patterns
template <typename LHS, typename RHS>
class MatMulExpr : public ExprBase<MatMulExpr<LHS, RHS>> {
    LHS lhs_;
    RHS rhs_;
    bool transpose_lhs_;
    bool transpose_rhs_;
    bool noalias_;

  public:
    using lhs_type = LHS;
    using rhs_type = RHS;

    MatMulExpr(LHS lhs, RHS rhs, bool transpose_lhs = false,
               bool transpose_rhs = false)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)),
          transpose_lhs_(transpose_lhs), transpose_rhs_(transpose_rhs),
          noalias_(false) {}

    // Private constructor for noalias copy
    MatMulExpr(LHS lhs, RHS rhs, bool transpose_lhs, bool transpose_rhs,
               bool noalias)
        : lhs_(std::move(lhs)), rhs_(std::move(rhs)),
          transpose_lhs_(transpose_lhs), transpose_rhs_(transpose_rhs),
          noalias_(noalias) {}

    // Expression interface implementation
    DType dtype_impl() const {
        return ops::promote_types(lhs_.dtype(), rhs_.dtype());
    }

    Shape shape_impl() const {
        Shape lhs_shape = lhs_.shape();
        Shape rhs_shape = rhs_.shape();

        // Get effective dimensions considering transpose flags
        size_t lhs_ndim = lhs_shape.size();
        size_t rhs_ndim = rhs_shape.size();

        if (lhs_ndim == 0 || rhs_ndim == 0) {
            return {}; // Error case, will be handled in eval
        }

        // Get matrix dimensions
        size_t lhs_rows, lhs_cols, rhs_rows, rhs_cols;

        if (lhs_ndim == 1) {
            lhs_rows = 1;
            lhs_cols = lhs_shape[0];
        } else {
            lhs_rows = lhs_shape[lhs_ndim - 2];
            lhs_cols = lhs_shape[lhs_ndim - 1];
        }

        if (rhs_ndim == 1) {
            rhs_rows = rhs_shape[0];
            rhs_cols = 1;
        } else {
            rhs_rows = rhs_shape[rhs_ndim - 2];
            rhs_cols = rhs_shape[rhs_ndim - 1];
        }

        if (transpose_lhs_)
            std::swap(lhs_rows, lhs_cols);
        if (transpose_rhs_)
            std::swap(rhs_rows, rhs_cols);

        // Compute batch shape
        Shape batch_shape;
        if (lhs_ndim > 2 || rhs_ndim > 2) {
            Shape lhs_batch, rhs_batch;
            for (size_t i = 0; i < (lhs_ndim > 2 ? lhs_ndim - 2 : 0); ++i) {
                lhs_batch.push_back(lhs_shape[i]);
            }
            for (size_t i = 0; i < (rhs_ndim > 2 ? rhs_ndim - 2 : 0); ++i) {
                rhs_batch.push_back(rhs_shape[i]);
            }
            if (!lhs_batch.empty() || !rhs_batch.empty()) {
                if (lhs_batch.empty()) {
                    batch_shape = rhs_batch;
                } else if (rhs_batch.empty()) {
                    batch_shape = lhs_batch;
                } else {
                    batch_shape =
                        ShapeUtils::broadcast_shape(lhs_batch, rhs_batch);
                }
            }
        }

        // Build result shape
        Shape result_shape = batch_shape;
        if (lhs_ndim == 1 && rhs_ndim == 1) {
            result_shape = {1}; // Dot product
        } else if (lhs_ndim == 1) {
            result_shape.push_back(rhs_cols);
        } else if (rhs_ndim == 1) {
            result_shape.push_back(lhs_rows);
        } else {
            result_shape.push_back(lhs_rows);
            result_shape.push_back(rhs_cols);
        }
        if (result_shape.empty()) {
            result_shape = {1};
        }

        return result_shape;
    }

    Device device_impl() const {
        Device lhs_dev = lhs_.device();
        Device rhs_dev = rhs_.device();
        if (lhs_dev == Device::GPU || rhs_dev == Device::GPU) {
            return Device::GPU;
        }
        return Device::CPU;
    }

    // Evaluate the expression
    Tensor eval_impl() const {
        Tensor lhs_tensor = lhs_.eval();
        Tensor rhs_tensor = rhs_.eval();
        return ops::matmul(lhs_tensor, rhs_tensor, transpose_lhs_,
                           transpose_rhs_);
    }

    // Access operands
    const LHS &lhs() const { return lhs_; }
    const RHS &rhs() const { return rhs_; }

    // Transpose flags
    bool transpose_lhs() const { return transpose_lhs_; }
    bool transpose_rhs() const { return transpose_rhs_; }

    // Aliasing control
    bool is_noalias() const { return noalias_; }
    bool needs_alias_check() const { return !noalias_; }

    // Create a copy with noalias flag set
    // User asserts that the output does not alias any input
    MatMulExpr noalias() const {
        return MatMulExpr(lhs_, rhs_, transpose_lhs_, transpose_rhs_, true);
    }
};

// Helper function to create matmul expressions
template <typename LHS, typename RHS>
auto make_matmul_expr(LHS &&lhs, RHS &&rhs, bool transpose_lhs = false,
                      bool transpose_rhs = false) {
    return MatMulExpr<std::decay_t<LHS>, std::decay_t<RHS>>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs), transpose_lhs,
        transpose_rhs);
}

} // namespace expr
} // namespace axiom
