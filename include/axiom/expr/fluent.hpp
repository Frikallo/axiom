#pragma once

// This header adds fluent method chaining to expression templates.
// Include this AFTER all expression types are defined.
//
// Enables syntax like:
//   auto expr = (a + b).relu().sigmoid();  // Still lazy!
//   Tensor result = expr;  // Evaluates entire fused chain

#include "base.hpp"
#include "binary.hpp"
#include "unary.hpp"

namespace axiom {
namespace expr {

// ============================================================================
// Fluent unary operations for ExprBase
// These are defined as free functions that work on any expression type
// ============================================================================

// Activation functions
template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto relu(Expr &&expr) {
    return UnaryExpr<ReluOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto sigmoid(Expr &&expr) {
    return UnaryExpr<SigmoidOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto tanh(Expr &&expr) {
    return UnaryExpr<TanhActivationOp, std::decay_t<Expr>>(
        std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto gelu(Expr &&expr) {
    return UnaryExpr<GeluOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto silu(Expr &&expr) {
    return UnaryExpr<SiluOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

// Math functions
template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto sqrt(Expr &&expr) {
    return UnaryExpr<SqrtOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto exp(Expr &&expr) {
    return UnaryExpr<ExpOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto log(Expr &&expr) {
    return UnaryExpr<LogOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto abs(Expr &&expr) {
    return UnaryExpr<AbsOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto neg(Expr &&expr) {
    return UnaryExpr<NegOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto square(Expr &&expr) {
    return UnaryExpr<SquareOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto reciprocal(Expr &&expr) {
    return UnaryExpr<ReciprocalOp, std::decay_t<Expr>>(
        std::forward<Expr>(expr));
}

// Trigonometric
template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto sin(Expr &&expr) {
    return UnaryExpr<SinOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto cos(Expr &&expr) {
    return UnaryExpr<CosOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto tan(Expr &&expr) {
    return UnaryExpr<TanOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

// Rounding
template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto floor(Expr &&expr) {
    return UnaryExpr<FloorOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto ceil(Expr &&expr) {
    return UnaryExpr<CeilOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

template <typename Expr,
          typename = std::enable_if_t<is_expression_v<std::decay_t<Expr>>>>
auto round(Expr &&expr) {
    return UnaryExpr<RoundOp, std::decay_t<Expr>>(std::forward<Expr>(expr));
}

// ============================================================================
// ExpressionWrapper - enables method chaining on expressions
// Wraps any expression and provides fluent methods
// ============================================================================

template <typename Expr>
class ExpressionWrapper : public ExprBase<ExpressionWrapper<Expr>> {
    Expr expr_;

  public:
    explicit ExpressionWrapper(Expr expr) : expr_(std::move(expr)) {}

    // Forward ExprBase requirements
    DType dtype_impl() const { return expr_.dtype(); }
    Shape shape_impl() const { return expr_.shape(); }
    Device device_impl() const { return expr_.device(); }
    Tensor eval_impl() const { return expr_.eval(); }

    // Access wrapped expression
    const Expr &expr() const { return expr_; }

    // ========================================================================
    // Fluent unary methods - return wrapped UnaryExpr
    // ========================================================================

    auto relu() const { return wrap(UnaryExpr<ReluOp, Expr>(expr_)); }
    auto sigmoid() const { return wrap(UnaryExpr<SigmoidOp, Expr>(expr_)); }
    auto tanh() const { return wrap(UnaryExpr<TanhActivationOp, Expr>(expr_)); }
    auto gelu() const { return wrap(UnaryExpr<GeluOp, Expr>(expr_)); }
    auto silu() const { return wrap(UnaryExpr<SiluOp, Expr>(expr_)); }

    auto sqrt() const { return wrap(UnaryExpr<SqrtOp, Expr>(expr_)); }
    auto exp() const { return wrap(UnaryExpr<ExpOp, Expr>(expr_)); }
    auto log() const { return wrap(UnaryExpr<LogOp, Expr>(expr_)); }
    auto abs() const { return wrap(UnaryExpr<AbsOp, Expr>(expr_)); }
    auto neg() const { return wrap(UnaryExpr<NegOp, Expr>(expr_)); }
    auto square() const { return wrap(UnaryExpr<SquareOp, Expr>(expr_)); }
    auto reciprocal() const {
        return wrap(UnaryExpr<ReciprocalOp, Expr>(expr_));
    }

    auto sin() const { return wrap(UnaryExpr<SinOp, Expr>(expr_)); }
    auto cos() const { return wrap(UnaryExpr<CosOp, Expr>(expr_)); }
    auto tan() const { return wrap(UnaryExpr<TanOp, Expr>(expr_)); }

    auto floor() const { return wrap(UnaryExpr<FloorOp, Expr>(expr_)); }
    auto ceil() const { return wrap(UnaryExpr<CeilOp, Expr>(expr_)); }
    auto round() const { return wrap(UnaryExpr<RoundOp, Expr>(expr_)); }

    auto sign() const { return wrap(UnaryExpr<SignOp, Expr>(expr_)); }
    auto erf() const { return wrap(UnaryExpr<ErfOp, Expr>(expr_)); }

  private:
    template <typename E> static auto wrap(E &&e) {
        return ExpressionWrapper<std::decay_t<E>>(std::forward<E>(e));
    }
};

// Helper to wrap an expression for fluent chaining
template <typename Expr> auto fluent(Expr &&expr) {
    return ExpressionWrapper<std::decay_t<Expr>>(std::forward<Expr>(expr));
}

} // namespace expr

// Bring fluent() into axiom namespace for convenience
using expr::fluent;

} // namespace axiom
