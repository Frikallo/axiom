#pragma once

#include <type_traits>

#include "expr/base.hpp"
#include "expr/binary.hpp"
#include "expr/traits.hpp"
#include "expr/unary.hpp"
#include "operations.hpp"
#include "tensor.hpp"

namespace axiom {

// ============================================================================
// Lazy Evaluation Operators (Expression Templates)
//
// These operators return expression templates instead of eagerly evaluated
// Tensors. The expressions are automatically converted to Tensors when needed
// via implicit conversion, preserving backward compatibility while enabling
// automatic kernel fusion for chained operations.
//
// Example:
//   Tensor a, b, c;
//   Tensor d = (a + b) * c;  // Single fused evaluation instead of two
// ============================================================================

// ============================================================================
// Arithmetic operators (return expression templates)
// ============================================================================

inline auto operator+(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::AddOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator-(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::SubOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator*(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::MulOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator/(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::DivOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator%(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::ModOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

// ============================================================================
// Expression + Tensor operators (for chaining)
// ============================================================================

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator+(const Expr &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::AddOp, Expr, expr::TensorRef>(
        lhs, expr::TensorRef(rhs));
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator+(const Tensor &lhs, const Expr &rhs) {
    return expr::BinaryExpr<expr::AddOp, expr::TensorRef, Expr>(
        expr::TensorRef(lhs), rhs);
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator-(const Expr &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::SubOp, Expr, expr::TensorRef>(
        lhs, expr::TensorRef(rhs));
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator-(const Tensor &lhs, const Expr &rhs) {
    return expr::BinaryExpr<expr::SubOp, expr::TensorRef, Expr>(
        expr::TensorRef(lhs), rhs);
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator*(const Expr &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::MulOp, Expr, expr::TensorRef>(
        lhs, expr::TensorRef(rhs));
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator*(const Tensor &lhs, const Expr &rhs) {
    return expr::BinaryExpr<expr::MulOp, expr::TensorRef, Expr>(
        expr::TensorRef(lhs), rhs);
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator/(const Expr &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::DivOp, Expr, expr::TensorRef>(
        lhs, expr::TensorRef(rhs));
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline auto operator/(const Tensor &lhs, const Expr &rhs) {
    return expr::BinaryExpr<expr::DivOp, expr::TensorRef, Expr>(
        expr::TensorRef(lhs), rhs);
}

// ============================================================================
// Expression + Expression operators (for chaining)
// ============================================================================

template <typename LHS, typename RHS,
          typename = std::enable_if_t<expr::is_expression_v<LHS> &&
                                      expr::is_expression_v<RHS>>>
inline auto operator+(const LHS &lhs, const RHS &rhs) {
    return expr::BinaryExpr<expr::AddOp, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS,
          typename = std::enable_if_t<expr::is_expression_v<LHS> &&
                                      expr::is_expression_v<RHS>>>
inline auto operator-(const LHS &lhs, const RHS &rhs) {
    return expr::BinaryExpr<expr::SubOp, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS,
          typename = std::enable_if_t<expr::is_expression_v<LHS> &&
                                      expr::is_expression_v<RHS>>>
inline auto operator*(const LHS &lhs, const RHS &rhs) {
    return expr::BinaryExpr<expr::MulOp, LHS, RHS>(lhs, rhs);
}

template <typename LHS, typename RHS,
          typename = std::enable_if_t<expr::is_expression_v<LHS> &&
                                      expr::is_expression_v<RHS>>>
inline auto operator/(const LHS &lhs, const RHS &rhs) {
    return expr::BinaryExpr<expr::DivOp, LHS, RHS>(lhs, rhs);
}

// ============================================================================
// Comparison operators (return expression templates)
// ============================================================================

inline auto operator==(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::EqOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator!=(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::NeOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator<(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::LtOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator<=(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::LeOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator>(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::GtOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator>=(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::GeOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

// ============================================================================
// Logical operators (return expression templates)
// ============================================================================

inline auto operator&&(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::AndOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator||(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::OrOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

// ============================================================================
// Bitwise operators (return expression templates)
// ============================================================================

inline auto operator&(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::BitAndOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator|(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::BitOrOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator^(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::BitXorOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator<<(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::LShiftOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

inline auto operator>>(const Tensor &lhs, const Tensor &rhs) {
    return expr::BinaryExpr<expr::RShiftOp, expr::TensorRef, expr::TensorRef>(
        expr::TensorRef(lhs), expr::TensorRef(rhs));
}

// ============================================================================
// Scalar comparison operators (for cleaner masking syntax)
// ============================================================================

// tensor > scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator>(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::GtOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator>(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::GtOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

// tensor >= scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator>=(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::GeOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator>=(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::GeOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

// tensor < scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator<(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::LtOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator<(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::LtOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

// tensor <= scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator<=(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::LeOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator<=(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::LeOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

// tensor == scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator==(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::EqOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator==(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::EqOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

// tensor != scalar
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator!=(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::NeOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator!=(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::NeOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

// Logical NOT for boolean tensors
inline auto operator!(const Tensor &tensor) {
    return expr::UnaryExpr<expr::NotOp, expr::TensorRef>(
        expr::TensorRef(tensor));
}

// ============================================================================
// Scalar arithmetic operations (convenience overloads)
// ============================================================================

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator+(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::AddOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator+(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::AddOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator-(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::SubOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator-(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::SubOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator*(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::MulOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator*(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::MulOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator/(const Tensor &tensor, T scalar) {
    return expr::BinaryExpr<expr::DivOp, expr::TensorRef, expr::ScalarExpr<T>>(
        expr::TensorRef(tensor), expr::ScalarExpr<T>(scalar, tensor.device()));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto operator/(T scalar, const Tensor &tensor) {
    return expr::BinaryExpr<expr::DivOp, expr::ScalarExpr<T>, expr::TensorRef>(
        expr::ScalarExpr<T>(scalar, tensor.device()), expr::TensorRef(tensor));
}

// ============================================================================
// In-place operators (still eager - they modify existing tensors)
// ============================================================================

inline Tensor &operator+=(Tensor &lhs, const Tensor &rhs) {
    ops::add_inplace(lhs, rhs);
    return lhs;
}

inline Tensor &operator-=(Tensor &lhs, const Tensor &rhs) {
    ops::subtract_inplace(lhs, rhs);
    return lhs;
}

inline Tensor &operator*=(Tensor &lhs, const Tensor &rhs) {
    ops::multiply_inplace(lhs, rhs);
    return lhs;
}

inline Tensor &operator/=(Tensor &lhs, const Tensor &rhs) {
    ops::divide_inplace(lhs, rhs);
    return lhs;
}

// Convenience overloads for scalars
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor &operator+=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::add_inplace(lhs, scalar_tensor);
    return lhs;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor &operator-=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::subtract_inplace(lhs, scalar_tensor);
    return lhs;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor &operator*=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::multiply_inplace(lhs, scalar_tensor);
    return lhs;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline Tensor &operator/=(Tensor &lhs, T scalar) {
    Tensor scalar_tensor = Tensor::full({}, scalar, lhs.device());
    ops::divide_inplace(lhs, scalar_tensor);
    return lhs;
}

// ============================================================================
// Expression in-place operators
// Evaluate the expression and assign to the tensor
// ============================================================================

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline Tensor &operator+=(Tensor &lhs, const Expr &rhs) {
    lhs = lhs + rhs.eval();
    return lhs;
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline Tensor &operator-=(Tensor &lhs, const Expr &rhs) {
    lhs = lhs - rhs.eval();
    return lhs;
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline Tensor &operator*=(Tensor &lhs, const Expr &rhs) {
    lhs = lhs * rhs.eval();
    return lhs;
}

template <typename Expr,
          typename = std::enable_if_t<expr::is_expression_v<Expr>>>
inline Tensor &operator/=(Tensor &lhs, const Expr &rhs) {
    lhs = lhs / rhs.eval();
    return lhs;
}

} // namespace axiom
