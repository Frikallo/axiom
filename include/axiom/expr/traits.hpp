#pragma once

#include <type_traits>

#include "base.hpp"

namespace axiom {
namespace expr {

// is_expression_v is defined in base.hpp

// Check if type is a BinaryExpr
template <typename T> struct is_binary_expr : std::false_type {};

template <typename Op, typename LHS, typename RHS>
struct is_binary_expr<BinaryExpr<Op, LHS, RHS>> : std::true_type {};

template <typename T>
inline constexpr bool is_binary_expr_v = is_binary_expr<T>::value;

// Check if type is a UnaryExpr
template <typename T> struct is_unary_expr : std::false_type {};

template <typename Op, typename Operand>
struct is_unary_expr<UnaryExpr<Op, Operand>> : std::true_type {};

template <typename T>
inline constexpr bool is_unary_expr_v = is_unary_expr<T>::value;

// Check if type is a MatMulExpr
template <typename T> struct is_matmul_expr : std::false_type {};

template <typename LHS, typename RHS>
struct is_matmul_expr<MatMulExpr<LHS, RHS>> : std::true_type {};

template <typename T>
inline constexpr bool is_matmul_expr_v = is_matmul_expr<T>::value;

// Check if type is a TensorRef
template <typename T> struct is_tensor_ref : std::false_type {};

template <> struct is_tensor_ref<TensorRef> : std::true_type {};

template <typename T>
inline constexpr bool is_tensor_ref_v = is_tensor_ref<T>::value;

// Check if type is a ScalarExpr
template <typename T> struct is_scalar_expr : std::false_type {};

template <typename S> struct is_scalar_expr<ScalarExpr<S>> : std::true_type {};

template <typename T>
inline constexpr bool is_scalar_expr_v = is_scalar_expr<T>::value;

// Check if type is a leaf expression (TensorRef or ScalarExpr)
template <typename T>
inline constexpr bool is_leaf_expr_v =
    is_tensor_ref_v<T> || is_scalar_expr_v<T>;

// Expression depth (for compile-time optimization decisions)
template <typename T> struct expr_depth {
    static constexpr size_t value = 0;
};

template <> struct expr_depth<TensorRef> {
    static constexpr size_t value = 0;
};

template <typename S> struct expr_depth<ScalarExpr<S>> {
    static constexpr size_t value = 0;
};

template <typename Op, typename LHS, typename RHS>
struct expr_depth<BinaryExpr<Op, LHS, RHS>> {
    static constexpr size_t value =
        1 + (expr_depth<LHS>::value > expr_depth<RHS>::value
                 ? expr_depth<LHS>::value
                 : expr_depth<RHS>::value);
};

template <typename Op, typename Operand>
struct expr_depth<UnaryExpr<Op, Operand>> {
    static constexpr size_t value = 1 + expr_depth<Operand>::value;
};

template <typename LHS, typename RHS> struct expr_depth<MatMulExpr<LHS, RHS>> {
    static constexpr size_t value =
        1 + (expr_depth<LHS>::value > expr_depth<RHS>::value
                 ? expr_depth<LHS>::value
                 : expr_depth<RHS>::value);
};

template <typename T>
inline constexpr size_t expr_depth_v = expr_depth<T>::value;

// Count number of tensor references in expression tree (for aliasing checks)
template <typename T> struct tensor_ref_count {
    static constexpr size_t value = 0;
};

template <> struct tensor_ref_count<TensorRef> {
    static constexpr size_t value = 1;
};

template <typename S> struct tensor_ref_count<ScalarExpr<S>> {
    static constexpr size_t value = 0;
};

template <typename Op, typename LHS, typename RHS>
struct tensor_ref_count<BinaryExpr<Op, LHS, RHS>> {
    static constexpr size_t value =
        tensor_ref_count<LHS>::value + tensor_ref_count<RHS>::value;
};

template <typename Op, typename Operand>
struct tensor_ref_count<UnaryExpr<Op, Operand>> {
    static constexpr size_t value = tensor_ref_count<Operand>::value;
};

template <typename LHS, typename RHS>
struct tensor_ref_count<MatMulExpr<LHS, RHS>> {
    static constexpr size_t value =
        tensor_ref_count<LHS>::value + tensor_ref_count<RHS>::value;
};

template <typename T>
inline constexpr size_t tensor_ref_count_v = tensor_ref_count<T>::value;

// Determine if any expression in the tree potentially requires aliasing check
template <typename T> struct may_need_alias_check : std::false_type {};

template <typename LHS, typename RHS>
struct may_need_alias_check<MatMulExpr<LHS, RHS>> : std::true_type {};

template <typename Op, typename LHS, typename RHS>
struct may_need_alias_check<BinaryExpr<Op, LHS, RHS>> {
    static constexpr bool value =
        may_need_alias_check<LHS>::value || may_need_alias_check<RHS>::value;
};

template <typename Op, typename Operand>
struct may_need_alias_check<UnaryExpr<Op, Operand>> {
    static constexpr bool value = may_need_alias_check<Operand>::value;
};

template <typename T>
inline constexpr bool may_need_alias_check_v = may_need_alias_check<T>::value;

} // namespace expr
} // namespace axiom
