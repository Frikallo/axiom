#pragma once

#include <utility>
#include <vector>

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"
#include "axiom/shape.hpp"
#include "base.hpp"
#include "traits.hpp"

// Forward declaration for CPU fusion (implemented in cpu_fused.hpp)
namespace axiom {
namespace expr {
template <typename Expr> Tensor cpu_eval_fused(const Expr &expr);

template <typename Expr> bool shouldFuseCPU(const Expr &expr);
} // namespace expr
} // namespace axiom

namespace axiom {
namespace expr {

// Operation tags for unary operations

struct NegOp {
    static constexpr ops::OpType type = ops::OpType::Negate;
    static constexpr const char *name = "negate";
};

struct AbsOp {
    static constexpr ops::OpType type = ops::OpType::Abs;
    static constexpr const char *name = "abs";
};

struct SqrtOp {
    static constexpr ops::OpType type = ops::OpType::Sqrt;
    static constexpr const char *name = "sqrt";
};

struct ExpOp {
    static constexpr ops::OpType type = ops::OpType::Exp;
    static constexpr const char *name = "exp";
};

struct LogOp {
    static constexpr ops::OpType type = ops::OpType::Log;
    static constexpr const char *name = "log";
};

struct SinOp {
    static constexpr ops::OpType type = ops::OpType::Sin;
    static constexpr const char *name = "sin";
};

struct CosOp {
    static constexpr ops::OpType type = ops::OpType::Cos;
    static constexpr const char *name = "cos";
};

struct TanOp {
    static constexpr ops::OpType type = ops::OpType::Tan;
    static constexpr const char *name = "tan";
};

struct ErfOp {
    static constexpr ops::OpType type = ops::OpType::Erf;
    static constexpr const char *name = "erf";
};

// NumPy-like math operations
struct SignOp {
    static constexpr ops::OpType type = ops::OpType::Sign;
    static constexpr const char *name = "sign";
};

struct FloorOp {
    static constexpr ops::OpType type = ops::OpType::Floor;
    static constexpr const char *name = "floor";
};

struct CeilOp {
    static constexpr ops::OpType type = ops::OpType::Ceil;
    static constexpr const char *name = "ceil";
};

struct TruncOp {
    static constexpr ops::OpType type = ops::OpType::Trunc;
    static constexpr const char *name = "trunc";
};

struct RoundOp {
    static constexpr ops::OpType type = ops::OpType::Round;
    static constexpr const char *name = "round";
};

struct ReciprocalOp {
    static constexpr ops::OpType type = ops::OpType::Reciprocal;
    static constexpr const char *name = "reciprocal";
};

struct SquareOp {
    static constexpr ops::OpType type = ops::OpType::Square;
    static constexpr const char *name = "square";
};

struct CbrtOp {
    static constexpr ops::OpType type = ops::OpType::Cbrt;
    static constexpr const char *name = "cbrt";
};

// Testing operations (return Bool)
struct IsNaNOp {
    static constexpr ops::OpType type = ops::OpType::IsNaN;
    static constexpr const char *name = "isnan";
    static constexpr DType result_dtype = DType::Bool;
};

struct IsInfOp {
    static constexpr ops::OpType type = ops::OpType::IsInf;
    static constexpr const char *name = "isinf";
    static constexpr DType result_dtype = DType::Bool;
};

struct IsFiniteOp {
    static constexpr ops::OpType type = ops::OpType::IsFinite;
    static constexpr const char *name = "isfinite";
    static constexpr DType result_dtype = DType::Bool;
};

// Logical not
struct NotOp {
    static constexpr ops::OpType type = ops::OpType::LogicalNot;
    static constexpr const char *name = "logical_not";
    static constexpr DType result_dtype = DType::Bool;
};

// Activation functions
struct ReluOp {
    static constexpr ops::OpType type = ops::OpType::ReLU;
    static constexpr const char *name = "relu";
};

struct SigmoidOp {
    static constexpr ops::OpType type = ops::OpType::Sigmoid;
    static constexpr const char *name = "sigmoid";
};

struct TanhActivationOp {
    static constexpr ops::OpType type = ops::OpType::Tanh;
    static constexpr const char *name = "tanh";
};

struct GeluOp {
    static constexpr ops::OpType type = ops::OpType::GELU;
    static constexpr const char *name = "gelu";
};

struct SiluOp {
    static constexpr ops::OpType type = ops::OpType::SiLU;
    static constexpr const char *name = "silu";
};

// Complex operations
struct ConjOp {
    static constexpr ops::OpType type = ops::OpType::Conj;
    static constexpr const char *name = "conj";
};

// Trait to check if an operation has a fixed result dtype
template <typename Op, typename = void>
struct unary_has_fixed_result_dtype : std::false_type {};

template <typename Op>
struct unary_has_fixed_result_dtype<Op, std::void_t<decltype(Op::result_dtype)>>
    : std::true_type {};

template <typename Op>
inline constexpr bool unary_has_fixed_result_dtype_v =
    unary_has_fixed_result_dtype<Op>::value;

// Unary expression template
template <typename Op, typename Operand>
class UnaryExpr : public ExprBase<UnaryExpr<Op, Operand>> {
    Operand operand_;

  public:
    using op_type = Op;
    using operand_type = Operand;

    explicit UnaryExpr(Operand operand) : operand_(std::move(operand)) {}

    // Expression interface implementation
    DType dtype_impl() const {
        if constexpr (unary_has_fixed_result_dtype_v<Op>) {
            return Op::result_dtype;
        } else {
            return operand_.dtype();
        }
    }

    Shape shape_impl() const { return operand_.shape(); }

    Device device_impl() const { return operand_.device(); }

    // Evaluate the expression
    // Dispatches to fused or eager evaluation based on expression
    // characteristics
    Tensor eval_impl() const {
        // Check if CPU fusion is beneficial at compile time
        // expr_depth_v > 1 means we have nested expressions worth fusing
        if constexpr (expr_depth_v<UnaryExpr<Op, Operand>> > 1) {
            // Runtime check for device and contiguity
            if (shouldFuseCPU(*this)) {
                return cpu_eval_fused(*this);
            }
        }
        // GPU fusion is handled separately in gpu_eval.mm
        // Fall through to eager evaluation for:
        // - Single operations (depth == 1)
        // - GPU tensors (handled by existing MPSGraph infra)
        // - Non-contiguous tensors
        return eval_eager();
    }

    // Eager evaluation - creates intermediate tensors
    // Used for single ops or when fusion is not beneficial
    Tensor eval_eager() const {
        Tensor operand_tensor = operand_.eval();

        const ops::Operation *op = ops::OperationRegistry::get_operation(
            Op::type, operand_tensor.device());
        if (!op) {
            throw RuntimeError::not_implemented(std::string("Operation ") +
                                                Op::name + " not available");
        }
        return op->execute_unary(operand_tensor);
    }

    // Access operand (for fusion/traversal)
    const Operand &operand() const { return operand_; }

    // Get operation type
    static constexpr ops::OpType op_type_value() { return Op::type; }
};

// Helper function to create unary expressions
template <typename Op, typename Operand>
auto make_unary_expr(Operand &&operand) {
    return UnaryExpr<Op, std::decay_t<Operand>>(std::forward<Operand>(operand));
}

// ============================================================================
// ExprBase fluent method implementations
// Defined here after UnaryExpr is complete
// ============================================================================

template <typename Derived> auto ExprBase<Derived>::relu() const {
    return UnaryExpr<ReluOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::sigmoid() const {
    return UnaryExpr<SigmoidOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::tanh() const {
    return UnaryExpr<TanhActivationOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::gelu() const {
    return UnaryExpr<GeluOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::silu() const {
    return UnaryExpr<SiluOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::sqrt() const {
    return UnaryExpr<SqrtOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::exp() const {
    return UnaryExpr<ExpOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::log() const {
    return UnaryExpr<LogOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::abs() const {
    return UnaryExpr<AbsOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::neg() const {
    return UnaryExpr<NegOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::square() const {
    return UnaryExpr<SquareOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::reciprocal() const {
    return UnaryExpr<ReciprocalOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::sin() const {
    return UnaryExpr<SinOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::cos() const {
    return UnaryExpr<CosOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::tan() const {
    return UnaryExpr<TanOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::floor() const {
    return UnaryExpr<FloorOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::ceil() const {
    return UnaryExpr<CeilOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::round() const {
    return UnaryExpr<RoundOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::sign() const {
    return UnaryExpr<SignOp, Derived>(derived());
}

template <typename Derived> auto ExprBase<Derived>::erf() const {
    return UnaryExpr<ErfOp, Derived>(derived());
}

} // namespace expr
} // namespace axiom
