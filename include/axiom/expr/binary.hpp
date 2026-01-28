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

// Operation tags for binary operations
// Each tag carries the operation type for dispatch

struct AddOp {
    static constexpr ops::OpType type = ops::OpType::Add;
    static constexpr const char *name = "add";
};

struct SubOp {
    static constexpr ops::OpType type = ops::OpType::Subtract;
    static constexpr const char *name = "subtract";
};

struct MulOp {
    static constexpr ops::OpType type = ops::OpType::Multiply;
    static constexpr const char *name = "multiply";
};

struct DivOp {
    static constexpr ops::OpType type = ops::OpType::Divide;
    static constexpr const char *name = "divide";
};

struct ModOp {
    static constexpr ops::OpType type = ops::OpType::Modulo;
    static constexpr const char *name = "modulo";
};

struct PowOp {
    static constexpr ops::OpType type = ops::OpType::Power;
    static constexpr const char *name = "power";
};

// Comparison operations
struct EqOp {
    static constexpr ops::OpType type = ops::OpType::Equal;
    static constexpr const char *name = "equal";
    static constexpr DType result_dtype = DType::Bool;
};

struct NeOp {
    static constexpr ops::OpType type = ops::OpType::NotEqual;
    static constexpr const char *name = "not_equal";
    static constexpr DType result_dtype = DType::Bool;
};

struct LtOp {
    static constexpr ops::OpType type = ops::OpType::Less;
    static constexpr const char *name = "less";
    static constexpr DType result_dtype = DType::Bool;
};

struct LeOp {
    static constexpr ops::OpType type = ops::OpType::LessEqual;
    static constexpr const char *name = "less_equal";
    static constexpr DType result_dtype = DType::Bool;
};

struct GtOp {
    static constexpr ops::OpType type = ops::OpType::Greater;
    static constexpr const char *name = "greater";
    static constexpr DType result_dtype = DType::Bool;
};

struct GeOp {
    static constexpr ops::OpType type = ops::OpType::GreaterEqual;
    static constexpr const char *name = "greater_equal";
    static constexpr DType result_dtype = DType::Bool;
};

// Logical operations
struct AndOp {
    static constexpr ops::OpType type = ops::OpType::LogicalAnd;
    static constexpr const char *name = "logical_and";
    static constexpr DType result_dtype = DType::Bool;
};

struct OrOp {
    static constexpr ops::OpType type = ops::OpType::LogicalOr;
    static constexpr const char *name = "logical_or";
    static constexpr DType result_dtype = DType::Bool;
};

struct XorOp {
    static constexpr ops::OpType type = ops::OpType::LogicalXor;
    static constexpr const char *name = "logical_xor";
    static constexpr DType result_dtype = DType::Bool;
};

// Bitwise operations
struct BitAndOp {
    static constexpr ops::OpType type = ops::OpType::BitwiseAnd;
    static constexpr const char *name = "bitwise_and";
};

struct BitOrOp {
    static constexpr ops::OpType type = ops::OpType::BitwiseOr;
    static constexpr const char *name = "bitwise_or";
};

struct BitXorOp {
    static constexpr ops::OpType type = ops::OpType::BitwiseXor;
    static constexpr const char *name = "bitwise_xor";
};

struct LShiftOp {
    static constexpr ops::OpType type = ops::OpType::LeftShift;
    static constexpr const char *name = "left_shift";
};

struct RShiftOp {
    static constexpr ops::OpType type = ops::OpType::RightShift;
    static constexpr const char *name = "right_shift";
};

// Math operations
struct MaxOp {
    static constexpr ops::OpType type = ops::OpType::Maximum;
    static constexpr const char *name = "maximum";
};

struct MinOp {
    static constexpr ops::OpType type = ops::OpType::Minimum;
    static constexpr const char *name = "minimum";
};

// Trait to check if an operation has a fixed result dtype
template <typename Op, typename = void>
struct has_fixed_result_dtype : std::false_type {};

template <typename Op>
struct has_fixed_result_dtype<Op, std::void_t<decltype(Op::result_dtype)>>
    : std::true_type {};

template <typename Op>
inline constexpr bool has_fixed_result_dtype_v =
    has_fixed_result_dtype<Op>::value;

// Binary expression template
// Stores both operands by value (TensorRef holds pointer, so this is cheap)
template <typename Op, typename LHS, typename RHS>
class BinaryExpr : public ExprBase<BinaryExpr<Op, LHS, RHS>> {
    LHS lhs_;
    RHS rhs_;

  public:
    using op_type = Op;
    using lhs_type = LHS;
    using rhs_type = RHS;

    BinaryExpr(LHS lhs, RHS rhs) : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    // Expression interface implementation
    DType dtype_impl() const {
        if constexpr (has_fixed_result_dtype_v<Op>) {
            return Op::result_dtype;
        } else {
            return ops::promote_types(lhs_.dtype(), rhs_.dtype());
        }
    }

    Shape shape_impl() const {
        return ShapeUtils::broadcast_shape(lhs_.shape(), rhs_.shape());
    }

    Device device_impl() const {
        Device lhs_dev = lhs_.device();
        Device rhs_dev = rhs_.device();
        // GPU takes priority - if either operand is on GPU, result is on GPU
        if (lhs_dev == Device::GPU || rhs_dev == Device::GPU) {
            return Device::GPU;
        }
        return Device::CPU;
    }

    // Evaluate the expression
    // Dispatches to fused or eager evaluation based on expression
    // characteristics
    Tensor eval_impl() const {
        // Check if CPU fusion is beneficial at compile time
        // expr_depth_v > 1 means we have nested expressions worth fusing
        if constexpr (expr_depth_v<BinaryExpr<Op, LHS, RHS>> > 1) {
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
        Tensor lhs_tensor = lhs_.eval();
        Tensor rhs_tensor = rhs_.eval();

        const ops::Operation *op = ops::OperationRegistry::get_operation(
            Op::type, lhs_tensor.device());
        if (!op) {
            throw RuntimeError::not_implemented(std::string("Operation ") +
                                                Op::name + " not available");
        }
        return op->execute_binary(lhs_tensor, rhs_tensor);
    }

    // Access operands (for fusion/traversal)
    const LHS &lhs() const { return lhs_; }
    const RHS &rhs() const { return rhs_; }

    // Get operation type
    static constexpr ops::OpType op_type_value() { return Op::type; }
};

// Helper function to create binary expressions
template <typename Op, typename LHS, typename RHS>
auto make_binary_expr(LHS &&lhs, RHS &&rhs) {
    return BinaryExpr<Op, std::decay_t<LHS>, std::decay_t<RHS>>(
        std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

} // namespace expr
} // namespace axiom
