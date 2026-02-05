#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "../dtype.hpp"
#include "../operations.hpp"
#include "../shape.hpp"
#include "../storage.hpp"

namespace axiom {
namespace graph {

// Forward declaration
class GraphRegistry;

// Represents a deferred computation node in the lazy evaluation graph
struct GraphNode {
    // Unique identifier for this node
    uint64_t id;

    // The operation this node represents
    ops::OpType op_type;

    // Input nodes (other lazy tensors or constant nodes)
    std::vector<std::shared_ptr<GraphNode>> inputs;

    // Computed metadata (no allocation yet)
    Shape output_shape;
    DType output_dtype;
    Device target_device;

    // Operation-specific parameters
    struct Params {
        // For reductions
        std::vector<int> axes;
        bool keep_dims = false;

        // For matmul
        bool transpose_a = false;
        bool transpose_b = false;

        // For leaky_relu
        float alpha = 0.01f;

        // For reshape/view
        Shape new_shape;

        // For softmax/log_softmax
        int axis = -1;
    } params;

    // For constant inputs (eager tensors fed into lazy ops)
    // When a materialized tensor is an input to a lazy op, we wrap it
    std::shared_ptr<Storage> constant_storage;
    Strides constant_strides;
    size_t constant_offset = 0;
    bool is_constant = false;

    // Reference count - how many other nodes depend on this one
    std::atomic<size_t> ref_count{0};

    // Result cache (populated after first materialization)
    mutable std::shared_ptr<Storage> cached_result_;
    mutable Shape cached_shape_;
    mutable Strides cached_strides_;
    mutable bool is_materialized_ = false;

    // Fusion metadata
    bool is_fused = false;
    std::shared_ptr<GraphNode> fused_parent = nullptr;

    GraphNode() : id(next_id()), ref_count(0) {}

    // Check if this node is ready to execute (all inputs materialized or
    // constant)
    bool inputs_ready() const {
        for (const auto &input : inputs) {
            if (!input->is_constant && !input->is_materialized_) {
                return false;
            }
        }
        return true;
    }

    // Get the total byte size of the output
    size_t byte_size() const {
        return ShapeUtils::size(output_shape) * dtype_size(output_dtype);
    }

  private:
    static uint64_t next_id() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1);
    }
};

// Helper to check if an operation is element-wise (same shape in, same shape
// out)
inline bool is_elementwise_op(ops::OpType op) {
    using ops::OpType;
    switch (op) {
    // Binary element-wise
    case OpType::Add:
    case OpType::Subtract:
    case OpType::Multiply:
    case OpType::Divide:
    case OpType::Power:
    case OpType::Modulo:
    case OpType::Maximum:
    case OpType::Minimum:
    case OpType::Atan2:
    case OpType::Hypot:
    // Comparison (element-wise, returns bool)
    case OpType::Equal:
    case OpType::NotEqual:
    case OpType::Less:
    case OpType::LessEqual:
    case OpType::Greater:
    case OpType::GreaterEqual:
    // Logical (element-wise)
    case OpType::LogicalAnd:
    case OpType::LogicalOr:
    case OpType::LogicalXor:
    case OpType::LogicalNot:
    // Bitwise (element-wise)
    case OpType::BitwiseAnd:
    case OpType::BitwiseOr:
    case OpType::BitwiseXor:
    case OpType::LeftShift:
    case OpType::RightShift:
    // Unary element-wise
    case OpType::Negate:
    case OpType::Abs:
    case OpType::Sqrt:
    case OpType::Exp:
    case OpType::Log:
    case OpType::Sin:
    case OpType::Cos:
    case OpType::Tan:
    case OpType::Erf:
    case OpType::Sign:
    case OpType::Floor:
    case OpType::Ceil:
    case OpType::Trunc:
    case OpType::Round:
    case OpType::Reciprocal:
    case OpType::Square:
    case OpType::Cbrt:
    case OpType::IsNaN:
    case OpType::IsInf:
    case OpType::IsFinite:
    case OpType::Conj:
    // Activations (element-wise)
    case OpType::ReLU:
    case OpType::LeakyReLU:
    case OpType::SiLU:
    case OpType::Sigmoid:
    case OpType::Tanh:
    case OpType::GELU:
        return true;
    default:
        return false;
    }
}

// Check if operation is a reduction
inline bool is_reduction_op(ops::OpType op) {
    using ops::OpType;
    switch (op) {
    case OpType::Sum:
    case OpType::Mean:
    case OpType::Max:
    case OpType::Min:
    case OpType::ArgMax:
    case OpType::ArgMin:
    case OpType::Any:
    case OpType::All:
    case OpType::Prod:
        return true;
    default:
        return false;
    }
}

// Check if operation is a unary operation
inline bool is_unary_op(ops::OpType op) {
    using ops::OpType;
    switch (op) {
    case OpType::Negate:
    case OpType::Abs:
    case OpType::Sqrt:
    case OpType::Exp:
    case OpType::Log:
    case OpType::Sin:
    case OpType::Cos:
    case OpType::Tan:
    case OpType::Erf:
    case OpType::Sign:
    case OpType::Floor:
    case OpType::Ceil:
    case OpType::Trunc:
    case OpType::Round:
    case OpType::Reciprocal:
    case OpType::Square:
    case OpType::Cbrt:
    case OpType::IsNaN:
    case OpType::IsInf:
    case OpType::IsFinite:
    case OpType::Conj:
    case OpType::Real:
    case OpType::Imag:
    case OpType::ReLU:
    case OpType::LeakyReLU:
    case OpType::SiLU:
    case OpType::Sigmoid:
    case OpType::Tanh:
    case OpType::GELU:
    case OpType::LogicalNot:
        return true;
    default:
        return false;
    }
}

// Check if operation is a binary operation
inline bool is_binary_op(ops::OpType op) {
    using ops::OpType;
    switch (op) {
    case OpType::Add:
    case OpType::Subtract:
    case OpType::Multiply:
    case OpType::Divide:
    case OpType::Power:
    case OpType::Modulo:
    case OpType::Maximum:
    case OpType::Minimum:
    case OpType::Atan2:
    case OpType::Hypot:
    case OpType::Equal:
    case OpType::NotEqual:
    case OpType::Less:
    case OpType::LessEqual:
    case OpType::Greater:
    case OpType::GreaterEqual:
    case OpType::LogicalAnd:
    case OpType::LogicalOr:
    case OpType::LogicalXor:
    case OpType::BitwiseAnd:
    case OpType::BitwiseOr:
    case OpType::BitwiseXor:
    case OpType::LeftShift:
    case OpType::RightShift:
        return true;
    default:
        return false;
    }
}

// ============================================================================
// Fused Operation Representation
// ============================================================================

// Represents a chain of operations that can be executed together
struct FusedOpChain {
    // Sequence of operations to apply (in order)
    std::vector<ops::OpType> ops;

    // For each op, which inputs to use (index into input_nodes or previous
    // result) -1 means use the result of the previous op in the chain
    std::vector<std::vector<int>> input_indices;

    // Input nodes (constants or other graph nodes)
    std::vector<std::shared_ptr<GraphNode>> input_nodes;

    // Output metadata
    Shape output_shape;
    DType output_dtype;
    Device target_device;

    // Original nodes that were fused (for debugging/analysis)
    std::vector<GraphNode *> original_nodes;

    // In-place execution support
    // If can_execute_inplace is true, the chain can reuse the first input's
    // storage for output (when that input has ref_count == 1)
    bool can_execute_inplace = false;
    int inplace_input_index = -1; // Which input can be reused (-1 = none)

    // Check if this is a simple unary chain (single input, all unary ops)
    bool is_unary_chain() const {
        if (input_nodes.size() != 1)
            return false;
        for (auto op : ops) {
            if (!is_unary_op(op))
                return false;
        }
        return true;
    }

    // Check if this is a binary+unary pattern like (a+b).relu()
    bool is_binary_then_unary() const {
        if (ops.empty())
            return false;
        if (!is_binary_op(ops[0]))
            return false;
        for (size_t i = 1; i < ops.size(); ++i) {
            if (!is_unary_op(ops[i]))
                return false;
        }
        return true;
    }
};

// Common fused patterns for special-case optimization
enum class FusedPattern {
    None,
    // Binary + Unary (2 inputs)
    AddReLU,    // relu(a + b)
    SubAbs,     // |a - b|
    AddSquare,  // (a + b)^2
    MulReLU,    // relu(a * b)
    SubSquare,  // (a - b)^2
    AddSigmoid, // sigmoid(a + b)
    MulSigmoid, // sigmoid(a * b)

    // Ternary (3 inputs)
    MulAdd,         // a * b + c (FMA)
    MulSub,         // a * b - c
    AddMulReLU,     // relu((a + b) * c)
    SubMulAbs,      // |((a - b) * c)|
    ScaleShiftReLU, // relu(a * scale + bias)

    // Special patterns
    BiasReLU, // add bias then relu (same as AddReLU)
    ExpSum,   // exp then sum (softmax helper)
    Softmax,  // full softmax: exp(x - max) / sum(exp(x - max))
};

// Detect known fusion patterns
inline FusedPattern detect_pattern(const FusedOpChain &chain) {
    // 2-op patterns (Binary + Unary)
    if (chain.ops.size() == 2) {
        auto op0 = chain.ops[0];
        auto op1 = chain.ops[1];

        // Add + activation
        if (op0 == ops::OpType::Add && op1 == ops::OpType::ReLU)
            return FusedPattern::AddReLU;
        if (op0 == ops::OpType::Add && op1 == ops::OpType::Square)
            return FusedPattern::AddSquare;
        if (op0 == ops::OpType::Add && op1 == ops::OpType::Sigmoid)
            return FusedPattern::AddSigmoid;

        // Sub + activation
        if (op0 == ops::OpType::Subtract && op1 == ops::OpType::Abs)
            return FusedPattern::SubAbs;
        if (op0 == ops::OpType::Subtract && op1 == ops::OpType::Square)
            return FusedPattern::SubSquare;

        // Mul + activation
        if (op0 == ops::OpType::Multiply && op1 == ops::OpType::ReLU)
            return FusedPattern::MulReLU;
        if (op0 == ops::OpType::Multiply && op1 == ops::OpType::Sigmoid)
            return FusedPattern::MulSigmoid;
    }

    // 3-op patterns (Ternary)
    if (chain.ops.size() == 3) {
        auto op0 = chain.ops[0];
        auto op1 = chain.ops[1];
        auto op2 = chain.ops[2];

        // Scale-shift-activation: mul -> add -> activation
        if (op0 == ops::OpType::Multiply && op1 == ops::OpType::Add) {
            if (op2 == ops::OpType::ReLU)
                return FusedPattern::ScaleShiftReLU;
        }

        // Mul -> Add/Sub (FMA patterns)
        if (op0 == ops::OpType::Multiply && op1 == ops::OpType::Add &&
            op2 == ops::OpType::ReLU)
            return FusedPattern::ScaleShiftReLU;

        // Add/Sub -> Mul -> activation
        if (op0 == ops::OpType::Add && op1 == ops::OpType::Multiply &&
            op2 == ops::OpType::ReLU)
            return FusedPattern::AddMulReLU;
        if (op0 == ops::OpType::Subtract && op1 == ops::OpType::Multiply &&
            op2 == ops::OpType::Abs)
            return FusedPattern::SubMulAbs;
    }

    // Check for MulAdd/MulSub (2 ops but ternary inputs)
    if (chain.ops.size() == 2 && chain.input_nodes.size() >= 3) {
        if (chain.ops[0] == ops::OpType::Multiply &&
            chain.ops[1] == ops::OpType::Add)
            return FusedPattern::MulAdd;
        if (chain.ops[0] == ops::OpType::Multiply &&
            chain.ops[1] == ops::OpType::Subtract)
            return FusedPattern::MulSub;
    }

    return FusedPattern::None;
}

} // namespace graph
} // namespace axiom
