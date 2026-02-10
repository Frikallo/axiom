#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"
#include "axiom/shape.hpp"
#include "axiom/storage.hpp"
#include "op_traits.hpp"

namespace axiom {
namespace graph {

// Forward declaration
class GraphRegistry;

// ============================================================================
// Operation-specific parameter types
// ============================================================================

struct NoParams {};

struct ReductionParams {
    std::vector<int> axes;
    bool keep_dims = false;
};

struct MatMulParams {
    bool transpose_a = false;
    bool transpose_b = false;
};

struct ActivationParams {
    float alpha = 0.01f; // LeakyReLU
    int axis = -1;       // Softmax/LogSoftmax
};

struct ReshapeParams {
    Shape new_shape;
};

using OpParams = std::variant<NoParams, ReductionParams, MatMulParams,
                              ActivationParams, ReshapeParams>;

// Convenience accessor: returns T& if the variant holds T, throws otherwise
template <typename T> const T &get_params(const OpParams &p) {
    return std::get<T>(p);
}

// Represents a deferred computation node in the lazy evaluation graph
struct GraphNode {
    // Unique identifier for this node
    uint64_t id;

    // The operation this node represents
    ops::OpType op_type{};

    // Input nodes (other lazy tensors or constant nodes)
    std::vector<std::shared_ptr<GraphNode>> inputs;

    // Computed metadata (no allocation yet)
    Shape output_shape;
    DType output_dtype{};
    Device target_device = Device::CPU;

    // Operation-specific parameters
    OpParams params;

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

};

// detect_pattern() is declared in fused_patterns.hpp

} // namespace graph
} // namespace axiom
