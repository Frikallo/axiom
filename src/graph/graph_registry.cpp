#include "axiom/graph/graph_registry.hpp"
#include "axiom/error.hpp"
#include "axiom/graph/compiled_graph.hpp"
#include "axiom/graph/graph_cache.hpp"
#include "axiom/graph/graph_executor.hpp"
#include "axiom/graph/graph_signature.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"
#include "axiom/type_conversion.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <unordered_set>

namespace axiom {
namespace graph {

// Thread-local storage for pending nodes
thread_local std::vector<std::weak_ptr<GraphNode>>
    GraphRegistry::pending_nodes_;

// Global settings — atomic to avoid data races across threads
static std::atomic<bool> g_eager_mode{false};
static std::atomic<size_t> g_max_pending_nodes{10000};

// Thread-local eager mode for EagerModeScope
static thread_local bool tl_eager_mode = false;

bool is_eager_mode_enabled() {
    // Check environment variable once (thread-safe static init per C++11)
    static const bool env_eager = [] {
        const char *env = std::getenv("AXIOM_EAGER_MODE");
        return env != nullptr && std::string(env) == "1";
    }();
    return env_eager || g_eager_mode.load(std::memory_order_relaxed) ||
           tl_eager_mode;
}

size_t get_max_pending_nodes() {
    return g_max_pending_nodes.load(std::memory_order_relaxed);
}

void set_max_pending_nodes(size_t max_nodes) {
    g_max_pending_nodes.store(max_nodes, std::memory_order_relaxed);
}

// ============================================================================
// Shape Inference
// ============================================================================

Shape infer_binary_shape(const Shape &lhs, const Shape &rhs) {
    // NumPy-style broadcasting
    size_t max_dims = std::max(lhs.size(), rhs.size());
    Shape result(max_dims);

    for (size_t i = 0; i < max_dims; ++i) {
        size_t lhs_idx =
            (i < max_dims - lhs.size()) ? 0 : i - (max_dims - lhs.size());
        size_t rhs_idx =
            (i < max_dims - rhs.size()) ? 0 : i - (max_dims - rhs.size());

        size_t lhs_dim = (i < max_dims - lhs.size()) ? 1 : lhs[lhs_idx];
        size_t rhs_dim = (i < max_dims - rhs.size()) ? 1 : rhs[rhs_idx];

        if (lhs_dim == rhs_dim) {
            result[i] = lhs_dim;
        } else if (lhs_dim == 1) {
            result[i] = rhs_dim;
        } else if (rhs_dim == 1) {
            result[i] = lhs_dim;
        } else {
            throw ShapeError::broadcast_incompatible(
                "shapes are not broadcastable");
        }
    }
    return result;
}

Shape infer_unary_shape(const Shape &input) { return input; }

// Check if op is argmax/argmin which return empty shape for scalar
static bool is_argmax_argmin_op(ops::OpType op) {
    return op == ops::OpType::ArgMax || op == ops::OpType::ArgMin;
}

Shape infer_reduction_shape(const Shape &input, const std::vector<int> &axes,
                            bool keep_dims, ops::OpType op = ops::OpType::Sum) {
    std::vector<int> norm_axes = axes;

    // If no axes specified, reduce all dimensions
    if (norm_axes.empty()) {
        if (keep_dims) {
            return Shape(input.size(), 1);
        }
        // ArgMax/ArgMin return empty shape for scalar, others return {1}
        if (is_argmax_argmin_op(op)) {
            return Shape{};
        }
        return Shape{1};
    }

    // Normalize negative axes
    for (int &ax : norm_axes) {
        if (ax < 0) {
            ax += static_cast<int>(input.size());
        }
    }

    // Sort axes for consistent processing
    std::sort(norm_axes.begin(), norm_axes.end());

    Shape result;
    for (size_t i = 0; i < input.size(); ++i) {
        bool is_reduced = std::find(norm_axes.begin(), norm_axes.end(),
                                    static_cast<int>(i)) != norm_axes.end();
        if (is_reduced) {
            if (keep_dims) {
                result.push_back(1);
            }
        } else {
            result.push_back(input[i]);
        }
    }

    // If all dimensions are reduced (result is empty)
    if (result.empty()) {
        // ArgMax/ArgMin return empty shape for scalar, others return {1}
        if (is_argmax_argmin_op(op)) {
            return Shape{};
        }
        return Shape{1};
    }
    return result;
}

Shape infer_matmul_shape(const Shape &a, const Shape &b, bool transpose_a,
                         bool transpose_b) {
    // Get effective shapes after transpose
    Shape a_shape = a;
    Shape b_shape = b;

    if (a_shape.size() < 2) {
        a_shape.insert(a_shape.begin(), 1);
    }
    if (b_shape.size() < 2) {
        b_shape.push_back(1);
    }

    size_t a_rows =
        transpose_a ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
    size_t a_cols =
        transpose_a ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
    size_t b_rows =
        transpose_b ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
    size_t b_cols =
        transpose_b ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];

    if (a_cols != b_rows) {
        throw ShapeError("Matrix dimensions don't match for matmul");
    }

    // Handle batch dimensions
    Shape result;
    size_t a_batch_dims = a_shape.size() - 2;
    size_t b_batch_dims = b_shape.size() - 2;
    size_t max_batch_dims = std::max(a_batch_dims, b_batch_dims);

    // Broadcast batch dimensions
    for (size_t i = 0; i < max_batch_dims; ++i) {
        size_t a_idx = (i < max_batch_dims - a_batch_dims)
                           ? 0
                           : i - (max_batch_dims - a_batch_dims);
        size_t b_idx = (i < max_batch_dims - b_batch_dims)
                           ? 0
                           : i - (max_batch_dims - b_batch_dims);

        size_t a_dim = (i < max_batch_dims - a_batch_dims) ? 1 : a_shape[a_idx];
        size_t b_dim = (i < max_batch_dims - b_batch_dims) ? 1 : b_shape[b_idx];

        if (a_dim == b_dim) {
            result.push_back(a_dim);
        } else if (a_dim == 1) {
            result.push_back(b_dim);
        } else if (b_dim == 1) {
            result.push_back(a_dim);
        } else {
            throw ShapeError("Batch dimensions don't match for matmul");
        }
    }

    result.push_back(a_rows);
    result.push_back(b_cols);

    // Squeeze out dimensions if original inputs were 1D
    // For 1D x 1D -> scalar (0-dimensional, empty shape)
    // For 1D x 2D -> 1D (remove first dimension)
    // For 2D x 1D -> 1D (remove last dimension)
    if (a.size() == 1 && result.size() > 1) {
        result.erase(result.begin() + result.size() - 2);
    }
    if (b.size() == 1 && !result.empty()) {
        result.pop_back();
    }

    // Note: if both a and b were 1D, result is now empty, which is correct
    // for a 0-dimensional scalar tensor (matches eager matmul behavior)

    return result;
}

// ============================================================================
// Dtype Inference
// ============================================================================

DType infer_binary_dtype(DType lhs, DType rhs) {
    // Use ops::promote_types for consistency with eager execution path
    return ops::promote_types(lhs, rhs);
}

// Infer output dtype for binary operations considering the op type
DType infer_binary_output_dtype(ops::OpType op, DType lhs, DType rhs) {
    // Comparison and logical operations return bool
    if (is_comparison_op(op) || is_logical_op(op)) {
        return DType::Bool;
    }
    // Other operations use standard type promotion
    return ops::promote_types(lhs, rhs);
}

DType infer_unary_dtype(ops::OpType op, DType input) {
    using ops::OpType;

    // Operations that always return bool
    switch (op) {
    case OpType::IsNaN:
    case OpType::IsInf:
    case OpType::IsFinite:
    case OpType::Equal:
    case OpType::NotEqual:
    case OpType::Less:
    case OpType::LessEqual:
    case OpType::Greater:
    case OpType::GreaterEqual:
    case OpType::LogicalNot:
        return DType::Bool;
    default:
        break;
    }

    // Most unary operations preserve dtype
    return input;
}

DType infer_reduction_dtype(ops::OpType op, DType input) {
    using ops::OpType;

    // ArgMax/ArgMin return Int64
    if (op == OpType::ArgMax || op == OpType::ArgMin) {
        return DType::Int64;
    }

    // Any/All return Bool
    if (op == OpType::Any || op == OpType::All) {
        return DType::Bool;
    }

    // Mean always returns float
    if (op == OpType::Mean) {
        if (input == DType::Float64 || input == DType::Complex128) {
            return input;
        }
        if (input == DType::Complex64) {
            return DType::Complex64;
        }
        return DType::Float32;
    }

    // Other reductions preserve dtype
    return input;
}

// ============================================================================
// Helper to create a constant node from a materialized tensor
// ============================================================================

static std::shared_ptr<GraphNode> make_constant_node(const Tensor &t) {
    auto node = std::make_shared<GraphNode>();
    node->is_constant = true;
    node->constant_storage = t.storage();
    node->constant_strides = t.strides();
    node->constant_offset = t.offset();
    node->output_shape = t.shape();
    node->output_dtype = t.dtype();
    node->target_device = t.device();
    node->is_materialized_ = true;
    node->cached_result_ = t.storage();
    node->cached_shape_ = t.shape();
    node->cached_strides_ = t.strides();
    return node;
}

// ============================================================================
// GraphRegistry Implementation
// ============================================================================

// Helper to check if a tensor is lazy
static bool tensor_is_lazy(const Tensor &t);

// Helper to get lazy node from tensor
static std::shared_ptr<GraphNode> get_lazy_node(const Tensor &t);

// Helper to create tensor from lazy node
static Tensor create_tensor_from_node(std::shared_ptr<GraphNode> node);

// Forward declarations - these will be implemented after Tensor modifications
extern bool tensor_is_lazy_impl(const Tensor &t);
extern std::shared_ptr<GraphNode> get_lazy_node_impl(const Tensor &t);
extern Tensor create_tensor_from_node_impl(std::shared_ptr<GraphNode> node);

static bool tensor_is_lazy(const Tensor &t) { return tensor_is_lazy_impl(t); }

static std::shared_ptr<GraphNode> get_lazy_node(const Tensor &t) {
    return get_lazy_node_impl(t);
}

static Tensor create_tensor_from_node(std::shared_ptr<GraphNode> node) {
    return create_tensor_from_node_impl(std::move(node));
}

// Helper to create a tensor from a materialized node (returns proper tensor,
// not lazy)
static Tensor create_materialized_tensor(GraphNode *node) {
    if (!node->is_materialized_) {
        throw RuntimeError("Cannot create tensor from non-materialized node");
    }
    return Tensor(node->cached_result_, node->cached_shape_,
                  node->cached_strides_, node->output_dtype);
}

Tensor GraphRegistry::create_lazy_unary(ops::OpType op, const Tensor &input,
                                        const OpParams &params) {
    // In eager mode, execute immediately and return a materialized tensor
    if (is_eager_mode_enabled()) {
        auto node = std::make_shared<GraphNode>();
        node->op_type = op;
        node->params = params;
        node->output_shape = infer_unary_shape(input.shape());
        node->output_dtype = infer_unary_dtype(op, input.dtype());
        node->target_device = input.device();

        if (tensor_is_lazy(input)) {
            node->inputs.push_back(get_lazy_node(input));
        } else {
            node->inputs.push_back(make_constant_node(input));
        }

        materialize(node.get());
        // Return a proper materialized tensor, not a lazy one
        return create_materialized_tensor(node.get());
    }

    auto node = std::make_shared<GraphNode>();
    node->op_type = op;
    node->params = params;
    node->output_shape = infer_unary_shape(input.shape());
    node->output_dtype = infer_unary_dtype(op, input.dtype());
    node->target_device = input.device();

    // Add input node
    if (tensor_is_lazy(input)) {
        auto input_node = get_lazy_node(input);
        node->inputs.push_back(input_node);
        input_node->ref_count++;
    } else {
        node->inputs.push_back(make_constant_node(input));
    }

    // Track this node
    pending_nodes_.push_back(node);

    // Auto-materialize if too many pending nodes
    if (pending_nodes_.size() >
        g_max_pending_nodes.load(std::memory_order_relaxed)) {
        materialize(node.get());
    }

    return create_tensor_from_node(node);
}

Tensor GraphRegistry::create_lazy_binary(ops::OpType op, const Tensor &lhs,
                                         const Tensor &rhs,
                                         const OpParams &params) {
    if (is_eager_mode_enabled()) {
        auto node = std::make_shared<GraphNode>();
        node->op_type = op;
        node->params = params;
        node->output_shape = infer_binary_shape(lhs.shape(), rhs.shape());
        // Use infer_binary_output_dtype to handle comparison/logical ops
        node->output_dtype =
            infer_binary_output_dtype(op, lhs.dtype(), rhs.dtype());
        node->target_device =
            (lhs.device() == Device::GPU || rhs.device() == Device::GPU)
                ? Device::GPU
                : Device::CPU;

        if (tensor_is_lazy(lhs)) {
            node->inputs.push_back(get_lazy_node(lhs));
        } else {
            node->inputs.push_back(make_constant_node(lhs));
        }
        if (tensor_is_lazy(rhs)) {
            node->inputs.push_back(get_lazy_node(rhs));
        } else {
            node->inputs.push_back(make_constant_node(rhs));
        }

        materialize(node.get());
        return create_materialized_tensor(node.get());
    }

    auto node = std::make_shared<GraphNode>();
    node->op_type = op;
    node->params = params;
    node->output_shape = infer_binary_shape(lhs.shape(), rhs.shape());
    // Use infer_binary_output_dtype to handle comparison/logical ops
    node->output_dtype =
        infer_binary_output_dtype(op, lhs.dtype(), rhs.dtype());
    node->target_device =
        (lhs.device() == Device::GPU || rhs.device() == Device::GPU)
            ? Device::GPU
            : Device::CPU;

    // Add input nodes
    if (tensor_is_lazy(lhs)) {
        auto lhs_node = get_lazy_node(lhs);
        node->inputs.push_back(lhs_node);
        lhs_node->ref_count++;
    } else {
        node->inputs.push_back(make_constant_node(lhs));
    }

    if (tensor_is_lazy(rhs)) {
        auto rhs_node = get_lazy_node(rhs);
        node->inputs.push_back(rhs_node);
        rhs_node->ref_count++;
    } else {
        node->inputs.push_back(make_constant_node(rhs));
    }

    pending_nodes_.push_back(node);

    if (pending_nodes_.size() >
        g_max_pending_nodes.load(std::memory_order_relaxed)) {
        materialize(node.get());
    }

    return create_tensor_from_node(node);
}

Tensor GraphRegistry::create_lazy_reduction(ops::OpType op, const Tensor &input,
                                            const std::vector<int> &axes,
                                            bool keep_dims) {
    if (is_eager_mode_enabled()) {
        auto node = std::make_shared<GraphNode>();
        node->op_type = op;
        node->params = ReductionParams{axes, keep_dims};
        node->output_shape =
            infer_reduction_shape(input.shape(), axes, keep_dims, op);
        node->output_dtype = infer_reduction_dtype(op, input.dtype());
        node->target_device = input.device();

        if (tensor_is_lazy(input)) {
            node->inputs.push_back(get_lazy_node(input));
        } else {
            node->inputs.push_back(make_constant_node(input));
        }

        materialize(node.get());
        return create_materialized_tensor(node.get());
    }

    auto node = std::make_shared<GraphNode>();
    node->op_type = op;
    node->params = ReductionParams{axes, keep_dims};
    node->output_shape =
        infer_reduction_shape(input.shape(), axes, keep_dims, op);
    node->output_dtype = infer_reduction_dtype(op, input.dtype());
    node->target_device = input.device();

    if (tensor_is_lazy(input)) {
        auto input_node = get_lazy_node(input);
        node->inputs.push_back(input_node);
        input_node->ref_count++;
    } else {
        node->inputs.push_back(make_constant_node(input));
    }

    pending_nodes_.push_back(node);

    if (pending_nodes_.size() >
        g_max_pending_nodes.load(std::memory_order_relaxed)) {
        materialize(node.get());
    }

    return create_tensor_from_node(node);
}

Tensor GraphRegistry::create_lazy_matmul(const Tensor &a, const Tensor &b,
                                         bool transpose_a, bool transpose_b) {
    if (is_eager_mode_enabled()) {
        auto node = std::make_shared<GraphNode>();
        node->op_type = ops::OpType::MatMul;
        node->params = MatMulParams{transpose_a, transpose_b};
        node->output_shape =
            infer_matmul_shape(a.shape(), b.shape(), transpose_a, transpose_b);
        node->output_dtype = infer_binary_dtype(a.dtype(), b.dtype());
        node->target_device =
            (a.device() == Device::GPU || b.device() == Device::GPU)
                ? Device::GPU
                : Device::CPU;

        if (tensor_is_lazy(a)) {
            node->inputs.push_back(get_lazy_node(a));
        } else {
            node->inputs.push_back(make_constant_node(a));
        }
        if (tensor_is_lazy(b)) {
            node->inputs.push_back(get_lazy_node(b));
        } else {
            node->inputs.push_back(make_constant_node(b));
        }

        materialize(node.get());
        return create_materialized_tensor(node.get());
    }

    auto node = std::make_shared<GraphNode>();
    node->op_type = ops::OpType::MatMul;
    node->params = MatMulParams{transpose_a, transpose_b};
    node->output_shape =
        infer_matmul_shape(a.shape(), b.shape(), transpose_a, transpose_b);
    node->output_dtype = infer_binary_dtype(a.dtype(), b.dtype());
    node->target_device =
        (a.device() == Device::GPU || b.device() == Device::GPU) ? Device::GPU
                                                                 : Device::CPU;

    if (tensor_is_lazy(a)) {
        auto a_node = get_lazy_node(a);
        node->inputs.push_back(a_node);
        a_node->ref_count++;
    } else {
        node->inputs.push_back(make_constant_node(a));
    }

    if (tensor_is_lazy(b)) {
        auto b_node = get_lazy_node(b);
        node->inputs.push_back(b_node);
        b_node->ref_count++;
    } else {
        node->inputs.push_back(make_constant_node(b));
    }

    pending_nodes_.push_back(node);

    if (pending_nodes_.size() >
        g_max_pending_nodes.load(std::memory_order_relaxed)) {
        materialize(node.get());
    }

    return create_tensor_from_node(node);
}

// Collect constant/materialized input tensors in topological order,
// matching the order expected by the compiled plan's input_slots.
static std::vector<Tensor>
collect_constant_inputs(const GraphNode *root, const CompiledGraph & /*plan*/) {
    // Walk the graph in topological order and collect constants.
    // Stop at constant/materialized boundaries (don't recurse into
    // their inputs) to match the compiler's topological sort.
    std::vector<const GraphNode *> sorted;
    std::unordered_set<const GraphNode *> visited;
    std::vector<std::pair<const GraphNode *, size_t>> stack;
    stack.push_back({root, 0});

    while (!stack.empty()) {
        auto &[n, idx] = stack.back();
        if (visited.count(n)) {
            stack.pop_back();
            continue;
        }
        // Leaf boundary: include but don't recurse
        if (n != root && (n->is_constant || n->is_materialized_)) {
            visited.insert(n);
            sorted.push_back(n);
            stack.pop_back();
            continue;
        }
        if (idx < n->inputs.size()) {
            const GraphNode *child = n->inputs[idx].get();
            idx++;
            if (!visited.count(child))
                stack.push_back({child, 0});
        } else {
            visited.insert(n);
            sorted.push_back(n);
            stack.pop_back();
        }
    }

    // Collect input tensors from constant/materialized nodes
    std::vector<Tensor> inputs;
    std::unordered_set<const GraphNode *> seen;
    for (const auto *n : sorted) {
        if (n->is_constant || n->is_materialized_) {
            if (seen.insert(n).second) {
                if (n->is_constant) {
                    inputs.emplace_back(n->constant_storage, n->output_shape,
                                        n->constant_strides, n->output_dtype,
                                        n->constant_offset);
                } else {
                    inputs.emplace_back(n->cached_result_, n->cached_shape_,
                                        n->cached_strides_, n->output_dtype);
                }
            }
        }
    }
    return inputs;
}

void GraphRegistry::materialize(GraphNode *node) {
    if (node->is_materialized_)
        return;

    // Walk the subgraph and materialize shared nodes (ref_count > 1)
    // first so their results are preserved for other consumers.
    // Single-consumer intermediates are left unmaterialized so the
    // compiler can see the full chain and fuse operations.
    {
        std::vector<GraphNode *> shared_topo;
        std::unordered_set<GraphNode *> visited;
        std::vector<std::pair<GraphNode *, size_t>> stk;
        stk.push_back({node, 0});

        while (!stk.empty()) {
            auto &[n, idx] = stk.back();
            if (visited.count(n)) {
                stk.pop_back();
                continue;
            }
            if (idx < n->inputs.size()) {
                GraphNode *child = n->inputs[idx].get();
                idx++;
                if (!visited.count(child) && !child->is_constant &&
                    !child->is_materialized_) {
                    stk.push_back({child, 0});
                }
            } else {
                visited.insert(n);
                if (n != node && !n->is_constant && !n->is_materialized_ &&
                    n->ref_count > 1) {
                    shared_topo.push_back(n);
                }
                stk.pop_back();
            }
        }

        for (auto *shared : shared_topo) {
            if (!shared->is_materialized_) {
                materialize(shared);
            }
        }
    }

    // Compute structural signature of the graph
    auto sig = compute_signature(node);

    // Get compiled plan from cache (compile on miss)
    auto plan = GraphCache::instance().get_or_compile(sig, node);

    // Collect input tensors from constant/materialized nodes
    auto inputs = collect_constant_inputs(node, *plan);

    // Execute the compiled plan (hot path — no graph walking)
    auto result = execute(*plan, inputs);

    // Store result back into node
    node->cached_result_ = result.storage();
    node->cached_shape_ = result.shape();
    node->cached_strides_ = result.strides();
    node->is_materialized_ = true;

    // Decrement ref_count for inputs — this node no longer needs them
    // for fusion purposes since its result is now computed.
    for (auto &inp : node->inputs) {
        if (inp && !inp->is_constant && inp->ref_count > 0) {
            inp->ref_count--;
        }
    }

    // Clean up pending nodes that are now materialized
    pending_nodes_.erase(std::remove_if(pending_nodes_.begin(),
                                        pending_nodes_.end(),
                                        [](const std::weak_ptr<GraphNode> &wp) {
                                            auto sp = wp.lock();
                                            return !sp || sp->is_materialized_;
                                        }),
                         pending_nodes_.end());
}

size_t GraphRegistry::pending_node_count() {
    // Clean up expired weak pointers
    pending_nodes_.erase(std::remove_if(pending_nodes_.begin(),
                                        pending_nodes_.end(),
                                        [](const std::weak_ptr<GraphNode> &wp) {
                                            return wp.expired();
                                        }),
                         pending_nodes_.end());
    return pending_nodes_.size();
}

void GraphRegistry::materialize_all() {
    // Copy to local vector first — materialize() mutates pending_nodes_
    // via erase-remove, so iterating the original is undefined behavior.
    auto snapshot = pending_nodes_;
    for (auto &wp : snapshot) {
        if (auto node = wp.lock()) {
            if (!node->is_materialized_) {
                materialize(node.get());
            }
        }
    }
    pending_nodes_.clear();
}

// ============================================================================
// EagerModeScope Implementation
// ============================================================================

EagerModeScope::EagerModeScope() : previous_mode_(tl_eager_mode) {
    tl_eager_mode = true;
}

EagerModeScope::~EagerModeScope() { tl_eager_mode = previous_mode_; }

} // namespace graph
} // namespace axiom
