#include "axiom/graph/graph_compiler.hpp"
#include "axiom/graph/graph_node.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace axiom {
namespace graph {

// ============================================================================
// Topological Sort (iterative)
// ============================================================================

std::vector<const GraphNode *>
GraphCompiler::topological_sort(const GraphNode *root) {
    std::vector<const GraphNode *> result;
    std::unordered_set<const GraphNode *> visited;
    std::vector<std::pair<const GraphNode *, size_t>> stack;

    stack.push_back({root, 0});

    while (!stack.empty()) {
        auto &[node, idx] = stack.back();

        if (visited.count(node)) {
            stack.pop_back();
            continue;
        }

        if (idx < node->inputs.size()) {
            const GraphNode *child = node->inputs[idx].get();
            idx++;
            if (!visited.count(child)) {
                stack.push_back({child, 0});
            }
        } else {
            visited.insert(node);
            result.push_back(node);
            stack.pop_back();
        }
    }

    return result;
}

// ============================================================================
// Dead Code Elimination
// ============================================================================

std::vector<const GraphNode *>
GraphCompiler::dead_code_elimination(
    const std::vector<const GraphNode *> &sorted,
    const GraphNode *root) {

    // Walk backward from root to mark reachable nodes
    std::unordered_set<const GraphNode *> reachable;
    std::vector<const GraphNode *> worklist;
    worklist.push_back(root);
    reachable.insert(root);

    while (!worklist.empty()) {
        const GraphNode *node = worklist.back();
        worklist.pop_back();
        for (const auto &inp : node->inputs) {
            if (reachable.insert(inp.get()).second) {
                worklist.push_back(inp.get());
            }
        }
    }

    // Filter sorted to only reachable nodes
    std::vector<const GraphNode *> result;
    result.reserve(sorted.size());
    for (const auto *n : sorted) {
        if (reachable.count(n)) {
            result.push_back(n);
        }
    }
    return result;
}

// ============================================================================
// Fusion Analysis
// ============================================================================

// Check if two nodes can be fused (producer into consumer)
static bool can_fuse_pair(const GraphNode *producer,
                          const GraphNode *consumer) {
    if (!is_elementwise_op(producer->op_type) ||
        !is_elementwise_op(consumer->op_type))
        return false;

    // Producer must have single consumer
    if (producer->ref_count > 1)
        return false;

    // Same device
    if (producer->target_device != consumer->target_device)
        return false;

    // Compatible shapes
    if (producer->output_shape != consumer->output_shape) {
        if (!ShapeUtils::broadcastable(producer->output_shape,
                                       consumer->output_shape))
            return false;
        Shape bcast = ShapeUtils::broadcast_shape(
            producer->output_shape, consumer->output_shape);
        if (bcast != consumer->output_shape)
            return false;
    }

    // Don't fuse already-materialized nodes
    if (producer->is_materialized_)
        return false;

    return true;
}

std::vector<GraphCompiler::FusionGroup>
GraphCompiler::fusion_analysis(
    const std::vector<const GraphNode *> &sorted) {

    std::unordered_set<const GraphNode *> assigned;
    std::vector<FusionGroup> groups;

    // Process in reverse topological order (consumers first)
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        const GraphNode *node = *it;

        if (assigned.count(node))
            continue;
        if (node->is_constant || node->is_materialized_)
            continue;

        // Non-elementwise ops get their own single-node group
        if (!is_elementwise_op(node->op_type)) {
            assigned.insert(node);
            FusionGroup group;
            group.nodes = {node};
            group.pattern = FusedPattern::None;
            group.is_fused = false;
            groups.push_back(std::move(group));
            continue;
        }

        // Try to build a chain going backward through first input
        std::vector<const GraphNode *> chain;
        const GraphNode *current = node;

        while (current && !assigned.count(current)) {
            if (!is_elementwise_op(current->op_type))
                break;
            if (current->is_constant || current->is_materialized_)
                break;

            chain.push_back(current);
            assigned.insert(current);

            const GraphNode *next = nullptr;
            if (!current->inputs.empty()) {
                const auto *first_inp = current->inputs[0].get();
                if (!first_inp->is_constant &&
                    !first_inp->is_materialized_ &&
                    can_fuse_pair(first_inp, current)) {
                    next = first_inp;
                }
            }
            current = next;
        }

        // Reverse to execution order (producer first)
        std::reverse(chain.begin(), chain.end());

        FusionGroup group;
        group.nodes = std::move(chain);
        group.is_fused = group.nodes.size() > 1;

        // Detect known pattern if fused
        if (group.is_fused) {
            FusedOpChain foc;
            for (const auto *n : group.nodes) {
                foc.ops.push_back(n->op_type);
            }
            // Build input_nodes list for pattern detection
            std::unordered_set<const GraphNode *> chain_set(
                group.nodes.begin(), group.nodes.end());
            for (const auto *n : group.nodes) {
                for (const auto &inp : n->inputs) {
                    if (!chain_set.count(inp.get())) {
                        bool found = false;
                        for (const auto &existing : foc.input_nodes) {
                            if (existing.get() == inp.get()) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            foc.input_nodes.push_back(inp);
                        }
                    }
                }
            }
            // Build input_indices for pattern detection
            for (size_t i = 0; i < group.nodes.size(); ++i) {
                std::vector<int> indices;
                const auto *n = group.nodes[i];
                for (const auto &inp : n->inputs) {
                    if (i > 0 && inp.get() == group.nodes[i - 1]) {
                        indices.push_back(-1);
                    } else {
                        for (size_t j = 0; j < foc.input_nodes.size(); ++j) {
                            if (foc.input_nodes[j].get() == inp.get()) {
                                indices.push_back(static_cast<int>(j));
                                break;
                            }
                        }
                    }
                }
                foc.input_indices.push_back(indices);
            }
            group.pattern = detect_pattern(foc);
        } else {
            group.pattern = FusedPattern::None;
        }

        groups.push_back(std::move(group));
    }

    // Reverse so groups are in topological order
    std::reverse(groups.begin(), groups.end());
    return groups;
}

// ============================================================================
// Memory Planning
// ============================================================================

void GraphCompiler::memory_plan(CompiledGraph &plan) {
    auto &slots = plan.buffer_slots;

    // Compute liveness: first_use and last_use for each slot
    for (int step_idx = 0; step_idx < static_cast<int>(plan.steps.size());
         ++step_idx) {
        const auto &step = plan.steps[step_idx];

        // Output slot is produced here
        int out = step.output_slot;
        if (out >= 0 && out < static_cast<int>(slots.size())) {
            if (slots[out].first_use < 0)
                slots[out].first_use = step_idx;
            slots[out].last_use = step_idx;
        }

        // Input slots are read here
        for (const auto &per_op : step.input_slot_indices) {
            for (int s : per_op) {
                if (s >= 0 && s < static_cast<int>(slots.size())) {
                    if (slots[s].first_use < 0)
                        slots[s].first_use = step_idx;
                    slots[s].last_use = step_idx;
                }
            }
        }
    }

    // Linear scan allocation: reuse dead slots
    // free_list: sorted by byte_size (ascending)
    struct FreeEntry {
        int alloc_id;
        size_t byte_size;
    };
    std::vector<FreeEntry> free_list;

    int next_alloc = 0;
    plan.slot_to_allocation.resize(slots.size(), -1);

    // Input slots don't need allocation (they come from existing tensors)
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].is_input) {
            plan.slot_to_allocation[i] = -1; // no allocation needed
        }
    }

    for (int step_idx = 0; step_idx < static_cast<int>(plan.steps.size());
         ++step_idx) {

        // Free slots whose last_use < step_idx
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].is_input)
                continue;
            if (plan.slot_to_allocation[i] >= 0 &&
                slots[i].last_use >= 0 &&
                slots[i].last_use < step_idx) {
                // This slot is dead — put its allocation on the free list
                free_list.push_back(
                    {plan.slot_to_allocation[i], slots[i].byte_size});
            }
        }

        // Allocate for output slot of this step
        int out = plan.steps[step_idx].output_slot;
        if (out >= 0 && !slots[out].is_input &&
            plan.slot_to_allocation[out] < 0) {
            size_t needed = slots[out].byte_size;

            // Find smallest free allocation that fits
            int best = -1;
            size_t best_size = SIZE_MAX;
            for (size_t f = 0; f < free_list.size(); ++f) {
                if (free_list[f].byte_size >= needed &&
                    free_list[f].byte_size < best_size) {
                    best = static_cast<int>(f);
                    best_size = free_list[f].byte_size;
                }
            }

            if (best >= 0) {
                plan.slot_to_allocation[out] = free_list[best].alloc_id;
                slots[out].reuses_slot = free_list[best].alloc_id;
                free_list.erase(free_list.begin() + best);
            } else {
                plan.slot_to_allocation[out] = next_alloc++;
            }
        }
    }

    // Compute allocation sizes
    plan.num_allocations = static_cast<size_t>(next_alloc);
    plan.allocation_sizes.resize(plan.num_allocations, 0);
    for (size_t i = 0; i < slots.size(); ++i) {
        int a = plan.slot_to_allocation[i];
        if (a >= 0 && a < static_cast<int>(plan.num_allocations)) {
            plan.allocation_sizes[a] =
                std::max(plan.allocation_sizes[a], slots[i].byte_size);
        }
    }
}

// ============================================================================
// Loop Parameter Computation
// ============================================================================

void GraphCompiler::compute_loop_params(CompiledGraph &plan) {
    for (auto &step : plan.steps) {
        step.total_elements = ShapeUtils::size(step.output_shape);

        // L1 tile size: ~16K float32 elements (64KB per buffer)
        // Apple M-series L1 is 192KB, 2 buffers = 128KB
        size_t elem_size = dtype_size(step.output_dtype);
        if (elem_size > 0) {
            step.tile_size = (64 * 1024) / elem_size;
            if (step.tile_size == 0)
                step.tile_size = 1024;
        } else {
            step.tile_size = 16384;
        }

        // Classify input access patterns
        step.input_access.clear();
        for (const auto &per_op : step.input_slot_indices) {
            for (int slot_idx : per_op) {
                if (slot_idx < 0 ||
                    slot_idx >= static_cast<int>(plan.buffer_slots.size())) {
                    step.input_access.push_back(AccessPattern::Contiguous);
                    continue;
                }
                const auto &slot = plan.buffer_slots[slot_idx];
                if (slot.shape == step.output_shape) {
                    // Check strides for contiguity
                    auto expected = ShapeUtils::calculate_strides(
                        slot.shape,
                        static_cast<int64_t>(dtype_size(slot.dtype)));
                    if (slot.strides == expected) {
                        step.input_access.push_back(
                            AccessPattern::Contiguous);
                    } else {
                        step.input_access.push_back(AccessPattern::Strided);
                    }
                } else if (ShapeUtils::size(slot.shape) == 1) {
                    step.input_access.push_back(
                        AccessPattern::ScalarBroadcast);
                } else {
                    step.input_access.push_back(AccessPattern::Broadcast);
                }
            }
        }
    }
}

// ============================================================================
// Main Compile Pipeline
// ============================================================================

std::shared_ptr<CompiledGraph>
GraphCompiler::compile(const GraphSignature &sig, const GraphNode *root) {
    auto plan = std::make_shared<CompiledGraph>();
    plan->signature = sig;

    // 1. Topological sort
    auto sorted = topological_sort(root);

    // 2. Dead code elimination
    sorted = dead_code_elimination(sorted, root);

    // 3. Fusion analysis
    auto groups = fusion_analysis(sorted);

    // Build node → slot mapping
    std::unordered_map<const GraphNode *, int> node_to_slot;
    int next_slot = 0;

    // First pass: assign slots for all constant/materialized input nodes
    std::vector<int> input_slot_list;
    int input_idx = 0;
    for (const auto *node : sorted) {
        if (node->is_constant || node->is_materialized_) {
            if (!node_to_slot.count(node)) {
                int slot = next_slot++;
                node_to_slot[node] = slot;

                BufferSlot bs;
                bs.byte_size = node->byte_size();
                bs.dtype = node->output_dtype;
                bs.shape = node->output_shape;
                bs.strides = node->is_constant
                                 ? node->constant_strides
                                 : node->cached_strides_;
                bs.device = node->target_device;
                bs.is_input = true;
                bs.input_index = input_idx++;
                plan->buffer_slots.push_back(bs);
                input_slot_list.push_back(slot);
            }
        }
    }
    plan->input_slots = input_slot_list;

    // Second pass: create execution steps from fusion groups
    for (const auto &group : groups) {
        if (group.nodes.empty())
            continue;

        // Skip constant/materialized nodes (they're inputs, not steps)
        bool all_const = true;
        for (const auto *n : group.nodes) {
            if (!n->is_constant && !n->is_materialized_) {
                all_const = false;
                break;
            }
        }
        if (all_const)
            continue;

        const GraphNode *last = group.nodes.back();

        // Allocate output slot
        int out_slot = next_slot++;
        node_to_slot[last] = out_slot;

        BufferSlot out_bs;
        out_bs.byte_size = last->byte_size();
        out_bs.dtype = last->output_dtype;
        out_bs.shape = last->output_shape;
        out_bs.strides = ShapeUtils::calculate_strides(
            last->output_shape,
            static_cast<int64_t>(dtype_size(last->output_dtype)));
        out_bs.device = last->target_device;
        plan->buffer_slots.push_back(out_bs);

        // For intermediate nodes in fused chain, assign the out_slot to
        // all intermediate nodes too (they share the output)
        for (size_t i = 0; i < group.nodes.size() - 1; ++i) {
            // Intermediate nodes get temporary slot assignments if needed
            // by other consumers (but in a fused chain they shouldn't)
            if (!node_to_slot.count(group.nodes[i])) {
                node_to_slot[group.nodes[i]] = out_slot;
            }
        }

        ExecutionStep step;
        step.output_slot = out_slot;
        step.output_shape = last->output_shape;
        step.output_dtype = last->output_dtype;
        step.device = last->target_device;
        step.total_elements = ShapeUtils::size(last->output_shape);

        if (group.is_fused && group.pattern != FusedPattern::None) {
            // Known SIMD pattern
            step.kind = ExecutionStep::Kind::FusedKnown;
            step.pattern = group.pattern;
            for (const auto *n : group.nodes) {
                step.op_chain.push_back(n->op_type);
            }
        } else if (group.is_fused) {
            // Generic fused
            step.kind = ExecutionStep::Kind::FusedGeneric;
            step.pattern = FusedPattern::None;
            for (const auto *n : group.nodes) {
                step.op_chain.push_back(n->op_type);
            }
        } else if (group.nodes.size() == 1) {
            const auto *n = group.nodes[0];
            if (n->op_type == ops::OpType::MatMul ||
                n->op_type == ops::OpType::BatchMatMul) {
                step.kind = ExecutionStep::Kind::SingleOp;
            } else {
                step.kind = ExecutionStep::Kind::SingleOp;
            }
            step.op_type = n->op_type;
            step.params = n->params;
        }

        // Build input slot indices for each op in the chain
        std::unordered_set<const GraphNode *> chain_set(
            group.nodes.begin(), group.nodes.end());

        for (size_t i = 0; i < group.nodes.size(); ++i) {
            const auto *n = group.nodes[i];
            std::vector<int> indices;

            for (const auto &inp : n->inputs) {
                if (i > 0 && inp.get() == group.nodes[i - 1]) {
                    // Previous result in chain
                    indices.push_back(-1);
                } else {
                    auto it = node_to_slot.find(inp.get());
                    if (it != node_to_slot.end()) {
                        indices.push_back(it->second);
                    } else {
                        // Node not yet assigned — it's an external input
                        // that should have been assigned above
                        indices.push_back(-2); // error sentinel
                    }
                }
            }
            step.input_slot_indices.push_back(indices);
        }

        // For single ops, copy params
        if (group.nodes.size() == 1) {
            step.params = group.nodes[0]->params;
        }

        // In-place analysis: check if output can reuse an input slot
        step.can_inplace = false;
        step.inplace_input_slot = -1;
        if (!step.input_slot_indices.empty()) {
            for (const auto &per_op : step.input_slot_indices) {
                for (int s : per_op) {
                    if (s >= 0 &&
                        s < static_cast<int>(plan->buffer_slots.size())) {
                        const auto &slot = plan->buffer_slots[s];
                        if (slot.shape == step.output_shape &&
                            slot.dtype == step.output_dtype &&
                            slot.is_input) {
                            step.can_inplace = true;
                            step.inplace_input_slot = s;
                            break;
                        }
                    }
                }
                if (step.can_inplace)
                    break;
            }
        }

        plan->steps.push_back(std::move(step));
    }

    // Set output slot to the slot of the root node
    auto root_it = node_to_slot.find(root);
    if (root_it != node_to_slot.end()) {
        plan->output_slot = root_it->second;
    }

    // 4. Memory planning
    memory_plan(*plan);

    // 5. Loop parameter computation
    compute_loop_params(*plan);

    return plan;
}

} // namespace graph
} // namespace axiom
