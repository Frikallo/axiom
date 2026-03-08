#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <variant>
#include <vector>

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"
#include "axiom/shape.hpp"
#include "axiom/storage.hpp"
#include "graph_node.hpp"
#include "graph_signature.hpp"

namespace axiom {
namespace graph {

enum class AccessPattern : uint8_t {
    Contiguous,
    Strided,
    Broadcast,
    ScalarBroadcast
};

// ============================================================================
// ExecutionStep: variant of typed step structs
// ============================================================================

struct StepBase {
    int output_slot = -1;
    size_t total_elements = 0;
    Shape output_shape;
    DType output_dtype{};
    Device device = Device::CPU;
    std::vector<std::vector<int>> input_slot_indices;
    std::vector<AccessPattern> input_access;
    size_t tile_size = 0;
};

struct SingleOpStep : StepBase {
    ops::OpType op_type{};
    OpParams params;
};

struct FusedKnownStep : StepBase {
    FusedPattern pattern = FusedPattern::None;
    std::vector<ops::OpType> op_chain;
};

struct FusedGenericStep : StepBase {
    std::vector<ops::OpType> op_chain;
};

struct MatMulActivationStep : StepBase {
    ops::OpType op_type{}; // MatMul or BatchMatMul
    OpParams params;
    std::vector<ops::OpType> op_chain;
};

struct FusedReductionStep : StepBase {
    ops::OpType reduction_op{};
    DType chain_dtype{};
    OpParams params;
    std::vector<ops::OpType> op_chain;
    Shape chain_shape; // pre-reduction shape (elementwise chain output)
};

using ExecutionStep =
    std::variant<SingleOpStep, FusedKnownStep, FusedGenericStep,
                 MatMulActivationStep, FusedReductionStep>;

// Visitor to access common StepBase fields from any variant alternative
inline const StepBase &step_base(const ExecutionStep &step) {
    return std::visit(
        [](const auto &s) -> const StepBase & {
            return static_cast<const StepBase &>(s);
        },
        step);
}

inline StepBase &step_base(ExecutionStep &step) {
    return std::visit(
        [](auto &s) -> StepBase & { return static_cast<StepBase &>(s); }, step);
}

// ============================================================================
// Buffer and arena types
// ============================================================================

struct BufferSlot {
    size_t byte_size;
    DType dtype;
    Shape shape;
    Strides strides;
    Device device;

    int first_use = -1; // Step index that produces this
    int last_use = -1;  // Last step that reads this

    bool is_input = false;
    int input_index = -1; // Index into the flat list of constant inputs

    int reuses_slot = -1; // Memory reuse from a dead slot
};

struct BufferArena {
    std::vector<std::shared_ptr<Storage>> backing; // M allocations
};

struct CompiledGraph {
    GraphSignature signature;
    std::vector<ExecutionStep> steps;
    std::vector<BufferSlot> buffer_slots;
    std::vector<int> input_slots; // Slot indices for external/constant inputs
    int output_slot = -1;
    size_t num_allocations = 0;
    std::vector<int> slot_to_allocation;
    std::vector<size_t> allocation_sizes;

    // Arena pool for buffer reuse across executions
    mutable std::mutex arena_mutex_;
    mutable std::vector<std::unique_ptr<BufferArena>> free_arenas_;

    std::unique_ptr<BufferArena> acquire_arena() const {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        if (free_arenas_.empty())
            return nullptr;
        auto arena = std::move(free_arenas_.back());
        free_arenas_.pop_back();
        return arena;
    }

    void release_arena(std::unique_ptr<BufferArena> arena) const {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        // Cap pool size to prevent unbounded growth under concurrent load
        static constexpr size_t MAX_FREE_ARENAS = 4;
        if (free_arenas_.size() < MAX_FREE_ARENAS) {
            free_arenas_.push_back(std::move(arena));
        }
        // else: arena is dropped (unique_ptr destroyed)
    }
};

} // namespace graph
} // namespace axiom
