#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "../dtype.hpp"
#include "../operations.hpp"
#include "../shape.hpp"
#include "../storage.hpp"
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

struct ExecutionStep {
    enum class Kind : uint8_t {
        SingleOp,         // Dispatch via OperationRegistry
        FusedKnown,       // Matched HWY SIMD pattern
        FusedGeneric,     // Generic fused loop with fn-ptr dispatch
        MatMulActivation, // MatMul + activation fused
        FusedReduction    // Elementwise chain + full reduction
    };

    Kind kind{};
    ops::OpType op_type{};      // For SingleOp
    ops::OpType reduction_op{}; // For FusedReduction
    GraphNode::Params params;
    Device device = Device::CPU;
    FusedPattern pattern = FusedPattern::None; // For FusedKnown

    // Chain of ops for fused steps
    std::vector<ops::OpType> op_chain;

    // Per-op: which buffer slots are inputs
    // -1 means "use result of previous op in chain"
    std::vector<std::vector<int>> input_slot_indices;

    int output_slot = -1;
    size_t total_elements = 0;
    Shape output_shape;
    DType output_dtype{};
    DType chain_dtype{}; // For FusedReduction: dtype of the elementwise chain
    std::vector<AccessPattern> input_access;

    size_t tile_size = 0;
};

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
