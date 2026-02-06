#include "axiom/graph/graph_executor.hpp"
#include "axiom/error.hpp"
#include "axiom/graph/fused_kernel.hpp"
#include "axiom/graph/graph_node.hpp"
#include "axiom/operations.hpp"
#include "backends/cpu/simd/simd_dispatch.hpp"

#include <algorithm>
#include <unordered_set>

namespace axiom {
namespace graph {

// ============================================================================
// RAII guard to return arena to pool on scope exit
// ============================================================================

namespace {

struct ArenaGuard {
    const CompiledGraph &plan;
    std::unique_ptr<BufferArena> arena;
    ~ArenaGuard() {
        if (arena)
            plan.release_arena(std::move(arena));
    }
};

} // namespace

// ============================================================================
// Execute the full compiled plan
// ============================================================================

Tensor GraphExecutor::execute(const CompiledGraph &plan,
                              const std::vector<Tensor> &inputs) {
    // Allocate buffer vector — one Tensor per slot
    std::vector<Tensor> buffers(plan.buffer_slots.size());

    // Bind input tensors to their slots
    for (size_t i = 0; i < plan.input_slots.size() && i < inputs.size(); ++i) {
        int slot = plan.input_slots[i];
        if (slot >= 0 && slot < static_cast<int>(buffers.size())) {
            buffers[slot] = inputs[i];
        }
    }

    // Acquire or create arena with M backing storages
    auto arena = plan.acquire_arena();
    if (!arena && plan.num_allocations > 0) {
        // Determine device from first non-input slot
        Device arena_device = Device::CPU;
        for (const auto &slot : plan.buffer_slots) {
            if (!slot.is_input) {
                arena_device = slot.device;
                break;
            }
        }

        arena = std::make_unique<BufferArena>();
        arena->backing.resize(plan.num_allocations);
        for (size_t a = 0; a < plan.num_allocations; ++a) {
            arena->backing[a] = std::shared_ptr<Storage>(
                make_storage(plan.allocation_sizes[a], arena_device));
        }
    }
    ArenaGuard guard{plan, std::move(arena)};

    // Bind slots to arena backing via Tensor views
    for (size_t i = 0; i < plan.buffer_slots.size(); ++i) {
        if (plan.buffer_slots[i].is_input)
            continue;
        if (buffers[i].storage())
            continue; // already bound

        const auto &slot = plan.buffer_slots[i];

        // Output slot always gets fresh allocation
        if (static_cast<int>(i) == plan.output_slot) {
            buffers[i] = Tensor(slot.shape, slot.dtype, slot.device);
            continue;
        }

        // Try arena-backed view for intermediates
        if (guard.arena && i < plan.slot_to_allocation.size()) {
            int alloc_id = plan.slot_to_allocation[i];
            if (alloc_id >= 0 &&
                alloc_id < static_cast<int>(guard.arena->backing.size())) {
                buffers[i] = Tensor(guard.arena->backing[alloc_id], slot.shape,
                                    slot.strides, slot.dtype);
                continue;
            }
        }

        // Fallback: fresh allocation
        buffers[i] = Tensor(slot.shape, slot.dtype, slot.device);
    }

    // Execute each step
    for (const auto &step : plan.steps) {
        execute_step(step, plan, buffers);
    }

    // Return the output
    if (plan.output_slot >= 0 &&
        plan.output_slot < static_cast<int>(buffers.size())) {
        return buffers[plan.output_slot];
    }

    throw RuntimeError("CompiledGraph has no valid output slot");
}

// ============================================================================
// Step Dispatch
// ============================================================================

void GraphExecutor::execute_step(const ExecutionStep &step,
                                 const CompiledGraph &plan,
                                 std::vector<Tensor> &buffers) {
    switch (step.kind) {
    case ExecutionStep::Kind::SingleOp:
        execute_single_op(step, buffers);
        break;
    case ExecutionStep::Kind::FusedKnown:
        execute_fused_known(step, plan, buffers);
        break;
    case ExecutionStep::Kind::FusedGeneric:
        execute_fused_generic(step, plan, buffers);
        break;
    case ExecutionStep::Kind::MatMulActivation:
        execute_matmul_activation(step, buffers);
        break;
    }
}

// ============================================================================
// SingleOp: dispatch via OperationRegistry (current behavior)
// ============================================================================

void GraphExecutor::execute_single_op(const ExecutionStep &step,
                                      std::vector<Tensor> &buffers) {
    Device device = step.device;
    ops::OpType op = step.op_type;

    const ops::Operation *operation =
        ops::OperationRegistry::get_operation(op, device);
    if (!operation) {
        device = Device::CPU;
        operation = ops::OperationRegistry::get_operation(op, device);
    }
    if (!operation) {
        throw DeviceError("Operation not available for any device");
    }

    // Collect inputs from the first (only) op's input slots
    if (step.input_slot_indices.empty()) {
        throw RuntimeError("SingleOp step has no input slot indices");
    }

    const auto &indices = step.input_slot_indices[0];

    auto get_input = [&](int slot_idx) -> Tensor {
        if (slot_idx < 0 || slot_idx >= static_cast<int>(buffers.size())) {
            throw RuntimeError("Invalid buffer slot index in SingleOp");
        }
        Tensor t = buffers[slot_idx];
        if (t.device() != device) {
            t = t.to(device);
        }
        return t;
    };

    Tensor result;

    if (is_unary_op(op)) {
        result = operation->execute_unary(get_input(indices[0]));
    } else if (is_binary_op(op)) {
        result = operation->execute_binary(get_input(indices[0]),
                                           get_input(indices[1]));
    } else if (is_reduction_op(op)) {
        result = operation->execute_reduction(
            get_input(indices[0]), step.params.axes, step.params.keep_dims);
    } else if (op == ops::OpType::MatMul || op == ops::OpType::BatchMatMul) {
        result = operation->execute_matmul(
            get_input(indices[0]), get_input(indices[1]),
            step.params.transpose_a, step.params.transpose_b);
    } else if (op == ops::OpType::Softmax || op == ops::OpType::LogSoftmax) {
        result = operation->execute_reduction(get_input(indices[0]),
                                              {step.params.axis}, false);
    } else {
        throw RuntimeError("Unsupported op type in GraphExecutor::SingleOp");
    }

    if (step.output_slot >= 0 &&
        step.output_slot < static_cast<int>(buffers.size())) {
        buffers[step.output_slot] = result;
    }
}

// ============================================================================
// FusedKnown: use existing HWY SIMD fused kernels
// ============================================================================

// Helper to check if dtype is supported for SIMD fused patterns
static bool is_fused_simd_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::Int32 || dtype == DType::Int64;
}

static bool is_integer_pattern_supported(FusedPattern pattern) {
    switch (pattern) {
    case FusedPattern::AddReLU:
    case FusedPattern::SubAbs:
    case FusedPattern::AddSquare:
    case FusedPattern::SubSquare:
    case FusedPattern::MulAdd:
        return true;
    default:
        return false;
    }
}

// Execute a known fused pattern using optimized SIMD kernels
static bool dispatch_fused_simd(FusedPattern pattern,
                                const std::vector<Tensor> &inputs,
                                Tensor &result) {
    if (inputs.empty())
        return false;

    DType dtype = inputs[0].dtype();
    size_t n = inputs[0].size();

    bool is_integer = (dtype == DType::Int32 || dtype == DType::Int64);
    if (is_integer && !is_integer_pattern_supported(pattern))
        return false;

#define DISPATCH_BINARY(PATTERN, DISPATCH_FN)                                  \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 2) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>(),        \
                                         inputs[1].typed_data<float>(),        \
                                         result.typed_data<float>(), n);       \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(inputs[0].typed_data<double>(),      \
                                          inputs[1].typed_data<double>(),      \
                                          result.typed_data<double>(), n);     \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_BINARY_WITH_INT(PATTERN, DISPATCH_FN)                         \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 2) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>(),        \
                                         inputs[1].typed_data<float>(),        \
                                         result.typed_data<float>(), n);       \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(inputs[0].typed_data<double>(),      \
                                          inputs[1].typed_data<double>(),      \
                                          result.typed_data<double>(), n);     \
                return true;                                                   \
            } else if (dtype == DType::Int32) {                                \
                simd::DISPATCH_FN<int32_t>(inputs[0].typed_data<int32_t>(),    \
                                           inputs[1].typed_data<int32_t>(),    \
                                           result.typed_data<int32_t>(), n);   \
                return true;                                                   \
            } else if (dtype == DType::Int64) {                                \
                simd::DISPATCH_FN<int64_t>(inputs[0].typed_data<int64_t>(),    \
                                           inputs[1].typed_data<int64_t>(),    \
                                           result.typed_data<int64_t>(), n);   \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_TERNARY(PATTERN, DISPATCH_FN)                                 \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 3) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>(),        \
                                         inputs[1].typed_data<float>(),        \
                                         inputs[2].typed_data<float>(),        \
                                         result.typed_data<float>(), n);       \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(inputs[0].typed_data<double>(),      \
                                          inputs[1].typed_data<double>(),      \
                                          inputs[2].typed_data<double>(),      \
                                          result.typed_data<double>(), n);     \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_TERNARY_WITH_INT(PATTERN, DISPATCH_FN)                        \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 3) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>(),        \
                                         inputs[1].typed_data<float>(),        \
                                         inputs[2].typed_data<float>(),        \
                                         result.typed_data<float>(), n);       \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(inputs[0].typed_data<double>(),      \
                                          inputs[1].typed_data<double>(),      \
                                          inputs[2].typed_data<double>(),      \
                                          result.typed_data<double>(), n);     \
                return true;                                                   \
            } else if (dtype == DType::Int32) {                                \
                simd::DISPATCH_FN<int32_t>(inputs[0].typed_data<int32_t>(),    \
                                           inputs[1].typed_data<int32_t>(),    \
                                           inputs[2].typed_data<int32_t>(),    \
                                           result.typed_data<int32_t>(), n);   \
                return true;                                                   \
            } else if (dtype == DType::Int64) {                                \
                simd::DISPATCH_FN<int64_t>(inputs[0].typed_data<int64_t>(),    \
                                           inputs[1].typed_data<int64_t>(),    \
                                           inputs[2].typed_data<int64_t>(),    \
                                           result.typed_data<int64_t>(), n);   \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

    switch (pattern) {
        DISPATCH_BINARY_WITH_INT(AddReLU, dispatch_fused_add_relu);
        DISPATCH_BINARY_WITH_INT(SubAbs, dispatch_fused_sub_abs);
        DISPATCH_BINARY_WITH_INT(AddSquare, dispatch_fused_add_square);
        DISPATCH_BINARY_WITH_INT(SubSquare, dispatch_fused_sub_square);
        DISPATCH_BINARY(MulReLU, dispatch_fused_mul_relu);
        DISPATCH_BINARY(AddSigmoid, dispatch_fused_add_sigmoid);
        DISPATCH_BINARY(MulSigmoid, dispatch_fused_mul_sigmoid);
        DISPATCH_TERNARY_WITH_INT(MulAdd, dispatch_fused_mul_add);
        DISPATCH_TERNARY(MulSub, dispatch_fused_mul_sub);
        DISPATCH_TERNARY(ScaleShiftReLU, dispatch_fused_scale_shift_relu);
        DISPATCH_TERNARY(AddMulReLU, dispatch_fused_add_mul_relu);
        DISPATCH_TERNARY(SubMulAbs, dispatch_fused_sub_mul_abs);
    default:
        break;
    }

#undef DISPATCH_BINARY
#undef DISPATCH_BINARY_WITH_INT
#undef DISPATCH_TERNARY
#undef DISPATCH_TERNARY_WITH_INT

    return false;
}

void GraphExecutor::execute_fused_known(const ExecutionStep &step,
                                        const CompiledGraph &plan,
                                        std::vector<Tensor> &buffers) {
    // Collect external inputs (not chain-internal results)
    std::vector<Tensor> inputs;
    std::unordered_set<int> chain_internal;

    // For fused chains, inputs with index -1 are chain-internal
    for (const auto &per_op : step.input_slot_indices) {
        for (int s : per_op) {
            if (s >= 0 && s < static_cast<int>(buffers.size())) {
                bool already = false;
                for (const auto &t : inputs) {
                    if (t.storage() == buffers[s].storage()) {
                        already = true;
                        break;
                    }
                }
                if (!already) {
                    inputs.push_back(buffers[s]);
                }
            }
        }
    }

    if (!is_fused_simd_dtype(step.output_dtype) || step.device != Device::CPU) {
        // Fallback to generic
        execute_fused_generic(step, plan, buffers);
        return;
    }

    // Check inputs are contiguous and same size as output (no broadcast)
    for (const auto &t : inputs) {
        if (!t.is_contiguous() || t.size() != step.total_elements) {
            execute_fused_generic(step, plan, buffers);
            return;
        }
    }

    // Allocate output
    Tensor result = buffers[step.output_slot];
    if (!result.storage()) {
        result = Tensor(step.output_shape, step.output_dtype, step.device);
    }

    if (dispatch_fused_simd(step.pattern, inputs, result)) {
        buffers[step.output_slot] = result;
        return;
    }

    // Fallback
    execute_fused_generic(step, plan, buffers);
}

// ============================================================================
// FusedGeneric: tiled fused loop with devirtualized fn-ptr dispatch
// ============================================================================

// Tile size: 16K elements = 64KB for float32, fits in Apple M-series L1
static constexpr size_t TILE_ELEMENTS = 16384;

// Stack-allocated tile buffers (2 for double-buffering)
struct alignas(64) TileBuffer {
    char data[TILE_ELEMENTS * sizeof(double)]; // worst case: float64
};

// Attempt the tiled fused loop. Returns false if any op in the chain
// lacks a fn-ptr (unsupported dtype/op), in which case the caller
// falls back to OperationRegistry dispatch.
static bool try_tiled_fused_loop(const ExecutionStep &step,
                                 std::vector<Tensor> &buffers) {
    if (step.device != Device::CPU)
        return false;

    DType dtype = step.output_dtype;
    size_t elem_size = dtype_size(dtype);
    if (elem_size == 0)
        return false;

    // Resolve all function pointers up front
    struct OpDesc {
        bool is_unary;
        UnaryFn unary_fn;
        BinaryFn binary_fn;
        int input_slot_a; // -1 = previous result
        int input_slot_b; // -1 = previous result, -2 = N/A
    };

    std::vector<OpDesc> descs;
    descs.reserve(step.op_chain.size());

    for (size_t i = 0; i < step.op_chain.size(); ++i) {
        ops::OpType op = step.op_chain[i];
        const auto &indices = step.input_slot_indices[i];
        OpDesc d{};

        if (is_unary_op(op)) {
            d.is_unary = true;
            d.unary_fn = get_unary_fn(op, dtype);
            if (!d.unary_fn)
                return false;
            d.input_slot_a = indices.empty() ? -1 : indices[0];
            d.input_slot_b = -2;
        } else if (is_binary_op(op)) {
            d.is_unary = false;
            d.binary_fn = get_binary_fn(op, dtype);
            if (!d.binary_fn)
                return false;
            d.input_slot_a = indices.empty() ? -1 : indices[0];
            d.input_slot_b = (indices.size() > 1) ? indices[1] : -1;
        } else {
            return false; // unsupported op kind in chain
        }

        descs.push_back(d);
    }

    // All fn-ptrs resolved — proceed with tiled loop

    // Check inputs are contiguous and match output size (no broadcast)
    for (const auto &per_op : step.input_slot_indices) {
        for (int s : per_op) {
            if (s >= 0 && s < static_cast<int>(buffers.size())) {
                if (!buffers[s].is_contiguous() ||
                    buffers[s].size() != step.total_elements)
                    return false;
            }
        }
    }

    size_t total = step.total_elements;
    size_t tile_size = std::min(TILE_ELEMENTS, total);

    // Output buffer
    Tensor &output = buffers[step.output_slot];
    if (!output.storage()) {
        output = Tensor(step.output_shape, dtype, Device::CPU);
    }
    char *out_ptr = reinterpret_cast<char *>(output.data());

    // Stack tile buffers for intermediate results (double-buffer)
    TileBuffer tile_a_buf, tile_b_buf;
    void *tile_a = tile_a_buf.data;
    void *tile_b = tile_b_buf.data;

    auto resolve_ptr = [&](int slot, size_t offset) -> const void * {
        if (slot >= 0 && slot < static_cast<int>(buffers.size())) {
            return reinterpret_cast<const char *>(buffers[slot].data()) +
                   offset * elem_size;
        }
        return nullptr;
    };

    for (size_t base = 0; base < total; base += tile_size) {
        size_t count = std::min(tile_size, total - base);

        // Current intermediate result pointer alternates between tile_a
        // and tile_b. After the last op, we copy directly to output.
        void *prev = nullptr;
        bool prev_is_a = true; // tracks which tile holds `prev`

        for (size_t oi = 0; oi < descs.size(); ++oi) {
            const auto &d = descs[oi];
            bool is_last = (oi == descs.size() - 1);

            // Where to write this op's result
            void *dst;
            if (is_last) {
                dst = out_ptr + base * elem_size;
            } else {
                dst = prev_is_a ? tile_b : tile_a;
            }

            if (d.is_unary) {
                const void *src;
                if (d.input_slot_a == -1) {
                    src = prev;
                } else {
                    src = resolve_ptr(d.input_slot_a, base);
                }
                d.unary_fn(src, dst, count);
            } else {
                const void *src_a;
                const void *src_b;
                if (d.input_slot_a == -1) {
                    src_a = prev;
                } else {
                    src_a = resolve_ptr(d.input_slot_a, base);
                }
                if (d.input_slot_b == -1) {
                    src_b = prev;
                } else {
                    src_b = resolve_ptr(d.input_slot_b, base);
                }
                d.binary_fn(src_a, src_b, dst, count);
            }

            prev = dst;
            if (!is_last) {
                prev_is_a = !prev_is_a;
            }
        }
    }

    return true;
}

void GraphExecutor::execute_fused_generic(const ExecutionStep &step,
                                          const CompiledGraph & /*plan*/,
                                          std::vector<Tensor> &buffers) {
    // Try the fast tiled path first
    if (try_tiled_fused_loop(step, buffers))
        return;

    // Fallback: sequential op-by-op via OperationRegistry
    Device device = step.device;
    Tensor current;

    for (size_t i = 0; i < step.op_chain.size(); ++i) {
        ops::OpType op = step.op_chain[i];

        const ops::Operation *operation =
            ops::OperationRegistry::get_operation(op, device);
        if (!operation) {
            operation = ops::OperationRegistry::get_operation(op, Device::CPU);
            device = Device::CPU;
        }
        if (!operation) {
            throw DeviceError("Operation not available in fused generic");
        }

        const auto &indices = step.input_slot_indices[i];

        auto resolve = [&](int idx) -> Tensor {
            if (idx == -1)
                return current;
            if (idx >= 0 && idx < static_cast<int>(buffers.size())) {
                Tensor t = buffers[idx];
                if (t.device() != device)
                    t = t.to(device);
                return t;
            }
            throw RuntimeError("Invalid slot index in fused generic");
        };

        if (is_unary_op(op)) {
            current = operation->execute_unary(resolve(indices[0]));
        } else if (is_binary_op(op)) {
            Tensor lhs = resolve(indices[0]);
            Tensor rhs = (indices.size() > 1) ? resolve(indices[1]) : current;
            current = operation->execute_binary(lhs, rhs);
        } else {
            throw RuntimeError("Unsupported op in fused chain");
        }
    }

    if (step.output_slot >= 0 &&
        step.output_slot < static_cast<int>(buffers.size())) {
        buffers[step.output_slot] = current;
    }
}

// ============================================================================
// MatMulActivation: matmul + in-place activation via devirtualized fn-ptr
// ============================================================================

void GraphExecutor::execute_matmul_activation(const ExecutionStep &step,
                                              std::vector<Tensor> &buffers) {
    if (step.op_chain.size() < 2) {
        // Fallback: just execute as single matmul
        execute_single_op(step, buffers);
        return;
    }

    // Execute matmul
    Device device = step.device;
    ops::OpType mm_op = step.op_chain[0];

    const ops::Operation *operation =
        ops::OperationRegistry::get_operation(mm_op, device);
    if (!operation) {
        device = Device::CPU;
        operation = ops::OperationRegistry::get_operation(mm_op, device);
    }
    if (!operation) {
        throw DeviceError("MatMul operation not available");
    }

    const auto &mm_indices = step.input_slot_indices[0];
    if (mm_indices.size() < 2) {
        throw RuntimeError("MatMulActivation: matmul needs 2 inputs");
    }

    auto get_buf = [&](int idx) -> Tensor {
        if (idx < 0 || idx >= static_cast<int>(buffers.size()))
            throw RuntimeError("Invalid buffer slot in MatMulActivation");
        Tensor t = buffers[idx];
        if (t.device() != device)
            t = t.to(device);
        return t;
    };

    Tensor result = operation->execute_matmul(
        get_buf(mm_indices[0]), get_buf(mm_indices[1]), step.params.transpose_a,
        step.params.transpose_b);

    // Apply activation in-place using devirtualized fn-ptr
    ops::OpType act_op = step.op_chain[1];
    DType dtype = result.dtype();
    UnaryFn fn = get_unary_fn(act_op, dtype);

    if (fn && result.is_contiguous()) {
        fn(result.data(), result.data(), result.size());
    } else {
        // Fallback via OperationRegistry
        const ops::Operation *act =
            ops::OperationRegistry::get_operation(act_op, device);
        if (!act) {
            act = ops::OperationRegistry::get_operation(act_op, Device::CPU);
        }
        if (act) {
            result = act->execute_unary(result);
        }
    }

    if (step.output_slot >= 0 &&
        step.output_slot < static_cast<int>(buffers.size())) {
        buffers[step.output_slot] = result;
    }
}

} // namespace graph
} // namespace axiom
