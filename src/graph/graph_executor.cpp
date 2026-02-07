#include "axiom/graph/graph_executor.hpp"
#include "axiom/error.hpp"
#include "axiom/graph/fused_kernel.hpp"
#include "axiom/graph/graph_node.hpp"
#include "axiom/operations.hpp"
#include "backends/cpu/simd/simd_dispatch.hpp"

#include <algorithm>
#include <unordered_set>

#include "axiom/parallel.hpp"

namespace axiom {
namespace graph {

// Tile size: 16K elements = 64KB for float32, fits in Apple M-series L1
static constexpr size_t TILE_ELEMENTS = 16384;

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

    // Auto-tune tile size on first execution of a graph with fused steps.
    // Use heuristic: longer chains benefit from smaller tiles (better L1
    // residency), shorter chains benefit from larger tiles (less overhead).
    if (!plan.tuned_.load(std::memory_order_relaxed)) {
        size_t best_ts = TILE_ELEMENTS; // default 16K elements
        for (const auto &step : plan.steps) {
            if ((step.kind == ExecutionStep::Kind::FusedGeneric ||
                 step.kind == ExecutionStep::Kind::FusedReduction) &&
                step.device == Device::CPU && step.total_elements >= 100000) {
                size_t chain_len = step.op_chain.size();
                if (chain_len >= 6) {
                    // Long chains: smaller tiles to keep working set in L1
                    best_ts = 4096;
                } else if (chain_len >= 3) {
                    // Medium chains: default
                    best_ts = 16384;
                } else {
                    // Short chains: larger tiles, less overhead
                    best_ts = 65536;
                }
                break;
            }
        }
        plan.tuned_tile_size_.store(best_ts, std::memory_order_relaxed);
        plan.tuned_.store(true, std::memory_order_relaxed);
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
    case ExecutionStep::Kind::FusedReduction:
        execute_fused_reduction(step, plan, buffers);
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

// FusedKnown SIMD kernels do minimal work per call, so OpenMP thread
// spawn overhead is only amortized at larger sizes (~1M+ elements).
static constexpr size_t FUSED_KNOWN_MIN_PARALLEL = 524288;

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

// Execute a known fused pattern using optimized SIMD kernels.
// off/cnt allow processing a sub-range for parallel dispatch.
static bool dispatch_fused_simd(FusedPattern pattern,
                                const std::vector<Tensor> &inputs,
                                Tensor &result, size_t off = 0,
                                size_t cnt = 0) {
    if (inputs.empty())
        return false;

    DType dtype = inputs[0].dtype();
    if (cnt == 0)
        cnt = inputs[0].size();

    bool is_integer = (dtype == DType::Int32 || dtype == DType::Int64);
    if (is_integer && !is_integer_pattern_supported(pattern))
        return false;

#define DISPATCH_BINARY(PATTERN, DISPATCH_FN)                                  \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 2) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_BINARY_WITH_INT(PATTERN, DISPATCH_FN)                         \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 2) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            } else if (dtype == DType::Int32) {                                \
                simd::DISPATCH_FN<int32_t>(                                    \
                    inputs[0].typed_data<int32_t>() + off,                     \
                    inputs[1].typed_data<int32_t>() + off,                     \
                    result.typed_data<int32_t>() + off, cnt);                  \
                return true;                                                   \
            } else if (dtype == DType::Int64) {                                \
                simd::DISPATCH_FN<int64_t>(                                    \
                    inputs[0].typed_data<int64_t>() + off,                     \
                    inputs[1].typed_data<int64_t>() + off,                     \
                    result.typed_data<int64_t>() + off, cnt);                  \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_TERNARY(PATTERN, DISPATCH_FN)                                 \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 3) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         inputs[2].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    inputs[2].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_TERNARY_WITH_INT(PATTERN, DISPATCH_FN)                        \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 3) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         inputs[2].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    inputs[2].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            } else if (dtype == DType::Int32) {                                \
                simd::DISPATCH_FN<int32_t>(                                    \
                    inputs[0].typed_data<int32_t>() + off,                     \
                    inputs[1].typed_data<int32_t>() + off,                     \
                    inputs[2].typed_data<int32_t>() + off,                     \
                    result.typed_data<int32_t>() + off, cnt);                  \
                return true;                                                   \
            } else if (dtype == DType::Int64) {                                \
                simd::DISPATCH_FN<int64_t>(                                    \
                    inputs[0].typed_data<int64_t>() + off,                     \
                    inputs[1].typed_data<int64_t>() + off,                     \
                    inputs[2].typed_data<int64_t>() + off,                     \
                    result.typed_data<int64_t>() + off, cnt);                  \
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

    size_t n = step.total_elements;

    if (parallel::should_parallelize(n, FUSED_KNOWN_MIN_PARALLEL)) {
        // Parallel: chunk data across threads
        size_t nt = parallel::get_num_threads();
        size_t chunk = (n + nt - 1) / nt;
        ptrdiff_t nchunks = static_cast<ptrdiff_t>((n + chunk - 1) / chunk);
        bool ok = true;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t ci = 0; ci < nchunks; ++ci) {
            size_t start = static_cast<size_t>(ci) * chunk;
            size_t count = std::min(chunk, n - start);
            if (!dispatch_fused_simd(step.pattern, inputs, result, start,
                                     count)) {
#pragma omp atomic write
                ok = false;
            }
        }
        if (ok) {
            buffers[step.output_slot] = result;
            return;
        }
    } else {
        if (dispatch_fused_simd(step.pattern, inputs, result)) {
            buffers[step.output_slot] = result;
            return;
        }
    }

    // Fallback
    execute_fused_generic(step, plan, buffers);
}

// ============================================================================
// FusedGeneric: tiled fused loop with devirtualized fn-ptr dispatch
// ============================================================================

// Stack-allocated tile buffers (2 for double-buffering)
struct alignas(64) TileBuffer {
    char data[TILE_ELEMENTS * sizeof(double)]; // worst case: float64
};

// Attempt the tiled fused loop. Returns false if any op in the chain
// lacks a fn-ptr (unsupported dtype/op), in which case the caller
// falls back to OperationRegistry dispatch.
static bool try_tiled_fused_loop(const ExecutionStep &step,
                                 std::vector<Tensor> &buffers,
                                 const CompiledGraph &plan) {
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
    size_t ts = plan.tuned_.load(std::memory_order_relaxed)
                    ? plan.tuned_tile_size_.load(std::memory_order_relaxed)
                    : TILE_ELEMENTS;
    if (ts == 0)
        ts = TILE_ELEMENTS;
    size_t tile_size = std::min(ts, total);

    // Output buffer
    Tensor &output = buffers[step.output_slot];
    if (!output.storage()) {
        output = Tensor(step.output_shape, dtype, Device::CPU);
    }
    char *out_ptr = reinterpret_cast<char *>(output.data());

    auto resolve_ptr = [&](int slot, size_t offset) -> const void * {
        if (slot >= 0 && slot < static_cast<int>(buffers.size())) {
            return reinterpret_cast<const char *>(buffers[slot].data()) +
                   offset * elem_size;
        }
        return nullptr;
    };

    // Inner loop body: process one tile of the fused chain
    auto process_tile = [&](size_t base, size_t count, void *tile_a,
                            void *tile_b) {
        void *prev = nullptr;
        bool prev_is_a = true;

        for (size_t oi = 0; oi < descs.size(); ++oi) {
            const auto &d = descs[oi];
            bool is_last = (oi == descs.size() - 1);

            void *dst;
            if (is_last) {
                dst = out_ptr + base * elem_size;
            } else {
                dst = prev_is_a ? tile_b : tile_a;
            }

            if (d.is_unary) {
                const void *src = (d.input_slot_a == -1)
                                      ? prev
                                      : resolve_ptr(d.input_slot_a, base);
                d.unary_fn(src, dst, count);
            } else {
                const void *src_a = (d.input_slot_a == -1)
                                        ? prev
                                        : resolve_ptr(d.input_slot_a, base);
                const void *src_b = (d.input_slot_b == -1)
                                        ? prev
                                        : resolve_ptr(d.input_slot_b, base);
                d.binary_fn(src_a, src_b, dst, count);
            }

            prev = dst;
            if (!is_last)
                prev_is_a = !prev_is_a;
        }
    };

    if (parallel::should_parallelize(total)) {
        // Parallel: each thread gets its own tile buffers
        ptrdiff_t num_tiles =
            static_cast<ptrdiff_t>((total + tile_size - 1) / tile_size);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t ti = 0; ti < num_tiles; ++ti) {
            size_t base = static_cast<size_t>(ti) * tile_size;
            size_t count = std::min(tile_size, total - base);
            TileBuffer local_a, local_b;
            process_tile(base, count, local_a.data, local_b.data);
        }
    } else {
        // Sequential: single set of tile buffers
        TileBuffer tile_a_buf, tile_b_buf;
        for (size_t base = 0; base < total; base += tile_size) {
            size_t count = std::min(tile_size, total - base);
            process_tile(base, count, tile_a_buf.data, tile_b_buf.data);
        }
    }

    return true;
}

void GraphExecutor::execute_fused_generic(const ExecutionStep &step,
                                          const CompiledGraph &plan,
                                          std::vector<Tensor> &buffers) {
    // Try the fast tiled path first
    if (try_tiled_fused_loop(step, buffers, plan))
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

// ============================================================================
// FusedReduction: elementwise chain + full reduction in a single pass
// ============================================================================

// Fallback for FusedReduction: execute elementwise chain, then reduce
void GraphExecutor::execute_fused_reduction_fallback(
    const ExecutionStep &step, const CompiledGraph &plan,
    std::vector<Tensor> &buffers) {
    // Build a temporary step for just the elementwise chain
    ExecutionStep ew_step;
    ew_step.kind = ExecutionStep::Kind::FusedGeneric;
    ew_step.op_chain = step.op_chain;
    // input_slot_indices: only the elementwise ops' indices
    // (excl. the reduction's input which is the chain output)
    ew_step.input_slot_indices.assign(
        step.input_slot_indices.begin(),
        step.input_slot_indices.begin() +
            static_cast<ptrdiff_t>(step.op_chain.size()));
    ew_step.device = step.device;
    ew_step.output_dtype = step.output_dtype;
    // Determine output shape from the elementwise chain input
    if (!step.input_slot_indices.empty() &&
        !step.input_slot_indices[0].empty()) {
        int first_slot = step.input_slot_indices[0][0];
        if (first_slot >= 0 && first_slot < static_cast<int>(buffers.size())) {
            ew_step.output_shape = buffers[first_slot].shape();
        }
    }
    ew_step.total_elements = step.total_elements;

    // Allocate a temporary slot for elementwise output
    int temp_slot = static_cast<int>(buffers.size());
    buffers.emplace_back();
    ew_step.output_slot = temp_slot;

    GraphExecutor::execute_fused_generic(ew_step, plan, buffers);

    // Now execute the reduction on the elementwise result
    Tensor ew_result = buffers[temp_slot];
    ops::OpType red_op = step.reduction_op;
    const ops::Operation *operation =
        ops::OperationRegistry::get_operation(red_op, step.device);
    if (!operation) {
        operation = ops::OperationRegistry::get_operation(red_op, Device::CPU);
    }
    if (operation) {
        Tensor result = operation->execute_reduction(
            ew_result, step.params.axes, step.params.keep_dims);
        if (step.output_slot >= 0 &&
            step.output_slot < static_cast<int>(buffers.size())) {
            buffers[step.output_slot] = result;
        }
    }
    buffers.pop_back(); // remove temporary slot
}

void GraphExecutor::execute_fused_reduction(const ExecutionStep &step,
                                            const CompiledGraph &plan,
                                            std::vector<Tensor> &buffers) {
    if (step.device != Device::CPU) {
        // GPU: execute elementwise chain + reduction separately
        execute_fused_reduction_fallback(step, plan, buffers);
        return;
    }

    DType dtype = step.output_dtype;
    size_t elem_size = dtype_size(dtype);
    if (elem_size == 0) {
        execute_fused_reduction_fallback(step, plan, buffers);
        return;
    }

    // Resolve fn-ptrs for the elementwise chain (excl. reduction)
    struct OpDesc {
        bool is_unary;
        UnaryFn unary_fn;
        BinaryFn binary_fn;
        int input_slot_a;
        int input_slot_b;
    };

    std::vector<OpDesc> descs;
    descs.reserve(step.op_chain.size());

    // The op_chain contains only the elementwise ops (reduction is separate)
    for (size_t i = 0; i < step.op_chain.size(); ++i) {
        ops::OpType op = step.op_chain[i];
        const auto &indices = step.input_slot_indices[i];
        OpDesc d{};

        if (is_unary_op(op)) {
            d.is_unary = true;
            d.unary_fn = get_unary_fn(op, dtype);
            if (!d.unary_fn) {
                execute_fused_generic(step, plan, buffers);
                return;
            }
            d.input_slot_a = indices.empty() ? -1 : indices[0];
            d.input_slot_b = -2;
        } else if (is_binary_op(op)) {
            d.is_unary = false;
            d.binary_fn = get_binary_fn(op, dtype);
            if (!d.binary_fn) {
                execute_fused_generic(step, plan, buffers);
                return;
            }
            d.input_slot_a = indices.empty() ? -1 : indices[0];
            d.input_slot_b = (indices.size() > 1) ? indices[1] : -1;
        } else {
            execute_fused_generic(step, plan, buffers);
            return;
        }

        descs.push_back(d);
    }

    // Check inputs are contiguous and match the elementwise chain size
    // (broadcast inputs would require coordinate remapping, not supported)
    for (const auto &per_op : step.input_slot_indices) {
        for (int s : per_op) {
            if (s >= 0 && s < static_cast<int>(buffers.size())) {
                if (!buffers[s].is_contiguous() ||
                    buffers[s].size() != step.total_elements) {
                    execute_fused_reduction_fallback(step, plan, buffers);
                    return;
                }
            }
        }
    }

    // Determine the input element count (the reduction input, not output)
    // step.total_elements was set to the elementwise chain input size
    size_t total = step.total_elements;
    if (total == 0) {
        execute_fused_generic(step, plan, buffers);
        return;
    }

    size_t ts = plan.tuned_.load(std::memory_order_relaxed)
                    ? plan.tuned_tile_size_.load(std::memory_order_relaxed)
                    : TILE_ELEMENTS;
    if (ts == 0)
        ts = TILE_ELEMENTS;
    size_t tile_size = std::min(ts, total);

    auto resolve_ptr = [&](int slot, size_t offset) -> const void * {
        if (slot >= 0 && slot < static_cast<int>(buffers.size())) {
            return reinterpret_cast<const char *>(buffers[slot].data()) +
                   offset * elem_size;
        }
        return nullptr;
    };

    // Determine reduction identity and SIMD reduce function
    ops::OpType red_op = step.reduction_op;
    bool is_mean = (red_op == ops::OpType::Mean);
    if (is_mean)
        red_op = ops::OpType::Sum; // Mean = Sum / N

    // We only support float32/float64 for fused reduction SIMD
    bool use_simd = (dtype == DType::Float32 || dtype == DType::Float64);

    // Process tile: run elementwise chain, then reduce tile
    auto process_tile_and_reduce = [&](size_t base, size_t count, void *tile_a,
                                       void *tile_b) -> double {
        void *prev = nullptr;
        bool prev_is_a = true;

        for (size_t oi = 0; oi < descs.size(); ++oi) {
            const auto &d = descs[oi];
            // All ops write to tile buffer (never directly to output)
            void *dst = prev_is_a ? tile_b : tile_a;

            if (d.is_unary) {
                const void *src = (d.input_slot_a == -1)
                                      ? prev
                                      : resolve_ptr(d.input_slot_a, base);
                d.unary_fn(src, dst, count);
            } else {
                const void *src_a = (d.input_slot_a == -1)
                                        ? prev
                                        : resolve_ptr(d.input_slot_a, base);
                const void *src_b = (d.input_slot_b == -1)
                                        ? prev
                                        : resolve_ptr(d.input_slot_b, base);
                d.binary_fn(src_a, src_b, dst, count);
            }

            prev = dst;
            prev_is_a = !prev_is_a;
        }

        // Reduce the tile
        double tile_acc = 0.0;
        if (use_simd && dtype == DType::Float32) {
            const float *tile = static_cast<const float *>(prev);
            if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean)
                tile_acc = simd::dispatch_reduce_sum(tile, count);
            else if (red_op == ops::OpType::Max)
                tile_acc = simd::dispatch_reduce_max(tile, count);
            else if (red_op == ops::OpType::Min)
                tile_acc = simd::dispatch_reduce_min(tile, count);
            else if (red_op == ops::OpType::Prod)
                tile_acc = simd::dispatch_reduce_prod(tile, count);
        } else if (use_simd && dtype == DType::Float64) {
            const double *tile = static_cast<const double *>(prev);
            if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean)
                tile_acc = simd::dispatch_reduce_sum(tile, count);
            else if (red_op == ops::OpType::Max)
                tile_acc = simd::dispatch_reduce_max(tile, count);
            else if (red_op == ops::OpType::Min)
                tile_acc = simd::dispatch_reduce_min(tile, count);
            else if (red_op == ops::OpType::Prod)
                tile_acc = simd::dispatch_reduce_prod(tile, count);
        }
        return tile_acc;
    };

    double acc = 0.0;
    bool is_max_min =
        (red_op == ops::OpType::Max || red_op == ops::OpType::Min);
    bool is_prod = (red_op == ops::OpType::Prod);

    if (is_prod)
        acc = 1.0;
    if (is_max_min)
        acc = (red_op == ops::OpType::Max)
                  ? -std::numeric_limits<double>::infinity()
                  : std::numeric_limits<double>::infinity();

    if (parallel::should_parallelize(total)) {
        ptrdiff_t num_tiles =
            static_cast<ptrdiff_t>((total + tile_size - 1) / tile_size);

        if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean) {
#pragma omp parallel for reduction(+ : acc) schedule(static)
            for (ptrdiff_t ti = 0; ti < num_tiles; ++ti) {
                size_t base = static_cast<size_t>(ti) * tile_size;
                size_t count = std::min(tile_size, total - base);
                TileBuffer la, lb;
                acc += process_tile_and_reduce(base, count, la.data, lb.data);
            }
        } else if (red_op == ops::OpType::Max) {
            double local_max = acc;
#pragma omp parallel
            {
                double thread_max = acc;
                TileBuffer la, lb;
#pragma omp for schedule(static) nowait
                for (ptrdiff_t ti = 0; ti < num_tiles; ++ti) {
                    size_t base = static_cast<size_t>(ti) * tile_size;
                    size_t count = std::min(tile_size, total - base);
                    double v =
                        process_tile_and_reduce(base, count, la.data, lb.data);
                    if (v > thread_max)
                        thread_max = v;
                }
#pragma omp critical
                {
                    if (thread_max > local_max)
                        local_max = thread_max;
                }
            }
            acc = local_max;
        } else if (red_op == ops::OpType::Min) {
            double local_min = acc;
#pragma omp parallel
            {
                double thread_min = acc;
                TileBuffer la, lb;
#pragma omp for schedule(static) nowait
                for (ptrdiff_t ti = 0; ti < num_tiles; ++ti) {
                    size_t base = static_cast<size_t>(ti) * tile_size;
                    size_t count = std::min(tile_size, total - base);
                    double v =
                        process_tile_and_reduce(base, count, la.data, lb.data);
                    if (v < thread_min)
                        thread_min = v;
                }
#pragma omp critical
                {
                    if (thread_min < local_min)
                        local_min = thread_min;
                }
            }
            acc = local_min;
        } else {
            // Prod: sequential to avoid floating-point ordering issues
            TileBuffer la, lb;
            for (size_t base = 0; base < total; base += tile_size) {
                size_t count = std::min(tile_size, total - base);
                acc *= process_tile_and_reduce(base, count, la.data, lb.data);
            }
        }
    } else {
        TileBuffer la, lb;
        for (size_t base = 0; base < total; base += tile_size) {
            size_t count = std::min(tile_size, total - base);
            double v = process_tile_and_reduce(base, count, la.data, lb.data);
            if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean)
                acc += v;
            else if (red_op == ops::OpType::Max)
                acc = std::max(acc, v);
            else if (red_op == ops::OpType::Min)
                acc = std::min(acc, v);
            else if (red_op == ops::OpType::Prod)
                acc *= v;
        }
    }

    if (is_mean)
        acc /= static_cast<double>(total);

    // Store scalar result
    Tensor result(step.output_shape, dtype, Device::CPU);
    if (dtype == DType::Float32) {
        result.typed_data<float>()[0] = static_cast<float>(acc);
    } else if (dtype == DType::Float64) {
        result.typed_data<double>()[0] = acc;
    }

    if (step.output_slot >= 0 &&
        step.output_slot < static_cast<int>(buffers.size())) {
        buffers[step.output_slot] = result;
    }
}

} // namespace graph
} // namespace axiom
