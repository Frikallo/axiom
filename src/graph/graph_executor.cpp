#include "axiom/graph/graph_executor.hpp"
#include "axiom/error.hpp"
#include "axiom/graph/fused_kernel.hpp"
#include "axiom/graph/fused_patterns.hpp"
#include "axiom/graph/graph_node.hpp"
#include "axiom/operations.hpp"
#include "backends/cpu/simd/simd_dispatch.hpp"

#ifdef AXIOM_METAL_SUPPORT
#include "axiom/graph/gpu_fusion.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <unordered_set>

#include "axiom/parallel.hpp"

namespace axiom {
namespace graph {

// Opt-in debug logging for fusion fallbacks (set AXIOM_DEBUG_FUSION=1)
static bool debug_fusion() {
    static const bool enabled = std::getenv("AXIOM_DEBUG_FUSION") != nullptr;
    return enabled;
}

// Tile size: 16K elements = 64KB for float32, fits in Apple M-series L1
static constexpr size_t TILE_ELEMENTS = 16384;

namespace {

// ============================================================================
// RAII guard to return arena to pool on scope exit
// ============================================================================

struct ArenaGuard {
    const CompiledGraph &plan;
    std::unique_ptr<BufferArena> arena;
    ~ArenaGuard() {
        if (arena)
            plan.release_arena(std::move(arena));
    }
};

// Describes a single op in a fused chain with resolved fn-ptrs.
// Shared by FusedGeneric and FusedReduction execution paths.
struct FusedOpDesc {
    bool is_unary;
    UnaryFn unary_fn;
    BinaryFn binary_fn;
    int input_slot_a; // -1 = previous result
    int input_slot_b; // -1 = previous result, -2 = N/A
};

// Resolve fn-ptrs for every op in a chain. Returns empty vector on failure
// (unsupported op or missing fn-ptr for the given dtype).
std::vector<FusedOpDesc>
resolve_chain_fn_ptrs(const std::vector<ops::OpType> &op_chain,
                      const std::vector<std::vector<int>> &input_slot_indices,
                      DType dtype) {
    std::vector<FusedOpDesc> descs;
    descs.reserve(op_chain.size());

    for (size_t i = 0; i < op_chain.size(); ++i) {
        ops::OpType op = op_chain[i];
        const auto &indices = input_slot_indices[i];
        FusedOpDesc d{};

        if (is_unary_op(op)) {
            d.is_unary = true;
            d.unary_fn = get_unary_fn(op, dtype);
            if (!d.unary_fn)
                return {};
            d.input_slot_a = indices.empty() ? -1 : indices[0];
            d.input_slot_b = -2;
        } else if (is_binary_op(op)) {
            d.is_unary = false;
            d.binary_fn = get_binary_fn(op, dtype);
            if (!d.binary_fn)
                return {};
            d.input_slot_a = indices.empty() ? -1 : indices[0];
            d.input_slot_b = (indices.size() > 1) ? indices[1] : -1;
        } else {
            return {};
        }

        descs.push_back(d);
    }
    return descs;
}

// Scalar reduction over a contiguous integer buffer. Returns result as double.
double scalar_reduce_tile(const void *data, size_t count, DType dtype,
                          ops::OpType red_op) {
    if (dtype == DType::Int32) {
        const int32_t *p = static_cast<const int32_t *>(data);
        if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean) {
            int64_t s = 0;
            for (size_t i = 0; i < count; ++i)
                s += p[i];
            return static_cast<double>(s);
        } else if (red_op == ops::OpType::Max) {
            int32_t m = p[0];
            for (size_t i = 1; i < count; ++i)
                m = std::max(m, p[i]);
            return static_cast<double>(m);
        } else if (red_op == ops::OpType::Min) {
            int32_t m = p[0];
            for (size_t i = 1; i < count; ++i)
                m = std::min(m, p[i]);
            return static_cast<double>(m);
        } else if (red_op == ops::OpType::Prod) {
            int64_t m = 1;
            for (size_t i = 0; i < count; ++i)
                m *= p[i];
            return static_cast<double>(m);
        }
    } else if (dtype == DType::Int64) {
        const int64_t *p = static_cast<const int64_t *>(data);
        if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean) {
            int64_t s = 0;
            for (size_t i = 0; i < count; ++i)
                s += p[i];
            return static_cast<double>(s);
        } else if (red_op == ops::OpType::Max) {
            int64_t m = p[0];
            for (size_t i = 1; i < count; ++i)
                m = std::max(m, p[i]);
            return static_cast<double>(m);
        } else if (red_op == ops::OpType::Min) {
            int64_t m = p[0];
            for (size_t i = 1; i < count; ++i)
                m = std::min(m, p[i]);
            return static_cast<double>(m);
        } else if (red_op == ops::OpType::Prod) {
            int64_t m = 1;
            for (size_t i = 0; i < count; ++i)
                m *= p[i];
            return static_cast<double>(m);
        }
    }
    return 0.0;
}

// Reduce a tile using SIMD (float types) or scalar (integer types).
double reduce_tile(const void *data, size_t count, DType dtype,
                   ops::OpType red_op) {
    if (dtype == DType::Float32) {
        const float *tile = static_cast<const float *>(data);
        if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean)
            return simd::dispatch_reduce_sum(tile, count);
        if (red_op == ops::OpType::Max)
            return simd::dispatch_reduce_max(tile, count);
        if (red_op == ops::OpType::Min)
            return simd::dispatch_reduce_min(tile, count);
        if (red_op == ops::OpType::Prod)
            return simd::dispatch_reduce_prod(tile, count);
    } else if (dtype == DType::Float64) {
        const double *tile = static_cast<const double *>(data);
        if (red_op == ops::OpType::Sum || red_op == ops::OpType::Mean)
            return simd::dispatch_reduce_sum(tile, count);
        if (red_op == ops::OpType::Max)
            return simd::dispatch_reduce_max(tile, count);
        if (red_op == ops::OpType::Min)
            return simd::dispatch_reduce_min(tile, count);
        if (red_op == ops::OpType::Prod)
            return simd::dispatch_reduce_prod(tile, count);
    }
    return scalar_reduce_tile(data, count, dtype, red_op);
}

// Store a double accumulator into a scalar tensor of the given dtype.
void store_scalar_result(Tensor &result, double acc) {
    DType dt = result.dtype();
    if (dt == DType::Float32)
        result.typed_data<float>()[0] = static_cast<float>(acc);
    else if (dt == DType::Float64)
        result.typed_data<double>()[0] = acc;
    else if (dt == DType::Int32)
        result.typed_data<int32_t>()[0] = static_cast<int32_t>(acc);
    else if (dt == DType::Int64)
        result.typed_data<int64_t>()[0] = static_cast<int64_t>(acc);
}

// Check that all external input slots are contiguous and match expected size.
bool inputs_are_contiguous(const std::vector<std::vector<int>> &slot_indices,
                           const std::vector<Tensor> &buffers,
                           size_t expected_size) {
    for (const auto &per_op : slot_indices) {
        for (int s : per_op) {
            if (s >= 0 && s < static_cast<int>(buffers.size())) {
                if (!buffers[s].is_contiguous() ||
                    buffers[s].size() != expected_size)
                    return false;
            }
        }
    }
    return true;
}

// Forward declarations for internal dispatch functions
void execute_single_op(const SingleOpStep &step, std::vector<Tensor> &buffers);
void execute_fused_known(const FusedKnownStep &step, const CompiledGraph &plan,
                         std::vector<Tensor> &buffers);
void execute_fused_generic(const FusedGenericStep &step,
                           const CompiledGraph &plan,
                           std::vector<Tensor> &buffers);
void execute_matmul_activation(const MatMulActivationStep &step,
                               std::vector<Tensor> &buffers);
void execute_fused_reduction(const FusedReductionStep &step,
                             const CompiledGraph &plan,
                             std::vector<Tensor> &buffers);
void execute_fused_reduction_fallback(const FusedReductionStep &step,
                                      const CompiledGraph &plan,
                                      std::vector<Tensor> &buffers);

} // anonymous namespace

// ============================================================================
// Execute the full compiled plan
// ============================================================================

Tensor execute(const CompiledGraph &plan, const std::vector<Tensor> &inputs) {
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

    // Execute each step via std::visit
    for (const auto &step : plan.steps) {
        std::visit(
            [&](const auto &s) {
                using T = std::decay_t<decltype(s)>;
                if constexpr (std::is_same_v<T, SingleOpStep>) {
                    execute_single_op(s, buffers);
                } else if constexpr (std::is_same_v<T, FusedKnownStep>) {
                    execute_fused_known(s, plan, buffers);
                } else if constexpr (std::is_same_v<T, FusedGenericStep>) {
                    execute_fused_generic(s, plan, buffers);
                } else if constexpr (std::is_same_v<T, MatMulActivationStep>) {
                    execute_matmul_activation(s, buffers);
                } else if constexpr (std::is_same_v<T, FusedReductionStep>) {
                    execute_fused_reduction(s, plan, buffers);
                }
            },
            step);
    }

    // Return the output
    if (plan.output_slot >= 0 &&
        plan.output_slot < static_cast<int>(buffers.size())) {
        return buffers[plan.output_slot];
    }

    throw RuntimeError("CompiledGraph has no valid output slot");
}

namespace {

// ============================================================================
// SingleOp: dispatch via OperationRegistry (current behavior)
// ============================================================================

void execute_single_op(const SingleOpStep &step, std::vector<Tensor> &buffers) {
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
        const auto &rp = get_params<ReductionParams>(step.params);
        result = operation->execute_reduction(get_input(indices[0]), rp.axes,
                                              rp.keep_dims);
    } else if (op == ops::OpType::MatMul || op == ops::OpType::BatchMatMul) {
        const auto &mp = get_params<MatMulParams>(step.params);
        result = operation->execute_matmul(get_input(indices[0]),
                                           get_input(indices[1]),
                                           mp.transpose_a, mp.transpose_b);
    } else if (op == ops::OpType::Softmax || op == ops::OpType::LogSoftmax) {
        const auto &ap = get_params<ActivationParams>(step.params);
        result = operation->execute_reduction(get_input(indices[0]), {ap.axis},
                                              false);
    } else {
        throw RuntimeError("Unsupported op type in execute_single_op");
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
// spawn overhead is only amortized at larger sizes (~512K+ elements).
static constexpr size_t FUSED_KNOWN_MIN_PARALLEL = 524288;

void execute_fused_known(const FusedKnownStep &step, const CompiledGraph &plan,
                         std::vector<Tensor> &buffers) {
    // Collect external inputs (not chain-internal results)
    std::vector<Tensor> inputs;
    std::unordered_set<int> seen_slots;

    // For fused chains, inputs with index -1 are chain-internal
    for (const auto &per_op : step.input_slot_indices) {
        for (int s : per_op) {
            if (s >= 0 && s < static_cast<int>(buffers.size())) {
                // Dedup by slot index (not storage pointer — arena-backed
                // slots can share a Storage yet hold distinct data).
                if (seen_slots.insert(s).second) {
                    inputs.push_back(buffers[s]);
                }
            }
        }
    }

    if (!is_fused_simd_dtype(step.output_dtype) || step.device != Device::CPU) {
#ifdef AXIOM_METAL_SUPPORT
        // Try GPU fused path (single MPSGraph for entire chain)
        if (step.device == Device::GPU &&
            execute_gpu_fused_chain(step, step.op_chain, buffers))
            return;
#endif
        // Fallback to generic
        if (debug_fusion())
            std::fprintf(stderr, "[axiom:fusion] FusedKnown -> generic: "
                                 "unsupported dtype or non-CPU device\n");
        FusedGenericStep generic_step;
        static_cast<StepBase &>(generic_step) =
            static_cast<const StepBase &>(step);
        generic_step.op_chain = step.op_chain;
        execute_fused_generic(generic_step, plan, buffers);
        return;
    }

    // Check inputs are contiguous and same size as output (no broadcast)
    for (const auto &t : inputs) {
        if (!t.is_contiguous() || t.size() != step.total_elements) {
            if (debug_fusion())
                std::fprintf(stderr, "[axiom:fusion] FusedKnown -> generic: "
                                     "non-contiguous or broadcast input\n");
            FusedGenericStep generic_step;
            static_cast<StepBase &>(generic_step) =
                static_cast<const StepBase &>(step);
            generic_step.op_chain = step.op_chain;
            execute_fused_generic(generic_step, plan, buffers);
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
        std::atomic<bool> ok{true};
#pragma omp parallel for schedule(static)
        for (ptrdiff_t ci = 0; ci < nchunks; ++ci) {
            size_t start = static_cast<size_t>(ci) * chunk;
            size_t count = std::min(chunk, n - start);
            if (!dispatch_fused_pattern(step.pattern, inputs, result, start,
                                        count)) {
                ok.store(false, std::memory_order_relaxed);
            }
        }
        if (ok.load(std::memory_order_relaxed)) {
            buffers[step.output_slot] = result;
            return;
        }
    } else {
        if (dispatch_fused_pattern(step.pattern, inputs, result)) {
            buffers[step.output_slot] = result;
            return;
        }
    }

    // Fallback
    if (debug_fusion())
        std::fprintf(stderr,
                     "[axiom:fusion] FusedKnown -> generic: SIMD dispatch "
                     "failed for pattern\n");
    FusedGenericStep generic_step;
    static_cast<StepBase &>(generic_step) = static_cast<const StepBase &>(step);
    generic_step.op_chain = step.op_chain;
    execute_fused_generic(generic_step, plan, buffers);
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
static bool try_tiled_fused_loop(const FusedGenericStep &step,
                                 std::vector<Tensor> &buffers) {
    if (step.device != Device::CPU)
        return false;

    DType dtype = step.output_dtype;
    size_t elem_size = dtype_size(dtype);
    if (elem_size == 0)
        return false;

    auto descs =
        resolve_chain_fn_ptrs(step.op_chain, step.input_slot_indices, dtype);
    if (descs.empty() && !step.op_chain.empty())
        return false;

    if (!inputs_are_contiguous(step.input_slot_indices, buffers,
                               step.total_elements))
        return false;

    size_t total = step.total_elements;
    size_t tile_size = std::min(TILE_ELEMENTS, total);

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

void execute_fused_generic(const FusedGenericStep &step,
                           const CompiledGraph & /*plan*/,
                           std::vector<Tensor> &buffers) {
    // Try the fast tiled path first
    if (try_tiled_fused_loop(step, buffers))
        return;

#ifdef AXIOM_METAL_SUPPORT
    // Try GPU fused path before falling back to op-by-op
    if (step.device == Device::GPU &&
        execute_gpu_fused_chain(step, step.op_chain, buffers))
        return;
#endif

    // Fallback: sequential op-by-op via OperationRegistry
    if (debug_fusion())
        std::fprintf(stderr, "[axiom:fusion] FusedGeneric -> op-by-op: "
                             "tiled loop and GPU paths unavailable\n");
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

void execute_matmul_activation(const MatMulActivationStep &step,
                               std::vector<Tensor> &buffers) {
    if (step.op_chain.size() < 2) {
        // Fallback: just execute as single matmul
        SingleOpStep single;
        static_cast<StepBase &>(single) = static_cast<const StepBase &>(step);
        single.op_type = step.op_type;
        single.params = step.params;
        execute_single_op(single, buffers);
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

    const auto &mp = get_params<MatMulParams>(step.params);
    Tensor result = operation->execute_matmul(get_buf(mm_indices[0]),
                                              get_buf(mm_indices[1]),
                                              mp.transpose_a, mp.transpose_b);

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
void execute_fused_reduction_fallback(const FusedReductionStep &step,
                                      const CompiledGraph &plan,
                                      std::vector<Tensor> &buffers) {
    // Build a temporary step for just the elementwise chain
    FusedGenericStep ew_step;
    ew_step.op_chain = step.op_chain;
    // input_slot_indices: only the elementwise ops' indices
    // (excl. the reduction's input which is the chain output)
    ew_step.input_slot_indices.assign(
        step.input_slot_indices.begin(),
        step.input_slot_indices.begin() +
            static_cast<ptrdiff_t>(step.op_chain.size()));
    ew_step.device = step.device;
    // Use chain_dtype for the elementwise chain, NOT output_dtype
    // (e.g. mean(int32) has output_dtype=Float32 but chain operates on Int32)
    ew_step.output_dtype = step.chain_dtype;
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

    execute_fused_generic(ew_step, plan, buffers);

    // Now execute the reduction on the elementwise result
    Tensor ew_result = buffers[temp_slot];
    ops::OpType red_op = step.reduction_op;
    const ops::Operation *operation =
        ops::OperationRegistry::get_operation(red_op, step.device);
    if (!operation) {
        operation = ops::OperationRegistry::get_operation(red_op, Device::CPU);
    }
    if (!operation) {
        buffers.pop_back(); // remove temporary slot
        throw DeviceError("Reduction operation not available for any device");
    }
    const auto &rp = get_params<ReductionParams>(step.params);
    Tensor result =
        operation->execute_reduction(ew_result, rp.axes, rp.keep_dims);
    if (step.output_slot >= 0 &&
        step.output_slot < static_cast<int>(buffers.size())) {
        buffers[step.output_slot] = result;
    }
    buffers.pop_back(); // remove temporary slot
}

void execute_fused_reduction(const FusedReductionStep &step,
                             const CompiledGraph &plan,
                             std::vector<Tensor> &buffers) {
    if (step.device != Device::CPU) {
#ifdef AXIOM_METAL_SUPPORT
        // Try GPU fused reduction (single MPSGraph for chain + reduction)
        if (step.device == Device::GPU &&
            execute_gpu_fused_reduction(step, buffers))
            return;
#endif
        // Fallback: execute elementwise chain + reduction separately
        execute_fused_reduction_fallback(step, plan, buffers);
        return;
    }

    // Use chain_dtype for fn-ptr resolution — this is the dtype the
    // elementwise chain operates on, which may differ from output_dtype
    // (e.g. mean(int32) has output_dtype=Float32 but chain is Int32).
    DType chain_dtype = step.chain_dtype;
    // Fallback: if chain_dtype wasn't set, use output_dtype
    if (chain_dtype == DType{})
        chain_dtype = step.output_dtype;

    size_t elem_size = dtype_size(chain_dtype);
    if (elem_size == 0) {
        execute_fused_reduction_fallback(step, plan, buffers);
        return;
    }

    // Resolve fn-ptrs for the elementwise chain (excl. reduction)
    auto descs = resolve_chain_fn_ptrs(step.op_chain, step.input_slot_indices,
                                       chain_dtype);
    if (descs.empty() && !step.op_chain.empty()) {
        execute_fused_reduction_fallback(step, plan, buffers);
        return;
    }

    // Check inputs are contiguous (broadcast would need coordinate remapping)
    if (!inputs_are_contiguous(step.input_slot_indices, buffers,
                               step.total_elements)) {
        execute_fused_reduction_fallback(step, plan, buffers);
        return;
    }

    // Determine the input element count (the reduction input, not output)
    // step.total_elements was set to the elementwise chain input size
    size_t total = step.total_elements;
    if (total == 0) {
        FusedGenericStep generic_step;
        static_cast<StepBase &>(generic_step) =
            static_cast<const StepBase &>(step);
        generic_step.op_chain = step.op_chain;
        execute_fused_generic(generic_step, plan, buffers);
        return;
    }

    size_t tile_size = std::min(TILE_ELEMENTS, total);

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

    // Process tile: run elementwise chain, then reduce the result
    auto process_tile_and_reduce = [&](size_t base, size_t count, void *tile_a,
                                       void *tile_b) -> double {
        void *prev = nullptr;
        bool prev_is_a = true;

        for (size_t oi = 0; oi < descs.size(); ++oi) {
            const auto &d = descs[oi];
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

        return reduce_tile(prev, count, chain_dtype, red_op);
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
#pragma omp parallel
            {
                double thread_max = -std::numeric_limits<double>::infinity();
                TileBuffer la, lb;
#pragma omp for schedule(static)
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
                    if (thread_max > acc)
                        acc = thread_max;
                }
            }
        } else if (red_op == ops::OpType::Min) {
#pragma omp parallel
            {
                double thread_min = std::numeric_limits<double>::infinity();
                TileBuffer la, lb;
#pragma omp for schedule(static)
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
                    if (thread_min < acc)
                        acc = thread_min;
                }
            }
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

    Tensor result(step.output_shape, step.output_dtype, Device::CPU);
    store_scalar_result(result, acc);

    if (step.output_slot >= 0 &&
        step.output_slot < static_cast<int>(buffers.size())) {
        buffers[step.output_slot] = result;
    }
}

} // anonymous namespace

} // namespace graph
} // namespace axiom
