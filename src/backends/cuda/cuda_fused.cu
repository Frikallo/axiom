#include "cuda_fused.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include "cuda_allocator.hpp"
#include "cuda_buffer_provider.hpp"
#include "cuda_context.hpp"
#include "cuda_fusion_cache.hpp"
#include "cuda_kernels.hpp"
#include "nvrtc_codegen.hpp"
#include "nvrtc_compiler.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <unordered_set>
#include <vector>
#endif

namespace axiom {
namespace backends {
namespace cuda {

void register_cuda_fused_operations() {
    // Fused patterns are handled via NVRTC codegen in execute_cuda_fused_chain.
}

} // namespace cuda
} // namespace backends

#ifdef AXIOM_CUDA_SUPPORT

namespace graph {

using namespace backends::cuda;

// ============================================================================
// Helper: collect unique external input slots from step.input_slot_indices
// ============================================================================

static std::vector<int> collect_external_slots(
    const std::vector<std::vector<int>> &input_slot_indices) {
    std::vector<int> slots;
    std::unordered_set<int> seen;
    for (const auto &per_op : input_slot_indices) {
        for (int s : per_op) {
            if (s >= 0 && seen.insert(s).second)
                slots.push_back(s);
        }
    }
    return slots;
}

// ============================================================================
// Map slot indices to a contiguous external-input numbering.
// Returns a vector where result[i] is the external input ordinal for slot i,
// or -1 if the slot is not an external input.
// ============================================================================

static std::vector<int> build_slot_to_external(const std::vector<int> &ext_slots,
                                               int max_slot) {
    std::vector<int> mapping(max_slot + 1, -1);
    for (size_t i = 0; i < ext_slots.size(); ++i)
        mapping[ext_slots[i]] = static_cast<int>(i);
    return mapping;
}

// ============================================================================
// Remap input_slot_indices from buffer-slot numbers to external-input ordinals
// ============================================================================

static std::vector<std::vector<int>>
remap_slots(const std::vector<std::vector<int>> &input_slot_indices,
            const std::vector<int> &slot_to_external) {
    std::vector<std::vector<int>> remapped;
    remapped.reserve(input_slot_indices.size());
    for (const auto &per_op : input_slot_indices) {
        std::vector<int> r;
        r.reserve(per_op.size());
        for (int s : per_op)
            r.push_back(s < 0 ? -1 : slot_to_external[s]);
        remapped.push_back(std::move(r));
    }
    return remapped;
}

// ============================================================================
// execute_cuda_fused_chain
// ============================================================================

bool execute_cuda_fused_chain(const StepBase &step,
                              const std::vector<ops::OpType> &op_chain,
                              std::vector<Tensor> &buffers) {
    if (op_chain.empty())
        return false;

    // Check dtype support
    if (!is_nvrtc_supported_dtype(step.output_dtype))
        return false;

    // Check all ops supported
    for (auto op : op_chain) {
        if (!is_nvrtc_supported_op(op))
            return false;
    }

    // Collect external input slots
    auto ext_slots = collect_external_slots(step.input_slot_indices);
    if (ext_slots.empty())
        return false;

    // Validate all inputs are GPU + contiguous
    for (int slot : ext_slots) {
        const Tensor &t = buffers[slot];
        if (t.device() != Device::GPU)
            return false;
        if (!t.is_contiguous())
            return false;
    }

    // Detect broadcast
    bool needs_broadcast = false;
    for (int slot : ext_slots) {
        if (static_cast<size_t>(buffers[slot].size()) != step.total_elements) {
            needs_broadcast = true;
            break;
        }
    }

    // Build cache key
    FusionCacheKey key;
    key.op_chain = op_chain;
    key.output_dtype = step.output_dtype;
    key.has_broadcast = needs_broadcast;

    for (int slot : ext_slots)
        key.input_dtypes.push_back(buffers[slot].dtype());

    if (needs_broadcast) {
        key.ndim = static_cast<int>(step.output_shape.size());
        for (int slot : ext_slots) {
            const auto &t = buffers[slot];
            uint32_t mask = 0;
            auto t_shape = t.shape();
            int t_ndim = static_cast<int>(t_shape.size());
            int offset = key.ndim - t_ndim;
            for (int d = 0; d < key.ndim; ++d) {
                int td = d - offset;
                if (td < 0 || t_shape[td] == 1)
                    mask |= (1u << d);
            }
            key.broadcast_masks.push_back(mask);
        }
    }

    // Find max slot for mapping
    int max_slot = 0;
    for (int s : ext_slots)
        max_slot = std::max(max_slot, s);
    auto slot_to_ext = build_slot_to_external(ext_slots, max_slot);
    auto remapped = remap_slots(step.input_slot_indices, slot_to_ext);

    // Lookup or compile
    size_t num_inputs = ext_slots.size();
    DType compute_dtype = step.output_dtype;
    int ndim = static_cast<int>(step.output_shape.size());

    CachedFusedKernel *cached = FusedKernelCache::instance().lookup_or_compile(
        key, [&]() -> GeneratedKernel {
            FusedKernelSpec spec;
            spec.op_chain = op_chain;
            spec.input_slot_indices = remapped;
            spec.compute_dtype = compute_dtype;
            spec.output_dtype = compute_dtype;
            spec.num_external_inputs = num_inputs;
            spec.needs_broadcast = needs_broadcast;
            if (needs_broadcast) {
                spec.ndim = ndim;
                spec.output_shape.assign(step.output_shape.begin(),
                                         step.output_shape.end());
                // Compute broadcast strides
                for (int slot : ext_slots) {
                    const auto &t = buffers[slot];
                    auto t_shape = t.shape();
                    int t_ndim = static_cast<int>(t_shape.size());
                    int offset = ndim - t_ndim;
                    std::vector<int64_t> strides(ndim, 0);
                    // Compute contiguous strides for this input, with 0 for
                    // broadcast dims
                    int64_t stride = 1;
                    for (int d = t_ndim - 1; d >= 0; --d) {
                        if (t_shape[d] != 1)
                            strides[d + offset] = stride;
                        stride *= t_shape[d];
                    }
                    spec.input_strides.push_back(std::move(strides));
                }
            }
            return generate_fused_kernel(spec);
        });

    if (!cached)
        return false;

    // Allocate output
    Tensor output(step.output_shape, step.output_dtype, Device::GPU);
    size_t n = step.total_elements;
    if (n == 0) {
        buffers[step.output_slot] = std::move(output);
        return true;
    }

    // Gather device pointers
    std::vector<void *> input_ptrs;
    input_ptrs.reserve(num_inputs);
    for (int slot : ext_slots) {
        auto *bp = as_cuda_buffer_provider(buffers[slot].storage().get());
        if (!bp)
            return false;
        input_ptrs.push_back(bp->device_ptr());
    }

    auto *out_bp = as_cuda_buffer_provider(output.storage().get());
    if (!out_bp)
        return false;
    void *out_ptr = out_bp->device_ptr();

    auto stream = static_cast<cudaStream_t>(CudaContext::instance().stream());
    int block = cached->block_size;
    int grid = static_cast<int>((n + block - 1) / block);
    unsigned long long n_ull = static_cast<unsigned long long>(n);

    if (!needs_broadcast) {
        // Build args: [&in0, &in1, ..., &out, &n]
        std::vector<void *> args;
        args.reserve(num_inputs + 2);
        for (auto &p : input_ptrs)
            args.push_back(&p);
        args.push_back(&out_ptr);
        args.push_back(&n_ull);

        auto err = cuLaunchKernel(
            cached->compiled.function, grid, 1, 1, block, 1, 1, 0,
            static_cast<CUstream>(stream), args.data(), nullptr);
        if (err != CUDA_SUCCESS)
            return false;
    } else {
        // Broadcast path: upload strides and shape arrays to device
        size_t strides_total = num_inputs * ndim;
        size_t strides_bytes = strides_total * sizeof(int64_t);
        size_t shape_bytes = ndim * sizeof(int64_t);

        auto &alloc = CudaAllocator::instance();
        auto *d_strides =
            static_cast<int64_t *>(alloc.allocate(strides_bytes));
        auto *d_shape = static_cast<int64_t *>(alloc.allocate(shape_bytes));

        // Fill strides (interleaved: input0[dim0..ndim-1], input1[...], ...)
        std::vector<int64_t> h_strides(strides_total);
        for (size_t i = 0; i < num_inputs; ++i) {
            const auto &t = buffers[ext_slots[i]];
            auto t_shape = t.shape();
            int t_ndim = static_cast<int>(t_shape.size());
            int off = ndim - t_ndim;
            int64_t stride = 1;
            for (int d = t_ndim - 1; d >= 0; --d) {
                if (t_shape[d] != 1)
                    h_strides[i * ndim + d + off] = stride;
                stride *= t_shape[d];
            }
        }

        std::vector<int64_t> h_shape(step.output_shape.begin(),
                                     step.output_shape.end());

        cudaMemcpyAsync(d_strides, h_strides.data(), strides_bytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_shape, h_shape.data(), shape_bytes,
                        cudaMemcpyHostToDevice, stream);

        // Build args: [&in0, ..., &out, &n, &d_strides, &d_shape, &ndim]
        std::vector<void *> args;
        args.reserve(num_inputs + 5);
        for (auto &p : input_ptrs)
            args.push_back(&p);
        args.push_back(&out_ptr);
        args.push_back(&n_ull);
        args.push_back(&d_strides);
        args.push_back(&d_shape);
        args.push_back(&ndim);

        auto err = cuLaunchKernel(
            cached->compiled.function, grid, 1, 1, block, 1, 1, 0,
            static_cast<CUstream>(stream), args.data(), nullptr);

        // Schedule deallocation after stream completes
        // For simplicity, sync and free (strides/shape are small)
        cudaStreamSynchronize(stream);
        alloc.deallocate(d_strides, strides_bytes);
        alloc.deallocate(d_shape, shape_bytes);

        if (err != CUDA_SUCCESS)
            return false;
    }

    CudaExecutionStream::instance().increment_batch();
    buffers[step.output_slot] = std::move(output);
    return true;
}

// ============================================================================
// execute_cuda_fused_reduction
// ============================================================================

// Map graph OpType to ReduceOpKind for the CUB reduction dispatch.
static ReduceOpKind reduction_op_kind(ops::OpType op) {
    switch (op) {
    case ops::OpType::Sum:
        return ReduceOpKind::Sum;
    case ops::OpType::Mean:
        return ReduceOpKind::Sum; // divide by n after
    case ops::OpType::Max:
        return ReduceOpKind::Max;
    case ops::OpType::Min:
        return ReduceOpKind::Min;
    case ops::OpType::Prod:
        return ReduceOpKind::Prod;
    case ops::OpType::Any:
        return ReduceOpKind::Any;
    case ops::OpType::All:
        return ReduceOpKind::All;
    default:
        return ReduceOpKind::Sum;
    }
}

bool execute_cuda_fused_reduction(const FusedReductionStep &step,
                                  std::vector<Tensor> &buffers) {
    // Step 1: execute elementwise chain into a temp tensor
    if (step.op_chain.empty()) {
        // No chain — just a plain reduction, let the normal path handle it
        return false;
    }

    // Build a temporary StepBase for the chain portion
    StepBase chain_step;
    chain_step.output_slot = -1; // will use a temporary
    chain_step.total_elements = step.total_elements;
    chain_step.output_shape =
        step.chain_shape.empty() ? step.output_shape : step.chain_shape;
    DType chain_dtype =
        step.chain_dtype != DType{} ? step.chain_dtype : step.output_dtype;
    chain_step.output_dtype = chain_dtype;
    chain_step.device = step.device;
    chain_step.input_slot_indices = step.input_slot_indices;
    chain_step.input_access = step.input_access;

    // Allocate a temporary slot for chain output
    size_t temp_slot = buffers.size();
    buffers.emplace_back(); // placeholder
    chain_step.output_slot = static_cast<int>(temp_slot);

    bool chain_ok =
        execute_cuda_fused_chain(chain_step, step.op_chain, buffers);
    if (!chain_ok) {
        buffers.pop_back();
        return false;
    }

    // Step 2: reduce the temp tensor
    const Tensor &chain_out = buffers[temp_slot];
    auto *src_bp = as_cuda_buffer_provider(chain_out.storage().get());
    if (!src_bp) {
        buffers.pop_back();
        return false;
    }

    size_t n = chain_out.size();
    size_t elem_size = dtype_size(chain_dtype);
    auto stream = static_cast<cudaStream_t>(CudaContext::instance().stream());
    ReduceOpKind kind = reduction_op_kind(step.reduction_op);

    // Allocate scalar output
    Tensor result({}, step.output_dtype, Device::GPU);
    auto *dst_bp = as_cuda_buffer_provider(result.storage().get());
    if (!dst_bp) {
        buffers.pop_back();
        return false;
    }

    // CUB two-pass: query temp size, allocate, execute
    size_t temp_bytes = 0;
    launch_full_reduce(kind, src_bp->device_ptr(), dst_bp->device_ptr(), n,
                       elem_size, nullptr, temp_bytes, stream);

    auto &alloc = CudaAllocator::instance();
    void *temp_storage = alloc.allocate(temp_bytes);
    launch_full_reduce(kind, src_bp->device_ptr(), dst_bp->device_ptr(), n,
                       elem_size, temp_storage, temp_bytes, stream);
    alloc.deallocate(temp_storage, temp_bytes);

    // Mean: divide by n
    if (step.reduction_op == ops::OpType::Mean && n > 0) {
        // Launch a simple scalar division kernel via existing unary reciprocal
        // + multiply, or just use a memcpy + host divide after sync.
        // Simplest correct approach: sync, read, divide, write back.
        cudaStreamSynchronize(stream);

        if (chain_dtype == DType::Float32) {
            float val = 0;
            cudaMemcpy(&val, dst_bp->device_ptr(), sizeof(float),
                       cudaMemcpyDeviceToHost);
            val /= static_cast<float>(n);
            cudaMemcpy(dst_bp->device_ptr(), &val, sizeof(float),
                       cudaMemcpyHostToDevice);
        } else if (chain_dtype == DType::Float64) {
            double val = 0;
            cudaMemcpy(&val, dst_bp->device_ptr(), sizeof(double),
                       cudaMemcpyDeviceToHost);
            val /= static_cast<double>(n);
            cudaMemcpy(dst_bp->device_ptr(), &val, sizeof(double),
                       cudaMemcpyHostToDevice);
        }
    }

    CudaExecutionStream::instance().increment_batch();
    buffers[step.output_slot] = std::move(result);

    // Clean up temp slot
    buffers[temp_slot] = Tensor();
    if (temp_slot == buffers.size() - 1)
        buffers.pop_back();

    return true;
}

} // namespace graph
#endif

} // namespace axiom
