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

    // Pre-allocate output buffers using the memory plan
    for (size_t i = 0; i < plan.buffer_slots.size(); ++i) {
        if (plan.buffer_slots[i].is_input)
            continue;
        if (buffers[i].storage())
            continue; // already bound

        const auto &slot = plan.buffer_slots[i];
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
        // MatMul + activation: execute matmul first, then activation
        execute_single_op(step, buffers);
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
    } else if (op == ops::OpType::MatMul ||
               op == ops::OpType::BatchMatMul) {
        result = operation->execute_matmul(
            get_input(indices[0]), get_input(indices[1]),
            step.params.transpose_a, step.params.transpose_b);
    } else if (op == ops::OpType::Softmax ||
               op == ops::OpType::LogSoftmax) {
        result = operation->execute_reduction(
            get_input(indices[0]), {step.params.axis}, false);
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
                    if (t.storage() == buffers[s].storage())  {
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

    if (!is_fused_simd_dtype(step.output_dtype) ||
        step.device != Device::CPU) {
        // Fallback to generic
        execute_fused_generic(step, plan, buffers);
        return;
    }

    // Check inputs are contiguous
    for (const auto &t : inputs) {
        if (!t.is_contiguous()) {
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
// FusedGeneric: sequential op-by-op execution (fallback)
// ============================================================================

void GraphExecutor::execute_fused_generic(const ExecutionStep &step,
                                          const CompiledGraph & /*plan*/,
                                          std::vector<Tensor> &buffers) {
    Device device = step.device;
    Tensor current;

    for (size_t i = 0; i < step.op_chain.size(); ++i) {
        ops::OpType op = step.op_chain[i];

        const ops::Operation *operation =
            ops::OperationRegistry::get_operation(op, device);
        if (!operation) {
            operation =
                ops::OperationRegistry::get_operation(op, Device::CPU);
            device = Device::CPU;
        }
        if (!operation) {
            throw DeviceError("Operation not available in fused generic");
        }

        const auto &indices = step.input_slot_indices[i];

        auto resolve = [&](int idx) -> Tensor {
            if (idx == -1)
                return current; // previous result
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

} // namespace graph
} // namespace axiom
