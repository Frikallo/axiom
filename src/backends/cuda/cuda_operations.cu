#include "cuda_operations.hpp"
#include "cublas_operations.hpp"
#include "cuda_buffer_provider.hpp"
#include "cuda_context.hpp"
#include "cuda_kernels.hpp"

#include "axiom/dtype.hpp"
#include "axiom/error.hpp"
#include "axiom/operations.hpp"
#include "axiom/shape.hpp"
#include "axiom/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace axiom {
namespace backends {
namespace cuda {

// ============================================================================
// ensure_gpu_contiguous — every CUDA op should call this on its inputs
// ============================================================================

Tensor ensure_gpu_contiguous(const Tensor &t) {
    if (t.is_contiguous()) return t;

#ifdef AXIOM_CUDA_SUPPORT
    Tensor result(t.shape(), t.dtype(), Device::GPU);

    auto *src_provider = as_cuda_buffer_provider(t.storage().get());
    auto *dst_provider = as_cuda_buffer_provider(result.storage().get());
    if (!src_provider || !dst_provider) {
        throw DeviceError("ensure_gpu_contiguous: storage is not CUDA-backed");
    }

    const auto *src_ptr =
        static_cast<const uint8_t *>(src_provider->device_ptr()) +
        src_provider->offset() + t.offset();
    auto *dst_ptr =
        static_cast<uint8_t *>(dst_provider->device_ptr()) +
        dst_provider->offset();

    GatherStridedParams params{};
    params.ndim = static_cast<unsigned int>(t.ndim());
    params.numel = static_cast<unsigned int>(t.size());
    params.offset = 0;
    params.itemsize = static_cast<unsigned int>(t.itemsize());
    params.flip_mask = 0;

    for (size_t i = 0; i < t.ndim(); ++i) {
        params.shape[i] = static_cast<unsigned int>(t.shape()[i]);
        int64_t stride = t.strides()[i];
        if (stride < 0) {
            params.flip_mask |= (1u << i);
        }
        params.src_strides[i] =
            static_cast<unsigned int>(std::abs(stride) /
                                      static_cast<int64_t>(t.itemsize()));
    }

    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());

    launch_gather_strided(src_ptr, dst_ptr, params, t.itemsize(), stream);

    CudaExecutionStream::instance().increment_batch();
    return result;
#else
    throw DeviceError("CUDA support not compiled");
#endif
}

// ============================================================================
// OpType → BinaryOpKind mapping
// ============================================================================

static bool is_comparison_or_logical(ops::OpType op) {
    switch (op) {
    case ops::OpType::Equal:
    case ops::OpType::NotEqual:
    case ops::OpType::Less:
    case ops::OpType::LessEqual:
    case ops::OpType::Greater:
    case ops::OpType::GreaterEqual:
    case ops::OpType::LogicalAnd:
    case ops::OpType::LogicalOr:
    case ops::OpType::LogicalXor:
        return true;
    default:
        return false;
    }
}

static BinaryOpKind to_binary_op_kind(ops::OpType op) {
    switch (op) {
    case ops::OpType::Add:          return BinaryOpKind::Add;
    case ops::OpType::Subtract:     return BinaryOpKind::Sub;
    case ops::OpType::Multiply:     return BinaryOpKind::Mul;
    case ops::OpType::Divide:       return BinaryOpKind::Div;
    case ops::OpType::Power:        return BinaryOpKind::Pow;
    case ops::OpType::Modulo:       return BinaryOpKind::Mod;
    case ops::OpType::Maximum:      return BinaryOpKind::Max;
    case ops::OpType::Minimum:      return BinaryOpKind::Min;
    case ops::OpType::Atan2:        return BinaryOpKind::Atan2;
    case ops::OpType::Hypot:        return BinaryOpKind::Hypot;
    case ops::OpType::Equal:        return BinaryOpKind::Equal;
    case ops::OpType::NotEqual:     return BinaryOpKind::NotEqual;
    case ops::OpType::Less:         return BinaryOpKind::Less;
    case ops::OpType::LessEqual:    return BinaryOpKind::LessEqual;
    case ops::OpType::Greater:      return BinaryOpKind::Greater;
    case ops::OpType::GreaterEqual: return BinaryOpKind::GreaterEqual;
    case ops::OpType::LogicalAnd:   return BinaryOpKind::LogicalAnd;
    case ops::OpType::LogicalOr:    return BinaryOpKind::LogicalOr;
    case ops::OpType::LogicalXor:   return BinaryOpKind::LogicalXor;
    default:
        throw DeviceError("Unsupported binary OpType for CUDA");
    }
}

// ============================================================================
// CudaBinaryOperation
// ============================================================================

class CudaBinaryOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string op_name_;

  public:
    CudaBinaryOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    bool supports_binary(const Tensor &lhs,
                         const Tensor &rhs) const override {
        return ops::are_broadcastable(lhs.shape(), rhs.shape());
    }

    Tensor execute_binary(const Tensor &lhs,
                          const Tensor &rhs) const override {
#ifdef AXIOM_CUDA_SUPPORT
        // Type promotion
        DType result_dtype = ops::result_type(lhs, rhs);
        if (is_comparison_or_logical(op_type_)) {
            result_dtype = DType::Bool;
        }

        // Promote inputs if needed
        Tensor lhs_p = (lhs.dtype() == result_dtype || is_comparison_or_logical(op_type_))
                            ? lhs : lhs.astype(result_dtype);
        Tensor rhs_p = (rhs.dtype() == result_dtype || is_comparison_or_logical(op_type_))
                            ? rhs : rhs.astype(result_dtype);

        // For comparison/logical, promote both inputs to the same type
        if (is_comparison_or_logical(op_type_) && lhs_p.dtype() != rhs_p.dtype()) {
            DType common = ops::promote_types(lhs_p.dtype(), rhs_p.dtype());
            if (lhs_p.dtype() != common) lhs_p = lhs_p.astype(common);
            if (rhs_p.dtype() != common) rhs_p = rhs_p.astype(common);
        }

        // Ensure contiguous
        Tensor lhs_c = ensure_gpu_contiguous(lhs_p);
        Tensor rhs_c = ensure_gpu_contiguous(rhs_p);

        // Compute broadcast info
        auto bcast = ops::compute_broadcast_info(lhs_c.shape(), rhs_c.shape());

        // Allocate output
        Tensor result(bcast.result_shape, result_dtype, Device::GPU);
        size_t numel = result.size();
        if (numel == 0) return result;

        // Extract device pointers
        auto *lhs_buf = as_cuda_buffer_provider(lhs_c.storage().get());
        auto *rhs_buf = as_cuda_buffer_provider(rhs_c.storage().get());
        auto *out_buf = as_cuda_buffer_provider(result.storage().get());
        if (!lhs_buf || !rhs_buf || !out_buf) {
            throw DeviceError("CudaBinaryOperation: storage is not CUDA-backed");
        }

        const void *lhs_ptr = lhs_buf->device_ptr();
        const void *rhs_ptr = rhs_buf->device_ptr();
        void *out_ptr = out_buf->device_ptr();

        auto stream =
            static_cast<cudaStream_t>(CudaContext::instance().stream());
        BinaryOpKind kind = to_binary_op_kind(op_type_);

        // Input element size (for comparison/logical, inputs may differ
        // from output size which is always 1 byte / Bool).
        DType input_dtype = is_comparison_or_logical(op_type_)
                                ? lhs_c.dtype()
                                : result_dtype;
        size_t input_elem = dtype_size(input_dtype);

        if (bcast.needs_broadcast) {
            // Build broadcast params
            BroadcastParams bp{};
            bp.ndim = static_cast<int>(bcast.result_shape.size());

            Strides a_strides = ShapeUtils::broadcast_strides(
                lhs_c.shape(), lhs_c.strides(), bcast.result_shape);
            Strides b_strides = ShapeUtils::broadcast_strides(
                rhs_c.shape(), rhs_c.strides(), bcast.result_shape);

            for (int i = 0; i < bp.ndim; ++i) {
                bp.out_shape[i] =
                    static_cast<int64_t>(bcast.result_shape[i]);
                // Convert byte strides to element strides
                bp.a_strides[i] = a_strides[i] /
                    static_cast<int64_t>(input_elem);
                bp.b_strides[i] = b_strides[i] /
                    static_cast<int64_t>(input_elem);
            }

            launch_binary_broadcast(kind, lhs_ptr, rhs_ptr, out_ptr,
                                    numel, bp, input_elem, stream);
        } else {
            launch_binary_elementwise(kind, lhs_ptr, rhs_ptr, out_ptr,
                                      numel, input_elem, stream);
        }

        CudaExecutionStream::instance().increment_batch();
        return result;
#else
        (void)lhs;
        (void)rhs;
        throw DeviceError("CUDA support not compiled");
#endif
    }
};

// ============================================================================
// Operation registration
// ============================================================================

static void register_binary_op(ops::OpType op_type,
                                const std::string &name) {
    ops::OperationRegistry::register_operation(
        op_type, Device::GPU,
        std::make_unique<CudaBinaryOperation>(op_type, name));
}

void register_cuda_operations() {
    if (!is_cuda_available()) return;

    // Arithmetic
    register_binary_op(ops::OpType::Add, "add");
    register_binary_op(ops::OpType::Subtract, "subtract");
    register_binary_op(ops::OpType::Multiply, "multiply");
    register_binary_op(ops::OpType::Divide, "divide");
    register_binary_op(ops::OpType::Power, "power");
    register_binary_op(ops::OpType::Modulo, "modulo");

    // Math
    register_binary_op(ops::OpType::Maximum, "maximum");
    register_binary_op(ops::OpType::Minimum, "minimum");
    register_binary_op(ops::OpType::Atan2, "atan2");
    register_binary_op(ops::OpType::Hypot, "hypot");

    // Comparison
    register_binary_op(ops::OpType::Equal, "equal");
    register_binary_op(ops::OpType::NotEqual, "not_equal");
    register_binary_op(ops::OpType::Less, "less");
    register_binary_op(ops::OpType::LessEqual, "less_equal");
    register_binary_op(ops::OpType::Greater, "greater");
    register_binary_op(ops::OpType::GreaterEqual, "greater_equal");

    // Logical
    register_binary_op(ops::OpType::LogicalAnd, "logical_and");
    register_binary_op(ops::OpType::LogicalOr, "logical_or");
    register_binary_op(ops::OpType::LogicalXor, "logical_xor");

    register_cublas_operations();

    // TODO: register unary and reduction operations
}

} // namespace cuda
} // namespace backends
} // namespace axiom
