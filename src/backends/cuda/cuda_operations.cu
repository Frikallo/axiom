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
// OpType → UnaryOpKind mapping
// ============================================================================

static bool is_unary_test_op(ops::OpType op) {
    switch (op) {
    case ops::OpType::IsNaN:
    case ops::OpType::IsInf:
    case ops::OpType::IsFinite:
        return true;
    default:
        return false;
    }
}

static UnaryOpKind to_unary_op_kind(ops::OpType op) {
    switch (op) {
    case ops::OpType::Negate:     return UnaryOpKind::Negate;
    case ops::OpType::Abs:        return UnaryOpKind::Abs;
    case ops::OpType::Sqrt:       return UnaryOpKind::Sqrt;
    case ops::OpType::Exp:        return UnaryOpKind::Exp;
    case ops::OpType::Log:        return UnaryOpKind::Log;
    case ops::OpType::Sin:        return UnaryOpKind::Sin;
    case ops::OpType::Cos:        return UnaryOpKind::Cos;
    case ops::OpType::Tan:        return UnaryOpKind::Tan;
    case ops::OpType::Tanh:       return UnaryOpKind::Tanh;
    case ops::OpType::Sign:       return UnaryOpKind::Sign;
    case ops::OpType::Floor:      return UnaryOpKind::Floor;
    case ops::OpType::Ceil:       return UnaryOpKind::Ceil;
    case ops::OpType::Trunc:      return UnaryOpKind::Trunc;
    case ops::OpType::Round:      return UnaryOpKind::Round;
    case ops::OpType::Reciprocal: return UnaryOpKind::Reciprocal;
    case ops::OpType::Square:     return UnaryOpKind::Square;
    case ops::OpType::Cbrt:       return UnaryOpKind::Cbrt;
    case ops::OpType::Erf:        return UnaryOpKind::Erf;
    case ops::OpType::IsNaN:      return UnaryOpKind::IsNaN;
    case ops::OpType::IsInf:      return UnaryOpKind::IsInf;
    case ops::OpType::IsFinite:   return UnaryOpKind::IsFinite;
    case ops::OpType::ReLU:       return UnaryOpKind::ReLU;
    case ops::OpType::LeakyReLU:  return UnaryOpKind::LeakyReLU;
    case ops::OpType::Sigmoid:    return UnaryOpKind::Sigmoid;
    case ops::OpType::SiLU:       return UnaryOpKind::SiLU;
    case ops::OpType::GELU:       return UnaryOpKind::GELU;
    default:
        throw DeviceError("Unsupported unary OpType for CUDA");
    }
}

// ============================================================================
// CudaUnaryOperation
// ============================================================================

class CudaUnaryOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string op_name_;

  public:
    CudaUnaryOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    bool supports_binary(const Tensor & /*lhs*/,
                         const Tensor & /*rhs*/) const override {
        return false;
    }

    Tensor execute_binary(const Tensor & /*lhs*/,
                          const Tensor & /*rhs*/) const override {
        throw RuntimeError::internal(
            "execute_binary called on unary operation");
    }

    Tensor execute_unary(const Tensor &input) const override {
#ifdef AXIOM_CUDA_SUPPORT
        Tensor in_c = ensure_gpu_contiguous(input);

        // Output dtype: Bool for test ops, same as input otherwise
        DType out_dtype = is_unary_test_op(op_type_) ? DType::Bool
                                                     : in_c.dtype();

        Tensor result(in_c.shape(), out_dtype, Device::GPU);
        size_t numel = result.size();
        if (numel == 0) return result;

        auto *src_buf = as_cuda_buffer_provider(in_c.storage().get());
        auto *dst_buf = as_cuda_buffer_provider(result.storage().get());
        if (!src_buf || !dst_buf) {
            throw DeviceError(
                "CudaUnaryOperation: storage is not CUDA-backed");
        }

        auto stream =
            static_cast<cudaStream_t>(CudaContext::instance().stream());

        launch_unary_elementwise(to_unary_op_kind(op_type_),
                                 src_buf->device_ptr(),
                                 dst_buf->device_ptr(), numel,
                                 dtype_size(in_c.dtype()), stream);

        CudaExecutionStream::instance().increment_batch();
        return result;
#else
        (void)input;
        throw DeviceError("CUDA support not compiled");
#endif
    }
};

// ============================================================================
// OpType → ReduceOpKind mapping
// ============================================================================

static ReduceOpKind to_reduce_op_kind(ops::OpType op) {
    switch (op) {
    case ops::OpType::Sum:  return ReduceOpKind::Sum;
    case ops::OpType::Mean: return ReduceOpKind::Sum; // Mean = Sum / count
    case ops::OpType::Max:  return ReduceOpKind::Max;
    case ops::OpType::Min:  return ReduceOpKind::Min;
    case ops::OpType::Prod: return ReduceOpKind::Prod;
    case ops::OpType::Any:  return ReduceOpKind::Any;
    case ops::OpType::All:  return ReduceOpKind::All;
    default:
        throw DeviceError("Unsupported reduction OpType for CUDA");
    }
}

// ============================================================================
// CudaReductionOperation
// ============================================================================

class CudaReductionOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string op_name_;

  public:
    CudaReductionOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    bool supports_binary(const Tensor & /*lhs*/,
                         const Tensor & /*rhs*/) const override {
        return false;
    }

    Tensor execute_binary(const Tensor & /*lhs*/,
                          const Tensor & /*rhs*/) const override {
        throw RuntimeError::internal(
            "execute_binary called on reduction operation");
    }

    Tensor execute_unary(const Tensor &input) const override {
        (void)input;
        throw RuntimeError::internal(
            "execute_unary called on reduction operation");
    }

    Tensor execute_reduction(const Tensor &input,
                             const std::vector<int> &axes,
                             bool keep_dims) const override {
#ifdef AXIOM_CUDA_SUPPORT
        Tensor in_c = ensure_gpu_contiguous(input);

        // Normalize axes: empty means reduce all
        std::vector<int> norm_axes = axes;
        if (norm_axes.empty()) {
            for (size_t i = 0; i < in_c.ndim(); ++i)
                norm_axes.push_back(static_cast<int>(i));
        }

        bool is_full_reduction = (norm_axes.size() == in_c.ndim());

        // Compute output shape
        Shape output_shape;
        if (is_full_reduction) {
            output_shape = keep_dims ? Shape(in_c.ndim(), 1) : Shape{1};
        } else {
            std::vector<bool> is_reduced(in_c.ndim(), false);
            for (int ax : norm_axes)
                is_reduced[ax] = true;
            for (size_t i = 0; i < in_c.ndim(); ++i) {
                if (is_reduced[i]) {
                    if (keep_dims) output_shape.push_back(1);
                } else {
                    output_shape.push_back(in_c.shape()[i]);
                }
            }
        }

        DType out_dtype = in_c.dtype();
        Tensor result(output_shape, out_dtype, Device::GPU);
        if (result.size() == 0) return result;

        auto *src_buf = as_cuda_buffer_provider(in_c.storage().get());
        auto *dst_buf = as_cuda_buffer_provider(result.storage().get());
        if (!src_buf || !dst_buf) {
            throw DeviceError(
                "CudaReductionOperation: storage is not CUDA-backed");
        }

        const void *src_ptr = src_buf->device_ptr();
        void *dst_ptr = dst_buf->device_ptr();
        size_t elem_size = dtype_size(in_c.dtype());
        auto stream =
            static_cast<cudaStream_t>(CudaContext::instance().stream());

        ReduceOpKind kind = to_reduce_op_kind(op_type_);

        if (is_full_reduction) {
            // Full reduction via CUB — two-pass: query temp size, then execute
            size_t temp_bytes = 0;
            launch_full_reduce(kind, src_ptr, dst_ptr, in_c.size(),
                               elem_size, nullptr, temp_bytes, stream);

            void *temp = nullptr;
            cudaMalloc(&temp, temp_bytes);
            launch_full_reduce(kind, src_ptr, dst_ptr, in_c.size(),
                               elem_size, temp, temp_bytes, stream);
            cudaFree(temp);
        } else {
            // Axis reduction — decompose into (outer, axis_len, inner).
            // Currently supports single-axis reduction.  Multi-axis is
            // handled by reducing one axis at a time.
            if (norm_axes.size() == 1) {
                int ax = norm_axes[0];
                size_t outer = 1;
                for (int i = 0; i < ax; ++i)
                    outer *= in_c.shape()[i];
                size_t axis_len = in_c.shape()[ax];
                size_t inner = 1;
                for (size_t i = ax + 1; i < in_c.ndim(); ++i)
                    inner *= in_c.shape()[i];

                launch_axis_reduce(kind, src_ptr, dst_ptr, outer, axis_len,
                                   inner, elem_size, stream);
            } else {
                // Multi-axis: reduce axes one at a time (highest first
                // to keep earlier axis indices valid)
                std::vector<int> sorted_axes = norm_axes;
                std::sort(sorted_axes.rbegin(), sorted_axes.rend());

                Tensor current = in_c;
                for (int ax : sorted_axes) {
                    Tensor cur_c = ensure_gpu_contiguous(current);
                    size_t outer = 1;
                    for (int i = 0; i < ax; ++i)
                        outer *= cur_c.shape()[i];
                    size_t axis_len = cur_c.shape()[ax];
                    size_t inner = 1;
                    for (size_t i = ax + 1; i < cur_c.ndim(); ++i)
                        inner *= cur_c.shape()[i];

                    // Intermediate shape: remove the reduced axis
                    Shape inter_shape;
                    for (size_t i = 0; i < cur_c.ndim(); ++i) {
                        if (static_cast<int>(i) != ax)
                            inter_shape.push_back(cur_c.shape()[i]);
                    }
                    if (inter_shape.empty()) inter_shape.push_back(1);

                    Tensor inter(inter_shape, out_dtype, Device::GPU);
                    auto *inter_buf =
                        as_cuda_buffer_provider(inter.storage().get());
                    auto *cur_buf =
                        as_cuda_buffer_provider(cur_c.storage().get());

                    launch_axis_reduce(kind, cur_buf->device_ptr(),
                                       inter_buf->device_ptr(), outer,
                                       axis_len, inner, elem_size, stream);
                    current = inter;
                }

                // If keep_dims, reshape to insert 1s at reduced positions
                if (keep_dims) {
                    current = current.reshape(output_shape);
                }

                // For Mean, divide by total reduced element count
                if (op_type_ == ops::OpType::Mean) {
                    size_t count = 1;
                    for (int ax : norm_axes)
                        count *= in_c.shape()[ax];
                    // Use binary division: current / count
                    Tensor divisor = Tensor::full(current.shape(),
                                                  static_cast<double>(count),
                                                  current.dtype(), Device::GPU);
                    Tensor cur_contig = ensure_gpu_contiguous(current);
                    Tensor div_contig = ensure_gpu_contiguous(divisor);

                    Tensor mean_result(current.shape(), out_dtype, Device::GPU);
                    auto *mean_src =
                        as_cuda_buffer_provider(cur_contig.storage().get());
                    auto *mean_div =
                        as_cuda_buffer_provider(div_contig.storage().get());
                    auto *mean_dst =
                        as_cuda_buffer_provider(mean_result.storage().get());

                    launch_binary_elementwise(BinaryOpKind::Div,
                                              mean_src->device_ptr(),
                                              mean_div->device_ptr(),
                                              mean_dst->device_ptr(),
                                              current.size(), elem_size,
                                              stream);
                    CudaExecutionStream::instance().increment_batch();
                    return mean_result;
                }

                CudaExecutionStream::instance().increment_batch();
                return current;
            }
        }

        // Handle Mean: divide result by total reduced element count
        if (op_type_ == ops::OpType::Mean) {
            size_t count = 1;
            for (int ax : norm_axes)
                count *= in_c.shape()[ax];
            Tensor divisor = Tensor::full(result.shape(),
                                          static_cast<double>(count),
                                          result.dtype(), Device::GPU);
            Tensor res_contig = ensure_gpu_contiguous(result);
            Tensor div_contig = ensure_gpu_contiguous(divisor);

            Tensor mean_result(result.shape(), out_dtype, Device::GPU);
            auto *mean_src =
                as_cuda_buffer_provider(res_contig.storage().get());
            auto *mean_div =
                as_cuda_buffer_provider(div_contig.storage().get());
            auto *mean_dst =
                as_cuda_buffer_provider(mean_result.storage().get());

            launch_binary_elementwise(BinaryOpKind::Div,
                                      mean_src->device_ptr(),
                                      mean_div->device_ptr(),
                                      mean_dst->device_ptr(),
                                      result.size(), elem_size, stream);
            CudaExecutionStream::instance().increment_batch();
            return mean_result;
        }

        CudaExecutionStream::instance().increment_batch();
        return result;
#else
        (void)input;
        (void)axes;
        (void)keep_dims;
        throw DeviceError("CUDA support not compiled");
#endif
    }
};

// ============================================================================
// CudaArgReduceOperation (ArgMax / ArgMin)
// ============================================================================

class CudaArgReduceOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string op_name_;
    bool is_max_;

  public:
    CudaArgReduceOperation(ops::OpType op_type, std::string op_name,
                           bool is_max)
        : op_type_(op_type), op_name_(std::move(op_name)), is_max_(is_max) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    bool supports_binary(const Tensor & /*lhs*/,
                         const Tensor & /*rhs*/) const override {
        return false;
    }

    Tensor execute_binary(const Tensor & /*lhs*/,
                          const Tensor & /*rhs*/) const override {
        throw RuntimeError::internal(
            "execute_binary called on argreduce operation");
    }

    Tensor execute_unary(const Tensor &input) const override {
        (void)input;
        throw RuntimeError::internal(
            "execute_unary called on argreduce operation");
    }

    Tensor execute_reduction(const Tensor &input,
                             const std::vector<int> &axes,
                             bool keep_dims) const override {
#ifdef AXIOM_CUDA_SUPPORT
        Tensor in_c = ensure_gpu_contiguous(input);

        // ArgMax/ArgMin operates on a single axis
        int ax = axes.empty() ? -1 : axes[0];

        // Full reduction (all elements)
        if (ax == -1 || axes.size() > 1) {
            Tensor flat = in_c.flatten();
            flat = ensure_gpu_contiguous(flat);

            Shape output_shape = keep_dims ? Shape(in_c.ndim(), 1) : Shape{1};
            Tensor result(output_shape, DType::Int64, Device::GPU);

            auto *src_buf = as_cuda_buffer_provider(flat.storage().get());
            auto *dst_buf = as_cuda_buffer_provider(result.storage().get());
            if (!src_buf || !dst_buf) {
                throw DeviceError(
                    "CudaArgReduceOperation: storage is not CUDA-backed");
            }

            auto stream =
                static_cast<cudaStream_t>(CudaContext::instance().stream());
            size_t elem_size = dtype_size(flat.dtype());

            // Two-pass CUB: query temp, then execute
            size_t temp_bytes = 0;
            launch_full_argreduce(is_max_, src_buf->device_ptr(),
                                  dst_buf->device_ptr(), flat.size(),
                                  elem_size, nullptr, temp_bytes, stream);

            void *temp = nullptr;
            cudaMalloc(&temp, temp_bytes);
            launch_full_argreduce(is_max_, src_buf->device_ptr(),
                                  dst_buf->device_ptr(), flat.size(),
                                  elem_size, temp, temp_bytes, stream);
            cudaFree(temp);

            CudaExecutionStream::instance().increment_batch();
            return result;
        }

        // Single-axis argreduce
        if (ax < 0) ax += static_cast<int>(in_c.ndim());

        Shape output_shape;
        for (size_t i = 0; i < in_c.ndim(); ++i) {
            if (static_cast<int>(i) == ax) {
                if (keep_dims) output_shape.push_back(1);
            } else {
                output_shape.push_back(in_c.shape()[i]);
            }
        }
        if (output_shape.empty()) output_shape.push_back(1);

        Tensor result(output_shape, DType::Int64, Device::GPU);
        if (result.size() == 0) return result;

        size_t outer = 1;
        for (int i = 0; i < ax; ++i)
            outer *= in_c.shape()[i];
        size_t axis_len = in_c.shape()[ax];
        size_t inner = 1;
        for (size_t i = ax + 1; i < in_c.ndim(); ++i)
            inner *= in_c.shape()[i];

        auto *src_buf = as_cuda_buffer_provider(in_c.storage().get());
        auto *dst_buf = as_cuda_buffer_provider(result.storage().get());
        if (!src_buf || !dst_buf) {
            throw DeviceError(
                "CudaArgReduceOperation: storage is not CUDA-backed");
        }

        auto stream =
            static_cast<cudaStream_t>(CudaContext::instance().stream());

        launch_axis_argreduce(is_max_, src_buf->device_ptr(),
                              dst_buf->device_ptr(), outer, axis_len, inner,
                              dtype_size(in_c.dtype()), stream);

        CudaExecutionStream::instance().increment_batch();
        return result;
#else
        (void)input;
        (void)axes;
        (void)keep_dims;
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

static void register_unary_op(ops::OpType op_type,
                               const std::string &name) {
    ops::OperationRegistry::register_operation(
        op_type, Device::GPU,
        std::make_unique<CudaUnaryOperation>(op_type, name));
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

    // Unary math
    register_unary_op(ops::OpType::Negate, "negate");
    register_unary_op(ops::OpType::Abs, "abs");
    register_unary_op(ops::OpType::Sqrt, "sqrt");
    register_unary_op(ops::OpType::Exp, "exp");
    register_unary_op(ops::OpType::Log, "log");
    register_unary_op(ops::OpType::Sin, "sin");
    register_unary_op(ops::OpType::Cos, "cos");
    register_unary_op(ops::OpType::Tan, "tan");
    register_unary_op(ops::OpType::Tanh, "tanh");
    register_unary_op(ops::OpType::Erf, "erf");

    // Rounding / algebraic
    register_unary_op(ops::OpType::Sign, "sign");
    register_unary_op(ops::OpType::Floor, "floor");
    register_unary_op(ops::OpType::Ceil, "ceil");
    register_unary_op(ops::OpType::Trunc, "trunc");
    register_unary_op(ops::OpType::Round, "round");
    register_unary_op(ops::OpType::Reciprocal, "reciprocal");
    register_unary_op(ops::OpType::Square, "square");
    register_unary_op(ops::OpType::Cbrt, "cbrt");

    // Testing
    register_unary_op(ops::OpType::IsNaN, "isnan");
    register_unary_op(ops::OpType::IsInf, "isinf");
    register_unary_op(ops::OpType::IsFinite, "isfinite");

    // Activations
    register_unary_op(ops::OpType::ReLU, "relu");
    register_unary_op(ops::OpType::LeakyReLU, "leaky_relu");
    register_unary_op(ops::OpType::Sigmoid, "sigmoid");
    register_unary_op(ops::OpType::SiLU, "silu");
    register_unary_op(ops::OpType::GELU, "gelu");

    // Reductions
    auto register_reduce_op = [](ops::OpType op_type, const std::string &name) {
        ops::OperationRegistry::register_operation(
            op_type, Device::GPU,
            std::make_unique<CudaReductionOperation>(op_type, name));
    };

    register_reduce_op(ops::OpType::Sum, "sum");
    register_reduce_op(ops::OpType::Mean, "mean");
    register_reduce_op(ops::OpType::Max, "max");
    register_reduce_op(ops::OpType::Min, "min");
    register_reduce_op(ops::OpType::Prod, "prod");
    register_reduce_op(ops::OpType::Any, "any");
    register_reduce_op(ops::OpType::All, "all");

    // ArgMax / ArgMin
    ops::OperationRegistry::register_operation(
        ops::OpType::ArgMax, Device::GPU,
        std::make_unique<CudaArgReduceOperation>(ops::OpType::ArgMax,
                                                  "argmax", true));
    ops::OperationRegistry::register_operation(
        ops::OpType::ArgMin, Device::GPU,
        std::make_unique<CudaArgReduceOperation>(ops::OpType::ArgMin,
                                                  "argmin", false));

    register_cublas_operations();
}

} // namespace cuda
} // namespace backends
} // namespace axiom
