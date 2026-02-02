#include "cpu_operations.hpp"

#include <cstddef>

#include "axiom/error.hpp"
#include "axiom/operations.hpp"
#include "axiom/parallel.hpp"
#include "axiom/shape.hpp"
#include "axiom/tensor.hpp"

// SIMD optimization headers
#include "simd/simd_dispatch.hpp"
#include "vdsp.hpp"

// BLAS backend abstraction
#include "blas/blas_backend.hpp"

namespace axiom {
namespace backends {
namespace cpu {

// ============================================================================
// Simple implementation focusing on basic types
// ============================================================================

template <typename Func>
Tensor CPUBinaryOperation<Func>::execute_binary(const Tensor &lhs,
                                                const Tensor &rhs) const {
    // Ensure tensors are on CPU
    if (lhs.device() != Device::CPU || rhs.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU binary operations");
    }

    // Compute broadcast information
    auto broadcast_info = ops::compute_broadcast_info(lhs.shape(), rhs.shape());

    // Determine result type
    DType result_dtype = ops::result_type(lhs, rhs);

    // Handle comparison and logical operations - they always return bool
    if (op_type_ == ops::OpType::Equal || op_type_ == ops::OpType::NotEqual ||
        op_type_ == ops::OpType::Less || op_type_ == ops::OpType::LessEqual ||
        op_type_ == ops::OpType::Greater ||
        op_type_ == ops::OpType::GreaterEqual ||
        op_type_ == ops::OpType::LogicalAnd ||
        op_type_ == ops::OpType::LogicalOr ||
        op_type_ == ops::OpType::LogicalXor) {
        result_dtype = DType::Bool;
    }

    // Check for bitwise operations - they only work on integer types
    bool is_bitwise_op = (op_type_ == ops::OpType::BitwiseAnd ||
                          op_type_ == ops::OpType::BitwiseOr ||
                          op_type_ == ops::OpType::BitwiseXor ||
                          op_type_ == ops::OpType::LeftShift ||
                          op_type_ == ops::OpType::RightShift);
    if (is_bitwise_op) {
        if (result_dtype == DType::Float16 || result_dtype == DType::Float32 ||
            result_dtype == DType::Float64 ||
            result_dtype == DType::Complex64 ||
            result_dtype == DType::Complex128) {
            throw TypeError(
                "Bitwise operations only support integer types, got " +
                dtype_name(result_dtype));
        }
    }

    // Create result tensor
    Tensor result(broadcast_info.result_shape, result_dtype, Device::CPU);

#define DISPATCH_CPU_BINARY_OP(TYPE_ENUM, TYPE)                                \
    case TYPE_ENUM:                                                            \
        execute_binary_typed<TYPE>(lhs, rhs, result);                          \
        break;

    switch (result_dtype) {
    case DType::Float32:
    case DType::Float64:
    case DType::Float16:
        // Skip float types for bitwise ops - we throw above
        if (!is_bitwise_op) {
            if (result_dtype == DType::Float32)
                execute_binary_typed<float>(lhs, rhs, result);
            else if (result_dtype == DType::Float64)
                execute_binary_typed<double>(lhs, rhs, result);
            else
                execute_binary_typed<float16_t>(lhs, rhs, result);
        }
        break;
        DISPATCH_CPU_BINARY_OP(DType::Int8, int8_t)
        DISPATCH_CPU_BINARY_OP(DType::Int16, int16_t)
        DISPATCH_CPU_BINARY_OP(DType::Int32, int32_t)
        DISPATCH_CPU_BINARY_OP(DType::Int64, int64_t)
        DISPATCH_CPU_BINARY_OP(DType::UInt8, uint8_t)
        DISPATCH_CPU_BINARY_OP(DType::UInt16, uint16_t)
        DISPATCH_CPU_BINARY_OP(DType::UInt32, uint32_t)
        DISPATCH_CPU_BINARY_OP(DType::UInt64, uint64_t)
    case DType::Bool:
        execute_binary_typed<bool>(lhs, rhs, result);
        break;
    case DType::Complex64:
    case DType::Complex128:
        // Complex types are handled separately - only arithmetic ops are
        // allowed The operation registry and operations.cpp should enforce this
        // before we get here
        throw TypeError::unsupported_dtype(
            dtype_name(result_dtype),
            "CPU binary operations (complex types require special handling)");
    default:
        throw TypeError::unsupported_dtype(dtype_name(result_dtype),
                                           "CPU binary operations");
    }
#undef DISPATCH_CPU_BINARY_OP

    return result;
}

template <typename Func>
template <typename T>
void CPUBinaryOperation<Func>::execute_binary_typed(const Tensor &lhs,
                                                    const Tensor &rhs,
                                                    Tensor &result) const {
    auto broadcast_info = ops::compute_broadcast_info(lhs.shape(), rhs.shape());

    // Use strided iteration if broadcast is needed OR if either tensor is
    // non-contiguous (including negative strides from flip operations)
    if (broadcast_info.needs_broadcast || !lhs.is_contiguous() ||
        !rhs.is_contiguous()) {
        execute_binary_broadcast<T>(lhs, rhs, result, broadcast_info);
    } else {
        execute_binary_same_shape<T>(lhs, rhs, result);
    }
}

template <typename Func>
template <typename T>
void CPUBinaryOperation<Func>::execute_binary_same_shape(const Tensor &lhs,
                                                         const Tensor &rhs,
                                                         Tensor &result) const {
    size_t total_elements = result.size();

    // For comparison operations with bool output
    if constexpr (std::is_same_v<T, bool>) {
        // Convert inputs to Float32 for comparison
        Tensor lhs_float = lhs.astype(DType::Float32);
        Tensor rhs_float = rhs.astype(DType::Float32);

        const float *lhs_data = lhs_float.template typed_data<float>();
        const float *rhs_data = rhs_float.template typed_data<float>();
        bool *result_data = result.template typed_data<bool>();

        for (size_t i = 0; i < total_elements; ++i) {
            result_data[i] = func_(lhs_data[i], rhs_data[i]);
        }
    } else {
        // Convert tensors to result type
        Tensor lhs_converted = lhs.astype(result.dtype());
        Tensor rhs_converted = rhs.astype(result.dtype());

        const T *lhs_data = lhs_converted.template typed_data<T>();
        const T *rhs_data = rhs_converted.template typed_data<T>();
        T *result_data = result.template typed_data<T>();

        // Tier 1: Try Accelerate (vDSP) for contiguous float32/float64
        // Accelerate is already internally parallelized and highly optimized
#ifdef AXIOM_USE_ACCELERATE
        if constexpr (std::is_same_v<T, float>) {
            if constexpr (std::is_same_v<Func, AddFunc>) {
                accelerate::vadd_f32(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SubtractFunc>) {
                accelerate::vsub_f32(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, MultiplyFunc>) {
                accelerate::vmul_f32(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, DivideFunc>) {
                accelerate::vdiv_f32(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, PowerFunc>) {
                accelerate::vpow_f32(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Atan2Func>) {
                accelerate::vatan2_f32(lhs_data, rhs_data, result_data,
                                       total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, HypotFunc>) {
                accelerate::vhypot_f32(lhs_data, rhs_data, result_data,
                                       total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ModuloFunc>) {
                accelerate::vfmod_f32(lhs_data, rhs_data, result_data,
                                      total_elements);
                return;
            }
        }
        if constexpr (std::is_same_v<T, double>) {
            if constexpr (std::is_same_v<Func, AddFunc>) {
                accelerate::vadd_f64(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SubtractFunc>) {
                accelerate::vsub_f64(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, MultiplyFunc>) {
                accelerate::vmul_f64(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, DivideFunc>) {
                accelerate::vdiv_f64(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, PowerFunc>) {
                accelerate::vpow_f64(lhs_data, rhs_data, result_data,
                                     total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Atan2Func>) {
                accelerate::vatan2_f64(lhs_data, rhs_data, result_data,
                                       total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, HypotFunc>) {
                accelerate::vhypot_f64(lhs_data, rhs_data, result_data,
                                       total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ModuloFunc>) {
                accelerate::vfmod_f64(lhs_data, rhs_data, result_data,
                                      total_elements);
                return;
            }
        }
#endif // AXIOM_USE_ACCELERATE

        // Tier 2: XSIMD for vectorizable types (all platforms)
        // Uses runtime dispatch for optimal SIMD on each CPU
        if constexpr (simd::has_support<T>) {
            // Basic arithmetic - runtime dispatched for best performance
            if constexpr (std::is_same_v<Func, AddFunc>) {
                simd::dispatch_binary_add(lhs_data, rhs_data, result_data,
                                          total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SubtractFunc>) {
                simd::dispatch_binary_sub(lhs_data, rhs_data, result_data,
                                          total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, MultiplyFunc>) {
                simd::dispatch_binary_mul(lhs_data, rhs_data, result_data,
                                          total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, DivideFunc>) {
                simd::dispatch_binary_div(lhs_data, rhs_data, result_data,
                                          total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, MaximumFunc>) {
                simd::dispatch_binary_max(lhs_data, rhs_data, result_data,
                                          total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, MinimumFunc>) {
                simd::dispatch_binary_min(lhs_data, rhs_data, result_data,
                                          total_elements);
                return;
            }
            // Math functions (floating point only) - runtime dispatched
            if constexpr (std::is_floating_point_v<T>) {
                if constexpr (std::is_same_v<Func, PowerFunc>) {
                    simd::dispatch_binary_pow(lhs_data, rhs_data, result_data,
                                              total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, Atan2Func>) {
                    simd::dispatch_binary_atan2(lhs_data, rhs_data, result_data,
                                                total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, HypotFunc>) {
                    simd::dispatch_binary_hypot(lhs_data, rhs_data, result_data,
                                                total_elements);
                    return;
                }
            }
        }

        // Tier 3: Scalar fallback (with OpenMP for large tensors)
#ifdef AXIOM_USE_OPENMP
        if (parallel::should_parallelize(total_elements)) {
            ptrdiff_t n = static_cast<ptrdiff_t>(total_elements);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) {
                result_data[i] = func_(lhs_data[i], rhs_data[i]);
            }
        } else
#endif
        {
            for (size_t i = 0; i < total_elements; ++i) {
                result_data[i] = func_(lhs_data[i], rhs_data[i]);
            }
        }
    }
}

template <typename Func>
template <typename T>
void CPUBinaryOperation<Func>::execute_binary_broadcast(
    const Tensor &lhs, const Tensor &rhs, Tensor &result,
    const ops::BroadcastInfo &broadcast_info) const {
    const Shape &result_shape = broadcast_info.result_shape;

    // For comparison operations with bool output
    if constexpr (std::is_same_v<T, bool>) {
        Tensor lhs_float = lhs.astype(DType::Float32);
        Tensor rhs_float = rhs.astype(DType::Float32);

        const float *lhs_data = lhs_float.template typed_data<float>();
        const float *rhs_data = rhs_float.template typed_data<float>();
        bool *result_data = result.template typed_data<bool>();

        execute_broadcast_loop<float, bool>(
            lhs_data, rhs_data, result_data, lhs.shape(), rhs.shape(),
            result_shape, lhs_float.strides(), rhs_float.strides());
    } else {
        Tensor lhs_converted = lhs.astype(result.dtype());
        Tensor rhs_converted = rhs.astype(result.dtype());

        const T *lhs_data = lhs_converted.template typed_data<T>();
        const T *rhs_data = rhs_converted.template typed_data<T>();
        T *result_data = result.template typed_data<T>();

        execute_broadcast_loop<T, T>(
            lhs_data, rhs_data, result_data, lhs.shape(), rhs.shape(),
            result_shape, lhs_converted.strides(), rhs_converted.strides());
    }
}

template <typename Func>
template <typename InputT, typename OutputT>
void CPUBinaryOperation<Func>::execute_broadcast_loop(
    const InputT *lhs_data, const InputT *rhs_data, OutputT *result_data,
    const Shape &lhs_shape, const Shape &rhs_shape, const Shape &result_shape,
    const Strides &lhs_strides_in, const Strides &rhs_strides_in) const {
    size_t total_elements = ShapeUtils::size(result_shape);
    size_t ndim = result_shape.size();

    // Prepare broadcasted strides
    Strides lhs_bcast_strides(ndim, 0);
    Strides rhs_bcast_strides(ndim, 0);

    int lhs_dim_offset = ndim - lhs_shape.size();
    for (size_t i = 0; i < lhs_shape.size(); ++i) {
        if (lhs_shape[i] != 1) {
            lhs_bcast_strides[i + lhs_dim_offset] = lhs_strides_in[i];
        }
    }

    int rhs_dim_offset = ndim - rhs_shape.size();
    for (size_t i = 0; i < rhs_shape.size(); ++i) {
        if (rhs_shape[i] != 1) {
            rhs_bcast_strides[i + rhs_dim_offset] = rhs_strides_in[i];
        }
    }

    // OpenMP parallel path for large broadcast operations
#ifdef AXIOM_USE_OPENMP
    if (parallel::should_parallelize(total_elements)) {
        // MSVC OpenMP requires signed loop index
        ptrdiff_t n = static_cast<ptrdiff_t>(total_elements);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            // Compute coordinates from flat index
            size_t remaining = static_cast<size_t>(i);
            int64_t lhs_byte_offset = 0;
            int64_t rhs_byte_offset = 0;

            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                size_t coord = remaining % result_shape[d];
                remaining /= result_shape[d];
                lhs_byte_offset +=
                    static_cast<int64_t>(coord) * lhs_bcast_strides[d];
                rhs_byte_offset +=
                    static_cast<int64_t>(coord) * rhs_bcast_strides[d];
            }

            const auto &lhs_val = *reinterpret_cast<const InputT *>(
                reinterpret_cast<const uint8_t *>(lhs_data) + lhs_byte_offset);
            const auto &rhs_val = *reinterpret_cast<const InputT *>(
                reinterpret_cast<const uint8_t *>(rhs_data) + rhs_byte_offset);

            result_data[i] = func_(lhs_val, rhs_val);
        }
        return;
    }
#endif

    // Sequential fallback with coordinate tracking
    std::vector<size_t> result_coords(ndim, 0);

    for (size_t i = 0; i < total_elements; ++i) {
        // Use signed arithmetic for negative stride support
        int64_t lhs_byte_offset = 0;
        int64_t rhs_byte_offset = 0;

        for (size_t j = 0; j < ndim; ++j) {
            lhs_byte_offset +=
                static_cast<int64_t>(result_coords[j]) * lhs_bcast_strides[j];
            rhs_byte_offset +=
                static_cast<int64_t>(result_coords[j]) * rhs_bcast_strides[j];
        }

        const auto &lhs_val = *reinterpret_cast<const InputT *>(
            reinterpret_cast<const uint8_t *>(lhs_data) + lhs_byte_offset);
        const auto &rhs_val = *reinterpret_cast<const InputT *>(
            reinterpret_cast<const uint8_t *>(rhs_data) + rhs_byte_offset);

        result_data[i] = func_(lhs_val, rhs_val);

        // Increment coordinates
        for (int j = ndim - 1; j >= 0; --j) {
            if (++result_coords[j] < result_shape[j]) {
                break;
            }
            result_coords[j] = 0;
        }
    }
}

template <typename Func>
void CPUBinaryOperation<Func>::execute_binary_inplace(Tensor &lhs,
                                                      const Tensor &rhs) const {
    // In-place operations require lhs to be writeable
    if (!lhs.flags().writeable) {
        throw MemoryError(
            "Cannot perform in-place operation on a non-writeable tensor");
    }

    // Check for type safety. In-place ops do not promote the lhs tensor.
    DType promoted_dtype = ops::promote_types(lhs.dtype(), rhs.dtype());
    if (promoted_dtype != lhs.dtype()) {
        throw TypeError(
            "In-place operation would require unsafe type casting from " +
            dtype_name(lhs.dtype()) + " to " + dtype_name(promoted_dtype));
    }

    // Check for broadcast safety. In-place ops cannot change the lhs shape.
    if (!ops::are_broadcastable(lhs.shape(), rhs.shape()) ||
        ops::compute_broadcast_info(lhs.shape(), rhs.shape()).result_shape !=
            lhs.shape()) {
        throw ShapeError(
            "In-place operation with broadcasting cannot change tensor shape");
    }

// Dispatch to the typed implementation
#define DISPATCH_CPU_INPLACE_OP(TYPE_ENUM, TYPE)                               \
    case TYPE_ENUM:                                                            \
        execute_inplace_typed<TYPE>(lhs, rhs);                                 \
        break;

    switch (lhs.dtype()) {
        DISPATCH_CPU_INPLACE_OP(DType::Float32, float)
        DISPATCH_CPU_INPLACE_OP(DType::Float64, double)
        DISPATCH_CPU_INPLACE_OP(DType::Float16, float16_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int8, int8_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int16, int16_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int32, int32_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int64, int64_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt8, uint8_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt16, uint16_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt32, uint32_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt64, uint64_t)
        DISPATCH_CPU_INPLACE_OP(DType::Bool, bool)
    default:
        throw TypeError::unsupported_dtype(dtype_name(lhs.dtype()),
                                           "CPU in-place operations");
    }
#undef DISPATCH_CPU_INPLACE_OP
}

template <typename Func>
template <typename T>
void CPUBinaryOperation<Func>::execute_inplace_typed(Tensor &lhs,
                                                     const Tensor &rhs) const {
    if (lhs.shape() == rhs.shape() && lhs.is_contiguous() &&
        rhs.is_contiguous()) {
        T *lhs_data = lhs.template typed_data<T>();
        const T *rhs_data = rhs.template typed_data<T>();
        size_t n_elements = lhs.size();

#ifdef AXIOM_USE_OPENMP
        if (parallel::should_parallelize(n_elements)) {
            ptrdiff_t n = static_cast<ptrdiff_t>(n_elements);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) {
                lhs_data[i] = func_(lhs_data[i], rhs_data[i]);
            }
        } else
#endif
        {
            for (size_t i = 0; i < n_elements; ++i) {
                lhs_data[i] = func_(lhs_data[i], rhs_data[i]);
            }
        }
    } else {
        execute_inplace_broadcast<T>(lhs, rhs);
    }
}

template <typename Func>
template <typename T>
void CPUBinaryOperation<Func>::execute_inplace_broadcast(
    Tensor &lhs, const Tensor &rhs) const {
    const Shape &lhs_shape = lhs.shape();
    const Shape &rhs_shape = rhs.shape();
    size_t lhs_ndim = lhs_shape.size();
    size_t rhs_ndim = rhs_shape.size();

    // Prepare broadcasted strides for rhs (lhs strides remain as-is)
    Strides rhs_bcast_strides(lhs_ndim, 0);
    size_t rhs_dim_offset = lhs_ndim - rhs_ndim;

    for (size_t i = 0; i < rhs_ndim; ++i) {
        if (rhs_shape[i] != 1) {
            rhs_bcast_strides[i + rhs_dim_offset] = rhs.strides()[i];
        }
    }

    T *lhs_data = lhs.template typed_data<T>();
    const T *rhs_data = rhs.template typed_data<T>();
    const Strides &lhs_strides = lhs.strides();
    size_t total_elements = lhs.size();

    std::vector<size_t> coords(lhs_ndim, 0);

    for (size_t i = 0; i < total_elements; ++i) {
        // Use signed arithmetic for negative stride support
        int64_t lhs_byte_offset = 0;
        int64_t rhs_byte_offset = 0;

        for (size_t j = 0; j < lhs_ndim; ++j) {
            lhs_byte_offset += static_cast<int64_t>(coords[j]) * lhs_strides[j];
            rhs_byte_offset +=
                static_cast<int64_t>(coords[j]) * rhs_bcast_strides[j];
        }

        T &lhs_val = *reinterpret_cast<T *>(
            reinterpret_cast<uint8_t *>(lhs_data) + lhs_byte_offset);
        const T &rhs_val = *reinterpret_cast<const T *>(
            reinterpret_cast<const uint8_t *>(rhs_data) + rhs_byte_offset);

        lhs_val = func_(lhs_val, rhs_val);

        // Increment coordinates
        for (int j = static_cast<int>(lhs_ndim) - 1; j >= 0; --j) {
            if (++coords[j] < lhs_shape[j]) {
                break;
            }
            coords[j] = 0;
        }
    }
}

// ============================================================================
// CPU Complex Binary Operation Implementation
// ============================================================================

Tensor CPUComplexBinaryOperation::execute_binary(const Tensor &lhs,
                                                 const Tensor &rhs) const {
    if (lhs.device() != Device::CPU || rhs.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU complex binary operations");
    }

    auto broadcast_info = ops::compute_broadcast_info(lhs.shape(), rhs.shape());
    DType result_dtype = ops::result_type(lhs, rhs);

    if (!is_complex_dtype(result_dtype)) {
        throw TypeError(
            "CPUComplexBinaryOperation requires complex input types");
    }

    Tensor result(broadcast_info.result_shape, result_dtype, Device::CPU);

    if (result_dtype == DType::Complex64) {
        execute_complex_typed<complex64_t>(lhs, rhs, result);
    } else {
        execute_complex_typed<complex128_t>(lhs, rhs, result);
    }

    return result;
}

template <typename T>
void CPUComplexBinaryOperation::execute_complex_typed(const Tensor &lhs,
                                                      const Tensor &rhs,
                                                      Tensor &result) const {
    Tensor lhs_conv = lhs.astype(result.dtype());
    Tensor rhs_conv = rhs.astype(result.dtype());

    const T *lhs_data = lhs_conv.typed_data<T>();
    const T *rhs_data = rhs_conv.typed_data<T>();
    T *result_data = result.typed_data<T>();

    auto broadcast_info = ops::compute_broadcast_info(lhs.shape(), rhs.shape());
    const Shape &result_shape = broadcast_info.result_shape;
    size_t total_elements = ShapeUtils::size(result_shape);
    size_t ndim = result_shape.size();

    // Prepare broadcasted strides
    Strides lhs_bcast_strides(ndim, 0);
    Strides rhs_bcast_strides(ndim, 0);

    size_t lhs_dim_offset = ndim - lhs.shape().size();
    for (size_t i = 0; i < lhs.shape().size(); ++i) {
        if (lhs.shape()[i] != 1) {
            lhs_bcast_strides[i + lhs_dim_offset] = lhs_conv.strides()[i];
        }
    }

    size_t rhs_dim_offset = ndim - rhs.shape().size();
    for (size_t i = 0; i < rhs.shape().size(); ++i) {
        if (rhs.shape()[i] != 1) {
            rhs_bcast_strides[i + rhs_dim_offset] = rhs_conv.strides()[i];
        }
    }

    std::vector<size_t> result_coords(ndim, 0);

    for (size_t i = 0; i < total_elements; ++i) {
        // Use signed arithmetic for negative stride support
        int64_t lhs_byte_offset = 0;
        int64_t rhs_byte_offset = 0;

        for (size_t j = 0; j < ndim; ++j) {
            lhs_byte_offset +=
                static_cast<int64_t>(result_coords[j]) * lhs_bcast_strides[j];
            rhs_byte_offset +=
                static_cast<int64_t>(result_coords[j]) * rhs_bcast_strides[j];
        }

        const T &lhs_val = *reinterpret_cast<const T *>(
            reinterpret_cast<const uint8_t *>(lhs_data) + lhs_byte_offset);
        const T &rhs_val = *reinterpret_cast<const T *>(
            reinterpret_cast<const uint8_t *>(rhs_data) + rhs_byte_offset);

        // Apply operation based on op_type_
        switch (op_type_) {
        case ops::OpType::Add:
            result_data[i] = lhs_val + rhs_val;
            break;
        case ops::OpType::Subtract:
            result_data[i] = lhs_val - rhs_val;
            break;
        case ops::OpType::Multiply:
            result_data[i] = lhs_val * rhs_val;
            break;
        case ops::OpType::Divide:
            result_data[i] = lhs_val / rhs_val;
            break;
        default:
            throw TypeError("Unsupported operation for complex types: " +
                            name());
        }

        // Increment coordinates
        for (int j = ndim - 1; j >= 0; --j) {
            if (++result_coords[j] < result_shape[j]) {
                break;
            }
            result_coords[j] = 0;
        }
    }
}

// Explicit template instantiations
template void CPUComplexBinaryOperation::execute_complex_typed<complex64_t>(
    const Tensor &, const Tensor &, Tensor &) const;
template void CPUComplexBinaryOperation::execute_complex_typed<complex128_t>(
    const Tensor &, const Tensor &, Tensor &) const;

// ============================================================================
// CPU Unary Operation Implementation
// ============================================================================

template <typename Func>
Tensor CPUUnaryOperation<Func>::execute_unary(const Tensor &input) const {
    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU unary operations");
    }

    // Check if this is an abs operation - complex abs returns real type
    bool is_abs_op = (op_type_ == ops::OpType::Abs);
    bool is_logical_not = (op_type_ == ops::OpType::LogicalNot);
    bool is_element_test =
        (op_type_ == ops::OpType::IsNaN || op_type_ == ops::OpType::IsInf ||
         op_type_ == ops::OpType::IsFinite);

    // Unary ops usually return the same dtype as input, except:
    // - abs on complex returns the corresponding real type
    // - logical_not always returns Bool
    // - isnan/isinf/isfinite always return Bool
    DType result_dtype = input.dtype();
    if (is_abs_op && is_complex_dtype(input.dtype())) {
        result_dtype = (input.dtype() == DType::Complex64) ? DType::Float32
                                                           : DType::Float64;
    }
    if (is_logical_not || is_element_test) {
        result_dtype = DType::Bool;
    }

    Tensor result(input.shape(), result_dtype, Device::CPU);

    // Special handling for logical_not - always outputs Bool
    if (is_logical_not) {
        bool *out_data = result.typed_data<bool>();
        // Handle different input dtypes
        switch (input.dtype()) {
#define DISPATCH_LOGICAL_NOT(DTYPE, CTYPE)                                     \
    case DTYPE: {                                                              \
        const CTYPE *in_data = input.typed_data<CTYPE>();                      \
        for (size_t i = 0; i < input.size(); ++i) {                            \
            out_data[i] = !static_cast<bool>(in_data[i]);                      \
        }                                                                      \
        break;                                                                 \
    }
            DISPATCH_LOGICAL_NOT(DType::Bool, bool)
            DISPATCH_LOGICAL_NOT(DType::Int8, int8_t)
            DISPATCH_LOGICAL_NOT(DType::Int16, int16_t)
            DISPATCH_LOGICAL_NOT(DType::Int32, int32_t)
            DISPATCH_LOGICAL_NOT(DType::Int64, int64_t)
            DISPATCH_LOGICAL_NOT(DType::UInt8, uint8_t)
            DISPATCH_LOGICAL_NOT(DType::UInt16, uint16_t)
            DISPATCH_LOGICAL_NOT(DType::UInt32, uint32_t)
            DISPATCH_LOGICAL_NOT(DType::UInt64, uint64_t)
            DISPATCH_LOGICAL_NOT(DType::Float16, float16_t)
            DISPATCH_LOGICAL_NOT(DType::Float32, float)
            DISPATCH_LOGICAL_NOT(DType::Float64, double)
#undef DISPATCH_LOGICAL_NOT
        default:
            throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                               "logical_not");
        }
        return result;
    }

    // Special handling for element tests (isnan, isinf, isfinite) - always
    // outputs Bool
    if (is_element_test) {
        bool *out_data = result.typed_data<bool>();
        switch (input.dtype()) {
#define DISPATCH_ELEMENT_TEST(DTYPE, CTYPE)                                    \
    case DTYPE: {                                                              \
        const CTYPE *in_data = input.typed_data<CTYPE>();                      \
        for (size_t i = 0; i < input.size(); ++i) {                            \
            out_data[i] = func_(in_data[i]);                                   \
        }                                                                      \
        break;                                                                 \
    }
            DISPATCH_ELEMENT_TEST(DType::Float16, float16_t)
            DISPATCH_ELEMENT_TEST(DType::Float32, float)
            DISPATCH_ELEMENT_TEST(DType::Float64, double)
            // For integer types, isnan/isinf return false, isfinite returns
            // true
            DISPATCH_ELEMENT_TEST(DType::Bool, bool)
            DISPATCH_ELEMENT_TEST(DType::Int8, int8_t)
            DISPATCH_ELEMENT_TEST(DType::Int16, int16_t)
            DISPATCH_ELEMENT_TEST(DType::Int32, int32_t)
            DISPATCH_ELEMENT_TEST(DType::Int64, int64_t)
            DISPATCH_ELEMENT_TEST(DType::UInt8, uint8_t)
            DISPATCH_ELEMENT_TEST(DType::UInt16, uint16_t)
            DISPATCH_ELEMENT_TEST(DType::UInt32, uint32_t)
            DISPATCH_ELEMENT_TEST(DType::UInt64, uint64_t)
#undef DISPATCH_ELEMENT_TEST
        default:
            throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                               name());
        }
        return result;
    }

    // Special handling for complex abs - returns magnitude as float
    if (is_abs_op && input.dtype() == DType::Complex64) {
        const complex64_t *in_data = input.typed_data<complex64_t>();
        float *out_data = result.typed_data<float>();
        for (size_t i = 0; i < input.size(); ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
        return result;
    }
    if (is_abs_op && input.dtype() == DType::Complex128) {
        const complex128_t *in_data = input.typed_data<complex128_t>();
        double *out_data = result.typed_data<double>();
        for (size_t i = 0; i < input.size(); ++i) {
            out_data[i] = std::abs(in_data[i]);
        }
        return result;
    }

#define DISPATCH_CPU_UNARY_OP(TYPE_ENUM, TYPE)                                 \
    case TYPE_ENUM:                                                            \
        execute_unary_typed<TYPE>(input, result);                              \
        break;

    switch (result_dtype) {
        DISPATCH_CPU_UNARY_OP(DType::Float32, float)
        DISPATCH_CPU_UNARY_OP(DType::Float64, double)
        DISPATCH_CPU_UNARY_OP(DType::Float16, float16_t)
        DISPATCH_CPU_UNARY_OP(DType::Int8, int8_t)
        DISPATCH_CPU_UNARY_OP(DType::Int16, int16_t)
        DISPATCH_CPU_UNARY_OP(DType::Int32, int32_t)
        DISPATCH_CPU_UNARY_OP(DType::Int64, int64_t)
        DISPATCH_CPU_UNARY_OP(DType::UInt8, uint8_t)
        DISPATCH_CPU_UNARY_OP(DType::UInt16, uint16_t)
        DISPATCH_CPU_UNARY_OP(DType::UInt32, uint32_t)
        DISPATCH_CPU_UNARY_OP(DType::UInt64, uint64_t)
        DISPATCH_CPU_UNARY_OP(DType::Bool, bool)
    case DType::Complex64:
    case DType::Complex128:
        // Complex unary operations should be handled at the higher level in
        // operations.cpp except for abs which is already handled above
        throw TypeError::unsupported_dtype(
            dtype_name(result_dtype), name() + " (use higher-level dispatch)");
    default:
        throw TypeError::unsupported_dtype(dtype_name(result_dtype),
                                           "CPU unary operations");
    }
#undef DISPATCH_CPU_UNARY_OP

    return result;
}

template <typename Func>
template <typename T>
void CPUUnaryOperation<Func>::execute_unary_typed(const Tensor &input,
                                                  Tensor &result) const {
    size_t total_elements = input.size();
    const T *input_data = input.template typed_data<T>();
    T *result_data = result.template typed_data<T>();

    // Tier 1: Try Accelerate (vForce/vDSP) for contiguous float32/float64
    // Accelerate is already internally parallelized and highly optimized
#ifdef AXIOM_USE_ACCELERATE
    if (input.is_contiguous()) {
        if constexpr (std::is_same_v<T, float>) {
            if constexpr (std::is_same_v<Func, ExpFunc>) {
                accelerate::vexp_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, LogFunc>) {
                accelerate::vlog_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SqrtFunc>) {
                accelerate::vsqrt_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SinFunc>) {
                accelerate::vsin_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CosFunc>) {
                accelerate::vcos_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, TanFunc>) {
                accelerate::vtan_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, TanhFunc>) {
                accelerate::vtanh_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AbsFunc>) {
                accelerate::vabs_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, NegateFunc>) {
                accelerate::vneg_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, FloorFunc>) {
                accelerate::vfloor_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CeilFunc>) {
                accelerate::vceil_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, RoundFunc>) {
                accelerate::vround_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, TruncFunc>) {
                accelerate::vtrunc_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ReciprocalFunc>) {
                accelerate::vrecip_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SquareFunc>) {
                accelerate::vsquare_f32(input_data, result_data,
                                        total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ErfFunc>) {
                accelerate::verf_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CbrtFunc>) {
                accelerate::vcbrt_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ReLUFunc>) {
                accelerate::vrelu_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SinhFunc>) {
                accelerate::vsinh_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CoshFunc>) {
                accelerate::vcosh_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AsinFunc>) {
                accelerate::vasin_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AcosFunc>) {
                accelerate::vacos_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AtanFunc>) {
                accelerate::vatan_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Log2Func>) {
                accelerate::vlog2_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Log10Func>) {
                accelerate::vlog10_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Log1pFunc>) {
                accelerate::vlog1p_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Exp2Func>) {
                accelerate::vexp2_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Expm1Func>) {
                accelerate::vexpm1_f32(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, RsqrtFunc>) {
                accelerate::vrsqrt_f32(input_data, result_data, total_elements);
                return;
            }
        }
        if constexpr (std::is_same_v<T, double>) {
            if constexpr (std::is_same_v<Func, ExpFunc>) {
                accelerate::vexp_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, LogFunc>) {
                accelerate::vlog_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SqrtFunc>) {
                accelerate::vsqrt_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SinFunc>) {
                accelerate::vsin_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CosFunc>) {
                accelerate::vcos_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, TanFunc>) {
                accelerate::vtan_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, TanhFunc>) {
                accelerate::vtanh_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AbsFunc>) {
                accelerate::vabs_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, NegateFunc>) {
                accelerate::vneg_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, FloorFunc>) {
                accelerate::vfloor_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CeilFunc>) {
                accelerate::vceil_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, RoundFunc>) {
                accelerate::vround_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, TruncFunc>) {
                accelerate::vtrunc_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ReciprocalFunc>) {
                accelerate::vrecip_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SquareFunc>) {
                accelerate::vsquare_f64(input_data, result_data,
                                        total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ErfFunc>) {
                accelerate::verf_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CbrtFunc>) {
                accelerate::vcbrt_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, ReLUFunc>) {
                accelerate::vrelu_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SinhFunc>) {
                accelerate::vsinh_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, CoshFunc>) {
                accelerate::vcosh_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AsinFunc>) {
                accelerate::vasin_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AcosFunc>) {
                accelerate::vacos_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, AtanFunc>) {
                accelerate::vatan_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Log2Func>) {
                accelerate::vlog2_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Log10Func>) {
                accelerate::vlog10_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Log1pFunc>) {
                accelerate::vlog1p_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Exp2Func>) {
                accelerate::vexp2_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, Expm1Func>) {
                accelerate::vexpm1_f64(input_data, result_data, total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, RsqrtFunc>) {
                accelerate::vrsqrt_f64(input_data, result_data, total_elements);
                return;
            }
        }
    }
#endif // AXIOM_USE_ACCELERATE

    // Tier 2: XSIMD for vectorizable types (all platforms)
    // Uses runtime dispatch for optimal SIMD on each CPU
    if constexpr (simd::has_support<T>) {
        if (input.is_contiguous()) {
            // Basic unary operations - runtime dispatched
            if constexpr (std::is_same_v<Func, AbsFunc>) {
                if constexpr (std::is_signed_v<T>) {
                    simd::dispatch_unary_abs(input_data, result_data,
                                             total_elements);
                    return;
                }
            } else if constexpr (std::is_same_v<Func, NegateFunc>) {
                if constexpr (std::is_signed_v<T>) {
                    simd::dispatch_unary_neg(input_data, result_data,
                                             total_elements);
                    return;
                }
            } else if constexpr (std::is_same_v<Func, SquareFunc>) {
                simd::dispatch_unary_square(input_data, result_data,
                                            total_elements);
                return;
            } else if constexpr (std::is_same_v<Func, SignFunc>) {
                if constexpr (std::is_signed_v<T>) {
                    simd::dispatch_unary_sign(input_data, result_data,
                                              total_elements);
                    return;
                }
            }

            // Activation functions - runtime dispatched
            if constexpr (std::is_signed_v<T>) {
                if constexpr (std::is_same_v<Func, ReLUFunc>) {
                    simd::dispatch_activation_relu(input_data, result_data,
                                                   total_elements);
                    return;
                }
            }

            // Floating point operations - all runtime dispatched
            if constexpr (std::is_floating_point_v<T>) {
                // Activation functions
                if constexpr (std::is_same_v<Func, SigmoidFunc>) {
                    simd::dispatch_activation_sigmoid(input_data, result_data,
                                                      total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, SiLUFunc>) {
                    simd::dispatch_activation_silu(input_data, result_data,
                                                   total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, GELUFunc>) {
                    simd::dispatch_activation_gelu(input_data, result_data,
                                                   total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, ReciprocalFunc>) {
                    simd::dispatch_unary_reciprocal(input_data, result_data,
                                                    total_elements);
                    return;
                }

                // Rounding operations
#ifndef AXIOM_USE_ACCELERATE
                if constexpr (std::is_same_v<Func, FloorFunc>) {
                    simd::dispatch_unary_floor(input_data, result_data,
                                               total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, CeilFunc>) {
                    simd::dispatch_unary_ceil(input_data, result_data,
                                              total_elements);
                    return;
                } else
#endif
                    if constexpr (std::is_same_v<Func, RoundFunc>) {
                    simd::dispatch_unary_round(input_data, result_data,
                                               total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, TruncFunc>) {
                    simd::dispatch_unary_trunc(input_data, result_data,
                                               total_elements);
                    return;
                }

#ifndef AXIOM_USE_ACCELERATE
                // Math functions for non-Apple platforms
                if constexpr (std::is_same_v<Func, ExpFunc>) {
                    simd::dispatch_unary_exp(input_data, result_data,
                                             total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, LogFunc>) {
                    simd::dispatch_unary_log(input_data, result_data,
                                             total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, SqrtFunc>) {
                    simd::dispatch_unary_sqrt(input_data, result_data,
                                              total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, SinFunc>) {
                    simd::dispatch_unary_sin(input_data, result_data,
                                             total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, CosFunc>) {
                    simd::dispatch_unary_cos(input_data, result_data,
                                             total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, TanFunc>) {
                    simd::dispatch_unary_tan(input_data, result_data,
                                             total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, TanhFunc>) {
                    simd::dispatch_unary_tanh(input_data, result_data,
                                              total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, ErfFunc>) {
                    simd::dispatch_unary_erf(input_data, result_data,
                                             total_elements);
                    return;
                } else if constexpr (std::is_same_v<Func, CbrtFunc>) {
                    simd::dispatch_unary_cbrt(input_data, result_data,
                                              total_elements);
                    return;
                }
#endif // !AXIOM_USE_ACCELERATE
            }
        }
    }

    // Tier 3: Scalar fallback (with OpenMP for large contiguous tensors)
#ifdef AXIOM_USE_OPENMP
    if (input.is_contiguous() && parallel::should_parallelize(total_elements)) {
        ptrdiff_t n = static_cast<ptrdiff_t>(total_elements);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            result_data[i] = func_(input_data[i]);
        }
    } else
#endif
    {
        for (size_t i = 0; i < total_elements; ++i) {
            result_data[i] = func_(input_data[i]);
        }
    }
}

// ============================================================================
// CPU Reduction Operation Implementation
// ============================================================================
namespace { // Anonymous namespace for helpers

Shape calculate_reduction_shape(const Shape &input_shape,
                                const std::vector<int> &axes, bool keep_dims) {
    if (axes.empty()) {
        return keep_dims ? Shape(input_shape.size(), 1) : Shape{1};
    }

    Shape output_shape;
    std::vector<bool> is_reduced_axis(input_shape.size(), false);
    for (int axis : axes) {
        is_reduced_axis[axis] = true;
    }

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (is_reduced_axis[i]) {
            if (keep_dims) {
                output_shape.push_back(1);
            }
        } else {
            output_shape.push_back(input_shape[i]);
        }
    }
    return output_shape;
}
} // namespace

template <typename Func>
Tensor CPUReductionOperation<Func>::execute_reduction(
    const Tensor &input, const std::vector<int> &axis, bool keep_dims) const {
    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU reduction operations");
    }

    // Dispatch based on input dtype
    switch (input.dtype()) {
    case DType::Float16:
        // Use Float32 accumulation for numerical stability, convert result back
        return execute_reduction_typed<float16_t>(input, axis, keep_dims);
    case DType::Float32:
        return execute_reduction_typed<float>(input, axis, keep_dims);
    case DType::Float64:
        return execute_reduction_typed<double>(input, axis, keep_dims);
    case DType::Int32:
        return execute_reduction_typed<int32_t>(input, axis, keep_dims);
    case DType::Int64:
        return execute_reduction_typed<int64_t>(input, axis, keep_dims);
    case DType::Int16:
        return execute_reduction_typed<int16_t>(input, axis, keep_dims);
    case DType::Int8:
        return execute_reduction_typed<int8_t>(input, axis, keep_dims);
    case DType::UInt8:
        return execute_reduction_typed<uint8_t>(input, axis, keep_dims);
    case DType::UInt16:
        return execute_reduction_typed<uint16_t>(input, axis, keep_dims);
    case DType::UInt32:
        return execute_reduction_typed<uint32_t>(input, axis, keep_dims);
    case DType::UInt64:
        return execute_reduction_typed<uint64_t>(input, axis, keep_dims);
    case DType::Bool:
        return execute_reduction_typed<bool>(input, axis, keep_dims);
    case DType::Complex64:
    case DType::Complex128:
        // Complex types only support Sum and Mean reductions
        // Max, Min, Any, All, ArgMax, ArgMin don't have a total ordering on
        // complex
        if (op_type_ != ops::OpType::Sum && op_type_ != ops::OpType::Mean) {
            throw TypeError(
                "Reduction '" + name() +
                "' not supported for complex types (no total ordering)");
        }
        // Fall back to direct implementation instead of template
        // This avoids template instantiation issues with SumFunc identity
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           name() +
                                               " (use higher-level reduction)");
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "CPU reduction operations");
    }
}

template <typename Func>
template <typename T>
void CPUReductionOperation<Func>::reduction_recursive_helper(
    const Tensor &input, Tensor &result, const std::vector<int> &axes,
    std::vector<size_t> &current_coords, int current_dim, const Func &func,
    bool keep_dims) {
    if (current_dim == static_cast<int>(input.ndim())) {
        std::vector<size_t> result_coords;
        if (keep_dims) {
            result_coords = current_coords;
            for (int axis : axes) {
                result_coords[axis] = 0;
            }
        } else {
            for (size_t i = 0; i < input.ndim(); ++i) {
                bool is_reduced = false;
                for (int axis : axes) {
                    if (i == static_cast<size_t>(axis)) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced) {
                    result_coords.push_back(current_coords[i]);
                }
            }
        }

        if (result_coords.empty()) {
            result_coords.push_back(0);
        }

        size_t result_offset =
            ShapeUtils::linear_index(result_coords, result.strides()) /
            result.itemsize();
        T &result_val = result.template typed_data<T>()[result_offset];

        size_t input_offset =
            ShapeUtils::linear_index(current_coords, input.strides()) /
            input.itemsize();
        const T &input_val = input.template typed_data<T>()[input_offset];

        result_val = func(result_val, input_val);
        return;
    }

    for (size_t i = 0; i < input.shape()[current_dim]; ++i) {
        current_coords[current_dim] = i;
        reduction_recursive_helper<T>(input, result, axes, current_coords,
                                      current_dim + 1, func, keep_dims);
    }
}

template <typename Func>
template <typename T>
Tensor CPUReductionOperation<Func>::execute_reduction_typed(
    const Tensor &input, const std::vector<int> &axes, bool keep_dims) const {
    // For Float16, use Float32 accumulation for numerical stability
    constexpr bool use_float32_accum = std::is_same_v<T, float16_t>;
    using AccumT = std::conditional_t<use_float32_accum, float, T>;

    Shape result_shape =
        calculate_reduction_shape(input.shape(), axes, keep_dims);

    // Create result tensor with the appropriate dtype
    DType result_dtype = dtype_of_v<T>;
    DType accum_dtype = dtype_of_v<AccumT>;

    std::vector<int> norm_axes = axes;
    if (norm_axes.empty()) {
        for (size_t i = 0; i < input.ndim(); ++i)
            norm_axes.push_back(static_cast<int>(i));
    }

    // Fast path: Full reduction on contiguous tensor
    bool is_full_reduction = (norm_axes.size() == input.ndim());

#ifdef AXIOM_USE_ACCELERATE
    // Use Accelerate for full contiguous reductions on float32/float64
    if (is_full_reduction && input.is_contiguous()) {
        if constexpr (std::is_same_v<AccumT, float>) {
            const float *data = input.template typed_data<float>();
            size_t n = input.size();
            float result_val;

            if constexpr (std::is_same_v<Func, SumFunc>) {
                result_val = accelerate::vsum_f32(data, n);
            } else if constexpr (std::is_same_v<Func, MaxFunc>) {
                result_val = accelerate::vmax_f32(data, n);
            } else if constexpr (std::is_same_v<Func, MinFunc>) {
                result_val = accelerate::vmin_f32(data, n);
            } else {
                goto fallback_path;
            }

            if (op_type_ == ops::OpType::Mean) {
                result_val /= static_cast<float>(n);
            }

            Shape scalar_shape = keep_dims ? Shape(input.ndim(), 1) : Shape{1};
            Tensor result(scalar_shape, accum_dtype, Device::CPU);
            result.template typed_data<float>()[0] = result_val;

            if constexpr (use_float32_accum) {
                return result.astype(result_dtype);
            }
            return result;
        }
        if constexpr (std::is_same_v<AccumT, double>) {
            const double *data = input.template typed_data<double>();
            size_t n = input.size();
            double result_val;

            if constexpr (std::is_same_v<Func, SumFunc>) {
                result_val = accelerate::vsum_f64(data, n);
            } else if constexpr (std::is_same_v<Func, MaxFunc>) {
                result_val = accelerate::vmax_f64(data, n);
            } else if constexpr (std::is_same_v<Func, MinFunc>) {
                result_val = accelerate::vmin_f64(data, n);
            } else {
                goto fallback_path;
            }

            if (op_type_ == ops::OpType::Mean) {
                result_val /= static_cast<double>(n);
            }

            Shape scalar_shape = keep_dims ? Shape(input.ndim(), 1) : Shape{1};
            Tensor result(scalar_shape, accum_dtype, Device::CPU);
            result.template typed_data<double>()[0] = result_val;
            return result;
        }
    }
fallback_path:
#endif // AXIOM_USE_ACCELERATE

#ifndef AXIOM_USE_ACCELERATE
    // XSIMD fast path for full contiguous reductions
    if (is_full_reduction && input.is_contiguous()) {
        if constexpr (simd::has_support<AccumT>) {
            const AccumT *data;
            Tensor input_converted;
            if constexpr (use_float32_accum) {
                input_converted = input.astype(DType::Float32);
                data = input_converted.template typed_data<AccumT>();
            } else {
                data = input.template typed_data<AccumT>();
            }
            size_t n = input.size();
            AccumT result_val;

            // Runtime dispatched reductions for optimal SIMD
            if constexpr (std::is_same_v<Func, SumFunc>) {
                result_val = simd::dispatch_reduce_sum(data, n);
            } else if constexpr (std::is_same_v<Func, MaxFunc>) {
                result_val = simd::dispatch_reduce_max(data, n);
            } else if constexpr (std::is_same_v<Func, MinFunc>) {
                result_val = simd::dispatch_reduce_min(data, n);
            } else if constexpr (std::is_same_v<Func, ProdFunc>) {
                result_val = simd::dispatch_reduce_prod(data, n);
            } else {
                goto scalar_fallback;
            }

            if (op_type_ == ops::OpType::Mean) {
                result_val /= static_cast<AccumT>(n);
            }

            Shape scalar_shape = keep_dims ? Shape(input.ndim(), 1) : Shape{1};
            Tensor result(scalar_shape, accum_dtype, Device::CPU);
            result.template typed_data<AccumT>()[0] = result_val;

            if constexpr (use_float32_accum) {
                return result.astype(result_dtype);
            }
            return result;
        }
    }
scalar_fallback:
#endif // !AXIOM_USE_ACCELERATE

    // For accumulation, use Float32 if needed
    Tensor result(result_shape, accum_dtype, Device::CPU);
    result.fill(Func::template identity<AccumT>());

    // If using float32 accumulation for float16, we need to convert input
    if constexpr (use_float32_accum) {
        Tensor input_f32 = input.astype(DType::Float32);
        std::vector<size_t> current_coords(input.ndim(), 0);
        reduction_recursive_helper<AccumT>(input_f32, result, norm_axes,
                                           current_coords, 0, func_, keep_dims);
    } else {
        std::vector<size_t> current_coords(input.ndim(), 0);
        reduction_recursive_helper<AccumT>(input, result, norm_axes,
                                           current_coords, 0, func_, keep_dims);
    }

    if (op_type_ == ops::OpType::Mean) {
        size_t reduction_size = 1;
        for (int axis : norm_axes) {
            reduction_size *= input.shape()[axis];
        }

        AccumT *result_data = result.template typed_data<AccumT>();
        for (size_t i = 0; i < result.size(); ++i) {
            result_data[i] /= static_cast<AccumT>(reduction_size);
        }
    }

    // Convert back to original dtype if we used float32 accumulation
    if constexpr (use_float32_accum) {
        return result.astype(result_dtype);
    } else {
        return result;
    }
}

// ============================================================================
// CPU MatMul Operation Implementation
// ============================================================================

void CPUMatMulOperation::get_matmul_dims(const Tensor &a, const Tensor &b,
                                         bool transpose_a, bool transpose_b,
                                         size_t &M, size_t &N, size_t &K,
                                         size_t &K_b) {
    size_t a_ndim = a.ndim();
    size_t b_ndim = b.ndim();

    // For 1D tensors, treat as row/column vector
    size_t a_rows, a_cols, b_rows, b_cols;

    if (a_ndim == 1) {
        a_rows = 1;
        a_cols = a.shape()[0];
    } else {
        a_rows = a.shape()[a_ndim - 2];
        a_cols = a.shape()[a_ndim - 1];
    }

    if (b_ndim == 1) {
        b_rows = b.shape()[0];
        b_cols = 1;
    } else {
        b_rows = b.shape()[b_ndim - 2];
        b_cols = b.shape()[b_ndim - 1];
    }

    // Apply transpose flags
    if (transpose_a)
        std::swap(a_rows, a_cols);
    if (transpose_b)
        std::swap(b_rows, b_cols);

    M = a_rows;
    K = a_cols;
    K_b = b_rows;
    N = b_cols;
}

Shape CPUMatMulOperation::compute_batch_shape(const Tensor &a,
                                              const Tensor &b) {
    // Get batch dimensions (all dims except last 2)
    size_t a_batch_dims = a.ndim() > 2 ? a.ndim() - 2 : 0;
    size_t b_batch_dims = b.ndim() > 2 ? b.ndim() - 2 : 0;

    Shape a_batch, b_batch;
    for (size_t i = 0; i < a_batch_dims; ++i)
        a_batch.push_back(a.shape()[i]);
    for (size_t i = 0; i < b_batch_dims; ++i)
        b_batch.push_back(b.shape()[i]);

    // Broadcast batch dimensions
    return ShapeUtils::broadcast_shape(a_batch, b_batch);
}

template <typename T>
void CPUMatMulOperation::matmul_2d(const T *a_data, const T *b_data, T *c_data,
                                   size_t M, size_t N, size_t K,
                                   size_t a_row_stride, size_t a_col_stride,
                                   size_t b_row_stride, size_t b_col_stride,
                                   size_t c_row_stride, size_t c_col_stride) {

    // Try to use BLAS backend for float32/float64 if strides are compatible
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        // Check if A is row-major or column-major (or transposed version)
        bool a_row_major = (a_col_stride == 1);
        bool a_col_major = (a_row_stride == 1);
        bool b_row_major = (b_col_stride == 1);
        bool b_col_major = (b_row_stride == 1);
        bool c_row_major = (c_col_stride == 1) || (c_col_stride == 0 && N == 1);

        // BLAS requires contiguous storage in some dimension
        // Also require non-zero M, N, K and valid leading dimensions
        if (M > 0 && N > 0 && K > 0 && (a_row_major || a_col_major) &&
            (b_row_major || b_col_major) && c_row_major) {

            // Determine transpose flags and leading dimensions for BLAS
            bool trans_a =
                !a_row_major; // If A is col-major, it looks transposed to BLAS
            bool trans_b = !b_row_major;

            // Leading dimensions (stride of the non-unit dimension)
            // For row-major: lda = stride of row (number of columns in memory)
            // For col-major: lda = stride of column (number of rows in memory)
            size_t lda = a_row_major ? (a_row_stride > 0 ? a_row_stride : K)
                                     : (a_col_stride > 0 ? a_col_stride : M);
            size_t ldb = b_row_major ? (b_row_stride > 0 ? b_row_stride : N)
                                     : (b_col_stride > 0 ? b_col_stride : K);
            size_t ldc = (c_row_stride > 0) ? c_row_stride : N;

            // BLAS requires lda >= K (or M if transposed), ldb >= N (or K if
            // transposed), ldc >= N
            size_t min_lda = trans_a ? M : K;
            size_t min_ldb = trans_b ? K : N;
            size_t min_ldc = N;

            if (lda >= min_lda && ldb >= min_ldb && ldc >= min_ldc && lda > 0 &&
                ldb > 0 && ldc > 0) {

                // Use the BLAS backend abstraction layer
                auto &backend = blas::get_blas_backend();

                if constexpr (std::is_same_v<T, float>) {
                    backend.sgemm(trans_a, trans_b, M, N, K, 1.0f, a_data, lda,
                                  b_data, ldb, 0.0f, c_data, ldc);
                } else {
                    backend.dgemm(trans_a, trans_b, M, N, K, 1.0, a_data, lda,
                                  b_data, ldb, 0.0, c_data, ldc);
                }
                return;
            }
        }
    }

    // Fallback: Cache-blocked matrix multiplication with proper memory access
    // Key optimization: accumulate C in a local tile buffer to avoid repeated
    // loads/stores to C on every K iteration (which causes memory bandwidth
    // bottleneck)
    constexpr size_t TILE_M = 64;
    constexpr size_t TILE_N = 64;
    constexpr size_t TILE_K = 256;

    // Local tile buffer for C accumulation - critical for performance
    alignas(64) T c_tile[TILE_M][TILE_N];

    // Process tiles of C
    for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
        size_t tile_m = std::min(TILE_M, M - i0);

        for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
            size_t tile_n = std::min(TILE_N, N - j0);

            // Initialize c_tile to zero (we're computing from scratch)
            for (size_t i = 0; i < tile_m; ++i) {
                for (size_t j = 0; j < tile_n; ++j) {
                    c_tile[i][j] = T(0);
                }
            }

            // Accumulate A*B into c_tile - NO C memory access in this loop!
            for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                size_t tile_k = std::min(TILE_K, K - k0);

                // Micro-kernel: accumulate into c_tile
                for (size_t i = 0; i < tile_m; ++i) {
                    for (size_t k = 0; k < tile_k; ++k) {
                        T a_val = a_data[(i0 + i) * a_row_stride +
                                         (k0 + k) * a_col_stride];
                        for (size_t j = 0; j < tile_n; ++j) {
                            T b_val = b_data[(k0 + k) * b_row_stride +
                                             (j0 + j) * b_col_stride];
                            c_tile[i][j] += a_val * b_val;
                        }
                    }
                }
            }

            // Store c_tile back to C - ONE store per element
            for (size_t i = 0; i < tile_m; ++i) {
                for (size_t j = 0; j < tile_n; ++j) {
                    c_data[(i0 + i) * c_row_stride + (j0 + j) * c_col_stride] =
                        c_tile[i][j];
                }
            }
        }
    }
}

template <typename T>
Tensor CPUMatMulOperation::execute_matmul_typed(const Tensor &a,
                                                const Tensor &b,
                                                bool transpose_a,
                                                bool transpose_b) const {
    size_t M, N, K, K_b;
    get_matmul_dims(a, b, transpose_a, transpose_b, M, N, K, K_b);

    if (K != K_b) {
        throw ShapeError("MatMul dimension mismatch: A has " +
                         std::to_string(K) + " columns but B has " +
                         std::to_string(K_b) + " rows");
    }

    size_t a_ndim = a.ndim();
    size_t b_ndim = b.ndim();

    // Compute output shape with broadcasted batch dimensions
    Shape result_shape;

    if (a_ndim > 2 || b_ndim > 2) {
        Shape batch_shape = compute_batch_shape(a, b);
        result_shape = batch_shape;
    }

    // Handle 1D cases for output shape
    if (a_ndim == 1 && b_ndim == 1) {
        // Vector dot product: returns scalar
        result_shape = {};
    } else if (a_ndim == 1) {
        // (K,) @ (..., K, N) -> (..., N)
        result_shape.push_back(N);
    } else if (b_ndim == 1) {
        // (..., M, K) @ (K,) -> (..., M)
        result_shape.push_back(M);
    } else {
        // Standard case: (..., M, K) @ (..., K, N) -> (..., M, N)
        result_shape.push_back(M);
        result_shape.push_back(N);
    }

    // Scalar result (rank-0 tensor) is handled naturally with empty shape
    // Use empty() instead of zeros() - BLAS will write all values, and fallback
    // path zeros explicitly. This avoids unnecessary memory initialization.
    Tensor result = Tensor::empty(result_shape, a.dtype(), Device::CPU);

    // Get strides for the matrix dimensions
    size_t a_itemsize = a.itemsize();
    size_t b_itemsize = b.itemsize();
    size_t c_itemsize = result.itemsize();

    // Calculate element strides (converting byte strides to element strides)
    size_t a_row_stride, a_col_stride;
    size_t b_row_stride, b_col_stride;
    size_t c_row_stride, c_col_stride;

    if (a_ndim == 1) {
        a_row_stride = 0;
        a_col_stride = a.strides()[0] / a_itemsize;
    } else {
        a_row_stride = a.strides()[a_ndim - 2] / a_itemsize;
        a_col_stride = a.strides()[a_ndim - 1] / a_itemsize;
    }

    if (b_ndim == 1) {
        b_row_stride = b.strides()[0] / b_itemsize;
        b_col_stride = 0;
    } else {
        b_row_stride = b.strides()[b_ndim - 2] / b_itemsize;
        b_col_stride = b.strides()[b_ndim - 1] / b_itemsize;
    }

    // Handle transpose via stride swapping (zero-copy!)
    if (transpose_a)
        std::swap(a_row_stride, a_col_stride);
    if (transpose_b)
        std::swap(b_row_stride, b_col_stride);

    size_t result_ndim = result.ndim();
    if (result_ndim >= 2) {
        c_row_stride = result.strides()[result_ndim - 2] / c_itemsize;
        c_col_stride = result.strides()[result_ndim - 1] / c_itemsize;
    } else if (result_ndim == 1) {
        // For 1D result (from 1D @ 2D), treat as row vector: c_row=0,
        // c_col=stride Or as column vector (from 2D @ 1D): c_row=stride,
        // c_col=0 We need to determine which based on input shapes
        if (a_ndim == 1 && b_ndim >= 2) {
            // (K,) @ (..., K, N) -> (..., N) - result is conceptually a row, so
            // col varies
            c_row_stride = 0;
            c_col_stride = result.strides()[0] / c_itemsize;
        } else {
            // (..., M, K) @ (K,) -> (..., M) - result is conceptually a column,
            // so row varies
            c_row_stride = result.strides()[0] / c_itemsize;
            c_col_stride = 0;
        }
    } else {
        c_row_stride = 0;
        c_col_stride = 0;
    }

    const T *a_base = a.typed_data<T>();
    const T *b_base = b.typed_data<T>();
    T *c_base = result.typed_data<T>();

    // For simple 2D case without batching
    if (a_ndim <= 2 && b_ndim <= 2) {
        matmul_2d<T>(a_base, b_base, c_base, M, N, K, a_row_stride,
                     a_col_stride, b_row_stride, b_col_stride, c_row_stride,
                     c_col_stride);
    } else {
        // Batch matmul with broadcasting
        Shape batch_shape = compute_batch_shape(a, b);
        size_t batch_size = ShapeUtils::size(batch_shape);
        size_t batch_ndim = batch_shape.size();

        // Compute batch strides for a, b, c
        Strides a_batch_strides(batch_ndim, 0);
        Strides b_batch_strides(batch_ndim, 0);
        Strides c_batch_strides(batch_ndim, 0);

        size_t a_batch_offset = batch_ndim - (a_ndim > 2 ? a_ndim - 2 : 0);
        size_t b_batch_offset = batch_ndim - (b_ndim > 2 ? b_ndim - 2 : 0);

        for (size_t i = 0; i < batch_ndim; ++i) {
            if (i >= a_batch_offset && a_ndim > 2) {
                size_t a_dim_idx = i - a_batch_offset;
                if (a.shape()[a_dim_idx] != 1) {
                    a_batch_strides[i] = a.strides()[a_dim_idx] / a_itemsize;
                }
            }
            if (i >= b_batch_offset && b_ndim > 2) {
                size_t b_dim_idx = i - b_batch_offset;
                if (b.shape()[b_dim_idx] != 1) {
                    b_batch_strides[i] = b.strides()[b_dim_idx] / b_itemsize;
                }
            }
            c_batch_strides[i] = result.strides()[i] / c_itemsize;
        }

        // Iterate over batch dimensions
        // Parallelize batch dimension when batch_size > 1
#ifdef AXIOM_USE_OPENMP
        if (batch_size > 1) {
#pragma omp parallel for schedule(dynamic)
            // MSVC OpenMP requires signed loop index
            for (ptrdiff_t batch_idx = 0;
                 batch_idx < static_cast<ptrdiff_t>(batch_size); ++batch_idx) {
                // Compute batch coordinates from flat index
                size_t remaining = batch_idx;
                int64_t a_batch_off = 0, b_batch_off = 0, c_batch_off = 0;

                for (int d = static_cast<int>(batch_ndim) - 1; d >= 0; --d) {
                    size_t coord = remaining % batch_shape[d];
                    remaining /= batch_shape[d];
                    a_batch_off +=
                        static_cast<int64_t>(coord) * a_batch_strides[d];
                    b_batch_off +=
                        static_cast<int64_t>(coord) * b_batch_strides[d];
                    c_batch_off +=
                        static_cast<int64_t>(coord) * c_batch_strides[d];
                }

                matmul_2d<T>(a_base + a_batch_off, b_base + b_batch_off,
                             c_base + c_batch_off, M, N, K, a_row_stride,
                             a_col_stride, b_row_stride, b_col_stride,
                             c_row_stride, c_col_stride);
            }
        } else
#endif
        {
            // Sequential fallback
            std::vector<size_t> batch_coords(batch_ndim, 0);
            for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                // Compute batch offsets (signed for negative stride support)
                int64_t a_batch_off = 0, b_batch_off = 0, c_batch_off = 0;
                for (size_t i = 0; i < batch_ndim; ++i) {
                    a_batch_off += static_cast<int64_t>(batch_coords[i]) *
                                   a_batch_strides[i];
                    b_batch_off += static_cast<int64_t>(batch_coords[i]) *
                                   b_batch_strides[i];
                    c_batch_off += static_cast<int64_t>(batch_coords[i]) *
                                   c_batch_strides[i];
                }

                matmul_2d<T>(a_base + a_batch_off, b_base + b_batch_off,
                             c_base + c_batch_off, M, N, K, a_row_stride,
                             a_col_stride, b_row_stride, b_col_stride,
                             c_row_stride, c_col_stride);

                // Increment batch coordinates
                for (int i = batch_ndim - 1; i >= 0; --i) {
                    if (++batch_coords[i] < batch_shape[i])
                        break;
                    batch_coords[i] = 0;
                }
            }
        }
    }

    return result;
}

Tensor CPUMatMulOperation::execute_matmul(const Tensor &a, const Tensor &b,
                                          bool transpose_a,
                                          bool transpose_b) const {
    if (a.device() != Device::CPU || b.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU MatMul");
    }

    if (a.ndim() == 0 || b.ndim() == 0) {
        throw ShapeError("MatMul does not support 0-dimensional tensors");
    }

    // Fast path: Contiguous 2D float32/float64 matrices with same dtype
    // Skip all the overhead of type promotion and complex stride handling
    // This path uses the BLAS backend abstraction for all platforms
    if (a.ndim() == 2 && b.ndim() == 2 && a.dtype() == b.dtype() &&
        a.is_contiguous() && b.is_contiguous() &&
        (a.dtype() == DType::Float32 || a.dtype() == DType::Float64)) {

        size_t M = transpose_a ? a.shape()[1] : a.shape()[0];
        size_t K_a = transpose_a ? a.shape()[0] : a.shape()[1];
        size_t K_b = transpose_b ? b.shape()[1] : b.shape()[0];
        size_t N = transpose_b ? b.shape()[0] : b.shape()[1];

        if (K_a != K_b) {
            throw ShapeError("MatMul dimension mismatch: A has " +
                             std::to_string(K_a) + " columns but B has " +
                             std::to_string(K_b) + " rows");
        }

        Tensor result = Tensor::empty({M, N}, a.dtype(), Device::CPU);

        size_t lda = a.shape()[1]; // Leading dimension of A (cols)
        size_t ldb = b.shape()[1]; // Leading dimension of B (cols)
        size_t ldc = N;            // Leading dimension of C (cols)

        // Use the BLAS backend abstraction layer
        auto &backend = blas::get_blas_backend();

        if (a.dtype() == DType::Float32) {
            backend.sgemm(transpose_a, transpose_b, M, N, K_a, 1.0f,
                          a.typed_data<float>(), lda, b.typed_data<float>(),
                          ldb, 0.0f, result.typed_data<float>(), ldc);
        } else {
            backend.dgemm(transpose_a, transpose_b, M, N, K_a, 1.0,
                          a.typed_data<double>(), lda, b.typed_data<double>(),
                          ldb, 0.0, result.typed_data<double>(), ldc);
        }
        return result;
    }

    // Type promote and dispatch
    DType result_dtype = ops::promote_types(a.dtype(), b.dtype());
    Tensor a_promoted =
        (a.dtype() == result_dtype) ? a : a.astype(result_dtype);
    Tensor b_promoted =
        (b.dtype() == result_dtype) ? b : b.astype(result_dtype);

#define DISPATCH_MATMUL(DTYPE, CTYPE)                                          \
    case DTYPE:                                                                \
        return execute_matmul_typed<CTYPE>(a_promoted, b_promoted,             \
                                           transpose_a, transpose_b);

    switch (result_dtype) {
        DISPATCH_MATMUL(DType::Float16, float16_t)
        DISPATCH_MATMUL(DType::Float32, float)
        DISPATCH_MATMUL(DType::Float64, double)
        DISPATCH_MATMUL(DType::Int32, int32_t)
        DISPATCH_MATMUL(DType::Int64, int64_t)
        DISPATCH_MATMUL(DType::Complex64, complex64_t)
        DISPATCH_MATMUL(DType::Complex128, complex128_t)
    default:
        throw TypeError::unsupported_dtype(dtype_name(result_dtype), "MatMul");
    }
#undef DISPATCH_MATMUL
}

// ============================================================================
// CPU ArgMax/ArgMin Operation Implementation
// ============================================================================

template <typename T>
Tensor CPUArgMaxOperation::execute_argmax_typed(const Tensor &input, int axis,
                                                bool keep_dims) const {
    size_t ndim = input.ndim();

    // Normalize axis
    if (axis < 0)
        axis += static_cast<int>(ndim);
    if (axis < 0 || axis >= static_cast<int>(ndim)) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // Calculate output shape
    Shape output_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int>(i) == axis) {
            if (keep_dims)
                output_shape.push_back(1);
        } else {
            output_shape.push_back(input.shape()[i]);
        }
    }
    // Scalar result (rank-0 tensor) is handled naturally with empty shape

    // Create output tensor with Int64 dtype for indices
    Tensor result = Tensor::zeros(output_shape, DType::Int64, Device::CPU);
    int64_t *result_data = result.typed_data<int64_t>();
    const T *input_data = input.typed_data<T>();

#ifdef AXIOM_USE_ACCELERATE
    // Fast path: Use Accelerate for full reduction on contiguous
    // float32/float64
    if (ndim == 1 && axis == 0 && input.is_contiguous()) {
        if constexpr (std::is_same_v<T, float>) {
            size_t idx = accelerate::vargmax_f32(input_data, input.size());
            result_data[0] = static_cast<int64_t>(idx);
            return result;
        } else if constexpr (std::is_same_v<T, double>) {
            size_t idx = accelerate::vargmax_f64(input_data, input.size());
            result_data[0] = static_cast<int64_t>(idx);
            return result;
        }
    }
#endif

    // Calculate sizes
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i)
        outer_size *= input.shape()[i];

    size_t axis_size = input.shape()[axis];

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < ndim; ++i)
        inner_size *= input.shape()[i];

    // Iterate over all positions
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            // Find max along axis
            size_t best_idx = 0;
            T best_val = std::numeric_limits<T>::lowest();

            for (size_t k = 0; k < axis_size; ++k) {
                // Calculate input index
                std::vector<size_t> coords(ndim);
                size_t temp_outer = outer;
                for (int i = axis - 1; i >= 0; --i) {
                    coords[i] = temp_outer % input.shape()[i];
                    temp_outer /= input.shape()[i];
                }
                coords[axis] = k;
                size_t temp_inner = inner;
                for (int i = static_cast<int>(ndim) - 1; i > axis; --i) {
                    coords[i] = temp_inner % input.shape()[i];
                    temp_inner /= input.shape()[i];
                }

                size_t input_offset = 0;
                for (size_t i = 0; i < ndim; ++i) {
                    input_offset +=
                        coords[i] * (input.strides()[i] / input.itemsize());
                }

                T val = input_data[input_offset];
                if (val > best_val) {
                    best_val = val;
                    best_idx = k;
                }
            }

            // Store result
            size_t result_idx = outer * inner_size + inner;
            result_data[result_idx] = static_cast<int64_t>(best_idx);
        }
    }

    return result;
}

Tensor CPUArgMaxOperation::execute_reduction(const Tensor &input,
                                             const std::vector<int> &axis,
                                             bool keep_dims) const {
    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU ArgMax");
    }

    int ax = axis.empty() ? -1 : axis[0];

    // For full reduction (axis=-1 or all axes), flatten first
    if (ax == -1 || axis.size() > 1) {
        auto flat = input.flatten();
        ax = 0;
    }

#define DISPATCH_ARGMAX(DTYPE, CTYPE)                                          \
    case DTYPE:                                                                \
        return execute_argmax_typed<CTYPE>(input, ax, keep_dims);

    switch (input.dtype()) {
        DISPATCH_ARGMAX(DType::Float16, float16_t)
        DISPATCH_ARGMAX(DType::Float32, float)
        DISPATCH_ARGMAX(DType::Float64, double)
        DISPATCH_ARGMAX(DType::Int32, int32_t)
        DISPATCH_ARGMAX(DType::Int64, int64_t)
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()), "ArgMax");
    }
#undef DISPATCH_ARGMAX
}

template <typename T>
Tensor CPUArgMinOperation::execute_argmin_typed(const Tensor &input, int axis,
                                                bool keep_dims) const {
    size_t ndim = input.ndim();

    // Normalize axis
    if (axis < 0)
        axis += static_cast<int>(ndim);
    if (axis < 0 || axis >= static_cast<int>(ndim)) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // Calculate output shape
    Shape output_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int>(i) == axis) {
            if (keep_dims)
                output_shape.push_back(1);
        } else {
            output_shape.push_back(input.shape()[i]);
        }
    }
    // Scalar result (rank-0 tensor) is handled naturally with empty shape

    // Create output tensor with Int64 dtype for indices
    Tensor result = Tensor::zeros(output_shape, DType::Int64, Device::CPU);
    int64_t *result_data = result.typed_data<int64_t>();
    const T *input_data = input.typed_data<T>();

#ifdef AXIOM_USE_ACCELERATE
    // Fast path: Use Accelerate for full reduction on contiguous
    // float32/float64
    if (ndim == 1 && axis == 0 && input.is_contiguous()) {
        if constexpr (std::is_same_v<T, float>) {
            size_t idx = accelerate::vargmin_f32(input_data, input.size());
            result_data[0] = static_cast<int64_t>(idx);
            return result;
        } else if constexpr (std::is_same_v<T, double>) {
            size_t idx = accelerate::vargmin_f64(input_data, input.size());
            result_data[0] = static_cast<int64_t>(idx);
            return result;
        }
    }
#endif

    // Calculate sizes
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i)
        outer_size *= input.shape()[i];

    size_t axis_size = input.shape()[axis];

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < ndim; ++i)
        inner_size *= input.shape()[i];

    // Iterate over all positions
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            // Find min along axis
            size_t best_idx = 0;
            T best_val = std::numeric_limits<T>::max();

            for (size_t k = 0; k < axis_size; ++k) {
                // Calculate input index
                std::vector<size_t> coords(ndim);
                size_t temp_outer = outer;
                for (int i = axis - 1; i >= 0; --i) {
                    coords[i] = temp_outer % input.shape()[i];
                    temp_outer /= input.shape()[i];
                }
                coords[axis] = k;
                size_t temp_inner = inner;
                for (int i = static_cast<int>(ndim) - 1; i > axis; --i) {
                    coords[i] = temp_inner % input.shape()[i];
                    temp_inner /= input.shape()[i];
                }

                size_t input_offset = 0;
                for (size_t i = 0; i < ndim; ++i) {
                    input_offset +=
                        coords[i] * (input.strides()[i] / input.itemsize());
                }

                T val = input_data[input_offset];
                if (val < best_val) {
                    best_val = val;
                    best_idx = k;
                }
            }

            // Store result
            size_t result_idx = outer * inner_size + inner;
            result_data[result_idx] = static_cast<int64_t>(best_idx);
        }
    }

    return result;
}

Tensor CPUArgMinOperation::execute_reduction(const Tensor &input,
                                             const std::vector<int> &axis,
                                             bool keep_dims) const {
    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU ArgMin");
    }

    int ax = axis.empty() ? -1 : axis[0];

    // For full reduction (axis=-1 or all axes), flatten first
    if (ax == -1 || axis.size() > 1) {
        auto flat = input.flatten();
        ax = 0;
    }

#define DISPATCH_ARGMIN(DTYPE, CTYPE)                                          \
    case DTYPE:                                                                \
        return execute_argmin_typed<CTYPE>(input, ax, keep_dims);

    switch (input.dtype()) {
        DISPATCH_ARGMIN(DType::Float16, float16_t)
        DISPATCH_ARGMIN(DType::Float32, float)
        DISPATCH_ARGMIN(DType::Float64, double)
        DISPATCH_ARGMIN(DType::Int32, int32_t)
        DISPATCH_ARGMIN(DType::Int64, int64_t)
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()), "ArgMin");
    }
#undef DISPATCH_ARGMIN
}

// ============================================================================
// CPU Where Operation Implementation
// ============================================================================

// Helper to compute broadcast strides: maps input shape to output shape strides
// Returns strides in bytes where broadcasted dimensions have stride 0
static Strides compute_broadcast_strides(const Shape &input_shape,
                                         const Shape &output_shape,
                                         const Strides &input_strides) {
    size_t out_ndim = output_shape.size();
    size_t in_ndim = input_shape.size();
    Strides result(out_ndim, 0);

    // Align shapes from the right
    for (size_t i = 0; i < out_ndim; ++i) {
        size_t out_idx = out_ndim - 1 - i;
        if (i < in_ndim) {
            size_t in_idx = in_ndim - 1 - i;
            // If input dimension is 1, it's broadcast (stride = 0)
            // Otherwise, use the input stride
            if (input_shape[in_idx] == output_shape[out_idx]) {
                result[out_idx] = input_strides[in_idx];
            } else if (input_shape[in_idx] == 1) {
                result[out_idx] = 0; // Broadcast dimension
            } else {
                // This shouldn't happen if shapes are properly broadcastable
                result[out_idx] = input_strides[in_idx];
            }
        } else {
            // Input has fewer dimensions - broadcast with stride 0
            result[out_idx] = 0;
        }
    }
    return result;
}

// Helper to get value from tensor at byte offset, converting to output type T
template <typename T>
static T get_tensor_value_at(const void *data, size_t byte_offset,
                             DType dtype) {
    switch (dtype) {
    case DType::Bool:
        return static_cast<T>(*reinterpret_cast<const bool *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::Int8:
        return static_cast<T>(*reinterpret_cast<const int8_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::Int16:
        return static_cast<T>(*reinterpret_cast<const int16_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::Int32:
        return static_cast<T>(*reinterpret_cast<const int32_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::Int64:
        return static_cast<T>(*reinterpret_cast<const int64_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::UInt8:
        return static_cast<T>(*reinterpret_cast<const uint8_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::UInt16:
        return static_cast<T>(*reinterpret_cast<const uint16_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::UInt32:
        return static_cast<T>(*reinterpret_cast<const uint32_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::UInt64:
        return static_cast<T>(*reinterpret_cast<const uint64_t *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::Float16:
        return static_cast<T>(
            static_cast<float>(*reinterpret_cast<const float16_t *>(
                static_cast<const uint8_t *>(data) + byte_offset)));
    case DType::Float32:
        return static_cast<T>(*reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    case DType::Float64:
        return static_cast<T>(*reinterpret_cast<const double *>(
            static_cast<const uint8_t *>(data) + byte_offset));
    default:
        return T(0);
    }
}

template <typename T>
Tensor CPUWhereOperation::execute_where_typed(const Tensor &condition,
                                              const Tensor &a,
                                              const Tensor &b) const {
    // Compute broadcast shape for all three inputs
    Shape temp_shape =
        ShapeUtils::broadcast_shape(condition.shape(), a.shape());
    Shape output_shape = ShapeUtils::broadcast_shape(temp_shape, b.shape());

    // Determine output dtype from a and b (condition is bool)
    DType output_dtype = ops::promote_types(a.dtype(), b.dtype());

    // Create output tensor
    Tensor result(output_shape, output_dtype, Device::CPU);

    size_t numel = ShapeUtils::size(output_shape);
    if (numel == 0)
        return result;

    // Get strides for broadcasting
    auto cond_strides = compute_broadcast_strides(
        condition.shape(), output_shape, condition.strides());
    auto a_strides =
        compute_broadcast_strides(a.shape(), output_shape, a.strides());
    auto b_strides =
        compute_broadcast_strides(b.shape(), output_shape, b.strides());
    auto result_strides = result.strides();

    // Get data pointers
    const uint8_t *cond_data = static_cast<const uint8_t *>(condition.data());
    const void *a_data = a.data();
    const void *b_data = b.data();
    T *result_data = result.typed_data<T>();

    size_t result_itemsize = result.itemsize();
    size_t cond_itemsize = condition.itemsize();
    size_t ndim = output_shape.size();
    DType a_dtype = a.dtype();
    DType b_dtype = b.dtype();
    DType cond_dtype = condition.dtype();

    // Parallel path for large tensors
#ifdef AXIOM_USE_OPENMP
    if (parallel::should_parallelize(numel)) {
        ptrdiff_t n = static_cast<ptrdiff_t>(numel);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            // Compute coordinates from flat index
            size_t remaining = static_cast<size_t>(i);
            int64_t cond_offset = 0;
            int64_t a_offset = 0;
            int64_t b_offset = 0;
            int64_t result_offset = 0;

            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                size_t coord = remaining % output_shape[d];
                remaining /= output_shape[d];
                cond_offset += static_cast<int64_t>(coord) * cond_strides[d];
                a_offset += static_cast<int64_t>(coord) * a_strides[d];
                b_offset += static_cast<int64_t>(coord) * b_strides[d];
                result_offset +=
                    static_cast<int64_t>(coord) * result_strides[d];
            }

            // Get condition value
            bool cond_val = false;
            if (cond_dtype == DType::Bool) {
                cond_val =
                    *reinterpret_cast<const bool *>(cond_data + cond_offset);
            } else {
                for (size_t byte = 0; byte < cond_itemsize; ++byte) {
                    if (cond_data[cond_offset + byte] != 0) {
                        cond_val = true;
                        break;
                    }
                }
            }

            T value = cond_val
                          ? get_tensor_value_at<T>(a_data, a_offset, a_dtype)
                          : get_tensor_value_at<T>(b_data, b_offset, b_dtype);
            result_data[result_offset / result_itemsize] = value;
        }
        return result;
    }
#endif

    // Sequential fallback with coordinate tracking
    std::vector<size_t> coords(ndim, 0);

    for (size_t i = 0; i < numel; ++i) {
        // Compute byte offsets for each input (signed for negative stride
        // support)
        int64_t cond_offset = 0;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t result_offset = 0;

        for (size_t d = 0; d < ndim; ++d) {
            cond_offset += static_cast<int64_t>(coords[d]) * cond_strides[d];
            a_offset += static_cast<int64_t>(coords[d]) * a_strides[d];
            b_offset += static_cast<int64_t>(coords[d]) * b_strides[d];
            result_offset +=
                static_cast<int64_t>(coords[d]) * result_strides[d];
        }

        // Get condition value (handle different condition dtypes)
        bool cond_val = false;
        if (cond_dtype == DType::Bool) {
            cond_val = *reinterpret_cast<const bool *>(cond_data + cond_offset);
        } else {
            // For numeric types, non-zero is true
            // Check if any byte is non-zero
            for (size_t byte = 0; byte < cond_itemsize; ++byte) {
                if (cond_data[cond_offset + byte] != 0) {
                    cond_val = true;
                    break;
                }
            }
        }

        // Select from a or b based on condition, with proper type conversion
        T value = cond_val ? get_tensor_value_at<T>(a_data, a_offset, a_dtype)
                           : get_tensor_value_at<T>(b_data, b_offset, b_dtype);
        result_data[result_offset / result_itemsize] = value;

        // Increment coordinates
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < output_shape[d])
                break;
            coords[d] = 0;
        }
    }

    return result;
}

Tensor CPUWhereOperation::execute_where(const Tensor &condition,
                                        const Tensor &a,
                                        const Tensor &b) const {
    if (condition.device() != Device::CPU || a.device() != Device::CPU ||
        b.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU Where");
    }

    // Determine output dtype from a and b
    DType output_dtype = ops::promote_types(a.dtype(), b.dtype());

#define DISPATCH_WHERE(DTYPE, CTYPE)                                           \
    case DTYPE:                                                                \
        return execute_where_typed<CTYPE>(condition, a, b);

    switch (output_dtype) {
        DISPATCH_WHERE(DType::Float16, float16_t)
        DISPATCH_WHERE(DType::Float32, float)
        DISPATCH_WHERE(DType::Float64, double)
        DISPATCH_WHERE(DType::Int32, int32_t)
        DISPATCH_WHERE(DType::Int64, int64_t)
        DISPATCH_WHERE(DType::Int16, int16_t)
        DISPATCH_WHERE(DType::Int8, int8_t)
        DISPATCH_WHERE(DType::UInt8, uint8_t)
        DISPATCH_WHERE(DType::UInt16, uint16_t)
        DISPATCH_WHERE(DType::UInt32, uint32_t)
        DISPATCH_WHERE(DType::UInt64, uint64_t)
        DISPATCH_WHERE(DType::Bool, bool)
    default:
        throw TypeError::unsupported_dtype(dtype_name(output_dtype), "Where");
    }
#undef DISPATCH_WHERE
}

// ============================================================================
// CPU MaskedFill Implementation
// ============================================================================

template <typename T>
Tensor CPUMaskedFillOperation::execute_masked_fill_typed(
    const Tensor &input, const Tensor &mask, const Tensor &value) const {

    // Create output tensor (copy of input)
    Tensor result = input.copy();

    // Get value to fill
    T fill_value;
    if (value.dtype() == input.dtype()) {
        fill_value = value.typed_data<T>()[0];
    } else {
        // Convert value to target type
        Tensor value_converted = value.astype(input.dtype());
        fill_value = value_converted.typed_data<T>()[0];
    }

    // Handle broadcasting: mask and input may have different shapes
    auto broadcast_info =
        ops::compute_broadcast_info(input.shape(), mask.shape());
    const auto &result_shape = broadcast_info.result_shape;

    // If shapes match, fast path
    if (input.shape() == mask.shape() && input.is_contiguous() &&
        mask.is_contiguous()) {
        T *result_data = result.typed_data<T>();
        const uint8_t *mask_data = mask.typed_data<uint8_t>();
        size_t n_elements = input.size();

#ifdef AXIOM_USE_OPENMP
        if (parallel::should_parallelize(n_elements)) {
            ptrdiff_t n = static_cast<ptrdiff_t>(n_elements);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) {
                if (mask_data[i]) {
                    result_data[i] = fill_value;
                }
            }
        } else
#endif
        {
            for (size_t i = 0; i < n_elements; ++i) {
                if (mask_data[i]) {
                    result_data[i] = fill_value;
                }
            }
        }
        return result;
    }

    // General case with broadcasting
    Tensor input_expanded = input.broadcast_to(result_shape);
    Tensor mask_expanded = mask.broadcast_to(result_shape);
    result = Tensor(result_shape, input.dtype(), Device::CPU);

    T *result_data = result.typed_data<T>();
    size_t total_size = ShapeUtils::size(result_shape);
    const auto &mask_strides = mask_expanded.strides();
    const auto &input_strides = input_expanded.strides();
    const void *mask_ptr = mask_expanded.data();
    const void *input_ptr = input_expanded.data();

#ifdef AXIOM_USE_OPENMP
    if (parallel::should_parallelize(total_size)) {
        ptrdiff_t n = static_cast<ptrdiff_t>(total_size);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            auto coords =
                ShapeUtils::unravel_index(static_cast<size_t>(i), result_shape);
            size_t mask_offset = ShapeUtils::linear_index(coords, mask_strides);
            bool mask_val =
                *reinterpret_cast<const uint8_t *>(
                    static_cast<const uint8_t *>(mask_ptr) + mask_offset) != 0;
            size_t input_offset =
                ShapeUtils::linear_index(coords, input_strides);
            T input_val = *reinterpret_cast<const T *>(
                static_cast<const uint8_t *>(input_ptr) + input_offset);
            result_data[i] = mask_val ? fill_value : input_val;
        }
        return result;
    }
#endif

    for (size_t i = 0; i < total_size; ++i) {
        auto coords = ShapeUtils::unravel_index(i, result_shape);

        // Get mask value
        size_t mask_offset = ShapeUtils::linear_index(coords, mask_strides);
        bool mask_val =
            *reinterpret_cast<const uint8_t *>(
                static_cast<const uint8_t *>(mask_ptr) + mask_offset) != 0;

        // Get input value
        size_t input_offset = ShapeUtils::linear_index(coords, input_strides);
        T input_val = *reinterpret_cast<const T *>(
            static_cast<const uint8_t *>(input_ptr) + input_offset);

        result_data[i] = mask_val ? fill_value : input_val;
    }

    return result;
}

Tensor CPUMaskedFillOperation::execute_masked_fill(const Tensor &input,
                                                   const Tensor &mask,
                                                   const Tensor &value) const {
    if (input.device() != Device::CPU || mask.device() != Device::CPU ||
        value.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU MaskedFill");
    }

#define DISPATCH_MASKED_FILL(DTYPE, CTYPE)                                     \
    case DTYPE:                                                                \
        return execute_masked_fill_typed<CTYPE>(input, mask, value);

    switch (input.dtype()) {
        DISPATCH_MASKED_FILL(DType::Float16, float16_t)
        DISPATCH_MASKED_FILL(DType::Float32, float)
        DISPATCH_MASKED_FILL(DType::Float64, double)
        DISPATCH_MASKED_FILL(DType::Int32, int32_t)
        DISPATCH_MASKED_FILL(DType::Int64, int64_t)
        DISPATCH_MASKED_FILL(DType::Int16, int16_t)
        DISPATCH_MASKED_FILL(DType::Int8, int8_t)
        DISPATCH_MASKED_FILL(DType::UInt8, uint8_t)
        DISPATCH_MASKED_FILL(DType::UInt16, uint16_t)
        DISPATCH_MASKED_FILL(DType::UInt32, uint32_t)
        DISPATCH_MASKED_FILL(DType::UInt64, uint64_t)
        DISPATCH_MASKED_FILL(DType::Bool, bool)
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "MaskedFill");
    }
#undef DISPATCH_MASKED_FILL
}

// ============================================================================
// CPU MaskedSelect Implementation
// ============================================================================

template <typename T>
Tensor CPUMaskedSelectOperation::execute_masked_select_typed(
    const Tensor &input, const Tensor &mask) const {

    // Handle broadcasting
    auto broadcast_info =
        ops::compute_broadcast_info(input.shape(), mask.shape());
    const auto &result_shape = broadcast_info.result_shape;

    Tensor input_expanded = input.broadcast_to(result_shape);
    Tensor mask_expanded = mask.broadcast_to(result_shape);

    // First pass: count selected elements
    size_t count = 0;
    size_t total_size = ShapeUtils::size(result_shape);

    for (size_t i = 0; i < total_size; ++i) {
        auto coords = ShapeUtils::unravel_index(i, result_shape);
        size_t mask_offset =
            ShapeUtils::linear_index(coords, mask_expanded.strides());
        bool mask_val = *reinterpret_cast<const uint8_t *>(
                            static_cast<const uint8_t *>(mask_expanded.data()) +
                            mask_offset) != 0;
        if (mask_val) {
            count++;
        }
    }

    // Create output tensor (1D)
    Tensor result({count}, input.dtype(), Device::CPU);
    if (count == 0) {
        return result;
    }

    // Second pass: gather selected elements
    T *result_data = result.typed_data<T>();
    size_t out_idx = 0;

    for (size_t i = 0; i < total_size; ++i) {
        auto coords = ShapeUtils::unravel_index(i, result_shape);

        size_t mask_offset =
            ShapeUtils::linear_index(coords, mask_expanded.strides());
        bool mask_val = *reinterpret_cast<const uint8_t *>(
                            static_cast<const uint8_t *>(mask_expanded.data()) +
                            mask_offset) != 0;

        if (mask_val) {
            size_t input_offset =
                ShapeUtils::linear_index(coords, input_expanded.strides());
            T input_val = *reinterpret_cast<const T *>(
                static_cast<const uint8_t *>(input_expanded.data()) +
                input_offset);
            result_data[out_idx++] = input_val;
        }
    }

    return result;
}

Tensor
CPUMaskedSelectOperation::execute_masked_select(const Tensor &input,
                                                const Tensor &mask) const {
    if (input.device() != Device::CPU || mask.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU MaskedSelect");
    }

#define DISPATCH_MASKED_SELECT(DTYPE, CTYPE)                                   \
    case DTYPE:                                                                \
        return execute_masked_select_typed<CTYPE>(input, mask);

    switch (input.dtype()) {
        DISPATCH_MASKED_SELECT(DType::Float16, float16_t)
        DISPATCH_MASKED_SELECT(DType::Float32, float)
        DISPATCH_MASKED_SELECT(DType::Float64, double)
        DISPATCH_MASKED_SELECT(DType::Int32, int32_t)
        DISPATCH_MASKED_SELECT(DType::Int64, int64_t)
        DISPATCH_MASKED_SELECT(DType::Int16, int16_t)
        DISPATCH_MASKED_SELECT(DType::Int8, int8_t)
        DISPATCH_MASKED_SELECT(DType::UInt8, uint8_t)
        DISPATCH_MASKED_SELECT(DType::UInt16, uint16_t)
        DISPATCH_MASKED_SELECT(DType::UInt32, uint32_t)
        DISPATCH_MASKED_SELECT(DType::UInt64, uint64_t)
        DISPATCH_MASKED_SELECT(DType::Bool, bool)
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "MaskedSelect");
    }
#undef DISPATCH_MASKED_SELECT
}

// ============================================================================
// CPU Gather Implementation
// ============================================================================

template <typename T>
Tensor CPUGatherOperation::execute_gather_typed(const Tensor &input, int dim,
                                                const Tensor &indices) const {
    // Normalize dim
    int norm_dim = dim;
    if (norm_dim < 0) {
        norm_dim += static_cast<int>(input.ndim());
    }

    // Output has same shape as indices
    Tensor result(indices.shape(), input.dtype(), Device::CPU);

    // Get indices data (always int64)
    Tensor indices_i64 = indices.astype(DType::Int64);
    const int64_t *indices_data = indices_i64.typed_data<int64_t>();

    T *result_data = result.typed_data<T>();

    size_t total_size = indices.size();
    const auto &input_shape = input.shape();
    const auto &indices_shape = indices.shape();
    const auto &input_strides = input.strides();
    const void *input_ptr = input.data();
    int64_t dim_size = static_cast<int64_t>(input_shape[norm_dim]);

#ifdef AXIOM_USE_OPENMP
    if (parallel::should_parallelize(total_size)) {
        ptrdiff_t n = static_cast<ptrdiff_t>(total_size);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            auto coords = ShapeUtils::unravel_index(static_cast<size_t>(i),
                                                    indices_shape);
            int64_t idx = indices_data[i];
            if (idx < 0) {
                idx += dim_size;
            }
            coords[norm_dim] = static_cast<size_t>(idx);
            size_t input_offset =
                ShapeUtils::linear_index(coords, input_strides);
            result_data[i] = *reinterpret_cast<const T *>(
                static_cast<const uint8_t *>(input_ptr) + input_offset);
        }
        return result;
    }
#endif

    for (size_t i = 0; i < total_size; ++i) {
        auto coords = ShapeUtils::unravel_index(i, indices_shape);

        // Get the index value
        int64_t idx = indices_data[i];

        // Handle negative indices
        if (idx < 0) {
            idx += dim_size;
        }

        // Build input coordinates: replace dim with idx
        coords[norm_dim] = static_cast<size_t>(idx);

        // Get the value from input
        size_t input_offset = ShapeUtils::linear_index(coords, input_strides);
        result_data[i] = *reinterpret_cast<const T *>(
            static_cast<const uint8_t *>(input_ptr) + input_offset);
    }

    return result;
}

Tensor CPUGatherOperation::execute_gather(const Tensor &input, int dim,
                                          const Tensor &indices) const {
    if (input.device() != Device::CPU || indices.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU Gather");
    }

#define DISPATCH_GATHER(DTYPE, CTYPE)                                          \
    case DTYPE:                                                                \
        return execute_gather_typed<CTYPE>(input, dim, indices);

    switch (input.dtype()) {
        DISPATCH_GATHER(DType::Float16, float16_t)
        DISPATCH_GATHER(DType::Float32, float)
        DISPATCH_GATHER(DType::Float64, double)
        DISPATCH_GATHER(DType::Int32, int32_t)
        DISPATCH_GATHER(DType::Int64, int64_t)
        DISPATCH_GATHER(DType::Int16, int16_t)
        DISPATCH_GATHER(DType::Int8, int8_t)
        DISPATCH_GATHER(DType::UInt8, uint8_t)
        DISPATCH_GATHER(DType::UInt16, uint16_t)
        DISPATCH_GATHER(DType::UInt32, uint32_t)
        DISPATCH_GATHER(DType::UInt64, uint64_t)
        DISPATCH_GATHER(DType::Bool, bool)
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()), "Gather");
    }
#undef DISPATCH_GATHER
}

// ============================================================================
// CPU Scatter Implementation
// ============================================================================

template <typename T>
Tensor CPUScatterOperation::execute_scatter_typed(const Tensor &input, int dim,
                                                  const Tensor &indices,
                                                  const Tensor &src) const {
    // Normalize dim
    int norm_dim = dim;
    if (norm_dim < 0) {
        norm_dim += static_cast<int>(input.ndim());
    }

    // Output starts as copy of input
    Tensor result = input.copy();

    // Get indices data (always int64)
    Tensor indices_i64 = indices.astype(DType::Int64);
    const int64_t *indices_data = indices_i64.typed_data<int64_t>();

    // Convert src to same dtype as input
    Tensor src_converted =
        (src.dtype() == input.dtype()) ? src : src.astype(input.dtype());
    const T *src_data = src_converted.typed_data<T>();

    size_t total_size = indices.size();
    const auto &result_shape = result.shape();
    const auto &indices_shape = indices.shape();
    const auto &result_strides = result.strides();

    for (size_t i = 0; i < total_size; ++i) {
        auto coords = ShapeUtils::unravel_index(i, indices_shape);

        // Get the index value
        int64_t idx = indices_data[i];

        // Handle negative indices
        if (idx < 0) {
            idx += static_cast<int64_t>(result_shape[norm_dim]);
        }

        // Build result coordinates: replace dim with idx
        std::vector<size_t> result_coords = coords;
        result_coords[norm_dim] = static_cast<size_t>(idx);

        // Set the value in result
        size_t result_offset =
            ShapeUtils::linear_index(result_coords, result_strides);
        *reinterpret_cast<T *>(static_cast<uint8_t *>(result.data()) +
                               result_offset) = src_data[i];
    }

    return result;
}

Tensor CPUScatterOperation::execute_scatter(const Tensor &input, int dim,
                                            const Tensor &indices,
                                            const Tensor &src) const {
    if (input.device() != Device::CPU || indices.device() != Device::CPU ||
        src.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU Scatter");
    }

#define DISPATCH_SCATTER(DTYPE, CTYPE)                                         \
    case DTYPE:                                                                \
        return execute_scatter_typed<CTYPE>(input, dim, indices, src);

    switch (input.dtype()) {
        DISPATCH_SCATTER(DType::Float16, float16_t)
        DISPATCH_SCATTER(DType::Float32, float)
        DISPATCH_SCATTER(DType::Float64, double)
        DISPATCH_SCATTER(DType::Int32, int32_t)
        DISPATCH_SCATTER(DType::Int64, int64_t)
        DISPATCH_SCATTER(DType::Int16, int16_t)
        DISPATCH_SCATTER(DType::Int8, int8_t)
        DISPATCH_SCATTER(DType::UInt8, uint8_t)
        DISPATCH_SCATTER(DType::UInt16, uint16_t)
        DISPATCH_SCATTER(DType::UInt32, uint32_t)
        DISPATCH_SCATTER(DType::UInt64, uint64_t)
        DISPATCH_SCATTER(DType::Bool, bool)
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "Scatter");
    }
#undef DISPATCH_SCATTER
}

// ============================================================================
// CPU IndexSelect Implementation
// ============================================================================

template <typename T>
Tensor CPUIndexSelectOperation::execute_index_select_typed(
    const Tensor &input, int dim, const Tensor &indices) const {
    // Normalize dim
    int norm_dim = dim;
    if (norm_dim < 0) {
        norm_dim += static_cast<int>(input.ndim());
    }

    // Get indices data
    Tensor indices_i64 = indices.astype(DType::Int64);
    const int64_t *indices_data = indices_i64.typed_data<int64_t>();
    size_t num_indices = indices.size();

    // Compute output shape: replace dim size with num_indices
    Shape output_shape = input.shape();
    output_shape[norm_dim] = num_indices;

    Tensor result(output_shape, input.dtype(), Device::CPU);

    T *result_data = result.typed_data<T>();

    const auto &input_shape = input.shape();
    const auto &input_strides = input.strides();
    const void *input_ptr = input.data();
    int64_t dim_size = static_cast<int64_t>(input_shape[norm_dim]);

    size_t total_size = result.size();

#ifdef AXIOM_USE_OPENMP
    if (parallel::should_parallelize(total_size)) {
        ptrdiff_t n = static_cast<ptrdiff_t>(total_size);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            auto out_coords =
                ShapeUtils::unravel_index(static_cast<size_t>(i), output_shape);
            size_t idx_pos = out_coords[norm_dim];
            int64_t idx = indices_data[idx_pos];
            if (idx < 0) {
                idx += dim_size;
            }
            out_coords[norm_dim] = static_cast<size_t>(idx);
            size_t input_offset =
                ShapeUtils::linear_index(out_coords, input_strides);
            result_data[i] = *reinterpret_cast<const T *>(
                static_cast<const uint8_t *>(input_ptr) + input_offset);
        }
        return result;
    }
#endif

    for (size_t i = 0; i < total_size; ++i) {
        auto out_coords = ShapeUtils::unravel_index(i, output_shape);

        // Get the index for this dimension
        size_t idx_pos = out_coords[norm_dim];
        int64_t idx = indices_data[idx_pos];

        // Handle negative indices
        if (idx < 0) {
            idx += dim_size;
        }

        // Build input coordinates
        out_coords[norm_dim] = static_cast<size_t>(idx);

        // Get value from input
        size_t input_offset =
            ShapeUtils::linear_index(out_coords, input_strides);
        result_data[i] = *reinterpret_cast<const T *>(
            static_cast<const uint8_t *>(input_ptr) + input_offset);
    }

    return result;
}

Tensor
CPUIndexSelectOperation::execute_index_select(const Tensor &input, int dim,
                                              const Tensor &indices) const {
    if (input.device() != Device::CPU || indices.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU IndexSelect");
    }

    if (indices.ndim() != 1) {
        throw ShapeError("index_select requires 1D indices tensor");
    }

#define DISPATCH_INDEX_SELECT(DTYPE, CTYPE)                                    \
    case DTYPE:                                                                \
        return execute_index_select_typed<CTYPE>(input, dim, indices);

    switch (input.dtype()) {
        DISPATCH_INDEX_SELECT(DType::Float16, float16_t)
        DISPATCH_INDEX_SELECT(DType::Float32, float)
        DISPATCH_INDEX_SELECT(DType::Float64, double)
        DISPATCH_INDEX_SELECT(DType::Int32, int32_t)
        DISPATCH_INDEX_SELECT(DType::Int64, int64_t)
        DISPATCH_INDEX_SELECT(DType::Int16, int16_t)
        DISPATCH_INDEX_SELECT(DType::Int8, int8_t)
        DISPATCH_INDEX_SELECT(DType::UInt8, uint8_t)
        DISPATCH_INDEX_SELECT(DType::UInt16, uint16_t)
        DISPATCH_INDEX_SELECT(DType::UInt32, uint32_t)
        DISPATCH_INDEX_SELECT(DType::UInt64, uint64_t)
        DISPATCH_INDEX_SELECT(DType::Bool, bool)
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()),
                                           "IndexSelect");
    }
#undef DISPATCH_INDEX_SELECT
}

// ============================================================================
// CPU Softmax/LogSoftmax Implementation
// ============================================================================

template <typename T>
Tensor CPUSoftmaxOperation::execute_softmax_typed(const Tensor &input,
                                                  int axis) const {
    // Normalize axis
    int norm_axis = axis;
    if (norm_axis < 0) {
        norm_axis += static_cast<int>(input.ndim());
    }

    // Create output tensor
    Tensor result(input.shape(), input.dtype(), Device::CPU);

    size_t outer_size = 1;
    for (int i = 0; i < norm_axis; ++i)
        outer_size *= input.shape()[i];

    size_t axis_size = input.shape()[norm_axis];

    size_t inner_size = 1;
    for (size_t i = norm_axis + 1; i < input.ndim(); ++i)
        inner_size *= input.shape()[i];

    const T *input_data = input.typed_data<T>();
    T *result_data = result.typed_data<T>();

#ifdef AXIOM_USE_ACCELERATE
    // Use Accelerate for contiguous softmax along the last axis (common case)
    if (input.is_contiguous() && inner_size == 1) {
        for (size_t outer = 0; outer < outer_size; ++outer) {
            const T *slice_in = input_data + outer * axis_size;
            T *slice_out = result_data + outer * axis_size;

            if constexpr (std::is_same_v<T, float>) {
                if (is_log_) {
                    accelerate::vlog_softmax_f32(slice_in, slice_out,
                                                 axis_size);
                } else {
                    accelerate::vsoftmax_f32(slice_in, slice_out, axis_size);
                }
            } else if constexpr (std::is_same_v<T, double>) {
                if (is_log_) {
                    accelerate::vlog_softmax_f64(slice_in, slice_out,
                                                 axis_size);
                } else {
                    accelerate::vsoftmax_f64(slice_in, slice_out, axis_size);
                }
            }
        }
        return result;
    }
#endif // AXIOM_USE_ACCELERATE

    // Process each softmax independently (general case)
    // Each (outer, inner) pair is independent and can be parallelized
    size_t total_softmax_ops = outer_size * inner_size;
    bool is_log = is_log_;

#ifdef AXIOM_USE_OPENMP
    if (parallel::should_parallelize(total_softmax_ops * axis_size)) {
        ptrdiff_t n = static_cast<ptrdiff_t>(total_softmax_ops);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t flat_idx = 0; flat_idx < n; ++flat_idx) {
            size_t outer = static_cast<size_t>(flat_idx) / inner_size;
            size_t inner = static_cast<size_t>(flat_idx) % inner_size;

            // Find max for numerical stability
            T max_val = std::numeric_limits<T>::lowest();
            for (size_t k = 0; k < axis_size; ++k) {
                size_t idx = (outer * axis_size + k) * inner_size + inner;
                max_val = std::max(max_val, input_data[idx]);
            }

            // Compute exp(x - max) and sum
            T sum_exp = T(0);
            for (size_t k = 0; k < axis_size; ++k) {
                size_t idx = (outer * axis_size + k) * inner_size + inner;
                T exp_val = std::exp(input_data[idx] - max_val);
                result_data[idx] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            if (is_log) {
                T log_sum = std::log(sum_exp);
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t idx = (outer * axis_size + k) * inner_size + inner;
                    result_data[idx] = input_data[idx] - max_val - log_sum;
                }
            } else {
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t idx = (outer * axis_size + k) * inner_size + inner;
                    result_data[idx] /= sum_exp;
                }
            }
        }
        return result;
    }
#endif

    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            // Find max for numerical stability
            T max_val = std::numeric_limits<T>::lowest();
            for (size_t k = 0; k < axis_size; ++k) {
                size_t idx = (outer * axis_size + k) * inner_size + inner;
                max_val = std::max(max_val, input_data[idx]);
            }

            // Compute exp(x - max) and sum
            T sum_exp = T(0);
            for (size_t k = 0; k < axis_size; ++k) {
                size_t idx = (outer * axis_size + k) * inner_size + inner;
                T exp_val = std::exp(input_data[idx] - max_val);
                result_data[idx] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize
            if (is_log) {
                // log_softmax = x - max - log(sum_exp)
                T log_sum = std::log(sum_exp);
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t idx = (outer * axis_size + k) * inner_size + inner;
                    result_data[idx] = input_data[idx] - max_val - log_sum;
                }
            } else {
                // softmax = exp(x - max) / sum_exp
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t idx = (outer * axis_size + k) * inner_size + inner;
                    result_data[idx] /= sum_exp;
                }
            }
        }
    }

    return result;
}

Tensor CPUSoftmaxOperation::execute_reduction(const Tensor &input,
                                              const std::vector<int> &axis,
                                              bool keep_dims) const {
    (void)keep_dims; // Softmax preserves shape
    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU Softmax");
    }

    int ax = axis.empty() ? -1 : axis[0];

    switch (input.dtype()) {
    case DType::Float16: {
        // For float16, compute in float32 for numerical stability, then convert
        // back
        Tensor input_f32 = input.astype(DType::Float32);
        Tensor result_f32 = execute_softmax_typed<float>(input_f32, ax);
        return result_f32.astype(DType::Float16);
    }
    case DType::Float32:
        return execute_softmax_typed<float>(input, ax);
    case DType::Float64:
        return execute_softmax_typed<double>(input, ax);
    default:
        throw TypeError::unsupported_dtype(dtype_name(input.dtype()), name());
    }
}

// ============================================================================
// Factory functions
// ============================================================================

void register_cpu_operations() {
    using namespace ops;

    // Register arithmetic operations
    OperationRegistry::register_operation(
        OpType::Add, Device::CPU,
        std::make_unique<CPUBinaryOperation<AddFunc>>(OpType::Add, "add",
                                                      AddFunc{}));

    OperationRegistry::register_operation(
        OpType::Subtract, Device::CPU,
        std::make_unique<CPUBinaryOperation<SubtractFunc>>(
            OpType::Subtract, "subtract", SubtractFunc{}));

    OperationRegistry::register_operation(
        OpType::Multiply, Device::CPU,
        std::make_unique<CPUBinaryOperation<MultiplyFunc>>(
            OpType::Multiply, "multiply", MultiplyFunc{}));

    OperationRegistry::register_operation(
        OpType::Divide, Device::CPU,
        std::make_unique<CPUBinaryOperation<DivideFunc>>(
            OpType::Divide, "divide", DivideFunc{}));

    OperationRegistry::register_operation(
        OpType::Power, Device::CPU,
        std::make_unique<CPUBinaryOperation<PowerFunc>>(OpType::Power, "power",
                                                        PowerFunc{}));

    OperationRegistry::register_operation(
        OpType::Modulo, Device::CPU,
        std::make_unique<CPUBinaryOperation<ModuloFunc>>(
            OpType::Modulo, "modulo", ModuloFunc{}));

    // Register comparison operations
    OperationRegistry::register_operation(
        OpType::Equal, Device::CPU,
        std::make_unique<CPUBinaryOperation<EqualFunc>>(OpType::Equal, "equal",
                                                        EqualFunc{}));

    OperationRegistry::register_operation(
        OpType::NotEqual, Device::CPU,
        std::make_unique<CPUBinaryOperation<NotEqualFunc>>(
            OpType::NotEqual, "not_equal", NotEqualFunc{}));

    OperationRegistry::register_operation(
        OpType::Less, Device::CPU,
        std::make_unique<CPUBinaryOperation<LessFunc>>(OpType::Less, "less",
                                                       LessFunc{}));

    OperationRegistry::register_operation(
        OpType::LessEqual, Device::CPU,
        std::make_unique<CPUBinaryOperation<LessEqualFunc>>(
            OpType::LessEqual, "less_equal", LessEqualFunc{}));

    OperationRegistry::register_operation(
        OpType::Greater, Device::CPU,
        std::make_unique<CPUBinaryOperation<GreaterFunc>>(
            OpType::Greater, "greater", GreaterFunc{}));

    OperationRegistry::register_operation(
        OpType::GreaterEqual, Device::CPU,
        std::make_unique<CPUBinaryOperation<GreaterEqualFunc>>(
            OpType::GreaterEqual, "greater_equal", GreaterEqualFunc{}));

    // Register logical operations
    OperationRegistry::register_operation(
        OpType::LogicalAnd, Device::CPU,
        std::make_unique<CPUBinaryOperation<LogicalAndFunc>>(
            OpType::LogicalAnd, "logical_and", LogicalAndFunc{}));

    OperationRegistry::register_operation(
        OpType::LogicalOr, Device::CPU,
        std::make_unique<CPUBinaryOperation<LogicalOrFunc>>(
            OpType::LogicalOr, "logical_or", LogicalOrFunc{}));

    OperationRegistry::register_operation(
        OpType::LogicalXor, Device::CPU,
        std::make_unique<CPUBinaryOperation<LogicalXorFunc>>(
            OpType::LogicalXor, "logical_xor", LogicalXorFunc{}));

    // Register bitwise operations
    OperationRegistry::register_operation(
        OpType::BitwiseAnd, Device::CPU,
        std::make_unique<CPUBinaryOperation<BitwiseAndFunc>>(
            OpType::BitwiseAnd, "bitwise_and", BitwiseAndFunc{}));

    OperationRegistry::register_operation(
        OpType::BitwiseOr, Device::CPU,
        std::make_unique<CPUBinaryOperation<BitwiseOrFunc>>(
            OpType::BitwiseOr, "bitwise_or", BitwiseOrFunc{}));

    OperationRegistry::register_operation(
        OpType::BitwiseXor, Device::CPU,
        std::make_unique<CPUBinaryOperation<BitwiseXorFunc>>(
            OpType::BitwiseXor, "bitwise_xor", BitwiseXorFunc{}));

    OperationRegistry::register_operation(
        OpType::LeftShift, Device::CPU,
        std::make_unique<CPUBinaryOperation<LeftShiftFunc>>(
            OpType::LeftShift, "left_shift", LeftShiftFunc{}));

    OperationRegistry::register_operation(
        OpType::RightShift, Device::CPU,
        std::make_unique<CPUBinaryOperation<RightShiftFunc>>(
            OpType::RightShift, "right_shift", RightShiftFunc{}));

    // Register math operations
    OperationRegistry::register_operation(
        OpType::Maximum, Device::CPU,
        std::make_unique<CPUBinaryOperation<MaximumFunc>>(
            OpType::Maximum, "maximum", MaximumFunc{}));

    OperationRegistry::register_operation(
        OpType::Minimum, Device::CPU,
        std::make_unique<CPUBinaryOperation<MinimumFunc>>(
            OpType::Minimum, "minimum", MinimumFunc{}));

    OperationRegistry::register_operation(
        OpType::Atan2, Device::CPU,
        std::make_unique<CPUBinaryOperation<Atan2Func>>(OpType::Atan2, "atan2",
                                                        Atan2Func{}));

    OperationRegistry::register_operation(
        OpType::Hypot, Device::CPU,
        std::make_unique<CPUBinaryOperation<HypotFunc>>(OpType::Hypot, "hypot",
                                                        HypotFunc{}));

    // Register unary operations
    OperationRegistry::register_operation(
        OpType::Negate, Device::CPU,
        std::make_unique<CPUUnaryOperation<NegateFunc>>(
            OpType::Negate, "negate", NegateFunc{}));
    OperationRegistry::register_operation(
        OpType::Abs, Device::CPU,
        std::make_unique<CPUUnaryOperation<AbsFunc>>(OpType::Abs, "abs",
                                                     AbsFunc{}));
    OperationRegistry::register_operation(
        OpType::Sqrt, Device::CPU,
        std::make_unique<CPUUnaryOperation<SqrtFunc>>(OpType::Sqrt, "sqrt",
                                                      SqrtFunc{}));
    OperationRegistry::register_operation(
        OpType::Exp, Device::CPU,
        std::make_unique<CPUUnaryOperation<ExpFunc>>(OpType::Exp, "exp",
                                                     ExpFunc{}));
    OperationRegistry::register_operation(
        OpType::Log, Device::CPU,
        std::make_unique<CPUUnaryOperation<LogFunc>>(OpType::Log, "log",
                                                     LogFunc{}));
    OperationRegistry::register_operation(
        OpType::Sin, Device::CPU,
        std::make_unique<CPUUnaryOperation<SinFunc>>(OpType::Sin, "sin",
                                                     SinFunc{}));
    OperationRegistry::register_operation(
        OpType::Cos, Device::CPU,
        std::make_unique<CPUUnaryOperation<CosFunc>>(OpType::Cos, "cos",
                                                     CosFunc{}));
    OperationRegistry::register_operation(
        OpType::Tan, Device::CPU,
        std::make_unique<CPUUnaryOperation<TanFunc>>(OpType::Tan, "tan",
                                                     TanFunc{}));
    OperationRegistry::register_operation(
        OpType::Erf, Device::CPU,
        std::make_unique<CPUUnaryOperation<ErfFunc>>(OpType::Erf, "erf",
                                                     ErfFunc{}));
    OperationRegistry::register_operation(
        OpType::GELU, Device::CPU,
        std::make_unique<CPUUnaryOperation<GELUFunc>>(OpType::GELU, "gelu",
                                                      GELUFunc{}));
    OperationRegistry::register_operation(
        OpType::ReLU, Device::CPU,
        std::make_unique<CPUUnaryOperation<ReLUFunc>>(OpType::ReLU, "relu",
                                                      ReLUFunc{}));
    OperationRegistry::register_operation(
        OpType::LeakyReLU, Device::CPU,
        std::make_unique<CPUUnaryOperation<LeakyReLUFunc>>(
            OpType::LeakyReLU, "leaky_relu", LeakyReLUFunc{}));
    OperationRegistry::register_operation(
        OpType::Sigmoid, Device::CPU,
        std::make_unique<CPUUnaryOperation<SigmoidFunc>>(
            OpType::Sigmoid, "sigmoid", SigmoidFunc{}));
    OperationRegistry::register_operation(
        OpType::Tanh, Device::CPU,
        std::make_unique<CPUUnaryOperation<TanhFunc>>(OpType::Tanh, "tanh",
                                                      TanhFunc{}));
    OperationRegistry::register_operation(
        OpType::SiLU, Device::CPU,
        std::make_unique<CPUUnaryOperation<SiLUFunc>>(OpType::SiLU, "silu",
                                                      SiLUFunc{}));
    OperationRegistry::register_operation(
        OpType::Conj, Device::CPU,
        std::make_unique<CPUUnaryOperation<ConjFunc>>(OpType::Conj, "conj",
                                                      ConjFunc{}));
    OperationRegistry::register_operation(
        OpType::LogicalNot, Device::CPU,
        std::make_unique<CPUUnaryOperation<LogicalNotFunc>>(
            OpType::LogicalNot, "logical_not", LogicalNotFunc{}));

    // NumPy-like math operations
    OperationRegistry::register_operation(
        OpType::Sign, Device::CPU,
        std::make_unique<CPUUnaryOperation<SignFunc>>(OpType::Sign, "sign",
                                                      SignFunc{}));
    OperationRegistry::register_operation(
        OpType::Floor, Device::CPU,
        std::make_unique<CPUUnaryOperation<FloorFunc>>(OpType::Floor, "floor",
                                                       FloorFunc{}));
    OperationRegistry::register_operation(
        OpType::Ceil, Device::CPU,
        std::make_unique<CPUUnaryOperation<CeilFunc>>(OpType::Ceil, "ceil",
                                                      CeilFunc{}));
    OperationRegistry::register_operation(
        OpType::Trunc, Device::CPU,
        std::make_unique<CPUUnaryOperation<TruncFunc>>(OpType::Trunc, "trunc",
                                                       TruncFunc{}));
    OperationRegistry::register_operation(
        OpType::Round, Device::CPU,
        std::make_unique<CPUUnaryOperation<RoundFunc>>(OpType::Round, "round",
                                                       RoundFunc{}));
    OperationRegistry::register_operation(
        OpType::Reciprocal, Device::CPU,
        std::make_unique<CPUUnaryOperation<ReciprocalFunc>>(
            OpType::Reciprocal, "reciprocal", ReciprocalFunc{}));
    OperationRegistry::register_operation(
        OpType::Square, Device::CPU,
        std::make_unique<CPUUnaryOperation<SquareFunc>>(
            OpType::Square, "square", SquareFunc{}));
    OperationRegistry::register_operation(
        OpType::Cbrt, Device::CPU,
        std::make_unique<CPUUnaryOperation<CbrtFunc>>(OpType::Cbrt, "cbrt",
                                                      CbrtFunc{}));
    OperationRegistry::register_operation(
        OpType::IsNaN, Device::CPU,
        std::make_unique<CPUUnaryOperation<IsNaNFunc>>(OpType::IsNaN, "isnan",
                                                       IsNaNFunc{}));
    OperationRegistry::register_operation(
        OpType::IsInf, Device::CPU,
        std::make_unique<CPUUnaryOperation<IsInfFunc>>(OpType::IsInf, "isinf",
                                                       IsInfFunc{}));
    OperationRegistry::register_operation(
        OpType::IsFinite, Device::CPU,
        std::make_unique<CPUUnaryOperation<IsFiniteFunc>>(
            OpType::IsFinite, "isfinite", IsFiniteFunc{}));

    // Register reduction operations
    OperationRegistry::register_operation(
        OpType::Sum, Device::CPU,
        std::make_unique<CPUReductionOperation<SumFunc>>(OpType::Sum, "sum",
                                                         SumFunc{}));
    OperationRegistry::register_operation(
        OpType::Mean, Device::CPU,
        std::make_unique<CPUReductionOperation<SumFunc>>(OpType::Mean, "mean",
                                                         SumFunc{}));
    OperationRegistry::register_operation(
        OpType::Max, Device::CPU,
        std::make_unique<CPUReductionOperation<MaxFunc>>(OpType::Max, "max",
                                                         MaxFunc{}));
    OperationRegistry::register_operation(
        OpType::Min, Device::CPU,
        std::make_unique<CPUReductionOperation<MinFunc>>(OpType::Min, "min",
                                                         MinFunc{}));
    OperationRegistry::register_operation(
        OpType::Any, Device::CPU,
        std::make_unique<CPUReductionOperation<AnyFunc>>(OpType::Any, "any",
                                                         AnyFunc{}));
    OperationRegistry::register_operation(
        OpType::All, Device::CPU,
        std::make_unique<CPUReductionOperation<AllFunc>>(OpType::All, "all",
                                                         AllFunc{}));
    OperationRegistry::register_operation(
        OpType::Prod, Device::CPU,
        std::make_unique<CPUReductionOperation<ProdFunc>>(OpType::Prod, "prod",
                                                          ProdFunc{}));

    // Register argmax/argmin operations
    OperationRegistry::register_operation(
        OpType::ArgMax, Device::CPU, std::make_unique<CPUArgMaxOperation>());
    OperationRegistry::register_operation(
        OpType::ArgMin, Device::CPU, std::make_unique<CPUArgMinOperation>());

    // Register matrix multiplication operation
    OperationRegistry::register_operation(
        OpType::MatMul, Device::CPU, std::make_unique<CPUMatMulOperation>());

    // Register where (conditional selection) operation
    OperationRegistry::register_operation(
        OpType::Where, Device::CPU, std::make_unique<CPUWhereOperation>());

    // Register masking operations
    OperationRegistry::register_operation(
        OpType::MaskedFill, Device::CPU,
        std::make_unique<CPUMaskedFillOperation>());
    OperationRegistry::register_operation(
        OpType::MaskedSelect, Device::CPU,
        std::make_unique<CPUMaskedSelectOperation>());

    // Register indexing operations
    OperationRegistry::register_operation(
        OpType::Gather, Device::CPU, std::make_unique<CPUGatherOperation>());
    OperationRegistry::register_operation(
        OpType::Scatter, Device::CPU, std::make_unique<CPUScatterOperation>());
    OperationRegistry::register_operation(
        OpType::IndexSelect, Device::CPU,
        std::make_unique<CPUIndexSelectOperation>());

    // Register softmax operations
    OperationRegistry::register_operation(
        OpType::Softmax, Device::CPU,
        std::make_unique<CPUSoftmaxOperation>(false));
    OperationRegistry::register_operation(
        OpType::LogSoftmax, Device::CPU,
        std::make_unique<CPUSoftmaxOperation>(true));
}

} // namespace cpu
} // namespace backends
} // namespace axiom