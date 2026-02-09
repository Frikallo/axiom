#pragma once

#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "axiom/dtype.hpp"
#include "axiom/error.hpp"
#include "axiom/indexing.hpp"
#include "axiom/operations.hpp"
#include "axiom/parallel.hpp"
#include "axiom/shape.hpp"
#include "axiom/tensor.hpp"

namespace axiom {
namespace ops {

namespace detail {

// ============================================================================
// callable_traits: Deduce arity, arg types, and return type from a
// lambda/functor's operator().
// ============================================================================

template <typename T>
struct callable_traits : callable_traits<decltype(&T::operator())> {};

// Const unary
template <typename C, typename R, typename A>
struct callable_traits<R (C::*)(A) const> {
    using return_type = R;
    using arg_type = A;
    static constexpr size_t arity = 1;
};

// Mutable unary
template <typename C, typename R, typename A>
struct callable_traits<R (C::*)(A)> {
    using return_type = R;
    using arg_type = A;
    static constexpr size_t arity = 1;
};

// Const binary
template <typename C, typename R, typename A1, typename A2>
struct callable_traits<R (C::*)(A1, A2) const> {
    using return_type = R;
    using arg1_type = A1;
    using arg2_type = A2;
    static constexpr size_t arity = 2;
};

// Mutable binary
template <typename C, typename R, typename A1, typename A2>
struct callable_traits<R (C::*)(A1, A2)> {
    using return_type = R;
    using arg1_type = A1;
    using arg2_type = A2;
    static constexpr size_t arity = 2;
};

} // namespace detail

// ============================================================================
// Unary apply: apply a lambda element-wise to a tensor
// ============================================================================

template <typename Func> Tensor apply(const Tensor &input, Func &&func) {
    using traits = detail::callable_traits<std::decay_t<Func>>;
    using InputT = std::decay_t<typename traits::arg_type>;
    using OutputT = std::decay_t<typename traits::return_type>;

    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("apply (custom functor)");
    }

    // Empty tensor: return empty with correct dtype and shape
    if (input.empty()) {
        return Tensor(input.shape(), dtype_of_v<OutputT>, Device::CPU);
    }

    // Auto-cast to the lambda's expected input type
    Tensor src = input;
    if (src.dtype() != dtype_of_v<InputT>) {
        src = src.astype(dtype_of_v<InputT>);
    }

    Tensor result(src.shape(), dtype_of_v<OutputT>, Device::CPU);
    OutputT *result_data = result.typed_data<OutputT>();

    if (src.is_contiguous()) {
        const InputT *src_data = src.typed_data<InputT>();
        size_t n = src.size();

#ifdef AXIOM_USE_OPENMP
        if (parallel::should_parallelize(n)) {
            auto sz = static_cast<ptrdiff_t>(n);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < sz; ++i) {
                result_data[i] = func(src_data[i]);
            }
            return result;
        }
#endif

        for (size_t i = 0; i < n; ++i) {
            result_data[i] = func(src_data[i]);
        }
    } else {
        // Non-contiguous: stride-based iteration with coordinate
        // tracking
        size_t total = src.size();
        size_t ndim = src.ndim();
        const auto &strides = src.strides();
        const auto *base = reinterpret_cast<const uint8_t *>(src.data());
        std::vector<size_t> coords(ndim, 0);

        for (size_t i = 0; i < total; ++i) {
            int64_t byte_offset = 0;
            for (size_t d = 0; d < ndim; ++d) {
                byte_offset += static_cast<int64_t>(coords[d]) * strides[d];
            }
            const auto &val =
                *reinterpret_cast<const InputT *>(base + byte_offset);
            result_data[i] = func(val);

            // Increment coordinates
            for (int j = static_cast<int>(ndim) - 1; j >= 0; --j) {
                if (++coords[j] < src.shape()[j]) {
                    break;
                }
                coords[j] = 0;
            }
        }
    }

    return result;
}

// ============================================================================
// Binary apply: apply a lambda element-wise to two tensors with
// broadcasting
// ============================================================================

template <typename Func>
Tensor apply(const Tensor &a, const Tensor &b, Func &&func) {
    using traits = detail::callable_traits<std::decay_t<Func>>;
    using LhsT = std::decay_t<typename traits::arg1_type>;
    using RhsT = std::decay_t<typename traits::arg2_type>;
    using OutputT = std::decay_t<typename traits::return_type>;

    if (a.device() != Device::CPU) {
        throw DeviceError::cpu_only("apply (custom functor)");
    }
    if (b.device() != Device::CPU) {
        throw DeviceError::cpu_only("apply (custom functor)");
    }

    if (!are_broadcastable(a.shape(), b.shape())) {
        throw ShapeError::broadcast_incompatible(
            "cannot broadcast shapes for apply");
    }

    // Auto-cast inputs
    Tensor lhs = a;
    if (lhs.dtype() != dtype_of_v<LhsT>) {
        lhs = lhs.astype(dtype_of_v<LhsT>);
    }
    Tensor rhs = b;
    if (rhs.dtype() != dtype_of_v<RhsT>) {
        rhs = rhs.astype(dtype_of_v<RhsT>);
    }

    auto info = compute_broadcast_info(lhs.shape(), rhs.shape());
    const Shape &result_shape = info.result_shape;
    size_t total = ShapeUtils::size(result_shape);

    // Handle empty result
    if (total == 0) {
        return Tensor(result_shape, dtype_of_v<OutputT>, Device::CPU);
    }

    Tensor result(result_shape, dtype_of_v<OutputT>, Device::CPU);
    OutputT *result_data = result.typed_data<OutputT>();

    // Same-shape contiguous fast path
    if (!info.needs_broadcast && lhs.is_contiguous() && rhs.is_contiguous()) {
        const LhsT *lhs_data = lhs.typed_data<LhsT>();
        const RhsT *rhs_data = rhs.typed_data<RhsT>();

#ifdef AXIOM_USE_OPENMP
        if (parallel::should_parallelize(total)) {
            auto n = static_cast<ptrdiff_t>(total);
#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) {
                result_data[i] = func(lhs_data[i], rhs_data[i]);
            }
            return result;
        }
#endif

        for (size_t i = 0; i < total; ++i) {
            result_data[i] = func(lhs_data[i], rhs_data[i]);
        }
        return result;
    }

    // Broadcast path: stride-based element access
    size_t ndim = result_shape.size();

    Strides lhs_bcast(ndim, 0);
    Strides rhs_bcast(ndim, 0);

    int lhs_off = static_cast<int>(ndim - lhs.shape().size());
    for (size_t i = 0; i < lhs.shape().size(); ++i) {
        if (lhs.shape()[i] != 1) {
            lhs_bcast[i + lhs_off] = lhs.strides()[i];
        }
    }

    int rhs_off = static_cast<int>(ndim - rhs.shape().size());
    for (size_t i = 0; i < rhs.shape().size(); ++i) {
        if (rhs.shape()[i] != 1) {
            rhs_bcast[i + rhs_off] = rhs.strides()[i];
        }
    }

    const auto *lhs_base = reinterpret_cast<const uint8_t *>(lhs.data());
    const auto *rhs_base = reinterpret_cast<const uint8_t *>(rhs.data());

    ShapeDivisors shape_div(result_shape);

#ifdef AXIOM_USE_OPENMP
    if (parallel::should_parallelize(total)) {
        auto n = static_cast<ptrdiff_t>(total);
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            size_t remaining = static_cast<size_t>(i);
            int64_t lhs_off_b = 0;
            int64_t rhs_off_b = 0;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                auto r = fxdiv_divide_size_t(remaining, shape_div[d]);
                size_t coord = r.remainder;
                remaining = r.quotient;
                lhs_off_b += static_cast<int64_t>(coord) * lhs_bcast[d];
                rhs_off_b += static_cast<int64_t>(coord) * rhs_bcast[d];
            }
            const auto &lv =
                *reinterpret_cast<const LhsT *>(lhs_base + lhs_off_b);
            const auto &rv =
                *reinterpret_cast<const RhsT *>(rhs_base + rhs_off_b);
            result_data[i] = func(lv, rv);
        }
        return result;
    }
#endif

    // Sequential with coordinate tracking
    std::vector<size_t> coords(ndim, 0);
    for (size_t i = 0; i < total; ++i) {
        int64_t lhs_off_b = 0;
        int64_t rhs_off_b = 0;
        for (size_t d = 0; d < ndim; ++d) {
            lhs_off_b += static_cast<int64_t>(coords[d]) * lhs_bcast[d];
            rhs_off_b += static_cast<int64_t>(coords[d]) * rhs_bcast[d];
        }
        const auto &lv = *reinterpret_cast<const LhsT *>(lhs_base + lhs_off_b);
        const auto &rv = *reinterpret_cast<const RhsT *>(rhs_base + rhs_off_b);
        result_data[i] = func(lv, rv);

        for (int j = static_cast<int>(ndim) - 1; j >= 0; --j) {
            if (++coords[j] < result_shape[j]) {
                break;
            }
            coords[j] = 0;
        }
    }

    return result;
}

// ============================================================================
// vectorize: wrap a lambda into a reusable callable
// ============================================================================

template <typename Func> auto vectorize(Func &&func) {
    using traits = detail::callable_traits<std::decay_t<Func>>;

    if constexpr (traits::arity == 1) {
        return [f = std::forward<Func>(func)](const Tensor &input) {
            return apply(input, f);
        };
    } else {
        return [f = std::forward<Func>(func)](
                   const Tensor &a, const Tensor &b) { return apply(a, b, f); };
    }
}

// ============================================================================
// apply_along_axis: apply a function to 1-D slices along a given axis
// func1d takes a 1-D Tensor and returns a scalar Tensor or 1-D Tensor.
// ============================================================================

template <typename Func>
Tensor apply_along_axis(Func &&func1d, int axis, const Tensor &arr) {
    if (arr.device() != Device::CPU) {
        throw DeviceError::cpu_only("apply_along_axis");
    }

    int ndim = static_cast<int>(arr.ndim());
    if (ndim == 0) {
        throw ValueError("apply_along_axis requires at least 1-D input");
    }

    // Normalize axis
    int norm_axis = axis < 0 ? axis + ndim : axis;
    if (norm_axis < 0 || norm_axis >= ndim) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // Build the "outer" shape = all dims except the axis dim
    Shape outer_shape;
    for (int d = 0; d < ndim; ++d) {
        if (d != norm_axis) {
            outer_shape.push_back(arr.shape()[d]);
        }
    }

    // For 1-D input, just call func1d directly
    if (ndim == 1) {
        return func1d(arr);
    }

    // Probe: call func1d on the first slice to determine output shape
    // Build slices for index 0 on all outer dims
    std::vector<Slice> probe_slices(ndim);
    for (int d = 0; d < ndim; ++d) {
        if (d == norm_axis) {
            probe_slices[d] = Slice(); // full axis
        } else {
            probe_slices[d] = Slice(0, 1); // first element
        }
    }
    Tensor probe_slice = arr.slice(probe_slices);
    // Squeeze all non-axis dims to get a 1-D tensor
    // Reshape to just the axis length
    probe_slice = probe_slice.reshape({arr.shape()[norm_axis]});

    Tensor probe_result = func1d(probe_slice);

    // Determine result shape based on probe_result
    bool scalar_output =
        (probe_result.ndim() == 0 ||
         (probe_result.ndim() == 1 && probe_result.size() == 1 &&
          probe_result.shape()[0] == 1));

    // If func1d returns a scalar, result shape = outer_shape
    // If func1d returns a 1-D tensor of length M, result shape
    // replaces the axis dim with M
    Shape result_shape;
    if (scalar_output) {
        result_shape = outer_shape;
    } else {
        if (probe_result.ndim() != 1) {
            throw ValueError(
                "apply_along_axis: func1d must return a scalar or 1-D tensor");
        }
        for (int d = 0; d < ndim; ++d) {
            if (d == norm_axis) {
                result_shape.push_back(probe_result.shape()[0]);
            } else {
                result_shape.push_back(arr.shape()[d]);
            }
        }
    }

    size_t outer_total = ShapeUtils::size(outer_shape);
    size_t axis_len = arr.shape()[norm_axis];

    Tensor result(result_shape, probe_result.dtype(), Device::CPU);

    // Iterate over all outer index combinations
    std::vector<size_t> outer_coords(outer_shape.size(), 0);

    for (size_t iter = 0; iter < outer_total; ++iter) {
        // Build slice args for this outer coordinate
        std::vector<Slice> slices(ndim);
        size_t oi = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d == norm_axis) {
                slices[d] = Slice(); // full axis
            } else {
                auto c = static_cast<int64_t>(outer_coords[oi]);
                slices[d] = Slice(c, c + 1);
                ++oi;
            }
        }

        // Extract 1-D slice and reshape to (axis_len,)
        Tensor slice_1d = arr.slice(slices).reshape({axis_len});

        // Apply function
        Tensor out;
        if (iter == 0) {
            out = probe_result; // reuse the probe result
        } else {
            out = func1d(slice_1d);
        }

        // Copy result into the output tensor
        if (scalar_output) {
            // out is a scalar; place it at outer_coords in result
            if (outer_shape.empty()) {
                // result is scalar
                std::memcpy(result.data(), out.data(), result.itemsize());
            } else {
                // Build index into result
                std::vector<size_t> idx = outer_coords;
                std::memcpy(static_cast<uint8_t *>(result.data()) +
                                ShapeUtils::linear_index(idx, result.strides()),
                            out.data(), result.itemsize());
            }
        } else {
            // out is 1-D of length M; place along axis in result
            size_t out_len = out.shape()[0];
            const uint8_t *src = static_cast<const uint8_t *>(out.data());
            size_t item_sz = result.itemsize();

            for (size_t k = 0; k < out_len; ++k) {
                // Build full index into result
                std::vector<size_t> idx(ndim);
                size_t oi2 = 0;
                for (int d = 0; d < ndim; ++d) {
                    if (d == norm_axis) {
                        idx[d] = k;
                    } else {
                        idx[d] = outer_coords[oi2++];
                    }
                }
                std::memcpy(static_cast<uint8_t *>(result.data()) +
                                ShapeUtils::linear_index(idx, result.strides()),
                            src + k * item_sz, item_sz);
            }
        }

        // Increment outer coordinates
        for (int j = static_cast<int>(outer_coords.size()) - 1; j >= 0; --j) {
            if (++outer_coords[j] < outer_shape[j]) {
                break;
            }
            outer_coords[j] = 0;
        }
    }

    return result;
}

// ============================================================================
// apply_over_axes: apply a function repeatedly over multiple axes
// func takes (Tensor, int axis) and returns a tensor that is either
// the same ndim or has the reduced axis kept.
// ============================================================================

template <typename Func>
Tensor apply_over_axes(Func &&func, const Tensor &a,
                       const std::vector<int> &axes) {
    if (a.device() != Device::CPU) {
        throw DeviceError::cpu_only("apply_over_axes");
    }

    Tensor result = a;
    int ndim = static_cast<int>(a.ndim());

    for (int ax : axes) {
        int norm = ax < 0 ? ax + ndim : ax;
        if (norm < 0 || norm >= ndim) {
            throw ShapeError::invalid_axis(ax, ndim);
        }

        Tensor out = func(result, norm);

        if (out.ndim() != static_cast<size_t>(ndim)) {
            throw ValueError(
                "apply_over_axes: function did not preserve ndim (got " +
                std::to_string(out.ndim()) + ", expected " +
                std::to_string(ndim) + ")");
        }

        result = out;
    }

    return result;
}

// ============================================================================
// fromfunc: alias for vectorize
// Wraps a scalar function into a callable that operates on tensors.
// ============================================================================

template <typename Func> auto fromfunc(Func &&func) {
    return vectorize(std::forward<Func>(func));
}

} // namespace ops
} // namespace axiom
