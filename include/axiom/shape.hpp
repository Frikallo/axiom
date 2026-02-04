#pragma once

#include <cstdint>
#include <vector>

#include <fxdiv.h>

namespace axiom {

using Shape = std::vector<size_t>;
using Strides = std::vector<int64_t>;

enum class MemoryOrder {
    RowMajor, // Row-major (C-style) - default
    ColMajor  // Column-major (Fortran-style)
};

// Precomputed divisors for fast index calculations using FXdiv
// Use when repeatedly dividing by the same shape dimensions (e.g., in loops)
class ShapeDivisors {
  public:
    ShapeDivisors() = default;

    explicit ShapeDivisors(const Shape &shape) : divisors_(shape.size()) {
        for (size_t i = 0; i < shape.size(); ++i) {
            divisors_[i] = fxdiv_init_size_t(shape[i]);
        }
    }

    // Fast unravel: convert linear index to multi-dimensional coordinates
    // Output written to pre-allocated indices vector (avoids allocation)
    void unravel_into(size_t linear_idx, std::vector<size_t> &indices) const {
        size_t remaining = linear_idx;
        for (int i = static_cast<int>(divisors_.size()) - 1; i >= 0; --i) {
            auto result = fxdiv_divide_size_t(remaining, divisors_[i]);
            indices[i] = result.remainder;
            remaining = result.quotient;
        }
    }

    // Fast coordinate calculation for broadcast loops
    // Computes coords and byte offsets in a single pass
    void unravel_with_offsets(size_t linear_idx, const Strides &strides,
                              int64_t &byte_offset) const {
        size_t remaining = linear_idx;
        byte_offset = 0;
        for (int i = static_cast<int>(divisors_.size()) - 1; i >= 0; --i) {
            auto result = fxdiv_divide_size_t(remaining, divisors_[i]);
            byte_offset += static_cast<int64_t>(result.remainder) * strides[i];
            remaining = result.quotient;
        }
    }

    // Compute offsets for two tensors with different strides (broadcast ops)
    void unravel_with_dual_offsets(size_t linear_idx, const Strides &strides_a,
                                   const Strides &strides_b, int64_t &offset_a,
                                   int64_t &offset_b) const {
        size_t remaining = linear_idx;
        offset_a = 0;
        offset_b = 0;
        for (int i = static_cast<int>(divisors_.size()) - 1; i >= 0; --i) {
            auto result = fxdiv_divide_size_t(remaining, divisors_[i]);
            size_t coord = result.remainder;
            offset_a += static_cast<int64_t>(coord) * strides_a[i];
            offset_b += static_cast<int64_t>(coord) * strides_b[i];
            remaining = result.quotient;
        }
    }

    size_t ndim() const { return divisors_.size(); }
    bool empty() const { return divisors_.empty(); }

    const fxdiv_divisor_size_t &operator[](size_t i) const {
        return divisors_[i];
    }

  private:
    std::vector<fxdiv_divisor_size_t> divisors_;
};

class ShapeUtils {
  public:
    static size_t size(const Shape &shape);

    static Strides
    get_contiguous_strides(const Shape &shape, size_t itemsize,
                           MemoryOrder order = MemoryOrder::RowMajor);

    static Strides calculate_strides(const Shape &shape, size_t itemsize,
                                     MemoryOrder order = MemoryOrder::RowMajor);

    static bool broadcastable(const Shape &shape1, const Shape &shape2);

    static Shape broadcast_shape(const Shape &shape1, const Shape &shape2);

    static bool shapes_equal(const Shape &shape1, const Shape &shape2);

    static bool is_valid_shape(const Shape &shape);

    static size_t linear_index(const std::vector<size_t> &indices,
                               const Strides &strides);

    // Standard unravel_index (allocates result vector)
    static std::vector<size_t> unravel_index(size_t linear_idx,
                                             const Shape &shape);

    // Fast unravel using precomputed divisors (no allocation)
    static void unravel_index_fast(size_t linear_idx,
                                   const ShapeDivisors &divisors,
                                   std::vector<size_t> &indices);
};

Shape squeeze_shape(const Shape &shape, int axis = -1);
Shape unsqueeze_shape(const Shape &shape, int axis);
Shape reshape_shape(const Shape &current_shape, const Shape &new_shape);

} // namespace axiom