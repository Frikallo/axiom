#pragma once

#include <vector>

namespace axiom {

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

// Memory layout order
enum class MemoryOrder {
    RowMajor,        // Row-major (C-style) - default
    ColMajor      // Column-major (Fortran-style)
};

// Shape utilities
class ShapeUtils {
public:
    // Calculate total number of elements from shape
    static size_t size(const Shape& shape);
    
    // Calculate strides from shape and memory order
    static Strides calculate_strides(const Shape& shape, size_t itemsize, MemoryOrder order = MemoryOrder::RowMajor);
    
    // Check if two shapes are compatible for broadcasting
    static bool broadcastable(const Shape& shape1, const Shape& shape2);
    
    // Calculate broadcasted shape
    static Shape broadcast_shape(const Shape& shape1, const Shape& shape2);
    
    // Check if strides represent a contiguous array
    static bool is_contiguous(const Shape& shape, const Strides& strides, size_t itemsize);
    
    // Check if shapes are equal
    static bool shapes_equal(const Shape& shape1, const Shape& shape2);
    
    // Validate shape (no negative dimensions, etc.)
    static bool is_valid_shape(const Shape& shape);
    
    // Calculate linear index from multi-dimensional indices
    static size_t linear_index(const std::vector<size_t>& indices, const Strides& strides);
    
    // Calculate multi-dimensional indices from linear index
    static std::vector<size_t> unravel_index(size_t linear_idx, const Shape& shape);
};

// Shape manipulation utilities
Shape squeeze_shape(const Shape& shape, int axis = -1);
Shape unsqueeze_shape(const Shape& shape, int axis);
Shape reshape_shape(const Shape& current_shape, const Shape& new_shape);

} // namespace axiom