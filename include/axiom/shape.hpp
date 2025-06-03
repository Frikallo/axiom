#pragma once

#include <vector>

namespace axiom {

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

enum class MemoryOrder {
  RowMajor,  // Row-major (C-style) - default
  ColMajor   // Column-major (Fortran-style)
};

class ShapeUtils {
 public:
  static size_t size(const Shape& shape);

  static Strides calculate_strides(const Shape& shape, size_t itemsize,
                                   MemoryOrder order = MemoryOrder::RowMajor);

  static bool broadcastable(const Shape& shape1, const Shape& shape2);

  static Shape broadcast_shape(const Shape& shape1, const Shape& shape2);

  static bool is_contiguous(const Shape& shape, const Strides& strides,
                            size_t itemsize);

  static bool shapes_equal(const Shape& shape1, const Shape& shape2);

  static bool is_valid_shape(const Shape& shape);

  static size_t linear_index(const std::vector<size_t>& indices,
                             const Strides& strides);

  static std::vector<size_t> unravel_index(size_t linear_idx,
                                           const Shape& shape);
};

Shape squeeze_shape(const Shape& shape, int axis = -1);
Shape unsqueeze_shape(const Shape& shape, int axis);
Shape reshape_shape(const Shape& current_shape, const Shape& new_shape);

}  // namespace axiom