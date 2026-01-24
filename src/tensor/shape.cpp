#include "axiom/shape.hpp"

#include "axiom/error.hpp"
#include <algorithm>
#include <numeric>

namespace axiom {

size_t ShapeUtils::size(const Shape& shape) {
  if (shape.empty()) return 1;
  return std::accumulate(shape.begin(), shape.end(), size_t(1),
                         std::multiplies<size_t>());
}

Strides ShapeUtils::calculate_strides(const Shape& shape, size_t itemsize,
                                      MemoryOrder order) {
  if (shape.empty()) return {};

  Strides strides(shape.size());

  if (order == MemoryOrder::RowMajor) {
    // Row-major (C-style): last dimension has stride of itemsize
    strides.back() = itemsize;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  } else {
    // Column-major (Fortran-style): first dimension has stride of itemsize
    strides[0] = itemsize;
    for (size_t i = 1; i < shape.size(); ++i) {
      strides[i] = strides[i - 1] * shape[i - 1];
    }
  }

  return strides;
}

Strides ShapeUtils::get_contiguous_strides(const Shape& shape, size_t itemsize, MemoryOrder order) {
    return calculate_strides(shape, itemsize, order);
}

bool ShapeUtils::broadcastable(const Shape& shape1, const Shape& shape2) {
  size_t ndim1 = shape1.size();
  size_t ndim2 = shape2.size();
  size_t max_ndim = std::max(ndim1, ndim2);

  for (size_t i = 0; i < max_ndim; ++i) {
    size_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
    size_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return false;
    }
  }

  return true;
}

Shape ShapeUtils::broadcast_shape(const Shape& shape1, const Shape& shape2) {
  if (!broadcastable(shape1, shape2)) {
    throw ShapeError::broadcast_incompatible("shapes not broadcastable");
  }

  size_t ndim1 = shape1.size();
  size_t ndim2 = shape2.size();
  size_t max_ndim = std::max(ndim1, ndim2);

  Shape result(max_ndim);

  for (size_t i = 0; i < max_ndim; ++i) {
    size_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
    size_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;

    result[max_ndim - 1 - i] = std::max(dim1, dim2);
  }

  return result;
}

bool ShapeUtils::shapes_equal(const Shape& shape1, const Shape& shape2) {
  return shape1 == shape2;
}

bool ShapeUtils::is_valid_shape(const Shape& shape) {
  // All dimensions must be non-negative (size_t is unsigned, so this is
  // automatically true) Just check for reasonable size to avoid overflow
  return size(shape) <= SIZE_MAX;
}

size_t ShapeUtils::linear_index(const std::vector<size_t>& indices,
                                const Strides& strides) {
  if (indices.size() != strides.size()) {
    throw ShapeError("Number of indices (" + std::to_string(indices.size()) + 
                     ") must match number of dimensions (" + std::to_string(strides.size()) + ")");
  }

  size_t linear_idx = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    linear_idx += indices[i] * strides[i];
  }

  return linear_idx;
}

std::vector<size_t> ShapeUtils::unravel_index(size_t linear_idx,
                                              const Shape& shape) {
  std::vector<size_t> indices(shape.size());

  for (int i = shape.size() - 1; i >= 0; --i) {
    indices[i] = linear_idx % shape[i];
    linear_idx /= shape[i];
  }

  return indices;
}

// Shape manipulation utilities
Shape squeeze_shape(const Shape& shape, int axis) {
  Shape result;

  if (axis == -1) {
    // Remove all dimensions of size 1
    for (size_t dim : shape) {
      if (dim != 1) {
        result.push_back(dim);
      }
    }
  } else {
    // Remove specific axis if it has size 1
    if (axis < 0) axis += shape.size();
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
      throw ShapeError::invalid_axis(axis, shape.size());
    }
    if (shape[axis] != 1) {
      throw ShapeError("Cannot squeeze dimension " + std::to_string(axis) + 
                       " that has size " + std::to_string(shape[axis]) + " (not 1)");
    }

    for (size_t i = 0; i < shape.size(); ++i) {
      if (static_cast<int>(i) != axis) {
        result.push_back(shape[i]);
      }
    }
  }

  return result;
}

Shape unsqueeze_shape(const Shape& shape, int axis) {
  Shape result = shape;

  if (axis < 0) axis += shape.size() + 1;
  if (axis < 0 || axis > static_cast<int>(shape.size())) {
    throw ShapeError("Axis " + std::to_string(axis) + " out of bounds for unsqueeze (shape has " + 
                     std::to_string(shape.size()) + " dimensions)");
  }

  result.insert(result.begin() + axis, 1);
  return result;
}

Shape reshape_shape(const Shape& current_shape, const Shape& new_shape) {
  size_t current_size = ShapeUtils::size(current_shape);
  size_t new_size = ShapeUtils::size(new_shape);

  // Handle -1 in new_shape (infer dimension)
  Shape result = new_shape;
  int infer_idx = -1;
  for (size_t i = 0; i < result.size(); ++i) {
    if (static_cast<int>(result[i]) == -1) {
      if (infer_idx != -1) {
        throw ShapeError("Only one dimension can be inferred in reshape");
      }
      infer_idx = i;
    }
  }

  if (infer_idx != -1) {
    size_t known_size = 1;
    for (size_t i = 0; i < result.size(); ++i) {
      if (static_cast<int>(i) != infer_idx) {
        known_size *= result[i];
      }
    }

    if (current_size % known_size != 0) {
      throw ShapeError::invalid_reshape(current_size, known_size);
    }

    result[infer_idx] = current_size / known_size;
    new_size = current_size;
  }

  if (current_size != new_size) {
    throw ShapeError::invalid_reshape(current_size, new_size);
  }

  return result;
}

}  // namespace axiom