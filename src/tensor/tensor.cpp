#include "axiom/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

#include "axiom/einops.hpp"
#include "axiom/error.hpp"
#include "axiom/io/io.hpp"
#include "axiom/numeric.hpp"
#include "axiom/operations.hpp"
#include "axiom/random.hpp"
#include "axiom/system.hpp"

namespace axiom {

template <typename T> std::string vec_to_string(const std::vector<T> &vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i < vec.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

size_t Tensor::calculate_storage_size() const { return size() * itemsize(); }

void Tensor::validate_indices(const std::vector<size_t> &indices) const {
    if (indices.size() != ndim()) {
        throw ShapeError(
            "Number of indices (" + std::to_string(indices.size()) +
            ") must match tensor dimensions (" + std::to_string(ndim()) + ")");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw IndexError::out_of_bounds(indices[i], shape_[i], i);
        }
    }
}

void Tensor::update_contiguity_flags() {
    if (shape_.empty()) {
        flags_.c_contiguous = true;
        flags_.f_contiguous = true;
        return;
    }

    // Tensors with negative strides are never contiguous
    for (int64_t s : strides_) {
        if (s < 0) {
            flags_.c_contiguous = false;
            flags_.f_contiguous = false;
            return;
        }
    }

    auto c_strides = ShapeUtils::calculate_strides(shape_, itemsize(),
                                                   MemoryOrder::RowMajor);
    flags_.c_contiguous = (strides_ == c_strides);

    auto f_strides = ShapeUtils::calculate_strides(shape_, itemsize(),
                                                   MemoryOrder::ColMajor);
    flags_.f_contiguous = (strides_ == f_strides);
}

Tensor::Tensor()
    : storage_(nullptr), shape_(), strides_(), dtype_(DType::Float32),
      offset_(0), flags_(), memory_order_(MemoryOrder::RowMajor) {
    flags_.owndata = false;
}

Tensor::Tensor(const Shape &shape, DType dtype, Device device,
               MemoryOrder order)
    : shape_(shape), dtype_(dtype), offset_(0), flags_(), memory_order_(order) {
    if (!ShapeUtils::is_valid_shape(shape_)) {
        throw ShapeError("Invalid shape: " + vec_to_string(shape));
    }

    strides_ = ShapeUtils::calculate_strides(shape_, dtype_size(dtype_), order);
    storage_ = make_storage(calculate_storage_size(), device);

    update_contiguity_flags();
    flags_.owndata = true;
}

Tensor::Tensor(std::initializer_list<size_t> shape, DType dtype, Device device,
               MemoryOrder order)
    : Tensor(Shape(shape), dtype, device, order) {}

Tensor::Tensor(std::shared_ptr<Storage> storage, const Shape &shape,
               const Strides &strides, DType dtype, size_t offset,
               MemoryOrder order)
    : storage_(storage), shape_(shape), strides_(strides), dtype_(dtype),
      offset_(offset), flags_(), memory_order_(order) {
    if (!storage_) {
        throw MemoryError("Storage cannot be null");
    }

    if (shape_.size() != strides_.size()) {
        throw ShapeError("Shape and strides must have same length: shape has " +
                         std::to_string(shape_.size()) +
                         " dimensions but strides has " +
                         std::to_string(strides_.size()));
    }

    size_t required_size = 0;
    if (!shape_.empty()) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] > 1) {
                // Use absolute value of stride for size calculation
                required_size = std::max(
                    required_size, static_cast<size_t>((shape_[i] - 1) *
                                                       std::abs(strides_[i])));
            }
        }
        required_size += dtype_size(dtype_);
    }

    if (offset_ + required_size > storage_->size_bytes()) {
        throw MemoryError::storage_too_small(offset_ + required_size,
                                             storage_->size_bytes());
    }

    update_contiguity_flags();
    flags_.owndata = (offset_ == 0);
}

Tensor::Tensor(const Tensor &other)
    : storage_(other.storage_), shape_(other.shape_), strides_(other.strides_),
      dtype_(other.dtype_), offset_(other.offset_), flags_(other.flags_),
      memory_order_(other.memory_order_) {}

Tensor &Tensor::operator=(const Tensor &other) {
    if (this != &other) {
        storage_ = other.storage_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        dtype_ = other.dtype_;
        offset_ = other.offset_;
        flags_ = other.flags_;
        memory_order_ = other.memory_order_;
    }
    return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : storage_(std::move(other.storage_)), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), dtype_(other.dtype_),
      offset_(other.offset_), flags_(other.flags_),
      memory_order_(other.memory_order_) {}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        storage_ = std::move(other.storage_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        offset_ = other.offset_;
        flags_ = other.flags_;
        memory_order_ = other.memory_order_;
    }
    return *this;
}

void *Tensor::data() {
    if (!storage_) {
        return nullptr;
    }
    // For negative strides, adjust base pointer to account for flipped view
    // The data pointer should point to the "first" element in iteration order
    size_t adjustment = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (strides_[i] < 0 && shape_[i] > 1) {
            adjustment += static_cast<size_t>((shape_[i] - 1) * (-strides_[i]));
        }
    }
    return static_cast<uint8_t *>(storage_->data()) + offset_ + adjustment;
}

const void *Tensor::data() const {
    if (!storage_) {
        return nullptr;
    }
    // For negative strides, adjust base pointer to account for flipped view
    // The data pointer should point to the "first" element in iteration order
    size_t adjustment = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (strides_[i] < 0 && shape_[i] > 1) {
            adjustment += static_cast<size_t>((shape_[i] - 1) * (-strides_[i]));
        }
    }
    return static_cast<const uint8_t *>(storage_->data()) + offset_ +
           adjustment;
}

Tensor Tensor::slice(const std::vector<Slice> &slice_args) const {
    if (slice_args.size() > ndim()) {
        throw IndexError("Too many indices for tensor: got " +
                         std::to_string(slice_args.size()) +
                         " but tensor has " + std::to_string(ndim()) +
                         " dimensions");
    }

    Shape new_shape;
    Strides new_strides;
    std::vector<size_t> start_indices;

    int current_dim = 0;
    for (const auto &arg : slice_args) {
        int64_t dim_size = shape_[current_dim];

        // Normalize start
        int64_t start = arg.start.value_or(0);
        if (start < 0)
            start += dim_size;
        start = std::max((int64_t)0, std::min(start, dim_size));

        // Normalize stop
        int64_t stop = arg.stop.value_or(dim_size);
        if (arg.stop && stop < 0)
            stop += dim_size;
        stop = std::max((int64_t)0, std::min(stop, dim_size));

        // Normalize step
        int64_t step = arg.step.value_or(1);
        if (step == 0)
            throw IndexError::invalid_slice("step cannot be zero");

        start_indices.push_back(start);

        if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
            new_shape.push_back(0);
            new_strides.push_back(0);
        } else {
            new_shape.push_back((stop - start + (step > 0 ? step : -step) - 1) /
                                std::abs(step));
            new_strides.push_back(strides_[current_dim] * step);
        }
        current_dim++;
    }

    start_indices.resize(ndim(), 0);
    size_t new_offset =
        offset_ + ShapeUtils::linear_index(start_indices, strides_);

    // Copy remaining dims
    for (size_t i = current_dim; i < ndim(); ++i) {
        new_shape.push_back(shape_[i]);
        new_strides.push_back(strides_[i]);
    }

    return Tensor(storage_, new_shape, new_strides, dtype_, new_offset,
                  memory_order_);
}

Tensor Tensor::operator[](std::initializer_list<Index> indices) const {
    if (indices.size() > ndim()) {
        throw IndexError("Too many indices for tensor: got " +
                         std::to_string(indices.size()) + " but tensor has " +
                         std::to_string(ndim()) + " dimensions");
    }

    // Check for TensorIndex (boolean mask or integer indices)
    for (const auto &index : indices) {
        if (std::holds_alternative<TensorIndex>(index)) {
            // If we have a TensorIndex, treat it as boolean masking
            const auto &tensor_idx = std::get<TensorIndex>(index);
            if (tensor_idx.indices) {
                // Check if it's a boolean mask or integer indices
                if (tensor_idx.indices->dtype() == DType::Bool) {
                    // Boolean masking - return selected elements
                    return masked_select(*tensor_idx.indices);
                } else {
                    // Integer indices - use gather on dimension 0
                    return gather(0, *tensor_idx.indices);
                }
            }
        }
    }

    // Regular indexing path
    std::vector<Slice> slice_args;
    std::vector<int> dims_to_squeeze;

    int current_dim = 0;
    for (const auto &index : indices) {
        std::visit(
            [&](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int64_t>) {
                    // Convert integer index to a slice of size 1
                    if (arg >= 0) {
                        slice_args.emplace_back(arg, arg + 1, 1);
                    } else {
                        // For negative indices, let slice handle the end
                        // boundary
                        slice_args.emplace_back(arg, std::nullopt, 1);
                    }
                    dims_to_squeeze.push_back(current_dim);
                } else if constexpr (std::is_same_v<T, Slice>) {
                    slice_args.push_back(arg);
                } else if constexpr (std::is_same_v<T, TensorIndex>) {
                    // Already handled above
                }
            },
            index);
        current_dim++;
    }

    // Call the main slice method
    Tensor sliced_view = this->slice(slice_args);

    // Squeeze the dimensions that were indexed by an integer
    // We must squeeze from the largest index to smallest to avoid shifting
    // subsequent indices.
    std::sort(dims_to_squeeze.rbegin(), dims_to_squeeze.rend());
    for (int dim : dims_to_squeeze) {
        if (sliced_view.shape()[dim] == 1) {
            sliced_view = sliced_view.squeeze(dim);
        }
    }

    return sliced_view;
}

void recursive_copy(uint8_t *dst, const uint8_t *src, const Shape &shape,
                    const Strides &dst_strides, const Strides &src_strides,
                    size_t itemsize, int dim) {
    if (dim == static_cast<int>(shape.size()) - 1) {
        for (size_t i = 0; i < shape[dim]; ++i) {
            std::memcpy(dst + i * dst_strides[dim], src + i * src_strides[dim],
                        itemsize);
        }
    } else {
        for (size_t i = 0; i < shape[dim]; ++i) {
            recursive_copy(dst + i * dst_strides[dim],
                           src + i * src_strides[dim], shape, dst_strides,
                           src_strides, itemsize, dim + 1);
        }
    }
}

Tensor Tensor::ascontiguousarray() const {
    if (is_c_contiguous()) {
        return *this;
    }

    auto new_tensor = Tensor(shape_, dtype_, device(), MemoryOrder::RowMajor);

    if (device() == Device::CPU) {
        recursive_copy(static_cast<uint8_t *>(new_tensor.data()),
                       static_cast<const uint8_t *>(this->data()), shape_,
                       new_tensor.strides(), this->strides(), itemsize(), 0);
    } else {
        new_tensor.storage_->copy_from(*storage_);
    }

    return new_tensor;
}

Tensor Tensor::asfortranarray() const {
    if (is_f_contiguous()) {
        return *this;
    }

    auto new_tensor = Tensor(shape_, dtype_, device(), MemoryOrder::ColMajor);

    if (device() == Device::CPU) {
        recursive_copy(static_cast<uint8_t *>(new_tensor.data()),
                       static_cast<const uint8_t *>(this->data()), shape_,
                       new_tensor.strides(), this->strides(), itemsize(), 0);
    } else {
        new_tensor.storage_->copy_from(*storage_);
    }

    return new_tensor;
}

Tensor Tensor::reshape(const Shape &new_shape, MemoryOrder order) const {
    Shape validated_shape = reshape_shape(shape_, new_shape);

    bool can_view = false;
    if (order == memory_order_) {
        if ((order == MemoryOrder::RowMajor && is_c_contiguous()) ||
            (order == MemoryOrder::ColMajor && is_f_contiguous())) {
            can_view = true;
        }
    }

    if (can_view) {
        Strides new_strides =
            ShapeUtils::calculate_strides(validated_shape, itemsize(), order);
        return Tensor(storage_, validated_shape, new_strides, dtype_, offset_,
                      order);
    } else {
        auto new_tensor = Tensor(validated_shape, dtype_, device(), order);
        if (device() == Device::CPU) {
            for (size_t i = 0; i < size(); ++i) {
                auto indices = ShapeUtils::unravel_index(i, shape_);
                size_t src_byte_offset =
                    ShapeUtils::linear_index(indices, strides_);
                auto new_indices =
                    ShapeUtils::unravel_index(i, validated_shape);
                size_t dst_byte_offset =
                    ShapeUtils::linear_index(new_indices, new_tensor.strides_);

                std::memcpy(
                    static_cast<uint8_t *>(new_tensor.storage_->data()) +
                        dst_byte_offset,
                    static_cast<const uint8_t *>(storage_->data()) + offset_ +
                        src_byte_offset,
                    itemsize());
            }
        } else {
            new_tensor.storage_->copy_from(*storage_);
        }
        return new_tensor;
    }
}

Tensor Tensor::reshape(std::initializer_list<size_t> new_shape,
                       MemoryOrder order) const {
    return reshape(Shape(new_shape), order);
}

Tensor
Tensor::rearrange(const std::string &pattern,
                  const std::map<std::string, size_t> &axis_sizes) const {
    return einops::rearrange(*this, pattern, axis_sizes);
}

Tensor Tensor::reduce(const std::string &pattern, const std::string &reduction,
                      const std::map<std::string, size_t> &axis_sizes) const {
    return einops::reduce(*this, pattern, reduction, axis_sizes);
}

Tensor Tensor::transpose() const {
    if (ndim() < 2) {
        return *this;
    }

    Shape new_shape = shape_;
    Strides new_strides = strides_;

    std::swap(new_shape[ndim() - 2], new_shape[ndim() - 1]);
    std::swap(new_strides[ndim() - 2], new_strides[ndim() - 1]);

    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::transpose(const std::vector<int> &axes) const {
    if (axes.size() != ndim()) {
        throw ShapeError("Number of axes (" + std::to_string(axes.size()) +
                         ") must match tensor dimensions (" +
                         std::to_string(ndim()) + ")");
    }

    Shape new_shape(ndim());
    Strides new_strides(ndim());

    for (size_t i = 0; i < axes.size(); ++i) {
        int axis = axes[i];
        if (axis < 0)
            axis += ndim();
        if (axis < 0 || axis >= static_cast<int>(ndim())) {
            throw ShapeError::invalid_axis(axis, ndim());
        }

        new_shape[i] = shape_[axis];
        new_strides[i] = strides_[axis];
    }

    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::squeeze(int axis) const {
    Shape new_shape;
    Strides new_strides;

    if (axis == -1) {
        // Squeeze all dimensions of size 1
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] != 1) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
    } else {
        int real_axis = axis < 0 ? axis + ndim() : axis;
        if (real_axis < 0 || real_axis >= (int)ndim()) {
            throw ShapeError::invalid_axis(real_axis, ndim());
        }
        if (shape_[real_axis] != 1) {
            return *this; // It's a no-op
        }
        for (size_t i = 0; i < shape_.size(); ++i) {
            if ((int)i != real_axis) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
    }

    // If the tensor becomes a scalar
    if (new_shape.empty() && !shape_.empty()) {
        return Tensor(storage_, {}, {}, dtype_, offset_);
    }

    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::unsqueeze(int axis) const {
    Shape new_shape = unsqueeze_shape(shape_, axis);

    Strides new_strides = strides_;
    if (axis < 0)
        axis += shape_.size() + 1;
    new_strides.insert(new_strides.begin() + axis, 0);

    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::view(const Shape &new_shape) const {
    if (ShapeUtils::size(new_shape) != size()) {
        throw ShapeError::invalid_reshape(size(), ShapeUtils::size(new_shape));
    }

    if (!is_contiguous()) {
        throw MemoryError::not_contiguous("view");
    }

    Strides new_strides = ShapeUtils::calculate_strides(new_shape, itemsize(),
                                                        MemoryOrder::RowMajor);
    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    // Normalize negative indices
    int ndims = static_cast<int>(ndim());
    if (start_dim < 0)
        start_dim += ndims;
    if (end_dim < 0)
        end_dim += ndims;

    if (start_dim < 0 || start_dim >= ndims || end_dim < 0 ||
        end_dim >= ndims || start_dim > end_dim) {
        throw ShapeError(
            "Invalid flatten dimensions: start_dim=" +
            std::to_string(start_dim) + ", end_dim=" + std::to_string(end_dim) +
            " for tensor with " + std::to_string(ndims) + " dimensions");
    }

    // Calculate new shape
    Shape new_shape;
    size_t flattened_size = 1;

    for (int i = 0; i < start_dim; ++i) {
        new_shape.push_back(shape_[i]);
    }

    for (int i = start_dim; i <= end_dim; ++i) {
        flattened_size *= shape_[i];
    }
    new_shape.push_back(flattened_size);

    for (int i = end_dim + 1; i < ndims; ++i) {
        new_shape.push_back(shape_[i]);
    }

    // Check if we can return a view (must be contiguous in flattened region)
    bool can_view = true;
    if (end_dim > start_dim) {
        // Check that strides are contiguous in the flattened region
        // Also cannot create view if any stride is negative
        for (int i = start_dim; i < end_dim; ++i) {
            if (strides_[i] < 0 || strides_[i + 1] < 0 ||
                strides_[i] !=
                    static_cast<int64_t>(shape_[i + 1]) * strides_[i + 1]) {
                can_view = false;
                break;
            }
        }
    }

    if (can_view) {
        // Create view with new strides
        Strides new_strides;
        for (int i = 0; i < start_dim; ++i) {
            new_strides.push_back(strides_[i]);
        }
        new_strides.push_back(strides_[end_dim]); // Stride of flattened dim
        for (int i = end_dim + 1; i < ndims; ++i) {
            new_strides.push_back(strides_[i]);
        }
        return Tensor(storage_, new_shape, new_strides, dtype_, offset_,
                      memory_order_);
    } else {
        // Need to copy
        return ascontiguousarray().reshape(new_shape);
    }
}

// ============================================================================
// NumPy-like aliases and view operations
// ============================================================================

Tensor Tensor::negative() const { return -(*this); }

Tensor Tensor::flipud() const { return flip(0); }

Tensor Tensor::fliplr() const { return flip(1); }

Tensor Tensor::swapaxes(int axis1, int axis2) const {
    int ndims = static_cast<int>(ndim());
    // Normalize negative axes
    if (axis1 < 0)
        axis1 += ndims;
    if (axis2 < 0)
        axis2 += ndims;

    if (axis1 < 0 || axis1 >= ndims || axis2 < 0 || axis2 >= ndims) {
        throw ShapeError("swapaxes: axis out of range for tensor with " +
                         std::to_string(ndims) + " dimensions");
    }

    if (axis1 == axis2) {
        return *this;
    }

    // Build permutation
    std::vector<int> axes(ndims);
    for (int i = 0; i < ndims; ++i) {
        axes[i] = i;
    }
    std::swap(axes[axis1], axes[axis2]);

    return transpose(axes);
}

Tensor Tensor::moveaxis(int source, int destination) const {
    int ndims = static_cast<int>(ndim());
    // Normalize negative axes
    if (source < 0)
        source += ndims;
    if (destination < 0)
        destination += ndims;

    if (source < 0 || source >= ndims || destination < 0 ||
        destination >= ndims) {
        throw ShapeError("moveaxis: axis out of range for tensor with " +
                         std::to_string(ndims) + " dimensions");
    }

    if (source == destination) {
        return *this;
    }

    // Build permutation: remove source, insert at destination
    std::vector<int> axes;
    for (int i = 0; i < ndims; ++i) {
        if (i != source) {
            axes.push_back(i);
        }
    }
    axes.insert(axes.begin() + destination, source);

    return transpose(axes);
}

Tensor Tensor::flip(int axis) const { return flip(std::vector<int>{axis}); }

Tensor Tensor::flip(const std::vector<int> &axes) const {
    int ndims = static_cast<int>(ndim());

    // Normalize and validate axes
    std::vector<int> norm_axes;
    for (int ax : axes) {
        int norm = ax < 0 ? ax + ndims : ax;
        if (norm < 0 || norm >= ndims) {
            throw ShapeError::invalid_axis(ax, ndim());
        }
        norm_axes.push_back(norm);
    }

    // Zero-copy flip: negate strides for flipped axes
    // The data() method handles the pointer adjustment for negative strides
    Strides new_strides = strides_;
    for (int ax : norm_axes) {
        new_strides[ax] = -new_strides[ax];
    }

    return Tensor(storage_, shape_, new_strides, dtype_, offset_,
                  memory_order_);
}

Tensor Tensor::rot90(int k, const std::vector<int> &axes) const {
    if (axes.size() != 2) {
        throw ShapeError("rot90 requires exactly 2 axes");
    }

    int ndims = static_cast<int>(ndim());
    int ax0 = axes[0] < 0 ? axes[0] + ndims : axes[0];
    int ax1 = axes[1] < 0 ? axes[1] + ndims : axes[1];

    if (ax0 < 0 || ax0 >= ndims || ax1 < 0 || ax1 >= ndims) {
        throw ShapeError("rot90: axes out of range");
    }
    if (ax0 == ax1) {
        throw ShapeError("rot90: axes must be different");
    }

    // Normalize k to [0, 3]
    k = ((k % 4) + 4) % 4;

    if (k == 0) {
        return *this;
    }

    Tensor result = *this;
    for (int i = 0; i < k; ++i) {
        // One 90-degree rotation = flip along ax1, then swap axes
        result = result.flip(ax1).swapaxes(ax0, ax1);
    }

    return result;
}

Tensor Tensor::roll(int64_t shift, int axis) const {
    if (axis == -1) {
        // Roll over flattened tensor, then reshape back
        auto flat = flatten();
        auto rolled = flat.roll(shift, 0);
        return rolled.reshape(shape_);
    }

    int ndims = static_cast<int>(ndim());
    int norm_axis = axis < 0 ? axis + ndims : axis;
    if (norm_axis < 0 || norm_axis >= ndims) {
        throw ShapeError::invalid_axis(axis, ndim());
    }

    int64_t axis_size = static_cast<int64_t>(shape_[norm_axis]);
    if (axis_size == 0)
        return *this;

    // Normalize shift to [0, axis_size)
    shift = ((shift % axis_size) + axis_size) % axis_size;
    if (shift == 0)
        return *this;

    Tensor result(shape_, dtype_, device(), memory_order_);

    if (device() == Device::CPU) {
        std::vector<size_t> coords(ndim(), 0);
        for (size_t i = 0; i < size(); ++i) {
            // Calculate rolled source coordinate
            std::vector<size_t> src_coords = coords;
            int64_t src_idx = static_cast<int64_t>(coords[norm_axis]) - shift;
            if (src_idx < 0)
                src_idx += axis_size;
            src_coords[norm_axis] = static_cast<size_t>(src_idx);

            size_t src_offset = ShapeUtils::linear_index(src_coords, strides_);
            size_t dst_offset =
                ShapeUtils::linear_index(coords, result.strides());

            std::memcpy(static_cast<uint8_t *>(result.data()) + dst_offset,
                        static_cast<const uint8_t *>(data()) + src_offset,
                        itemsize());

            // Increment coordinates
            for (int j = static_cast<int>(ndim()) - 1; j >= 0; --j) {
                if (++coords[j] < shape_[j])
                    break;
                coords[j] = 0;
            }
        }
    } else {
        auto cpu_rolled = cpu().roll(shift, axis);
        result = cpu_rolled.to(device());
    }

    return result;
}

Tensor Tensor::roll(const std::vector<int64_t> &shifts,
                    const std::vector<int> &axes) const {
    if (shifts.size() != axes.size()) {
        throw ValueError("roll: shifts and axes must have same length");
    }

    Tensor result = *this;
    for (size_t i = 0; i < shifts.size(); ++i) {
        result = result.roll(shifts[i], axes[i]);
    }
    return result;
}

Tensor Tensor::diagonal(int offset, int axis1, int axis2) const {
    int ndims = static_cast<int>(ndim());
    if (ndims < 2) {
        throw ShapeError("diagonal requires tensor with at least 2 dimensions");
    }

    // Normalize axes
    if (axis1 < 0)
        axis1 += ndims;
    if (axis2 < 0)
        axis2 += ndims;

    if (axis1 < 0 || axis1 >= ndims || axis2 < 0 || axis2 >= ndims) {
        throw ShapeError("diagonal: axes out of range");
    }
    if (axis1 == axis2) {
        throw ShapeError("diagonal: axis1 and axis2 must be different");
    }

    // Make axis1 < axis2 for simplicity
    if (axis1 > axis2) {
        std::swap(axis1, axis2);
        offset = -offset;
    }

    int64_t n1 = static_cast<int64_t>(shape_[axis1]);
    int64_t n2 = static_cast<int64_t>(shape_[axis2]);

    // Calculate diagonal length
    int64_t diag_size;
    if (offset >= 0) {
        diag_size = std::max(int64_t(0), std::min(n1, n2 - offset));
    } else {
        diag_size = std::max(int64_t(0), std::min(n1 + offset, n2));
    }

    // Build output shape: remove axis1 and axis2, add diagonal dimension at end
    Shape out_shape;
    for (int i = 0; i < ndims; ++i) {
        if (i != axis1 && i != axis2) {
            out_shape.push_back(shape_[i]);
        }
    }
    out_shape.push_back(static_cast<size_t>(diag_size));

    Tensor result(out_shape, dtype_, device(), memory_order_);

    if (diag_size == 0) {
        return result;
    }

    if (device() == Device::CPU) {
        size_t out_size = result.size();
        std::vector<size_t> out_coords(result.ndim(), 0);

        for (size_t i = 0; i < out_size; ++i) {
            // Build input coordinates from output coordinates
            std::vector<size_t> in_coords;
            size_t out_idx = 0;
            for (int j = 0; j < ndims; ++j) {
                if (j == axis1 || j == axis2) {
                    in_coords.push_back(0); // Placeholder
                } else {
                    in_coords.push_back(out_coords[out_idx++]);
                }
            }

            // Diagonal index is the last dimension of output
            size_t diag_idx = out_coords[result.ndim() - 1];
            if (offset >= 0) {
                in_coords[axis1] = diag_idx;
                in_coords[axis2] = diag_idx + offset;
            } else {
                in_coords[axis1] = diag_idx - offset;
                in_coords[axis2] = diag_idx;
            }

            size_t src_offset = ShapeUtils::linear_index(in_coords, strides_);
            size_t dst_offset =
                ShapeUtils::linear_index(out_coords, result.strides());

            std::memcpy(static_cast<uint8_t *>(result.data()) + dst_offset,
                        static_cast<const uint8_t *>(data()) + src_offset,
                        itemsize());

            // Increment output coordinates
            for (int j = static_cast<int>(result.ndim()) - 1; j >= 0; --j) {
                if (++out_coords[j] < result.shape()[j])
                    break;
                out_coords[j] = 0;
            }
        }
    } else {
        auto cpu_diag = cpu().diagonal(offset, axis1, axis2);
        result = cpu_diag.to(device());
    }

    return result;
}

Tensor Tensor::trace(int offset, int axis1, int axis2) const {
    return diagonal(offset, axis1, axis2).sum(-1);
}

Tensor Tensor::expand(const Shape &new_shape) const {
    // expand creates a view by setting stride to 0 for expanded dimensions
    // Only dimensions of size 1 can be expanded

    if (new_shape.size() < shape_.size()) {
        throw ShapeError(
            "expand: new shape must have at least as many dimensions (got " +
            std::to_string(new_shape.size()) + " but tensor has " +
            std::to_string(shape_.size()) + ")");
    }

    size_t dim_diff = new_shape.size() - shape_.size();

    // Validate and compute new strides
    Strides new_strides(new_shape.size(), 0);

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i < dim_diff) {
            // New leading dimensions - stride is 0 (broadcast)
            if (new_shape[i] == 0) {
                throw ShapeError("expand: cannot expand to size 0");
            }
            new_strides[i] = 0;
        } else {
            size_t old_idx = i - dim_diff;
            size_t old_size = shape_[old_idx];
            size_t new_size = new_shape[i];

            if (old_size == new_size) {
                // Same size - keep stride
                new_strides[i] = strides_[old_idx];
            } else if (old_size == 1) {
                // Expanding from 1 - set stride to 0
                new_strides[i] = 0;
            } else if (new_size == old_size) {
                new_strides[i] = strides_[old_idx];
            } else {
                throw ShapeError(
                    "expand: can only expand dimensions of size 1, got size " +
                    std::to_string(old_size) + " at dimension " +
                    std::to_string(old_idx));
            }
        }
    }

    return Tensor(storage_, new_shape, new_strides, dtype_, offset_,
                  memory_order_);
}

Tensor Tensor::repeat(const std::vector<size_t> &repeats) const {
    // repeat actually copies data
    // Each dimension is repeated by the corresponding factor

    if (repeats.size() != shape_.size()) {
        throw ShapeError("repeat: number of repeat values (" +
                         std::to_string(repeats.size()) +
                         ") must match tensor dimensions (" +
                         std::to_string(shape_.size()) + ")");
    }

    // Calculate new shape
    Shape new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
        new_shape[i] = shape_[i] * repeats[i];
    }

    // Create output tensor
    Tensor result(new_shape, dtype_, device(), memory_order_);

    if (device() == Device::CPU) {
        // For CPU, copy data with proper repetition
        size_t total_elements = result.size();
        const uint8_t *src_base = static_cast<const uint8_t *>(data());
        uint8_t *dst_base = static_cast<uint8_t *>(result.data());

        std::vector<size_t> dst_coords(new_shape.size(), 0);

        for (size_t i = 0; i < total_elements; ++i) {
            // Map dst coords to src coords using modulo
            std::vector<size_t> src_coords(shape_.size());
            for (size_t d = 0; d < shape_.size(); ++d) {
                src_coords[d] = dst_coords[d] % shape_[d];
            }

            // Calculate byte offsets
            size_t src_offset = ShapeUtils::linear_index(src_coords, strides_);
            size_t dst_offset =
                ShapeUtils::linear_index(dst_coords, result.strides());

            // Copy element
            std::memcpy(dst_base + dst_offset, src_base + src_offset,
                        itemsize());

            // Increment dst coords
            for (int d = static_cast<int>(new_shape.size()) - 1; d >= 0; --d) {
                if (++dst_coords[d] < new_shape[d])
                    break;
                dst_coords[d] = 0;
            }
        }
    } else {
        // For GPU, we need a kernel (for now, go through CPU)
        auto cpu_result = cpu().repeat(repeats);
        result = cpu_result.to(device());
    }

    return result;
}

Tensor Tensor::matmul(const Tensor &other, bool transpose_self,
                      bool transpose_other) const {
    return ops::matmul(*this, other, transpose_self, transpose_other);
}

// ============================================================================
// Conditional and masking operations (fluent API)
// ============================================================================

Tensor Tensor::where(const Tensor &condition, const Tensor &other) const {
    return ops::where(condition, *this, other);
}

Tensor Tensor::where(const Tensor &condition, float other) const {
    auto other_tensor = Tensor::full(shape_, other, device());
    return ops::where(condition, *this, other_tensor);
}

Tensor Tensor::where(const Tensor &condition, double other) const {
    auto other_tensor =
        Tensor::full(shape_, static_cast<float>(other), device());
    return ops::where(condition, *this, other_tensor);
}

Tensor Tensor::where(const Tensor &condition, int32_t other) const {
    auto other_tensor =
        Tensor::full(shape_, static_cast<float>(other), device());
    return ops::where(condition, *this, other_tensor);
}

Tensor Tensor::masked_fill(const Tensor &mask, float value) const {
    return ops::masked_fill(*this, mask, value);
}

Tensor Tensor::masked_fill(const Tensor &mask, double value) const {
    return ops::masked_fill(*this, mask, static_cast<float>(value));
}

Tensor Tensor::masked_fill(const Tensor &mask, int32_t value) const {
    return ops::masked_fill(*this, mask, static_cast<float>(value));
}

Tensor Tensor::masked_fill(const Tensor &mask, const Tensor &value) const {
    return ops::masked_fill(*this, mask, value);
}

Tensor &Tensor::masked_fill_(const Tensor &mask, float value) {
    *this = masked_fill(mask, value);
    return *this;
}

Tensor &Tensor::masked_fill_(const Tensor &mask, double value) {
    *this = masked_fill(mask, value);
    return *this;
}

Tensor &Tensor::masked_fill_(const Tensor &mask, int32_t value) {
    *this = masked_fill(mask, value);
    return *this;
}

Tensor Tensor::masked_select(const Tensor &mask) const {
    return ops::masked_select(*this, mask);
}

// ============================================================================
// Indexing operations (fluent API)
// ============================================================================

Tensor Tensor::gather(int dim, const Tensor &indices) const {
    return ops::gather(*this, dim, indices);
}

Tensor Tensor::scatter(int dim, const Tensor &indices,
                       const Tensor &src) const {
    return ops::scatter(*this, dim, indices, src);
}

Tensor &Tensor::scatter_(int dim, const Tensor &indices, const Tensor &src) {
    *this = scatter(dim, indices, src);
    return *this;
}

Tensor Tensor::index_select(int dim, const Tensor &indices) const {
    return ops::index_select(*this, dim, indices);
}

Tensor Tensor::sum(int axis, bool keep_dims) const {
    if (axis == -1) {
        return ops::sum(*this, {}, keep_dims);
    }
    return ops::sum(*this, {axis}, keep_dims);
}

Tensor Tensor::sum(const std::vector<int> &axes, bool keep_dims) const {
    return ops::sum(*this, axes, keep_dims);
}

Tensor Tensor::mean(int axis, bool keep_dims) const {
    if (axis == -1) {
        return ops::mean(*this, {}, keep_dims);
    }
    return ops::mean(*this, {axis}, keep_dims);
}

Tensor Tensor::mean(const std::vector<int> &axes, bool keep_dims) const {
    return ops::mean(*this, axes, keep_dims);
}

Tensor Tensor::max(int axis, bool keep_dims) const {
    if (axis == -1) {
        return ops::max(*this, {}, keep_dims);
    }
    return ops::max(*this, {axis}, keep_dims);
}

Tensor Tensor::min(int axis, bool keep_dims) const {
    if (axis == -1) {
        return ops::min(*this, {}, keep_dims);
    }
    return ops::min(*this, {axis}, keep_dims);
}

Tensor Tensor::argmax(int axis, bool keep_dims) const {
    return ops::argmax(*this, axis, keep_dims);
}

Tensor Tensor::argmin(int axis, bool keep_dims) const {
    return ops::argmin(*this, axis, keep_dims);
}

// Additional reduction member functions
Tensor Tensor::prod(int axis, bool keep_dims) const {
    if (axis == -1) {
        return ops::prod(*this, {}, keep_dims);
    }
    return ops::prod(*this, {axis}, keep_dims);
}

Tensor Tensor::prod(const std::vector<int> &axes, bool keep_dims) const {
    return ops::prod(*this, axes, keep_dims);
}

Tensor Tensor::any(int axis, bool keep_dims) const {
    if (axis == -1) {
        return ops::any(*this, {}, keep_dims);
    }
    return ops::any(*this, {axis}, keep_dims);
}

Tensor Tensor::any(const std::vector<int> &axes, bool keep_dims) const {
    return ops::any(*this, axes, keep_dims);
}

Tensor Tensor::all(int axis, bool keep_dims) const {
    if (axis == -1) {
        return ops::all(*this, {}, keep_dims);
    }
    return ops::all(*this, {axis}, keep_dims);
}

Tensor Tensor::all(const std::vector<int> &axes, bool keep_dims) const {
    return ops::all(*this, axes, keep_dims);
}

// Statistical operations (composition-based)
Tensor Tensor::var(int axis, int ddof, bool keep_dims) const {
    if (axis == -1) {
        return var(std::vector<int>{}, ddof, keep_dims);
    }
    return var(std::vector<int>{axis}, ddof, keep_dims);
}

Tensor Tensor::var(const std::vector<int> &axes, int ddof,
                   bool keep_dims) const {
    // var = mean((x - mean(x))^2) with ddof correction
    auto m = mean(axes.empty() ? std::vector<int>{} : axes, true);
    auto centered = ops::subtract(*this, m);
    auto sq = ops::square(centered);
    auto sum_sq = sq.sum(axes.empty() ? std::vector<int>{} : axes, keep_dims);

    // Calculate reduction size
    size_t n = 1;
    if (axes.empty()) {
        n = size();
    } else {
        for (int ax : axes) {
            int norm_ax = ax < 0 ? ax + static_cast<int>(ndim()) : ax;
            n *= shape_[norm_ax];
        }
    }

    auto divisor =
        Tensor::full({1}, static_cast<float>(n - ddof), sum_sq.device());
    return ops::divide(sum_sq, divisor);
}

Tensor Tensor::std(int axis, int ddof, bool keep_dims) const {
    return var(axis, ddof, keep_dims).sqrt();
}

Tensor Tensor::std(const std::vector<int> &axes, int ddof,
                   bool keep_dims) const {
    return var(axes, ddof, keep_dims).sqrt();
}

Tensor Tensor::ptp(int axis, bool keep_dims) const {
    // ptp = max - min (peak-to-peak)
    return ops::subtract(max(axis, keep_dims), min(axis, keep_dims));
}

// Unary math operations
Tensor Tensor::abs() const { return ops::abs(*this); }
Tensor Tensor::sqrt() const { return ops::sqrt(*this); }
Tensor Tensor::exp() const { return ops::exp(*this); }
Tensor Tensor::log() const { return ops::log(*this); }
Tensor Tensor::sin() const { return ops::sin(*this); }
Tensor Tensor::cos() const { return ops::cos(*this); }
Tensor Tensor::tan() const { return ops::tan(*this); }

// NumPy-like math operations
Tensor Tensor::sign() const { return ops::sign(*this); }
Tensor Tensor::floor() const { return ops::floor(*this); }
Tensor Tensor::ceil() const { return ops::ceil(*this); }
Tensor Tensor::trunc() const { return ops::trunc(*this); }
Tensor Tensor::round(int decimals) const { return ops::round(*this, decimals); }
Tensor Tensor::reciprocal() const { return ops::reciprocal(*this); }
Tensor Tensor::square() const { return ops::square(*this); }
Tensor Tensor::cbrt() const { return ops::cbrt(*this); }

// Element-wise testing operations
Tensor Tensor::isnan() const { return ops::isnan(*this); }
Tensor Tensor::isinf() const { return ops::isinf(*this); }
Tensor Tensor::isfinite() const { return ops::isfinite(*this); }

// Clipping operations
Tensor Tensor::clip(const Tensor &min_val, const Tensor &max_val) const {
    return ops::clip(*this, min_val, max_val);
}

Tensor Tensor::clip(double min_val, double max_val) const {
    auto min_tensor = Tensor::full({1}, static_cast<float>(min_val), device());
    auto max_tensor = Tensor::full({1}, static_cast<float>(max_val), device());
    return ops::clip(*this, min_tensor, max_tensor);
}

// Activation operations
Tensor Tensor::relu() const { return ops::relu(*this); }
Tensor Tensor::leaky_relu(float negative_slope) const {
    return ops::leaky_relu(*this, negative_slope);
}
Tensor Tensor::gelu() const { return ops::gelu(*this); }
Tensor Tensor::silu() const { return ops::silu(*this); }
Tensor Tensor::sigmoid() const { return ops::sigmoid(*this); }
Tensor Tensor::tanh() const { return ops::tanh(*this); }
Tensor Tensor::softmax(int axis) const { return ops::softmax(*this, axis); }
Tensor Tensor::log_softmax(int axis) const {
    return ops::log_softmax(*this, axis);
}

Tensor Tensor::copy(MemoryOrder order) const {
    auto new_tensor = Tensor(shape_, dtype_, device(), order);

    if (device() == Device::CPU && order != memory_order_) {
        for (size_t i = 0; i < size(); ++i) {
            auto indices = ShapeUtils::unravel_index(i, shape_);
            size_t src_byte_offset =
                ShapeUtils::linear_index(indices, strides_);
            size_t dst_byte_offset =
                ShapeUtils::linear_index(indices, new_tensor.strides_);

            std::memcpy(static_cast<uint8_t *>(new_tensor.storage_->data()) +
                            dst_byte_offset,
                        static_cast<const uint8_t *>(storage_->data()) +
                            offset_ + src_byte_offset,
                        itemsize());
        }
    } else {
        new_tensor.storage_->copy_from(*storage_);
    }

    return new_tensor;
}

Tensor Tensor::to(Device target_device, MemoryOrder order) const {
    if (device() == target_device && order == memory_order_) {
        return *this;
    }

    auto new_tensor = Tensor(shape_, dtype_, target_device, order);

    if (order != memory_order_ && device() == Device::CPU &&
        target_device == Device::CPU) {
        for (size_t i = 0; i < size(); ++i) {
            auto indices = ShapeUtils::unravel_index(i, shape_);
            size_t src_byte_offset =
                ShapeUtils::linear_index(indices, strides_);
            size_t dst_byte_offset =
                ShapeUtils::linear_index(indices, new_tensor.strides_);

            std::memcpy(static_cast<uint8_t *>(new_tensor.storage_->data()) +
                            dst_byte_offset,
                        static_cast<const uint8_t *>(storage_->data()) +
                            offset_ + src_byte_offset,
                        itemsize());
        }
    } else {
        new_tensor.storage_->copy_from(*storage_);
    }

    return new_tensor;
}

Tensor Tensor::cpu() const { return to(Device::CPU, memory_order_); }

Tensor Tensor::gpu() const { return to(Device::GPU, memory_order_); }

Tensor Tensor::astype(DType new_dtype) const {
    if (new_dtype == dtype_) {
        return *this;
    }

    auto new_tensor = Tensor(shape_, new_dtype, device(), memory_order_);

    if (device() == Device::CPU && new_tensor.device() == Device::CPU) {
        if (is_contiguous() && new_tensor.is_contiguous()) {
            type_conversion::convert_dtype(new_tensor.data(), data(), size(),
                                           new_dtype, dtype_);
        } else {
            type_conversion::convert_dtype_strided(
                new_tensor.data(), data(), shape_, new_tensor.strides(),
                strides_, new_dtype, dtype_, 0, offset_);
        }
    } else {
        auto cpu_source = (device() == Device::CPU) ? *this : this->cpu();
        auto cpu_target = Tensor(shape_, new_dtype, Device::CPU, memory_order_);

        if (cpu_source.is_contiguous() && cpu_target.is_contiguous()) {
            type_conversion::convert_dtype(cpu_target.data(), cpu_source.data(),
                                           size(), new_dtype, dtype_);
        } else {
            type_conversion::convert_dtype_strided(
                cpu_target.data(), cpu_source.data(), shape_,
                cpu_target.strides(), cpu_source.strides(), new_dtype, dtype_,
                0, cpu_source.offset_);
        }

        if (new_tensor.device() != Device::CPU) {
            new_tensor.storage_->copy_from(*cpu_target.storage_);
        } else {
            new_tensor = cpu_target;
        }
    }

    return new_tensor;
}

Tensor Tensor::astype_safe(DType new_dtype) const {
    if (type_conversion::conversion_may_lose_precision(dtype_, new_dtype)) {
        throw TypeError::conversion_not_safe(dtype_name(),
                                             axiom::dtype_name(new_dtype));
    }
    return astype(new_dtype);
}

std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=" << dtype_name() << ", device=";
    oss << (device() == Device::CPU ? "CPU" : "GPU");
    if (shape_.size() > 0) {
        oss << ", order="
            << (memory_order_ == MemoryOrder::RowMajor ? "RowMajor"
                                                       : "ColMajor");
    }
    oss << ")";
    return oss.str();
}

std::string Tensor::str() const { return repr(); }

bool Tensor::same_shape(const Tensor &other) const {
    return ShapeUtils::shapes_equal(shape_, other.shape_);
}

bool Tensor::same_dtype(const Tensor &other) const {
    return dtype_ == other.dtype_;
}

bool Tensor::same_device(const Tensor &other) const {
    return device() == other.device();
}

bool Tensor::same_memory_order(const Tensor &other) const {
    return memory_order_ == other.memory_order_;
}

// Comparison/testing methods (NumPy-like)
Tensor Tensor::isclose(const Tensor &other, double rtol, double atol) const {
    // |a - b| <= atol + rtol * |b|
    auto diff = ops::subtract(*this, other).abs();
    auto atol_tensor = Tensor::full({1}, static_cast<float>(atol), device());
    auto rtol_tensor = Tensor::full({1}, static_cast<float>(rtol), device());
    auto threshold =
        ops::add(atol_tensor, ops::multiply(rtol_tensor, other.abs()));
    return ops::less_equal(diff, threshold);
}

bool Tensor::allclose(const Tensor &other, double rtol, double atol) const {
    auto close = isclose(other, rtol, atol);
    auto all_result = close.all();
    return all_result.item<bool>();
}

bool Tensor::array_equal(const Tensor &other) const {
    if (!same_shape(other)) {
        return false;
    }
    auto eq = ops::equal(*this, other);
    auto all_result = eq.all();
    return all_result.item<bool>();
}

Tensor Tensor::zeros(const Shape &shape, DType dtype, Device device,
                     MemoryOrder order) {
    // Always create and initialize on CPU first, then transfer to target device
    auto tensor = Tensor(shape, dtype, Device::CPU, order);
    std::memset(tensor.data(), 0, tensor.nbytes());
    if (device == Device::GPU) {
        return tensor.to(device, order);
    }
    return tensor;
}

Tensor Tensor::zeros(std::initializer_list<size_t> shape, DType dtype,
                     Device device, MemoryOrder order) {
    return zeros(Shape(shape), dtype, device, order);
}

Tensor Tensor::ones(const Shape &shape, DType dtype, Device device,
                    MemoryOrder order) {
    // Always create and initialize on CPU first, then transfer to target device
    auto tensor = Tensor(shape, dtype, Device::CPU, order);
    switch (dtype) {
    case DType::Bool:
        tensor.fill<bool>(true);
        break;
    case DType::Int8:
        tensor.fill<int8_t>(1);
        break;
    case DType::Int16:
        tensor.fill<int16_t>(1);
        break;
    case DType::Int32:
        tensor.fill<int32_t>(1);
        break;
    case DType::Int64:
        tensor.fill<int64_t>(1);
        break;
    case DType::UInt8:
        tensor.fill<uint8_t>(1);
        break;
    case DType::UInt16:
        tensor.fill<uint16_t>(1);
        break;
    case DType::UInt32:
        tensor.fill<uint32_t>(1);
        break;
    case DType::UInt64:
        tensor.fill<uint64_t>(1);
        break;
    case DType::Float16:
        tensor.fill<float16_t>(float16_t(1.0f));
        break;
    case DType::Float32:
        tensor.fill<float>(1.0f);
        break;
    case DType::Float64:
        tensor.fill<double>(1.0);
        break;
    case DType::Complex64:
        tensor.fill<complex64_t>(complex64_t(1.0f, 0.0f));
        break;
    case DType::Complex128:
        tensor.fill<complex128_t>(complex128_t(1.0, 0.0));
        break;
    }
    if (device == Device::GPU) {
        return tensor.to(device, order);
    }
    return tensor;
}

Tensor Tensor::ones(std::initializer_list<size_t> shape, DType dtype,
                    Device device, MemoryOrder order) {
    return ones(Shape(shape), dtype, device, order);
}

Tensor Tensor::empty(const Shape &shape, DType dtype, Device device,
                     MemoryOrder order) {
    return Tensor(shape, dtype, device, order);
}

Tensor Tensor::empty(std::initializer_list<size_t> shape, DType dtype,
                     Device device, MemoryOrder order) {
    return empty(Shape(shape), dtype, device, order);
}

Tensor Tensor::eye(size_t n, DType dtype, Device device, MemoryOrder order) {
    auto tensor = zeros({n, n}, dtype, device, order);

    if (device == Device::CPU) {
        switch (dtype) {
        case DType::Bool:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<bool>({i, i}, true);
            break;
        case DType::Int8:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<int8_t>({i, i}, 1);
            break;
        case DType::Int16:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<int16_t>({i, i}, 1);
            break;
        case DType::Int32:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<int32_t>({i, i}, 1);
            break;
        case DType::Int64:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<int64_t>({i, i}, 1);
            break;
        case DType::UInt8:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<uint8_t>({i, i}, 1);
            break;
        case DType::UInt16:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<uint16_t>({i, i}, 1);
            break;
        case DType::UInt32:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<uint32_t>({i, i}, 1);
            break;
        case DType::UInt64:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<uint64_t>({i, i}, 1);
            break;
        case DType::Float16:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<float16_t>({i, i}, float16_t(1.0f));
            break;
        case DType::Float32:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<float>({i, i}, 1.0f);
            break;
        case DType::Float64:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<double>({i, i}, 1.0);
            break;
        case DType::Complex64:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<complex64_t>({i, i}, complex64_t(1.0f, 0.0f));
            break;
        case DType::Complex128:
            for (size_t i = 0; i < n; ++i)
                tensor.set_item<complex128_t>({i, i}, complex128_t(1.0, 0.0));
            break;
        }
    }

    return tensor;
}

Tensor Tensor::identity(size_t n, DType dtype, Device device,
                        MemoryOrder order) {
    return eye(n, dtype, device, order);
}

Tensor Tensor::arange(int64_t start, int64_t end, int64_t step, DType dtype,
                      Device device) {
    if (step == 0) {
        throw ValueError("Step cannot be zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        return Tensor::empty({0}, dtype, device);
    }
    size_t size = (end - start + step + (step > 0 ? -1 : 1)) / step;
    Tensor t({size}, dtype, device);

    // This implementation is for CPU only for now.
    if (device != Device::CPU) {
        throw DeviceError::cpu_only("arange");
    }

    switch (dtype) {
    case DType::Float32: {
        auto *data = t.typed_data<float>();
        for (size_t i = 0; i < size; ++i)
            data[i] = start + i * step;
        break;
    }
    case DType::Int32: {
        auto *data = t.typed_data<int32_t>();
        for (size_t i = 0; i < size; ++i)
            data[i] = start + i * step;
        break;
    }
    case DType::Int64: {
        auto *data = t.typed_data<int64_t>();
        for (size_t i = 0; i < size; ++i)
            data[i] = start + i * step;
        break;
    }
    // Add other types as needed
    default:
        throw TypeError::unsupported_dtype(axiom::dtype_name(dtype), "arange");
    }

    return t;
}

Tensor Tensor::arange(int64_t end, DType dtype, Device device) {
    return arange(0, end, 1, dtype, device);
}

void Tensor::manual_seed(uint64_t seed) { axiom::manual_seed(seed); }

Tensor Tensor::randn(const Shape &shape, DType dtype, Device device,
                     MemoryOrder order) {
    // Always create and initialize on CPU first, then transfer to target device
    auto tensor = Tensor(shape, dtype, Device::CPU, order);
    auto &rng = RandomGenerator::instance();

    switch (dtype) {
    case DType::Float32: {
        float *data = tensor.typed_data<float>();
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = rng.normal<float>();
        }
        break;
    }
    case DType::Float64: {
        double *data = tensor.typed_data<double>();
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = rng.normal<double>();
        }
        break;
    }
    case DType::Float16: {
        float16_t *data = tensor.typed_data<float16_t>();
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = float16_t(rng.normal<float>());
        }
        break;
    }
    default:
        throw TypeError("randn only supports floating point types, got " +
                        axiom::dtype_name(dtype));
    }

    if (device == Device::GPU) {
        return tensor.to(device, order);
    }
    return tensor;
}

// ============================================================================
// File I/O method implementations
// ============================================================================

void Tensor::save(const std::string &filename) const {
    io::save(*this, filename);
}

Tensor Tensor::load(const std::string &filename, Device device) {
    return io::load(filename, device);
}

void Tensor::save_tensors(const std::map<std::string, Tensor> &tensors,
                          const std::string &filename) {
    io::save_archive(tensors, filename);
}

std::map<std::string, Tensor> Tensor::load_tensors(const std::string &filename,
                                                   Device device) {
    return io::load_archive(filename, device);
}

std::vector<std::string>
Tensor::list_tensors_in_archive(const std::string &filename) {
    return io::flatbuffers::list_archive(filename);
}

Tensor Tensor::load_tensor_from_archive(const std::string &filename,
                                        const std::string &tensor_name,
                                        Device device) {
    return io::flatbuffers::load_from_archive(filename, tensor_name, device);
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << io::to_string(tensor);
    return os;
}

Tensor operator-(const Tensor &tensor) { return ops::negate(tensor); }

// ============================================================================
// View/materialization introspection
// ============================================================================

bool Tensor::has_zero_stride() const {
    for (int64_t s : strides_) {
        if (s == 0)
            return true;
    }
    return false;
}

bool Tensor::has_negative_stride() const {
    for (int64_t s : strides_) {
        if (s < 0)
            return true;
    }
    return false;
}

bool Tensor::would_materialize_on_reshape(const Shape &new_shape) const {
    if (ShapeUtils::size(new_shape) != size())
        return true;

    bool can_view = false;
    if ((memory_order_ == MemoryOrder::RowMajor && is_c_contiguous()) ||
        (memory_order_ == MemoryOrder::ColMajor && is_f_contiguous())) {
        can_view = true;
    }
    return !can_view;
}

// ============================================================================
// Safety rails
// ============================================================================

bool Tensor::has_nan() const {
    if (device() != Device::CPU) {
        return cpu().has_nan();
    }

    // Check floating point types for NaN
    if (!is_floating_dtype(dtype_) && !is_complex_dtype(dtype_))
        return false;

    auto check_nan = [this]<typename T>() {
        const T *data = typed_data<T>();
        for (size_t i = 0; i < size(); ++i) {
            if (std::isnan(static_cast<double>(data[i])))
                return true;
        }
        return false;
    };

    // Complex types need to check both real and imaginary parts
    auto check_nan_complex = [this]<typename T>() {
        const T *data = typed_data<T>();
        for (size_t i = 0; i < size(); ++i) {
            if (std::isnan(data[i].real()) || std::isnan(data[i].imag()))
                return true;
        }
        return false;
    };

    switch (dtype_) {
    case DType::Float16:
        return check_nan.template operator()<float16_t>();
    case DType::Float32:
        return check_nan.template operator()<float>();
    case DType::Float64:
        return check_nan.template operator()<double>();
    case DType::Complex64:
        return check_nan_complex.template operator()<complex64_t>();
    case DType::Complex128:
        return check_nan_complex.template operator()<complex128_t>();
    default:
        return false;
    }
}

bool Tensor::has_inf() const {
    if (device() != Device::CPU) {
        return cpu().has_inf();
    }

    // Check floating point types for Inf
    if (!is_floating_dtype(dtype_) && !is_complex_dtype(dtype_))
        return false;

    auto check_inf = [this]<typename T>() {
        const T *data = typed_data<T>();
        for (size_t i = 0; i < size(); ++i) {
            if (std::isinf(static_cast<double>(data[i])))
                return true;
        }
        return false;
    };

    // Complex types need to check both real and imaginary parts
    auto check_inf_complex = [this]<typename T>() {
        const T *data = typed_data<T>();
        for (size_t i = 0; i < size(); ++i) {
            if (std::isinf(data[i].real()) || std::isinf(data[i].imag()))
                return true;
        }
        return false;
    };

    switch (dtype_) {
    case DType::Float16:
        return check_inf.template operator()<float16_t>();
    case DType::Float32:
        return check_inf.template operator()<float>();
    case DType::Float64:
        return check_inf.template operator()<double>();
    case DType::Complex64:
        return check_inf_complex.template operator()<complex64_t>();
    case DType::Complex128:
        return check_inf_complex.template operator()<complex128_t>();
    default:
        return false;
    }
}

Tensor &Tensor::nan_guard() {
    if (has_nan()) {
        throw ValueError::nan_detected(repr());
    }
    return *this;
}

Tensor &Tensor::assert_finite() {
    if (has_nan()) {
        throw ValueError::nan_detected(repr());
    }
    if (has_inf()) {
        throw ValueError::inf_detected(repr());
    }
    return *this;
}

Tensor &Tensor::assert_shape(const Shape &expected) {
    if (shape_ != expected) {
        throw ShapeError::mismatch(expected, shape_);
    }
    return *this;
}

Tensor &Tensor::assert_shape(const std::string &pattern) {
    std::vector<std::string> tokens;
    std::istringstream iss(pattern);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    if (tokens.size() != ndim()) {
        throw ShapeError(
            "pattern '" + pattern + "' has " + std::to_string(tokens.size()) +
            " dimensions but tensor has " + std::to_string(ndim()));
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        try {
            size_t expected = std::stoull(tokens[i]);
            if (shape_[i] != expected) {
                throw ShapeError("dimension " + std::to_string(i) +
                                 " expected " + std::to_string(expected) +
                                 " but got " + std::to_string(shape_[i]));
            }
        } catch (const std::invalid_argument &) {
            // Named dimension, just check that it exists (any size is ok)
        }
    }
    return *this;
}

std::string Tensor::debug_info() const {
    std::ostringstream oss;
    oss << "Tensor Debug Info:\n";
    oss << "  Shape: " << vec_to_string(shape_) << "\n";
    oss << "  Strides: " << vec_to_string(strides_) << "\n";
    oss << "  DType: " << dtype_name() << "\n";
    oss << "  Device: " << (device() == Device::CPU ? "CPU" : "GPU") << "\n";
    oss << "  Size: " << size() << " elements, " << nbytes() << " bytes\n";
    oss << "  Memory order: "
        << (memory_order_ == MemoryOrder::RowMajor ? "RowMajor" : "ColMajor")
        << "\n";
    oss << "  Contiguous: " << (is_contiguous() ? "yes" : "no") << "\n";
    oss << "  Is view: " << (is_view() ? "yes" : "no") << "\n";
    oss << "  Owns data: " << (owns_data() ? "yes" : "no") << "\n";
    oss << "  Has zero stride: " << (has_zero_stride() ? "yes" : "no") << "\n";
    oss << "  Has negative stride: " << (has_negative_stride() ? "yes" : "no")
        << "\n";
    oss << "  Storage offset: " << offset_ << " bytes\n";
    if (is_floating_dtype(dtype_) && device() == Device::CPU && size() > 0) {
        oss << "  Has NaN: " << (has_nan() ? "yes" : "no") << "\n";
        oss << "  Has Inf: " << (has_inf() ? "yes" : "no") << "\n";
    }
    return oss.str();
}

// ============================================================================
// Complex number operations
// ============================================================================

Tensor Tensor::real() const {
    if (!is_complex_dtype(dtype_)) {
        throw TypeError("real() requires complex tensor, got " + dtype_name());
    }

    // Complex64 (8 bytes) -> Float32 (4 bytes)
    // Complex128 (16 bytes) -> Float64 (8 bytes)
    DType base_dtype =
        (dtype_ == DType::Complex64) ? DType::Float32 : DType::Float64;

    // Strides stay the same - we still step between complex elements,
    // just interpreting the real part (first half) of each
    // Layout: [real0][imag0][real1][imag1]... stride moves us to next complex
    // Real part is at the same offset (complex is laid out as [real, imag])
    return Tensor(storage_, shape_, strides_, base_dtype, offset_,
                  memory_order_);
}

Tensor Tensor::imag() const {
    if (!is_complex_dtype(dtype_)) {
        throw TypeError("imag() requires complex tensor, got " + dtype_name());
    }

    // Complex64 (8 bytes) -> Float32 (4 bytes)
    // Complex128 (16 bytes) -> Float64 (8 bytes)
    DType base_dtype =
        (dtype_ == DType::Complex64) ? DType::Float32 : DType::Float64;
    size_t base_size = dtype_size(base_dtype);

    // Strides stay the same - we still step between complex elements,
    // just interpreting the imag part (second half) of each
    // Layout: [real0][imag0][real1][imag1]... stride moves us to next complex
    // Imag part is offset by the size of the base type (to skip the real part)
    return Tensor(storage_, shape_, strides_, base_dtype, offset_ + base_size,
                  memory_order_);
}

Tensor Tensor::conj() const {
    if (!is_complex_dtype(dtype_)) {
        throw TypeError("conj() requires complex tensor, got " + dtype_name());
    }

    return ops::conj(*this);
}

// ============================================================================
// Stacking and Concatenation Operations
// ============================================================================

Tensor Tensor::concatenate(const std::vector<Tensor> &tensors, int axis) {
    if (tensors.empty()) {
        throw ValueError("concatenate requires at least one tensor");
    }

    const Tensor &first = tensors[0];
    int ndim = static_cast<int>(first.ndim());

    // Normalize axis
    int norm_axis = axis < 0 ? axis + ndim : axis;
    if (norm_axis < 0 || norm_axis >= ndim) {
        throw ValueError("axis " + std::to_string(axis) +
                         " out of bounds for tensor of dimension " +
                         std::to_string(ndim));
    }

    // Validate all tensors have compatible shapes
    DType result_dtype = first.dtype();
    Device result_device = first.device();

    for (size_t i = 1; i < tensors.size(); ++i) {
        const Tensor &t = tensors[i];
        if (t.ndim() != first.ndim()) {
            throw ShapeError(
                "concatenate: all tensors must have same number of dimensions");
        }
        for (int d = 0; d < ndim; ++d) {
            if (d != norm_axis && t.shape()[d] != first.shape()[d]) {
                throw ShapeError(
                    "concatenate: shapes don't match on dimension " +
                    std::to_string(d));
            }
        }
        // Promote dtype if needed
        result_dtype = ops::promote_types(result_dtype, t.dtype());
        // Use GPU if any tensor is on GPU
        if (t.device() == Device::GPU) {
            result_device = Device::GPU;
        }
    }

    // Calculate output shape
    Shape result_shape = first.shape();
    size_t total_concat_dim = 0;
    for (const auto &t : tensors) {
        total_concat_dim += t.shape()[norm_axis];
    }
    result_shape[norm_axis] = total_concat_dim;

    // Create result tensor
    Tensor result(result_shape, result_dtype, result_device);

    // Copy data from each tensor
    size_t concat_offset = 0;
    for (const auto &t : tensors) {
        Tensor src = t.astype(result_dtype);
        if (src.device() != result_device) {
            src = src.to(result_device);
        }

        // Create slice range for this tensor
        std::vector<Slice> slices;
        for (int d = 0; d < ndim; ++d) {
            if (d == norm_axis) {
                slices.push_back(
                    Slice(static_cast<int64_t>(concat_offset),
                          static_cast<int64_t>(concat_offset + t.shape()[d])));
            } else {
                slices.push_back(Slice()); // Full range
            }
        }

        // Get view of destination and copy
        Tensor dest_view = result.slice(slices);

        // Copy data element by element (could be optimized)
        if (result_device == Device::CPU && src.is_contiguous() &&
            dest_view.is_contiguous()) {
            std::memcpy(dest_view.data(), src.data(), src.nbytes());
        } else if (result_device == Device::CPU) {
            // Non-contiguous copy
            size_t total_elements = src.size();
            std::vector<size_t> coords(ndim, 0);
            for (size_t i = 0; i < total_elements; ++i) {
                size_t src_offset =
                    ShapeUtils::linear_index(coords, src.strides());
                size_t dst_offset =
                    ShapeUtils::linear_index(coords, dest_view.strides());
                std::memcpy(
                    static_cast<uint8_t *>(dest_view.data()) + dst_offset,
                    static_cast<const uint8_t *>(src.data()) + src_offset,
                    src.itemsize());

                // Increment coordinates
                for (int j = ndim - 1; j >= 0; --j) {
                    if (++coords[j] < src.shape()[j])
                        break;
                    coords[j] = 0;
                }
            }
        } else {
            // GPU: use CPU fallback for now
            auto cpu_result = concatenate(
                [&]() {
                    std::vector<Tensor> cpu_tensors;
                    for (const auto &tensor : tensors) {
                        cpu_tensors.push_back(tensor.cpu());
                    }
                    return cpu_tensors;
                }(),
                axis);
            return cpu_result.to(result_device);
        }

        concat_offset += t.shape()[norm_axis];
    }

    return result;
}

Tensor Tensor::stack(const std::vector<Tensor> &tensors, int axis) {
    if (tensors.empty()) {
        throw ValueError("stack requires at least one tensor");
    }

    // All tensors must have the same shape
    const Shape &first_shape = tensors[0].shape();
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape() != first_shape) {
            throw ShapeError("stack: all tensors must have the same shape");
        }
    }

    // Normalize axis (can be in range [0, ndim])
    int ndim = static_cast<int>(first_shape.size());
    int norm_axis = axis < 0 ? axis + ndim + 1 : axis;
    if (norm_axis < 0 || norm_axis > ndim) {
        throw ValueError("axis " + std::to_string(axis) +
                         " out of bounds for stack");
    }

    // unsqueeze each tensor at the stack axis, then concatenate
    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());
    for (const auto &t : tensors) {
        expanded.push_back(t.unsqueeze(norm_axis));
    }

    return concatenate(expanded, norm_axis);
}

Tensor Tensor::vstack(const std::vector<Tensor> &tensors) {
    if (tensors.empty()) {
        throw ValueError("vstack requires at least one tensor");
    }

    // Handle 1D arrays - stack along new first axis
    if (tensors[0].ndim() == 1) {
        return stack(tensors, 0);
    }

    // For 2D and above, concatenate along axis 0
    return concatenate(tensors, 0);
}

Tensor Tensor::hstack(const std::vector<Tensor> &tensors) {
    if (tensors.empty()) {
        throw ValueError("hstack requires at least one tensor");
    }

    // Handle 1D arrays - concatenate along axis 0
    if (tensors[0].ndim() == 1) {
        return concatenate(tensors, 0);
    }

    // For 2D and above, concatenate along axis 1
    return concatenate(tensors, 1);
}

Tensor Tensor::dstack(const std::vector<Tensor> &tensors) {
    if (tensors.empty()) {
        throw ValueError("dstack requires at least one tensor");
    }

    // Expand 1D and 2D arrays to 3D
    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());

    for (const auto &t : tensors) {
        if (t.ndim() == 1) {
            // Shape (N,) -> (1, N, 1)
            expanded.push_back(t.reshape({1, t.shape()[0], 1}));
        } else if (t.ndim() == 2) {
            // Shape (M, N) -> (M, N, 1)
            expanded.push_back(t.unsqueeze(2));
        } else {
            expanded.push_back(t);
        }
    }

    return concatenate(expanded, 2);
}

Tensor Tensor::column_stack(const std::vector<Tensor> &tensors) {
    if (tensors.empty()) {
        throw ValueError("column_stack requires at least one tensor");
    }

    // 1D arrays become columns, 2D arrays are stacked as-is
    std::vector<Tensor> columns;
    columns.reserve(tensors.size());

    for (const auto &t : tensors) {
        if (t.ndim() == 1) {
            // Shape (N,) -> (N, 1)
            columns.push_back(t.reshape({t.size(), 1}));
        } else {
            columns.push_back(t);
        }
    }

    return concatenate(columns, 1);
}

std::vector<Tensor> Tensor::split(size_t sections, int axis) const {
    int ndim_val = static_cast<int>(ndim());
    int norm_axis = axis < 0 ? axis + ndim_val : axis;

    if (norm_axis < 0 || norm_axis >= ndim_val) {
        throw ValueError("axis " + std::to_string(axis) + " out of bounds");
    }

    size_t axis_size = shape_[norm_axis];
    if (axis_size % sections != 0) {
        throw ValueError("split: tensor size " + std::to_string(axis_size) +
                         " not divisible by " + std::to_string(sections));
    }

    size_t section_size = axis_size / sections;
    std::vector<Tensor> result;
    result.reserve(sections);

    for (size_t i = 0; i < sections; ++i) {
        std::vector<Slice> slices;
        for (int d = 0; d < ndim_val; ++d) {
            if (d == norm_axis) {
                slices.push_back(
                    Slice(static_cast<int64_t>(i * section_size),
                          static_cast<int64_t>((i + 1) * section_size)));
            } else {
                slices.push_back(Slice());
            }
        }
        result.push_back(slice(slices));
    }

    return result;
}

std::vector<Tensor> Tensor::split(const std::vector<size_t> &indices,
                                  int axis) const {
    int ndim_val = static_cast<int>(ndim());
    int norm_axis = axis < 0 ? axis + ndim_val : axis;

    if (norm_axis < 0 || norm_axis >= ndim_val) {
        throw ValueError("axis " + std::to_string(axis) + " out of bounds");
    }

    std::vector<Tensor> result;
    size_t prev_idx = 0;

    for (size_t idx : indices) {
        std::vector<Slice> slices;
        for (int d = 0; d < ndim_val; ++d) {
            if (d == norm_axis) {
                slices.push_back(Slice(static_cast<int64_t>(prev_idx),
                                       static_cast<int64_t>(idx)));
            } else {
                slices.push_back(Slice());
            }
        }
        result.push_back(slice(slices));
        prev_idx = idx;
    }

    // Final section from last index to end
    std::vector<Slice> slices;
    for (int d = 0; d < ndim_val; ++d) {
        if (d == norm_axis) {
            slices.push_back(Slice(static_cast<int64_t>(prev_idx),
                                   static_cast<int64_t>(shape_[norm_axis])));
        } else {
            slices.push_back(Slice());
        }
    }
    result.push_back(slice(slices));

    return result;
}

std::vector<Tensor> Tensor::chunk(size_t n_chunks, int axis) const {
    int ndim_val = static_cast<int>(ndim());
    int norm_axis = axis < 0 ? axis + ndim_val : axis;

    if (norm_axis < 0 || norm_axis >= ndim_val) {
        throw ValueError("axis " + std::to_string(axis) + " out of bounds");
    }

    size_t axis_size = shape_[norm_axis];
    size_t base_chunk_size = axis_size / n_chunks;
    size_t remainder = axis_size % n_chunks;

    std::vector<Tensor> result;
    result.reserve(n_chunks);

    size_t start = 0;
    for (size_t i = 0; i < n_chunks; ++i) {
        // First 'remainder' chunks get one extra element
        size_t chunk_size = base_chunk_size + (i < remainder ? 1 : 0);
        size_t end = start + chunk_size;

        if (chunk_size > 0) {
            std::vector<Slice> slices;
            for (int d = 0; d < ndim_val; ++d) {
                if (d == norm_axis) {
                    slices.push_back(Slice(static_cast<int64_t>(start),
                                           static_cast<int64_t>(end)));
                } else {
                    slices.push_back(Slice());
                }
            }
            result.push_back(slice(slices));
        }
        start = end;
    }

    return result;
}

} // namespace axiom