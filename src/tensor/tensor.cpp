//=============================================================================
// src/tensor/tensor.cpp - Complete implementation with memory order support
//=============================================================================

#include "axiom/tensor.hpp"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace axiom {

// ============================================================================
// Private helper methods
// ============================================================================

size_t Tensor::calculate_storage_size() const {
    return size() * itemsize();
}

void Tensor::validate_indices(const std::vector<size_t>& indices) const {
    if (indices.size() != ndim()) {
        throw std::runtime_error("Number of indices must match tensor dimensions");
    }
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds");
        }
    }
}

void Tensor::update_contiguity_flags() {
    if (shape_.empty()) {
        flags_.c_contiguous = true;
        flags_.f_contiguous = true;
        return;
    }
    
    // Check C contiguity (row-major)
    auto c_strides = ShapeUtils::calculate_strides(shape_, itemsize(), MemoryOrder::RowMajor);
    flags_.c_contiguous = (strides_ == c_strides);
    
    // Check F contiguity (column-major)  
    auto f_strides = ShapeUtils::calculate_strides(shape_, itemsize(), MemoryOrder::ColMajor);
    flags_.f_contiguous = (strides_ == f_strides);
}

// ============================================================================
// Constructors and factory methods
// ============================================================================

Tensor::Tensor() 
    : storage_(nullptr)
    , shape_()
    , strides_()
    , dtype_(DType::Float32)
    , offset_(0)
    , flags_()
    , memory_order_(MemoryOrder::RowMajor) {
    flags_.owndata = false;
}

Tensor::Tensor(const Shape& shape, DType dtype, Device device, MemoryOrder order)
    : shape_(shape)
    , dtype_(dtype)
    , offset_(0)
    , flags_()
    , memory_order_(order) {
    
    if (!ShapeUtils::is_valid_shape(shape_)) {
        throw std::runtime_error("Invalid shape");
    }
    
    strides_ = ShapeUtils::calculate_strides(shape_, dtype_size(dtype_), order);
    storage_ = make_storage(calculate_storage_size(), device);
    
    update_contiguity_flags();
    flags_.owndata = true;
}

Tensor::Tensor(std::initializer_list<size_t> shape, DType dtype, Device device, MemoryOrder order)
    : Tensor(Shape(shape), dtype, device, order) {
}

Tensor::Tensor(std::shared_ptr<Storage> storage, const Shape& shape, const Strides& strides, 
               DType dtype, size_t offset, MemoryOrder order)
    : storage_(storage)
    , shape_(shape)
    , strides_(strides)
    , dtype_(dtype)
    , offset_(offset)
    , flags_()
    , memory_order_(order) {
    
    if (!storage_) {
        throw std::runtime_error("Storage cannot be null");
    }
    
    if (shape_.size() != strides_.size()) {
        throw std::runtime_error("Shape and strides must have same length");
    }
    
    // Check bounds
    size_t required_size = 0;
    if (!shape_.empty()) {
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] > 1) {
                required_size = std::max(required_size, (shape_[i] - 1) * strides_[i]);
            }
        }
        required_size += dtype_size(dtype_);
    }
    
    if (offset_ + required_size > storage_->size_bytes()) {
        throw std::runtime_error("Storage too small for tensor view");
    }
    
    update_contiguity_flags();
    flags_.owndata = !storage_->is_view();
}

// Copy constructor - now handles memory_order_
Tensor::Tensor(const Tensor& other)
    : storage_(other.storage_)
    , shape_(other.shape_)
    , strides_(other.strides_)
    , dtype_(other.dtype_)
    , offset_(other.offset_)
    , flags_(other.flags_)
    , memory_order_(other.memory_order_) {
}

Tensor& Tensor::operator=(const Tensor& other) {
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

Tensor::Tensor(Tensor&& other) noexcept
    : storage_(std::move(other.storage_))
    , shape_(std::move(other.shape_))
    , strides_(std::move(other.strides_))
    , dtype_(other.dtype_)
    , offset_(other.offset_)
    , flags_(other.flags_)
    , memory_order_(other.memory_order_) {
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
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

// ============================================================================
// Data access - CRITICAL: These were missing!
// ============================================================================

void* Tensor::data() {
    if (!storage_ || storage_->device() != Device::CPU) {
        throw std::runtime_error("Direct data access only available for CPU tensors");
    }
    return static_cast<uint8_t*>(storage_->data()) + offset_;
}

const void* Tensor::data() const {
    if (!storage_ || storage_->device() != Device::CPU) {
        throw std::runtime_error("Direct data access only available for CPU tensors");
    }
    return static_cast<const uint8_t*>(storage_->data()) + offset_;
}

Tensor Tensor::base() const {
    if (!is_view()) {
        return *this; // Return copy if not a view
    }
    
    auto base_storage = storage_->base();
    if (!base_storage) {
        return *this;
    }
    
    // Calculate the shape and strides of the base tensor
    // For now, return a simple version - this could be more sophisticated
    Shape base_shape = {storage_->size_bytes() / itemsize()};
    Strides base_strides = {itemsize()};
    
    return Tensor(base_storage, base_shape, base_strides, dtype_, 0);
}

// ============================================================================
// Memory order operations
// ============================================================================

Tensor Tensor::ascontiguousarray() const {
    if (is_c_contiguous()) {
        return *this; // Already C-contiguous
    }
    
    auto new_tensor = Tensor(shape_, dtype_, device(), MemoryOrder::RowMajor);
    
    if (device() == Device::CPU) {
        // Copy data with reordering
        for (size_t i = 0; i < size(); ++i) {
            auto indices = ShapeUtils::unravel_index(i, shape_);
            
            // Calculate source offset using current strides
            size_t src_offset = ShapeUtils::linear_index(indices, strides_);
            // Calculate destination offset using C-order strides  
            size_t dst_offset = ShapeUtils::linear_index(indices, new_tensor.strides_);
            
            std::memcpy(
                static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
                static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
                itemsize()
            );
        }
    } else {
        // For GPU, copy through storage interface
        new_tensor.storage_->copy_from(*storage_);
    }
    
    return new_tensor;
}

Tensor Tensor::asfortranarray() const {
    if (is_f_contiguous()) {
        return *this; // Already F-contiguous
    }
    
    auto new_tensor = Tensor(shape_, dtype_, device(), MemoryOrder::ColMajor);
    
    if (device() == Device::CPU) {
        // Copy data with reordering
        for (size_t i = 0; i < size(); ++i) {
            auto indices = ShapeUtils::unravel_index(i, shape_);
            
            // Calculate source offset using current strides
            size_t src_offset = ShapeUtils::linear_index(indices, strides_);
            // Calculate destination offset using F-order strides
            size_t dst_offset = ShapeUtils::linear_index(indices, new_tensor.strides_);
            
            std::memcpy(
                static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
                static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
                itemsize()
            );
        }
    } else {
        // For GPU, copy through storage interface
        new_tensor.storage_->copy_from(*storage_);
    }
    
    return new_tensor;
}

// ============================================================================
// Shape manipulation (view operations) - CRITICAL: These were missing!
// ============================================================================

Tensor Tensor::reshape(const Shape& new_shape, MemoryOrder order) const {
    Shape validated_shape = reshape_shape(shape_, new_shape);
    
    // Check if we can create a view (only if order matches and is contiguous)
    bool can_view = false;
    if (order == memory_order_) {
        if ((order == MemoryOrder::RowMajor && is_c_contiguous()) ||
            (order == MemoryOrder::ColMajor && is_f_contiguous())) {
            can_view = true;
        }
    }
    
    if (can_view) {
        Strides new_strides = ShapeUtils::calculate_strides(validated_shape, itemsize(), order);
        return Tensor(storage_, validated_shape, new_strides, dtype_, offset_, order);
    } else {
        // Need to create a copy with new order
        auto new_tensor = Tensor(validated_shape, dtype_, device(), order);
        // Copy data appropriately
        if (device() == Device::CPU) {
            for (size_t i = 0; i < size(); ++i) {
                auto indices = ShapeUtils::unravel_index(i, shape_);
                size_t src_offset = ShapeUtils::linear_index(indices, strides_);
                
                // Recalculate indices for new shape
                auto new_indices = ShapeUtils::unravel_index(i, validated_shape);
                size_t dst_offset = ShapeUtils::linear_index(new_indices, new_tensor.strides_);
                
                std::memcpy(
                    static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
                    static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
                    itemsize()
                );
            }
        } else {
            new_tensor.storage_->copy_from(*storage_);
        }
        return new_tensor;
    }
}

// Overload with default MemoryOrder for backward compatibility
Tensor Tensor::reshape(const Shape& new_shape) const {
    return reshape(new_shape, MemoryOrder::RowMajor);
}

Tensor Tensor::reshape(std::initializer_list<size_t> new_shape, MemoryOrder order) const {
    return reshape(Shape(new_shape), order);
}

Tensor Tensor::reshape(std::initializer_list<size_t> new_shape) const {
    return reshape(Shape(new_shape), MemoryOrder::RowMajor);
}

Tensor Tensor::transpose() const {
    if (ndim() < 2) {
        return *this; // 0D and 1D tensors are unchanged by transpose
    }
    
    Shape new_shape = shape_;
    Strides new_strides = strides_;
    
    // Reverse the last two dimensions
    std::swap(new_shape[ndim() - 2], new_shape[ndim() - 1]);
    std::swap(new_strides[ndim() - 2], new_strides[ndim() - 1]);
    
    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::transpose(const std::vector<int>& axes) const {
    if (axes.size() != ndim()) {
        throw std::runtime_error("Number of axes must match tensor dimensions");
    }
    
    Shape new_shape(ndim());
    Strides new_strides(ndim());
    
    for (size_t i = 0; i < axes.size(); ++i) {
        int axis = axes[i];
        if (axis < 0) axis += ndim();
        if (axis < 0 || axis >= static_cast<int>(ndim())) {
            throw std::runtime_error("Axis out of bounds");
        }
        
        new_shape[i] = shape_[axis];
        new_strides[i] = strides_[axis];
    }
    
    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::squeeze(int axis) const {
    Shape new_shape = squeeze_shape(shape_, axis);
    
    // Calculate corresponding strides
    Strides new_strides;
    if (axis == -1) {
        // Remove all dimensions of size 1
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] != 1) {
                new_strides.push_back(strides_[i]);
            }
        }
    } else {
        // Remove specific axis
        if (axis < 0) axis += shape_.size();
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (static_cast<int>(i) != axis) {
                new_strides.push_back(strides_[i]);
            }
        }
    }
    
    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::unsqueeze(int axis) const {
    Shape new_shape = unsqueeze_shape(shape_, axis);
    
    // Insert stride of 0 for new dimension (broadcasting)
    Strides new_strides = strides_;
    if (axis < 0) axis += shape_.size() + 1;
    new_strides.insert(new_strides.begin() + axis, 0);
    
    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::view(const Shape& new_shape) const {
    if (ShapeUtils::size(new_shape) != size()) {
        throw std::runtime_error("View must have same number of elements");
    }
    
    if (!is_contiguous()) {
        throw std::runtime_error("Cannot create view of non-contiguous tensor");
    }
    
    Strides new_strides = ShapeUtils::calculate_strides(new_shape, itemsize(), MemoryOrder::RowMajor);
    return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

// ============================================================================
// Memory operations
// ============================================================================

Tensor Tensor::copy(MemoryOrder order) const {
    auto new_tensor = Tensor(shape_, dtype_, device(), order);
    
    if (device() == Device::CPU && order != memory_order_) {
        // Need to reorder data during copy
        for (size_t i = 0; i < size(); ++i) {
            auto indices = ShapeUtils::unravel_index(i, shape_);
            size_t src_offset = ShapeUtils::linear_index(indices, strides_);
            size_t dst_offset = ShapeUtils::linear_index(indices, new_tensor.strides_);
            
            std::memcpy(
                static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
                static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
                itemsize()
            );
        }
    } else {
        // Simple copy (same order or GPU)
        new_tensor.storage_->copy_from(*storage_);
    }
    
    return new_tensor;
}

// Backward compatible copy() method
Tensor Tensor::copy() const {
    return copy(memory_order_);
}

Tensor Tensor::to(Device target_device, MemoryOrder order) const {
    if (device() == target_device && order == memory_order_) {
        return *this; // Already on target device with correct order
    }
    
    auto new_tensor = Tensor(shape_, dtype_, target_device, order);
    
    if (order != memory_order_ && device() == Device::CPU && target_device == Device::CPU) {
        // Reorder during copy
        for (size_t i = 0; i < size(); ++i) {
            auto indices = ShapeUtils::unravel_index(i, shape_);
            size_t src_offset = ShapeUtils::linear_index(indices, strides_);
            size_t dst_offset = ShapeUtils::linear_index(indices, new_tensor.strides_);
            
            std::memcpy(
                static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
                static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
                itemsize()
            );
        }
    } else {
        // Standard copy
        new_tensor.storage_->copy_from(*storage_);
    }
    
    return new_tensor;
}

// Backward compatible device transfer methods
Tensor Tensor::to(Device device) const {
    return to(device, memory_order_);
}

Tensor Tensor::cpu() const { 
    return to(Device::CPU, memory_order_); 
}

Tensor Tensor::gpu() const { 
    return to(Device::GPU, memory_order_); 
}

Tensor Tensor::astype(DType new_dtype) const {
    if (new_dtype == dtype_) {
        return *this; // No conversion needed
    }
    
    // TODO: Implement proper dtype conversion
    // For now, just create a new tensor with new dtype
    throw std::runtime_error("Type conversion not yet implemented");
}

// ============================================================================
// Utility methods
// ============================================================================

std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=" << dtype_name() << ", device=";
    oss << (device() == Device::CPU ? "CPU" : "GPU");
    if (shape_.size() > 0) {  // Only show order for non-empty tensors
        oss << ", order=" << (memory_order_ == MemoryOrder::RowMajor ? "RowMajor" : "ColMajor");
    }
    oss << ")";
    return oss.str();
}

std::string Tensor::str() const {
    return repr(); // For now, same as repr
}

bool Tensor::same_shape(const Tensor& other) const {
    return ShapeUtils::shapes_equal(shape_, other.shape_);
}

bool Tensor::same_dtype(const Tensor& other) const {
    return dtype_ == other.dtype_;
}

bool Tensor::same_device(const Tensor& other) const {
    return device() == other.device();
}

bool Tensor::same_memory_order(const Tensor& other) const {
    return memory_order_ == other.memory_order_;
}

// ============================================================================
// Tensor creation functions
// ============================================================================

Tensor zeros(const Shape& shape, DType dtype, Device device, MemoryOrder order) {
    auto tensor = Tensor(shape, dtype, device, order);
    if (device == Device::CPU) {
        std::memset(tensor.data(), 0, tensor.nbytes());
    }
    return tensor;
}

Tensor zeros(std::initializer_list<size_t> shape, DType dtype, Device device, MemoryOrder order) {
    return zeros(Shape(shape), dtype, device, order);
}

Tensor ones(const Shape& shape, DType dtype, Device device, MemoryOrder order) {
    auto tensor = Tensor(shape, dtype, device, order);
    if (device == Device::CPU) {
        // Fill with ones - implementation depends on dtype
        switch (dtype) {
            case DType::Float32:
                tensor.fill<float>(1.0f);
                break;
            case DType::Float64:
                tensor.fill<double>(1.0);
                break;
            case DType::Int32:
                tensor.fill<int32_t>(1);
                break;
            case DType::Int64:
                tensor.fill<int64_t>(1);
                break;
            default:
                throw std::runtime_error("Unsupported dtype for ones");
        }
    }
    return tensor;
}

Tensor ones(std::initializer_list<size_t> shape, DType dtype, Device device, MemoryOrder order) {
    return ones(Shape(shape), dtype, device, order);
}

Tensor empty(const Shape& shape, DType dtype, Device device, MemoryOrder order) {
    return Tensor(shape, dtype, device, order);
}

Tensor empty(std::initializer_list<size_t> shape, DType dtype, Device device, MemoryOrder order) {
    return empty(Shape(shape), dtype, device, order);
}

Tensor eye(size_t n, DType dtype, Device device, MemoryOrder order) {
    auto tensor = zeros({n, n}, dtype, device, order);
    
    if (device == Device::CPU) {
        // Set diagonal to 1
        switch (dtype) {
            case DType::Float32: {
                for (size_t i = 0; i < n; ++i) {
                    tensor.set_item<float>({i, i}, 1.0f);
                }
                break;
            }
            case DType::Float64: {
                for (size_t i = 0; i < n; ++i) {
                    tensor.set_item<double>({i, i}, 1.0);
                }
                break;
            }
            case DType::Int32: {
                for (size_t i = 0; i < n; ++i) {
                    tensor.set_item<int32_t>({i, i}, 1);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for eye");
        }
    }
    
    return tensor;
}

Tensor identity(size_t n, DType dtype, Device device, MemoryOrder order) {
    return eye(n, dtype, device, order);
}

// NumPy-compatible convenience functions
Tensor ascontiguousarray(const Tensor& tensor) {
    return tensor.ascontiguousarray();
}

Tensor asfortranarray(const Tensor& tensor) {
    return tensor.asfortranarray();
}

} // namespace axiom