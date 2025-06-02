#pragma once

#include "dtype.hpp"
#include "shape.hpp"
#include "storage.hpp"

#include <vector>
#include <memory>
#include <initializer_list>
#include <string>

namespace axiom {

// Forward declaration for tensor creation functions
class Tensor;

// Tensor flags (similar to numpy ndarray flags)
struct TensorFlags {
    bool writeable = true;
    bool c_contiguous = true;
    bool f_contiguous = false;
    bool aligned = true;
    bool owndata = true;
};

// Main tensor class - numpy ndarray compatible
class Tensor {
private:
    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Strides strides_;
    DType dtype_;
    size_t offset_;
    TensorFlags flags_;
    
    // Helper for calculating memory requirements
    size_t calculate_storage_size() const;
    
    // Helper for validating indexing
    void validate_indices(const std::vector<size_t>& indices) const;
    
public:
    // ============================================================================
    // Constructors and factory methods
    // ============================================================================
    
    // Default constructor - empty tensor
    Tensor();
    
    // Create tensor with given shape and dtype
    Tensor(const Shape& shape, DType dtype = DType::Float32, Device device = Device::CPU);
    Tensor(std::initializer_list<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);
    
    // Create tensor from existing storage (for views)
    Tensor(std::shared_ptr<Storage> storage, const Shape& shape, const Strides& strides, 
           DType dtype, size_t offset = 0);
    
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // ============================================================================
    // Core attributes (numpy ndarray compatible)
    // ============================================================================
    
    // Shape and dimensionality
    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return ShapeUtils::size(shape_); }
    
    // Strides and memory layout
    const Strides& strides() const { return strides_; }
    size_t itemsize() const { return dtype_size(dtype_); }
    size_t nbytes() const { return size() * itemsize(); }
    
    // Data type
    DType dtype() const { return dtype_; }
    std::string dtype_name() const { return axiom::dtype_name(dtype_); }
    
    // Memory and device info
    Device device() const { return storage_->device(); }
    const TensorFlags& flags() const { return flags_; }
    bool is_contiguous() const { return flags_.c_contiguous; }
    
    // Storage and base info
    std::shared_ptr<Storage> storage() const { return storage_; }
    bool is_view() const { return storage_->is_view(); }
    Tensor base() const;
    
    // ============================================================================
    // Data access
    // ============================================================================
    
    // Raw data access (CPU only)
    void* data();
    const void* data() const;
    
    // Typed data access (CPU only)
    template<typename T>
    T* typed_data() {
        if (device() != Device::CPU) {
            throw std::runtime_error("Direct data access only available for CPU tensors");
        }
        return reinterpret_cast<T*>(data());
    }
    
    template<typename T>
    const T* typed_data() const {
        if (device() != Device::CPU) {
            throw std::runtime_error("Direct data access only available for CPU tensors");
        }
        return reinterpret_cast<const T*>(data());
    }
    
    // ============================================================================
    // Indexing and slicing (basic element access)
    // ============================================================================
    
    // Single element access by multi-dimensional index
    template<typename T>
    T item(const std::vector<size_t>& indices) const {
        validate_indices(indices);
        if (device() != Device::CPU) {
            throw std::runtime_error("item() only available for CPU tensors");
        }
        size_t linear_idx = ShapeUtils::linear_index(indices, strides_);
        return typed_data<T>()[linear_idx];
    }
    
    // Single element assignment
    template<typename T>
    void set_item(const std::vector<size_t>& indices, const T& value) {
        validate_indices(indices);
        if (device() != Device::CPU) {
            throw std::runtime_error("set_item() only available for CPU tensors");
        }
        if (!flags_.writeable) {
            throw std::runtime_error("Tensor is not writeable");
        }
        size_t linear_idx = ShapeUtils::linear_index(indices, strides_);
        typed_data<T>()[linear_idx] = value;
    }
    
    // ============================================================================
    // Shape manipulation (view operations)
    // ============================================================================
    
    // Reshape tensor (returns view if possible)
    Tensor reshape(const Shape& new_shape) const;
    Tensor reshape(std::initializer_list<size_t> new_shape) const;
    
    // Transpose tensor
    Tensor transpose() const;
    Tensor transpose(const std::vector<int>& axes) const;
    
    // Add/remove dimensions
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(int axis) const;
    
    // Create view with different strides
    Tensor view(const Shape& new_shape) const;
    
    // ============================================================================
    // Memory operations
    // ============================================================================
    
    // Copy data
    Tensor copy() const;
    Tensor clone() const { return copy(); }
    
    // Move to different device
    Tensor to(Device device) const;
    Tensor cpu() const { return to(Device::CPU); }
    Tensor gpu() const { return to(Device::GPU); }
    
    // Convert dtype
    Tensor astype(DType new_dtype) const;
    
    // ============================================================================
    // Utility methods
    // ============================================================================
    
    // Check if tensor is empty
    bool empty() const { return size() == 0; }
    
    // String representation
    std::string repr() const;
    std::string str() const;
    
    // Comparison
    bool same_shape(const Tensor& other) const;
    bool same_dtype(const Tensor& other) const;
    bool same_device(const Tensor& other) const;
    
    // Fill tensor with value (CPU only for now)
    template<typename T>
    void fill(const T& value) {
        if (device() != Device::CPU) {
            throw std::runtime_error("fill() only available for CPU tensors");
        }
        if (!flags_.writeable) {
            throw std::runtime_error("Tensor is not writeable");
        }
        T* data_ptr = typed_data<T>();
        std::fill(data_ptr, data_ptr + size(), value);
    }
};

// ============================================================================
// Tensor creation functions (numpy-style factory functions)
// ============================================================================

// Create tensor filled with zeros
Tensor zeros(const Shape& shape, DType dtype = DType::Float32, Device device = Device::CPU);
Tensor zeros(std::initializer_list<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);

// Create tensor filled with ones  
Tensor ones(const Shape& shape, DType dtype = DType::Float32, Device device = Device::CPU);
Tensor ones(std::initializer_list<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);

// Create tensor filled with specific value
template<typename T>
Tensor full(const Shape& shape, const T& value, Device device = Device::CPU) {
    auto tensor = Tensor(shape, dtype_of_v<T>, device);
    if (device == Device::CPU) {
        tensor.fill(value);
    }
    return tensor;
}

// Create empty tensor (uninitialized)
Tensor empty(const Shape& shape, DType dtype = DType::Float32, Device device = Device::CPU);
Tensor empty(std::initializer_list<size_t> shape, DType dtype = DType::Float32, Device device = Device::CPU);

// Create tensor from existing data (CPU only)
template<typename T>
Tensor from_data(const T* data, const Shape& shape, bool copy = true) {
    auto tensor = Tensor(shape, dtype_of_v<T>, Device::CPU);
    if (copy) {
        std::memcpy(tensor.typed_data<T>(), data, tensor.nbytes());
    } else {
        // TODO: Implement non-owning tensor views
        throw std::runtime_error("Non-copying from_data not yet implemented");
    }
    return tensor;
}

// Create identity matrix
Tensor eye(size_t n, DType dtype = DType::Float32, Device device = Device::CPU);
Tensor identity(size_t n, DType dtype = DType::Float32, Device device = Device::CPU);

} // namespace axiom