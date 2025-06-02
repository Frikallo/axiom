//=============================================================================
// include/axiom/tensor.hpp - Complete header with memory order support
//=============================================================================

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
    MemoryOrder memory_order_;  // Track the intended memory order
    
    // Helper for calculating memory requirements
    size_t calculate_storage_size() const;
    
    // Helper for validating indexing
    void validate_indices(const std::vector<size_t>& indices) const;
    
    // Helper to update contiguity flags
    void update_contiguity_flags();
    
public:
    // ============================================================================
    // Constructors and factory methods
    // ============================================================================
    
    // Default constructor - empty tensor
    Tensor();
    
    // Create tensor with given shape, dtype, device, and memory order
    Tensor(const Shape& shape, DType dtype = DType::Float32, 
           Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);
    Tensor(std::initializer_list<size_t> shape, DType dtype = DType::Float32, 
           Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);
    
    // Create tensor from existing storage (for views)
    Tensor(std::shared_ptr<Storage> storage, const Shape& shape, const Strides& strides, 
           DType dtype, size_t offset = 0, MemoryOrder order = MemoryOrder::RowMajor);
    
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
    MemoryOrder memory_order() const { return memory_order_; }
    
    // Data type
    DType dtype() const { return dtype_; }
    std::string dtype_name() const { return axiom::dtype_name(dtype_); }
    
    // Memory and device info
    Device device() const { return storage_->device(); }
    const TensorFlags& flags() const { return flags_; }
    bool is_contiguous() const { return flags_.c_contiguous; }
    bool is_c_contiguous() const { return flags_.c_contiguous; }
    bool is_f_contiguous() const { return flags_.f_contiguous; }
    
    // Storage and base info
    std::shared_ptr<Storage> storage() const { return storage_; }
    bool is_view() const { return storage_->is_view(); }
    Tensor base() const;
    
    // ============================================================================
    // Memory order operations
    // ============================================================================
    
    // Convert to C-contiguous (row-major)
    Tensor ascontiguousarray() const;
    Tensor as_c_contiguous() const { return ascontiguousarray(); }
    
    // Convert to Fortran-contiguous (column-major)  
    Tensor asfortranarray() const;
    Tensor as_f_contiguous() const { return asfortranarray(); }
    
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
        const T* data_ptr = reinterpret_cast<const T*>(static_cast<const uint8_t*>(storage_->data()) + offset_);
        return data_ptr[linear_idx / sizeof(T)];
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
        T* data_ptr = reinterpret_cast<T*>(static_cast<uint8_t*>(storage_->data()) + offset_);
        data_ptr[linear_idx / sizeof(T)] = value;
    }
    
    // ============================================================================
    // Shape manipulation (view operations)
    // ============================================================================
    
    // Reshape tensor (returns view if possible)
    // New versions with memory order support
    Tensor reshape(const Shape& new_shape, MemoryOrder order) const;
    Tensor reshape(std::initializer_list<size_t> new_shape, MemoryOrder order) const;
    
    // Backward compatible versions (default to C-order)
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
    
    // Copy data (with optional memory order specification)
    Tensor copy(MemoryOrder order) const;
    Tensor copy() const;  // Backward compatible version
    Tensor clone() const { return copy(); }
    
    // Move to different device
    Tensor to(Device device, MemoryOrder order) const;
    Tensor to(Device device) const;  // Backward compatible version
    Tensor cpu() const;
    Tensor gpu() const;
    
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
    bool same_memory_order(const Tensor& other) const;
    
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

// Tensor creation functions with memory order support (backward compatible)
Tensor zeros(const Shape& shape, DType dtype = DType::Float32, 
             Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);
Tensor zeros(std::initializer_list<size_t> shape, DType dtype = DType::Float32, 
             Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);

Tensor ones(const Shape& shape, DType dtype = DType::Float32, 
            Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);
Tensor ones(std::initializer_list<size_t> shape, DType dtype = DType::Float32, 
            Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);

// Create tensor filled with specific value
template<typename T>
Tensor full(const Shape& shape, const T& value, Device device = Device::CPU, 
            MemoryOrder order = MemoryOrder::RowMajor) {
    auto tensor = Tensor(shape, dtype_of_v<T>, device, order);
    if (device == Device::CPU) {
        tensor.fill(value);
    }
    return tensor;
}

// Create empty tensor (uninitialized) with memory order support
Tensor empty(const Shape& shape, DType dtype = DType::Float32, 
             Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);
Tensor empty(std::initializer_list<size_t> shape, DType dtype = DType::Float32, 
             Device device = Device::CPU, MemoryOrder order = MemoryOrder::RowMajor);

// Create tensor from existing data (CPU only)
template<typename T>
Tensor from_data(const T* data, const Shape& shape, bool copy = true, 
                 MemoryOrder order = MemoryOrder::RowMajor) {
    auto tensor = Tensor(shape, dtype_of_v<T>, Device::CPU, order);
    if (copy) {
        if (order == MemoryOrder::RowMajor) {
            std::memcpy(tensor.typed_data<T>(), data, tensor.nbytes());
        } else {
            // For Fortran order, need to reorder the data
            auto src_strides = ShapeUtils::calculate_strides(shape, sizeof(T), MemoryOrder::RowMajor);
            auto dst_strides = tensor.strides();
            
            // Copy with stride reordering
            for (size_t i = 0; i < tensor.size(); ++i) {
                auto indices = ShapeUtils::unravel_index(i, shape);
                size_t src_idx = ShapeUtils::linear_index(indices, src_strides) / sizeof(T);
                size_t dst_idx = ShapeUtils::linear_index(indices, dst_strides) / sizeof(T);
                tensor.typed_data<T>()[dst_idx] = data[src_idx];
            }
        }
    } else {
        // TODO: Implement non-owning tensor views
        throw std::runtime_error("Non-copying from_data not yet implemented");
    }
    return tensor;
}

// Create identity matrix with memory order support
Tensor eye(size_t n, DType dtype = DType::Float32, Device device = Device::CPU, 
           MemoryOrder order = MemoryOrder::RowMajor);
Tensor identity(size_t n, DType dtype = DType::Float32, Device device = Device::CPU, 
                MemoryOrder order = MemoryOrder::RowMajor);

// NumPy-compatible functions
Tensor ascontiguousarray(const Tensor& tensor);
Tensor asfortranarray(const Tensor& tensor);

} // namespace axiom