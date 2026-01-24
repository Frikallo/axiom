#pragma once

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <variant>

#include "dtype.hpp"
#include "shape.hpp"
#include "storage.hpp"
#include "type_conversion.hpp"
#include "indexing.hpp"

// Forward declarations for I/O functionality
namespace axiom {
namespace io {
struct SerializationOptions;
}
}  // namespace axiom

namespace axiom {

struct TensorFlags {
  bool writeable = true;
  bool c_contiguous = true;
  bool f_contiguous = false;
  bool aligned = true;
  bool owndata = true;
};

class Tensor {
 private:
  std::shared_ptr<Storage> storage_;
  Shape shape_;
  Strides strides_;
  DType dtype_;
  size_t offset_;
  TensorFlags flags_;
  MemoryOrder memory_order_;

  size_t calculate_storage_size() const;
  void validate_indices(const std::vector<size_t>& indices) const;
  void update_contiguity_flags();

 public:
  // Constructors
  Tensor();
  Tensor(const Shape& shape, DType dtype = DType::Float32,
         Device device = Device::CPU,
         MemoryOrder order = MemoryOrder::RowMajor);
  Tensor(std::initializer_list<size_t> shape, DType dtype = DType::Float32,
         Device device = Device::CPU,
         MemoryOrder order = MemoryOrder::RowMajor);
  Tensor(std::shared_ptr<Storage> storage, const Shape& shape,
         const Strides& strides, DType dtype, size_t offset = 0,
         MemoryOrder order = MemoryOrder::RowMajor);

  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  // Core attributes
  const Shape& shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t size() const { return ShapeUtils::size(shape_); }
  const Strides& strides() const { return strides_; }
  size_t itemsize() const { return dtype_size(dtype_); }
  size_t nbytes() const { return size() * itemsize(); }
  MemoryOrder memory_order() const { return memory_order_; }
  DType dtype() const { return dtype_; }
  std::string dtype_name() const { return axiom::dtype_name(dtype_); }
  Device device() const { return storage_->device(); }
  const TensorFlags& flags() const { return flags_; }
  bool is_contiguous() const { return flags_.c_contiguous; }
  bool is_c_contiguous() const { return flags_.c_contiguous; }
  bool is_f_contiguous() const { return flags_.f_contiguous; }
  std::shared_ptr<Storage> storage() const { return storage_; }
  size_t offset() const { return offset_; }
  bool empty() const { return size() == 0; }

  // Data access
  void* data();
  const void* data() const;
  
  Tensor slice(const std::vector<Slice>& slice_args) const;

  Tensor operator[](std::initializer_list<Index> indices) const;

  template <typename T>
  T* typed_data() {
    if (device() != Device::CPU) {
      throw std::runtime_error(
          "Direct data access only available for CPU tensors");
    }
    return reinterpret_cast<T*>(data());
  }

  template <typename T>
  const T* typed_data() const {
    if (device() != Device::CPU) {
      throw std::runtime_error(
          "Direct data access only available for CPU tensors");
    }
    return reinterpret_cast<const T*>(data());
  }

  template <typename T>
  T item(const std::vector<size_t>& indices) const {
    validate_indices(indices);
    if (device() != Device::CPU) {
      throw std::runtime_error("item() only available for CPU tensors");
    }
    size_t byte_offset = ShapeUtils::linear_index(indices, strides_);
    const T* data_ptr = reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(storage_->data()) + offset_ + byte_offset);
    return *data_ptr;
  }

  template <typename T>
  void set_item(const std::vector<size_t>& indices, const T& value) {
    validate_indices(indices);
    if (device() != Device::CPU) {
      throw std::runtime_error("set_item() only available for CPU tensors");
    }
    if (!flags_.writeable) {
      throw std::runtime_error("Tensor is not writeable");
    }
    size_t byte_offset = ShapeUtils::linear_index(indices, strides_);
    T* data_ptr =
        reinterpret_cast<T*>(static_cast<uint8_t*>(storage_->data()) + offset_ + byte_offset);
    *data_ptr = value;
  }

  template <typename T>
  void fill(const T& value) {
    if (device() != Device::CPU) {
      throw std::runtime_error("fill() only available for CPU tensors");
    }
    if (!flags_.writeable) {
      throw std::runtime_error("Tensor is not writeable");
    }

    if (is_contiguous()) {
        T* data_ptr = typed_data<T>();
        std::fill(data_ptr, data_ptr + size(), value);
    } else {
        // Fallback for non-contiguous tensors
        std::vector<size_t> indices(ndim(), 0);
        for (size_t i = 0; i < size(); ++i) {
            set_item(indices, value);
            
            // Increment indices
            for (int j = ndim() - 1; j >= 0; --j) {
                if (++indices[j] < shape_[j]) {
                    break;
                }
                indices[j] = 0;
            }
        }
    }
  }

  // Memory layout operations
  Tensor ascontiguousarray() const;
  Tensor as_c_contiguous() const { return ascontiguousarray(); }
  Tensor asfortranarray() const;
  Tensor as_f_contiguous() const { return asfortranarray(); }

  // Shape manipulation
  Tensor reshape(const Shape& new_shape,
                 MemoryOrder order = MemoryOrder::RowMajor) const;
  Tensor reshape(std::initializer_list<size_t> new_shape,
                 MemoryOrder order = MemoryOrder::RowMajor) const;
  Tensor rearrange(const std::string& pattern,
                   const std::map<std::string, size_t>& axis_sizes = {}) const;
  Tensor transpose() const;
  Tensor transpose(const std::vector<int>& axes) const;
  Tensor squeeze(int axis = -1) const;
  Tensor unsqueeze(int axis) const;
  Tensor view(const Shape& new_shape) const;
  Tensor flatten(int start_dim = 0, int end_dim = -1) const;

  // Expand and repeat operations
  // expand: Zero-copy view using 0-stride for broadcasted dims
  // Only works when expanding dims of size 1
  Tensor expand(const Shape& new_shape) const;

  // repeat: Copies data to create repeated tensor
  // Each dim is repeated by the corresponding factor
  Tensor repeat(const std::vector<size_t>& repeats) const;
  Tensor tile(const std::vector<size_t>& reps) const { return repeat(reps); }  // NumPy alias

  // Matrix operations
  Tensor matmul(const Tensor& other, bool transpose_self = false,
                bool transpose_other = false) const;
  Tensor mm(const Tensor& other) const { return matmul(other); }  // Alias
  Tensor dot(const Tensor& other) const { return matmul(other); } // Alias for vectors

  // Reduction member functions
  Tensor sum(int axis = -1, bool keep_dims = false) const;
  Tensor sum(const std::vector<int>& axes, bool keep_dims = false) const;
  Tensor mean(int axis = -1, bool keep_dims = false) const;
  Tensor mean(const std::vector<int>& axes, bool keep_dims = false) const;
  Tensor max(int axis = -1, bool keep_dims = false) const;
  Tensor min(int axis = -1, bool keep_dims = false) const;
  Tensor argmax(int axis = -1, bool keep_dims = false) const;
  Tensor argmin(int axis = -1, bool keep_dims = false) const;

  // Memory operations
  Tensor copy(MemoryOrder order = MemoryOrder::RowMajor) const;
  Tensor clone() const { return copy(); }
  Tensor to(Device device, MemoryOrder order = MemoryOrder::RowMajor) const;
  Tensor cpu() const;
  Tensor gpu() const;

  // Type conversion
  Tensor astype(DType new_dtype) const;
  Tensor astype_safe(DType new_dtype) const;
  Tensor to_float() const { return astype(DType::Float32); }
  Tensor to_double() const { return astype(DType::Float64); }
  Tensor to_int() const { return astype(DType::Int32); }
  Tensor to_int64() const { return astype(DType::Int64); }
  Tensor to_bool() const { return astype(DType::Bool); }
  Tensor to_complex() const { return astype(DType::Complex64); }
  Tensor to_complex128() const { return astype(DType::Complex128); }

  // Utility methods
  std::string repr() const;
  std::string str() const;
  bool same_shape(const Tensor& other) const;
  bool same_dtype(const Tensor& other) const;
  bool same_device(const Tensor& other) const;
  bool same_memory_order(const Tensor& other) const;

  // File I/O methods
  void save(const std::string& filename) const;
  void save(const std::string& filename,
            const io::SerializationOptions& options) const;
  void save_to_stream(std::ostream& stream) const;
  void save_to_stream(std::ostream& stream,
                      const io::SerializationOptions& options) const;

  // Static loading methods
  static Tensor load(const std::string& filename, Device device = Device::CPU);
  static Tensor load_from_stream(std::istream& stream,
                                 Device device = Device::CPU);

  // Archive methods (for multiple tensors)
  static void save_tensors(const std::map<std::string, Tensor>& tensors,
                           const std::string& filename);
  static void save_tensors(const std::map<std::string, Tensor>& tensors,
                           const std::string& filename,
                           const io::SerializationOptions& options);
  static std::map<std::string, Tensor> load_tensors(
      const std::string& filename, Device device = Device::CPU);
  static std::vector<std::string> list_tensors_in_archive(
      const std::string& filename);
  static Tensor load_tensor_from_archive(const std::string& filename,
                                         const std::string& tensor_name,
                                         Device device = Device::CPU);

  // Static factory methods
  static Tensor zeros(const Shape& shape, DType dtype = DType::Float32,
                      Device device = Device::CPU,
                      MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor zeros(std::initializer_list<size_t> shape,
                      DType dtype = DType::Float32, Device device = Device::CPU,
                      MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor ones(const Shape& shape, DType dtype = DType::Float32,
                     Device device = Device::CPU,
                     MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor ones(std::initializer_list<size_t> shape,
                     DType dtype = DType::Float32, Device device = Device::CPU,
                     MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor empty(const Shape& shape, DType dtype = DType::Float32,
                      Device device = Device::CPU,
                      MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor empty(std::initializer_list<size_t> shape,
                      DType dtype = DType::Float32, Device device = Device::CPU,
                      MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor eye(size_t n, DType dtype = DType::Float32,
                    Device device = Device::CPU,
                    MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor identity(size_t n, DType dtype = DType::Float32,
                         Device device = Device::CPU,
                         MemoryOrder order = MemoryOrder::RowMajor);
  static Tensor randn(const Shape& shape, DType dtype = DType::Float32,
                      Device device = Device::CPU,
                      MemoryOrder order = MemoryOrder::RowMajor);
  static void manual_seed(uint64_t seed);
  static Tensor arange(int64_t start, int64_t end, int64_t step = 1,
                       DType dtype = DType::Int32, Device device = Device::CPU);
  static Tensor arange(int64_t end, DType dtype = DType::Int32,
                       Device device = Device::CPU);

  template <typename T>
  static Tensor full(const Shape& shape, const T& value,
                     Device device = Device::CPU,
                     MemoryOrder order = MemoryOrder::RowMajor) {
    auto tensor = Tensor(shape, dtype_of_v<T>, device, order);
    if (device == Device::CPU) {
      tensor.fill(value);
    }
    return tensor;
  }

  template <typename T>
  static Tensor from_data(const T* data, const Shape& shape, bool copy = true,
                          MemoryOrder order = MemoryOrder::RowMajor) {
    auto tensor = Tensor(shape, dtype_of_v<T>, Device::CPU, order);
    if (copy) {
      if (order == MemoryOrder::RowMajor) {
        std::memcpy(tensor.typed_data<T>(), data, tensor.nbytes());
      } else {
        auto src_strides = ShapeUtils::calculate_strides(shape, sizeof(T),
                                                         MemoryOrder::RowMajor);
        auto dst_strides = tensor.strides();
        for (size_t i = 0; i < tensor.size(); ++i) {
          auto indices = ShapeUtils::unravel_index(i, shape);
          size_t src_idx =
              ShapeUtils::linear_index(indices, src_strides) / sizeof(T);
          size_t dst_idx =
              ShapeUtils::linear_index(indices, dst_strides) / sizeof(T);
          tensor.typed_data<T>()[dst_idx] = data[src_idx];
        }
      }
    } else {
      throw std::runtime_error("Non-copying from_data not yet implemented");
    }
    return tensor;
  }

  template <typename T, size_t N>
  static Tensor from_array(const T (&data)[N], const Shape& shape,
                           DType target_dtype = dtype_of_v<T>,
                           Device device = Device::CPU,
                           MemoryOrder order = MemoryOrder::RowMajor) {
    if (ShapeUtils::size(shape) != N) {
      throw std::runtime_error("Array size doesn't match tensor shape");
    }
    auto source_tensor = from_data(data, shape, true, order);
    if (target_dtype != dtype_of_v<T>) {
      source_tensor = source_tensor.astype(target_dtype);
    }
    if (device != Device::CPU) {
      source_tensor = source_tensor.to(device);
    }
    return source_tensor;
  }

  template <typename T>
  static Tensor asarray(const Tensor& tensor) {
    return tensor.astype(dtype_of_v<T>);
  }

  template <typename T>
  static Tensor asarray(const Tensor& tensor, Device device) {
    return tensor.astype(dtype_of_v<T>).to(device);
  }

  static DType result_type(const Tensor& a, const Tensor& b) {
    return type_conversion::promote_dtypes(a.dtype(), b.dtype());
  }

  static Tensor ascontiguousarray(const Tensor& tensor) {
    return tensor.ascontiguousarray();
  }

  static Tensor asfortranarray(const Tensor& tensor) {
    return tensor.asfortranarray();
  }
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

// Operator overloads
Tensor operator-(const Tensor& tensor);

}  // namespace axiom