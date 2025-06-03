#include "axiom/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

#include "axiom/einops.hpp"
#include "axiom/io.hpp"

namespace axiom {

size_t Tensor::calculate_storage_size() const { return size() * itemsize(); }

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

  auto c_strides =
      ShapeUtils::calculate_strides(shape_, itemsize(), MemoryOrder::RowMajor);
  flags_.c_contiguous = (strides_ == c_strides);

  auto f_strides =
      ShapeUtils::calculate_strides(shape_, itemsize(), MemoryOrder::ColMajor);
  flags_.f_contiguous = (strides_ == f_strides);
}

Tensor::Tensor()
    : storage_(nullptr),
      shape_(),
      strides_(),
      dtype_(DType::Float32),
      offset_(0),
      flags_(),
      memory_order_(MemoryOrder::RowMajor) {
  flags_.owndata = false;
}

Tensor::Tensor(const Shape& shape, DType dtype, Device device,
               MemoryOrder order)
    : shape_(shape), dtype_(dtype), offset_(0), flags_(), memory_order_(order) {
  if (!ShapeUtils::is_valid_shape(shape_)) {
    throw std::runtime_error("Invalid shape");
  }

  strides_ = ShapeUtils::calculate_strides(shape_, dtype_size(dtype_), order);
  storage_ = make_storage(calculate_storage_size(), device);

  update_contiguity_flags();
  flags_.owndata = true;
}

Tensor::Tensor(std::initializer_list<size_t> shape, DType dtype, Device device,
               MemoryOrder order)
    : Tensor(Shape(shape), dtype, device, order) {}

Tensor::Tensor(std::shared_ptr<Storage> storage, const Shape& shape,
               const Strides& strides, DType dtype, size_t offset,
               MemoryOrder order)
    : storage_(storage),
      shape_(shape),
      strides_(strides),
      dtype_(dtype),
      offset_(offset),
      flags_(),
      memory_order_(order) {
  if (!storage_) {
    throw std::runtime_error("Storage cannot be null");
  }

  if (shape_.size() != strides_.size()) {
    throw std::runtime_error("Shape and strides must have same length");
  }

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

Tensor::Tensor(const Tensor& other)
    : storage_(other.storage_),
      shape_(other.shape_),
      strides_(other.strides_),
      dtype_(other.dtype_),
      offset_(other.offset_),
      flags_(other.flags_),
      memory_order_(other.memory_order_) {}

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
    : storage_(std::move(other.storage_)),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      dtype_(other.dtype_),
      offset_(other.offset_),
      flags_(other.flags_),
      memory_order_(other.memory_order_) {}

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

void* Tensor::data() {
  if (!storage_ || storage_->device() != Device::CPU) {
    throw std::runtime_error(
        "Direct data access only available for CPU tensors");
  }
  return static_cast<uint8_t*>(storage_->data()) + offset_;
}

const void* Tensor::data() const {
  if (!storage_ || storage_->device() != Device::CPU) {
    throw std::runtime_error(
        "Direct data access only available for CPU tensors");
  }
  return static_cast<const uint8_t*>(storage_->data()) + offset_;
}

Tensor Tensor::base() const {
  if (!is_view()) {
    return *this;
  }

  auto base_storage = storage_->base();
  if (!base_storage) {
    return *this;
  }

  Shape base_shape = {storage_->size_bytes() / itemsize()};
  Strides base_strides = {itemsize()};

  return Tensor(base_storage, base_shape, base_strides, dtype_, 0);
}

Tensor Tensor::ascontiguousarray() const {
  if (is_c_contiguous()) {
    return *this;
  }

  auto new_tensor = Tensor(shape_, dtype_, device(), MemoryOrder::RowMajor);

  if (device() == Device::CPU) {
    for (size_t i = 0; i < size(); ++i) {
      auto indices = ShapeUtils::unravel_index(i, shape_);
      size_t src_offset = ShapeUtils::linear_index(indices, strides_);
      size_t dst_offset =
          ShapeUtils::linear_index(indices, new_tensor.strides_);

      std::memcpy(
          static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
          static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
          itemsize());
    }
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
    for (size_t i = 0; i < size(); ++i) {
      auto indices = ShapeUtils::unravel_index(i, shape_);
      size_t src_offset = ShapeUtils::linear_index(indices, strides_);
      size_t dst_offset =
          ShapeUtils::linear_index(indices, new_tensor.strides_);

      std::memcpy(
          static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
          static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
          itemsize());
    }
  } else {
    new_tensor.storage_->copy_from(*storage_);
  }

  return new_tensor;
}

Tensor Tensor::reshape(const Shape& new_shape, MemoryOrder order) const {
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
        size_t src_offset = ShapeUtils::linear_index(indices, strides_);
        auto new_indices = ShapeUtils::unravel_index(i, validated_shape);
        size_t dst_offset =
            ShapeUtils::linear_index(new_indices, new_tensor.strides_);

        std::memcpy(
            static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
            static_cast<const uint8_t*>(storage_->data()) + offset_ +
                src_offset,
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

Tensor Tensor::rearrange(
    const std::string& pattern,
    const std::map<std::string, size_t>& axis_sizes) const {
  return einops::rearrange(*this, pattern, axis_sizes);
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

  Strides new_strides;
  if (axis == -1) {
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (shape_[i] != 1) {
        new_strides.push_back(strides_[i]);
      }
    }
  } else {
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

  Strides new_strides = ShapeUtils::calculate_strides(new_shape, itemsize(),
                                                      MemoryOrder::RowMajor);
  return Tensor(storage_, new_shape, new_strides, dtype_, offset_);
}

Tensor Tensor::copy(MemoryOrder order) const {
  auto new_tensor = Tensor(shape_, dtype_, device(), order);

  if (device() == Device::CPU && order != memory_order_) {
    for (size_t i = 0; i < size(); ++i) {
      auto indices = ShapeUtils::unravel_index(i, shape_);
      size_t src_offset = ShapeUtils::linear_index(indices, strides_);
      size_t dst_offset =
          ShapeUtils::linear_index(indices, new_tensor.strides_);

      std::memcpy(
          static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
          static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
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
      size_t src_offset = ShapeUtils::linear_index(indices, strides_);
      size_t dst_offset =
          ShapeUtils::linear_index(indices, new_tensor.strides_);

      std::memcpy(
          static_cast<uint8_t*>(new_tensor.storage_->data()) + dst_offset,
          static_cast<const uint8_t*>(storage_->data()) + offset_ + src_offset,
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
      type_conversion::convert_dtype_strided(new_tensor.data(), data(), shape_,
                                             new_tensor.strides(), strides_,
                                             new_dtype, dtype_, 0, offset_);
    }
  } else {
    auto cpu_source = (device() == Device::CPU) ? *this : this->cpu();
    auto cpu_target = Tensor(shape_, new_dtype, Device::CPU, memory_order_);

    if (cpu_source.is_contiguous() && cpu_target.is_contiguous()) {
      type_conversion::convert_dtype(cpu_target.data(), cpu_source.data(),
                                     size(), new_dtype, dtype_);
    } else {
      type_conversion::convert_dtype_strided(
          cpu_target.data(), cpu_source.data(), shape_, cpu_target.strides(),
          cpu_source.strides(), new_dtype, dtype_, 0, cpu_source.offset_);
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
    throw std::runtime_error(
        "Type conversion from " + dtype_name() + " to " +
        axiom::dtype_name(new_dtype) +
        " may lose precision. Use astype() to force conversion.");
  }
  return astype(new_dtype);
}

std::string Tensor::repr() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << shape_[i];
  }
  oss << "], dtype=" << dtype_name() << ", device=";
  oss << (device() == Device::CPU ? "CPU" : "GPU");
  if (shape_.size() > 0) {
    oss << ", order="
        << (memory_order_ == MemoryOrder::RowMajor ? "RowMajor" : "ColMajor");
  }
  oss << ")";
  return oss.str();
}

std::string Tensor::str() const { return repr(); }

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

Tensor Tensor::zeros(const Shape& shape, DType dtype, Device device,
                     MemoryOrder order) {
  auto tensor = Tensor(shape, dtype, device, order);
  if (device == Device::CPU) {
    std::memset(tensor.data(), 0, tensor.nbytes());
  }
  return tensor;
}

Tensor Tensor::zeros(std::initializer_list<size_t> shape, DType dtype,
                     Device device, MemoryOrder order) {
  return zeros(Shape(shape), dtype, device, order);
}

Tensor Tensor::ones(const Shape& shape, DType dtype, Device device,
                    MemoryOrder order) {
  auto tensor = Tensor(shape, dtype, device, order);
  if (device == Device::CPU) {
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
  }
  return tensor;
}

Tensor Tensor::ones(std::initializer_list<size_t> shape, DType dtype,
                    Device device, MemoryOrder order) {
  return ones(Shape(shape), dtype, device, order);
}

Tensor Tensor::empty(const Shape& shape, DType dtype, Device device,
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
        for (size_t i = 0; i < n; ++i) tensor.set_item<bool>({i, i}, true);
        break;
      case DType::Int8:
        for (size_t i = 0; i < n; ++i) tensor.set_item<int8_t>({i, i}, 1);
        break;
      case DType::Int16:
        for (size_t i = 0; i < n; ++i) tensor.set_item<int16_t>({i, i}, 1);
        break;
      case DType::Int32:
        for (size_t i = 0; i < n; ++i) tensor.set_item<int32_t>({i, i}, 1);
        break;
      case DType::Int64:
        for (size_t i = 0; i < n; ++i) tensor.set_item<int64_t>({i, i}, 1);
        break;
      case DType::UInt8:
        for (size_t i = 0; i < n; ++i) tensor.set_item<uint8_t>({i, i}, 1);
        break;
      case DType::UInt16:
        for (size_t i = 0; i < n; ++i) tensor.set_item<uint16_t>({i, i}, 1);
        break;
      case DType::UInt32:
        for (size_t i = 0; i < n; ++i) tensor.set_item<uint32_t>({i, i}, 1);
        break;
      case DType::UInt64:
        for (size_t i = 0; i < n; ++i) tensor.set_item<uint64_t>({i, i}, 1);
        break;
      case DType::Float16:
        for (size_t i = 0; i < n; ++i)
          tensor.set_item<float16_t>({i, i}, float16_t(1.0f));
        break;
      case DType::Float32:
        for (size_t i = 0; i < n; ++i) tensor.set_item<float>({i, i}, 1.0f);
        break;
      case DType::Float64:
        for (size_t i = 0; i < n; ++i) tensor.set_item<double>({i, i}, 1.0);
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

Tensor Tensor::randn(const Shape& shape, DType dtype, Device device,
                     MemoryOrder order) {
  auto tensor = Tensor(shape, dtype, device, order);

  // For CPU tensors, fill with random normal values
  if (device == Device::CPU) {
    // Simple random number generation for demonstration
    // In a real implementation, you'd use a proper random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    switch (dtype) {
      case DType::Float32: {
        float* data = tensor.typed_data<float>();
        for (size_t i = 0; i < tensor.size(); ++i) {
          data[i] = dis(gen);
        }
        break;
      }
      case DType::Float64: {
        double* data = tensor.typed_data<double>();
        for (size_t i = 0; i < tensor.size(); ++i) {
          data[i] = static_cast<double>(dis(gen));
        }
        break;
      }
      case DType::Float16: {
        float16_t* data = tensor.typed_data<float16_t>();
        for (size_t i = 0; i < tensor.size(); ++i) {
          data[i] = float16_t(dis(gen));
        }
        break;
      }
      default:
        throw std::runtime_error("randn only supports floating point types");
    }
  }

  return tensor;
}

// ============================================================================
// File I/O method implementations
// ============================================================================

void Tensor::save(const std::string& filename) const {
  io::save(*this, filename);
}

void Tensor::save(const std::string& filename, const io::SerializationOptions& options) const {
  io::save(*this, filename, options);
}

void Tensor::save_to_stream(std::ostream& stream) const {
  io::save_stream(*this, stream);
}

void Tensor::save_to_stream(std::ostream& stream, const io::SerializationOptions& options) const {
  io::save_stream(*this, stream, options);
}

Tensor Tensor::load(const std::string& filename, Device device) {
  return io::load(filename, device);
}

Tensor Tensor::load_from_stream(std::istream& stream, Device device) {
  return io::load_stream(stream, device);
}

void Tensor::save_tensors(const std::map<std::string, Tensor>& tensors,
                         const std::string& filename) {
  io::save_archive(tensors, filename);
}

void Tensor::save_tensors(const std::map<std::string, Tensor>& tensors,
                         const std::string& filename,
                         const io::SerializationOptions& options) {
  io::save_archive(tensors, filename, options);
}

std::map<std::string, Tensor> Tensor::load_tensors(const std::string& filename,
                                                   Device device) {
  return io::load_archive(filename, device);
}

std::vector<std::string> Tensor::list_tensors_in_archive(const std::string& filename) {
  return io::list_archive(filename);
}

Tensor Tensor::load_tensor_from_archive(const std::string& filename,
                                        const std::string& tensor_name,
                                        Device device) {
  return io::load_from_archive(filename, tensor_name, device);
}

}  // namespace axiom