#pragma once

#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "dtype.hpp"
#include "error.hpp"
#include "indexing.hpp"
#include "shape.hpp"
#include "storage.hpp"
#include "type_conversion.hpp"

// Forward declarations for I/O functionality
namespace axiom {
namespace io {}
} // namespace axiom

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
    DTypes dtype_;
    size_t offset_;
    TensorFlags flags_;
    MemoryOrder memory_order_;

    size_t calculate_storage_size() const;
    void validate_indices(const std::vector<size_t> &indices) const;
    void update_contiguity_flags();

  public:
    // Constructors
    Tensor();
    Tensor(const Shape &shape, DTypes dtype = Float32(),
           Device device = Device::CPU,
           MemoryOrder order = MemoryOrder::RowMajor);
    Tensor(std::initializer_list<size_t> shape, DTypes dtype = Float32(),
           Device device = Device::CPU,
           MemoryOrder order = MemoryOrder::RowMajor);
    Tensor(std::shared_ptr<Storage> storage, const Shape &shape,
           const Strides &strides, DTypes dtype, size_t offset = 0,
           MemoryOrder order = MemoryOrder::RowMajor);

    Tensor(const Tensor &other);
    Tensor &operator=(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;

    // Core attributes
    const Shape &shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return ShapeUtils::size(shape_); }
    const Strides &strides() const { return strides_; }
    size_t itemsize() const { return dtype_size(dtype_); }
    size_t nbytes() const { return size() * itemsize(); }
    MemoryOrder memory_order() const { return memory_order_; }
    DTypes dtype() const { return dtype_; }
    std::string dtype_name() const { return ::axiom::dtype_name(dtype_); }
    Device device() const { return storage_->device(); }
    const TensorFlags &flags() const { return flags_; }
    bool is_contiguous() const { return flags_.c_contiguous; }
    bool is_c_contiguous() const { return flags_.c_contiguous; }
    bool is_f_contiguous() const { return flags_.f_contiguous; }
    std::shared_ptr<Storage> storage() const { return storage_; }
    size_t offset() const { return offset_; }
    bool empty() const { return size() == 0; }

    // View/materialization introspection
    bool is_view() const { return !flags_.owndata || offset_ > 0; }
    bool owns_data() const { return flags_.owndata && offset_ == 0; }
    bool has_zero_stride() const; // True if any stride is 0 (broadcast view)
    bool has_negative_stride() const; // True if any stride < 0 (flipped view)
    bool shares_storage(const Tensor &other) const {
        return storage_.get() == other.storage_.get();
    }

    // Check if an operation would require materialization (data copy)
    bool would_materialize_on_reshape(const Shape &new_shape) const;
    bool would_materialize_on_transpose() const { return !is_contiguous(); }

    // Data access
    void *data();
    const void *data() const;

    Tensor slice(const std::vector<Slice> &slice_args) const;

    Tensor operator[](std::initializer_list<Index> indices) const;

    template <typename T> T *typed_data() {
        if (device() != Device::CPU) {
            throw DeviceError::cpu_only("direct data access");
        }
        return reinterpret_cast<T *>(data());
    }

    template <typename T> const T *typed_data() const {
        if (device() != Device::CPU) {
            throw DeviceError::cpu_only("direct data access");
        }
        return reinterpret_cast<const T *>(data());
    }

    template <typename T> T item(const std::vector<size_t> &indices) const {
        validate_indices(indices);
        if (device() != Device::CPU) {
            throw DeviceError::cpu_only("item()");
        }
        // Compute byte offset with signed arithmetic for negative strides
        int64_t byte_offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            byte_offset += static_cast<int64_t>(indices[i]) * strides_[i];
        }
        const T *data_ptr = reinterpret_cast<const T *>(
            static_cast<const uint8_t *>(data()) + byte_offset);
        return *data_ptr;
    }

    template <typename T>
    void set_item(const std::vector<size_t> &indices, const T &value) {
        validate_indices(indices);
        if (device() != Device::CPU) {
            throw DeviceError::cpu_only("set_item()");
        }
        if (!flags_.writeable) {
            throw MemoryError("Tensor is not writeable");
        }
        // Compute byte offset with signed arithmetic for negative strides
        int64_t byte_offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            byte_offset += static_cast<int64_t>(indices[i]) * strides_[i];
        }
        T *data_ptr =
            reinterpret_cast<T *>(static_cast<uint8_t *>(data()) + byte_offset);
        *data_ptr = value;
    }

    template <typename T> void fill(const T &value) {
        if (device() != Device::CPU) {
            throw DeviceError::cpu_only("fill()");
        }
        if (!flags_.writeable) {
            throw MemoryError("Tensor is not writeable");
        }

        if (is_contiguous()) {
            T *data_ptr = typed_data<T>();
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
    Tensor reshape(const Shape &new_shape,
                   MemoryOrder order = MemoryOrder::RowMajor) const;
    Tensor reshape(std::initializer_list<size_t> new_shape,
                   MemoryOrder order = MemoryOrder::RowMajor) const;
    Tensor
    rearrange(const std::string &pattern,
              const std::map<std::string, size_t> &axis_sizes = {}) const;
    Tensor reduce(const std::string &pattern, const std::string &reduction,
                  const std::map<std::string, size_t> &axis_sizes = {}) const;
    Tensor transpose() const;
    Tensor transpose(const std::vector<int> &axes) const;
    Tensor squeeze(int axis = -1) const;
    Tensor unsqueeze(int axis) const;
    Tensor view(const Shape &new_shape) const;
    Tensor flatten(int start_dim = 0, int end_dim = -1) const;
    Tensor unflatten(int dim, const Shape &sizes) const;

    // NumPy-like aliases and view operations
    Tensor T() const { return transpose(); }
    Tensor ravel() const { return flatten(); }
    Tensor negative() const;
    Tensor flipud() const;
    Tensor fliplr() const;
    Tensor swapaxes(int axis1, int axis2) const;
    Tensor moveaxis(int source, int destination) const;
    Tensor flip(int axis) const;
    Tensor flip(const std::vector<int> &axes) const;
    Tensor rot90(int k = 1, const std::vector<int> &axes = {0, 1}) const;
    Tensor roll(int64_t shift, int axis = -1) const;
    Tensor roll(const std::vector<int64_t> &shifts,
                const std::vector<int> &axes) const;
    Tensor diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) const;
    Tensor trace(int offset = 0, int axis1 = 0, int axis2 = 1) const;

    // Scalar extraction for single-element tensors
    template <typename T> T item() const {
        if (size() != 1) {
            throw ValueError("item() requires tensor with exactly 1 element, "
                             "got " +
                             std::to_string(size()));
        }
        if (device() != Device::CPU) {
            return cpu().item<T>();
        }
        return *reinterpret_cast<const T *>(data());
    }

    // Expand and repeat operations
    // expand: Zero-copy view using 0-stride for broadcasted dims
    // Only works when expanding dims of size 1
    Tensor expand(const Shape &new_shape) const;
    Tensor expand_as(const Tensor &other) const {
        return expand(other.shape());
    }
    Tensor broadcast_to(const Shape &shape) const { return expand(shape); }

    // repeat: Copies data to create repeated tensor
    // Each dim is repeated by the corresponding factor
    Tensor repeat(const std::vector<size_t> &repeats) const;
    Tensor tile(const std::vector<size_t> &reps) const {
        return repeat(reps);
    } // NumPy alias

    // Matrix operations
    Tensor matmul(const Tensor &other, bool transpose_self = false,
                  bool transpose_other = false) const;
    Tensor mm(const Tensor &other) const { return matmul(other); } // Alias
    Tensor dot(const Tensor &other) const {
        return matmul(other);
    } // Alias for vectors

    // Linear algebra shortcuts (see axiom::linalg for full API)
    Tensor det() const;
    Tensor inv() const;

    // =========================================================================
    // Conditional and masking operations (fluent API)
    // =========================================================================

    // where: Returns elements from *this where condition is true, other
    // otherwise Usage: x.where(x > 0, 0.0f) - similar to torch.where but as
    // member function
    Tensor where(const Tensor &condition, const Tensor &other) const;
    Tensor where(const Tensor &condition, float other) const;
    Tensor where(const Tensor &condition, double other) const;
    Tensor where(const Tensor &condition, int32_t other) const;

    // masked_fill: Fill elements where mask is true with value
    // Usage: x.masked_fill(x < 0, 0.0f) - zero out negative values
    Tensor masked_fill(const Tensor &mask, float value) const;
    Tensor masked_fill(const Tensor &mask, double value) const;
    Tensor masked_fill(const Tensor &mask, int32_t value) const;
    Tensor masked_fill(const Tensor &mask, const Tensor &value) const;

    // masked_fill_: In-place version of masked_fill
    Tensor &masked_fill_(const Tensor &mask, float value);
    Tensor &masked_fill_(const Tensor &mask, double value);
    Tensor &masked_fill_(const Tensor &mask, int32_t value);

    // masked_select: Select elements where mask is true
    // Usage: x.masked_select(x > 0) - get all positive values as 1D tensor
    Tensor masked_select(const Tensor &mask) const;

    // =========================================================================
    // Indexing operations (fluent API)
    // =========================================================================

    // gather: Gather values along an axis according to indices
    // Usage: x.gather(dim, indices) - like torch.gather
    Tensor gather(int dim, const Tensor &indices) const;

    // scatter: Scatter values into tensor at indices
    // Usage: x.scatter(dim, indices, src) - like tensor.scatter_
    Tensor scatter(int dim, const Tensor &indices, const Tensor &src) const;

    // scatter_: In-place scatter
    Tensor &scatter_(int dim, const Tensor &indices, const Tensor &src);

    // index_select: Select elements along a dimension using 1D indices
    // Usage: x.index_select(0, indices) - select rows by indices
    Tensor index_select(int dim, const Tensor &indices) const;

    // Reduction member functions
    Tensor sum(int axis = -1, bool keep_dims = false) const;
    Tensor sum(const std::vector<int> &axes, bool keep_dims = false) const;
    Tensor mean(int axis = -1, bool keep_dims = false) const;
    Tensor mean(const std::vector<int> &axes, bool keep_dims = false) const;
    Tensor max(int axis = -1, bool keep_dims = false) const;
    Tensor min(int axis = -1, bool keep_dims = false) const;
    Tensor argmax(int axis = -1, bool keep_dims = false) const;
    Tensor argmin(int axis = -1, bool keep_dims = false) const;

    // Additional reductions
    Tensor prod(int axis = -1, bool keep_dims = false) const;
    Tensor prod(const std::vector<int> &axes, bool keep_dims = false) const;
    Tensor any(int axis = -1, bool keep_dims = false) const;
    Tensor any(const std::vector<int> &axes, bool keep_dims = false) const;
    Tensor all(int axis = -1, bool keep_dims = false) const;
    Tensor all(const std::vector<int> &axes, bool keep_dims = false) const;

    // Statistical operations (composition-based)
    Tensor var(int axis = -1, int ddof = 0, bool keep_dims = false) const;
    Tensor var(const std::vector<int> &axes, int ddof = 0,
               bool keep_dims = false) const;
    Tensor std(int axis = -1, int ddof = 0, bool keep_dims = false) const;
    Tensor std(const std::vector<int> &axes, int ddof = 0,
               bool keep_dims = false) const;
    Tensor ptp(int axis = -1, bool keep_dims = false) const; // peak-to-peak

    // =========================================================================
    // Unary and activation operations (fluent API)
    // =========================================================================

    // Math operations
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor tan() const;

    // NumPy-like math operations
    Tensor sign() const;
    Tensor floor() const;
    Tensor ceil() const;
    Tensor trunc() const;
    Tensor round(int decimals = 0) const;
    Tensor reciprocal() const;
    Tensor square() const;
    Tensor cbrt() const;

    // Element-wise testing operations (return Bool tensor)
    Tensor isnan() const;
    Tensor isinf() const;
    Tensor isfinite() const;

    // Clipping operations
    Tensor clip(const Tensor &min_val, const Tensor &max_val) const;
    Tensor clip(double min_val, double max_val) const;
    Tensor clamp(const Tensor &min_val, const Tensor &max_val) const {
        return clip(min_val, max_val);
    }
    Tensor clamp(double min_val, double max_val) const {
        return clip(min_val, max_val);
    }

    // Activation functions
    Tensor relu() const;
    Tensor leaky_relu(float negative_slope = 0.01f) const;
    Tensor gelu() const;
    Tensor silu() const; // SiLU/Swish: x * sigmoid(x)
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax(int axis = -1) const;
    Tensor log_softmax(int axis = -1) const;

    // Memory operations
    Tensor copy(MemoryOrder order = MemoryOrder::RowMajor) const;
    Tensor clone() const { return copy(); }
    Tensor to(Device device, MemoryOrder order = MemoryOrder::RowMajor) const;
    Tensor cpu() const;
    Tensor gpu() const;

    // Type conversion
    Tensor astype(DTypes new_dtype) const;
    Tensor astype_safe(DTypes new_dtype) const;
    Tensor to_float() const { return astype(Float32()); }
    Tensor to_double() const { return astype(Float64()); }
    Tensor to_int() const { return astype(Int32()); }
    Tensor to_int64() const { return astype(Int64()); }
    Tensor to_bool() const { return astype(Bool()); }
    Tensor to_complex() const { return astype(Complex64()); }
    Tensor to_complex128() const { return astype(Complex128()); }
    Tensor half() const { return astype(Float16()); }

    // Complex number operations
    // real(): Zero-copy view of real components (requires complex input)
    Tensor real() const;
    // imag(): Zero-copy view of imaginary components (requires complex input)
    Tensor imag() const;
    // conj(): Complex conjugate
    Tensor conj() const;

    // Utility methods
    std::string repr() const;
    std::string str() const;
    bool same_shape(const Tensor &other) const;
    bool same_dtype(const Tensor &other) const;
    bool same_device(const Tensor &other) const;
    bool same_memory_order(const Tensor &other) const;

    // Comparison/testing methods (NumPy-like)
    Tensor isclose(const Tensor &other, double rtol = 1e-5,
                   double atol = 1e-8) const;
    bool allclose(const Tensor &other, double rtol = 1e-5,
                  double atol = 1e-8) const;
    bool array_equal(const Tensor &other) const;

    // Safety rails / debugging
    bool has_nan() const;
    bool has_inf() const;
    bool is_finite() const { return !has_nan() && !has_inf(); }
    Tensor &nan_guard();     // Returns *this, throws if NaN detected
    Tensor &assert_finite(); // Returns *this, throws if NaN or Inf detected
    Tensor &assert_shape(
        const std::string &pattern); // e.g. "b h w" or "batch 3 height width"
    Tensor &assert_shape(const Shape &expected);

    // Debug info
    std::string debug_info() const; // Detailed tensor info for debugging

    // File I/O methods
    void save(const std::string &filename) const;

    // Static loading methods (auto-detects format: .axfb, .npy)
    static Tensor load(const std::string &filename,
                       Device device = Device::CPU);

    // Archive methods (for multiple tensors)
    static void save_tensors(const std::map<std::string, Tensor> &tensors,
                             const std::string &filename);
    static std::map<std::string, Tensor>
    load_tensors(const std::string &filename, Device device = Device::CPU);
    static std::vector<std::string>
    list_tensors_in_archive(const std::string &filename);
    static Tensor load_tensor_from_archive(const std::string &filename,
                                           const std::string &tensor_name,
                                           Device device = Device::CPU);

    // Static factory methods
    static Tensor zeros(const Shape &shape, DTypes dtype = Float32(),
                        Device device = Device::CPU,
                        MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor zeros(std::initializer_list<size_t> shape,
                        DTypes dtype = Float32(), Device device = Device::CPU,
                        MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor ones(const Shape &shape, DTypes dtype = Float32(),
                       Device device = Device::CPU,
                       MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor ones(std::initializer_list<size_t> shape,
                       DTypes dtype = Float32(), Device device = Device::CPU,
                       MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor empty(const Shape &shape, DTypes dtype = Float32(),
                        Device device = Device::CPU,
                        MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor empty(std::initializer_list<size_t> shape,
                        DTypes dtype = Float32(), Device device = Device::CPU,
                        MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor eye(size_t n, DTypes dtype = Float32(),
                      Device device = Device::CPU,
                      MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor identity(size_t n, DTypes dtype = Float32(),
                           Device device = Device::CPU,
                           MemoryOrder order = MemoryOrder::RowMajor);
    static Tensor randn(const Shape &shape, DTypes dtype = Float32(),
                        Device device = Device::CPU,
                        MemoryOrder order = MemoryOrder::RowMajor);
    static void manual_seed(uint64_t seed);
    static Tensor arange(int64_t start, int64_t end, int64_t step = 1,
                         DTypes dtype = Int32(), Device device = Device::CPU);
    static Tensor arange(int64_t end, DTypes dtype = Int32(),
                         Device device = Device::CPU);

    // Numerical ranges
    static Tensor linspace(double start, double stop, size_t num = 50,
                           bool endpoint = true, DTypes dtype = Float64(),
                           Device device = Device::CPU);
    static Tensor logspace(double start, double stop, size_t num = 50,
                           bool endpoint = true, double base = 10.0,
                           DTypes dtype = Float64(),
                           Device device = Device::CPU);
    static Tensor geomspace(double start, double stop, size_t num = 50,
                            bool endpoint = true, DTypes dtype = Float64(),
                            Device device = Device::CPU);

    // Like variants (create tensor with same shape/dtype as prototype)
    static Tensor zeros_like(const Tensor &prototype);
    static Tensor ones_like(const Tensor &prototype);
    static Tensor empty_like(const Tensor &prototype);
    template <typename T>
    static Tensor full_like(const Tensor &prototype, const T &value) {
        return full(prototype.shape(), value, prototype.device(),
                    prototype.memory_order());
    }

    // Matrix building
    static Tensor diag(const Tensor &v, int64_t k = 0);
    static Tensor tri(size_t N, size_t M = 0, int64_t k = 0,
                      DTypes dtype = Float64(), Device device = Device::CPU);
    static Tensor tril(const Tensor &m, int64_t k = 0);
    static Tensor triu(const Tensor &m, int64_t k = 0);

    template <typename T>
    static Tensor full(const Shape &shape, const T &value,
                       Device device = Device::CPU,
                       MemoryOrder order = MemoryOrder::RowMajor) {
        // Always create and fill on CPU first, then transfer to target device
        auto tensor = Tensor(shape, dtype_of_v<T>(), Device::CPU, order);
        tensor.fill(value);
        if (device == Device::GPU) {
            return tensor.to(device, order);
        }
        return tensor;
    }

    template <typename T>
    static Tensor from_data(const T *data, const Shape &shape, bool copy = true,
                            MemoryOrder order = MemoryOrder::RowMajor) {
        auto tensor = Tensor(shape, dtype_of_v<T>, Device::CPU, order);
        if (copy) {
            if (order == MemoryOrder::RowMajor) {
                std::memcpy(tensor.typed_data<T>(), data, tensor.nbytes());
            } else {
                auto src_strides = ShapeUtils::calculate_strides(
                    shape, sizeof(T), MemoryOrder::RowMajor);
                auto dst_strides = tensor.strides();
                for (size_t i = 0; i < tensor.size(); ++i) {
                    auto indices = ShapeUtils::unravel_index(i, shape);
                    size_t src_idx =
                        ShapeUtils::linear_index(indices, src_strides) /
                        sizeof(T);
                    size_t dst_idx =
                        ShapeUtils::linear_index(indices, dst_strides) /
                        sizeof(T);
                    tensor.typed_data<T>()[dst_idx] = data[src_idx];
                }
            }
        } else {
            throw RuntimeError::not_implemented("non-copying from_data");
        }
        return tensor;
    }

    template <typename T, size_t N>
    static Tensor from_array(const T (&data)[N], const Shape &shape,
                             DTypes target_dtype = dtype_of_v<T>(),
                             Device device = Device::CPU,
                             MemoryOrder order = MemoryOrder::RowMajor) {
        if (ShapeUtils::size(shape) != N) {
            throw ShapeError("Array size " + std::to_string(N) +
                             " doesn't match tensor shape size " +
                             std::to_string(ShapeUtils::size(shape)));
        }
        auto source_tensor = from_data(data, shape, true, order);
        if (target_dtype.index() != dtype_of_v<T>().index()) {
            source_tensor = source_tensor.astype(target_dtype);
        }
        if (device != Device::CPU) {
            source_tensor = source_tensor.to(device);
        }
        return source_tensor;
    }

    template <typename T> static Tensor asarray(const Tensor &tensor) {
        return tensor.astype(dtype_of_v<T>);
    }

    template <typename T>
    static Tensor asarray(const Tensor &tensor, Device device) {
        return tensor.astype(dtype_of_v<T>).to(device);
    }

    static DTypes result_type(const Tensor &a, const Tensor &b) {
        return type_conversion::promote_dtypes(a.dtype(), b.dtype());
    }

    // =========================================================================
    // Stacking and Concatenation (Static Methods with Pythonic DX)
    // =========================================================================

    // Concatenate tensors along an existing axis
    // Usage: Tensor::concatenate({a, b, c}, axis)
    static Tensor concatenate(const std::vector<Tensor> &tensors, int axis = 0);

    // Short alias for concatenate - mimics np.concatenate / torch.cat
    // Usage: Tensor::cat({a, b, c}, axis)
    static Tensor cat(const std::vector<Tensor> &tensors, int axis = 0) {
        return concatenate(tensors, axis);
    }

    // Initializer list version for even more ergonomic syntax
    // Usage: Tensor::cat({a, b, c})
    static Tensor cat(std::initializer_list<Tensor> tensors, int axis = 0) {
        return concatenate(std::vector<Tensor>(tensors), axis);
    }

    // Stack tensors along a NEW axis (creates a new dimension)
    // Usage: Tensor::stack({a, b, c}, axis)
    static Tensor stack(const std::vector<Tensor> &tensors, int axis = 0);

    // Initializer list version
    static Tensor stack(std::initializer_list<Tensor> tensors, int axis = 0) {
        return stack(std::vector<Tensor>(tensors), axis);
    }

    // Convenience aliases for common stacking patterns (like NumPy)
    // vstack: Stack arrays vertically (row wise) - axis 0
    static Tensor vstack(const std::vector<Tensor> &tensors);
    static Tensor vstack(std::initializer_list<Tensor> tensors) {
        return vstack(std::vector<Tensor>(tensors));
    }

    // hstack: Stack arrays horizontally (column wise) - axis 1
    static Tensor hstack(const std::vector<Tensor> &tensors);
    static Tensor hstack(std::initializer_list<Tensor> tensors) {
        return hstack(std::vector<Tensor>(tensors));
    }

    // dstack: Stack arrays depth-wise (along third axis) - axis 2
    static Tensor dstack(const std::vector<Tensor> &tensors);
    static Tensor dstack(std::initializer_list<Tensor> tensors) {
        return dstack(std::vector<Tensor>(tensors));
    }

    // column_stack: Stack 1D arrays as columns into a 2D array
    static Tensor column_stack(const std::vector<Tensor> &tensors);
    static Tensor column_stack(std::initializer_list<Tensor> tensors) {
        return column_stack(std::vector<Tensor>(tensors));
    }

    // row_stack: Alias for vstack
    static Tensor row_stack(const std::vector<Tensor> &tensors) {
        return vstack(tensors);
    }
    static Tensor row_stack(std::initializer_list<Tensor> tensors) {
        return vstack(std::vector<Tensor>(tensors));
    }

    // Split operations
    // Split into n equal sections
    std::vector<Tensor> split(size_t sections, int axis = 0) const;
    // Split at given indices
    std::vector<Tensor> split(const std::vector<size_t> &indices,
                              int axis = 0) const;
    // Chunk into n parts (may be unequal if not divisible)
    std::vector<Tensor> chunk(size_t n_chunks, int axis = 0) const;

    // vsplit, hsplit, dsplit shortcuts
    std::vector<Tensor> vsplit(size_t sections) const {
        return split(sections, 0);
    }
    std::vector<Tensor> hsplit(size_t sections) const {
        return split(sections, 1);
    }
    std::vector<Tensor> dsplit(size_t sections) const {
        return split(sections, 2);
    }

    // Member function for chaining: a.cat(b, axis)
    Tensor cat(const Tensor &other, int axis = 0) const {
        return concatenate({*this, other}, axis);
    }

    static Tensor ascontiguousarray(const Tensor &tensor) {
        return tensor.ascontiguousarray();
    }

    static Tensor asfortranarray(const Tensor &tensor) {
        return tensor.asfortranarray();
    }
};

std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

// Operator overloads
Tensor operator-(const Tensor &tensor);

} // namespace axiom