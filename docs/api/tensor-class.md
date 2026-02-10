# Tensor Class

`axiom::Tensor` is the central data structure in Axiom. It represents a typed,
multi-dimensional array backed by CPU or Metal GPU storage.

> For a tutorial introduction, see the [User Guide: Tensor Basics](../user-guide/tensor-basics).

**Header:** `#include <axiom/tensor.hpp>`

**Namespace:** `axiom`

---

## Constructors

### Default constructor

```cpp
Tensor();
```

Creates an empty, uninitialized tensor with zero dimensions, `DType::Float32`,
and `Device::CPU`.

---

### Shape constructor

```cpp
Tensor(const Shape& shape,
       DType dtype = DType::Float32,
       Device device = Device::CPU,
       MemoryOrder order = MemoryOrder::RowMajor);
```

Allocates a tensor with the given shape. Storage is allocated but values are
uninitialized.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `shape` | `const Shape&` | Dimension sizes (e.g. `{3, 4}`). `Shape` is `std::vector<size_t>`. |
| `dtype` | `DType` | Element type. Default `DType::Float32`. |
| `device` | `Device` | `Device::CPU` or `Device::GPU`. Default `Device::CPU`. |
| `order` | `MemoryOrder` | `RowMajor` (C) or `ColMajor` (Fortran). Default `RowMajor`. |

```cpp
auto t = axiom::Tensor({3, 4}, axiom::DType::Float64, axiom::Device::CPU);
```

---

### Initializer-list shape constructor

```cpp
Tensor(std::initializer_list<size_t> shape,
       DType dtype = DType::Float32,
       Device device = Device::CPU,
       MemoryOrder order = MemoryOrder::RowMajor);
```

Same as the shape constructor but accepts a brace-enclosed list directly.

```cpp
axiom::Tensor t({2, 3, 4});  // Float32, CPU, RowMajor
```

---

### Storage constructor

```cpp
Tensor(std::shared_ptr<Storage> storage,
       const Shape& shape,
       const Strides& strides,
       DType dtype,
       size_t offset = 0,
       MemoryOrder order = MemoryOrder::RowMajor);
```

Constructs a tensor that wraps existing storage. Used internally to create
views, slices, and transposed tensors that share data.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `storage` | `std::shared_ptr<Storage>` | Shared pointer to backing memory. |
| `shape` | `const Shape&` | Logical shape of this view. |
| `strides` | `const Strides&` | Byte strides per dimension (`Strides` is `std::vector<int64_t>`). |
| `dtype` | `DType` | Element type. |
| `offset` | `size_t` | Byte offset into storage. Default `0`. |
| `order` | `MemoryOrder` | Memory layout. Default `RowMajor`. |

---

### Copy and move constructors

```cpp
Tensor(const Tensor& other);
Tensor& operator=(const Tensor& other);
Tensor(Tensor&& other) noexcept;
Tensor& operator=(Tensor&& other) noexcept;
```

Copy construction shares the underlying storage (reference-counted via
`std::shared_ptr`). No data is duplicated. Move construction transfers
ownership.

```cpp
auto a = axiom::Tensor::ones({3, 3});
auto b = a;          // b shares storage with a
auto c = std::move(a); // a is now empty; c owns the storage
```

---

## Attributes

### shape

```cpp
const Shape& shape() const;
```

Returns the dimension sizes as a `std::vector<size_t>`.

```cpp
auto t = axiom::Tensor({2, 3, 4});
// t.shape() == {2, 3, 4}
```

---

### ndim

```cpp
size_t ndim() const;
```

Returns the number of dimensions (rank). Equivalent to `shape().size()`.

---

### size

```cpp
size_t size() const;
```

Returns the total number of elements. Product of all dimension sizes.

```cpp
auto t = axiom::Tensor({2, 3, 4});
// t.size() == 24
```

---

### strides

```cpp
const Strides& strides() const;
```

Returns the byte strides per dimension as a `std::vector<int64_t>`. Strides may
be negative (flipped views) or zero (broadcast views).

---

### itemsize

```cpp
size_t itemsize() const;
```

Returns the size in bytes of a single element. Equivalent to
`dtype_size(dtype())`.

---

### nbytes

```cpp
size_t nbytes() const;
```

Returns the total byte count of all elements: `size() * itemsize()`.

---

### memory_order

```cpp
MemoryOrder memory_order() const;
```

Returns `MemoryOrder::RowMajor` or `MemoryOrder::ColMajor`.

---

### dtype

```cpp
DType dtype() const;
```

Returns the element data type. Available without materializing lazy tensors.

---

### dtype_name

```cpp
std::string dtype_name() const;
```

Returns a human-readable string for the dtype (e.g. `"float32"`, `"int64"`).

---

### device

```cpp
Device device() const;
```

Returns `Device::CPU` or `Device::GPU`. Available without materializing lazy
tensors.

---

### flags

```cpp
const TensorFlags& flags() const;
```

Returns the `TensorFlags` struct:

```cpp
struct TensorFlags {
    bool writeable = true;
    bool c_contiguous = true;
    bool f_contiguous = false;
    bool aligned = true;
    bool owndata = true;
};
```

---

### is_contiguous / is_c_contiguous / is_f_contiguous

```cpp
bool is_contiguous() const;
bool is_c_contiguous() const;
bool is_f_contiguous() const;
```

`is_contiguous()` and `is_c_contiguous()` both return `true` when elements are
laid out in row-major order without gaps. `is_f_contiguous()` returns `true` for
column-major layout.

---

### storage

```cpp
std::shared_ptr<Storage> storage() const;
```

Returns the underlying storage. Triggers materialization for lazy tensors.

---

### offset

```cpp
size_t offset() const;
```

Returns the byte offset into storage where this tensor's data begins. Non-zero
for views created by slicing.

---

### empty

```cpp
bool empty() const;
```

Returns `true` if the tensor has zero elements (`size() == 0`).

---

## Lazy Evaluation

### is_lazy

```cpp
bool is_lazy() const;
```

Returns `true` if this tensor represents a deferred computation that has not yet
been executed. Accessing data (via `data()`, `item()`, etc.) will trigger
materialization automatically.

---

### lazy_node

```cpp
std::shared_ptr<graph::GraphNode> lazy_node() const;
```

Returns the graph node backing this lazy tensor, or `nullptr` if the tensor is
already materialized.

---

## View and Materialization Introspection

### is_view

```cpp
bool is_view() const;
```

Returns `true` if this tensor does not own its storage outright -- it is a
slice, transpose, or other view into another tensor's memory.

---

### owns_data

```cpp
bool owns_data() const;
```

Returns `true` if this tensor owns its storage and has zero offset. The inverse
of `is_view()` in most cases.

---

### has_zero_stride

```cpp
bool has_zero_stride() const;
```

Returns `true` if any stride is zero, indicating a broadcast view created by
`expand()` or broadcasting.

---

### has_negative_stride

```cpp
bool has_negative_stride() const;
```

Returns `true` if any stride is negative, indicating a flipped view created by
`flip()` or `flipud()`/`fliplr()`.

---

### shares_storage

```cpp
bool shares_storage(const Tensor& other) const;
```

Returns `true` if both tensors point to the same underlying `Storage` object.
Triggers materialization on both tensors if either is lazy.

```cpp
auto a = axiom::Tensor::zeros({4, 4});
auto b = a.transpose();
// a.shares_storage(b) == true
```

---

### would_materialize_on_reshape

```cpp
bool would_materialize_on_reshape(const Shape& new_shape) const;
```

Returns `true` if reshaping to `new_shape` would require a data copy (because
the current strides are not compatible with the new shape).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `new_shape` | `const Shape&` | Target shape to test. |

---

### would_materialize_on_transpose

```cpp
bool would_materialize_on_transpose() const;
```

Returns `true` if the tensor is not contiguous, meaning a transpose would
require a data copy to produce a contiguous result.

---

## Data Access

All data-access methods trigger materialization for lazy tensors. Methods that
read or write raw pointers require `Device::CPU` and throw `DeviceError` for
GPU tensors.

### data

```cpp
void* data();
const void* data() const;
```

Returns a raw pointer to the tensor's data (at the current offset). The pointer
type is untyped; use `typed_data<T>()` for a typed alternative.

---

### typed_data

```cpp
template <typename T>
T* typed_data();

template <typename T>
const T* typed_data() const;
```

Returns a typed pointer to the tensor's data. Throws `DeviceError` if the
tensor is not on CPU.

**Template parameter:** `T` -- the C++ element type (e.g. `float`, `double`,
`int32_t`).

```cpp
auto t = axiom::Tensor::ones({3, 3});
float* ptr = t.typed_data<float>();
```

---

### item (indexed)

```cpp
template <typename T>
T item(const std::vector<size_t>& indices) const;
```

Returns a single element at the given multi-dimensional indices. Respects
strides, including negative strides.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `indices` | `const std::vector<size_t>&` | One index per dimension. |

**Throws:** `IndexError` if any index is out of range. `DeviceError` if not CPU.

```cpp
auto t = axiom::Tensor::ones({3, 4});
float val = t.item<float>({1, 2}); // 1.0f
```

---

### item (scalar extraction)

```cpp
template <typename T>
T item() const;
```

Extracts the single element from a tensor with exactly one element. Throws
`ValueError` if `size() != 1`.

```cpp
auto t = axiom::Tensor::full({1}, 42.0f);
float val = t.item<float>(); // 42.0f
```

---

### set_item

```cpp
template <typename T>
void set_item(const std::vector<size_t>& indices, const T& value);
```

Sets a single element at the given indices.

**Throws:** `IndexError` if out of range. `DeviceError` if not CPU.
`MemoryError` if the tensor is not writeable.

```cpp
auto t = axiom::Tensor::zeros({3, 3});
t.set_item<float>({1, 1}, 5.0f);
```

---

### fill

```cpp
template <typename T>
void fill(const T& value);
```

Sets every element to `value`. Works on both contiguous and non-contiguous
tensors (with a slower fallback path for the latter).

**Throws:** `DeviceError` if not CPU. `MemoryError` if the tensor is not
writeable.

```cpp
auto t = axiom::Tensor({4, 4});
t.fill<float>(3.14f);
```

---

### slice

```cpp
Tensor slice(const std::vector<Slice>& slice_args) const;
```

Returns a view into this tensor defined by the given slices. Each `Slice` has
optional `start`, `stop`, and `step` fields, mirroring Python's `start:stop:step`
syntax.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `slice_args` | `const std::vector<Slice>&` | One `Slice` per dimension. |

**Returns:** A `Tensor` view sharing storage with the original.

```cpp
using axiom::Slice;
auto t = axiom::Tensor::arange(12).reshape({3, 4});
auto row1 = t.slice({Slice(1, 2), Slice()});
```

---

### operator[]

```cpp
Tensor operator[](std::initializer_list<Index> indices) const;
```

Advanced indexing operator. Each `Index` is a `std::variant<int64_t, Slice, TensorIndex>`, supporting integer indexing, slicing, and boolean/integer tensor
indexing in a single call.

```cpp
using axiom::Slice;
auto t = axiom::Tensor::arange(12).reshape({3, 4});
auto col = t[{Slice(), 2}]; // third column
```

---

## Memory Layout

### ascontiguousarray

```cpp
Tensor ascontiguousarray() const;
```

Returns a C-contiguous (row-major) tensor. If the tensor is already
C-contiguous, returns `*this` (no copy). Otherwise copies data into a new
contiguous buffer.

**Alias:** `as_c_contiguous()` -- identical behavior.

---

### asfortranarray

```cpp
Tensor asfortranarray() const;
```

Returns a Fortran-contiguous (column-major) tensor. If already F-contiguous,
returns `*this`. Otherwise copies data.

**Alias:** `as_f_contiguous()` -- identical behavior.

---

## Utility Methods

### repr / str

```cpp
std::string repr() const;
std::string str() const;
```

`repr()` returns a detailed string suitable for debugging, including shape,
dtype, and data values. `str()` returns a compact representation. Both can be
streamed via `operator<<`:

```cpp
auto t = axiom::Tensor::arange(6).reshape({2, 3});
std::cout << t << std::endl;
```

---

### same_shape / same_dtype / same_device

```cpp
bool same_shape(const Tensor& other) const;
bool same_dtype(const Tensor& other) const;
bool same_device(const Tensor& other) const;
```

Element-wise checks that return `true` if the two tensors share the same shape,
dtype, or device respectively.

---

### same_memory_order

```cpp
bool same_memory_order(const Tensor& other) const;
```

Returns `true` if both tensors have the same `MemoryOrder`.

---

### isclose

```cpp
Tensor isclose(const Tensor& other,
               double rtol = 1e-5,
               double atol = 1e-8) const;
```

Returns a boolean tensor where each element is `true` if the corresponding
elements of `*this` and `other` satisfy
`|a - b| <= atol + rtol * |b|`.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `other` | `const Tensor&` | -- | Tensor to compare against. |
| `rtol` | `double` | `1e-5` | Relative tolerance. |
| `atol` | `double` | `1e-8` | Absolute tolerance. |

**Returns:** `Tensor` with `DType::Bool`.

---

### allclose

```cpp
bool allclose(const Tensor& other,
              double rtol = 1e-5,
              double atol = 1e-8) const;
```

Returns `true` if all elements satisfy the `isclose` condition.

---

### array_equal

```cpp
bool array_equal(const Tensor& other) const;
```

Returns `true` if both tensors have the same shape and all elements are exactly
equal (no tolerance).

---

## Safety and Debugging

### has_nan / has_inf / is_finite

```cpp
bool has_nan() const;
bool has_inf() const;
bool is_finite() const;
```

`has_nan()` returns `true` if any element is NaN. `has_inf()` returns `true` if
any element is positive or negative infinity. `is_finite()` returns `true` if no
element is NaN or Inf.

---

### nan_guard

```cpp
Tensor& nan_guard();
```

Checks for NaN values and throws if any are found. Returns `*this` for
chaining.

```cpp
auto result = (a + b).nan_guard();
```

---

### assert_finite

```cpp
Tensor& assert_finite();
```

Throws if any element is NaN or Inf. Returns `*this` for chaining.

---

### assert_shape

```cpp
Tensor& assert_shape(const std::string& pattern);
Tensor& assert_shape(const Shape& expected);
```

Throws if the tensor's shape does not match. The string overload accepts named
dimension patterns such as `"b h w"` or `"batch 3 height width"` where
integers are checked literally and names are treated as wildcards.

**Returns:** `*this` for chaining.

```cpp
auto t = axiom::Tensor::randn({8, 3, 224, 224});
t.assert_shape("batch 3 height width");
```

---

### debug_info

```cpp
std::string debug_info() const;
```

Returns a multi-line string with detailed tensor metadata: shape, strides,
dtype, device, contiguity flags, storage address, offset, and element count.
Intended for diagnostic use.

---

## Stream Operator

```cpp
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
```

Prints the tensor's `str()` representation to an output stream.

---

## Unary Negation Operator

```cpp
Tensor operator-(const Tensor& tensor);  // free function
```

Returns a new tensor with every element negated.

```cpp
auto t = axiom::Tensor::arange(5);
auto neg = -t;
```
