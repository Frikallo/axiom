# Tensor Basics

The `axiom::Tensor` is the fundamental data structure in Axiom. It is a typed,
multi-dimensional array that can live on the CPU or a Metal GPU, supports
zero-copy views through strides and offsets, and automatically manages its own
memory. This page explains the concepts you need to work with tensors
effectively.

## What Is a Tensor?

A tensor is a generalization of scalars, vectors, and matrices to an arbitrary
number of dimensions. Every Axiom tensor carries five pieces of information:

| Property | Accessor | Description |
|----------|----------|-------------|
| **Shape** | `shape()` | A `std::vector<size_t>` giving the size of each dimension. |
| **DType** | `dtype()` | The element type (`DType::Float32`, `DType::Int64`, etc.). |
| **Device** | `device()` | Where the data lives (`Device::CPU` or `Device::GPU`). |
| **Strides** | `strides()` | Byte offsets between consecutive elements along each dimension. |
| **Storage** | `storage()` | A shared pointer to the underlying memory buffer. |

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;

ops::OperationRegistry::initialize_builtin_operations();

auto t = Tensor::zeros({3, 4});
// shape:   {3, 4}
// dtype:   DType::Float32
// device:  Device::CPU
// strides: {16, 4}   (4 bytes per float * 4 columns, 4 bytes per float)
// size:    12 elements
// nbytes:  48 bytes
```

Multiple tensors can share the same storage. When they do, changes through one
tensor are visible through the other -- this is how views work.

---

## Data Types

Axiom supports 14 element types through the `DType` enum:

| DType | C++ Type | Size (bytes) | Category |
|-------|----------|:---:|----------|
| `DType::Bool` | `bool` | 1 | Boolean |
| `DType::Int8` | `int8_t` | 1 | Signed integer |
| `DType::Int16` | `int16_t` | 2 | Signed integer |
| `DType::Int32` | `int32_t` | 4 | Signed integer |
| `DType::Int64` | `int64_t` | 8 | Signed integer |
| `DType::UInt8` | `uint8_t` | 1 | Unsigned integer |
| `DType::UInt16` | `uint16_t` | 2 | Unsigned integer |
| `DType::UInt32` | `uint32_t` | 4 | Unsigned integer |
| `DType::UInt64` | `uint64_t` | 8 | Unsigned integer |
| `DType::Float16` | `float16_t` | 2 | Floating point |
| `DType::Float32` | `float` | 4 | Floating point |
| `DType::Float64` | `double` | 8 | Floating point |
| `DType::Complex64` | `std::complex<float>` | 8 | Complex |
| `DType::Complex128` | `std::complex<double>` | 16 | Complex |

**`DType::Float32` is the default.** Factory methods like `Tensor::zeros`,
`Tensor::ones`, and `Tensor::randn` all produce `Float32` tensors unless you
explicitly request a different type.

```cpp
auto a = Tensor::zeros({3, 4});                      // Float32
auto b = Tensor::ones({3, 4}, DType::Float64);       // Float64
auto c = Tensor::randn({3, 4}, DType::Complex64);    // Complex64
```

You can query the type at runtime:

```cpp
DType dt = a.dtype();              // DType::Float32
std::string name = a.dtype_name(); // "Float32"
size_t bytes = a.itemsize();       // 4
```

Category helpers are available for generic code:

```cpp
is_floating_dtype(DType::Float32);      // true
is_integer_dtype(DType::Int64);         // true
is_complex_dtype(DType::Complex128);    // true
is_signed_integer_dtype(DType::UInt8);  // false
```

---

## Shapes and Dimensions

The **shape** of a tensor is a vector of dimension sizes. Related accessors:

| Accessor | Returns | Description |
|----------|---------|-------------|
| `shape()` | `const Shape&` | The dimension sizes (e.g. `{2, 3, 4}`). |
| `ndim()` | `size_t` | Number of dimensions (`shape().size()`). |
| `size()` | `size_t` | Total number of elements (product of shape). |
| `empty()` | `bool` | True if any dimension is zero. |

```cpp
auto t = Tensor::zeros({2, 3, 4});
t.shape();  // {2, 3, 4}
t.ndim();   // 3
t.size();   // 24
t.empty();  // false
```

### Scalar tensors

A scalar tensor has shape `{}` (zero dimensions) and contains exactly one
element. You can extract its value with the `item<T>()` method:

```cpp
auto s = Tensor::full({}, 3.14f);
s.ndim();           // 0
s.size();           // 1
s.item<float>();    // 3.14f
```

### Empty tensors

A tensor with a zero in any dimension is empty -- it contains no data:

```cpp
auto e = Tensor::zeros({3, 0, 4});
e.size();   // 0
e.empty();  // true
```

---

## Memory Layout

Axiom tensors use a strided memory model. The **strides** vector records the
byte offset you must advance to reach the next element along each dimension.

### Row-major vs. column-major

By default, tensors are stored in **row-major** (C-style) order: the last
dimension varies fastest in memory. Column-major (Fortran-style) order is also
supported.

```cpp
// Row-major (default) -- last dimension is contiguous
auto c = Tensor::zeros({3, 4}, DType::Float32,
                       Device::CPU, MemoryOrder::RowMajor);
// strides: {16, 4}  (row stride = 4 floats * 4 bytes, col stride = 4 bytes)

// Column-major -- first dimension is contiguous
auto f = Tensor::zeros({3, 4}, DType::Float32,
                       Device::CPU, MemoryOrder::ColMajor);
// strides: {4, 12}  (row stride = 4 bytes, col stride = 3 floats * 4 bytes)
```

`MemoryOrder::RowMajor` is the default for all factory methods, constructors,
and operations like `copy()` and `reshape()`.

### Contiguity

A tensor is **contiguous** when its elements occupy a single, unbroken block of
memory with no gaps or overlaps. Axiom tracks contiguity with two flags:

| Method | Meaning |
|--------|---------|
| `is_contiguous()` | Alias for `is_c_contiguous()`. |
| `is_c_contiguous()` | Row-major contiguous (C order). |
| `is_f_contiguous()` | Column-major contiguous (Fortran order). |

Views created by `transpose()`, `slice()`, or `expand()` are often
non-contiguous. You can force contiguity with:

```cpp
auto contig = t.ascontiguousarray();  // C order copy (if needed)
auto fortran = t.asfortranarray();    // Fortran order copy (if needed)
```

---

## Views vs. Copies

Many shape-manipulation methods return a **view** -- a new `Tensor` object that
shares the same underlying storage as the original. Views are cheap: they
involve no data copying and no new memory allocation. They work by adjusting
the shape, strides, and offset.

### Operations that create views

- `reshape()` (when the tensor is contiguous)
- `transpose()` / `T()`
- `squeeze()` / `unsqueeze()`
- `expand()` / `broadcast_to()`
- `slice()` / `operator[]`
- `flatten()` (when contiguous)
- `swapaxes()` / `moveaxis()`
- `flip()` / `flipud()` / `fliplr()`
- `real()` / `imag()` (on complex tensors)

### Operations that create copies

- `copy()` / `clone()`
- `ascontiguousarray()` (if not already contiguous)
- `astype()` (always allocates new storage)
- `reshape()` when the tensor is non-contiguous
- Arithmetic and math operations (`+`, `*`, `exp()`, etc.)

### Detecting views

```cpp
auto a = Tensor::zeros({3, 4});
auto b = a.reshape({4, 3});     // View -- shares storage with a

b.is_view();          // true
b.owns_data();        // false
a.shares_storage(b);  // true
```

Because views share storage, writing through one tensor modifies the other:

```cpp
auto a = Tensor::zeros({3, 4});
auto b = a.reshape({4, 3});

b.set_item<float>({0, 0}, 99.0f);
a.item<float>({0, 0});  // 99.0f -- same memory
```

If you need an independent copy, call `copy()` or `clone()`:

```cpp
auto c = a.copy();       // Deep copy -- independent storage
c.shares_storage(a);     // false
```

### Predicting reshape behavior

You can check whether a reshape would require a copy before performing it:

```cpp
auto t = Tensor::zeros({3, 4}).transpose();
t.would_materialize_on_reshape({4, 3});  // true (non-contiguous)
```

---

## Devices

Axiom tensors live on one of two devices:

| Device | Description |
|--------|-------------|
| `Device::CPU` | System RAM. All operations supported. |
| `Device::GPU` | Metal GPU on Apple Silicon. Hardware-accelerated operations. |

### Moving between devices

```cpp
auto cpu_t = Tensor::randn({256, 256});

// CPU -> GPU
auto gpu_t = cpu_t.gpu();

// GPU -> CPU
auto back = gpu_t.cpu();

// Generic transfer
auto target = cpu_t.to(Device::GPU);
```

On Apple Silicon, CPU and GPU share unified memory, so transfers between
devices are efficient. Operations on a tensor automatically execute on the
device where that tensor resides.

### Creating tensors on a specific device

Every factory method accepts an optional `Device` parameter:

```cpp
auto gpu_zeros = Tensor::zeros({1024, 1024}, DType::Float32, Device::GPU);
auto gpu_randn = Tensor::randn({64, 128}, DType::Float32, Device::GPU);
auto gpu_eye   = Tensor::eye(256, DType::Float32, Device::GPU);
```

### Device queries

```cpp
auto t = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
t.device();                    // Device::GPU
t.same_device(other_tensor);   // true/false
```

Note: Direct data access via `data()`, `typed_data<T>()`, `item()`, and
`set_item()` is only available for CPU tensors. Attempting these on a GPU
tensor throws a `DeviceError`. Move the tensor to CPU first if you need
element-level access.

---

## Type Conversion

### astype and astype_safe

`astype()` converts a tensor to a new dtype. It always returns a new tensor
with freshly allocated storage:

```cpp
auto ints = Tensor::arange(5);             // Int32
auto floats = ints.astype(DType::Float64); // Float64 copy
```

`astype_safe()` does the same but checks for potential precision loss and
warns or throws if the conversion is lossy (e.g., `Float64` to `Float32`,
`Int64` to `Int32`):

```cpp
auto safe = ints.astype_safe(DType::Float64);  // OK -- no loss
```

### Convenience methods

For common conversions, named methods are provided:

```cpp
auto t = Tensor::arange(6);

t.to_float();       // -> Float32
t.to_double();      // -> Float64
t.to_int();         // -> Int32
t.to_int64();       // -> Int64
t.to_bool();        // -> Bool
t.to_complex();     // -> Complex64
t.to_complex128();  // -> Complex128
t.half();           // -> Float16
```

### Type promotion rules

When two tensors with different dtypes interact in a binary operation, Axiom
promotes both to a common type following NumPy-compatible rules:

- **Bool** is promoted to the other operand's type.
- **Integer + Float** promotes to the float type.
- **Smaller + Larger** promotes to the larger type within the same category.
- **Real + Complex** promotes to the complex type.

```cpp
auto i = Tensor::arange(5);                   // Int32
auto f = Tensor::ones({5}, DType::Float64);   // Float64
auto result = i + f;                           // Float64 (promoted)
```

You can query the result type without performing the operation:

```cpp
DType dt = Tensor::result_type(i, f);  // DType::Float64
```

---

## Creating Tensors

Axiom provides a range of static factory methods on the `Tensor` class. Here
is a brief overview of the most common ones:

```cpp
// Constant-filled tensors
auto z = Tensor::zeros({3, 4});
auto o = Tensor::ones({3, 4});
auto f = Tensor::full({3, 4}, 2.5f);
auto e = Tensor::empty({3, 4});        // Uninitialized (fast)

// Identity and diagonal matrices
auto eye = Tensor::eye(4);
auto id  = Tensor::identity(4);        // Alias for eye

// Numerical ranges
auto a = Tensor::arange(10);                   // [0, 1, ..., 9]
auto b = Tensor::arange(2, 10, 2);             // [2, 4, 6, 8]
auto c = Tensor::linspace(0.0, 1.0, 5);        // [0.0, 0.25, 0.5, 0.75, 1.0]
auto d = Tensor::logspace(-2.0, 2.0, 5);       // 10^(-2) to 10^2
auto g = Tensor::geomspace(1.0, 1000.0, 4);    // [1, 10, 100, 1000]

// Random tensors
auto r1 = Tensor::randn({3, 4});       // Standard normal
auto r2 = Tensor::rand({3, 4});        // Uniform [0, 1)
auto r3 = Tensor::randint(0, 10, {3, 4}); // Random integers in [0, 10)

// From existing data
float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
auto from_ptr   = Tensor::from_data(data, {2, 3});
auto from_array = Tensor::from_array(data, {2, 3});

// "Like" variants -- match shape and dtype of an existing tensor
auto zl = Tensor::zeros_like(r1);
auto ol = Tensor::ones_like(r1);
auto el = Tensor::empty_like(r1);
auto rl = Tensor::randn_like(r1);
```

All factory methods accept optional `DType`, `Device`, and `MemoryOrder`
parameters after the shape.

For complete function signatures, see [API Reference: Tensor Class](../api/tensor-class) and [API Reference: Data Types](../api/dtypes).
