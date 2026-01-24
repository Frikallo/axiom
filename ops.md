# Axiom Operations Reference

This document lists all operations available in Axiom, organized by category.

## Tensor Creation

| Function | Description | Example |
|----------|-------------|---------|
| `Tensor::zeros(shape)` | Create tensor filled with zeros | `Tensor::zeros({3, 4})` |
| `Tensor::ones(shape)` | Create tensor filled with ones | `Tensor::ones({3, 4})` |
| `Tensor::empty(shape)` | Create uninitialized tensor | `Tensor::empty({3, 4})` |
| `Tensor::full(shape, value)` | Create tensor filled with value | `Tensor::full({3, 4}, 3.14f)` |
| `Tensor::eye(n)` | Create identity matrix | `Tensor::eye(4)` |
| `Tensor::identity(n)` | Alias for eye | `Tensor::identity(4)` |
| `Tensor::randn(shape)` | Create tensor with random normal values | `Tensor::randn({3, 4})` |
| `Tensor::arange(start, end, step)` | Create range tensor | `Tensor::arange(0, 10, 2)` |
| `Tensor::from_data(ptr, shape)` | Create from raw data pointer | `Tensor::from_data(data, {3, 4})` |
| `Tensor::from_array(arr, shape)` | Create from C array | `Tensor::from_array(arr, {3, 4})` |

## Shape Manipulation

| Function | Description | Example |
|----------|-------------|---------|
| `tensor.reshape(new_shape)` | Reshape tensor (returns view if possible) | `t.reshape({6, 2})` |
| `tensor.view(new_shape)` | Create view with new shape (contiguous only) | `t.view({6, 2})` |
| `tensor.flatten(start, end)` | Flatten dimensions (zero-copy view if possible) | `t.flatten()` |
| `tensor.transpose()` | Swap last two dimensions | `t.transpose()` |
| `tensor.transpose(axes)` | Permute dimensions | `t.transpose({2, 0, 1})` |
| `tensor.squeeze(axis)` | Remove dimensions of size 1 | `t.squeeze(0)` |
| `tensor.unsqueeze(axis)` | Add dimension of size 1 | `t.unsqueeze(0)` |
| `tensor.rearrange(pattern)` | Einops-style reshape | `t.rearrange("b c h w -> b (c h) w")` |

## Expand and Repeat

| Function | Description | Zero-Copy | Example |
|----------|-------------|-----------|---------|
| `tensor.expand(shape)` | Expand dims of size 1 using 0-stride | Yes | `t.expand({64, 128, 256})` |
| `tensor.repeat(repeats)` | Repeat tensor by copying data | No | `t.repeat({2, 3, 1})` |
| `tensor.tile(reps)` | Alias for repeat (NumPy style) | No | `t.tile({2, 2})` |

**expand vs repeat:**
- `expand()` creates a **zero-cost view** by setting stride to 0 for broadcast dimensions
- `repeat()` **copies data** - use when views are unsafe or you need contiguous output

```cpp
// expand: zero-copy broadcast (size-1 dims only)
auto a = Tensor::ones({1, 64});
auto b = a.expand({128, 64});  // Shape: (128, 64), no data copy

// repeat: copies data
auto c = a.repeat({128, 1});   // Same shape, but data is copied
```

## Memory Operations

| Function | Description | Example |
|----------|-------------|---------|
| `tensor.copy()` | Create a copy of tensor | `t.copy()` |
| `tensor.clone()` | Alias for copy | `t.clone()` |
| `tensor.to(device)` | Move to device (CPU/GPU) | `t.to(Device::GPU)` |
| `tensor.cpu()` | Move to CPU | `t.cpu()` |
| `tensor.gpu()` | Move to GPU | `t.gpu()` |
| `tensor.ascontiguousarray()` | Make C-contiguous | `t.ascontiguousarray()` |
| `tensor.asfortranarray()` | Make Fortran-contiguous | `t.asfortranarray()` |

## Type Conversion

| Function | Description | Example |
|----------|-------------|---------|
| `tensor.astype(dtype)` | Convert to dtype | `t.astype(DType::Float32)` |
| `tensor.astype_safe(dtype)` | Safe conversion (throws if lossy) | `t.astype_safe(DType::Int32)` |
| `tensor.to_float()` | Convert to Float32 | `t.to_float()` |
| `tensor.to_double()` | Convert to Float64 | `t.to_double()` |
| `tensor.to_int()` | Convert to Int32 | `t.to_int()` |
| `tensor.to_int64()` | Convert to Int64 | `t.to_int64()` |
| `tensor.to_bool()` | Convert to Bool | `t.to_bool()` |

## Indexing and Slicing

| Function | Description | Example |
|----------|-------------|---------|
| `tensor[{indices}]` | Multi-dimensional indexing | `t[{0, Slice(1, 3)}]` |
| `tensor.slice(slices)` | Slice tensor | `t.slice({Slice(0, 2), Slice()})` |

---

## Binary Arithmetic Operations

All binary operations support broadcasting.

| Function | Operator | Description | Example |
|----------|----------|-------------|---------|
| `ops::add(a, b)` | `a + b` | Element-wise addition | `a + b` |
| `ops::subtract(a, b)` | `a - b` | Element-wise subtraction | `a - b` |
| `ops::multiply(a, b)` | `a * b` | Element-wise multiplication | `a * b` |
| `ops::divide(a, b)` | `a / b` | Element-wise division | `a / b` |
| `ops::power(a, b)` | - | Element-wise power | `ops::power(a, b)` |
| `ops::modulo(a, b)` | `a % b` | Element-wise modulo | `a % b` |

### In-Place Variants

| Operator | Description |
|----------|-------------|
| `a += b` | In-place addition |
| `a -= b` | In-place subtraction |
| `a *= b` | In-place multiplication |
| `a /= b` | In-place division |

---

## Comparison Operations

Return boolean tensors.

| Function | Operator | Description |
|----------|----------|-------------|
| `ops::equal(a, b)` | `a == b` | Element-wise equality |
| `ops::not_equal(a, b)` | `a != b` | Element-wise inequality |
| `ops::less(a, b)` | `a < b` | Element-wise less than |
| `ops::less_equal(a, b)` | `a <= b` | Element-wise less or equal |
| `ops::greater(a, b)` | `a > b` | Element-wise greater than |
| `ops::greater_equal(a, b)` | `a >= b` | Element-wise greater or equal |

---

## Logical Operations

| Function | Operator | Description |
|----------|----------|-------------|
| `ops::logical_and(a, b)` | `a && b` | Element-wise logical AND |
| `ops::logical_or(a, b)` | `a \|\| b` | Element-wise logical OR |
| `ops::logical_xor(a, b)` | - | Element-wise logical XOR |

---

## Bitwise Operations

Integer types only.

| Function | Operator | Description |
|----------|----------|-------------|
| `ops::bitwise_and(a, b)` | `a & b` | Bitwise AND |
| `ops::bitwise_or(a, b)` | `a \| b` | Bitwise OR |
| `ops::bitwise_xor(a, b)` | `a ^ b` | Bitwise XOR |
| `ops::left_shift(a, b)` | `a << b` | Left shift |
| `ops::right_shift(a, b)` | `a >> b` | Right shift |

---

## Math Binary Operations

| Function | Description | Example |
|----------|-------------|---------|
| `ops::maximum(a, b)` | Element-wise maximum | `ops::maximum(a, b)` |
| `ops::minimum(a, b)` | Element-wise minimum | `ops::minimum(a, b)` |
| `ops::atan2(y, x)` | Element-wise atan2 | `ops::atan2(y, x)` |
| `ops::hypot(a, b)` | Element-wise hypotenuse | `ops::hypot(a, b)` |

---

## Unary Operations

| Function | Operator | Description |
|----------|----------|-------------|
| `ops::negate(a)` | `-a` | Negation |
| `ops::abs(a)` | - | Absolute value |
| `ops::sqrt(a)` | - | Square root |
| `ops::exp(a)` | - | Exponential |
| `ops::log(a)` | - | Natural logarithm |
| `ops::sin(a)` | - | Sine |
| `ops::cos(a)` | - | Cosine |
| `ops::tan(a)` | - | Tangent |

---

## Reduction Operations

### Free Functions

| Function | Description | Example |
|----------|-------------|---------|
| `ops::sum(a, axis, keep_dims)` | Sum over axis | `ops::sum(a, {0}, true)` |
| `ops::mean(a, axis, keep_dims)` | Mean over axis | `ops::mean(a, {0})` |
| `ops::max(a, axis, keep_dims)` | Maximum over axis | `ops::max(a, {-1})` |
| `ops::min(a, axis, keep_dims)` | Minimum over axis | `ops::min(a)` |
| `ops::argmax(a, axis, keep_dims)` | Index of maximum | `ops::argmax(a, 0)` |
| `ops::argmin(a, axis, keep_dims)` | Index of minimum | `ops::argmin(a, -1)` |

### Member Functions

| Function | Description | Example |
|----------|-------------|---------|
| `tensor.sum(axis, keep_dims)` | Sum over axis | `t.sum(0)` |
| `tensor.sum(axes, keep_dims)` | Sum over multiple axes | `t.sum({0, 1})` |
| `tensor.mean(axis, keep_dims)` | Mean over axis | `t.mean(-1)` |
| `tensor.max(axis, keep_dims)` | Maximum over axis | `t.max(0)` |
| `tensor.min(axis, keep_dims)` | Minimum over axis | `t.min()` |
| `tensor.argmax(axis, keep_dims)` | Index of maximum (returns Int64) | `t.argmax(1)` |
| `tensor.argmin(axis, keep_dims)` | Index of minimum (returns Int64) | `t.argmin(-1)` |

**Parameters:**
- `axis`: Dimension to reduce. `-1` reduces all dimensions (default).
- `axes`: Vector of dimensions to reduce.
- `keep_dims`: If true, reduced dimensions become size 1 instead of being removed.

---

## Matrix Multiplication

| Function | Description | Example |
|----------|-------------|---------|
| `a.matmul(b)` | Matrix multiplication (member) | `a.matmul(b)` |
| `a.matmul(b, transpose_a, transpose_b)` | MatMul with optional transpose | `a.matmul(b, false, true)` |
| `a.mm(b)` | Alias for matmul | `a.mm(b)` |
| `a.dot(b)` | Alias for matmul (vectors) | `a.dot(b)` |
| `ops::matmul(a, b)` | Matrix multiplication (free function) | `ops::matmul(a, b)` |
| `ops::matmul(a, b, transpose_a, transpose_b)` | MatMul with optional transpose | `ops::matmul(a, b, false, true)` |

### Supported Shapes

| A Shape | B Shape | Result Shape | Description |
|---------|---------|--------------|-------------|
| `(M, K)` | `(K, N)` | `(M, N)` | Standard 2D matmul |
| `(K,)` | `(K, N)` | `(N,)` | Vector-matrix multiply |
| `(M, K)` | `(K,)` | `(M,)` | Matrix-vector multiply |
| `(K,)` | `(K,)` | `()` | Dot product |
| `(..., M, K)` | `(..., K, N)` | `(..., M, N)` | Batched matmul |

### Transpose Flags

The `transpose_a` and `transpose_b` flags allow **zero-copy** transposed matrix multiplication:

```cpp
// Instead of:
auto result = ops::matmul(a.transpose(), b);  // Creates transposed copy

// Use:
auto result = ops::matmul(a, b, true, false);  // Zero-copy transpose
```

### Batch Broadcasting

Batch dimensions are automatically broadcast:

```cpp
auto a = Tensor::randn({2, 1, 4, 3});  // Shape: (2, 1, 4, 3)
auto b = Tensor::randn({3, 5});        // Shape: (3, 5)
auto c = ops::matmul(a, b);            // Shape: (2, 1, 4, 5)

auto a2 = Tensor::randn({2, 1, 4, 3});
auto b2 = Tensor::randn({1, 8, 3, 5});
auto c2 = ops::matmul(a2, b2);         // Shape: (2, 8, 4, 5)
```

---

## Backend Support

| Operation Category | CPU | Metal GPU |
|-------------------|-----|-----------|
| Binary Arithmetic | Yes | Yes (float, int, half, uint8, int8) |
| Comparison | Yes | No |
| Logical | Yes | No |
| Bitwise | Yes | No |
| Math Binary | Yes | No |
| Unary | Yes | Yes (float, half) |
| Reduction | Yes | Yes (float, half) |
| Matrix Multiplication | Yes | Yes (float, half) |

---

## Data Types

| DType | C++ Type | Size |
|-------|----------|------|
| `DType::Bool` | `bool` | 1 |
| `DType::Int8` | `int8_t` | 1 |
| `DType::Int16` | `int16_t` | 2 |
| `DType::Int32` | `int32_t` | 4 |
| `DType::Int64` | `int64_t` | 8 |
| `DType::UInt8` | `uint8_t` | 1 |
| `DType::UInt16` | `uint16_t` | 2 |
| `DType::UInt32` | `uint32_t` | 4 |
| `DType::UInt64` | `uint64_t` | 8 |
| `DType::Float16` | `float16_t` | 2 |
| `DType::Float32` | `float` | 4 |
| `DType::Float64` | `double` | 8 |
| `DType::Complex64` | `complex64_t` | 8 |
| `DType::Complex128` | `complex128_t` | 16 |

---

## File I/O

| Function | Description |
|----------|-------------|
| `tensor.save(filename)` | Save tensor to file |
| `Tensor::load(filename)` | Load tensor from file |
| `Tensor::save_tensors(map, filename)` | Save multiple tensors |
| `Tensor::load_tensors(filename)` | Load multiple tensors |
| `Tensor::list_tensors_in_archive(filename)` | List tensors in archive |
