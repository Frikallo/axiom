# Axiom Operations Reference

This document lists all operations available in Axiom, organized by category.

## Test Coverage Status

This document is now maintained with test coverage tracking. Operations marked with:
- ✅ **Fully tested** - Comprehensive test coverage exists
- ⚠️ **Partially tested** - Some test coverage, but edge cases or variants may be missing
- ❌ **Not tested** - No dedicated test coverage exists

Last updated: 2026-01-24

## Tensor Creation

| Function | Description | Example | Status |
|----------|-------------|---------|--------|
| `Tensor::zeros(shape)` | Create tensor filled with zeros | `Tensor::zeros({3, 4})` | ✅ |
| `Tensor::ones(shape)` | Create tensor filled with ones | `Tensor::ones({3, 4})` | ✅ |
| `Tensor::empty(shape)` | Create uninitialized tensor | `Tensor::empty({3, 4})` | ✅ |
| `Tensor::full(shape, value)` | Create tensor filled with value | `Tensor::full({3, 4}, 3.14f)` | ✅ |
| `Tensor::eye(n)` | Create identity matrix | `Tensor::eye(4)` | ✅ |
| `Tensor::identity(n)` | Alias for eye | `Tensor::identity(4)` | ✅ |
| `Tensor::randn(shape)` | Create tensor with random normal values | `Tensor::randn({3, 4})` | ✅ |
| `Tensor::arange(start, end, step)` | Create range tensor | `Tensor::arange(0, 10, 2)` | ✅ |
| `Tensor::from_data(ptr, shape)` | Create from raw data pointer | `Tensor::from_data(data, {3, 4})` | ✅ |
| `Tensor::from_array(arr, shape)` | Create from C array | `Tensor::from_array(arr, {3, 4})` | ❌ |

## Random Number Generation

**Status: ✅ Fully tested (8/8 tests passing)**

Axiom uses the PCG (Permuted Congruential Generator) algorithm for random number generation. PCG provides excellent statistical properties, small state size (128 bits), and is suitable for parallel workloads.

### Seeding

| Function | Description | Example |
|----------|-------------|---------|
| `Tensor::manual_seed(seed)` | Set RNG seed for reproducibility | `Tensor::manual_seed(42)` |
| `axiom::manual_seed(seed)` | Free function alias | `axiom::manual_seed(42)` |
| `axiom::get_seed()` | Get current seed value | `auto seed = axiom::get_seed()` |

### Random Tensor Creation

| Function | Description | Example |
|----------|-------------|---------|
| `Tensor::randn(shape)` | Normal distribution (mean=0, std=1) | `Tensor::randn({3, 4})` |

### Direct RNG Access

For advanced use cases, access the `RandomGenerator` directly:

```cpp
auto& rng = axiom::RandomGenerator::instance();

// Generate values
float normal_val = rng.normal<float>(0.0f, 1.0f);  // mean, stddev
double uniform_val = rng.uniform<double>(0.0, 1.0); // low, high
int64_t int_val = rng.randint(0, 100);              // [low, high)
uint64_t raw = rng.random_uint64();                 // raw 64-bit value
double unit = rng.random_double();                  // [0, 1)
```

### Reproducibility

```cpp
// Same seed produces identical sequences
Tensor::manual_seed(42);
auto a = Tensor::randn({3, 3});

Tensor::manual_seed(42);
auto b = Tensor::randn({3, 3});  // b == a
```

**Note:** The RNG state is thread-local. Each thread has its own independent generator.

## Shape Manipulation

| Function | Description | Example | Status |
|----------|-------------|---------|--------|
| `tensor.reshape(new_shape)` | Reshape tensor (returns view if possible) | `t.reshape({6, 2})` | ✅ |
| `tensor.view(new_shape)` | Create view with new shape (contiguous only) | `t.view({6, 2})` | ✅ |
| `tensor.flatten(start, end)` | Flatten dimensions (zero-copy view if possible) | `t.flatten()` | ✅ |
| `tensor.transpose()` | Swap last two dimensions | `t.transpose()` | ✅ |
| `tensor.transpose(axes)` | Permute dimensions | `t.transpose({2, 0, 1})` | ✅ |
| `tensor.squeeze(axis)` | Remove dimensions of size 1 | `t.squeeze(0)` | ✅ |
| `tensor.unsqueeze(axis)` | Add dimension of size 1 | `t.unsqueeze(0)` | ✅ |
| `tensor.rearrange(pattern)` | Einops-style reshape | `t.rearrange("b c h w -> b (c h) w")` | ✅ |

## Expand and Repeat

**Status: ✅ Fully tested**

| Function | Description | Zero-Copy | Example | Status |
|----------|-------------|-----------|---------|--------|
| `tensor.expand(shape)` | Expand dims of size 1 using 0-stride | Yes | `t.expand({64, 128, 256})` | ✅ |
| `tensor.repeat(repeats)` | Repeat tensor by copying data | No | `t.repeat({2, 3, 1})` | ✅ |
| `tensor.tile(reps)` | Alias for repeat (NumPy style) | No | `t.tile({2, 2})` | ✅ |

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

**Status: ✅ Full GPU support via MPSGraph**

All binary operations support broadcasting.

| Function | Operator | Description | CPU | GPU |
|----------|----------|-------------|-----|-----|
| `ops::add(a, b)` | `a + b` | Element-wise addition | ✅ | ✅ |
| `ops::subtract(a, b)` | `a - b` | Element-wise subtraction | ✅ | ✅ |
| `ops::multiply(a, b)` | `a * b` | Element-wise multiplication | ✅ | ✅ |
| `ops::divide(a, b)` | `a / b` | Element-wise division | ✅ | ✅ |
| `ops::power(a, b)` | - | Element-wise power | ✅ | ✅ |
| `ops::modulo(a, b)` | `a % b` | Element-wise modulo | ✅ | ✅ |

### In-Place Variants

| Operator | Description |
|----------|-------------|
| `a += b` | In-place addition |
| `a -= b` | In-place subtraction |
| `a *= b` | In-place multiplication |
| `a /= b` | In-place division |

---

## Comparison Operations

**Status: ✅ Fully tested (20/20 tests passing - includes GPU tests!)**

Return boolean tensors. **Full GPU support via MPSGraph.**

| Function | Operator | Description | CPU | GPU | Status |
|----------|----------|-------------|-----|-----|--------|
| `ops::equal(a, b)` | `a == b` | Element-wise equality | ✅ | ✅ | ✅ |
| `ops::not_equal(a, b)` | `a != b` | Element-wise inequality | ✅ | ✅ | ✅ |
| `ops::less(a, b)` | `a < b` | Element-wise less than | ✅ | ✅ | ✅ |
| `ops::less_equal(a, b)` | `a <= b` | Element-wise less or equal | ✅ | ✅ | ✅ |
| `ops::greater(a, b)` | `a > b` | Element-wise greater than | ✅ | ✅ | ✅ |
| `ops::greater_equal(a, b)` | `a >= b` | Element-wise greater or equal | ✅ | ✅ | ✅ |

---

## Logical Operations

**Status: ✅ GPU support via MPSGraph**

| Function | Operator | Description | CPU | GPU |
|----------|----------|-------------|-----|-----|
| `ops::logical_and(a, b)` | `a && b` | Element-wise logical AND | ✅ | ✅ |
| `ops::logical_or(a, b)` | `a \|\| b` | Element-wise logical OR | ✅ | ✅ |
| `ops::logical_xor(a, b)` | - | Element-wise logical XOR | ✅ | ✅ |
| `ops::logical_not(a)` | - | Element-wise logical NOT | ✅ | ✅ |

---

## Bitwise Operations

**Status: ✅ GPU support via MPSGraph**

Integer types only.

| Function | Operator | Description | CPU | GPU |
|----------|----------|-------------|-----|-----|
| `ops::bitwise_and(a, b)` | `a & b` | Bitwise AND | ✅ | ✅ |
| `ops::bitwise_or(a, b)` | `a \| b` | Bitwise OR | ✅ | ✅ |
| `ops::bitwise_xor(a, b)` | `a ^ b` | Bitwise XOR | ✅ | ✅ |
| `ops::left_shift(a, b)` | `a << b` | Left shift | ✅ | ✅ |
| `ops::right_shift(a, b)` | `a >> b` | Right shift | ✅ | ✅ |

---

## Math Binary Operations

**Status: ✅ Full GPU support via MPSGraph**

| Function | Description | Example | CPU | GPU |
|----------|-------------|---------|-----|-----|
| `ops::maximum(a, b)` | Element-wise maximum | `ops::maximum(a, b)` | ✅ | ✅ |
| `ops::minimum(a, b)` | Element-wise minimum | `ops::minimum(a, b)` | ✅ | ✅ |
| `ops::atan2(y, x)` | Element-wise atan2 | `ops::atan2(y, x)` | ✅ | ✅ |
| `ops::power(a, b)` | Element-wise power | `ops::power(a, b)` | ✅ | ✅ |
| `ops::modulo(a, b)` | Element-wise modulo | `ops::modulo(a, b)` | ✅ | ✅ |
| `ops::hypot(a, b)` | Element-wise hypotenuse | `ops::hypot(a, b)` | ✅ | ❌ |

**Note:** `hypot` not yet implemented on GPU (can be computed as `sqrt(a*a + b*b)`).

---

## Unary Operations

**Status: ✅ Fully tested - Full GPU support via MPSGraph!**

| Function | Operator | Description | CPU | GPU | Status |
|----------|----------|-------------|-----|-----|--------|
| `ops::negate(a)` | `-a` | Negation | ✅ | ✅ | ✅ |
| `ops::abs(a)` | - | Absolute value | ✅ | ✅ | ✅ |
| `ops::sqrt(a)` | - | Square root | ✅ | ✅ | ✅ |
| `ops::exp(a)` | - | Exponential | ✅ | ✅ | ✅ |
| `ops::log(a)` | - | Natural logarithm | ✅ | ✅ | ✅ |
| `ops::sin(a)` | - | Sine | ✅ | ✅ | ✅ |
| `ops::cos(a)` | - | Cosine | ✅ | ✅ | ✅ |
| `ops::tan(a)` | - | Tangent | ✅ | ✅ | ✅ |

**Migration Complete:** All unary operations use MPSGraph for automatic kernel fusion and Apple Silicon optimization.

---

## Reduction Operations

**Status: ✅ Fully tested - Full GPU support via MPSGraph!**

### Free Functions

| Function | Description | Example | CPU | GPU |
|----------|-------------|---------|-----|-----|
| `ops::sum(a, axis, keep_dims)` | Sum over axis | `ops::sum(a, {0}, true)` | ✅ | ✅ |
| `ops::mean(a, axis, keep_dims)` | Mean over axis | `ops::mean(a, {0})` | ✅ | ✅ |
| `ops::max(a, axis, keep_dims)` | Maximum over axis | `ops::max(a, {-1})` | ✅ | ✅ |
| `ops::min(a, axis, keep_dims)` | Minimum over axis | `ops::min(a)` | ✅ | ✅ |
| `ops::argmax(a, axis, keep_dims)` | Index of maximum | `ops::argmax(a, 0)` | ✅ | ✅ |
| `ops::argmin(a, axis, keep_dims)` | Index of minimum | `ops::argmin(a, -1)` | ✅ | ✅ |

### Member Functions

| Function | Description | Example | CPU | GPU | Status |
|----------|-------------|---------|-----|-----|--------|
| `tensor.sum(axis, keep_dims)` | Sum over axis | `t.sum(0)` | ✅ | ✅ | ✅ |
| `tensor.sum(axes, keep_dims)` | Sum over multiple axes | `t.sum({0, 1})` | ✅ | ✅ | ✅ |
| `tensor.mean(axis, keep_dims)` | Mean over axis | `t.mean(-1)` | ✅ | ✅ | ✅ |
| `tensor.max(axis, keep_dims)` | Maximum over axis | `t.max(0)` | ✅ | ✅ | ✅ |
| `tensor.min(axis, keep_dims)` | Minimum over axis | `t.min()` | ✅ | ✅ | ✅ |
| `tensor.argmax(axis, keep_dims)` | Index of maximum (returns Int64) | `t.argmax(1)` | ✅ | ✅ | ✅ |
| `tensor.argmin(axis, keep_dims)` | Index of minimum (returns Int64) | `t.argmin(-1)` | ✅ | ✅ | ✅ |

**Parameters:**
- `axis`: Dimension to reduce. `-1` reduces all dimensions (default).
- `axes`: Vector of dimensions to reduce.
- `keep_dims`: If true, reduced dimensions become size 1 instead of being removed.

**Migration Complete:** All reduction operations now use MPSGraph instead of custom Metal kernels!

---

## Matrix Multiplication

**Status: ✅ Fully tested (14/14 tests passing) - Full GPU support via MPSGraph!**

| Function | Description | Example | CPU | GPU | Status |
|----------|-------------|---------|-----|-----|--------|
| `a.matmul(b)` | Matrix multiplication (member) | `a.matmul(b)` | ✅ | ✅ | ✅ |
| `a.matmul(b, transpose_a, transpose_b)` | MatMul with optional transpose | `a.matmul(b, false, true)` | ✅ | ✅ | ✅ |
| `a.mm(b)` | Alias for matmul | `a.mm(b)` | ✅ | ✅ | ✅ |
| `a.dot(b)` | Alias for matmul (vectors) | `a.dot(b)` | ✅ | ✅ | ✅ |
| `ops::matmul(a, b)` | Matrix multiplication (free function) | `ops::matmul(a, b)` | ✅ | ✅ | ✅ |
| `ops::matmul(a, b, transpose_a, transpose_b)` | MatMul with optional transpose | `ops::matmul(a, b, false, true)` | ✅ | ✅ | ✅ |

**Migration Complete:** MatMul now uses MPSGraph's `matrixMultiplicationWithPrimaryTensor` for Apple-optimized performance!

### Supported Shapes

| A Shape | B Shape | Result Shape | Description | Status |
|---------|---------|--------------|-------------|--------|
| `(M, K)` | `(K, N)` | `(M, N)` | Standard 2D matmul | ✅ |
| `(K,)` | `(K, N)` | `(N,)` | Vector-matrix multiply | ✅ |
| `(M, K)` | `(K,)` | `(M,)` | Matrix-vector multiply | ✅ |
| `(K,)` | `(K,)` | `()` | Dot product (scalar) | ✅ |
| `(..., M, K)` | `(..., K, N)` | `(..., M, N)` | Batched matmul | ✅ |

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

| Operation Category | CPU | Metal GPU | Implementation |
|-------------------|-----|-----------|----------------|
| Binary Arithmetic | ✅ | ✅ | MPSGraph |
| Comparison | ✅ | ✅ | MPSGraph |
| Logical | ✅ | ✅ | MPSGraph |
| Bitwise | ✅ | ✅ | MPSGraph (integer types) |
| Math Binary | ✅ | ✅ | MPSGraph |
| Unary | ✅ | ✅ | MPSGraph |
| Reduction (sum, mean, max, min) | ✅ | ✅ | MPSGraph |
| ArgMax/ArgMin | ✅ | ✅ | MPSGraph |
| Matrix Multiplication | ✅ | ✅ | MPSGraph |
| Conditional (where) | ✅ | ✅ | MPSGraph |

### Complete MPSGraph Migration (v0.1.0)

All GPU operations now use Apple's MPSGraph framework:

- ✅ **Binary arithmetic** (add, subtract, multiply, divide, power, modulo)
- ✅ **Unary operations** (negate, abs, sqrt, exp, log, sin, cos, tan)
- ✅ **Comparison operations** (equal, not_equal, less, less_equal, greater, greater_equal)
- ✅ **Logical operations** (and, or, xor, not)
- ✅ **Bitwise operations** (and, or, xor, shifts)
- ✅ **Reduction operations** (sum, mean, max, min)
- ✅ **ArgMax/ArgMin** - Returns Int64 indices
- ✅ **Matrix multiplication** - Batched matmul with broadcasting
- ✅ **Conditional selection** (where)
- ✅ **Math operations** (maximum, minimum, atan2, power, modulo)

### Benefits of MPSGraph Architecture

1. **Automatic kernel fusion** - Sequential operations are fused into single GPU kernels
2. **Apple Silicon optimization** - Uses Apple's hand-tuned kernels for M1/M2/M3 chips
3. **Simpler codebase** - ~500 lines of custom Metal kernel code removed
4. **Better maintainability** - New operations are 5-10 lines instead of 50+
5. **Non-contiguous tensor support** - GPU gather kernel handles strided memory layouts

### GPU Gather Kernel

Non-contiguous GPU tensors (from slicing, transposing, etc.) are automatically made contiguous using a GPU gather kernel before MPSGraph operations. This happens entirely on the GPU with no CPU fallback:

```cpp
// This works efficiently on GPU:
auto t = Tensor::randn({100, 100}, Device::GPU);
auto view = t.slice({{0, 50}, {25, 75}});  // Non-contiguous view
auto result = view.sum();  // Gather kernel + MPSGraph reduction
```

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

---

## View & Memory Introspection

Understand tensor memory layout and predict operation costs.

| Function | Description | Example |
|----------|-------------|---------|
| `tensor.is_view()` | True if tensor shares data with another | `t.is_view()` |
| `tensor.owns_data()` | True if tensor owns its storage | `t.owns_data()` |
| `tensor.has_zero_stride()` | True if any stride is 0 (broadcast view) | `t.has_zero_stride()` |
| `tensor.shares_storage(other)` | True if tensors share memory | `a.shares_storage(b)` |
| `tensor.would_materialize_on_reshape(shape)` | Predict if reshape copies | `t.would_materialize_on_reshape({2, 6})` |
| `tensor.would_materialize_on_transpose()` | Predict if transpose copies | `t.would_materialize_on_transpose()` |

```cpp
auto a = Tensor::ones({3, 4});
auto b = a.reshape({12});           // b.is_view() -> depends on contiguity
auto c = a.slice({{0, 2}});         // c.shares_storage(a) -> true
auto d = a.expand({2, 3, 4});       // d.has_zero_stride() -> true
```

---

## Explicit Broadcasting

Clearer intent for broadcasting operations.

| Function | Description | Example |
|----------|-------------|---------|
| `tensor.expand_as(other)` | Expand to match another tensor's shape | `x.expand_as(y)` |
| `tensor.broadcast_to(shape)` | Expand to target shape | `x.broadcast_to({64, 128})` |

```cpp
auto x = Tensor::ones({1, 64});
auto y = Tensor::ones({32, 64});
auto z = x.expand_as(y);           // Shape: (32, 64), zero-copy
auto w = x.broadcast_to({16, 64}); // Shape: (16, 64), zero-copy
```

---

## Safety Rails & Debugging

Catch numerical issues early.

### NaN/Inf Detection

| Function | Description | Returns |
|----------|-------------|---------|
| `tensor.has_nan()` | Check for NaN values | `bool` |
| `tensor.has_inf()` | Check for Inf values | `bool` |
| `tensor.is_finite()` | True if no NaN or Inf | `bool` |
| `tensor.nan_guard()` | Throw if NaN detected | `Tensor&` |
| `tensor.assert_finite()` | Throw if NaN or Inf detected | `Tensor&` |

```cpp
auto result = model.forward(input)
    .nan_guard()           // Throws if NaN
    .assert_finite();      // Throws if NaN or Inf

if (loss.has_nan()) {
    std::cerr << "Training diverged!" << std::endl;
}
```

### Shape Assertions

| Function | Description | Example |
|----------|-------------|---------|
| `tensor.assert_shape(shape)` | Assert exact shape | `t.assert_shape({32, 3, 224, 224})` |
| `tensor.assert_shape(pattern)` | Assert shape pattern | `t.assert_shape("batch 3 height width")` |

```cpp
// Named dimensions (any size ok, just checks ndim)
input.assert_shape("batch channels height width");

// Mixed: named + exact values
input.assert_shape("batch 3 224 224");  // Throws if channels != 3 or h/w != 224
```

### Debug Info

| Function | Description |
|----------|-------------|
| `tensor.debug_info()` | Full diagnostic string |

```cpp
std::cout << tensor.debug_info();
// Output:
// Tensor Debug Info:
//   Shape: [2, 3]
//   Strides: [12, 4]
//   DType: float32
//   Device: CPU
//   Size: 6 elements, 24 bytes
//   Memory order: RowMajor
//   Contiguous: yes
//   Is view: no
//   Owns data: yes
//   Has zero stride: no
//   Has NaN: no
//   Has Inf: no
```

---

## Profiling & Tracing

Track operations and performance.

### Tracing

```cpp
axiom::trace::enable();
axiom::trace::clear();

// ... tensor operations ...

std::cout << axiom::trace::dump();
axiom::trace::disable();
```

### Profiling

```cpp
axiom::profile::enable();

auto c = a.matmul(b);

auto& last = axiom::profile::last_op();
std::cout << "Op: " << last.name << ", Duration: " << last.duration.count() << "ns\n";
```

---

## Error Handling

Axiom provides a rich hierarchy of error types for precise exception handling.

### Error Hierarchy

```
AxiomError (base)
├── ShapeError      - Shape mismatches, invalid axes, broadcast failures
├── TypeError       - Unsupported dtypes, type mismatches
├── DeviceError     - Device unavailable, CPU/GPU mismatches
├── ValueError      - NaN/Inf detection, out-of-range values
├── IndexError      - Out-of-bounds indexing, invalid slices
├── MemoryError     - Allocation failures, non-contiguous requirements
└── RuntimeError    - Internal errors, unimplemented features
```

### Usage

```cpp
#include "axiom/error.hpp"

try {
    tensor.assert_shape({32, 3, 224, 224});
} catch (const ShapeError& e) {
    // Handle shape-specific error
} catch (const AxiomError& e) {
    // Catch any Axiom error
}
```

### Factory Methods

```cpp
// ShapeError
throw ShapeError::mismatch(expected_shape, actual_shape);
throw ShapeError::invalid_axis(axis, ndim);
throw ShapeError::broadcast_incompatible("(3,4) and (5,)");

// TypeError
throw TypeError::unsupported_dtype("complex128", "matmul");
throw TypeError::dtype_mismatch("float32", "int64");

// ValueError
throw ValueError::nan_detected("layer_output");
throw ValueError::inf_detected("gradients");
throw ValueError::out_of_range("learning_rate", 0.0, 1.0, 2.5);

// DeviceError
throw DeviceError::not_available("Metal GPU");
throw DeviceError::cpu_only("direct data access");

// IndexError
throw IndexError::out_of_bounds(10, 5, /*dim=*/0);
throw IndexError::invalid_slice("step cannot be zero");
```

---

## Numeric Utilities

Axiom provides utilities for handling special floating-point values.

### Constants

```cpp
#include "axiom/numeric.hpp"

// NaN constants
float nan_f = axiom::numeric::nan<float>();
double nan_d = axiom::numeric::nan_d;

// Infinity constants
float inf_f = axiom::numeric::inf<float>();
float neg_inf_f = axiom::numeric::neg_inf<float>();

// Machine epsilon
float eps = axiom::numeric::epsilon<float>();
```

### Value Classification

```cpp
axiom::is_nan(value);      // Check for NaN
axiom::is_inf(value);      // Check for +/- infinity
axiom::is_finite(value);   // Check for normal finite value

axiom::numeric::is_pos_inf(value);  // Positive infinity only
axiom::numeric::is_neg_inf(value);  // Negative infinity only
axiom::numeric::is_normal(value);   // Not zero, subnormal, inf, or nan
```

### String Formatting

```cpp
// Automatic NaN/Inf handling
axiom::numeric::to_string(NAN);        // "nan"
axiom::numeric::to_string(INFINITY);   // "inf"
axiom::numeric::to_string(-INFINITY);  // "-inf"
axiom::numeric::to_string(3.14159f);   // "3.1416"

// Tensor display automatically uses this formatting
std::cout << tensor;  // NaN shows as "nan", Inf as "inf"
```

### Safe Operations

Operations that return NaN/Inf instead of throwing:

```cpp
axiom::numeric::safe_div(1.0, 0.0);   // Returns inf
axiom::numeric::safe_div(0.0, 0.0);   // Returns nan
axiom::numeric::safe_log(-1.0);       // Returns nan
axiom::numeric::safe_sqrt(-1.0);      // Returns nan
```

### Approximate Equality

```cpp
// For floating-point comparison with tolerance
axiom::numeric::approx_equal(a, b);  // Uses default tolerances
axiom::numeric::approx_equal(a, b, rel_tol, abs_tol);

// NaN != NaN (IEEE semantics preserved)
axiom::numeric::approx_equal(NAN, NAN);  // false
```
