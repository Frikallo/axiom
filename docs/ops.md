# Axiom Operations Reference

Quick reference for all operations in Axiom. All operations support both CPU and Metal GPU unless noted.

## Tensor Creation

| Function | Description |
|----------|-------------|
| `Tensor::zeros(shape, dtype, device)` | Tensor filled with zeros |
| `Tensor::ones(shape, dtype, device)` | Tensor filled with ones |
| `Tensor::empty(shape, dtype, device)` | Uninitialized tensor |
| `Tensor::full(shape, value, dtype, device)` | Tensor filled with value |
| `Tensor::eye(n, dtype, device)` | Identity matrix |
| `Tensor::randn(shape, dtype, device)` | Random normal distribution |
| `Tensor::arange(start, end, step, dtype, device)` | Range tensor |
| `Tensor::from_data(ptr, shape, dtype, device)` | From raw data pointer |

## Shape Manipulation

| Function | Description |
|----------|-------------|
| `tensor.reshape(shape)` | Reshape (view if possible) |
| `tensor.view(shape)` | View with new shape (contiguous only) |
| `tensor.flatten(start, end)` | Flatten dimensions |
| `tensor.transpose()` | Swap last two dimensions |
| `tensor.transpose(axes)` | Permute dimensions |
| `tensor.squeeze(axis)` | Remove size-1 dimensions |
| `tensor.unsqueeze(axis)` | Add size-1 dimension |
| `tensor.expand(shape)` | Broadcast expand (zero-copy) |
| `tensor.repeat(repeats)` | Repeat with data copy |
| `tensor.rearrange(pattern)` | Einops-style reshape |

## Memory Operations

| Function | Description |
|----------|-------------|
| `tensor.copy()` / `tensor.clone()` | Deep copy |
| `tensor.to(device)` | Move to device |
| `tensor.cpu()` / `tensor.gpu()` | Device shortcuts |
| `tensor.ascontiguousarray()` | Make C-contiguous |
| `tensor.astype(dtype)` | Type conversion |

## Binary Arithmetic

| Function | Operator | CPU | GPU |
|----------|----------|-----|-----|
| `ops::add(a, b)` | `+` | ✓ | ✓ |
| `ops::subtract(a, b)` | `-` | ✓ | ✓ |
| `ops::multiply(a, b)` | `*` | ✓ | ✓ |
| `ops::divide(a, b)` | `/` | ✓ | ✓ |
| `ops::power(a, b)` | - | ✓ | ✓ |
| `ops::modulo(a, b)` | `%` | ✓ | ✓ |

In-place: `+=`, `-=`, `*=`, `/=`

## Comparison Operations

Returns boolean tensors.

| Function | Operator | CPU | GPU |
|----------|----------|-----|-----|
| `ops::equal(a, b)` | `==` | ✓ | ✓ |
| `ops::not_equal(a, b)` | `!=` | ✓ | ✓ |
| `ops::less(a, b)` | `<` | ✓ | ✓ |
| `ops::less_equal(a, b)` | `<=` | ✓ | ✓ |
| `ops::greater(a, b)` | `>` | ✓ | ✓ |
| `ops::greater_equal(a, b)` | `>=` | ✓ | ✓ |

## Logical Operations

| Function | CPU | GPU |
|----------|-----|-----|
| `ops::logical_and(a, b)` | ✓ | ✓ |
| `ops::logical_or(a, b)` | ✓ | ✓ |
| `ops::logical_xor(a, b)` | ✓ | ✓ |
| `ops::logical_not(a)` | ✓ | ✓ |

## Bitwise Operations

Integer types only.

| Function | Operator | CPU | GPU |
|----------|----------|-----|-----|
| `ops::bitwise_and(a, b)` | `&` | ✓ | ✓ |
| `ops::bitwise_or(a, b)` | `\|` | ✓ | ✓ |
| `ops::bitwise_xor(a, b)` | `^` | ✓ | ✓ |
| `ops::left_shift(a, b)` | `<<` | ✓ | ✓ |
| `ops::right_shift(a, b)` | `>>` | ✓ | ✓ |

## Math Operations

| Function | CPU | GPU |
|----------|-----|-----|
| `ops::maximum(a, b)` | ✓ | ✓ |
| `ops::minimum(a, b)` | ✓ | ✓ |
| `ops::atan2(y, x)` | ✓ | ✓ |
| `ops::hypot(a, b)` | ✓ | ✗ |

## Unary Operations

| Function | Operator | CPU | GPU |
|----------|----------|-----|-----|
| `ops::negate(a)` | `-a` | ✓ | ✓ |
| `ops::abs(a)` | - | ✓ | ✓ |
| `ops::sqrt(a)` | - | ✓ | ✓ |
| `ops::exp(a)` | - | ✓ | ✓ |
| `ops::log(a)` | - | ✓ | ✓ |
| `ops::sin(a)` | - | ✓ | ✓ |
| `ops::cos(a)` | - | ✓ | ✓ |
| `ops::tan(a)` | - | ✓ | ✓ |
| `ops::erf(a)` | - | ✓ | ✓ |

## Complex Operations

| Function | Description | CPU | GPU |
|----------|-------------|-----|-----|
| `ops::conj(a)` | Complex conjugate | ✓ | ✓ |
| `ops::real(a)` | Real part | ✓ | ✓ |
| `ops::imag(a)` | Imaginary part | ✓ | ✓ |

## Activation Functions

| Function | Description | CPU | GPU |
|----------|-------------|-----|-----|
| `ops::gelu(a)` | Gaussian Error Linear Unit | ✓ | ✓ |
| `ops::softmax(a, axis)` | Softmax (default axis=-1) | ✓ | ✓ |
| `ops::log_softmax(a, axis)` | Log-softmax (default axis=-1) | ✓ | ✓ |

## Reduction Operations

| Function | CPU | GPU |
|----------|-----|-----|
| `ops::sum(a, axis, keep_dims)` | ✓ | ✓ |
| `ops::mean(a, axis, keep_dims)` | ✓ | ✓ |
| `ops::max(a, axis, keep_dims)` | ✓ | ✓ |
| `ops::min(a, axis, keep_dims)` | ✓ | ✓ |
| `ops::argmax(a, axis, keep_dims)` | ✓ | ✓ |
| `ops::argmin(a, axis, keep_dims)` | ✓ | ✓ |
| `ops::any(a, axis, keep_dims)` | ✓ | ✓ |
| `ops::all(a, axis, keep_dims)` | ✓ | ✓ |

Member functions: `tensor.sum()`, `tensor.mean()`, `tensor.max()`, `tensor.min()`, `tensor.argmax()`, `tensor.argmin()`

**Boolean reductions:** `any` returns true if any element is non-zero; `all` returns true if all elements are non-zero.

## Matrix Multiplication

| Function | CPU | GPU |
|----------|-----|-----|
| `ops::matmul(a, b, transpose_a, transpose_b)` | ✓ | ✓ |
| `tensor.matmul(b)` / `tensor.mm(b)` / `tensor.dot(b)` | ✓ | ✓ |

Supported shapes:
- `(M,K) @ (K,N) → (M,N)` - Standard matmul
- `(K,) @ (K,N) → (N,)` - Vector-matrix
- `(M,K) @ (K,) → (M,)` - Matrix-vector
- `(...,M,K) @ (...,K,N) → (...,M,N)` - Batched with broadcasting

## Conditional Selection

| Function | CPU | GPU |
|----------|-----|-----|
| `ops::where(condition, a, b)` | ✓ | ✓ |

Returns elements from `a` where condition is true, `b` otherwise. All inputs broadcast together.

## Normalization Operations

| Function | Description | CPU | GPU |
|----------|-------------|-----|-----|
| `ops::layer_norm(x, weight, bias, axis, eps)` | Layer normalization | ✓ | ✓ |
| `ops::rms_norm(x, weight, axis, eps)` | RMS normalization | ✓ | ✓ |

- `layer_norm`: Computes `(x - mean) / sqrt(var + eps) * weight + bias`
- `rms_norm`: Computes `x / sqrt(mean(x²) + eps) * weight`
- Default `axis=-1`, `eps=1e-5`

## Dropout

| Function | Description | CPU | GPU |
|----------|-------------|-----|-----|
| `ops::dropout(x, p, training)` | Dropout regularization | ✓ | ✓ |

Returns `std::pair<Tensor, Tensor>` containing (output, mask). Scale factor `1/(1-p)` is applied when `training=true`. Default `p=0.5`.

## Data Types

| DType | C++ Type | Size |
|-------|----------|------|
| `Bool` | `bool` | 1 |
| `Int8` / `UInt8` | `int8_t` / `uint8_t` | 1 |
| `Int16` / `UInt16` | `int16_t` / `uint16_t` | 2 |
| `Int32` / `UInt32` | `int32_t` / `uint32_t` | 4 |
| `Int64` / `UInt64` | `int64_t` / `uint64_t` | 8 |
| `Float16` | `float16_t` | 2 |
| `Float32` | `float` | 4 |
| `Float64` | `double` | 8 |

## File I/O

| Function | Description |
|----------|-------------|
| `tensor.save(filename)` | Save single tensor |
| `Tensor::load(filename)` | Load single tensor |
| `Tensor::save_tensors(map, filename)` | Save multiple tensors |
| `Tensor::load_tensors(filename)` | Load multiple tensors |

## Introspection

| Function | Description |
|----------|-------------|
| `tensor.is_view()` | Shares data with another tensor |
| `tensor.is_contiguous()` | Memory is contiguous |
| `tensor.has_zero_stride()` | Has broadcast dimensions |
| `tensor.shares_storage(other)` | Shares memory with other |

## Debugging

| Function | Description |
|----------|-------------|
| `tensor.has_nan()` / `tensor.has_inf()` | Check for special values |
| `tensor.nan_guard()` / `tensor.assert_finite()` | Throw on special values |
| `tensor.assert_shape(shape)` | Assert exact shape |
| `tensor.debug_info()` | Full diagnostic string |

## Backend Summary

All operations use MPSGraph on Metal GPU for automatic kernel fusion and Apple Silicon optimization.

| Category | CPU | GPU |
|----------|-----|-----|
| Binary Arithmetic | ✓ | ✓ |
| Comparison | ✓ | ✓ |
| Logical | ✓ | ✓ |
| Bitwise | ✓ | ✓ |
| Math | ✓ | ✓ (except hypot) |
| Unary | ✓ | ✓ |
| Complex | ✓ | ✓ |
| Activations | ✓ | ✓ |
| Reductions | ✓ | ✓ |
| MatMul | ✓ | ✓ |
| Where | ✓ | ✓ |
| Normalization | ✓ | ✓ |
| Dropout | ✓ | ✓ |
