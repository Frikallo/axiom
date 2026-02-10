# Tensor Manipulation

Shape manipulation, transposition, flipping, rolling, and einops-style rearrangement operations on `axiom::Tensor`.

All operations return a new `Tensor`. Methods that can operate as zero-copy views (shared storage, no data movement) are noted. Negative axis indices are supported everywhere and resolve relative to `ndim()`.

*For a tutorial introduction, see the [User Guide: Shape Manipulation](../user-guide/shape-manipulation).*

---

## Reshape / View

### reshape

Returns a tensor with the given shape. Returns a **view** when the tensor is contiguous and the memory order matches; copies data otherwise.

```cpp
Tensor reshape(const Shape& new_shape,
               MemoryOrder order = MemoryOrder::RowMajor) const;
Tensor reshape(std::initializer_list<size_t> new_shape,
               MemoryOrder order = MemoryOrder::RowMajor) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `new_shape` | `Shape` / `initializer_list<size_t>` | Target shape. Total element count must match `size()`. |
| `order` | `MemoryOrder` | Memory layout for the result. Default `RowMajor`. |

**Returns** -- `Tensor` with the requested shape (view or copy).

**Throws** -- `ShapeError` if the total number of elements differs.

```cpp
auto a = Tensor::arange(12);              // shape (12,)
auto b = a.reshape({3, 4});               // shape (3, 4) -- view
auto c = a.reshape({2, 2, 3});            // shape (2, 2, 3) -- view
```

**Tip:** Use `would_materialize_on_reshape(new_shape)` to check whether the reshape will copy.

---

### view

Strict view-only reshape. Requires the tensor to be contiguous; throws if a copy would be needed.

```cpp
Tensor view(const Shape& new_shape) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `new_shape` | `Shape` | Target shape. Total element count must match `size()`. |

**Returns** -- `Tensor` view with shared storage.

**Throws** -- `ShapeError` if the element count differs. `MemoryError` if the tensor is not contiguous.

```cpp
auto a = Tensor::ones({4, 4});
auto b = a.view({16});                    // OK -- contiguous
auto c = a.transpose().view({16});        // throws MemoryError
```

---

### flatten

Flattens a contiguous range of dimensions into a single dimension. Returns a view when the flattened region is contiguous; copies otherwise.

```cpp
Tensor flatten(int start_dim = 0, int end_dim = -1) const;
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `start_dim` | `int` | `0` | First dimension to flatten (inclusive). |
| `end_dim` | `int` | `-1` | Last dimension to flatten (inclusive). |

**Returns** -- `Tensor` with the specified dimensions collapsed.

**Throws** -- `ShapeError` if dimension indices are out of range or `start_dim > end_dim`.

```cpp
auto a = Tensor::randn({2, 3, 4});
auto b = a.flatten();                     // shape (24,)
auto c = a.flatten(1, 2);                 // shape (2, 12)
```

---

### unflatten

Expands a single dimension into multiple dimensions. The product of `sizes` must equal the size of dimension `dim`.

```cpp
Tensor unflatten(int dim, const Shape& sizes) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `dim` | `int` | Dimension to unflatten. |
| `sizes` | `Shape` | Sizes of the replacement dimensions. Product must equal `shape()[dim]`. |

**Returns** -- `Tensor` with the target dimension split.

**Throws** -- `ShapeError` if `dim` is out of range or the product of `sizes` does not match.

```cpp
auto a = Tensor::randn({2, 12});
auto b = a.unflatten(1, {3, 4});          // shape (2, 3, 4)
auto c = a.unflatten(1, {2, 2, 3});       // shape (2, 2, 2, 3)
```

---

### ravel

Alias for `flatten()`. Returns a 1-D tensor.

```cpp
Tensor ravel() const;
```

**Returns** -- `Tensor` of shape `(size(),)`.

```cpp
auto a = Tensor::ones({3, 4});
auto flat = a.ravel();                    // shape (12,)
```

---

## Transpose / Permute

### transpose

Swap the last two dimensions (no-argument form) or permute all dimensions. Always returns a zero-copy **view**.

```cpp
Tensor transpose() const;                            // swap last 2 dims
Tensor transpose(const std::vector<int>& axes) const; // full permutation
```

**Parameters (permutation form)**

| Name | Type | Description |
|------|------|-------------|
| `axes` | `std::vector<int>` | Permutation of `[0, ndim())`. Length must equal `ndim()`. |

**Returns** -- `Tensor` view with permuted shape and strides.

**Throws** -- `ShapeError` if `axes.size() != ndim()` or any axis is out of range.

```cpp
auto a = Tensor::randn({2, 3, 4});
auto b = a.transpose();                  // shape (2, 4, 3)
auto c = a.transpose({2, 0, 1});         // shape (4, 2, 3)
```

For tensors with fewer than 2 dimensions, the no-argument form returns `*this` unchanged.

---

### T

Alias for `transpose()`. Swaps the last two dimensions.

```cpp
Tensor T() const;
```

```cpp
auto a = Tensor::randn({3, 4});
auto b = a.T();                           // shape (4, 3) -- view
```

---

### swapaxes

Swap two axes. Returns a zero-copy **view** (implemented via `transpose`).

```cpp
Tensor swapaxes(int axis1, int axis2) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `axis1` | `int` | First axis. |
| `axis2` | `int` | Second axis. |

**Returns** -- `Tensor` view with the two axes exchanged.

**Throws** -- `ShapeError` if either axis is out of range.

```cpp
auto a = Tensor::randn({2, 3, 4});
auto b = a.swapaxes(0, 2);               // shape (4, 3, 2)
```

---

### moveaxis

Move an axis from one position to another. Returns a zero-copy **view** (implemented via `transpose`).

```cpp
Tensor moveaxis(int source, int destination) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `source` | `int` | Original position of the axis. |
| `destination` | `int` | Target position. |

**Returns** -- `Tensor` view with the axis relocated.

**Throws** -- `ShapeError` if either position is out of range.

```cpp
auto a = Tensor::randn({2, 3, 4});
auto b = a.moveaxis(0, 2);               // shape (3, 4, 2)
auto c = a.moveaxis(-1, 0);              // shape (4, 2, 3)
```

---

## Dimension Manipulation

### squeeze

Remove dimensions of size 1. When called with the default `axis = -1`, removes **all** size-1 dimensions. When a specific axis is given, removes only that dimension (no-op if that dimension is not size 1). Always returns a zero-copy **view**.

```cpp
Tensor squeeze(int axis = -1) const;
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `axis` | `int` | `-1` | Specific axis to squeeze, or `-1` to squeeze all size-1 dims. |

**Returns** -- `Tensor` view with size-1 dimensions removed.

**Throws** -- `ShapeError` if a specific axis (not `-1`) is out of range.

```cpp
auto a = Tensor::randn({1, 3, 1, 4});
auto b = a.squeeze();                    // shape (3, 4)
auto c = a.squeeze(0);                   // shape (3, 1, 4)
auto d = a.squeeze(1);                   // shape (1, 3, 1, 4) -- no-op
```

---

### unsqueeze

Insert a dimension of size 1 at the given position. Always returns a zero-copy **view**.

```cpp
Tensor unsqueeze(int axis) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `axis` | `int` | Position for the new dimension. Valid range `[-ndim()-1, ndim()]`. |

**Returns** -- `Tensor` view with one additional dimension.

```cpp
auto a = Tensor::randn({3, 4});
auto b = a.unsqueeze(0);                 // shape (1, 3, 4)
auto c = a.unsqueeze(-1);                // shape (3, 4, 1)
```

---

### expand

Broadcast-expand dimensions of size 1 to a larger size. Always returns a zero-copy **view** that uses stride 0 for expanded dimensions. The result is read-only in practice (writes would alias).

```cpp
Tensor expand(const Shape& new_shape) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `new_shape` | `Shape` | Target shape. Must have `>= ndim()` dimensions. Only size-1 dims may be expanded. |

**Returns** -- `Tensor` view with broadcasted strides.

**Throws** -- `ShapeError` if the new shape has fewer dimensions, if a non-size-1 dimension would need to change, or if expanding to size 0.

```cpp
auto a = Tensor::ones({1, 4});
auto b = a.expand({3, 4});               // shape (3, 4) -- zero-copy
auto c = a.expand({2, 3, 4});            // shape (2, 3, 4) -- adds leading dim
```

---

### expand_as

Convenience wrapper that calls `expand(other.shape())`.

```cpp
Tensor expand_as(const Tensor& other) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `other` | `Tensor` | Tensor whose shape is used as the target. |

**Returns** -- `Tensor` view expanded to match `other.shape()`.

```cpp
auto weights = Tensor::ones({1, 4});
auto data = Tensor::randn({8, 4});
auto expanded = weights.expand_as(data);  // shape (8, 4)
```

---

### broadcast_to

Alias for `expand(shape)`. Matches the NumPy name.

```cpp
Tensor broadcast_to(const Shape& shape) const;
```

```cpp
auto a = Tensor::ones({1, 3});
auto b = a.broadcast_to({4, 3});          // shape (4, 3) -- zero-copy
```

---

### repeat

Repeat (tile) the tensor along each dimension by copying data. Unlike `expand`, this allocates new storage.

```cpp
Tensor repeat(const std::vector<size_t>& repeats) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `repeats` | `std::vector<size_t>` | Number of repetitions per dimension. Length must equal `ndim()`. |

**Returns** -- `Tensor` with each dimension `i` having size `shape()[i] * repeats[i]`.

**Throws** -- `ShapeError` if `repeats.size() != ndim()`.

```cpp
auto a = Tensor::arange(3);              // [0, 1, 2]
auto b = a.repeat({3});                  // [0, 1, 2, 0, 1, 2, 0, 1, 2]

auto m = Tensor::ones({2, 3});
auto n = m.repeat({2, 3});               // shape (4, 9)
```

---

### tile

Alias for `repeat`. Matches the NumPy name.

```cpp
Tensor tile(const std::vector<size_t>& reps) const;
```

```cpp
auto a = Tensor::arange(4).reshape({2, 2});
auto b = a.tile({1, 2});                 // shape (2, 4)
```

---

## Flipping / Rolling

### flip

Reverse elements along one or more axes. Always returns a zero-copy **view** (uses negative strides).

```cpp
Tensor flip(int axis) const;
Tensor flip(const std::vector<int>& axes) const;
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `axis` / `axes` | `int` / `std::vector<int>` | Axis or axes to flip. |

**Returns** -- `Tensor` view with negated strides along the specified axes.

**Throws** -- `ShapeError` if any axis is out of range.

```cpp
auto a = Tensor::arange(6).reshape({2, 3});
auto b = a.flip(1);                      // reverse columns
auto c = a.flip({0, 1});                 // reverse both axes
```

---

### flipud

Flip along axis 0 (rows). Alias for `flip(0)`.

```cpp
Tensor flipud() const;
```

**Returns** -- `Tensor` view with the first axis reversed.

```cpp
auto a = Tensor::arange(6).reshape({2, 3});
auto b = a.flipud();                     // equivalent to a.flip(0)
```

---

### fliplr

Flip along axis 1 (columns). Alias for `flip(1)`.

```cpp
Tensor fliplr() const;
```

**Returns** -- `Tensor` view with the second axis reversed.

```cpp
auto a = Tensor::arange(6).reshape({2, 3});
auto b = a.fliplr();                     // equivalent to a.flip(1)
```

---

### rot90

Rotate the tensor 90 degrees in the plane defined by two axes. Composed from `flip` and `swapaxes`, so intermediate results are views.

```cpp
Tensor rot90(int k = 1, const std::vector<int>& axes = {0, 1}) const;
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `k` | `int` | `1` | Number of 90-degree rotations. Negative values rotate clockwise. |
| `axes` | `std::vector<int>` | `{0, 1}` | The two axes that define the rotation plane. Must be distinct. |

**Returns** -- `Tensor` rotated `k` times.

**Throws** -- `ShapeError` if `axes` does not contain exactly 2 distinct valid axes.

```cpp
auto a = Tensor::arange(4).reshape({2, 2});
auto b = a.rot90();                       // 90 degrees counter-clockwise
auto c = a.rot90(2);                      // 180 degrees
auto d = a.rot90(-1);                     // 90 degrees clockwise
```

---

### roll

Circular shift of elements along an axis. When `axis = -1` (default), rolls over the flattened tensor and reshapes back. Copies data.

```cpp
Tensor roll(int64_t shift, int axis = -1) const;
Tensor roll(const std::vector<int64_t>& shifts,
            const std::vector<int>& axes) const;
```

**Parameters (single-axis form)**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `shift` | `int64_t` | | Number of positions to shift. Positive shifts toward higher indices. |
| `axis` | `int` | `-1` | Axis to roll along. `-1` rolls the flattened tensor. |

**Parameters (multi-axis form)**

| Name | Type | Description |
|------|------|-------------|
| `shifts` | `std::vector<int64_t>` | Shift amount for each axis. |
| `axes` | `std::vector<int>` | Corresponding axes. Must have the same length as `shifts`. |

**Returns** -- `Tensor` with elements circularly shifted.

**Throws** -- `ShapeError` if an axis is out of range. `ValueError` if `shifts` and `axes` have different lengths.

```cpp
auto a = Tensor::arange(5);              // [0, 1, 2, 3, 4]
auto b = a.roll(2);                      // [3, 4, 0, 1, 2]
auto c = a.roll(-1);                     // [1, 2, 3, 4, 0]

auto m = Tensor::arange(6).reshape({2, 3});
auto n = m.roll({1, -1}, {0, 1});        // roll axis 0 by +1, axis 1 by -1
```

---

## Diagonal

### diagonal

Extract a diagonal from a 2-D (or higher) tensor. For N-D tensors, the diagonal is taken from the plane defined by `axis1` and `axis2`, and a new axis is appended to hold the diagonal values.

```cpp
Tensor diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) const;
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `offset` | `int` | `0` | Offset from the main diagonal. Positive for upper diagonals, negative for lower. |
| `axis1` | `int` | `0` | First axis of the 2-D sub-array from which the diagonal is taken. |
| `axis2` | `int` | `1` | Second axis. Must differ from `axis1`. |

**Returns** -- `Tensor` containing the diagonal elements. For a 2-D input of shape `(M, N)`, the result has shape `(min(M, N - offset),)` when `offset >= 0`.

**Throws** -- `ShapeError` if `ndim() < 2`, axes are out of range, or `axis1 == axis2`.

```cpp
auto a = Tensor::arange(9).reshape({3, 3});
auto d = a.diagonal();                   // main diagonal [0, 4, 8]
auto d1 = a.diagonal(1);                 // upper diagonal [1, 5]
auto d_1 = a.diagonal(-1);               // lower diagonal [3, 7]
```

---

### trace

Sum along a diagonal. Equivalent to `diagonal(offset, axis1, axis2).sum(-1)`.

```cpp
Tensor trace(int offset = 0, int axis1 = 0, int axis2 = 1) const;
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `offset` | `int` | `0` | Diagonal offset. |
| `axis1` | `int` | `0` | First axis. |
| `axis2` | `int` | `1` | Second axis. |

**Returns** -- `Tensor` containing the sum of diagonal elements.

**Throws** -- Same as `diagonal`.

```cpp
auto a = Tensor::arange(9).reshape({3, 3});
float t = a.trace().item<float>();        // 0 + 4 + 8 = 12
```

---

## Einops

Semantic tensor manipulation using [einops](https://einops.rocks)-style pattern strings. See also the [Einops API Reference](einops).

### rearrange

Reshape, transpose, merge, and split axes in a single operation driven by a pattern string.

```cpp
Tensor rearrange(const std::string& pattern,
                 const std::map<std::string, size_t>& axis_sizes = {}) const;
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `pattern` | `std::string` | | Einops pattern such as `"b c h w -> b (c h) w"`. |
| `axis_sizes` | `std::map<std::string, size_t>` | `{}` | Named axis sizes for decomposition, e.g., `{{"h", 2}}`. |

**Returns** -- `Tensor` with the shape described by the right-hand side of the pattern.

```cpp
auto a = Tensor::randn({2, 3, 4, 5});

// Transpose
auto b = a.rearrange("b c h w -> b h w c");

// Flatten spatial dims
auto c = a.rearrange("b c h w -> b (c h w)");

// Split a dimension
auto d = Tensor::randn({6, 4});
auto e = d.rearrange("(h w) c -> h w c", {{"h", 2}, {"w", 3}});
```

---

### reduce

Reduce (collapse) axes using a named reduction while rearranging the remaining axes.

```cpp
Tensor reduce(const std::string& pattern,
              const std::string& reduction,
              const std::map<std::string, size_t>& axis_sizes = {}) const;
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `pattern` | `std::string` | | Einops pattern. Axes present on the left but absent on the right are reduced. |
| `reduction` | `std::string` | | Reduction operation: `"sum"`, `"mean"`, `"max"`, or `"min"`. |
| `axis_sizes` | `std::map<std::string, size_t>` | `{}` | Named axis sizes for decomposition. |

**Returns** -- `Tensor` with the reduced shape.

```cpp
auto img = Tensor::randn({8, 3, 32, 32});

// Global average pooling
auto gap = img.reduce("b c h w -> b c", "mean");

// 2x2 average pooling
auto pooled = img.reduce("b c (h p1) (w p2) -> b c h w", "mean",
                         {{"p1", 2}, {"p2", 2}});

// Channel-wise max
auto chmax = img.reduce("b c h w -> b h w", "max");
```

---

## Negative

### negative

Element-wise negation. Equivalent to the unary `-` operator.

```cpp
Tensor negative() const;
```

**Returns** -- `Tensor` with each element negated.

```cpp
auto a = Tensor::arange(5);
auto b = a.negative();                   // [0, -1, -2, -3, -4]
auto c = -a;                             // equivalent
```

---

## View vs. Copy Summary

| Method | View | Copy | Condition |
|--------|:----:|:----:|-----------|
| `reshape` | * | * | View when contiguous and matching memory order; copy otherwise. |
| `view` | * | | Always a view. Throws if not contiguous. |
| `flatten` | * | * | View when the flattened region is contiguous; copy otherwise. |
| `unflatten` | * | * | View when possible via `reshape`. |
| `ravel` | * | * | Same as `flatten`. |
| `transpose` / `T` | * | | Always a view (stride permutation). |
| `swapaxes` | * | | Always a view. |
| `moveaxis` | * | | Always a view. |
| `squeeze` | * | | Always a view. |
| `unsqueeze` | * | | Always a view. |
| `expand` / `expand_as` / `broadcast_to` | * | | Always a view (zero strides). |
| `flip` / `flipud` / `fliplr` | * | | Always a view (negative strides). |
| `rot90` | * | | Composed from views. |
| `repeat` / `tile` | | * | Always copies data. |
| `roll` | | * | Always copies data. |
| `diagonal` | | * | Copies data. |
| `trace` | | * | Copies (calls `diagonal` then `sum`). |
| `rearrange` | * | * | View when possible. |
| `reduce` | | * | Always reduces (produces new data). |
| `negative` | | * | Always computes new data. |
