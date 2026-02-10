# Indexing Operations

*For a tutorial introduction, see [User Guide: Indexing & Slicing](../user-guide/indexing-and-slicing).*

Advanced indexing operations for gathering and scattering tensor elements.

## ops::gather / Tensor::gather

```cpp
Tensor ops::gather(const Tensor &input, int dim, const Tensor &indices);
Tensor Tensor::gather(int dim, const Tensor &indices) const;
```

Gather values along an axis according to indices. Like PyTorch's `torch.gather`.

For a 3D tensor: `output[i][j][k] = input[indices[i][j][k]][j][k]` when `dim == 0`.

**Parameters:**
- `dim` (*int*) -- Dimension along which to gather.
- `indices` (*Tensor*) -- Index tensor. Must have the same number of dimensions as input.

**Example:**
```cpp
auto x = Tensor::randn({4, 5});
auto idx = Tensor::from_data(data, {4, 2});  // Int64 indices
auto gathered = x.gather(1, idx);  // Gather along columns
```

---

## ops::scatter / Tensor::scatter

```cpp
Tensor ops::scatter(const Tensor &input, int dim, const Tensor &indices,
                    const Tensor &src);
Tensor Tensor::scatter(int dim, const Tensor &indices, const Tensor &src) const;
```

Scatter values into a tensor at positions specified by indices.

For `dim == 0`: `output[indices[i][j][k]][j][k] = src[i][j][k]`.

**Parameters:**
- `dim` (*int*) -- Dimension along which to scatter.
- `indices` (*Tensor*) -- Index tensor.
- `src` (*Tensor*) -- Source values to scatter.

---

## Tensor::scatter_

```cpp
Tensor &Tensor::scatter_(int dim, const Tensor &indices, const Tensor &src);
```

In-place version of scatter.

---

## ops::index_select / Tensor::index_select

```cpp
Tensor ops::index_select(const Tensor &input, int dim, const Tensor &indices);
Tensor Tensor::index_select(int dim, const Tensor &indices) const;
```

Select slices along a dimension using 1D index tensor. More efficient than `gather` when selecting whole slices.

**Parameters:**
- `dim` (*int*) -- Dimension to select along.
- `indices` (*Tensor*) -- 1D tensor of indices.

**Example:**
```cpp
auto x = Tensor::randn({10, 5});
auto idx = Tensor::from_data(int64_t[]{0, 3, 7}, {3});
auto selected = x.index_select(0, idx);  // Select rows 0, 3, 7 -> shape (3, 5)
```

---

## ops::take

```cpp
Tensor ops::take(const Tensor &input, const Tensor &indices, int axis = -1);
```

Take elements from a tensor along an axis. If `axis == -1`, takes from the flattened tensor (like NumPy's `np.take`).

**Parameters:**
- `indices` (*Tensor*) -- Index tensor.
- `axis` (*int*) -- Axis to take along. Default: `-1` (flattened).

---

## ops::take_along_axis

```cpp
Tensor ops::take_along_axis(const Tensor &input, const Tensor &indices, int axis);
```

Take values along an axis using indices of the same shape (like NumPy's `np.take_along_axis`). Indices must broadcast with input except along the specified axis.

---

## ops::put_along_axis

```cpp
Tensor ops::put_along_axis(const Tensor &input, const Tensor &indices,
                           const Tensor &values, int axis);
```

Put values along an axis using indices. Returns a new tensor with values placed at the specified positions.

---

## Notes

- All operations support negative indices.
- Index tensors should be `Int64` dtype.

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Masking & Selection](masking-and-selection), [Tensor Manipulation](tensor-manipulation)
