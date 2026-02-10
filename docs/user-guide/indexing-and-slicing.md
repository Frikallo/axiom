# Indexing & Slicing

Axiom supports NumPy-style slicing and advanced indexing operations for selecting and modifying tensor elements.

## Basic Slicing

Use the `slice()` method with `Slice` objects to select sub-tensors. Slicing always returns a zero-copy view.

```cpp
using namespace axiom;

auto a = Tensor::arange(10);              // [0, 1, 2, ..., 9]

// Slice(start, stop, step)
auto b = a.slice({Slice(2, 7)});          // [2, 3, 4, 5, 6]
auto c = a.slice({Slice(0, 10, 2)});      // [0, 2, 4, 6, 8]
auto d = a.slice({Slice(5, 10)});         // [5, 6, 7, 8, 9]
```

## Multi-dimensional Slicing

```cpp
auto m = Tensor::arange(20).reshape({4, 5});

// Slice rows and columns
auto row = m.slice({Slice(1, 2)});        // Row 1, shape: {1, 5}
auto cols = m.slice({Slice(), Slice(0, 3)}); // All rows, first 3 cols

// Step slicing
auto every_other = m.slice({Slice(0, 4, 2), Slice()});  // Rows 0, 2
```

## Operator Indexing

The `operator[]` provides a compact syntax with `Index` objects:

```cpp
auto a = Tensor::arange(20).reshape({4, 5});
auto b = a[{Index(1), Index::all()}];     // Row 1
```

## Boolean Masking

Use comparison operators to create boolean masks, then select elements with `masked_select`:

```cpp
auto a = Tensor::randn({5, 5});

// Create boolean mask
auto mask = ops::greater(a, Tensor::zeros({1}));  // Elements > 0

// Select elements where mask is true (returns 1D tensor)
auto positives = a.masked_select(mask);

// Replace elements using masked_fill
auto clamped = a.masked_fill(mask, 0.0f);         // Zero out positives
```

## Where (Conditional Selection)

`ops::where` selects elements from two tensors based on a condition:

```cpp
auto x = Tensor::randn({3, 4});
auto y = Tensor::zeros({3, 4});

auto condition = ops::greater(x, Tensor::zeros({1}));
auto result = ops::where(condition, x, y);  // x where positive, else 0

// Member function version
auto clamped = x.where(condition, 0.0f);    // x where condition, else 0
```

## Gather and Scatter

`gather` collects values along an axis according to index tensors:

```cpp
auto data = Tensor::arange(12).reshape({3, 4}).to_float();

// Gather specific elements along axis 1
int64_t idx_data[] = {0, 2, 1, 3, 0, 2};
auto indices = Tensor::from_data(idx_data, {3, 2});
auto gathered = data.gather(1, indices);   // Shape: {3, 2}
```

`scatter` places values into a tensor at specified indices:

```cpp
auto dest = Tensor::zeros({3, 4});
auto src = Tensor::ones({3, 2});
int64_t idx_data[] = {0, 2, 1, 3, 0, 2};
auto indices = Tensor::from_data(idx_data, {3, 2});
auto result = dest.scatter(1, indices, src);

// In-place version
dest.scatter_(1, indices, src);
```

## Index Select

Select entire slices along a dimension:

```cpp
auto a = Tensor::arange(20).reshape({4, 5}).to_float();

// Select rows 0 and 2
int64_t idx[] = {0, 2};
auto indices = Tensor::from_data(idx, {2});
auto selected = a.index_select(0, indices);  // Shape: {2, 5}

// Select columns 1 and 3
int64_t col_idx[] = {1, 3};
auto col_indices = Tensor::from_data(col_idx, {2});
auto cols = a.index_select(1, col_indices);  // Shape: {4, 2}
```

## Take and Put Along Axis

```cpp
auto a = Tensor::arange(12).reshape({3, 4}).to_float();

// Take from flattened tensor
int64_t flat_idx[] = {0, 5, 11};
auto flat_indices = Tensor::from_data(flat_idx, {3});
auto taken = ops::take(a, flat_indices);     // Shape: {3}

// Take along a specific axis
int64_t axis_idx[] = {0, 2, 1, 3};
auto along_indices = Tensor::from_data(axis_idx, {1, 4});
auto along = ops::take_along_axis(a, along_indices, 0);

// Put values along axis
auto values = Tensor::ones({1, 4});
auto result = ops::put_along_axis(a, along_indices, values, 0);
```

## Padding

Add padding around tensor edges:

```cpp
auto a = Tensor::ones({3, 3});

// Constant padding: (before, after) pairs per dimension
auto padded = ops::pad(a, {{1, 1}, {2, 2}}, "constant", 0.0);
// Shape: {5, 7}, surrounded by zeros

// Reflect padding
auto reflected = ops::pad(a, {{1, 1}, {1, 1}}, "reflect");

// Replicate (edge) padding
auto replicated = ops::pad(a, {{1, 1}, {1, 1}}, "replicate");

// Circular (wrap) padding
auto circular = ops::pad(a, {{1, 1}, {1, 1}}, "circular");
```

## In-place Masked Operations

```cpp
auto a = Tensor::randn({4, 4});
auto mask = ops::less(a, Tensor::zeros({1}));

// In-place fill where mask is true
a.masked_fill_(mask, 0.0f);              // Zero out negatives in-place
```

For complete function signatures, see [API Reference: Masking & Selection](../api/masking-and-selection), [API Reference: Indexing Ops](../api/indexing-ops), and [API Reference: System](../api/system).
