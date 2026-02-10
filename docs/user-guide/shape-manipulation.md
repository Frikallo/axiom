# Shape Manipulation

Axiom provides a rich set of shape manipulation operations, many of which create zero-copy views rather than copying data.

## Views vs Copies

Understanding when an operation creates a view (shares memory) vs a copy is crucial for performance:

| Operation | View? | Notes |
|-----------|-------|-------|
| `reshape` | Usually | Copy only if non-contiguous and shape is incompatible |
| `view` | Always | Throws if not possible without copy |
| `transpose` | Always | Permutes strides, no data movement |
| `squeeze` / `unsqueeze` | Always | Adds/removes size-1 dimensions |
| `expand` | Always | Uses 0-stride for broadcasted dims |
| `flatten` | Usually | Copy if non-contiguous |
| `flip` | Always | Uses negative strides |
| `slice` | Always | Adjusts offset and strides |
| `copy` / `clone` | Never | Always allocates new storage |

Check with `tensor.is_view()` and `tensor.shares_storage(other)`.

## Reshape and View

`reshape` returns a view when possible, falling back to a copy:

```cpp
using namespace axiom;

auto a = Tensor::arange(12);             // Shape: {12}
auto b = a.reshape({3, 4});              // Shape: {3, 4} - view
auto c = a.reshape({2, 2, 3});           // Shape: {2, 2, 3} - view

// Check if reshape would require a copy
if (a.would_materialize_on_reshape({3, 4})) {
    // This reshape needs to copy data
}
```

`view` is stricter -- it throws `ShapeError` if a view is not possible:

```cpp
auto a = Tensor::arange(12);
auto b = a.view({3, 4});                 // OK: contiguous input
// Non-contiguous tensors may fail:
// a.transpose().view({12});             // Throws ShapeError
```

Use `-1` for one dimension to let Axiom infer it:

```cpp
auto a = Tensor::zeros({2, 3, 4});       // 24 elements
auto b = a.reshape({6, -1});             // Inferred: {6, 4}
```

## Transpose and Permute

```cpp
auto a = Tensor::zeros({3, 4});

// 2D transpose
auto b = a.transpose();                  // Shape: {4, 3}
auto c = a.T();                          // Same thing

// Arbitrary axis permutation
auto d = Tensor::zeros({2, 3, 4, 5});
auto e = d.transpose({0, 3, 1, 2});     // Shape: {2, 5, 3, 4}

// Swap two axes
auto f = d.swapaxes(1, 3);              // Shape: {2, 5, 4, 3}

// Move an axis to a new position
auto g = d.moveaxis(3, 1);              // Shape: {2, 5, 3, 4}
```

All transpose operations return views with permuted strides.

## Squeeze and Unsqueeze

```cpp
auto a = Tensor::zeros({1, 3, 1, 4});

// Remove all size-1 dimensions
auto b = a.squeeze();                    // Shape: {3, 4}

// Remove specific dimension
auto c = a.squeeze(0);                   // Shape: {3, 1, 4}
auto d = a.squeeze(2);                   // Shape: {1, 3, 4}

// Add a dimension
auto e = Tensor::zeros({3, 4});
auto f = e.unsqueeze(0);                 // Shape: {1, 3, 4}
auto g = e.unsqueeze(-1);               // Shape: {3, 4, 1}
```

## Flatten and Unflatten

```cpp
auto a = Tensor::zeros({2, 3, 4});

// Flatten all dimensions
auto b = a.flatten();                    // Shape: {24}
auto c = a.ravel();                      // Same thing (NumPy alias)

// Flatten a range of dimensions
auto d = a.flatten(1, 2);               // Shape: {2, 12}
auto e = a.flatten(0, 1);               // Shape: {6, 4}

// Unflatten a dimension
auto f = e.unflatten(0, {2, 3});        // Shape: {2, 3, 4}
```

## Expand and Repeat

`expand` creates a zero-copy view using 0-stride broadcasting:

```cpp
auto a = Tensor::ones({1, 3});
auto b = a.expand({4, 3});              // Shape: {4, 3} - view, no copy
auto c = a.broadcast_to({4, 3});        // Same thing (NumPy alias)

auto x = Tensor::ones({3, 1});
auto y = x.expand_as(Tensor::zeros({3, 5}));  // Shape: {3, 5}
```

`repeat` copies data to create a repeated tensor:

```cpp
auto a = Tensor::arange(3);             // [0, 1, 2]
auto b = a.repeat({3});                 // [0, 1, 2, 0, 1, 2, 0, 1, 2]
auto c = Tensor::ones({2, 3}).tile({2, 3}); // Shape: {4, 9}
```

## Flip and Roll

```cpp
auto a = Tensor::arange(12).reshape({3, 4});

// Flip along an axis (uses negative strides - zero-copy)
auto b = a.flip(0);                      // Reverse rows
auto c = a.flip(1);                      // Reverse columns
auto d = a.flip({0, 1});                 // Reverse both

// Convenience aliases
auto e = a.flipud();                     // Flip up-down (axis 0)
auto f = a.fliplr();                     // Flip left-right (axis 1)

// Roll elements along an axis
auto g = a.roll(2, 1);                  // Shift columns by 2
auto h = a.roll({1, -1}, {0, 1});       // Multi-axis roll

// Rotate 90 degrees
auto i = a.rot90(1, {0, 1});            // 90-degree rotation
auto j = a.rot90(2, {0, 1});            // 180-degree rotation
```

## Diagonal and Trace

```cpp
auto a = Tensor::arange(16).reshape({4, 4});

// Extract diagonal
auto d = a.diagonal();                   // Main diagonal
auto d1 = a.diagonal(1);                // 1 above main diagonal
auto dm1 = a.diagonal(-1);              // 1 below main diagonal

// Create diagonal matrix from 1D tensor
auto v = Tensor::arange(3);
auto m = Tensor::diag(v);               // 3x3 diagonal matrix
auto m1 = Tensor::diag(v, 1);           // 4x4 with offset diagonal

// Matrix trace (sum of diagonal)
auto t = a.trace();                      // Sum of main diagonal
```

## Stacking and Splitting

```cpp
auto a = Tensor::ones({2, 3});
auto b = Tensor::zeros({2, 3});

// Concatenate along existing axis
auto c = Tensor::cat({a, b}, 0);         // Shape: {4, 3}
auto d = Tensor::concatenate({a, b}, 1); // Shape: {2, 6}

// Stack along NEW axis
auto e = Tensor::stack({a, b}, 0);       // Shape: {2, 2, 3}
auto f = Tensor::stack({a, b}, -1);      // Shape: {2, 3, 2}

// Convenience stacking
auto g = Tensor::vstack({a, b});         // Shape: {4, 3}
auto h = Tensor::hstack({a, b});         // Shape: {2, 6}

// Split into equal sections
auto parts = c.split(2, 0);             // Two {2, 3} tensors

// Split at indices
auto pieces = c.split({1, 3}, 0);       // Sizes: {1,3}, {2,3}, {1,3}

// Chunk (may be unequal)
auto chunks = c.chunk(3, 0);            // ~equal chunks along axis 0
```

## Triangular Matrices

```cpp
auto a = Tensor::ones({4, 4});

// Lower/upper triangular
auto lo = Tensor::tril(a);              // Lower triangle
auto up = Tensor::triu(a);              // Upper triangle
auto lo1 = Tensor::tril(a, -1);         // Strict lower triangle

// Triangular matrix from scratch
auto t = Tensor::tri(4, 4, 0);          // Lower triangular ones
```

## Einops Rearrange

For complex reshaping and transposition, einops pattern syntax is often clearer:

```cpp
auto images = Tensor::randn({8, 3, 32, 32});

// Merge batch and channel dims
auto flat = images.rearrange("b c h w -> b (c h) w");

// Split a dimension
auto patches = images.rearrange(
    "b c (h ph) (w pw) -> b (h w) (c ph pw)",
    {{"ph", 8}, {"pw", 8}}
);

// Transpose with named dimensions
auto transposed = images.rearrange("b c h w -> b h w c");
```

For complete function signatures, see [API Reference: Tensor Manipulation](../api/tensor-manipulation) and [API Reference: Stacking & Splitting](../api/stacking-and-splitting).
