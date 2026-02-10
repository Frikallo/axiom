# Custom Functors

The `axiom::functors` namespace lets you apply custom C++ functions to tensor elements without writing backend kernels.

## apply (Unary)

Apply a function element-wise to a single tensor:

```cpp
using namespace axiom;

auto a = Tensor::randn({3, 4});

// Custom activation function
auto result = functors::apply(a, [](float x) -> float {
    return x > 0 ? x : 0.1f * x;  // Leaky ReLU
});
```

The functor receives and returns scalar values. The tensor is iterated element-by-element.

## apply (Binary)

Apply a binary function element-wise to two tensors:

```cpp
auto a = Tensor::randn({3, 4});
auto b = Tensor::randn({3, 4});

auto result = functors::apply(a, b, [](float x, float y) -> float {
    return std::max(x, y) - std::min(x, y);  // Absolute difference
});
```

## vectorize

Create a reusable vectorized function from a scalar function:

```cpp
// Define a custom function
auto smooth_l1 = functors::vectorize([](float x) -> float {
    float abs_x = std::abs(x);
    return abs_x < 1.0f ? 0.5f * x * x : abs_x - 0.5f;
});

// Use it like any other operation
auto a = Tensor::randn({100});
auto loss = smooth_l1(a);
```

`vectorize` returns a callable that accepts tensors and applies the function element-wise.

## fromfunc

Create a tensor from a function of indices:

```cpp
// Create a tensor where each element is f(i, j)
auto coords = functors::fromfunc(
    [](const std::vector<size_t> &indices) -> float {
        return static_cast<float>(indices[0] * 10 + indices[1]);
    },
    {5, 5},            // Shape
    DType::Float32      // Output dtype
);
// Result: [[0,1,2,3,4], [10,11,12,...], ...]
```

## apply_along_axis

Apply a function to 1D slices along a specific axis:

```cpp
auto a = Tensor::randn({4, 5});

// Custom reduction along axis 1
auto result = functors::apply_along_axis(
    [](const Tensor &slice) -> Tensor {
        // slice is a 1D tensor (length 5)
        return slice.max() - slice.min();  // Range
    },
    a, 1  // Apply along axis 1
);
// Shape: {4} - one result per row
```

The function receives a 1D tensor (a slice along the specified axis) and should return a tensor.

## apply_over_axes

Apply a function repeatedly over multiple axes:

```cpp
auto a = Tensor::randn({2, 3, 4});

auto result = functors::apply_over_axes(
    [](const Tensor &t, int axis) -> Tensor {
        return t.sum(axis, true);  // Sum with keepdims
    },
    a, {1, 2}  // Apply over axes 1 and 2
);
// Shape: {2, 1, 1}
```

## Performance Notes

Custom functors iterate elements in C++ and do not use SIMD vectorization or GPU acceleration. For performance-critical inner loops:

- Prefer built-in operations when available (they use SIMD and GPU)
- Use functors for prototyping or operations that aren't built-in
- Consider writing a custom backend kernel for hot paths

For complete function signatures, see [API Reference: Functors](../api/functors).
