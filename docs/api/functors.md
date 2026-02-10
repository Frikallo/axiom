# Custom Functors

*For a tutorial introduction, see [User Guide: Custom Functors](../user-guide/custom-functors).*

Apply custom element-wise functions to tensors. All in the `axiom::ops` namespace. CPU only.

## ops::apply (Unary)

```cpp
template <typename Func>
Tensor ops::apply(const Tensor &input, Func &&func);
```

Apply a unary lambda element-wise. The lambda's argument and return types determine the input/output dtypes. Auto-casts input if needed.

**Example:**
```cpp
auto doubled = ops::apply(x, [](float v) { return v * 2.0f; });
auto clamped = ops::apply(x, [](float v) { return std::max(0.0f, v); });
```

---

## ops::apply (Binary)

```cpp
template <typename Func>
Tensor ops::apply(const Tensor &a, const Tensor &b, Func &&func);
```

Apply a binary lambda element-wise with broadcasting. Types are inferred from the lambda signature.

**Example:**
```cpp
auto result = ops::apply(a, b, [](float x, float y) {
    return x * x + y * y;
});
```

---

## ops::vectorize

```cpp
template <typename Func>
auto ops::vectorize(Func &&func);
```

Wrap a scalar lambda into a reusable callable that operates on tensors. Automatically handles unary (1-arg) and binary (2-arg) lambdas.

**Example:**
```cpp
auto safe_log = ops::vectorize([](float x) -> float {
    return x > 0.0f ? std::log(x) : -999.0f;
});
auto result = safe_log(tensor);  // Applies to entire tensor
```

---

## ops::fromfunc

```cpp
template <typename Func>
auto ops::fromfunc(Func &&func);
```

Alias for `vectorize`. Wraps a scalar function into a tensor-operating callable.

---

## ops::apply_along_axis

```cpp
template <typename Func>
Tensor ops::apply_along_axis(Func &&func1d, int axis, const Tensor &arr);
```

Apply a function to 1-D slices along a given axis. The function receives a 1D tensor and should return a scalar tensor or a 1D tensor.

**Parameters:**
- `func1d` -- Function taking a 1D Tensor, returning a scalar or 1D Tensor.
- `axis` (*int*) -- Axis along which to apply.
- `arr` (*Tensor*) -- Input tensor.

**Example:**
```cpp
// Custom normalization along rows
auto normalized = ops::apply_along_axis(
    [](const Tensor &row) {
        auto mx = row.max();
        return ops::divide(row, mx);
    }, 1, matrix);
```

---

## ops::apply_over_axes

```cpp
template <typename Func>
Tensor ops::apply_over_axes(Func &&func, const Tensor &a,
                            const std::vector<int> &axes);
```

Apply a function repeatedly over multiple axes. The function takes `(Tensor, int axis)` and must preserve `ndim`.

**Parameters:**
- `func` -- Function taking `(Tensor, int)` and returning a Tensor with the same ndim.
- `a` (*Tensor*) -- Input tensor.
- `axes` -- Axes to apply over sequentially.

**Example:**
```cpp
auto result = ops::apply_over_axes(
    [](const Tensor &t, int ax) {
        return t.sum(ax, true);  // keep_dims=true preserves ndim
    }, tensor, {0, 2});
```

---

## Notes

- All functor operations are CPU only. GPU tensors throw `DeviceError`.
- Lambda types are deduced automatically via `callable_traits`.
- Parallelization via OpenMP is applied when the tensor is large enough.

**See Also:** [Unary Math](unary-math), [Reductions](reductions)
