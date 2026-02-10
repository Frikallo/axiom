# Reduction Operations

*For a tutorial introduction, see [User Guide: Reductions & Statistics](../user-guide/reductions-and-statistics).*

Reduce tensor dimensions by applying an aggregation function along one or more axes.

## ops::sum

```cpp
Tensor ops::sum(const Tensor &input, const std::vector<int> &axis = {},
                bool keep_dims = false);
Tensor Tensor::sum(int axis = -1, bool keep_dims = false) const;
Tensor Tensor::sum(const std::vector<int> &axes, bool keep_dims = false) const;
```

Sum of elements along the given axes. Empty `axis` reduces all dimensions.

**Parameters:**
- `axis` -- Axis or axes to reduce. Default: all axes.
- `keep_dims` (*bool*) -- If true, reduced dimensions are kept as size 1. Default: `false`.

**Example:**
```cpp
auto x = Tensor::ones({3, 4});
auto total = x.sum();        // scalar: 12.0
auto cols = x.sum(0);        // shape (4,): each column sum
auto rows = x.sum(1, true);  // shape (3, 1): each row sum, dim kept
```

---

## ops::mean

```cpp
Tensor ops::mean(const Tensor &input, const std::vector<int> &axis = {},
                 bool keep_dims = false);
Tensor Tensor::mean(int axis = -1, bool keep_dims = false) const;
Tensor Tensor::mean(const std::vector<int> &axes, bool keep_dims = false) const;
```

Arithmetic mean along the given axes.

---

## ops::max

```cpp
Tensor ops::max(const Tensor &input, const std::vector<int> &axis = {},
                bool keep_dims = false);
Tensor Tensor::max(int axis = -1, bool keep_dims = false) const;
```

Maximum value along the given axes.

---

## ops::min

```cpp
Tensor ops::min(const Tensor &input, const std::vector<int> &axis = {},
                bool keep_dims = false);
Tensor Tensor::min(int axis = -1, bool keep_dims = false) const;
```

Minimum value along the given axes.

---

## ops::argmax

```cpp
Tensor ops::argmax(const Tensor &input, int axis = -1, bool keep_dims = false);
Tensor Tensor::argmax(int axis = -1, bool keep_dims = false) const;
```

Index of the maximum value along an axis. Returns `Int64` tensor.

---

## ops::argmin

```cpp
Tensor ops::argmin(const Tensor &input, int axis = -1, bool keep_dims = false);
Tensor Tensor::argmin(int axis = -1, bool keep_dims = false) const;
```

Index of the minimum value along an axis. Returns `Int64` tensor.

---

## ops::prod

```cpp
Tensor ops::prod(const Tensor &input, const std::vector<int> &axis = {},
                 bool keep_dims = false);
Tensor Tensor::prod(int axis = -1, bool keep_dims = false) const;
Tensor Tensor::prod(const std::vector<int> &axes, bool keep_dims = false) const;
```

Product of elements along the given axes.

---

## ops::any

```cpp
Tensor ops::any(const Tensor &input, const std::vector<int> &axis = {},
                bool keep_dims = false);
Tensor Tensor::any(int axis = -1, bool keep_dims = false) const;
Tensor Tensor::any(const std::vector<int> &axes, bool keep_dims = false) const;
```

Returns `true` if any element is non-zero along the given axes.

---

## ops::all

```cpp
Tensor ops::all(const Tensor &input, const std::vector<int> &axis = {},
                bool keep_dims = false);
Tensor Tensor::all(int axis = -1, bool keep_dims = false) const;
Tensor Tensor::all(const std::vector<int> &axes, bool keep_dims = false) const;
```

Returns `true` if all elements are non-zero along the given axes.

---

## Statistical Operations

These are composition-based (built from other reductions).

### Tensor::var

```cpp
Tensor Tensor::var(int axis = -1, int ddof = 0, bool keep_dims = false) const;
Tensor Tensor::var(const std::vector<int> &axes, int ddof = 0,
                   bool keep_dims = false) const;
```

Variance along the given axes.

**Parameters:**
- `ddof` (*int*) -- Delta degrees of freedom. Default: `0` (population variance). Use `1` for sample variance.

**Example:**
```cpp
auto x = Tensor::randn({100});
auto pop_var = x.var();        // population variance (ddof=0)
auto samp_var = x.var(-1, 1);  // sample variance (ddof=1)
```

---

### Tensor::std

```cpp
Tensor Tensor::std(int axis = -1, int ddof = 0, bool keep_dims = false) const;
Tensor Tensor::std(const std::vector<int> &axes, int ddof = 0,
                   bool keep_dims = false) const;
```

Standard deviation (square root of variance).

---

### Tensor::ptp

```cpp
Tensor Tensor::ptp(int axis = -1, bool keep_dims = false) const;
```

Peak-to-peak value (max - min) along an axis.

---

## Axis Semantics

- **Positive axis:** Counts from the front (0 = first dimension).
- **Negative axis:** Counts from the back (-1 = last dimension).
- **Empty axis vector:** Reduces all dimensions to a scalar.

```cpp
auto x = Tensor::randn({2, 3, 4});
auto a = x.sum(0);            // shape (3, 4)
auto b = x.sum(-1);           // shape (2, 3)
auto c = x.sum({0, 2});       // shape (3,)
auto d = x.sum({}, false);    // scalar (all dims reduced)
```

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Unary Math](unary-math), [Activations](activations)
