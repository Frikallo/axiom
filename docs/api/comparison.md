# Comparison Operations

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

Element-wise comparison operations. All return `Bool` tensors and support broadcasting.

## ops::equal

```cpp
Tensor ops::equal(const Tensor &lhs, const Tensor &rhs);
```

Element-wise equality test.

**Returns:** Bool tensor. Also available as `a == b`.

---

## ops::not_equal

```cpp
Tensor ops::not_equal(const Tensor &lhs, const Tensor &rhs);
```

Element-wise inequality test.

**Returns:** Bool tensor. Also available as `a != b`.

---

## ops::less

```cpp
Tensor ops::less(const Tensor &lhs, const Tensor &rhs);
```

Element-wise less-than.

**Returns:** Bool tensor. Also available as `a < b`.

---

## ops::less_equal

```cpp
Tensor ops::less_equal(const Tensor &lhs, const Tensor &rhs);
```

Element-wise less-than-or-equal.

**Returns:** Bool tensor. Also available as `a <= b`.

---

## ops::greater

```cpp
Tensor ops::greater(const Tensor &lhs, const Tensor &rhs);
```

Element-wise greater-than.

**Returns:** Bool tensor. Also available as `a > b`.

---

## ops::greater_equal

```cpp
Tensor ops::greater_equal(const Tensor &lhs, const Tensor &rhs);
```

Element-wise greater-than-or-equal.

**Returns:** Bool tensor. Also available as `a >= b`.

---

## Scalar Comparisons

All comparison operators work with scalars on either side:

```cpp
auto mask = x > 0.0f;     // tensor > scalar
auto mask = 5.0f < x;     // scalar < tensor
auto mask = x == 3;        // tensor == scalar
```

---

## Approximate Comparison Methods

### Tensor::isclose

```cpp
Tensor Tensor::isclose(const Tensor &other, double rtol = 1e-5,
                       double atol = 1e-8) const;
```

Element-wise approximate equality. Returns Bool tensor where `|a - b| <= atol + rtol * |b|`.

**Parameters:**
- `other` (*Tensor*) -- Tensor to compare against.
- `rtol` (*double*) -- Relative tolerance. Default: `1e-5`.
- `atol` (*double*) -- Absolute tolerance. Default: `1e-8`.

**Returns:** Bool tensor.

---

### Tensor::allclose

```cpp
bool Tensor::allclose(const Tensor &other, double rtol = 1e-5,
                      double atol = 1e-8) const;
```

Returns `true` if all elements satisfy `isclose`.

**Example:**
```cpp
auto a = Tensor::full({3}, 1.0f);
auto b = Tensor::full({3}, 1.0f + 1e-7f);
bool close = a.allclose(b);  // true
```

---

### Tensor::array_equal

```cpp
bool Tensor::array_equal(const Tensor &other) const;
```

Returns `true` if shapes and all elements are exactly equal.

---

## Notes

- Ordering comparisons (`<`, `>`, `<=`, `>=`) throw `TypeError` on complex types.
- All operations support both CPU and GPU.

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Operators](operators), [Masking & Selection](masking-and-selection)
