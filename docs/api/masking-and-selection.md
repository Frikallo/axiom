# Masking & Selection

*For a tutorial introduction, see [User Guide: Indexing & Slicing](../user-guide/indexing-and-slicing).*

Conditional selection and masking operations.

## ops::where

```cpp
Tensor ops::where(const Tensor &condition, const Tensor &a, const Tensor &b);
```

Returns elements from `a` where `condition` is true, elements from `b` otherwise. All inputs are broadcast together.

**Parameters:**
- `condition` (*Tensor*) -- Boolean condition tensor.
- `a` (*Tensor*) -- Values where condition is true.
- `b` (*Tensor*) -- Values where condition is false.

**Example:**
```cpp
auto x = Tensor::randn({3, 4});
auto result = ops::where(x > 0.0f, x, Tensor::zeros({3, 4}));
```

---

## Tensor::where

```cpp
Tensor Tensor::where(const Tensor &condition, const Tensor &other) const;
Tensor Tensor::where(const Tensor &condition, float other) const;
Tensor Tensor::where(const Tensor &condition, double other) const;
Tensor Tensor::where(const Tensor &condition, int32_t other) const;
```

Member function form. Returns elements from `*this` where condition is true, `other` otherwise.

**Example:**
```cpp
auto result = x.where(x > 0.0f, 0.0f);  // ReLU equivalent
```

---

## ops::masked_fill / Tensor::masked_fill

```cpp
Tensor ops::masked_fill(const Tensor &input, const Tensor &mask, float value);
Tensor ops::masked_fill(const Tensor &input, const Tensor &mask, double value);
Tensor ops::masked_fill(const Tensor &input, const Tensor &mask,
                        const Tensor &value);

Tensor Tensor::masked_fill(const Tensor &mask, float value) const;
Tensor Tensor::masked_fill(const Tensor &mask, double value) const;
Tensor Tensor::masked_fill(const Tensor &mask, int32_t value) const;
Tensor Tensor::masked_fill(const Tensor &mask, const Tensor &value) const;
```

Returns a new tensor with positions where `mask` is true filled with `value`.

**Example:**
```cpp
// Attention masking
auto masked_scores = scores.masked_fill(!mask, -1e9f);

// Clamp to range
auto clamped = x.masked_fill(x < 0, 0.0f).masked_fill(x > 1, 1.0f);
```

---

## Tensor::masked_fill_

```cpp
Tensor &Tensor::masked_fill_(const Tensor &mask, float value);
Tensor &Tensor::masked_fill_(const Tensor &mask, double value);
Tensor &Tensor::masked_fill_(const Tensor &mask, int32_t value);
```

In-place version of `masked_fill`. Modifies the tensor directly.

---

## ops::masked_select / Tensor::masked_select

```cpp
Tensor ops::masked_select(const Tensor &input, const Tensor &mask);
Tensor Tensor::masked_select(const Tensor &mask) const;
```

Returns a 1D tensor containing elements where `mask` is true.

**Example:**
```cpp
auto x = Tensor::randn({3, 4});
auto positives = x.masked_select(x > 0.0f);  // 1D tensor of positive values
```

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Comparison](comparison), [Indexing Ops](indexing-ops), [Operators](operators)
