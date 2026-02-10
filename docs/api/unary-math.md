# Unary Math Operations

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

Element-wise unary math functions. All are available in both functional (`ops::`) and fluent (`tensor.`) forms.

## Core Math

### ops::negate / operator-

```cpp
Tensor ops::negate(const Tensor &input);
Tensor operator-(const Tensor &tensor);  // unary minus
```

Element-wise negation.

---

### ops::abs / Tensor::abs

```cpp
Tensor ops::abs(const Tensor &input);
Tensor Tensor::abs() const;
```

Element-wise absolute value. For complex types, returns the magnitude as a float tensor.

---

### ops::sqrt / Tensor::sqrt

```cpp
Tensor ops::sqrt(const Tensor &input);
Tensor Tensor::sqrt() const;
```

Element-wise square root.

---

### ops::exp / Tensor::exp

```cpp
Tensor ops::exp(const Tensor &input);
Tensor Tensor::exp() const;
```

Element-wise exponential (e^x).

---

### ops::log / Tensor::log

```cpp
Tensor ops::log(const Tensor &input);
Tensor Tensor::log() const;
```

Element-wise natural logarithm.

---

### ops::erf

```cpp
Tensor ops::erf(const Tensor &input);
```

Element-wise Gauss error function.

---

## Trigonometric

### ops::sin / Tensor::sin

```cpp
Tensor ops::sin(const Tensor &input);
Tensor Tensor::sin() const;
```

---

### ops::cos / Tensor::cos

```cpp
Tensor ops::cos(const Tensor &input);
Tensor Tensor::cos() const;
```

---

### ops::tan / Tensor::tan

```cpp
Tensor ops::tan(const Tensor &input);
Tensor Tensor::tan() const;
```

---

## NumPy-like Math

### ops::sign / Tensor::sign

```cpp
Tensor ops::sign(const Tensor &input);
Tensor Tensor::sign() const;
```

Returns -1, 0, or 1 for negative, zero, or positive elements.

---

### ops::floor / Tensor::floor

```cpp
Tensor ops::floor(const Tensor &input);
Tensor Tensor::floor() const;
```

Round down to nearest integer.

---

### ops::ceil / Tensor::ceil

```cpp
Tensor ops::ceil(const Tensor &input);
Tensor Tensor::ceil() const;
```

Round up to nearest integer.

---

### ops::trunc / Tensor::trunc

```cpp
Tensor ops::trunc(const Tensor &input);
Tensor Tensor::trunc() const;
```

Truncate toward zero.

---

### ops::round / Tensor::round

```cpp
Tensor ops::round(const Tensor &input, int decimals = 0);
Tensor Tensor::round(int decimals = 0) const;
```

Round to given number of decimal places.

**Parameters:**
- `decimals` (*int*) -- Number of decimal places. Default: `0`.

---

### ops::reciprocal / Tensor::reciprocal

```cpp
Tensor ops::reciprocal(const Tensor &input);
Tensor Tensor::reciprocal() const;
```

Element-wise 1/x.

---

### ops::square / Tensor::square

```cpp
Tensor ops::square(const Tensor &input);
Tensor Tensor::square() const;
```

Element-wise x^2.

---

### ops::cbrt / Tensor::cbrt

```cpp
Tensor ops::cbrt(const Tensor &input);
Tensor Tensor::cbrt() const;
```

Element-wise cube root.

---

## Element Testing

These return `Bool` tensors.

### ops::isnan / Tensor::isnan

```cpp
Tensor ops::isnan(const Tensor &input);
Tensor Tensor::isnan() const;
```

Returns `true` for NaN elements.

---

### ops::isinf / Tensor::isinf

```cpp
Tensor ops::isinf(const Tensor &input);
Tensor Tensor::isinf() const;
```

Returns `true` for infinite elements.

---

### ops::isfinite / Tensor::isfinite

```cpp
Tensor ops::isfinite(const Tensor &input);
Tensor Tensor::isfinite() const;
```

Returns `true` for finite (not NaN, not Inf) elements.

---

## Clipping

### ops::clip / Tensor::clip

```cpp
Tensor ops::clip(const Tensor &input, const Tensor &min_val,
                 const Tensor &max_val);
Tensor Tensor::clip(const Tensor &min_val, const Tensor &max_val) const;
Tensor Tensor::clip(double min_val, double max_val) const;
Tensor Tensor::clamp(double min_val, double max_val) const;  // alias
```

Clip values to the range `[min_val, max_val]`.

**Example:**
```cpp
auto x = Tensor::randn({3, 4});
auto clipped = x.clip(0.0, 1.0);  // values in [0, 1]
```

---

## Complex Number Operations

### ops::conj / Tensor::conj

```cpp
Tensor ops::conj(const Tensor &input);
Tensor Tensor::conj() const;
```

Complex conjugate.

---

### ops::real / Tensor::real

```cpp
Tensor ops::real(const Tensor &input);
Tensor Tensor::real() const;
```

Extract real part. Returns a zero-copy view for complex types.

---

### ops::imag / Tensor::imag

```cpp
Tensor ops::imag(const Tensor &input);
Tensor Tensor::imag() const;
```

Extract imaginary part. Returns a zero-copy view for complex types.

---

## Fluent Chaining

All unary operations can be chained:

```cpp
auto result = x.abs().sqrt().exp().log();
auto activated = (x * 2.0f + 1.0f).relu().sigmoid();
```

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Arithmetic](arithmetic), [Activations](activations)
