# Arithmetic Operations

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

Binary arithmetic operations with broadcasting and type promotion.

## ops::add

```cpp
Tensor ops::add(const Tensor &lhs, const Tensor &rhs);
```

Element-wise addition. Inputs are broadcast together.

**Parameters:**
- `lhs` (*Tensor*) -- Left operand.
- `rhs` (*Tensor*) -- Right operand.

**Returns:** Tensor with promoted dtype.

**Example:**
```cpp
auto a = Tensor::ones({3, 4});
auto b = Tensor::ones({3, 4});
auto c = ops::add(a, b);  // or: a + b
```

**See Also:** [operator+](operators), [subtract](#ops-subtract)

---

## ops::subtract

```cpp
Tensor ops::subtract(const Tensor &lhs, const Tensor &rhs);
```

Element-wise subtraction.

**Parameters:**
- `lhs` (*Tensor*) -- Left operand.
- `rhs` (*Tensor*) -- Right operand.

**Returns:** Tensor with promoted dtype.

**Example:**
```cpp
auto c = ops::subtract(a, b);  // or: a - b
```

---

## ops::multiply

```cpp
Tensor ops::multiply(const Tensor &lhs, const Tensor &rhs);
```

Element-wise multiplication.

**Parameters:**
- `lhs` (*Tensor*) -- Left operand.
- `rhs` (*Tensor*) -- Right operand.

**Returns:** Tensor with promoted dtype.

**Example:**
```cpp
auto c = ops::multiply(a, b);  // or: a * b
```

---

## ops::divide

```cpp
Tensor ops::divide(const Tensor &lhs, const Tensor &rhs);
```

Element-wise division.

**Parameters:**
- `lhs` (*Tensor*) -- Dividend.
- `rhs` (*Tensor*) -- Divisor.

**Returns:** Tensor with promoted dtype.

**Notes:**
- Integer division truncates toward zero (like C++).
- Floating-point division by zero returns `inf` or `nan`.

**Example:**
```cpp
auto c = ops::divide(a, b);  // or: a / b
```

---

## ops::power

```cpp
Tensor ops::power(const Tensor &lhs, const Tensor &rhs);
```

Element-wise exponentiation: `lhs ^ rhs`.

**Parameters:**
- `lhs` (*Tensor*) -- Base.
- `rhs` (*Tensor*) -- Exponent.

**Returns:** Tensor with promoted dtype.

**Example:**
```cpp
auto c = ops::power(a, b);
```

**Notes:** No operator overload -- use `ops::power()` directly.

---

## ops::modulo

```cpp
Tensor ops::modulo(const Tensor &lhs, const Tensor &rhs);
```

Element-wise modulo.

**Parameters:**
- `lhs` (*Tensor*) -- Dividend.
- `rhs` (*Tensor*) -- Divisor.

**Returns:** Tensor with promoted dtype.

**Example:**
```cpp
auto c = ops::modulo(a, b);  // or: a % b
```

---

## In-Place Operations

```cpp
void ops::add_inplace(Tensor &lhs, const Tensor &rhs);
void ops::subtract_inplace(Tensor &lhs, const Tensor &rhs);
void ops::multiply_inplace(Tensor &lhs, const Tensor &rhs);
void ops::divide_inplace(Tensor &lhs, const Tensor &rhs);
```

Modify `lhs` in place. Also available via operators `+=`, `-=`, `*=`, `/=`.

**Example:**
```cpp
auto x = Tensor::ones({3, 4});
x += Tensor::ones({3, 4});  // x is now 2.0 everywhere
```

---

## Broadcasting

All binary arithmetic operations support NumPy-style broadcasting. Shapes are compared element-wise from the trailing dimensions:

- Dimensions of size 1 are broadcast to match the other operand.
- Missing dimensions are treated as size 1.

```cpp
auto a = Tensor::randn({3, 4});   // (3, 4)
auto b = Tensor::randn({4});      // (4,)
auto c = a + b;                    // (3, 4) -- b is broadcast

auto d = Tensor::randn({3, 1});   // (3, 1)
auto e = a + d;                    // (3, 4) -- d is broadcast
```

## Type Promotion

When operands have different dtypes, the result dtype is determined by NumPy-compatible promotion rules. Use `ops::result_type(a, b)` to query the promoted dtype.

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |
