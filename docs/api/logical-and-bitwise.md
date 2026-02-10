# Logical & Bitwise Operations

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

## Logical Operations

### ops::logical_and

```cpp
Tensor ops::logical_and(const Tensor &lhs, const Tensor &rhs);
```

Element-wise logical AND. Also available as `a && b`.

---

### ops::logical_or

```cpp
Tensor ops::logical_or(const Tensor &lhs, const Tensor &rhs);
```

Element-wise logical OR. Also available as `a || b`.

---

### ops::logical_xor

```cpp
Tensor ops::logical_xor(const Tensor &lhs, const Tensor &rhs);
```

Element-wise logical XOR.

---

### ops::logical_not

```cpp
Tensor ops::logical_not(const Tensor &input);
```

Element-wise logical NOT. Also available as `!tensor`.

**Example:**
```cpp
auto mask = x > 0.0f;
auto inverted = !mask;  // logical NOT
```

---

## Bitwise Operations

Integer types only. Throws `TypeError` on floating-point or complex types.

### ops::bitwise_and

```cpp
Tensor ops::bitwise_and(const Tensor &lhs, const Tensor &rhs);
```

Element-wise bitwise AND. Also available as `a & b`.

---

### ops::bitwise_or

```cpp
Tensor ops::bitwise_or(const Tensor &lhs, const Tensor &rhs);
```

Element-wise bitwise OR. Also available as `a | b`.

---

### ops::bitwise_xor

```cpp
Tensor ops::bitwise_xor(const Tensor &lhs, const Tensor &rhs);
```

Element-wise bitwise XOR. Also available as `a ^ b`.

---

### ops::left_shift

```cpp
Tensor ops::left_shift(const Tensor &lhs, const Tensor &rhs);
```

Element-wise left shift. Also available as `a << b`.

---

### ops::right_shift

```cpp
Tensor ops::right_shift(const Tensor &lhs, const Tensor &rhs);
```

Element-wise right shift. Also available as `a >> b`.

---

## Notes

- All operations support broadcasting.
- Bitwise operations require integer dtypes (Int8 through UInt64).
- Bitwise operations throw `TypeError` on complex types.

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Operators](operators), [Comparison](comparison)
