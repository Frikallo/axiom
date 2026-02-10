# Operators

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

All overloaded C++ operators for the Tensor class. Defined in `axiom/tensor_operators.hpp`.

## Arithmetic Operators

| Operator | Function | Notes |
|----------|----------|-------|
| `a + b` | `ops::add(a, b)` | |
| `a - b` | `ops::subtract(a, b)` | |
| `a * b` | `ops::multiply(a, b)` | |
| `a / b` | `ops::divide(a, b)` | |
| `a % b` | `ops::modulo(a, b)` | |
| `-a` | `ops::negate(a)` | Unary minus |

All support Tensor-Tensor and Tensor-scalar operands:

```cpp
auto c = a + b;         // Tensor + Tensor
auto d = a + 2.0f;      // Tensor + scalar
auto e = 3.0f * a;      // scalar * Tensor
```

## In-Place Arithmetic

| Operator | Function |
|----------|----------|
| `a += b` | `ops::add_inplace(a, b)` |
| `a -= b` | `ops::subtract_inplace(a, b)` |
| `a *= b` | `ops::multiply_inplace(a, b)` |
| `a /= b` | `ops::divide_inplace(a, b)` |

Scalar overloads: `a += 1.0f`, `a *= 2`, etc.

## Comparison Operators

| Operator | Function | Returns |
|----------|----------|---------|
| `a == b` | `ops::equal(a, b)` | Bool tensor |
| `a != b` | `ops::not_equal(a, b)` | Bool tensor |
| `a < b` | `ops::less(a, b)` | Bool tensor |
| `a <= b` | `ops::less_equal(a, b)` | Bool tensor |
| `a > b` | `ops::greater(a, b)` | Bool tensor |
| `a >= b` | `ops::greater_equal(a, b)` | Bool tensor |

Scalar overloads on both sides: `x > 0.0f`, `5.0f < x`.

## Logical Operators

| Operator | Function |
|----------|----------|
| `a && b` | `ops::logical_and(a, b)` |
| `a \|\| b` | `ops::logical_or(a, b)` |
| `!a` | `ops::logical_xor(a, ones)` |

## Bitwise Operators

Integer types only.

| Operator | Function |
|----------|----------|
| `a & b` | `ops::bitwise_and(a, b)` |
| `a \| b` | `ops::bitwise_or(a, b)` |
| `a ^ b` | `ops::bitwise_xor(a, b)` |
| `a << b` | `ops::left_shift(a, b)` |
| `a >> b` | `ops::right_shift(a, b)` |

## Stream Output

```cpp
std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
```

Prints tensor in NumPy-like format.

```cpp
auto x = Tensor::arange(6).reshape({2, 3});
std::cout << x << std::endl;
```

**See Also:** [Arithmetic](arithmetic), [Comparison](comparison), [Logical & Bitwise](logical-and-bitwise)
