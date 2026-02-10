# Data Types

*For a tutorial introduction, see [User Guide: Tensor Basics](../user-guide/tensor-basics).*

## DType Enum

```cpp
enum class DType : uint8_t {
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Complex64, Complex128
};
```

| DType | C++ Type | Size (bytes) |
|-------|----------|:---:|
| `Bool` | `bool` | 1 |
| `Int8` | `int8_t` | 1 |
| `Int16` | `int16_t` | 2 |
| `Int32` | `int32_t` | 4 |
| `Int64` | `int64_t` | 8 |
| `UInt8` | `uint8_t` | 1 |
| `UInt16` | `uint16_t` | 2 |
| `UInt32` | `uint32_t` | 4 |
| `UInt64` | `uint64_t` | 8 |
| `Float16` | `float16_t` | 2 |
| `Float32` | `float` | 4 |
| `Float64` | `double` | 8 |
| `Complex64` | `std::complex<float>` | 8 |
| `Complex128` | `std::complex<double>` | 16 |

## Type Aliases

```cpp
using complex64_t = std::complex<float>;
using complex128_t = std::complex<double>;
// float16_t defined in axiom/float16.hpp
```

## Functions

### dtype_size

```cpp
constexpr size_t dtype_size(DType dtype);
```

Returns the size in bytes for a given dtype.

---

### dtype_name

```cpp
std::string dtype_name(DType dtype);
```

Returns a human-readable name (e.g., `"Float32"`).

---

### dtype_of_v

```cpp
template <typename T>
constexpr DType dtype_of_v;
```

Compile-time mapping from C++ type to DType. Example: `dtype_of_v<float>` is `DType::Float32`.

---

## Category Queries

```cpp
constexpr bool is_integer_dtype(DType dtype);
constexpr bool is_floating_dtype(DType dtype);
constexpr bool is_complex_dtype(DType dtype);
constexpr bool is_signed_integer_dtype(DType dtype);
constexpr bool is_unsigned_integer_dtype(DType dtype);
```

---

## Type Promotion

When two tensors with different dtypes interact, the result dtype is determined by NumPy-compatible promotion rules.

```cpp
DType type_conversion::promote_dtypes(DType dtype1, DType dtype2);
DType ops::promote_types(DType lhs_dtype, DType rhs_dtype);
DType ops::result_type(const Tensor &lhs, const Tensor &rhs);
```

General rules:
- Integer + Float = Float
- Smaller + Larger = Larger
- Real + Complex = Complex
- Bool is promoted to the other type

---

## Type Conversion

### Tensor::astype

```cpp
Tensor Tensor::astype(DType new_dtype) const;
```

Convert tensor to a new dtype. Returns a copy with the new type.

---

### Tensor::astype_safe

```cpp
Tensor Tensor::astype_safe(DType new_dtype) const;
```

Safe conversion that checks for precision loss.

---

### Convenience Methods

```cpp
Tensor Tensor::to_float() const;      // -> Float32
Tensor Tensor::to_double() const;     // -> Float64
Tensor Tensor::to_int() const;        // -> Int32
Tensor Tensor::to_int64() const;      // -> Int64
Tensor Tensor::to_bool() const;       // -> Bool
Tensor Tensor::to_complex() const;    // -> Complex64
Tensor Tensor::to_complex128() const; // -> Complex128
Tensor Tensor::half() const;          // -> Float16
```

---

## Precision Loss Detection

```cpp
bool type_conversion::conversion_may_lose_precision(DType from, DType to);
```

Returns `true` if converting `from` to `to` may lose information (e.g., Float64 to Float32, Int64 to Int32).

**See Also:** [Tensor Class](tensor-class), [Numeric Constants](numeric)
