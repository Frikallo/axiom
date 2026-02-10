# Errors

Exception hierarchy for Axiom. All exceptions inherit from `AxiomError` which inherits from `std::exception`.

## AxiomError

```cpp
class AxiomError : public std::exception {
public:
    explicit AxiomError(const std::string &message);
    const char *what() const noexcept override;
    const std::string &message() const;
};
```

Base exception for all Axiom errors.

---

## ShapeError

```cpp
class ShapeError : public AxiomError {
public:
    explicit ShapeError(const std::string &message);

    static ShapeError mismatch(expected, got);
    static ShapeError broadcast_incompatible(details);
    static ShapeError invalid_axis(axis, ndim);
    static ShapeError invalid_reshape(from_size, to_size);
};
```

Raised for shape mismatches, invalid axes, incompatible broadcasting, or invalid reshapes.

---

## TypeError

```cpp
class TypeError : public AxiomError {
public:
    explicit TypeError(const std::string &message);

    static TypeError unsupported_dtype(dtype, operation);
    static TypeError dtype_mismatch(expected, got);
    static TypeError conversion_not_safe(from, to);
};
```

Raised for unsupported dtypes, dtype mismatches, or unsafe type conversions.

---

## DeviceError

```cpp
class DeviceError : public AxiomError {
public:
    explicit DeviceError(const std::string &message);

    static DeviceError not_available(device);
    static DeviceError mismatch(expected, got);
    static DeviceError cpu_only(operation);
};
```

Raised when a device is not available, tensors are on different devices, or a CPU-only operation is called on a GPU tensor.

---

## ValueError

```cpp
class ValueError : public AxiomError {
public:
    explicit ValueError(const std::string &message);

    static ValueError nan_detected(context);
    static ValueError inf_detected(context);
    static ValueError not_finite(context);
    static ValueError out_of_range(what, min, max, got);
};
```

Raised by safety rails (`nan_guard`, `assert_finite`) and for out-of-range values.

---

## IndexError

```cpp
class IndexError : public AxiomError {
public:
    explicit IndexError(const std::string &message);

    static IndexError out_of_bounds(index, size, dim);
    static IndexError invalid_slice(details);
};
```

Raised for out-of-bounds indices and invalid slices.

---

## MemoryError

```cpp
class MemoryError : public AxiomError {
public:
    explicit MemoryError(const std::string &message);

    static MemoryError allocation_failed(bytes);
    static MemoryError storage_too_small(required, available);
    static MemoryError not_contiguous(operation);
};
```

Raised for allocation failures, insufficient storage, and operations requiring contiguous memory.

---

## RuntimeError

```cpp
class RuntimeError : public AxiomError {
public:
    explicit RuntimeError(const std::string &message);

    static RuntimeError not_implemented(feature);
    static RuntimeError internal(details);
};
```

Raised for unimplemented features and internal errors.

---

## I/O Exceptions

```cpp
class io::SerializationError : public std::runtime_error { ... };
class io::FileFormatError : public io::SerializationError { ... };
```

---

## Einops Exceptions

```cpp
class einops::EinopsError : public std::runtime_error { ... };
class einops::EinopsParseError : public einops::EinopsError { ... };
class einops::EinopsShapeError : public einops::EinopsError { ... };
```

---

## Hierarchy

```
std::exception
  AxiomError
    ShapeError
    TypeError
    DeviceError
    ValueError
    IndexError
    MemoryError
    RuntimeError
  std::runtime_error
    io::SerializationError
      io::FileFormatError
    einops::EinopsError
      einops::EinopsParseError
      einops::EinopsShapeError
```

**See Also:** [Tensor Class](tensor-class), [Debug & Profiling](debug)
