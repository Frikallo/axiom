# Error Handling

Axiom uses a structured exception hierarchy for clear, actionable error messages. All exceptions inherit from `AxiomError`, which inherits from `std::exception`.

## Exception Hierarchy

```
std::exception
  AxiomError
    ShapeError         Shape mismatches, invalid axes, bad reshapes
    TypeError          Unsupported dtypes, unsafe conversions
    DeviceError        Missing device, device mismatches
    ValueError         NaN/Inf detected, out-of-range values
    IndexError         Out-of-bounds indices, invalid slices
    MemoryError        Allocation failures, non-contiguous access
    RuntimeError       Unimplemented features, internal errors
  std::runtime_error
    io::SerializationError
      io::FileFormatError
    einops::EinopsError
      einops::EinopsParseError
      einops::EinopsShapeError
```

## Catching Errors

```cpp
using namespace axiom;

try {
    auto a = Tensor::zeros({3, 4});
    auto b = Tensor::zeros({5, 6});
    auto c = a + b;  // Shape mismatch!
} catch (const ShapeError &e) {
    std::cerr << "Shape error: " << e.what() << std::endl;
} catch (const AxiomError &e) {
    std::cerr << "Axiom error: " << e.what() << std::endl;
}
```

## Common Error Types

### ShapeError

```cpp
// Incompatible shapes for broadcasting
auto a = Tensor::zeros({3, 4});
auto b = Tensor::zeros({5, 6});
// a + b throws ShapeError::broadcast_incompatible(...)

// Invalid reshape
auto c = Tensor::zeros({3, 4});
// c.reshape({5, 5}) throws ShapeError::invalid_reshape(12, 25)

// Invalid axis
// c.sum(5) throws ShapeError::invalid_axis(5, 2)
```

### TypeError

```cpp
// Unsafe type conversion
auto a = Tensor::randn({3, 3});
// a.astype_safe(DType::Int32) throws if precision would be lost

// Unsupported dtype for operation
// Some operations don't support all dtypes
```

### DeviceError

```cpp
// GPU not available
// Tensor::zeros({3, 4}, DType::Float32, Device::GPU)
// throws DeviceError::not_available(Device::GPU) on non-Mac

// Direct data access on GPU tensor
auto gpu = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
// gpu.typed_data<float>() throws DeviceError::cpu_only("direct data access")
```

### ValueError

Thrown by safety rails (see below):

```cpp
auto a = Tensor::randn({3, 3});
// Manually introduce NaN
a.set_item<float>({0, 0}, std::numeric_limits<float>::quiet_NaN());

// a.nan_guard() throws ValueError::nan_detected(...)
// a.assert_finite() throws ValueError::not_finite(...)
```

## Safety Rails

Axiom provides chainable safety methods for detecting numerical issues during development:

### nan_guard

Throws `ValueError` if any NaN is detected:

```cpp
auto result = compute_something()
    .nan_guard();  // Throws if NaN present, otherwise returns *this
```

### assert_finite

Throws `ValueError` if any NaN or Inf is detected:

```cpp
auto result = compute_something()
    .assert_finite();  // Throws if NaN or Inf present
```

### assert_shape

Throws `ShapeError` if the tensor doesn't match the expected shape:

```cpp
auto tensor = get_model_output();

// Exact shape check
tensor.assert_shape({batch_size, 10});

// Named pattern check
tensor.assert_shape("batch classes");
```

### Inspection Without Throwing

```cpp
auto a = Tensor::randn({100, 100});

if (a.has_nan()) {
    std::cerr << "Warning: NaN detected" << std::endl;
}
if (a.has_inf()) {
    std::cerr << "Warning: Inf detected" << std::endl;
}
if (a.is_finite()) {
    // All values are finite
}
```

## Chaining Safety Rails

Safety methods return `*this`, so they chain naturally in computation pipelines:

```cpp
auto output = input
    .matmul(weights)
    .nan_guard()              // Check after matmul
    .relu()
    .assert_shape({batch, hidden})
    .assert_finite();         // Final check
```

## Best Practices

- Use `nan_guard()` after operations known to produce NaN (division, log, sqrt of negative)
- Use `assert_shape()` at module boundaries to catch shape bugs early
- Use `assert_finite()` before saving results or returning from functions
- Remove or disable safety checks in performance-critical production code
- Catch `AxiomError` as a generic fallback if you don't need type-specific handling

For complete API details, see [API Reference: Errors](../api/errors) and [API Reference: Debug](../api/debug).
