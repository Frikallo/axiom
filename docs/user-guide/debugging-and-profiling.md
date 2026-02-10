# Debugging & Profiling

Axiom includes built-in tools for inspecting tensors, tracing operations, profiling performance, and querying CPU capabilities.

## Tensor Inspection

### debug_info

Get a detailed diagnostic string for any tensor:

```cpp
using namespace axiom;

auto a = Tensor::randn({4, 5});
std::cout << a.debug_info() << std::endl;
```

This prints shape, dtype, device, strides, contiguity flags, memory usage, and whether the tensor is a view or lazy.

### Printing Tensors

```cpp
auto a = Tensor::arange(12).reshape({3, 4}).to_float();

// NumPy-style formatted output
std::cout << a << std::endl;
// Or:
std::cout << a.str() << std::endl;

// Detailed repr (includes shape and dtype)
std::cout << a.repr() << std::endl;
```

## Operation Tracing

The `trace` namespace records a timeline of operations for debugging computation sequences.

### Basic Tracing

```cpp
trace::enable();
trace::clear();

auto a = Tensor::randn({100, 100});
auto b = Tensor::randn({100, 100});
auto c = a.matmul(b);
auto d = c.relu();
auto e = d.sum();

std::cout << trace::dump() << std::endl;

trace::disable();
```

`trace::dump()` returns a formatted string showing each operation's name, description, timing, and memory usage.

### Scoped Tracing

Use `ScopedTrace` for RAII-style trace recording:

```cpp
trace::enable();

{
    trace::ScopedTrace t("forward_pass", "Batch forward",
                         input.nbytes());
    auto h = input.matmul(w1).relu();
    auto out = h.matmul(w2).softmax(1);
}  // Duration recorded here

std::cout << trace::dump() << std::endl;
```

### Trace Control

```cpp
trace::enable();       // Start recording
trace::disable();      // Stop recording
trace::clear();        // Clear all events
bool on = trace::is_enabled();
```

## Profiling

The `profile` namespace provides lightweight per-operation profiling.

```cpp
profile::enable();

auto c = a.matmul(b);

const auto &op = profile::last_op();
std::cout << "Operation: " << op.name << std::endl;
std::cout << "Duration: "
          << op.duration.count() / 1e6 << " ms" << std::endl;
std::cout << "Input bytes: " << op.input_bytes << std::endl;
std::cout << "Output bytes: " << op.output_bytes << std::endl;
std::cout << "Shapes: " << op.shape_info << std::endl;

profile::disable();
```

## CPU Diagnostics

Query the SIMD architecture and CPU capabilities:

```cpp
// Print SIMD info to stdout
cpu_info::print_simd_info();

// Get architecture name
const char *arch = cpu_info::simd_arch_name();
// e.g., "neon64", "avx2", "sse4.2"

// Get compact info string
std::string info = cpu_info::simd_info_string();
```

## Debugging Checklist

When diagnosing issues with tensor computations:

1. **Check shapes**: `std::cout << tensor.debug_info()` or `tensor.assert_shape(...)`
2. **Check for NaN/Inf**: `tensor.has_nan()`, `tensor.has_inf()`, `tensor.nan_guard()`
3. **Check device**: `tensor.device()` -- make sure operands are on the same device
4. **Check dtype**: `tensor.dtype_name()` -- unexpected promotions can cause issues
5. **Check contiguity**: `tensor.is_contiguous()` -- some operations need contiguous input
6. **Enable tracing**: `trace::enable()` to see the full operation sequence
7. **Profile bottlenecks**: `profile::enable()` to find slow operations

## Lazy Tensor Debugging

For lazy tensors, most inspection triggers materialization:

```cpp
auto lazy = /* lazy computation */;

// These do NOT trigger materialization:
lazy.is_lazy();     // true
lazy.shape();       // Available from graph metadata
lazy.dtype();       // Available from graph metadata
lazy.device();      // Available from graph metadata

// These DO trigger materialization:
lazy.data();        // Forces computation
std::cout << lazy;  // Forces computation for printing
lazy.debug_info();  // Forces computation for full diagnostics
```

For complete API details, see [API Reference: Debug & Profiling](../api/debug).
