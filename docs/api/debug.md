# Debug & Profiling

APIs for tracing, profiling, and CPU diagnostics.

## trace Namespace

Operation-level tracing for debugging computation sequences.

### trace::enable / trace::disable

```cpp
void trace::enable();
void trace::disable();
bool trace::is_enabled();
```

Enable/disable tracing globally.

---

### trace::clear

```cpp
void trace::clear();
```

Clear all recorded trace events.

---

### trace::dump

```cpp
std::string trace::dump();
```

Returns a formatted string of all recorded trace events.

---

### trace::ScopedTrace

```cpp
class trace::ScopedTrace {
public:
    ScopedTrace(const std::string &op_name, const std::string &desc = "",
                size_t memory_bytes = 0, bool materialized = false);
    ~ScopedTrace();  // Records duration on destruction
};
```

RAII-style trace recording. Records timing automatically.

```cpp
{
    trace::ScopedTrace t("matmul", "A @ B", A.nbytes() + B.nbytes());
    auto C = A.matmul(B);
}  // Duration recorded here
```

---

### TraceEvent

```cpp
struct trace::TraceEvent {
    std::string op_name;
    std::string description;
    std::chrono::steady_clock::time_point timestamp;
    std::chrono::nanoseconds duration;
    size_t memory_bytes;
    bool materialized;
};
```

---

## profile Namespace

Lightweight per-operation profiling.

### profile::enable / profile::disable

```cpp
void profile::enable();
void profile::disable();
```

---

### profile::last_op

```cpp
const profile::OpProfile &profile::last_op();
```

Returns profiling data for the most recently executed operation.

---

### OpProfile

```cpp
struct profile::OpProfile {
    std::string name;
    std::chrono::nanoseconds duration;
    size_t input_bytes;
    size_t output_bytes;
    std::string shape_info;
};
```

---

## cpu_info Namespace

CPU and SIMD architecture diagnostics.

### cpu_info::print_simd_info

```cpp
void cpu_info::print_simd_info();
```

Print SIMD architecture info to stdout.

---

### cpu_info::simd_arch_name

```cpp
const char *cpu_info::simd_arch_name();
```

Returns SIMD architecture name (e.g., `"neon64"`, `"avx2"`, `"sse4.2"`).

---

### cpu_info::simd_info_string

```cpp
std::string cpu_info::simd_info_string();
```

Returns SIMD info as a compact string.

---

## Tensor Debug Methods

### Tensor::debug_info

```cpp
std::string Tensor::debug_info() const;
```

Returns detailed diagnostic string including shape, dtype, device, strides, contiguity, and memory info.

---

### Tensor::has_nan / Tensor::has_inf

```cpp
bool Tensor::has_nan() const;
bool Tensor::has_inf() const;
bool Tensor::is_finite() const;
```

Check for special floating-point values. Works on Float16, Float32, Float64, Complex64, Complex128.

---

### Tensor::nan_guard / Tensor::assert_finite

```cpp
Tensor &Tensor::nan_guard();
Tensor &Tensor::assert_finite();
```

Throw `ValueError` if NaN or non-finite values are detected. Returns `*this` for chaining.

```cpp
auto result = compute().nan_guard();  // Throws if NaN present
```

---

### Tensor::assert_shape

```cpp
Tensor &Tensor::assert_shape(const std::string &pattern);
Tensor &Tensor::assert_shape(const Shape &expected);
```

Assert tensor matches expected shape. Throws `ShapeError` on mismatch.

```cpp
tensor.assert_shape({3, 4});            // Exact shape
tensor.assert_shape("batch height width");  // Named pattern
```

**See Also:** [Errors](errors), [System](system)
