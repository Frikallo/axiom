# System

System-level queries in the `axiom::system` namespace.

## system::is_metal_available

```cpp
bool system::is_metal_available();
```

Returns `true` if a Metal-capable GPU is available. Always `false` on non-Apple platforms.

---

## system::should_run_gpu_tests

```cpp
bool system::should_run_gpu_tests();
```

Returns `false` if Metal is not available or the `AXIOM_SKIP_GPU_TESTS` environment variable is set to `"1"`.

---

## system::device_to_string

```cpp
std::string system::device_to_string(Device device);
```

Returns `"CPU"` or `"GPU"`.

---

## Shape Utilities

### ops::broadcast_shapes

```cpp
Shape ops::broadcast_shapes(const std::vector<Shape> &shapes);
```

Compute the shape that all input shapes can be broadcast to.

---

### ops::broadcast_tensors

```cpp
std::vector<Tensor> ops::broadcast_tensors(const std::vector<Tensor> &tensors);
```

Broadcast multiple tensors to a common shape using zero-copy `expand()`.

---

### ops::meshgrid

```cpp
std::vector<Tensor> ops::meshgrid(const std::vector<Tensor> &tensors,
                                   const std::string &indexing = "xy");
```

Create coordinate matrices from coordinate vectors.

**Parameters:**
- `indexing` (*string*) -- `"xy"` (Cartesian, default) or `"ij"` (matrix indexing).

---

### ops::pad

```cpp
Tensor ops::pad(const Tensor &input,
                const std::vector<std::pair<size_t, size_t>> &pad_width,
                const std::string &mode = "constant", double value = 0.0);
```

Pad a tensor.

**Parameters:**
- `pad_width` -- Vector of `(before, after)` pairs for each dimension.
- `mode` (*string*) -- `"constant"` (default), `"reflect"`, `"replicate"`, `"circular"`.
- `value` (*double*) -- Fill value for constant mode. Default: `0.0`.

---

### ops::atleast_1d / 2d / 3d

```cpp
Tensor ops::atleast_1d(const Tensor &tensor);
Tensor ops::atleast_2d(const Tensor &tensor);
Tensor ops::atleast_3d(const Tensor &tensor);

std::vector<Tensor> ops::atleast_1d(const std::vector<Tensor> &tensors);
std::vector<Tensor> ops::atleast_2d(const std::vector<Tensor> &tensors);
std::vector<Tensor> ops::atleast_3d(const std::vector<Tensor> &tensors);
```

Ensure tensors have at least N dimensions by prepending size-1 dimensions.

**See Also:** [Devices & Storage](devices-and-storage), [Parallelism](parallel)
