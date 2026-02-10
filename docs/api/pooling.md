# Pooling Operations

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

Standard and adaptive pooling operations for 1D, 2D, and 3D inputs.

## 1D Pooling

Input shape: `(N, C, L)` or `(C, L)`. Output length: `L_out = (L + 2*padding - kernel_size) / stride + 1`.

### ops::max_pool1d

```cpp
Tensor ops::max_pool1d(const Tensor &input, int kernel_size,
                       int stride = 1, int padding = 0);
```

1D max pooling.

---

### ops::avg_pool1d

```cpp
Tensor ops::avg_pool1d(const Tensor &input, int kernel_size,
                       int stride = 1, int padding = 0,
                       bool count_include_pad = true);
```

1D average pooling.

**Parameters:**
- `count_include_pad` (*bool*) -- Include zero-padding in average calculation. Default: `true`.

---

## 2D Pooling

Input shape: `(N, C, H, W)` or `(C, H, W)`. Output: `H_out = (H + 2*padding[0] - kernel_size[0]) / stride[0] + 1`.

### ops::max_pool2d

```cpp
Tensor ops::max_pool2d(const Tensor &input,
                       const std::vector<int> &kernel_size,
                       const std::vector<int> &stride = {},
                       const std::vector<int> &padding = {});
```

2D max pooling. If `stride` is empty, it defaults to `kernel_size`.

**Example:**
```cpp
auto x = Tensor::randn({1, 3, 32, 32});
auto pooled = ops::max_pool2d(x, {2, 2});  // shape (1, 3, 16, 16)
```

---

### ops::avg_pool2d

```cpp
Tensor ops::avg_pool2d(const Tensor &input,
                       const std::vector<int> &kernel_size,
                       const std::vector<int> &stride = {},
                       const std::vector<int> &padding = {},
                       bool count_include_pad = true);
```

2D average pooling.

---

## 3D Pooling

Input shape: `(N, C, D, H, W)` or `(C, D, H, W)`.

### ops::max_pool3d

```cpp
Tensor ops::max_pool3d(const Tensor &input,
                       const std::vector<int> &kernel_size,
                       const std::vector<int> &stride = {},
                       const std::vector<int> &padding = {});
```

---

### ops::avg_pool3d

```cpp
Tensor ops::avg_pool3d(const Tensor &input,
                       const std::vector<int> &kernel_size,
                       const std::vector<int> &stride = {},
                       const std::vector<int> &padding = {},
                       bool count_include_pad = true);
```

---

## Adaptive Pooling

Adaptive pooling produces the specified output size regardless of input size.

### ops::adaptive_max_pool2d

```cpp
Tensor ops::adaptive_max_pool2d(const Tensor &input,
                                const std::vector<int> &output_size);
```

**Example:**
```cpp
auto x = Tensor::randn({1, 3, 32, 32});
auto pooled = ops::adaptive_max_pool2d(x, {1, 1});  // Global max pool
```

---

### ops::adaptive_avg_pool2d

```cpp
Tensor ops::adaptive_avg_pool2d(const Tensor &input,
                                const std::vector<int> &output_size);
```

---

### ops::adaptive_max_pool1d

```cpp
Tensor ops::adaptive_max_pool1d(const Tensor &input, int output_size);
```

---

### ops::adaptive_avg_pool1d

```cpp
Tensor ops::adaptive_avg_pool1d(const Tensor &input, int output_size);
```

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Activations](activations), [Normalization](normalization)
