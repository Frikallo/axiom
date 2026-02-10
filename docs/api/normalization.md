# Normalization

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

## ops::layer_norm

```cpp
Tensor ops::layer_norm(const Tensor &input, const Tensor &weight,
                       const Tensor &bias, int axis = -1, float eps = 1e-5f);
```

Layer normalization: `(x - mean) / sqrt(var + eps) * weight + bias`.

**Parameters:**
- `input` (*Tensor*) -- Input tensor.
- `weight` (*Tensor*) -- Scale parameter (same size as normalized axis).
- `bias` (*Tensor*) -- Shift parameter (same size as normalized axis).
- `axis` (*int*) -- Axis to normalize along. Default: `-1`.
- `eps` (*float*) -- Small constant for numerical stability. Default: `1e-5`.

**Example:**
```cpp
auto x = Tensor::randn({2, 4, 8});
auto weight = Tensor::ones({8});
auto bias = Tensor::zeros({8});
auto normed = ops::layer_norm(x, weight, bias);  // Normalize last dim
```

---

## ops::rms_norm

```cpp
Tensor ops::rms_norm(const Tensor &input, const Tensor &weight,
                     int axis = -1, float eps = 1e-5f);
```

RMS normalization: `x / sqrt(mean(x^2) + eps) * weight`. Does not subtract the mean (unlike layer_norm).

**Parameters:**
- `input` (*Tensor*) -- Input tensor.
- `weight` (*Tensor*) -- Scale parameter.
- `axis` (*int*) -- Axis to normalize along. Default: `-1`.
- `eps` (*float*) -- Small constant for numerical stability. Default: `1e-5`.

---

## ops::dropout

```cpp
std::pair<Tensor, Tensor> ops::dropout(const Tensor &input, float p = 0.5f,
                                       bool training = true);
```

Dropout regularization. Randomly zeroes elements with probability `p` during training, and scales remaining elements by `1 / (1 - p)`.

**Parameters:**
- `input` (*Tensor*) -- Input tensor.
- `p` (*float*) -- Probability of an element being zeroed. Default: `0.5`.
- `training` (*bool*) -- If false, returns input unchanged. Default: `true`.

**Returns:** `std::pair<Tensor, Tensor>` containing `(output, mask)` where `mask` is a Bool tensor of kept values.

**Example:**
```cpp
auto [output, mask] = ops::dropout(x, 0.1f, true);
```

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Activations](activations)
