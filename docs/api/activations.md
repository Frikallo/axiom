# Activation Functions

*For a tutorial introduction, see [User Guide: Operations](../user-guide/operations).*

Neural network activation functions. All available in both functional (`ops::`) and fluent (`tensor.`) forms.

## ops::relu / Tensor::relu

```cpp
Tensor ops::relu(const Tensor &input);
Tensor Tensor::relu() const;
```

Rectified Linear Unit: `max(0, x)`.

---

## ops::leaky_relu / Tensor::leaky_relu

```cpp
Tensor ops::leaky_relu(const Tensor &input, float negative_slope = 0.01f);
Tensor Tensor::leaky_relu(float negative_slope = 0.01f) const;
```

Leaky ReLU: `x` if `x > 0`, else `negative_slope * x`.

**Parameters:**
- `negative_slope` (*float*) -- Slope for negative values. Default: `0.01`.

---

## ops::sigmoid / Tensor::sigmoid

```cpp
Tensor ops::sigmoid(const Tensor &input);
Tensor Tensor::sigmoid() const;
```

Logistic sigmoid: `1 / (1 + exp(-x))`.

---

## ops::tanh / Tensor::tanh

```cpp
Tensor ops::tanh(const Tensor &input);
Tensor Tensor::tanh() const;
```

Hyperbolic tangent.

---

## ops::silu / Tensor::silu

```cpp
Tensor ops::silu(const Tensor &input);
Tensor Tensor::silu() const;
```

SiLU (Sigmoid Linear Unit) / Swish: `x * sigmoid(x)`.

---

## ops::gelu / Tensor::gelu

```cpp
Tensor ops::gelu(const Tensor &input);
Tensor Tensor::gelu() const;
```

Gaussian Error Linear Unit.

---

## ops::softmax / Tensor::softmax

```cpp
Tensor ops::softmax(const Tensor &input, int axis = -1);
Tensor Tensor::softmax(int axis = -1) const;
```

Softmax function along an axis: `exp(x) / sum(exp(x))`.

**Parameters:**
- `axis` (*int*) -- Axis along which to compute softmax. Default: `-1` (last axis).

**Example:**
```cpp
auto logits = Tensor::randn({4, 10});
auto probs = logits.softmax(-1);  // Probabilities along last axis
```

---

## ops::log_softmax / Tensor::log_softmax

```cpp
Tensor ops::log_softmax(const Tensor &input, int axis = -1);
Tensor Tensor::log_softmax(int axis = -1) const;
```

Log-softmax: `log(softmax(x))`. Numerically more stable than computing `log(softmax(x))` separately.

**Parameters:**
- `axis` (*int*) -- Axis along which to compute. Default: `-1`.

---

## Fluent Chaining

Activations compose naturally with other operations:

```cpp
auto output = x.relu();
auto output = (x * 2.0f + 1.0f).relu().sigmoid();
auto output = x.gelu().softmax(-1);
```

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Unary Math](unary-math), [Normalization](normalization)
