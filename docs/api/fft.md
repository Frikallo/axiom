# FFT

*For a tutorial introduction, see [User Guide: FFT](../user-guide/fft).*

Fast Fourier Transform operations in the `axiom::fft` namespace.

## 1D FFT

### fft::fft

```cpp
Tensor fft::fft(const Tensor &input, int64_t n = -1, int axis = -1,
                const std::string &norm = "backward");
```

1-dimensional discrete Fourier Transform.

**Parameters:**
- `input` -- Input tensor (real or complex).
- `n` (*int64_t*) -- Length of the transformed axis. Default: input size along `axis`.
- `axis` (*int*) -- Axis to compute FFT over. Default: `-1`.
- `norm` (*string*) -- Normalization mode: `"backward"` (default), `"forward"`, `"ortho"`.

**Returns:** Complex tensor.

---

### fft::ifft

```cpp
Tensor fft::ifft(const Tensor &input, int64_t n = -1, int axis = -1,
                 const std::string &norm = "backward");
```

1-dimensional inverse FFT.

---

### fft::rfft

```cpp
Tensor fft::rfft(const Tensor &input, int64_t n = -1, int axis = -1,
                 const std::string &norm = "backward");
```

1D FFT of real input. Output has length `n/2 + 1` (Hermitian symmetry).

---

### fft::irfft

```cpp
Tensor fft::irfft(const Tensor &input, int64_t n = -1, int axis = -1,
                  const std::string &norm = "backward");
```

Inverse FFT for real output.

---

## 2D FFT

### fft::fft2

```cpp
Tensor fft::fft2(const Tensor &input, const std::vector<int64_t> &s = {},
                 const std::vector<int> &axes = {-2, -1},
                 const std::string &norm = "backward");
```

2-dimensional FFT.

**Parameters:**
- `s` -- Shape of the output along transformed axes. Default: input sizes.
- `axes` -- Axes to compute FFT over. Default: `{-2, -1}`.

---

### fft::ifft2

```cpp
Tensor fft::ifft2(const Tensor &input, const std::vector<int64_t> &s = {},
                  const std::vector<int> &axes = {-2, -1},
                  const std::string &norm = "backward");
```

---

### fft::rfft2

```cpp
Tensor fft::rfft2(const Tensor &input, const std::vector<int64_t> &s = {},
                  const std::vector<int> &axes = {-2, -1},
                  const std::string &norm = "backward");
```

---

### fft::irfft2

```cpp
Tensor fft::irfft2(const Tensor &input, const std::vector<int64_t> &s = {},
                   const std::vector<int> &axes = {-2, -1},
                   const std::string &norm = "backward");
```

---

## N-dimensional FFT

### fft::fftn

```cpp
Tensor fft::fftn(const Tensor &input, const std::vector<int64_t> &s = {},
                 const std::vector<int> &axes = {},
                 const std::string &norm = "backward");
```

N-dimensional FFT. If `axes` is empty, transforms over all axes.

---

### fft::ifftn

```cpp
Tensor fft::ifftn(const Tensor &input, const std::vector<int64_t> &s = {},
                  const std::vector<int> &axes = {},
                  const std::string &norm = "backward");
```

---

## Utility Functions

### fft::fftshift

```cpp
Tensor fft::fftshift(const Tensor &input, const std::vector<int> &axes = {});
```

Shift zero-frequency component to the center of the spectrum.

---

### fft::ifftshift

```cpp
Tensor fft::ifftshift(const Tensor &input, const std::vector<int> &axes = {});
```

Inverse of `fftshift`.

---

### fft::fftfreq

```cpp
Tensor fft::fftfreq(int64_t n, double d = 1.0, DType dtype = DType::Float64,
                    Device device = Device::CPU);
```

DFT sample frequencies. Returns 1D tensor of length `n`.

---

### fft::rfftfreq

```cpp
Tensor fft::rfftfreq(int64_t n, double d = 1.0, DType dtype = DType::Float64,
                     Device device = Device::CPU);
```

Sample frequencies for `rfft`. Returns 1D tensor of length `n/2 + 1`.

---

## Window Functions

### fft::hann_window

```cpp
Tensor fft::hann_window(int64_t M, bool periodic = true,
                        DType dtype = DType::Float32, Device device = Device::CPU);
```

Hann (Hanning) window: `w[n] = 0.5 - 0.5 * cos(2*pi*n / (M-1))`.

---

### fft::hamming_window

```cpp
Tensor fft::hamming_window(int64_t M, bool periodic = true,
                           DType dtype = DType::Float32, Device device = Device::CPU);
```

Hamming window: `w[n] = 0.54 - 0.46 * cos(2*pi*n / (M-1))`.

---

### fft::blackman_window

```cpp
Tensor fft::blackman_window(int64_t M, bool periodic = true,
                            DType dtype = DType::Float32, Device device = Device::CPU);
```

---

### fft::bartlett_window

```cpp
Tensor fft::bartlett_window(int64_t M, bool periodic = true,
                            DType dtype = DType::Float32, Device device = Device::CPU);
```

Bartlett (triangular) window.

---

### fft::kaiser_window

```cpp
Tensor fft::kaiser_window(int64_t M, double beta = 12.0, bool periodic = true,
                          DType dtype = DType::Float32, Device device = Device::CPU);
```

**Parameters:**
- `beta` (*double*) -- Shape parameter. Default: `12.0`.

---

## Normalization Modes

| Mode | Forward | Backward |
|------|---------|----------|
| `"backward"` | No normalization | Divide by N |
| `"forward"` | Divide by N | No normalization |
| `"ortho"` | Divide by sqrt(N) | Divide by sqrt(N) |

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Unary Math](unary-math)
