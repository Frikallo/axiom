# FFT

The `axiom::fft` namespace provides Fast Fourier Transform operations for spectral analysis and signal processing.

## 1D FFT

### Complex-to-Complex

```cpp
using namespace axiom;

auto signal = Tensor::randn({1024}).to_complex();

// Forward FFT
auto spectrum = fft::fft(signal);          // Shape: {1024}, Complex64

// Inverse FFT
auto recovered = fft::ifft(spectrum);      // Shape: {1024}, Complex64
```

### Real-to-Complex

For real-valued input, `rfft` exploits Hermitian symmetry and returns only the non-redundant half:

```cpp
auto signal = Tensor::randn({1024});

auto spectrum = fft::rfft(signal);         // Shape: {513}, Complex64
auto recovered = fft::irfft(spectrum);     // Shape: {1024}, Float32
```

The output length of `rfft` is `n/2 + 1`. Pass `n` to `irfft` to specify the output length when the original length is ambiguous.

## 2D FFT

For images and 2D signals:

```cpp
auto image = Tensor::randn({256, 256});

// 2D FFT
auto freq = fft::fft2(image.to_complex());   // Shape: {256, 256}
auto recovered = fft::ifft2(freq);

// Real-input 2D FFT
auto freq_r = fft::rfft2(image);              // Shape: {256, 129}
auto recovered_r = fft::irfft2(freq_r);
```

## N-dimensional FFT

```cpp
auto data = Tensor::randn({16, 32, 64}).to_complex();

// Transform all dimensions
auto freq = fft::fftn(data);

// Transform specific dimensions
auto freq_partial = fft::fftn(data, {1, 2});  // Only dims 1 and 2

auto recovered = fft::ifftn(freq);
```

## Normalization Modes

All FFT functions accept a normalization parameter:

| Mode | Forward | Inverse |
|------|---------|---------|
| `"backward"` (default) | No scaling | Divide by N |
| `"forward"` | Divide by N | No scaling |
| `"ortho"` | Divide by sqrt(N) | Divide by sqrt(N) |

```cpp
auto signal = Tensor::randn({256}).to_complex();

auto s1 = fft::fft(signal, -1, "backward");  // Default
auto s2 = fft::fft(signal, -1, "ortho");     // Energy-preserving
auto s3 = fft::fft(signal, -1, "forward");   // Pre-normalized
```

## Frequency Utilities

### Frequency Bins

```cpp
// Frequencies for a 256-point FFT with sample spacing 1/256
auto freqs = fft::fftfreq(256, 1.0 / 256.0);     // Shape: {256}
auto rfreqs = fft::rfftfreq(256, 1.0 / 256.0);    // Shape: {129}
```

### Shifting

`fftshift` moves zero-frequency to the center (useful for visualization):

```cpp
auto spectrum = fft::fft(signal);
auto centered = fft::fftshift(spectrum);    // Zero-freq at center
auto unshifted = fft::ifftshift(centered);  // Undo the shift
```

## Window Functions

Window functions reduce spectral leakage when analyzing finite-length signals:

```cpp
size_t n = 1024;

auto hann    = fft::hann_window(n);        // Hann (cosine-squared)
auto hamming = fft::hamming_window(n);     // Hamming
auto black   = fft::blackman_window(n);    // Blackman
auto bart    = fft::bartlett_window(n);    // Bartlett (triangular)
auto kaiser  = fft::kaiser_window(n, 5.0); // Kaiser with beta=5

// Apply window before FFT
auto signal = Tensor::randn({1024});
auto windowed = signal * hann;
auto spectrum = fft::rfft(windowed);
```

## Common Patterns

### Power Spectrum

```cpp
auto signal = Tensor::randn({4096});
auto window = fft::hann_window(4096);

auto spectrum = fft::rfft(signal * window);
// Power = |FFT|^2
auto power = ops::square(spectrum.real()) + ops::square(spectrum.imag());
```

### 2D Frequency Filtering

```cpp
auto image = Tensor::randn({256, 256});

// Forward transform
auto freq = fft::fft2(image.to_complex());

// Apply frequency-domain filter (e.g., low-pass)
// ... modify freq ...

// Inverse transform
auto filtered = fft::ifft2(freq).real();
```

For complete function signatures, see [API Reference: FFT](../api/fft).
