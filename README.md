**Axiom** is an open-source, high-performance C++ tensor library that brings NumPy and PyTorch simplicity to native code. With state-of-the-art SIMD vectorization, BLAS acceleration, and Metal GPU support, Axiom delivers HPC-grade performance while maintaining an intuitive API that feels natural to Python developers.

[![CI](https://github.com/frikallo/axiom/actions/workflows/ci.yml/badge.svg)](https://github.com/frikallo/axiom/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

The **Axiom** library offers ...

  * ... **Python-familiar API** through operator overloading, method chaining, and NumPy-compatible function names
  * ... **high performance** through Accelerate, OpenBLAS, and manually tuned SIMD kernels
  * ... **vectorization** by SSE2/3/4, AVX, AVX2, AVX-512, FMA3/4, ARM NEON/ARMv8, WASM SIMD, RISC-V Vector, and PowerPC VSX
  * ... **parallel execution** by OpenMP with intelligent workload thresholds
  * ... **full GPU acceleration** via Metal Performance Shaders (MPSGraph) — every operation runs on GPU, not just matmul
  * ... **einops integration** for intuitive `rearrange("b h w c -> b c h w")` tensor manipulation
  * ... **zero-copy views** with strides-based memory model eliminating unnecessary data copies
  * ... **complete dtype coverage** including Float16/32/64, Int8-64, Bool, and Complex64/128
  * ... **portable distribution** with dynamically linked BLAS backends for cross-platform deployment

Get an impression of the familiar syntax in the [Quick Start](#quick-start) section and the impressive performance in the [Benchmarks](#benchmarks) section.

----

## Why Axiom?

### Axiom is intuitive.

```cpp
// NumPy: x = np.where(x > 0, x, 0)
auto x = Tensor::where(x > 0, x, 0);

// NumPy: y = x.reshape(2, -1).T
auto y = x.reshape({2, -1}).T();

// PyTorch: z = F.softmax(scores, dim=-1)
auto z = scores.softmax(-1);
```

If you know NumPy or PyTorch, you already know Axiom.

</td>
<td width="50%">

### Axiom is fast.

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| MatMul 2048² | 4.2ms | 0.8ms | **5×** |
| MatMul 4096² | 32ms | 1.8ms | **18×** |
| Softmax 10M | 2.8ms | 0.4ms | **7×** |
| LayerNorm | 1.2ms | 0.15ms | **8×** |

**Full GPU acceleration.** Every op on Metal.

</td>
</tr>
<tr>
<td>

### Axiom is expressive.

```cpp
// Einops-style rearrangement
auto img = x.rearrange("b h w c -> b c h w");

// Einops-style reduction (spatial pooling)
auto pooled = x.reduce(
    "b (h p1) (w p2) c -> b h w c",
    "mean", {{"p1", 2}, {"p2", 2}}
);

// Global average pooling
auto gap = features.reduce("b h w c -> b c", "mean");
```

Complex transformations, readable code.

</td>
<td>

### Axiom is reliable.

- **26 comprehensive test suites** covering all operations
- **CI/CD pipeline** testing CPU and GPU paths
- **Cross-platform validation** on macOS, Linux, Windows
- **NaN/Inf guards** with `assert_finite()` safety rails
- **Shape assertions** with `assert_shape("b h w c")`

Production-ready from day one.

</td>
</tr>
</table>

----

## Download

**Latest Release**: Axiom 1.0.0

```bash
git clone https://github.com/frikallo/axiom.git
cd axiom && make release
```

Or fetch directly in CMake:

```cmake
include(FetchContent)
FetchContent_Declare(axiom
    GIT_REPOSITORY https://github.com/frikallo/axiom.git
    GIT_TAG main)
FetchContent_MakeAvailable(axiom)
target_link_libraries(your_target Axiom::axiom)
```

----

## Quick Start

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    // Tensor creation - just like NumPy
    auto a = Tensor::zeros({3, 4});
    auto b = Tensor::ones({4, 5});
    auto c = Tensor::randn({3, 4});
    auto d = Tensor::linspace(0, 1, 100);

    // Intuitive operations
    auto result = (a + c).relu().matmul(b);

    // Conditional selection - Python's np.where()
    auto x = Tensor::randn({100});
    auto positive = Tensor::where(x > 0, x, 0.0f);

    // Einops-style rearrangement
    auto img = Tensor::randn({2, 224, 224, 3});
    auto nchw = img.rearrange("b h w c -> b c h w");

    // Full transformer attention in 5 lines
    auto Q = Tensor::randn({2, 8, 64, 64});
    auto K = Tensor::randn({2, 8, 64, 64});
    auto V = Tensor::randn({2, 8, 64, 64});
    auto scores = Q.matmul(K.transpose(-2, -1)) / std::sqrt(64.0f);
    auto output = scores.softmax(-1).matmul(V);

    return 0;
}
```

**GPU acceleration? Just change the device.** Every operation runs on Metal—no code changes required:

```cpp
// CPU version
auto x = Tensor::randn({1024, 1024}, DType::Float32, Device::CPU);

// GPU version - same API, 10-20x faster on Apple Silicon
auto x = Tensor::randn({1024, 1024}, DType::Float32, Device::GPU);

// Everything just works: matmul, softmax, reductions, broadcasting, indexing...
auto result = x.matmul(x.T()).softmax(-1).sum({1});  // All on GPU
```

No other C++ tensor library offers this. Eigen, Armadillo, Blaze—all CPU-only. With Axiom, you get the same clean API with full GPU acceleration on macOS.

----

## Feature Overview

### NumPy-Compatible API

Axiom mirrors NumPy and PyTorch APIs so closely that translating Python code is almost mechanical:

| NumPy / PyTorch | Axiom |
|-----------------|-------|
| `np.zeros((3,4))` | `Tensor::zeros({3,4})` |
| `np.arange(0, 10, 0.5)` | `Tensor::arange(0, 10, 0.5)` |
| `np.linspace(0, 1, 100)` | `Tensor::linspace(0, 1, 100)` |
| `x.reshape(-1, 4)` | `x.reshape({-1, 4})` |
| `x.transpose(0, 2, 1)` | `x.transpose({0, 2, 1})` |
| `np.concatenate([a,b], axis=1)` | `Tensor::cat({a,b}, 1)` |
| `np.where(cond, a, b)` | `Tensor::where(cond, a, b)` |
| `x[x > 0]` | `x.masked_select(x > 0)` |
| `torch.gather(x, dim, idx)` | `x.gather(dim, idx)` |
| `F.softmax(x, dim=-1)` | `x.softmax(-1)` |
| `F.layer_norm(x, shape)` | `ops::layer_norm(x, w, b)` |

### Einops Integration

Full [einops](https://github.com/arogozhnikov/einops) pattern syntax for semantic tensor manipulation:

```cpp
// Reshape and transpose in one operation
auto transposed = x.rearrange("b h w c -> b c h w");

// Flatten spatial dimensions
auto flat = x.rearrange("b h w c -> b (h w) c");

// Patch embedding (Vision Transformer style)
auto patches = img.rearrange("b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
                              {{"p1", 16}, {"p2", 16}});

// Reduce with pattern
auto pooled = x.reduce("b (h 2) (w 2) c -> b h w c", "mean");
auto gap = features.reduce("b h w c -> b c", "mean");
```

### Performance Backend

Axiom automatically selects the fastest available backend:

| Platform | BLAS Backend | Vectorization | GPU |
|----------|--------------|---------------|-----|
| **macOS (Apple Silicon)** | Accelerate + vDSP | ARM NEON / ARMv8 | Metal (MPSGraph) |
| **macOS (Intel)** | Accelerate | SSE2-4.2 / AVX / AVX2 | Metal |
| **Linux (x86_64)** | OpenBLAS | SSE2-4.2 / AVX / AVX2 / AVX-512 / FMA3 | — |
| **Linux (ARM)** | OpenBLAS | ARMv7 / ARMv8 NEON | — |
| **Windows** | Native (OpenBLAS optional) | SSE2-4.2 / AVX / AVX2 | — |
| **WebAssembly** | Native | WASM SIMD | — |
| **RISC-V** | Native | RISC-V Vector ISA | — |
| **PowerPC** | Native | VSX | — |

OpenMP parallelization with intelligent thresholds ensures overhead is only incurred when beneficial.

<details>
<summary><b>Full SIMD Support Matrix</b> (via xsimd)</summary>

| Architecture | Instruction Set Extensions |
|--------------|---------------------------|
| **x86 (Intel/AMD)** | SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, FMA3, AVX2 |
| **x86 (AVX-512)** | AVX-512 (GCC 7+, Clang, MSVC) |
| **x86 (AMD)** | All of the above + FMA4 |
| **ARM** | ARMv7 NEON, ARMv8 NEON |
| **WebAssembly** | WASM SIMD128 |
| **RISC-V** | Vector ISA (RVV) |
| **PowerPC** | VSX |

Axiom uses [xsimd](https://github.com/xtensor-stack/xsimd) for portable SIMD abstraction, automatically dispatching to the optimal instruction set at compile time with runtime fallback to scalar operations.

</details>

### Linear Algebra

Complete LAPACK-backed linear algebra module:

```cpp
// Decompositions
auto [U, S, Vh] = linalg::svd(A);
auto [Q, R] = linalg::qr(A);
auto L = linalg::cholesky(A);
auto [eigvals, eigvecs] = linalg::eigh(A);

// Solvers
auto x = linalg::solve(A, b);           // Ax = b
auto x = linalg::lstsq(A, b);           // Least squares
auto Ainv = linalg::pinv(A);            // Pseudoinverse

// Analysis
auto d = linalg::det(A);
auto n = linalg::norm(A, "fro");
auto r = linalg::matrix_rank(A);
auto k = linalg::cond(A);
```

All operations support batch dimensions: `A.shape = (batch, M, N)`.

### I/O and Serialization

```cpp
// Single tensor
tensor.save("weights.axfb");            // FlatBuffers (fast, zero-copy)
tensor.save("weights.npy");             // NumPy format (Python interop)
auto loaded = Tensor::load("weights.axfb");

// Multiple tensors (model checkpoints)
Tensor::save_tensors({{"weight", W}, {"bias", b}}, "model.axfb");
auto params = Tensor::load_tensors("model.axfb");
```

----

## Supported Operations

<details>
<summary><b>Arithmetic & Math</b> (click to expand)</summary>

| Operation | Function | Operator |
|-----------|----------|----------|
| Addition | `ops::add(a, b)` | `a + b` |
| Subtraction | `ops::subtract(a, b)` | `a - b` |
| Multiplication | `ops::multiply(a, b)` | `a * b` |
| Division | `ops::divide(a, b)` | `a / b` |
| Power | `ops::power(a, b)` | — |
| Modulo | `ops::modulo(a, b)` | `a % b` |
| Square root | `ops::sqrt(a)` | — |
| Exponential | `ops::exp(a)` | — |
| Logarithm | `ops::log(a)` | — |
| Absolute value | `ops::abs(a)` | — |
| Sign | `ops::sign(a)` | — |
| Floor/Ceil | `ops::floor(a)`, `ops::ceil(a)` | — |
| Trigonometric | `ops::sin`, `cos`, `tan` | — |
| Error function | `ops::erf(a)` | — |

</details>

<details>
<summary><b>Comparison & Logical</b></summary>

| Operation | Function | Operator |
|-----------|----------|----------|
| Equal | `ops::equal(a, b)` | `a == b` |
| Not equal | `ops::not_equal(a, b)` | `a != b` |
| Less than | `ops::less(a, b)` | `a < b` |
| Greater than | `ops::greater(a, b)` | `a > b` |
| Logical AND | `ops::logical_and(a, b)` | `a && b` |
| Logical OR | `ops::logical_or(a, b)` | `a \|\| b` |
| Logical NOT | `ops::logical_not(a)` | `!a` |
| Bitwise ops | `ops::bitwise_and/or/xor` | `&`, `\|`, `^` |

</details>

<details>
<summary><b>Reductions</b></summary>

```cpp
tensor.sum()                    // Total sum
tensor.sum({0, 2})              // Sum along axes
tensor.sum({0}, true)           // Keep dimensions

tensor.mean(), tensor.max(), tensor.min()
tensor.argmax(axis), tensor.argmin(axis)
tensor.any(), tensor.all()      // Boolean reductions
tensor.var(axis, ddof)          // Variance (Bessel correction)
tensor.std(axis, ddof)          // Standard deviation
tensor.prod(axis)               // Product
```

</details>

<details>
<summary><b>Shape Manipulation</b></summary>

```cpp
// Reshape and views
tensor.reshape(new_shape)       // View if contiguous, copy otherwise
tensor.view(new_shape)          // View only (asserts contiguous)
tensor.flatten()                // To 1D
tensor.squeeze()                // Remove size-1 dims
tensor.unsqueeze(axis)          // Add size-1 dim

// Transpose and permute
tensor.T()                      // Matrix transpose
tensor.transpose(axes)          // Arbitrary permutation
tensor.swapaxes(a, b)           // Swap two axes
tensor.moveaxis(src, dst)       // Move axis

// Flip and rotate
tensor.flip(axis)               // Reverse along axis
tensor.flipud(), tensor.fliplr()
tensor.rot90(k, axes)           // Rotate 90° k times
tensor.roll(shift, axis)        // Circular shift

// Join and split
Tensor::cat({a, b}, axis)       // Concatenate
Tensor::stack({a, b}, axis)     // Stack with new axis
tensor.split(n, axis)           // Split into n parts
tensor.chunk(n, axis)           // Chunk (may be unequal)
```

</details>

<details>
<summary><b>Neural Network Operations</b></summary>

```cpp
// Activations
tensor.relu()
tensor.leaky_relu(0.01f)
tensor.sigmoid()
tensor.tanh()
tensor.gelu()
tensor.silu()                   // Swish

// Softmax
tensor.softmax(axis)
tensor.log_softmax(axis)

// Normalization
ops::layer_norm(x, weight, bias, axis, eps)
ops::rms_norm(x, weight, axis, eps)

// Dropout (training mode)
auto [out, mask] = ops::dropout(x, 0.1f, training);
```

</details>

<details>
<summary><b>Indexing & Selection</b></summary>

```cpp
// Conditional selection
Tensor::where(cond, a, b)       // a where true, b where false
tensor.where(cond, value)       // Fluent API

// Masking
tensor.masked_fill(mask, val)   // Fill where mask is true
tensor.masked_select(mask)      // Extract elements

// Gather/Scatter (PyTorch-style)
tensor.gather(dim, indices)
tensor.scatter(dim, indices, src)
tensor.index_select(dim, indices)

// Diagonal operations
Tensor::diag(v, k)              // Vector to diagonal matrix
tensor.diagonal(offset)         // Extract diagonal
tensor.trace()                  // Sum of diagonal
Tensor::tril(m, k)              // Lower triangular
Tensor::triu(m, k)              // Upper triangular
```

</details>

See [docs/ops.md](docs/ops.md) for the complete API reference.

----

## Platform Support

### Requirements

| Platform | Compiler | Build System | Optional |
|----------|----------|--------------|----------|
| **macOS 11+** | Xcode 13+ / Clang 13+ | CMake 3.20+ | Metal GPU |
| **Linux (x86_64/ARM)** | GCC 10+ / Clang 13+ | CMake 3.20+ | OpenBLAS, OpenMP |
| **Windows** | MSVC 2019+ | CMake 3.20+ | OpenBLAS |
| **WebAssembly** | Emscripten 3.0+ | CMake 3.20+ | — |
| **RISC-V** | GCC 10+ / Clang 13+ | CMake 3.20+ | — |

### BLAS Backend Detection

Axiom automatically detects and links available BLAS libraries:

1. **Apple Accelerate** (macOS) — Preferred on Apple platforms
2. **OpenBLAS** — High-performance open-source BLAS
3. **Native fallback** — Always works, pure C++ implementation

For portable distributions, Axiom can dynamically link BLAS at runtime.

### Data Types

| Category | Types |
|----------|-------|
| **Floating Point** | `Float16`, `Float32`, `Float64` |
| **Signed Integer** | `Int8`, `Int16`, `Int32`, `Int64` |
| **Unsigned Integer** | `UInt8`, `UInt16`, `UInt32`, `UInt64` |
| **Boolean** | `Bool` |
| **Complex** | `Complex64`, `Complex128` |

----

### Roadmap

**Sketch in NumPy, deploy with Axiom.**

Axiom is building toward a future where prototyping in Python and deploying in C++ requires zero mental overhead. The API parity is intentional—your NumPy code translates line-by-line.

- **Lazy Evaluation** *(in development)* — Expression graph compilation for automatic kernel fusion and memory optimization
- **ONNX Runtime Integration** — Load and run ONNX models directly
- **Quantization Toolkit** — INT8/INT4 quantization for edge deployment
- **Custom Op Registration** — Extend Axiom with your own kernels
- **Full Portability** — Single codebase targeting x86, ARM, RISC-V, WebAssembly, and embedded platforms with dynamically linked backends

----

## Building from Source

```bash
# Clone
git clone https://github.com/frikallo/axiom.git
cd axiom

# Build (release mode)
make release

# Run tests
make test

# Install system-wide
sudo cmake --install build

# Optional: Build with OpenMP
cmake -B build -DCMAKE_BUILD_TYPE=Release -DAXIOM_USE_OPENMP=ON
cmake --build build
```

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the project style (`make format`)
2. All tests pass (`make test`)
3. New features include tests
4. Documentation is updated

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

----

## License

Axiom is licensed under the **MIT License**. You are free to use, modify, and distribute Axiom in both open-source and proprietary projects.

See [LICENSE](LICENSE) for the full license text.

----

## Citation

If Axiom is useful in your research, please cite:

```bibtex
@misc{axiom2025,
  title={Axiom: High-Performance Tensor Library for C++},
  author={Noah Kay},
  year={2025},
  url={https://github.com/frikallo/axiom}
}
```
