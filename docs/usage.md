# Usage Guide

This page showcases Axiom's API through practical examples. If you've used NumPy or PyTorch, most of this will feel immediately familiar.

For the full auto-generated API reference, see the [API Reference](api/index.md).

## Tensor Creation

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;

// Filled tensors
auto a = Tensor::zeros({3, 4});
auto b = Tensor::ones({4, 5});
auto c = Tensor::full({2, 3}, 3.14f);

// Random tensors
auto x = Tensor::randn({3, 4});              // Normal distribution
auto u = Tensor::rand({3, 4});               // Uniform [0, 1)
auto idx = Tensor::randint(0, 10, {3, 4});   // Random integers

// Sequences and grids
auto r = Tensor::arange(0, 10);              // [0, 1, ..., 9]
auto l = Tensor::linspace(0, 1, 100);        // 100 points in [0, 1]
auto g = Tensor::logspace(0, 3, 4);          // [1, 10, 100, 1000]

// Identity and diagonal
auto I = Tensor::eye(4);
auto d = Tensor::diag(Tensor::arange(3));

// From raw data
auto t = Tensor::from_data<float>({1, 2, 3, 4}, {2, 2});
```

## Arithmetic and Chaining

Operations return tensors, so they chain naturally:

```cpp
auto x = Tensor::randn({32, 64});
auto w = Tensor::randn({64, 10});

// Chain operations fluently
auto logits = x.matmul(w).relu().softmax(-1);

// Standard arithmetic with operator overloading
auto y = (a + b) * c - d / 2.0f;

// In-place operators
x += 1.0f;
x *= 2.0f;
x -= b;
```

## Slicing and Indexing

Slices create **zero-copy views** — no data is copied:

```cpp
auto t = Tensor::arange(16).reshape({4, 4});

// NumPy-style slicing
auto row = t[{0}];                           // First row
auto block = t.slice({Slice(1, 3), Slice(1, 3)});  // 2x2 submatrix
auto stepped = t.slice({Slice(0, 4, 2)});    // Every other row
auto last = t[{-1}];                         // Last row

// Modifying a view modifies the original
auto view = t.slice({Slice(0, 2), Slice(0, 2)});
view.fill(0.0f);  // Zeros out the top-left 2x2 block of t
```

## Broadcasting

Broadcasting follows NumPy rules. `expand` is zero-copy via stride tricks:

```cpp
auto col = Tensor::randn({3, 1});
auto row = Tensor::randn({1, 4});

// Implicit broadcast in binary ops
auto grid = col + row;  // Shape: {3, 4}

// Explicit zero-copy broadcast
auto expanded = col.expand({3, 4});          // No data copy
auto tiled = col.repeat({1, 4});             // Data copy
```

## Comparison and Masking

Comparisons return boolean tensors, not scalars:

```cpp
auto x = Tensor::randn({3, 3});

auto mask = x > 0;                           // Bool tensor
auto positive = x.where(mask, 0.0f);         // Zero out negatives
auto selected = x.masked_select(x > 0.5);   // Extract matching elements (1D)

// In-place masking
x.masked_fill_(x < 0, 0.0f);                // ReLU via mask

// Logical combinations
auto both = (x > 0) && (x < 1);
auto either = (x > 0) || (x < -1);

// Reductions on masks
bool any_neg = (x < 0).any().item<bool>();
bool all_pos = (x > 0).all().item<bool>();
```

## Shape Manipulation

```cpp
auto t = Tensor::arange(24);

// Reshape (infer one dimension with -1)
auto r = t.reshape({2, 3, -1});              // {2, 3, 4}

// Transpose
auto m = Tensor::randn({3, 4});
auto mt = m.T();                             // {4, 3}
auto p = t.reshape({2, 3, 4}).transpose({2, 0, 1});  // {4, 2, 3}

// Squeeze / unsqueeze
auto s = Tensor::ones({1, 3, 1, 4});
auto squeezed = s.squeeze();                 // {3, 4}
auto unsqueezed = m.unsqueeze(0);            // {1, 3, 4}

// Flatten
auto flat = t.reshape({2, 3, 4}).flatten();         // {24}
auto partial = t.reshape({2, 3, 4}).flatten(1, 2);  // {2, 12}

// Flip, roll, swap
auto flipped = m.flipud();
auto rolled = m.roll(2, 1);
auto swapped = t.reshape({2, 3, 4}).swapaxes(0, 2);
```

## Stacking and Splitting

```cpp
auto a = Tensor::ones({2, 3});
auto b = Tensor::full({2, 3}, 2.0f);

// Concatenate along existing axis
auto cat = Tensor::cat({a, b}, 0);           // {4, 3}

// Stack along new axis
auto stacked = Tensor::stack({a, b}, 0);     // {2, 2, 3}

// Convenience functions
auto v = Tensor::vstack({a, b});             // {4, 3}
auto h = Tensor::hstack({a, b});             // {2, 6}

// Split
auto parts = Tensor::arange(6).split(3, 0); // 3 equal parts
auto chunks = Tensor::arange(10).chunk(3);   // 3 chunks (may be unequal)
```

## Reductions

```cpp
auto x = Tensor::randn({3, 4});

// Global reductions
auto total = x.sum();
auto avg = x.mean();
auto mx = x.max();

// Axis reductions
auto row_sums = x.sum({1});                  // {3}
auto col_means = x.mean({0});               // {4}

// Keep dimensions
auto kept = x.sum({1}, true);               // {3, 1}

// Statistical
auto s = x.std();
auto v = x.var();

// Argmax / argmin
auto idx = x.argmax(1);                     // Index of max per row
```

## Lazy Evaluation

Axiom is **lazy by default**. Operations build a computation graph — nothing executes until a value is needed:

```cpp
auto a = Tensor::randn({1000, 1000});
auto b = Tensor::randn({1000, 1000});

auto c = ops::add(a, b);                    // No computation yet
auto d = ops::relu(c);                      // Still deferred
auto e = ops::multiply(d, d);               // Graph grows, nothing runs

e.is_lazy();                                // true

// Materialization happens automatically on data access
float val = e.item<float>({0, 0});          // NOW the graph executes
e.is_lazy();                                // false

// The graph compiler fuses operations automatically:
// add → relu → multiply becomes a single fused SIMD kernel
```

**Why lazy?** The graph compiler can see the full chain before executing, enabling operator fusion (12 SIMD-optimized patterns like `AddReLU`, `MulAdd`, `ScaleShiftReLU`), dead code elimination, and buffer reuse.

For debugging, force eager execution with a scoped helper:

```cpp
{
    graph::EagerModeScope eager;             // Everything in this scope runs immediately
    auto result = ops::relu(ops::add(a, b)); // Executes now, no graph
}
// Back to lazy outside the scope
```

## Type Conversions

```cpp
auto f32 = Tensor::randn({3, 3});

// Named conversions
auto f64 = f32.to_double();
auto i32 = f32.to_int();
auto fp16 = f32.half();
auto bf16 = f32.bfloat16();
auto cplx = f32.to_complex();

// Generic
auto custom = f32.astype(DType::Int64);

// Type promotion is automatic in mixed operations
auto a = Tensor::full({2, 2}, 10);           // Int32
auto b = Tensor::full({2, 2}, 5.5f);        // Float32
auto c = a + b;                              // Float32 (promoted)
```

## Device Transfer

On macOS, every operation runs on Metal GPU — same API, no code changes:

```cpp
auto cpu = Tensor::randn({1024, 1024});
auto gpu = cpu.gpu();                        // Transfer to Metal GPU

// All operations work identically on GPU
auto result = gpu.matmul(gpu.T()).softmax(-1).sum({1});

// Transfer back
auto back = result.cpu();
```

## Views and Copy Semantics

Axiom uses zero-copy views wherever possible. Slicing, transposing, and reshaping share underlying storage:

```cpp
auto original = Tensor::arange(16).reshape({4, 4});

// These are views — no data copied
auto sliced = original.slice({Slice(0, 2)});
auto transposed = original.T();
auto reshaped = original.reshape({2, 8});

original.shares_storage(sliced);             // true

// Explicit deep copy
auto independent = original.clone();
original.shares_storage(independent);        // false
```

## Einops

Pattern-based tensor rearrangement inspired by [einops](https://github.com/arogozhnikov/einops):

```cpp
auto img = Tensor::randn({2, 224, 224, 3});

// Rearrange: NHWC → NCHW
auto nchw = img.rearrange("b h w c -> b c h w");

// Reduce: spatial average pooling
auto pooled = img.reduce("b (h p1) (w p2) c -> b h w c",
    "mean", {{"p1", 2}, {"p2", 2}});

// Global average pooling
auto gap = img.reduce("b h w c -> b c", "mean");

// Einsum
auto attn = Tensor::einsum("bhqd,bhkd->bhqk", {Q, K});
```

## Linear Algebra

```cpp
auto A = Tensor::randn({4, 4});
auto b = Tensor::randn({4, 1});

// Decompositions
auto [U, S, Vt] = linalg::svd(A);
auto [Q, R] = linalg::qr(A);
auto L = linalg::cholesky(A.matmul(A.T()));   // Needs positive-definite

// Solvers
auto x = linalg::solve(A, b);
auto x_ls = linalg::lstsq(A, b);

// Properties
auto d = linalg::det(A);
auto n = linalg::norm(A);
auto r = linalg::matrix_rank(A);
auto inv = linalg::inv(A);
auto pinv = linalg::pinv(A);

// Eigendecomposition
auto [eigenvalues, eigenvectors] = linalg::eig(A);
```

## FFT

```cpp
auto signal = Tensor::randn({256});

// 1D transforms
auto spectrum = fft::fft(signal);
auto recovered = fft::ifft(spectrum);

// Real-valued FFT (more efficient)
auto rfft_out = fft::rfft(signal);           // Output: n/2 + 1 complex

// 2D transforms
auto image = Tensor::randn({64, 64});
auto freq = fft::fft2(image);

// Window functions
auto window = fft::hann_window(256);
auto windowed_signal = signal * window;
auto windowed_fft = fft::fft(windowed_signal);

// Frequencies
auto freqs = fft::fftfreq(256, 1.0 / 44100);
```

## Debugging

```cpp
auto t = Tensor::randn({3, 4});

// Print
std::cout << t << "\n";                      // NumPy-style formatted output

// Inspect
std::cout << t.debug_info() << "\n";         // Shape, dtype, device, strides, etc.

// Safety rails
t.assert_shape({3, 4});                      // Throws if wrong shape
t.assert_finite();                           // Throws if NaN or Inf

// Chainable guard
auto safe = compute().nan_guard().sqrt();    // Throws before sqrt if NaN

// Checks (no throw)
t.has_nan();                                 // bool
t.has_inf();                                 // bool
t.is_contiguous();                           // bool
t.is_view();                                 // bool
```

## I/O

```cpp
// NumPy format (.npy) — interop with Python
io::save("tensor.npy", t);
auto loaded = io::load("tensor.npy");

// Axiom FlatBuffers format (.axfb) — fast binary
io::save("model.axfb", t);

// Archives (multiple tensors)
io::save_archive("weights.axfb", {{"W1", w1}, {"b1", b1}, {"W2", w2}});
auto weights = io::load_archive("weights.axfb");
auto W1 = weights["W1"];
```
