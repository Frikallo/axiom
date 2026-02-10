# Operations

Axiom provides a rich set of tensor operations -- arithmetic, comparisons, unary math, and neural network activations -- all with automatic broadcasting, type promotion, and backend dispatch. This guide explains how the operation system works and how to use it effectively.

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;

// Initialize the operation registry before using any operations
ops::OperationRegistry::initialize_builtin_operations();
```

## The Fluent API

Every operation in Axiom is available in two forms: as a **member function** on `Tensor` (the fluent API) and as a **free function** in the `ops::` namespace. The fluent API allows natural method chaining, while the free-function form is useful for functional composition or when working with temporaries.

```cpp
auto x = Tensor::randn({3, 4});

// Fluent API -- chain operations left to right
auto result = x.abs().sqrt().sum();

// Free-function API -- equivalent, but nested
auto result2 = ops::sum(ops::sqrt(ops::abs(x)));
```

Both forms dispatch to exactly the same underlying kernel. Use whichever style reads better in context. The fluent API tends to work well for pipelines:

```cpp
auto logits = Tensor::randn({8, 10});
auto probs = logits.softmax(-1);

// Multi-step normalization pipeline
auto normalized = x.abs()
                   .log()
                   .clip(-10.0, 10.0)
                   .softmax(-1);
```

Binary operations use operators or free functions:

```cpp
auto a = Tensor::randn({3, 4});
auto b = Tensor::randn({3, 4});

auto c = a + b;                    // operator
auto d = ops::add(a, b);          // free function (identical result)
auto e = ops::power(a, b);        // no operator overload for power
```

## Backend Dispatch

Axiom uses a two-level dispatch system to route operations to the appropriate hardware backend.

**Level 1: OpType enum.** Every operation has a corresponding entry in the `ops::OpType` enum (e.g., `OpType::Add`, `OpType::Sqrt`, `OpType::Softmax`). This is the canonical identifier for the operation.

**Level 2: OperationRegistry.** At startup, `OperationRegistry::initialize_builtin_operations()` registers concrete kernel implementations for each `(OpType, Device)` pair. The CPU backend registers SIMD-accelerated kernels; the Metal GPU backend registers MPSGraph-based kernels.

When you call an operation, dispatch happens automatically based on the tensor's device:

```cpp
auto cpu_tensor = Tensor::randn({256, 256});
auto gpu_tensor = cpu_tensor.gpu();

// Dispatches to CPU SIMD kernel
auto cpu_result = cpu_tensor.sqrt();

// Dispatches to Metal GPU kernel
auto gpu_result = gpu_tensor.sqrt();
```

If a GPU kernel is not available for a given operation, Axiom falls back to the CPU backend transparently. You can query availability explicitly:

```cpp
bool has_gpu_add = ops::OperationRegistry::is_operation_available(
    ops::OpType::Add, Device::GPU);
```

Mixed-device operands in binary operations are resolved by moving tensors to a common device before execution.

## Broadcasting

Axiom follows NumPy-compatible broadcasting rules. When two tensors with different shapes are combined in a binary operation, their shapes are compared element-wise starting from the **trailing** (rightmost) dimensions:

1. If both dimensions are equal, they are compatible.
2. If one dimension is **1**, it is broadcast (stretched) to match the other.
3. If a tensor has fewer dimensions, it is treated as if it were prefixed with dimensions of size 1.

If the shapes are incompatible under these rules, the operation throws a `ShapeError`.

```cpp
auto a = Tensor::randn({3, 4});    // shape: (3, 4)
auto b = Tensor::randn({4});       // shape: (4,)
auto c = a + b;                     // shape: (3, 4)
// b is treated as (1, 4) and broadcast along axis 0

auto d = Tensor::randn({3, 1});    // shape: (3, 1)
auto e = a + d;                     // shape: (3, 4)
// d is broadcast along axis 1

auto f = Tensor::randn({1, 4});    // shape: (1, 4)
auto g = d + f;                     // shape: (3, 4)
// both are broadcast: d along axis 1, f along axis 0
```

A few more shape combinations:

| Shape A       | Shape B       | Result        | Notes                           |
|---------------|---------------|---------------|---------------------------------|
| `(5, 3, 4)`  | `(3, 4)`      | `(5, 3, 4)`  | B prefixed with 1               |
| `(5, 3, 4)`  | `(5, 1, 4)`  | `(5, 3, 4)`  | B broadcast on axis 1           |
| `(8, 1, 6)`  | `(7, 1)`      | `(8, 7, 6)`  | Both broadcast on different axes|
| `(3, 4)`     | `(5, 4)`      | **Error**     | 3 and 5 are incompatible        |

Broadcasting is zero-copy when possible -- Axiom uses stride manipulation (setting the stride to 0 for broadcast dimensions) rather than duplicating data.

You can also use the broadcasting utilities directly:

```cpp
bool ok = ops::are_broadcastable(a.shape(), b.shape());
auto info = ops::compute_broadcast_info(a.shape(), b.shape());
// info.result_shape contains the output shape
```

## Type Promotion

When operands have different dtypes, Axiom automatically promotes them to a common type before executing the operation. The rules follow NumPy conventions:

- **Bool + integer** produces the integer type: `Bool + Int32` becomes `Int32`.
- **Integer + float** produces the float type: `Int32 + Float32` becomes `Float32`.
- **Float32 + Float64** produces `Float64` (wider float wins).
- **Any + Complex** produces a complex type: `Float32 + Complex64` becomes `Complex64`.
- **Integer + integer** promotes to the wider type, preserving signedness when both agree. If one is signed and the other unsigned, the result is signed.

```cpp
auto ints = Tensor::arange(5);                       // Int32
auto floats = Tensor::ones({5});                     // Float32
auto result = ints + floats;                          // Float32

auto f32 = Tensor::randn({3, 3});                   // Float32
auto f64 = Tensor::randn({3, 3}, DType::Float64);   // Float64
auto promoted = f32 + f64;                            // Float64
```

You can query the promotion result without executing an operation:

```cpp
DType out_dtype = ops::promote_types(DType::Int32, DType::Float32);
// out_dtype == DType::Float64

DType out_dtype2 = ops::result_type(tensor_a, tensor_b);
// Takes both dtype and shape into account
```

## Arithmetic Operations

Element-wise arithmetic with full broadcasting and type promotion support.

| Operation  | Free function         | Operator   |
|------------|-----------------------|------------|
| Add        | `ops::add(a, b)`      | `a + b`    |
| Subtract   | `ops::subtract(a, b)` | `a - b`    |
| Multiply   | `ops::multiply(a, b)` | `a * b`    |
| Divide     | `ops::divide(a, b)`   | `a / b`    |
| Power      | `ops::power(a, b)`    | --         |
| Modulo     | `ops::modulo(a, b)`   | `a % b`    |

All operators support Tensor-scalar and scalar-Tensor combinations:

```cpp
auto x = Tensor::randn({4, 4});

auto y = x * 2.0f;         // scale every element
auto z = 1.0f - x;         // scalar on the left
auto w = x + 1.0f;         // scalar on the right
```

Additional binary math functions:

```cpp
auto mx = ops::maximum(a, b);   // element-wise max
auto mn = ops::minimum(a, b);   // element-wise min
auto at = ops::atan2(y_vals, x_vals);
auto hy = ops::hypot(a, b);     // sqrt(a^2 + b^2)
```

## Comparison Operations

Comparisons return a `Bool` dtype tensor. They support broadcasting and Tensor-scalar operands.

| Operation      | Free function              | Operator   |
|----------------|----------------------------|------------|
| Equal          | `ops::equal(a, b)`         | `a == b`   |
| Not equal      | `ops::not_equal(a, b)`     | `a != b`   |
| Less           | `ops::less(a, b)`          | `a < b`    |
| Less or equal  | `ops::less_equal(a, b)`    | `a <= b`   |
| Greater        | `ops::greater(a, b)`       | `a > b`    |
| Greater/equal  | `ops::greater_equal(a, b)` | `a >= b`   |

```cpp
auto x = Tensor::randn({3, 4});

auto mask = x > 0.0f;              // Bool tensor: true where positive
auto eq = x == 0.0f;               // Bool tensor: true where zero
auto cmp = x >= Tensor::zeros({3, 4}); // Tensor-Tensor comparison
```

Comparison results are commonly used with `where` and `masked_fill`:

```cpp
auto positive_only = ops::where(x > 0.0f, x, Tensor::zeros_like(x));
auto clamped = x.masked_fill(x < -1.0f, -1.0f);
```

## Unary Math

Element-wise mathematical functions, available in both `ops::` and fluent forms.

| Function     | ops:: form            | Fluent form        |
|--------------|-----------------------|--------------------|
| Absolute     | `ops::abs(x)`        | `x.abs()`          |
| Square root  | `ops::sqrt(x)`       | `x.sqrt()`         |
| Exponential  | `ops::exp(x)`        | `x.exp()`          |
| Logarithm    | `ops::log(x)`        | `x.log()`          |
| Sine         | `ops::sin(x)`        | `x.sin()`          |
| Cosine       | `ops::cos(x)`        | `x.cos()`          |
| Tangent      | `ops::tan(x)`        | `x.tan()`          |
| Error func   | `ops::erf(x)`        | --                 |
| Sign         | `ops::sign(x)`       | `x.sign()`         |
| Floor        | `ops::floor(x)`      | `x.floor()`        |
| Ceil         | `ops::ceil(x)`       | `x.ceil()`         |
| Truncate     | `ops::trunc(x)`      | `x.trunc()`        |
| Round        | `ops::round(x, n)`   | `x.round(n)`       |
| Reciprocal   | `ops::reciprocal(x)` | `x.reciprocal()`   |
| Square       | `ops::square(x)`     | `x.square()`       |
| Cube root    | `ops::cbrt(x)`       | `x.cbrt()`         |
| Negate       | `ops::negate(x)`     | `-x`               |

```cpp
auto x = Tensor::randn({4, 4});

// Chain unary operations
auto result = x.abs().log().floor();

// Element-wise testing (returns Bool tensors)
auto nan_mask  = x.isnan();
auto inf_mask  = x.isinf();
auto good_mask = x.isfinite();

// Clipping to a range
auto clipped = x.clip(-1.0, 1.0);   // clamp is an alias
```

Complex-valued tensors support a subset of unary operations (`abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `conj`) and also provide zero-copy views of real and imaginary components:

```cpp
auto z = Tensor::randn({3, 3}).to_complex();
auto re = z.real();     // zero-copy view of real part
auto im = z.imag();     // zero-copy view of imaginary part
auto zc = z.conj();     // complex conjugate
```

## Activations

Neural network activation functions, designed for use in model forward passes.

```cpp
auto x = Tensor::randn({8, 256});

auto r = x.relu();                   // max(0, x)
auto l = x.leaky_relu(0.1f);        // x if x>0, else 0.1*x
auto s = x.sigmoid();               // 1 / (1 + exp(-x))
auto t = x.tanh();                  // hyperbolic tangent
auto g = x.gelu();                  // Gaussian Error Linear Unit
auto w = x.silu();                  // x * sigmoid(x), a.k.a. Swish
```

Softmax and log-softmax operate along a specified axis (defaulting to the last):

```cpp
auto logits = Tensor::randn({8, 10});

auto probs = logits.softmax(-1);        // probabilities along last axis
auto log_probs = logits.log_softmax(-1);// numerically stable log-probs
```

Activations compose naturally with other operations:

```cpp
// A small feedforward block
auto hidden = input.matmul(W1).relu();
auto output = hidden.matmul(W2).softmax(-1);
```

## In-Place Operations

In-place operators modify the left-hand tensor directly, avoiding an extra allocation for the result. They are available through compound assignment operators and the `ops::` namespace.

| Operator  | Free function                     |
|-----------|-----------------------------------|
| `a += b`  | `ops::add_inplace(a, b)`          |
| `a -= b`  | `ops::subtract_inplace(a, b)`     |
| `a *= b`  | `ops::multiply_inplace(a, b)`     |
| `a /= b`  | `ops::divide_inplace(a, b)`       |

```cpp
auto x = Tensor::ones({3, 4});
auto y = Tensor::randn({3, 4});

x += y;           // x is modified in place
x *= 2.0f;        // scalar in-place
x -= 1.0f;

// Free-function form
ops::add_inplace(x, y);
```

In-place operations require that the left-hand tensor is writeable and owns its data. Attempting an in-place operation on a non-writeable view will throw a `MemoryError`. Broadcasting is supported for the right-hand operand, but the left-hand tensor's shape is never changed.

## Lazy Evaluation Mode

By default, Axiom uses a lazy evaluation strategy for most operations. Rather than executing each operation immediately, it builds a lightweight **computation graph** that records the sequence of operations. The graph is materialized (executed) automatically when you access the tensor's data -- for example, by printing it, reading element values, or passing it to an operation that requires concrete data.

```cpp
auto x = Tensor::randn({1024, 1024});

// These build a graph -- no computation happens yet
auto y = x.abs().sqrt().exp();

// Materialization happens here, when data is accessed
std::cout << y << std::endl;
```

Lazy evaluation enables **operation fusion**: the runtime can combine multiple element-wise operations into a single pass over memory, reducing memory bandwidth pressure and intermediate allocations. This is particularly beneficial for chains of unary operations and element-wise binary operations.

You can check whether a tensor is lazy (not yet materialized):

```cpp
bool pending = y.is_lazy();  // true if graph not yet executed
```

For a detailed treatment of the computation graph, fusion rules, and manual materialization, see [Lazy Evaluation](lazy-evaluation).

---

For complete function signatures, see [API Reference: Arithmetic](../api/arithmetic), [API Reference: Unary Math](../api/unary-math), and [API Reference: Operators](../api/operators).
