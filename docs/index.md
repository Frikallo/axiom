# Axiom

A C++20 tensor library optimized for Apple Silicon with Metal GPU acceleration and NumPy compatibility.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started/index
:link-type: doc

Install Axiom and run your first tensor operation in 5 minutes.
:::

:::{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc

In-depth guides on tensors, operations, GPU acceleration, and advanced features.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Complete reference for every class, function, and type.
:::

:::{grid-item-card} Developer Guide
:link: development/index
:link-type: doc

Build from source, add operations, and understand the architecture.
:::

::::

## Feature Highlights

- **NumPy-compatible API** -- Familiar tensor operations with C++ performance. [Learn more](user-guide/operations)
- **Metal GPU acceleration** -- Automatic dispatch to Apple Silicon GPU via MPSGraph. [Learn more](user-guide/gpu-acceleration)
- **Einops patterns** -- Readable tensor reshaping with `rearrange`, `reduce`, and `einsum`. [Learn more](user-guide/einops)
- **Lazy evaluation** -- Build computation graphs with automatic operation fusion. [Learn more](user-guide/lazy-evaluation)
- **LAPACK linear algebra** -- Full decomposition suite: SVD, QR, Cholesky, eigendecomposition. [Learn more](user-guide/linear-algebra)
- **Zero-copy views** -- Reshape, transpose, and slice without copying data. [Learn more](user-guide/tensor-basics)
- **14 data types** -- Bool through Complex128, with automatic type promotion. [Learn more](api/dtypes)

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
user-guide/index
api/index
development/index
BENCHMARKS
```
