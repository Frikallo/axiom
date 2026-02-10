# User Guide

In-depth guides on tensors, operations, GPU acceleration, and advanced features.

```{toctree}
:hidden:

tensor-basics
operations
shape-manipulation
indexing-and-slicing
reductions-and-statistics
linear-algebra
fft
einops
random
custom-functors
lazy-evaluation
gpu-acceleration
file-io
error-handling
debugging-and-profiling
```

## Fundamentals

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tensor Basics
:link: tensor-basics
:link-type: doc

Tensors, dtypes, shapes, devices, memory layout, and the core data model.
:::

:::{grid-item-card} Operations
:link: operations
:link-type: doc

Dispatch system, broadcasting, type promotion, and the fluent API.
:::

:::{grid-item-card} Shape Manipulation
:link: shape-manipulation
:link-type: doc

Reshape, transpose, views, stacking, splitting, and einops rearrange.
:::

:::{grid-item-card} Indexing & Slicing
:link: indexing-and-slicing
:link-type: doc

Slicing, boolean masking, gather/scatter, where, and pad.
:::
::::

## Numerical Computing

::::{grid} 2
:gutter: 3

:::{grid-item-card} Reductions & Statistics
:link: reductions-and-statistics
:link-type: doc

sum, mean, max, min, var, std, axis semantics, and element testing.
:::

:::{grid-item-card} Linear Algebra
:link: linear-algebra
:link-type: doc

Decompositions, solvers, batch operations, and complex-valued matrices.
:::

:::{grid-item-card} FFT
:link: fft
:link-type: doc

1D/2D/ND transforms, window functions, and normalization modes.
:::

:::{grid-item-card} Einops
:link: einops
:link-type: doc

Pattern syntax, rearrange, reduce, and einsum with practical examples.
:::
::::

## Advanced Features

::::{grid} 2
:gutter: 3

:::{grid-item-card} Random Number Generation
:link: random
:link-type: doc

PCG64 engine, seeding, distributions, and reproducibility.
:::

:::{grid-item-card} Custom Functors
:link: custom-functors
:link-type: doc

apply, vectorize, apply_along_axis, and apply_over_axes.
:::

:::{grid-item-card} Lazy Evaluation
:link: lazy-evaluation
:link-type: doc

Computation graphs, operation fusion, eager mode, and the graph cache.
:::

:::{grid-item-card} GPU Acceleration
:link: gpu-acceleration
:link-type: doc

Metal: device placement, transfers, unified memory, and performance tips.
:::
::::

## Practical

::::{grid} 2
:gutter: 3

:::{grid-item-card} File I/O
:link: file-io
:link-type: doc

`.npy` and `.axfb` formats, single/multi-tensor save and load.
:::

:::{grid-item-card} Error Handling
:link: error-handling
:link-type: doc

Exception hierarchy, safety rails (`nan_guard`, `assert_*`), and recovery.
:::

:::{grid-item-card} Debugging & Profiling
:link: debugging-and-profiling
:link-type: doc

`debug_info`, tracing, profiling, and CPU diagnostics.
:::

::::
