# API Reference

Complete reference for every class, function, and type in Axiom.

*For tutorial introductions, see the [User Guide](../user-guide/index).*

## Core

- [Tensor Class](tensor-class) -- Constructors, attributes, data access, introspection
- [Data Types](dtypes) -- DType enum, 14 types, promotion rules
- [Devices & Storage](devices-and-storage) -- Device enum, Storage class, CPU/Metal backends
- [Operators](operators) -- All overloaded C++ operators

## Tensor Operations

- [Tensor Creation](tensor-creation) -- Factory methods: zeros, ones, arange, linspace, etc.
- [Tensor Manipulation](tensor-manipulation) -- reshape, transpose, squeeze, flip, roll, etc.
- [Stacking & Splitting](stacking-and-splitting) -- cat, stack, vstack/hstack, split, chunk

## Math & Comparison

- [Arithmetic](arithmetic) -- add, subtract, multiply, divide, power, modulo
- [Comparison](comparison) -- equal, not_equal, less, greater, isclose, allclose
- [Unary Math](unary-math) -- abs, sqrt, exp, log, trig, sign, floor, ceil, round
- [Logical & Bitwise](logical-and-bitwise) -- logical_and/or/xor/not, bitwise ops, shifts

## Reductions & Statistics

- [Reductions](reductions) -- sum, mean, max, min, argmax, argmin, prod, any, all, var, std

## Neural Network

- [Activations](activations) -- relu, gelu, silu, sigmoid, tanh, softmax
- [Normalization](normalization) -- layer_norm, rms_norm, dropout
- [Pooling](pooling) -- max_pool, avg_pool (1D/2D/3D), adaptive variants

## Selection & Indexing

- [Masking & Selection](masking-and-selection) -- where, masked_fill, masked_select
- [Indexing Ops](indexing-ops) -- gather, scatter, index_select, take, put_along_axis

## Advanced

- [Linear Algebra](linalg) -- Full `axiom::linalg` namespace (30+ functions)
- [FFT](fft) -- Full `axiom::fft` namespace
- [Einops](einops) -- rearrange, reduce, einsum
- [Random](random) -- PCG64, RandomGenerator, manual_seed, distributions
- [Custom Functors](functors) -- apply, vectorize, apply_along_axis, fromfunc
- [File I/O](io) -- save/load, format detection, .axfb and .npy

## Utilities

- [Numeric Constants](numeric) -- NaN, Inf, epsilon, pi, safe arithmetic
- [Errors](errors) -- Full exception hierarchy
- [Debug & Profiling](debug) -- trace, profile, cpu_info APIs
- [Parallelism](parallel) -- Thread management, parallelization thresholds
- [System](system) -- is_metal_available, device_to_string

```{toctree}
:hidden:

tensor-class
tensor-creation
tensor-manipulation
arithmetic
comparison
unary-math
logical-and-bitwise
reductions
activations
masking-and-selection
indexing-ops
stacking-and-splitting
normalization
pooling
linalg
fft
einops
random
functors
io
dtypes
devices-and-storage
operators
numeric
errors
debug
parallel
system
```
