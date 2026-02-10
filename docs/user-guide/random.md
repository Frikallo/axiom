# Random Number Generation

Axiom uses the PCG64 algorithm for fast, high-quality random number generation with reproducible seeding.

## Seeding

Set the global seed for reproducible results:

```cpp
using namespace axiom;

Tensor::manual_seed(42);

auto a = Tensor::randn({3, 3});  // Same result every time with seed 42
```

## Distributions

### Normal Distribution

```cpp
// Standard normal (mean=0, std=1)
auto normal = Tensor::randn({1000, 100});

// Scaled normal: mean=5, std=2
auto scaled = Tensor::randn({1000}) * 2.0f + 5.0f;
```

### Uniform Distribution

```cpp
// Uniform in [0, 1)
auto uniform = Tensor::rand({3, 4});

// Uniform in [low, high)
auto custom = Tensor::uniform(-1.0, 1.0, {3, 4});
```

### Random Integers

```cpp
// Random integers in [low, high)
auto dice = Tensor::randint(1, 7, {1000});       // 1-6, Int64
auto bytes = Tensor::randint(0, 256, {100}, DType::Int32);
```

## "Like" Variants

Create random tensors matching the shape and dtype of an existing tensor:

```cpp
auto prototype = Tensor::zeros({4, 4}, DType::Float64);

auto a = Tensor::rand_like(prototype);      // Same shape, dtype
auto b = Tensor::randn_like(prototype);
auto c = Tensor::randint_like(prototype, 0, 10);
```

## Device and Dtype

Random tensors can be created directly on a specific device and dtype:

```cpp
auto gpu_rand = Tensor::randn({256, 256}, DType::Float32, Device::GPU);
auto double_rand = Tensor::rand({100}, DType::Float64);
```

## The PCG64 Engine

Under the hood, Axiom uses the PCG64 (Permuted Congruential Generator) algorithm:

- 128-bit state, 64-bit output
- Period of 2^128
- Excellent statistical quality
- Fast generation

The `RandomGenerator` singleton manages the global state:

```cpp
// Access the global generator (rarely needed directly)
auto &gen = random::RandomGenerator::instance();
gen.seed(42);
```

## Reproducibility Tips

- Always call `manual_seed()` at the start of your program for reproducible results
- The seed affects all subsequent random operations
- Random state is global and shared across threads
- GPU random generation uses the same seed but may produce different sequences than CPU due to different execution order

For complete function signatures, see [API Reference: Random](../api/random).
