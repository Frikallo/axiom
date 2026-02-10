# Reductions & Statistics

Reduction operations collapse one or more dimensions of a tensor, producing summary statistics.

## Axis Semantics

Every reduction accepts an `axis` parameter that specifies which dimension(s) to reduce:

```cpp
using namespace axiom;

auto a = Tensor::ones({2, 3, 4});

// Reduce all elements (axis = -1 with default)
auto total = a.sum();                     // Scalar: 24.0

// Reduce along a single axis
auto row_sums = a.sum(2);                // Shape: {2, 3}
auto col_sums = a.sum(1);                // Shape: {2, 4}
auto batch_sums = a.sum(0);              // Shape: {3, 4}

// Reduce along multiple axes
auto spatial = a.sum({1, 2});            // Shape: {2}
```

### Keeping Dimensions

Pass `keep_dims = true` to retain reduced axes as size-1 dimensions. This is useful for broadcasting the result back to the original shape:

```cpp
auto a = Tensor::randn({4, 5});
auto row_means = a.mean(1, true);         // Shape: {4, 1}

// Broadcast subtraction to center each row
auto centered = a - row_means;            // Shape: {4, 5}
```

## Basic Reductions

### Sum and Product

```cpp
auto a = Tensor::arange(1, 7).reshape({2, 3}).to_float();

auto s = a.sum();                         // Sum of all elements
auto row_s = a.sum(1);                    // Sum each row
auto p = a.prod();                        // Product of all elements
auto col_p = a.prod(0);                   // Product down each column
```

### Min and Max

```cpp
auto a = Tensor::randn({3, 4});

auto mx = a.max();                        // Global maximum
auto mn = a.min();                        // Global minimum
auto row_max = a.max(1);                  // Max of each row
auto col_min = a.min(0);                  // Min of each column

// Peak-to-peak (max - min)
auto range = a.ptp(1);                    // Range of each row
```

### Argmax and Argmin

Return the index of the maximum/minimum value along an axis:

```cpp
auto a = Tensor::randn({3, 4});

auto idx_max = a.argmax(1);              // Index of max in each row
auto idx_min = a.argmin(0);              // Index of min in each column

// Global argmax (on flattened tensor)
auto global_max_idx = a.argmax();
```

Results are `Int64` tensors.

## Statistical Operations

### Mean

```cpp
auto a = Tensor::randn({100, 50});

auto global_mean = a.mean();              // Mean of all elements
auto col_means = a.mean(0);              // Mean of each column
```

### Variance and Standard Deviation

```cpp
auto a = Tensor::randn({100, 50});

// Population variance (ddof=0, default)
auto var_pop = a.var(1);                  // Variance of each row

// Sample variance (ddof=1, Bessel's correction)
auto var_sample = a.var(1, 1);            // ddof=1

// Standard deviation
auto std_pop = a.std(1);                  // Population std
auto std_sample = a.std(1, 1);            // Sample std (ddof=1)

// Multi-axis
auto spatial_var = a.var({0, 1});         // Variance across all elements
```

The `ddof` parameter (delta degrees of freedom) adjusts the divisor: `N - ddof`.

## Boolean Reductions

### Any and All

Test whether elements satisfy a condition:

```cpp
auto a = Tensor::randn({3, 4});
auto mask = ops::greater(a, Tensor::zeros({1}));

// Check if any element is positive
auto has_positive = mask.any();           // Scalar Bool

// Check per-row
auto row_any = mask.any(1);              // Shape: {3}, Bool

// Check if all elements are positive
auto all_positive = mask.all();           // Scalar Bool
auto row_all = mask.all(1);              // Shape: {3}, Bool
```

## Free Function Variants

All reductions are available as both member functions and free functions in the `ops` namespace:

```cpp
auto a = Tensor::randn({3, 4});

// These are equivalent:
auto s1 = a.sum(1);
auto s2 = ops::sum(a, {1});

auto m1 = a.mean(0);
auto m2 = ops::mean(a, {0});

auto mx1 = a.max(1);
auto mx2 = ops::max(a, {1});
```

The free function versions accept `std::vector<int>` for axes, while member functions accept a single `int`.

## Reduction with Type Promotion

Reductions on integer types are promoted to avoid overflow:

- `Int8`, `Int16`, `Int32` sums promote to `Int64`
- `UInt8`, `UInt16`, `UInt32` sums promote to `UInt64`
- `Float16` reductions promote to `Float32`
- `Bool` reductions (any/all) stay `Bool`; sum promotes to `Int64`

## Common Patterns

### Normalization

```cpp
auto a = Tensor::randn({4, 100});

// Z-score normalization per row
auto mu = a.mean(1, true);               // Shape: {4, 1}
auto sigma = a.std(1, 0, true);          // Shape: {4, 1}
auto normalized = (a - mu) / sigma;
```

### Softmax (Manual)

```cpp
auto logits = Tensor::randn({4, 10});

auto max_val = logits.max(1, true);       // For numerical stability
auto shifted = logits - max_val;
auto exp_vals = shifted.exp();
auto probs = exp_vals / exp_vals.sum(1, true);

// Or just use the built-in:
auto probs2 = logits.softmax(1);
```

For complete function signatures, see [API Reference: Reductions](../api/reductions).
