# Einops

Axiom includes a built-in einops implementation for expressive tensor manipulation using pattern strings. This provides a readable alternative to chains of reshape, transpose, and reduction calls.

## Rearrange

`rearrange` reshapes and transposes tensors using a pattern string:

```cpp
using namespace axiom;

auto images = Tensor::randn({8, 3, 32, 32});

// Transpose: move channels to last
auto nhwc = images.rearrange("b c h w -> b h w c");
// Shape: {8, 32, 32, 3}

// Flatten spatial dimensions
auto flat = images.rearrange("b c h w -> b (c h w)");
// Shape: {8, 3072}

// Merge batch and channels
auto merged = images.rearrange("b c h w -> (b c) h w");
// Shape: {24, 32, 32}
```

### Splitting Dimensions

Use parentheses and axis sizes to split a dimension:

```cpp
auto a = Tensor::randn({8, 3, 32, 32});

// Split height into patches
auto patches = a.rearrange(
    "b c (h ph) (w pw) -> b (h w) (c ph pw)",
    {{"ph", 8}, {"pw", 8}}
);
// Shape: {8, 16, 192}  (4*4 patches, each 3*8*8)

// Split batch into grid
auto grid = a.rearrange(
    "(b1 b2) c h w -> (b1 h) (b2 w) c",
    {{"b1", 2}, {"b2", 4}}
);
// Shape: {64, 128, 3}
```

## Reduce

`reduce` combines rearrangement with a reduction operation:

```cpp
auto images = Tensor::randn({8, 3, 32, 32});

// Average pooling with einops
auto pooled = images.reduce(
    "b c (h h2) (w w2) -> b c h w", "mean",
    {{"h2", 2}, {"w2", 2}}
);
// Shape: {8, 3, 16, 16}

// Global average pooling
auto gap = images.reduce("b c h w -> b c", "mean");
// Shape: {8, 3}

// Max over spatial dimensions
auto spatial_max = images.reduce("b c h w -> b c", "max");
// Shape: {8, 3}
```

Supported reductions: `"sum"`, `"mean"`, `"max"`, `"min"`, `"prod"`.

## Einsum

`einsum` performs Einstein summation -- a general-purpose notation for contractions, traces, outer products, and more:

```cpp
// Matrix multiplication
auto A = Tensor::randn({3, 4});
auto B = Tensor::randn({4, 5});
auto C = einops::einsum("ij,jk->ik", {A, B});
// Shape: {3, 5}

// Batch matrix multiplication
auto X = Tensor::randn({8, 3, 4});
auto Y = Tensor::randn({8, 4, 5});
auto Z = einops::einsum("bij,bjk->bik", {X, Y});
// Shape: {8, 3, 5}

// Dot product
auto u = Tensor::randn({5});
auto v = Tensor::randn({5});
auto dot = einops::einsum("i,i->", {u, v});
// Shape: {} (scalar)

// Outer product
auto outer = einops::einsum("i,j->ij", {u, v});
// Shape: {5, 5}

// Trace
auto M = Tensor::randn({4, 4});
auto tr = einops::einsum("ii->", {M});
// Shape: {} (scalar)

// Bilinear form
auto x = Tensor::randn({3});
auto W = Tensor::randn({3, 3});
auto y = Tensor::randn({3});
auto result = einops::einsum("i,ij,j->", {x, W, y});
// Shape: {} (scalar)
```

## Pattern Syntax

Einops patterns follow this format:

```
input_axes -> output_axes
```

Rules:
- Space-separated axis names: `b c h w`
- Parenthesized groups merge/split axes: `(b c)` or `(h h2)`
- Axes present in input but not output are reduced (for `reduce`)
- Axis sizes can be specified via the map parameter for splitting

| Pattern | Effect |
|---------|--------|
| `b c h w -> b h w c` | Transpose |
| `b c h w -> b (c h w)` | Flatten |
| `(b1 b2) c h w -> b1 b2 c h w` | Split (needs axis sizes) |
| `b c h w -> b c` | Reduce h, w (reduce only) |
| `b c (h h2) (w w2) -> b c h w` | Pooling (reduce only, needs sizes) |

## Free Function API

Einops operations are also available as free functions:

```cpp
auto result1 = einops::rearrange(tensor, "b c h w -> b (c h w)");
auto result2 = einops::reduce(tensor, "b c h w -> b c", "mean");
auto result3 = einops::einsum("ij,jk->ik", {A, B});
```

For complete function signatures, see [API Reference: Einops](../api/einops).
