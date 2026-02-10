# Einops

*For a tutorial introduction, see [User Guide: Einops](../user-guide/einops).*

Einstein-inspired operations in the `axiom::einops` namespace. Also available as member functions on Tensor.

## einops::rearrange / Tensor::rearrange

```cpp
Tensor einops::rearrange(const Tensor &tensor, const std::string &pattern,
                         const std::map<std::string, size_t> &axis_sizes = {});
Tensor Tensor::rearrange(const std::string &pattern,
                         const std::map<std::string, size_t> &axis_sizes = {}) const;
```

Rearrange tensor dimensions using a pattern string.

**Parameters:**
- `pattern` (*string*) -- Pattern like `"b h w c -> b c h w"`.
- `axis_sizes` -- Optional axis size hints for decomposing axes.

**Example:**
```cpp
// Transpose
auto transposed = x.rearrange("h w c -> c h w");

// Flatten spatial dims
auto flat = x.rearrange("b h w c -> b (h w c)");

// Split channel dimension
auto split = x.rearrange("b h w (c g) -> b h w c g", {{"g", 4}});
```

---

## einops::reduce / Tensor::reduce

```cpp
Tensor einops::reduce(const Tensor &tensor, const std::string &pattern,
                      const std::string &reduction,
                      const std::map<std::string, size_t> &axis_sizes = {});
Tensor Tensor::reduce(const std::string &pattern, const std::string &reduction,
                      const std::map<std::string, size_t> &axis_sizes = {}) const;
```

Reduce tensor using a pattern. Axes in the input but not in the output are reduced.

**Parameters:**
- `pattern` (*string*) -- Pattern where output has fewer axes than input.
- `reduction` (*string*) -- Reduction operation: `"sum"`, `"mean"`, `"max"`, `"min"`, `"prod"`.
- `axis_sizes` -- Optional axis size hints.

**Example:**
```cpp
// Global average pooling
auto gap = features.reduce("b h w c -> b c", "mean");

// 2x2 average pooling
auto pooled = x.reduce("b (h p1) (w p2) c -> b h w c", "mean",
                       {{"p1", 2}, {"p2", 2}});

// Max pooling
auto maxpooled = x.reduce("b h w c -> b c", "max");
```

---

## einops::einsum

```cpp
Tensor einops::einsum(const std::string &equation,
                      const std::vector<Tensor> &operands);
```

Einstein summation convention.

**Parameters:**
- `equation` (*string*) -- Einsum equation like `"ij,jk->ik"`.
- `operands` -- Vector of input tensors.

**Supported patterns:**
```cpp
einops::einsum("ij,jk->ik", {A, B});     // Matrix multiply
einops::einsum("ii->", {A});              // Trace
einops::einsum("ij->ji", {A});            // Transpose
einops::einsum("ij,ij->ij", {A, B});      // Element-wise multiply
einops::einsum("bij,bjk->bik", {A, B});   // Batched matmul
einops::einsum("ijk->", {A});             // Sum all elements
einops::einsum("ij->j", {A});             // Sum over rows
```

---

## Pattern Syntax

- **Simple axes:** `h`, `w`, `c` -- single named dimensions.
- **Grouped axes:** `(h w)` -- dimensions to merge or split.
- **Ellipsis:** `...` -- remaining dimensions.
- **Arrow:** `->` -- separates input and output patterns.

## Exceptions

- `EinopsError` -- Base exception for einops operations.
- `EinopsParseError` -- Pattern syntax error.
- `EinopsShapeError` -- Shape incompatibility.

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Tensor Manipulation](tensor-manipulation), [Reductions](reductions)
