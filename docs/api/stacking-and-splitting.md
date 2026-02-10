# Stacking & Splitting

*For a tutorial introduction, see [User Guide: Shape Manipulation](../user-guide/shape-manipulation).*

## Concatenation

### Tensor::concatenate / Tensor::cat

```cpp
static Tensor Tensor::concatenate(const std::vector<Tensor> &tensors,
                                  int axis = 0);
static Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis = 0);
static Tensor Tensor::cat(std::initializer_list<Tensor> tensors, int axis = 0);
```

Join tensors along an existing axis. `cat` is an alias for `concatenate`.

**Parameters:**
- `tensors` -- Tensors to concatenate. Must have the same shape except along `axis`.
- `axis` (*int*) -- Axis to concatenate along. Default: `0`.

**Example:**
```cpp
auto a = Tensor::arange(3);  // [0, 1, 2]
auto b = Tensor::arange(3);  // [0, 1, 2]
auto c = Tensor::cat({a, b}, 0);  // [0, 1, 2, 0, 1, 2]
```

---

### Tensor::cat (member)

```cpp
Tensor Tensor::cat(const Tensor &other, int axis = 0) const;
```

Member function for chaining.

```cpp
auto c = a.cat(b, 0);  // Same as Tensor::cat({a, b}, 0)
```

---

## Stacking

### Tensor::stack

```cpp
static Tensor Tensor::stack(const std::vector<Tensor> &tensors, int axis = 0);
static Tensor Tensor::stack(std::initializer_list<Tensor> tensors, int axis = 0);
```

Stack tensors along a **new** axis. All tensors must have the same shape.

**Example:**
```cpp
auto a = Tensor::arange(3);  // shape (3,)
auto b = Tensor::arange(3);
auto s = Tensor::stack({a, b}, 0);  // shape (2, 3)
```

---

### Tensor::vstack

```cpp
static Tensor Tensor::vstack(const std::vector<Tensor> &tensors);
static Tensor Tensor::vstack(std::initializer_list<Tensor> tensors);
```

Stack vertically (along axis 0). 1D arrays are treated as row vectors.

---

### Tensor::hstack

```cpp
static Tensor Tensor::hstack(const std::vector<Tensor> &tensors);
static Tensor Tensor::hstack(std::initializer_list<Tensor> tensors);
```

Stack horizontally (along axis 1). 1D arrays are concatenated.

---

### Tensor::dstack

```cpp
static Tensor Tensor::dstack(const std::vector<Tensor> &tensors);
static Tensor Tensor::dstack(std::initializer_list<Tensor> tensors);
```

Stack depth-wise (along axis 2).

---

### Tensor::column_stack

```cpp
static Tensor Tensor::column_stack(const std::vector<Tensor> &tensors);
static Tensor Tensor::column_stack(std::initializer_list<Tensor> tensors);
```

Stack 1D arrays as columns into a 2D array.

---

### Tensor::row_stack

```cpp
static Tensor Tensor::row_stack(const std::vector<Tensor> &tensors);
```

Alias for `vstack`.

---

## Splitting

### Tensor::split

```cpp
std::vector<Tensor> Tensor::split(size_t sections, int axis = 0) const;
std::vector<Tensor> Tensor::split(const std::vector<size_t> &indices,
                                  int axis = 0) const;
```

Split a tensor into sub-tensors.

- **By count:** Split into `sections` equal parts along `axis`.
- **By indices:** Split at the given index positions along `axis`.

**Example:**
```cpp
auto x = Tensor::arange(6);  // [0, 1, 2, 3, 4, 5]
auto parts = x.split(3);      // [[0,1], [2,3], [4,5]]
```

---

### Tensor::chunk

```cpp
std::vector<Tensor> Tensor::chunk(size_t n_chunks, int axis = 0) const;
```

Split into `n_chunks` parts. The last chunk may be smaller if the dimension is not evenly divisible.

**Example:**
```cpp
auto x = Tensor::arange(6);
auto chunks = x.chunk(4);  // [[0,1], [2,3], [4], [5]]
```

---

### Tensor::vsplit / hsplit / dsplit

```cpp
std::vector<Tensor> Tensor::vsplit(size_t sections) const;  // split(sections, 0)
std::vector<Tensor> Tensor::hsplit(size_t sections) const;  // split(sections, 1)
std::vector<Tensor> Tensor::dsplit(size_t sections) const;  // split(sections, 2)
```

Convenience methods for splitting along axes 0, 1, and 2.

| CPU | GPU |
|-----|-----|
| ✓   | ✓   |

**See Also:** [Tensor Manipulation](tensor-manipulation)
