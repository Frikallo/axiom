# Coming from NumPy / PyTorch

Axiom's API is designed to feel familiar if you've used NumPy or PyTorch. This page provides a side-by-side comparison.

## Key Differences

| Concept | NumPy / PyTorch | Axiom |
|---------|----------------|-------|
| Language | Python | C++20 |
| Namespace | `np.` / `torch.` | `axiom::` / `ops::` |
| Default dtype | `float64` (NumPy) / `float32` (PyTorch) | `Float32` |
| GPU device | `"cuda"` | `Device::GPU` (Metal) |
| Init required | No | `ops::OperationRegistry::initialize_builtin_operations()` |

## Cheat Sheet

### Tensor Creation

| NumPy | PyTorch | Axiom |
|-------|---------|-------|
| `np.zeros((3, 4))` | `torch.zeros(3, 4)` | `Tensor::zeros({3, 4})` |
| `np.ones((3, 4))` | `torch.ones(3, 4)` | `Tensor::ones({3, 4})` |
| `np.eye(3)` | `torch.eye(3)` | `Tensor::eye(3)` |
| `np.arange(10)` | `torch.arange(10)` | `Tensor::arange(10)` |
| `np.linspace(0, 1, 5)` | `torch.linspace(0, 1, 5)` | `Tensor::linspace(0.0, 1.0, 5)` |
| `np.random.randn(3, 4)` | `torch.randn(3, 4)` | `Tensor::randn({3, 4})` |
| `np.full((3, 4), 5.0)` | `torch.full((3, 4), 5.0)` | `Tensor::full({3, 4}, 5.0f)` |
| `np.array([1, 2, 3])` | `torch.tensor([1, 2, 3])` | `Tensor::from_data(data, {3})` |

### Attributes

| NumPy | PyTorch | Axiom |
|-------|---------|-------|
| `x.shape` | `x.shape` | `x.shape()` |
| `x.dtype` | `x.dtype` | `x.dtype()` |
| `x.ndim` | `x.ndim` | `x.ndim()` |
| `x.size` | `x.numel()` | `x.size()` |
| `x.nbytes` | -- | `x.nbytes()` |
| `x.strides` | `x.stride()` | `x.strides()` |

### Arithmetic

| NumPy | Axiom |
|-------|-------|
| `x + y` | `x + y` |
| `x - y` | `x - y` |
| `x * y` | `x * y` |
| `x / y` | `x / y` |
| `x ** y` | `ops::power(x, y)` |
| `x % y` | `x % y` |
| `x + 2.0` | `x + 2.0f` |

### Shape Manipulation

| NumPy | PyTorch | Axiom |
|-------|---------|-------|
| `x.reshape(3, 4)` | `x.reshape(3, 4)` | `x.reshape({3, 4})` |
| `x.T` | `x.T` | `x.T()` |
| `x.transpose(1, 0)` | `x.permute(1, 0)` | `x.transpose({1, 0})` |
| `x.flatten()` | `x.flatten()` | `x.flatten()` |
| `x.squeeze()` | `x.squeeze()` | `x.squeeze()` |
| `np.expand_dims(x, 0)` | `x.unsqueeze(0)` | `x.unsqueeze(0)` |
| `np.flip(x, 0)` | `x.flip(0)` | `x.flip(0)` |
| `np.roll(x, 3)` | `torch.roll(x, 3)` | `x.roll(3)` |
| `np.concatenate([a, b])` | `torch.cat([a, b])` | `Tensor::cat({a, b})` |
| `np.stack([a, b])` | `torch.stack([a, b])` | `Tensor::stack({a, b})` |

### Reductions

| NumPy | PyTorch | Axiom |
|-------|---------|-------|
| `x.sum()` | `x.sum()` | `x.sum()` |
| `x.sum(axis=0)` | `x.sum(dim=0)` | `x.sum(0)` |
| `x.mean()` | `x.mean()` | `x.mean()` |
| `x.max()` | `x.max()` | `x.max()` |
| `x.argmax()` | `x.argmax()` | `x.argmax()` |
| `x.any()` | `x.any()` | `x.any()` |
| `x.all()` | `x.all()` | `x.all()` |
| `x.var(ddof=1)` | `x.var(unbiased=True)` | `x.var(-1, 1)` |
| `x.std()` | `x.std()` | `x.std()` |

### Math Functions

| NumPy | Axiom (functional) | Axiom (fluent) |
|-------|-------------------|----------------|
| `np.abs(x)` | `ops::abs(x)` | `x.abs()` |
| `np.sqrt(x)` | `ops::sqrt(x)` | `x.sqrt()` |
| `np.exp(x)` | `ops::exp(x)` | `x.exp()` |
| `np.log(x)` | `ops::log(x)` | `x.log()` |
| `np.sin(x)` | `ops::sin(x)` | `x.sin()` |
| `np.floor(x)` | `ops::floor(x)` | `x.floor()` |
| `np.sign(x)` | `ops::sign(x)` | `x.sign()` |
| `np.clip(x, 0, 1)` | `ops::clip(x, min, max)` | `x.clip(0.0, 1.0)` |

### Masking and Selection

| NumPy | PyTorch | Axiom |
|-------|---------|-------|
| `np.where(c, a, b)` | `torch.where(c, a, b)` | `ops::where(c, a, b)` |
| `x[mask]` | `x[mask]` | `x.masked_select(mask)` |
| -- | `x.masked_fill_(m, v)` | `x.masked_fill(m, v)` |
| `x[x > 0]` | `x[x > 0]` | `x.masked_select(x > 0.0f)` |

### Linear Algebra

| NumPy | Axiom |
|-------|-------|
| `np.linalg.svd(A)` | `linalg::svd(A)` |
| `np.linalg.qr(A)` | `linalg::qr(A)` |
| `np.linalg.solve(A, b)` | `linalg::solve(A, b)` |
| `np.linalg.inv(A)` | `linalg::inv(A)` |
| `np.linalg.det(A)` | `linalg::det(A)` |
| `np.linalg.norm(A)` | `linalg::norm(A)` |
| `np.linalg.eig(A)` | `linalg::eig(A)` |
| `A @ B` | `A.matmul(B)` |
| `np.dot(a, b)` | `linalg::dot(a, b)` |

### Einops

| Python einops | Axiom |
|---------------|-------|
| `rearrange(x, 'b h w c -> b c h w')` | `x.rearrange("b h w c -> b c h w")` |
| `reduce(x, 'b h w c -> b c', 'mean')` | `x.reduce("b h w c -> b c", "mean")` |
| `einsum('ij,jk->ik', A, B)` | `einops::einsum("ij,jk->ik", {A, B})` |

### Device Management

| PyTorch | Axiom |
|---------|-------|
| `x.to("cuda")` | `x.gpu()` |
| `x.to("cpu")` | `x.cpu()` |
| `x.device` | `x.device()` |
| `torch.cuda.is_available()` | `system::is_metal_available()` |

### File I/O

| NumPy | Axiom |
|-------|-------|
| `np.save("f.npy", x)` | `x.save("f.axfb")` |
| `np.load("f.npy")` | `Tensor::load("f.npy")` |

Axiom can read `.npy` files directly and uses its own `.axfb` (FlatBuffers) format for saving.

## Patterns That Differ

### No Dynamic Shapes

Axiom shapes use `std::vector<size_t>`, passed as initializer lists:

```cpp
// NumPy: np.zeros((3, 4))
// Axiom:
auto x = Tensor::zeros({3, 4});
```

### Explicit Type Templates

Scalar extraction requires a type template:

```cpp
// NumPy: val = x.item()
// Axiom:
float val = x.item<float>();
```

### Structured Return Types

Decompositions return named structs instead of tuples:

```cpp
// NumPy: U, S, Vh = np.linalg.svd(A)
// Axiom:
auto result = linalg::svd(A);
auto U  = result.U;
auto S  = result.S;
auto Vh = result.Vh;

// Or with structured bindings (C++17):
auto [U, S, Vh] = linalg::svd(A);
```
