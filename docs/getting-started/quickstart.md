# Quickstart

This guide walks you through creating tensors, performing operations, and using GPU acceleration.

## Include Axiom

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;
```

Before using any operations, initialize the operation registry:

```cpp
ops::OperationRegistry::initialize_builtin_operations();
```

## Creating Tensors

```cpp
// Zeros and ones
auto a = Tensor::zeros({3, 4});          // 3x4 float32 tensor
auto b = Tensor::ones({3, 4});           // 3x4 of ones

// Specific values
auto c = Tensor::full({2, 3}, 3.14f);   // Filled with 3.14
auto d = Tensor::eye(4);                 // 4x4 identity matrix

// Ranges
auto e = Tensor::arange(10);             // [0, 1, 2, ..., 9]
auto f = Tensor::linspace(0.0, 1.0, 5); // [0.0, 0.25, 0.5, 0.75, 1.0]

// Random
auto g = Tensor::randn({3, 3});          // Normal distribution
auto h = Tensor::rand({3, 3});           // Uniform [0, 1)

// From existing data
float data[] = {1, 2, 3, 4, 5, 6};
auto i = Tensor::from_data(data, {2, 3});
```

## Basic Operations

```cpp
auto x = Tensor::randn({3, 4});
auto y = Tensor::randn({3, 4});

// Arithmetic (with operator overloads)
auto sum  = x + y;
auto diff = x - y;
auto prod = x * y;
auto quot = x / y;

// Scalar operations
auto scaled = x * 2.0f + 1.0f;

// Math functions (fluent API)
auto result = x.abs().sqrt().exp();

// Or functional style
auto result2 = ops::exp(ops::sqrt(ops::abs(x)));
```

## Reductions

```cpp
auto x = Tensor::randn({4, 5});

auto total = x.sum();           // Sum all elements
auto col_sum = x.sum(0);        // Sum along axis 0 -> shape (5,)
auto row_mean = x.mean(1);      // Mean along axis 1 -> shape (4,)

auto max_val = x.max();         // Global max
auto min_idx = x.argmin(1);     // Indices of min per row
```

## Matrix Multiplication

```cpp
auto A = Tensor::randn({64, 128});
auto B = Tensor::randn({128, 32});
auto C = A.matmul(B);  // Shape: (64, 32)

// Batched matmul
auto batch_A = Tensor::randn({10, 64, 128});
auto batch_B = Tensor::randn({10, 128, 32});
auto batch_C = batch_A.matmul(batch_B);  // Shape: (10, 64, 32)
```

## Shape Manipulation

```cpp
auto x = Tensor::arange(12);

auto reshaped = x.reshape({3, 4});    // View if possible
auto transposed = reshaped.T();       // Transpose -> (4, 3)
auto flat = reshaped.flatten();       // Back to 1D

// Einops-style reshape
auto img = Tensor::randn({2, 32, 32, 3});
auto chw = img.rearrange("b h w c -> b c h w");
```

## GPU Acceleration

```cpp
// Move to GPU
auto cpu_tensor = Tensor::randn({1024, 1024});
auto gpu_tensor = cpu_tensor.gpu();

// Operations run on GPU automatically
auto result = gpu_tensor.matmul(gpu_tensor.T());

// Move back to CPU
auto cpu_result = result.cpu();

// Create directly on GPU
auto gpu_ones = Tensor::ones({256, 256}, DType::Float32, Device::GPU);
```

## Linear Algebra

```cpp
auto A = Tensor::randn({4, 4});

// Decompositions
auto [U, S, Vh] = linalg::svd(A);
auto [Q, R] = linalg::qr(A);

// Solve linear system Ax = b
auto b = Tensor::randn({4});
auto x = linalg::solve(A, b);

// Inverse and determinant
auto inv = linalg::inv(A);
auto det = linalg::det(A);
```

## Saving and Loading

```cpp
// Save single tensor
auto tensor = Tensor::randn({100, 100});
tensor.save("my_tensor.axfb");

// Load it back
auto loaded = Tensor::load("my_tensor.axfb");

// Save multiple tensors
Tensor::save_tensors({{"weights", w}, {"bias", b}}, "model.axfb");
auto model = Tensor::load_tensors("model.axfb");
```

## Printing Tensors

```cpp
auto x = Tensor::arange(6).reshape({2, 3});
std::cout << x << std::endl;
// Tensor(shape=[2, 3], dtype=Int32, device=CPU)
// [[0, 1, 2],
//  [3, 4, 5]]

// Detailed debug info
std::cout << x.debug_info() << std::endl;
```

## Next Steps

- [User Guide: Tensor Basics](../user-guide/tensor-basics) -- Deep dive into tensors, dtypes, and memory
- [User Guide: GPU Acceleration](../user-guide/gpu-acceleration) -- Metal GPU tips and best practices
- [API Reference](../api/index) -- Complete function signatures
- [NumPy Migration Guide](numpy-users) -- Coming from NumPy or PyTorch?
