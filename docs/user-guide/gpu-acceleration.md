# GPU Acceleration

Axiom provides GPU acceleration on macOS through Apple's Metal framework. Tensor operations dispatch to Metal Performance Shaders Graph (MPSGraph) kernels, taking advantage of the GPU cores and unified memory architecture on Apple Silicon. No additional drivers or toolkits are required -- Metal support is built into macOS.

## Checking Availability

Before using GPU features, verify that the current system has a Metal-capable device:

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;

if (system::is_metal_available()) {
    std::cout << "Metal GPU is available" << std::endl;
} else {
    std::cout << "No Metal device found; running on CPU" << std::endl;
}
```

`system::is_metal_available()` returns `true` when a Metal-capable GPU is present. It always returns `false` on non-Apple platforms (Linux, Windows).

**Platform requirements:**

- macOS 11 (Big Sur) or later
- Xcode 13 or later (for Metal and MPSGraph headers)
- Any Mac with Apple Silicon (M1/M2/M3/M4) or a supported AMD GPU

## Device Placement

Create tensors directly on the GPU by passing `Device::GPU` to factory methods:

```cpp
auto a = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
auto b = Tensor::ones({256, 256}, DType::Float32, Device::GPU);
auto c = Tensor::randn({1024, 1024}, DType::Float32, Device::GPU);
auto d = Tensor::eye(128, DType::Float32, Device::GPU);
```

You can check which device a tensor lives on with the `device()` method:

```cpp
auto t = Tensor::zeros({3, 4}, DType::Float32, Device::GPU);
std::cout << system::device_to_string(t.device()) << std::endl;  // "GPU"
```

## Transferring Between Devices

Move tensors between CPU and GPU with the convenience methods `.cpu()` and `.gpu()`, or the general-purpose `.to()`:

```cpp
auto cpu_tensor = Tensor::randn({512, 512});

// CPU -> GPU
auto gpu_tensor = cpu_tensor.gpu();
auto gpu_tensor2 = cpu_tensor.to(Device::GPU);

// GPU -> CPU
auto back_on_cpu = gpu_tensor.cpu();
auto back_on_cpu2 = gpu_tensor.to(Device::CPU);
```

Both `.cpu()` and `.gpu()` are no-ops if the tensor is already on the target device, so they are safe to call unconditionally.

### Automatic Transfers

When an operation receives operands on different devices, Axiom automatically transfers them to a common device before computing. This is convenient for quick experiments but adds transfer overhead:

```cpp
auto a = Tensor::randn({256, 256});                             // CPU
auto b = Tensor::randn({256, 256}, DType::Float32, Device::GPU); // GPU

// Axiom transfers `a` to GPU, then computes on GPU
auto c = a + b;
std::cout << system::device_to_string(c.device()) << std::endl;  // "GPU"
```

For performance-critical code, place all tensors on the same device explicitly rather than relying on automatic transfers.

## Unified Memory on Apple Silicon

Apple Silicon uses a unified memory architecture: CPU and GPU share the same physical memory. Axiom's `MetalStorage` allocates GPU buffers with `MTLResourceStorageModeShared`, which means the buffer is accessible from both CPU and GPU without an explicit copy.

In practice, this means:

- Device transfers between CPU and GPU are significantly cheaper than on discrete-GPU systems.
- When a GPU tensor is moved to CPU (or vice versa), the underlying data may already reside in the same physical memory region.
- Small tensors can be transferred with minimal overhead thanks to shared memory, reducing the penalty for mixed-device workflows.

You do not need to manage unified memory explicitly. Axiom handles buffer allocation and synchronization automatically.

## Supported Operations

Most built-in operations are implemented on the GPU via MPSGraph kernels, including:

- **Arithmetic:** add, subtract, multiply, divide, power, modulo
- **Unary math:** abs, sqrt, exp, log, sin, cos, tan, sign, floor, ceil, reciprocal, square, cbrt, negative
- **Comparisons:** equal, not_equal, less, less_equal, greater, greater_equal
- **Reductions:** sum, mean, max, min, argmax, argmin, prod, any, all
- **Matrix operations:** matmul (including batched)
- **Activations:** relu, leaky_relu, gelu, silu, sigmoid, tanh, softmax, log_softmax
- **Element-wise:** clip/clamp, where, masked_fill, isnan, isinf, isfinite

Operations on GPU tensors dispatch to Metal automatically -- no special syntax is needed:

```cpp
auto x = Tensor::randn({1024, 1024}, DType::Float32, Device::GPU);
auto y = Tensor::randn({1024, 1024}, DType::Float32, Device::GPU);

auto z = (x.matmul(y) + x).relu().softmax(1);  // Entire chain runs on GPU
```

## CPU-Only Operations

Some operations are only available on the CPU backend. When called on a GPU tensor, these operations automatically transfer the data to CPU, compute, and return a CPU result:

- **Linear algebra:** `linalg::svd`, `linalg::qr`, `linalg::eig`, `linalg::solve`, `linalg::inv`, `linalg::det`, and other `linalg::*` functions
- **FFT:** `fft::fft`, `fft::ifft`, `fft::fft2`, `fft::rfft`, and other `fft::*` functions
- **Einops:** `einops::rearrange`, `einops::reduce`, `einops::repeat`
- **Custom functors:** `ops::apply`, `ops::vectorize`, `ops::apply_along_axis`
- **File I/O:** `tensor.save()`, `Tensor::load()`

If you need the result back on GPU after a CPU-only operation, transfer it explicitly:

```cpp
auto gpu_matrix = Tensor::randn({128, 128}, DType::Float32, Device::GPU);

// SVD runs on CPU; result tensors are on CPU
auto [U, S, Vh] = linalg::svd(gpu_matrix.cpu());

// Move results back to GPU for further computation
auto U_gpu = U.gpu();
auto S_gpu = S.gpu();
```

## Performance Tips

**Keep data on GPU.** Every CPU-GPU transfer has overhead. Structure your pipeline so that tensors are created on GPU and stay there through as many operations as possible.

```cpp
// Good: create on GPU, compute on GPU, transfer once at the end
auto x = Tensor::randn({2048, 2048}, DType::Float32, Device::GPU);
auto y = Tensor::randn({2048, 2048}, DType::Float32, Device::GPU);
auto result = x.matmul(y).relu().sum();
float value = result.cpu().item<float>();
```

**Batch operations to amortize dispatch overhead.** Each GPU operation has a fixed dispatch cost. Combining work into fewer, larger operations is more efficient than many small ones.

**Use lazy evaluation for fused GPU execution.** Axiom's lazy evaluation mode can fuse multiple operations into a single GPU dispatch, reducing kernel launch overhead and intermediate memory allocations:

```cpp
auto x = Tensor::randn({1024, 1024}, DType::Float32, Device::GPU);
auto y = x.relu().sqrt().exp();  // May be fused into a single GPU kernel
```

**Large tensors benefit most from GPU.** The GPU excels at data-parallel workloads. Matrix multiplications, large element-wise operations, and batch computations see the greatest speedups.

**Small tensors may be faster on CPU.** For tensors with fewer than a few thousand elements, the overhead of GPU kernel dispatch can exceed the computation time. Keep small tensors on CPU.

```cpp
// Small tensor -- CPU is likely faster
auto small = Tensor::randn({4, 4});
auto det = linalg::det(small);

// Large tensor -- GPU shines here
auto large = Tensor::randn({2048, 2048}, DType::Float32, Device::GPU);
auto product = large.matmul(large.T());
```

## Error Handling

Axiom raises `DeviceError` exceptions for GPU-related problems:

- **`DeviceError::not_available`** -- Thrown when attempting to use `Device::GPU` on a system without Metal support (e.g., calling `.gpu()` on Linux).
- **`DeviceError::cpu_only`** -- Thrown when an operation requires direct CPU data access on a GPU tensor (e.g., `typed_data<T>()`, `item()` with indices, `fill()`, `set_item()`).

```cpp
try {
    auto t = Tensor::randn({3, 3}, DType::Float32, Device::GPU);
    float* ptr = t.typed_data<float>();  // Throws DeviceError::cpu_only
} catch (const DeviceError& e) {
    std::cerr << e.what() << std::endl;
}
```

To safely access element data from a GPU tensor, transfer it to CPU first:

```cpp
auto gpu_t = Tensor::randn({3, 3}, DType::Float32, Device::GPU);
auto cpu_t = gpu_t.cpu();
float val = cpu_t.item<float>({0, 0});  // Safe
```

For complete API details, see [API Reference: Devices & Storage](../api/devices-and-storage) and [API Reference: System](../api/system).
