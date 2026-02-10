# Backend Architecture

```{note}
This page expands on the original [Backend Architecture](../BACKEND_ARCHITECTURE) document with additional detail on the dispatch system and graph compilation.
```

## Overview

Axiom uses a modular backend system with two levels of abstraction:

1. **Storage layer** -- Abstract memory management (`Storage` interface)
2. **Operation layer** -- Kernel dispatch (`OperationRegistry`)

Each backend implements both layers for its target device.

## Current Backends

### CPU Backend (`src/backends/cpu/`)

- **Storage**: `cpu_storage.hpp/cpp` -- System memory via `malloc`/`new`
- **Operations**: `cpu_operations.cpp` -- SIMD-accelerated kernels via xsimd
- **Always available** on all platforms

### Metal Backend (`src/backends/metal/`)

- **Storage**: `metal_storage.hpp/mm` -- MTLBuffer with shared storage mode
- **Operations**: `mpsgraph_operations.mm` -- MPSGraph-based GPU kernels
- **macOS only** (requires Metal and MetalPerformanceShadersGraph)

## Storage Interface

All backends implement the abstract `Storage` class:

```cpp
class Storage {
public:
    virtual void *data() = 0;
    virtual size_t size_bytes() const = 0;
    virtual Device device() const = 0;
    virtual void copy_to(Storage &other) = 0;
    virtual void copy_from(Storage &other) = 0;
    virtual std::unique_ptr<Storage> clone() = 0;
    virtual bool is_view() const = 0;
    virtual std::shared_ptr<Storage> base() = 0;
};
```

The `make_storage()` factory function creates the appropriate storage type based on the target device.

## Operation Dispatch

### OpType Enum

Every operation has a canonical identifier in the `ops::OpType` enum (~70 operations):

```
Binary: Add, Subtract, Multiply, Divide, Power, Modulo
Comparison: Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual
Logical: LogicalAnd, LogicalOr, LogicalXor, LogicalNot
Bitwise: BitwiseAnd, BitwiseOr, BitwiseXor, LeftShift, RightShift
Unary: Negate, Abs, Sqrt, Exp, Log, Sin, Cos, Tan, Erf, Sign, ...
Activation: ReLU, LeakyReLU, SiLU, Sigmoid, Tanh, GELU, Softmax, ...
Reduction: Sum, Mean, Max, Min, ArgMax, ArgMin, Prod, Any, All
Matrix: MatMul, BatchMatMul
Conditional: Where, MaskedFill, MaskedSelect
Indexing: Gather, Scatter, IndexSelect, Take, TakeAlongAxis
Normalization: LayerNorm, RMSNorm, Dropout
Pooling: MaxPool1D/2D/3D, AvgPool1D/2D/3D, AdaptiveMaxPool2D, ...
```

### OperationRegistry

The registry maps `(OpType, Device)` pairs to concrete `Operation` implementations:

```cpp
OperationRegistry::register_operation(OpType::Add, Device::CPU,
                                       std::make_unique<CpuAdd>());
OperationRegistry::register_operation(OpType::Add, Device::GPU,
                                       std::make_unique<MetalAdd>());
```

At runtime, dispatch looks up the implementation:

```cpp
const Operation *op = OperationRegistry::get_operation(OpType::Add, device);
return op->execute_binary(lhs, rhs);
```

If a GPU implementation is not available, the system falls back to CPU automatically.

### Operation Base Class

Each kernel implements the appropriate virtual method:

```cpp
class Operation {
    virtual Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const;
    virtual Tensor execute_unary(const Tensor &input) const;
    virtual Tensor execute_reduction(const Tensor &input,
                                     const std::vector<int> &axis,
                                     bool keep_dims) const;
    virtual Tensor execute_matmul(const Tensor &a, const Tensor &b,
                                  bool transpose_a, bool transpose_b) const;
    virtual Tensor execute_where(const Tensor &condition,
                                 const Tensor &a, const Tensor &b) const;
    // ... and more
};
```

## Memory Model

### Zero-Copy Views

Tensors share storage through the `std::shared_ptr<Storage>` member. Views adjust shape, strides, and offset without allocating new memory:

```
Tensor A: storage=0x100, shape={4,4}, strides={16,4}, offset=0
Tensor B: storage=0x100, shape={2,4}, strides={16,4}, offset=32  (slice)
Tensor C: storage=0x100, shape={4,4}, strides={4,16}, offset=0   (transpose)
```

### Cross-Device Transfers

CPU and GPU storage types can copy data between each other via `copy_to`/`copy_from`. On Apple Silicon, the shared memory mode means CPU-accessible GPU buffers avoid redundant copies.

### Lazy Evaluation Graph

Operations can build a computation graph (`GraphNode` DAG) instead of executing immediately. The graph compiler then:

1. Topologically sorts nodes
2. Fuses element-wise chains
3. Plans memory reuse (buffer slot lifetimes)
4. Compiles into `ExecutionStep` sequences
5. Caches compiled plans by `GraphSignature`

See [Lazy Evaluation](../user-guide/lazy-evaluation) for details.

## Thread Safety

| Component | Thread Safety |
|-----------|--------------|
| `OperationRegistry` | Thread-safe (static registry) |
| Factory functions (`make_storage`) | Thread-safe |
| `Storage` objects | Not thread-safe (by design) |
| Cross-device copies | Synchronous, thread-safe |
| Graph cache | Thread-safe (mutex-protected LRU) |
| Arena pool | Thread-safe (per-graph mutex) |
| Memory views | Shared data, requires external synchronization |

## Adding a New Backend

To add a new backend (e.g., CUDA, Vulkan):

1. Create `src/backends/my_backend/`
2. Implement `Storage` subclass (`my_storage.hpp/cpp`)
3. Add `Device` enum value in `include/axiom/storage.hpp`
4. Update `make_storage()` factory in `src/storage/storage.cpp`
5. Implement `Operation` subclasses for supported ops
6. Register operations in `initialize_builtin_operations()`
7. Add CMake integration in `CMakeLists.txt`
8. Add tests for CPU-GPU parity

**See Also:** [Adding Operations](adding-operations), [User Guide: GPU Acceleration](../user-guide/gpu-acceleration)
