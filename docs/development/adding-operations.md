# Adding Operations

This guide walks through adding a new operation to Axiom end-to-end.

## Overview

Adding an operation involves six steps:

1. Define the `OpType` enum value
2. Implement the CPU kernel
3. Implement the GPU kernel (optional)
4. Add the high-level API
5. Declare in the public header
6. Add tests

## Step 1: Define OpType

Add your operation to the `OpType` enum in `include/axiom/operations.hpp`:

```cpp
enum class OpType {
    // ... existing ops ...

    // Your new operation
    MyNewOp,
};
```

## Step 2: Implement the CPU Kernel

Add a CPU implementation in `src/backends/cpu/cpu_operations.cpp`.

For a **unary** operation, implement the `execute_unary` method:

```cpp
class CpuMyNewOp : public Operation {
public:
    OpType type() const override { return OpType::MyNewOp; }
    std::string name() const override { return "MyNewOp"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_unary(const Tensor &input) const override {
        auto result = Tensor(input.shape(), input.dtype(), Device::CPU);
        // Implementation using typed_data pointers
        const float *in = input.typed_data<float>();
        float *out = result.typed_data<float>();
        for (size_t i = 0; i < input.size(); ++i) {
            out[i] = my_function(in[i]);
        }
        return result;
    }
};
```

For a **binary** operation, implement `execute_binary`:

```cpp
Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
    // Handle broadcasting
    auto info = compute_broadcast_info(lhs.shape(), rhs.shape());
    auto result = Tensor(info.result_shape, lhs.dtype(), Device::CPU);
    // ... implementation ...
    return result;
}
```

Register the CPU implementation in `initialize_builtin_operations()`:

```cpp
register_operation(OpType::MyNewOp, Device::CPU,
                   std::make_unique<CpuMyNewOp>());
```

## Step 3: Implement the GPU Kernel (Optional)

Add a Metal implementation in `src/backends/metal/mpsgraph_operations.mm`. GPU kernels use MPSGraph:

```objc
class MetalMyNewOp : public Operation {
public:
    OpType type() const override { return OpType::MyNewOp; }
    std::string name() const override { return "MyNewOp"; }
    Device device() const override { return Device::GPU; }

    Tensor execute_unary(const Tensor &input) const override {
        // Build MPSGraph computation
        // ...
    }
};
```

Register in the Metal backend's initialization.

## Step 4: Add the High-Level API

Add a user-facing function in `src/tensor/operations.cpp`:

```cpp
Tensor my_new_op(const Tensor &input) {
    return execute_unary_operation(OpType::MyNewOp, input);
}
```

This handles lazy evaluation, device dispatch, and type promotion automatically through the `execute_unary_operation` helper.

## Step 5: Declare in Public Header

Add the function declaration in `include/axiom/operations.hpp`:

```cpp
// In the ops namespace, with other operation declarations:
Tensor my_new_op(const Tensor &input);
```

If appropriate, add a fluent method to `include/axiom/tensor.hpp`:

```cpp
class Tensor {
    // ...
    Tensor my_new_op() const;
};
```

And implement it in `src/tensor/tensor.cpp`:

```cpp
Tensor Tensor::my_new_op() const {
    return ops::my_new_op(*this);
}
```

## Step 6: Add Tests

Create `tests/test_my_new_op.cpp`:

```cpp
#include "axiom_test_utils.hpp"
#include <axiom/axiom.hpp>

using namespace axiom;

void test_basic() {
    auto input = Tensor::randn({3, 4});
    auto result = ops::my_new_op(input);

    ASSERT(result.shape() == input.shape());
    ASSERT(result.dtype() == input.dtype());
    // Verify correctness against expected values
}

void test_dtypes() {
    // Test with different dtypes
    for (auto dtype : {DType::Float32, DType::Float64, DType::Int32}) {
        auto input = Tensor::ones({4, 4}, dtype);
        auto result = ops::my_new_op(input);
        ASSERT(result.dtype() == dtype);
    }
}

void test_gpu() {
    if (!system::should_run_gpu_tests()) return;

    auto input = Tensor::randn({64, 64}, DType::Float32, Device::GPU);
    auto result = ops::my_new_op(input);

    ASSERT(result.device() == Device::GPU);

    // CPU-GPU parity
    auto cpu_result = ops::my_new_op(input.cpu());
    ASSERT(cpu_result.allclose(result.cpu()));
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();
    RUN_TEST(test_basic);
    RUN_TEST(test_dtypes);
    RUN_TEST(test_gpu);
    return 0;
}
```

Add to `tests/CMakeLists.txt`:

```cmake
add_test_executable(test_my_new_op test_my_new_op.cpp)
```

## Step 7: Update Documentation

Add the operation to `docs/ops.md` and the appropriate API reference page under `docs/api/`.

## Lazy Evaluation Integration

The `execute_unary_operation` and `execute_binary_operation` helpers automatically integrate with lazy evaluation. For element-wise operations, add your `OpType` to the appropriate classifier in `include/axiom/graph/graph_node.hpp`:

```cpp
inline bool is_elementwise_op(ops::OpType op) {
    switch (op) {
    // ... existing ops ...
    case OpType::MyNewOp:
        return true;
    }
}

inline bool is_unary_op(ops::OpType op) {
    switch (op) {
    // ... existing ops ...
    case OpType::MyNewOp:
        return true;
    }
}
```

This enables your operation to participate in operation fusion.

## Checklist

- [ ] `OpType` enum value added
- [ ] CPU kernel implemented and registered
- [ ] GPU kernel implemented (if applicable)
- [ ] High-level API in `operations.cpp`
- [ ] Declaration in `operations.hpp`
- [ ] Fluent method on Tensor (if appropriate)
- [ ] Graph node classifiers updated (if element-wise)
- [ ] Tests with shape/dtype/device coverage
- [ ] `docs/ops.md` updated
- [ ] API reference page updated
