# Testing

Axiom uses a custom lightweight test framework. Tests are in `tests/` and use the `ASSERT()` macro and `RUN_TEST()` pattern.

## Running Tests

```bash
make test                           # Run all tests
make test-single TEST=tensor_basic  # Run one test
make test-verbose                   # Verbose output
make test-failed                    # Rerun failed tests
make test-list                      # List available tests
make test-debug                     # Run on debug build
```

## Test Structure

Each test file is a standalone executable. The entry point initializes the operation registry and runs test functions:

```cpp
#include "axiom_test_utils.hpp"
#include <axiom/axiom.hpp>

using namespace axiom;

void test_my_feature() {
    auto a = Tensor::zeros({3, 4});
    auto b = Tensor::ones({3, 4});
    auto c = a + b;

    ASSERT(c.shape() == Shape({3, 4}));
    ASSERT(c.dtype() == DType::Float32);
    ASSERT(c.allclose(Tensor::ones({3, 4})));
}

void test_edge_case() {
    auto empty = Tensor::zeros({0, 4});
    ASSERT(empty.empty());
    ASSERT(empty.size() == 0);
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    RUN_TEST(test_my_feature);
    RUN_TEST(test_edge_case);

    return 0;
}
```

## ASSERT Macro

`ASSERT(condition)` checks the condition and prints a diagnostic message on failure, including the file, line number, and expression:

```cpp
ASSERT(tensor.shape() == Shape({3, 4}));
ASSERT(tensor.allclose(expected, 1e-5, 1e-8));
ASSERT(tensor.dtype() == DType::Float32);
ASSERT(tensor.device() == Device::CPU);
ASSERT(!tensor.has_nan());
```

## GPU Tests

GPU tests check for Metal availability before running. Use `system::should_run_gpu_tests()` to skip gracefully:

```cpp
void test_gpu_matmul() {
    if (!system::should_run_gpu_tests()) return;

    auto a = Tensor::randn({64, 64}, DType::Float32, Device::GPU);
    auto b = Tensor::randn({64, 64}, DType::Float32, Device::GPU);
    auto c = a.matmul(b);

    ASSERT(c.device() == Device::GPU);
    ASSERT(c.shape() == Shape({64, 64}));
}
```

Set `AXIOM_SKIP_GPU_TESTS=1` to skip GPU tests in CI or on machines without Metal:

```bash
AXIOM_SKIP_GPU_TESTS=1 make test
```

## Adding a New Test File

1. Create `tests/test_my_feature.cpp` following the pattern above
2. Add it to `tests/CMakeLists.txt`:

```cmake
add_test_executable(test_my_feature test_my_feature.cpp)
```

3. Build and run:

```bash
make test-single TEST=test_my_feature
```

## Testing Patterns

### Numerical Comparison

Use `allclose` for floating-point comparisons:

```cpp
auto result = compute_something();
auto expected = Tensor::from_data(expected_data, {3, 3});

// Default tolerances: rtol=1e-5, atol=1e-8
ASSERT(result.allclose(expected));

// Custom tolerances for less precise operations
ASSERT(result.allclose(expected, /*rtol=*/1e-3, /*atol=*/1e-6));
```

### Shape Verification

```cpp
auto result = a.matmul(b);
ASSERT(result.shape() == Shape({M, N}));
ASSERT(result.ndim() == 2);
ASSERT(result.dtype() == DType::Float32);
```

### Exception Testing

```cpp
void test_invalid_reshape() {
    auto a = Tensor::zeros({3, 4});
    bool caught = false;
    try {
        a.reshape({5, 5});  // 12 != 25
    } catch (const ShapeError &) {
        caught = true;
    }
    ASSERT(caught);
}
```

### CPU-GPU Parity

Verify that CPU and GPU produce identical results:

```cpp
void test_add_cpu_gpu_parity() {
    if (!system::should_run_gpu_tests()) return;

    auto a = Tensor::randn({128, 128});
    auto b = Tensor::randn({128, 128});

    auto cpu_result = a + b;
    auto gpu_result = a.gpu() + b.gpu();

    ASSERT(cpu_result.allclose(gpu_result.cpu(), 1e-5, 1e-6));
}
```

## Test Organization

| File | Coverage |
|------|----------|
| `test_broadcast.cpp` | Broadcasting rules |
| `test_custom_functors.cpp` | apply, vectorize, apply_along_axis |
| `test_einops_reduce.cpp` | Einops rearrange and reduce |
| `test_einsum.cpp` | Einstein summation |
| `test_fft.cpp` | FFT operations |
| `test_fusion.cpp` | Operation fusion patterns |
| `test_graph_compiler.cpp` | Graph compilation and execution |
| `test_io_flatbuffers.cpp` | .axfb format I/O |
| `test_io_numpy.cpp` | .npy format I/O |
| `test_lazy_evaluation.cpp` | Lazy evaluation and materialization |
| `test_parallel.cpp` | OpenMP parallelization |
| `test_pooling.cpp` | Pooling operations |
| `test_cpu_gpu_parity.cpp` | CPU-GPU result agreement |
| `test_advanced_indexing.cpp` | Gather, scatter, index_select |
