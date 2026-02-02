# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
make release              # Build optimized release (default)
make debug                # Build debug version
make test                 # Run all tests
make test-single TEST=tensor_basic  # Run single test
make format               # Format all source files
make lint                 # Run clang-tidy static analysis
make compile-commands     # Generate compile_commands.json for IDE integration
make ci                   # Full CI pipeline (format-check + build + test)
```

## Architecture

Axiom is a C++20 tensor library optimized for Apple Silicon with Metal GPU acceleration and NumPy compatibility.

### Core Components

- **include/axiom/tensor.hpp** - Main Tensor class with shape, strides, dtype, and storage
- **include/axiom/operations.hpp** - Operation registry and 50+ operation type definitions
- **src/tensor/operations.cpp** - High-level operation APIs
- **src/backends/cpu/cpu_operations.cpp** - CPU kernels with SIMD (xsimd)
- **src/backends/metal/mpsgraph_operations.mm** - GPU kernels via MPSGraph

### Backend System

Two-level dispatch: OpType enum → Backend registry (CPU/GPU). Tensor operations automatically dispatch based on device.

**BLAS backends** (auto-detected):
- macOS: Accelerate framework
- Linux: OpenBLAS (fallback to native)
- Windows: Native implementation

### Memory Model

- Storage abstraction with CPU and Metal (GPU) backends
- Zero-copy views via strides and offset
- Automatic CPU ↔ GPU transfers
- Shared storage mode on Apple Silicon (unified memory)

### Key Patterns

**Adding a new operation:**
1. Define OpType enum in operations.hpp
2. Implement CPU kernel in cpu_operations.cpp
3. Implement GPU kernel in mpsgraph_operations.mm (optional)
4. Add high-level API in tensor/operations.cpp
5. Declare in operations.hpp
6. Add tests
7. Update docs/ops.md

## Code Style

- 80-column limit (clang-format configured)
- Classes/Structs: `CamelCase`
- Functions/Methods: `snake_case`
- Private members: suffix with `_`
- C++20 features, prefer `auto` and `constexpr`

## Testing

Tests are in `tests/`. Use `ASSERT()` macro and `RUN_TEST()` pattern:

```cpp
ops::OperationRegistry::initialize_builtin_operations();
RUN_TEST(test_my_feature);
```

Environment: `AXIOM_SKIP_GPU_TESTS=1` skips GPU tests (used in CI).

## Platform Notes

- macOS 11+, Xcode 13+, CMake 3.20+ required
- Metal GPU support is macOS-only
- GPU tests may be unstable in CI environments
