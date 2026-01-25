# Contributing to Axiom

Thank you for your interest in contributing to Axiom! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/axiom.git`
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Run tests: `make test`
6. Commit your changes: `git commit -am 'Add my feature'`
7. Push to your fork: `git push origin feature/my-feature`
8. Create a Pull Request

## Development Setup

### Prerequisites

- macOS 11.0+ (Big Sur or later)
- Xcode 13+ with Command Line Tools
- CMake 3.20+
- Ninja (recommended): `brew install ninja`
- clang-format and clang-tidy (for code quality): `brew install llvm`

### Building

```bash
# Release build
make release

# Debug build
make debug

# Run tests
make test

# Clean
make clean
```

## Code Style

We use clang-format for consistent code formatting. The configuration is in `.clang-format`.

### Formatting Your Code

```bash
# Format all source files
make format

# Check formatting without modifying
find include src -name '*.hpp' -o -name '*.cpp' | xargs clang-format --dry-run --Werror
```

### Style Guidelines

1. **Naming Conventions**:
   - Classes/Structs: `CamelCase` (e.g., `Tensor`, `ShapeUtils`)
   - Functions/Methods: `snake_case` (e.g., `execute_binary`, `get_shape`)
   - Variables: `snake_case` (e.g., `input_data`, `result_tensor`)
   - Constants: `snake_case` or `UPPER_CASE` for macros
   - Private members: suffix with `_` (e.g., `shape_`, `data_`)

2. **File Organization**:
   - Headers in `include/axiom/`
   - Implementation in `src/`
   - Tests in `tests/`
   - One class per file when practical

3. **Modern C++**:
   - Use C++20 features where appropriate
   - Prefer `auto` for complex types
   - Use `constexpr` where possible
   - Use smart pointers over raw pointers

## Testing

All new features must include tests. Tests are located in `tests/`.

### Writing Tests

```cpp
#include <axiom/axiom.hpp>
#include <cassert>

void test_my_feature() {
    // Setup
    auto input = Tensor::ones({2, 3}, DType::Float32, Device::CPU);
    
    // Execute
    auto result = ops::my_operation(input);
    
    // Verify
    ASSERT(result.shape() == Shape({2, 3}), "Shape mismatch");
    // More assertions...
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();
    
    RUN_TEST(test_my_feature);
    
    // Return 0 if all tests pass
    return (tests_passed == tests_run) ? 0 : 1;
}
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test
./build/tests/test_tensor_basic

# Run with verbose output
cd build && ctest --output-on-failure --verbose
```

## Adding New Operations

1. **Define the operation type** in `include/axiom/operations.hpp`:
   ```cpp
   enum class OpType {
       // ... existing ops ...
       MyNewOp,
   };
   ```

2. **Add the CPU implementation** in `src/backends/cpu/cpu_operations.cpp`:
   ```cpp
   struct MyNewOpFunc {
       template <typename T>
       T operator()(const T &a, const T &b) const {
           return /* computation */;
       }
   };
   
   // In register_cpu_operations():
   OperationRegistry::register_operation(
       OpType::MyNewOp, Device::CPU,
       std::make_unique<CPUBinaryOperation<MyNewOpFunc>>(
           OpType::MyNewOp, "my_new_op", MyNewOpFunc{}));
   ```

3. **Add the GPU implementation** in `src/backends/metal/mpsgraph_operations.mm`:
   ```objc
   static MPSGraphTensor* my_new_op(MPSGraph* graph, MPSGraphTensor* a, MPSGraphTensor* b) {
       return [graph /* MPSGraph method */ :a secondaryTensor:b name:nil];
   }
   
   // In register_mpsgraph_operations():
   OperationRegistry::register_operation(OpType::MyNewOp, Device::GPU,
       std::make_unique<MPSGraphBinaryOperation>(OpType::MyNewOp, "my_new_op", my_new_op));
   ```

4. **Add the high-level API** in `src/tensor/operations.cpp`:
   ```cpp
   Tensor my_new_op(const Tensor &a, const Tensor &b) {
       return execute_binary_operation(OpType::MyNewOp, a, b);
   }
   ```

5. **Declare the API** in `include/axiom/operations.hpp`:
   ```cpp
   Tensor my_new_op(const Tensor &a, const Tensor &b);
   ```

6. **Add tests** in `tests/test_tensor_operations.cpp` or create a new test file.

7. **Update documentation** in `docs/ops.md`.

## Pull Request Guidelines

1. **One feature per PR** - Keep PRs focused and manageable
2. **Write descriptive commits** - Use conventional commit messages
3. **Include tests** - All new features must have tests
4. **Update docs** - Update relevant documentation
5. **Pass CI** - Ensure all CI checks pass

### Commit Message Format

```
type: short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Reporting Issues

When reporting issues, please include:

1. macOS version
2. Xcode/Clang version
3. Axiom version or commit hash
4. Minimal reproduction code
5. Expected vs actual behavior
6. Any error messages

## Questions?

Feel free to open an issue for questions or discussions about the project.
