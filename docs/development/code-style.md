# Code Style

## Formatting

Axiom uses **clang-format** with an 80-column limit. The configuration is in `.clang-format` at the repository root.

```bash
make format          # Format all files
make format-check    # Check without modifying (used in CI)
make format-diff     # Show what would change
```

Formatting is enforced in CI -- PRs with formatting violations will fail the `format-check` step.

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes / Structs | `CamelCase` | `GraphNode`, `OperationRegistry` |
| Functions / Methods | `snake_case` | `execute_binary`, `is_contiguous` |
| Private members | `snake_case_` (trailing underscore) | `storage_`, `lazy_node_` |
| Constants | `UPPER_SNAKE_CASE` or `kCamelCase` | `DEFAULT_MIN_ELEMENTS`, `kMaxDims` |
| Namespaces | `lower_case` | `axiom`, `ops`, `linalg`, `fft` |
| Enum values | `CamelCase` | `DType::Float32`, `OpType::MatMul` |
| Template parameters | `CamelCase` | `template <typename T>` |
| File names | `snake_case` | `cpu_operations.cpp`, `graph_node.hpp` |

## C++20 Features

Axiom targets C++20. Preferred patterns:

- **`auto`**: Use for local variables when the type is obvious from context
- **`constexpr`**: Prefer for compile-time constants and functions
- **Structured bindings**: `auto [U, S, Vh] = linalg::svd(A);`
- **`std::shared_ptr`**: For shared storage ownership
- **`std::unique_ptr`**: For exclusive ownership (operations, arenas)
- **Range-based for**: Prefer over index-based iteration
- **`[[maybe_unused]]`**: For intentionally unused parameters
- **`if constexpr`**: For compile-time branching in templates

Avoid:

- `std::bind` (use lambdas)
- Raw `new`/`delete` (use smart pointers)
- C-style casts (use `static_cast`, `reinterpret_cast`)

## Header Organization

```cpp
#pragma once                           // Include guard

#include <standard_library_headers>    // System headers first
#include "axiom/local_headers.hpp"     // Project headers second

namespace axiom {

// Forward declarations
class Tensor;

// Constants
constexpr size_t DEFAULT_SIZE = 1024;

// Classes and functions
class MyClass {
public:
    // Public interface first
    void public_method();

private:
    // Private implementation
    int member_;
};

} // namespace axiom
```

## Static Analysis

Run clang-tidy for additional checks:

```bash
make lint
```

This checks for common issues like unused variables, performance anti-patterns, and potential bugs.

## Comments

- Add comments only where the logic is not self-evident
- Use `//` for single-line comments, `/* */` for block comments
- Document public APIs with brief Doxygen-style `/** */` comments in headers
- Avoid redundant comments that just restate the code

```cpp
// Good: explains WHY
// Use int64_t for byte offsets to handle negative strides from flip()
int64_t byte_offset = 0;

// Bad: restates WHAT
// Increment the counter
counter++;
```
