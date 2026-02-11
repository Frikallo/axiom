# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
make release          # Build release (default)
make debug            # Build debug
make test             # Run all tests
make test-single TEST=tensor_basic  # Run one test suite
make format           # Format all code (clang-format)
make format-check     # Check formatting without modifying
make lint             # Static analysis (clang-tidy)
make ci               # Full CI: format-check → build → test
```

Tests use Google Test. Test binaries are in `build/tests/`. You can also run via ctest:
```bash
cd build && ctest -R "TestName" --output-on-failure
```

## Architecture

Axiom is a C++20 high-performance tensor library with NumPy/PyTorch-compatible API, SIMD vectorization (via Google Highway), BLAS acceleration, and Metal GPU support on macOS.

### Core Data Flow

**Tensor** (`include/axiom/tensor.hpp`) holds a `shared_ptr<Storage>` (CPU or GPU), shape, strides, dtype, and an optional `lazy_node_` for deferred computation. Tensors support zero-copy views via custom strides.

**Two execution modes:**
1. **Eager** — operations execute immediately through the operation registry
2. **Lazy** — operations build a `GraphNode` DAG, compiled and executed when materialized

### Key Subsystems

| Subsystem | Headers | Implementation |
|-----------|---------|----------------|
| Tensor core | `include/axiom/tensor.hpp` | `src/tensor/tensor.cpp` |
| Operations | `include/axiom/operations.hpp` | `src/tensor/operations.cpp` |
| CPU backend | `src/backends/cpu/cpu_operations.hpp` | `src/backends/cpu/cpu_operations.cpp` |
| GPU backend | `src/backends/metal/mpsgraph_operations.hpp` | `src/backends/metal/mpsgraph_operations.mm` |
| Lazy eval / fusion | `include/axiom/graph/*.hpp` | `src/graph/*.cpp` |
| Linear algebra | `include/axiom/linalg.hpp` | `src/tensor/linalg.cpp` |
| Einops | `include/axiom/einops.hpp` | `src/tensor/einops.cpp` |
| FFT | `include/axiom/fft.hpp` | `src/tensor/fft.cpp` |
| I/O | `include/axiom/io/*.hpp` | FlatBuffers (.axfb) and NumPy (.npy) formats |

### Operation Registry Pattern

Operations are registered per-device (CPU/GPU) via `OperationRegistry`. CPU ops use functor-based dispatch with SIMD vectorization. GPU ops use MPSGraph. The `execute_binary_operation` / `execute_unary_operation` functions in `src/tensor/operations.cpp` route to the correct backend.

### Lazy Evaluation Pipeline

`GraphNode` → Graph compiler (topological sort + pattern matching) → Operator fusion (12+ patterns like AddReLU, MulAdd) → Fused kernel execution with arena-pooled buffers. Compiled graphs are cached by signature.

### BLAS/LAPACK Backend Selection

Three-tier fallback: Apple Accelerate (macOS) → OpenBLAS (Linux) → native C++ fallback. Auto-detected at CMake configure time. Backend headers in `src/backends/cpu/blas/` and `src/backends/cpu/lapack/`.

## Development Guidelines

### Design Principles: KISS, YAGNI, SOLID

All of these principles are critical for the development of Axiom.

**KISS (Keep It Simple, Stupid)** - Write straightforward, uncomplicated solutions. Avoid over-engineering and unnecessary complexity. Favor readable and maintainable code.

**YAGNI (You Aren't Gonna Need It)** - Don't add speculative features. Implement only what's currently needed. Reduce code bloat and maintenance overhead.

**SOLID Principles:**
- **S - Single Responsibility Principle (SRP)**: A class should have one job only. Each class should have a single reason to change.
- **O - Open/Closed Principle (OCP)**: Code should be open for extension, closed for modification. Add new features without rewriting existing code.
- **L - Liskov Substitution Principle (LSP)**: A child class should respect the contract of the parent.
- **I - Interface Segregation Principle (ISP)**: Don't force classes to implement methods they don't need. Better to have many small, specific interfaces than one fat, do-everything interface.
- **D - Dependency Inversion Principle (DIP)**: Depend on abstractions (interfaces), not concrete implementations. High-level modules shouldn't depend on low-level details. Both should depend on interfaces/contracts.

### Naming Standards

1. **Name by purpose, not by type or mechanics** - Bad: `data`, `result`, `list1`. Good: `pendingClaims`, `approvedInvoices`, `customerLookup`. Start names with a strong noun/verb: `loadPolicies`, `issueRefund`, `riskScore`.
2. **Match scope to specificity** - Tiny/local var in 3 lines? `i`, `row`, `sum` is fine. Public API/class? Use full, precise names: `ClaimDocumentClassifier`, `PaymentAuthorizationService`.
3. **Use domain language consistently** - Pick one term and stick to it. Prefer business terms over tech mush: `policyLapseDate` > `expiryTs`.
4. **Encode meaning, not metadata** - Don't add types: `customerListArr`. Do add units/context: `timeoutMs`, `premiumCHF`, `createdAtUtc`.
5. **Boolean names read like statements** - `isEligible`, `hasConsent`, `shouldRetry`, `canSettle`. Avoid negatives of negatives: prefer `isActive` over `isNotInactive`.
6. **Plurals and collections** - Singular for one, plural for many: `claim`, `claims`. For maps, say what's keyed by what: `claimsByCustomerId`, `ratesByCountry`.
7. **Avoid abbreviations unless truly common** - Good: `id`, `URL`, `CPU`. Risky: `cfg`, `pol`, `cust`. If you must abbreviate, document once in the README or glossary.
8. **Keep length as short as possible, as long as necessary** - `calculateMonthlyPremium()` good. `calculateMonthlyPremiumForHouseholdInSwissFrancs()` bad (push details to parameters).

## Code Conventions

- **Classes/Structs**: `CamelCase` — **Functions/Methods**: `snake_case` — **Private members**: `trailing_underscore_`
- **Enum values**: `CamelCase` (e.g., `OpType::Add`, `DType::Float32`)
- Google C++ style formatting (`.clang-format`), strict clang-tidy (`.clang-tidy`)
- `#include <axiom/axiom.hpp>` is the main entry point header

## Adding a New Operation

1. Add to `enum class OpType` in `include/axiom/operations.hpp`
2. CPU implementation in `src/backends/cpu/cpu_operations.cpp` (functor + register)
3. GPU implementation in `src/backends/metal/mpsgraph_operations.mm`
4. High-level API in `src/tensor/operations.cpp`, declared in `include/axiom/operations.hpp`
5. Tests in `tests/`

## Testing Patterns

Test utilities in `tests/axiom_test_utils.hpp` provide:
- `SKIP_IF_NO_GPU()` — skip Metal-dependent tests
- `ASSERT_TENSOR_NEAR()` — floating-point tensor comparison with tolerance
- `ASSERT_SHAPE()` — shape assertion helpers

## Platform Notes

- Metal GPU support is macOS-only (Objective-C++ in `.mm` files)
- OpenMP parallelization with intelligent thresholds (small ops skip threading)
- mimalloc used as memory allocator (`third_party/mimalloc/`)
- Highway SIMD abstraction (`third_party/highway/`) — portable across x86, ARM, WASM, RISC-V
