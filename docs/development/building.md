# Building

## Prerequisites

| Platform | Requirements |
|----------|-------------|
| macOS | macOS 11+, Xcode 13+ (for Metal), CMake 3.20+, C++20 compiler |
| Linux | CMake 3.20+, GCC 11+ or Clang 14+, OpenBLAS (recommended) |
| Windows | CMake 3.20+, MSVC 2022+ or Clang |

Optional: Ninja (faster builds), OpenMP (parallelization), clang-format, clang-tidy.

## Quick Start

```bash
git clone https://github.com/yourusername/axiom.git
cd axiom
make release    # Build optimized release
make test       # Run all tests
```

## Make Targets

| Target | Description |
|--------|-------------|
| `make release` | Build optimized release (default) |
| `make debug` | Build debug version with assertions |
| `make test` | Run all tests |
| `make test-single TEST=tensor_basic` | Run a single test |
| `make test-verbose` | Run tests with verbose output |
| `make test-failed` | Rerun only failed tests |
| `make test-list` | List all available tests |
| `make format` | Format all source files |
| `make format-check` | Check formatting without modifying |
| `make lint` | Run clang-tidy static analysis |
| `make compile-commands` | Generate `compile_commands.json` |
| `make ci` | Full CI pipeline (format-check + build + test) |
| `make lib` | Build only the library (no tests/examples) |
| `make clean` | Remove release build directory |
| `make rebuild` | Clean and rebuild release |

## CMake Options

Configure directly with CMake for fine-grained control:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DAXIOM_BUILD_TESTS=ON \
    -DAXIOM_BUILD_EXAMPLES=ON \
    -DAXIOM_BUILD_BENCHMARKS=OFF \
    -DAXIOM_USE_OPENMP=ON \
    -DAXIOM_BLAS_BACKEND=auto \
    -DAXIOM_LAPACK_BACKEND=auto \
    -DAXIOM_NATIVE_ARCH=OFF

cmake --build build -j$(nproc)
```

### Option Reference

| Option | Default | Description |
|--------|---------|-------------|
| `AXIOM_BUILD_TESTS` | `ON` | Build unit tests |
| `AXIOM_BUILD_EXAMPLES` | `ON` | Build example programs |
| `AXIOM_BUILD_BENCHMARKS` | `OFF` | Build benchmark suite |
| `AXIOM_USE_OPENMP` | `ON` | Enable OpenMP parallelization |
| `AXIOM_NATIVE_ARCH` | `OFF` | Use `-march=native` (non-portable) |
| `AXIOM_DIST_BUILD` | `OFF` | Bundle dependencies for distribution |
| `AXIOM_BLAS_BACKEND` | `auto` | BLAS backend: `auto`, `accelerate`, `openblas`, `native` |
| `AXIOM_LAPACK_BACKEND` | `auto` | LAPACK backend: `auto`, `accelerate`, `openblas`, `native` |

## BLAS/LAPACK Configuration

With `auto` (default), Axiom detects backends in this order:

1. **macOS**: Accelerate framework (always available)
2. **Linux**: OpenBLAS (if installed)
3. **Fallback**: Native C++ implementation

To force a specific backend:

```bash
# Force OpenBLAS on macOS
cmake -B build -DAXIOM_BLAS_BACKEND=openblas -DAXIOM_LAPACK_BACKEND=openblas

# Force native (no external dependencies)
cmake -B build -DAXIOM_BLAS_BACKEND=native -DAXIOM_LAPACK_BACKEND=native
```

Install OpenBLAS on Linux:

```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

## Platform-Specific Notes

### macOS (Apple Silicon)

Optimized by default for Apple M-series chips (`-mcpu=apple-m1`). Metal GPU support is automatic when Xcode command-line tools are installed.

### macOS (Intel)

Uses SSE4.2/AVX2 SIMD where available. Metal GPU support depends on the discrete/integrated GPU model.

### Linux

GPU support (Metal) is not available. All operations run on CPU. Install OpenBLAS for best BLAS performance.

### Cross-compilation

Use `AXIOM_NATIVE_ARCH=OFF` (default) for portable builds. Enable `AXIOM_NATIVE_ARCH=ON` only for local development where portability is not needed.

## IDE Integration

Generate `compile_commands.json` for IDE support:

```bash
make compile-commands
# Or: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

This enables clang-based tooling in VS Code, CLion, Neovim, and other editors.

## Using Axiom as a Dependency

### CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    axiom
    GIT_REPOSITORY https://github.com/yourusername/axiom.git
    GIT_TAG main
)
FetchContent_MakeAvailable(axiom)

target_link_libraries(your_target PRIVATE axiom)
```

### CMake subdirectory

```cmake
add_subdirectory(path/to/axiom)
target_link_libraries(your_target PRIVATE axiom)
```
