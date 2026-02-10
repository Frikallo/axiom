# Installation

## Requirements

- **C++20** compiler (Clang 13+, GCC 11+, MSVC 2022+)
- **CMake** 3.20+
- **macOS 11+** for Metal GPU support (optional)
- **Xcode 13+** on macOS

## Build from Source

```bash
git clone https://github.com/Frikallo/axiom.git
cd axiom
make release
```

This builds an optimized release binary. The build system auto-detects your platform and available BLAS backends.

### Build Variants

```bash
make release    # Optimized build (default)
make debug      # Debug build with assertions
make test       # Build and run all tests
make ci         # Full CI pipeline (format-check + build + test)
```

## CMake FetchContent

To use Axiom as a dependency in your CMake project:

```cmake
include(FetchContent)

FetchContent_Declare(
  axiom
  GIT_REPOSITORY https://github.com/Frikallo/axiom.git
  GIT_TAG        v1.0.0
)

FetchContent_MakeAvailable(axiom)

target_link_libraries(your_target PRIVATE axiom)
```

## Platform-Specific Notes

### macOS (Apple Silicon)

Axiom is optimized for Apple Silicon. The build automatically uses:

- **Accelerate framework** for BLAS/LAPACK (matrix multiply, linear algebra)
- **Metal** for GPU acceleration via MPSGraph
- **ARM NEON** SIMD intrinsics for vectorized operations

No additional dependencies are needed -- everything ships with Xcode.

### macOS (Intel)

Metal GPU support is available. Accelerate framework provides BLAS/LAPACK. NEON intrinsics are not available; the scalar fallback is used for non-BLAS operations.

### Linux

```bash
# Install OpenBLAS for LAPACK support (optional but recommended)
sudo apt install libopenblas-dev    # Debian/Ubuntu
sudo dnf install openblas-devel     # Fedora
```

Metal GPU is not available on Linux. All operations run on CPU. OpenBLAS provides BLAS/LAPACK acceleration.

### Windows

Metal GPU is not available on Windows. The native BLAS fallback is used unless you install OpenBLAS manually.

## Verifying the Installation

After building, run the test suite:

```bash
make test
```

To skip GPU tests (e.g., in CI or on machines without Metal):

```bash
AXIOM_SKIP_GPU_TESTS=1 make test
```

To run a single test:

```bash
make test-single TEST=tensor_basic
```

## IDE Integration

Generate `compile_commands.json` for IDE support (clangd, VSCode, CLion):

```bash
make compile-commands
```
