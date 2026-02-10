# Contributing

## Getting Started

1. Fork the repository
2. Clone your fork:

```bash
git clone https://github.com/yourusername/axiom.git
cd axiom
```

3. Build and run tests to verify your setup:

```bash
make release
make test
```

4. Create a feature branch:

```bash
git checkout -b feature/my-improvement
```

## Development Workflow

1. **Make changes** on your feature branch
2. **Format** your code: `make format`
3. **Run tests**: `make test`
4. **Run the full CI pipeline locally**: `make ci`
5. **Commit** with a clear message
6. **Push** and open a pull request

## Pull Request Checklist

Before submitting a PR, verify:

- [ ] Code compiles without warnings (`make release`)
- [ ] All existing tests pass (`make test`)
- [ ] New tests added for new functionality
- [ ] Code is formatted (`make format-check` passes)
- [ ] GPU tests pass if Metal is available
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive

## What to Contribute

**Good first contributions:**

- Bug fixes with test cases
- New test coverage for existing operations
- Documentation improvements
- Performance optimizations with benchmarks

**Larger contributions (discuss first):**

- New operations (follow the [Adding Operations](adding-operations) guide)
- New backend support (CUDA, Vulkan, etc.)
- Architectural changes

## Code Review

All PRs are reviewed before merging. Expect feedback on:

- Correctness and test coverage
- Code style consistency
- Performance implications
- Documentation completeness

## Reporting Issues

When reporting bugs, include:

- Platform (macOS version, Linux distro, CPU architecture)
- Axiom version or commit hash
- Minimal reproduction code
- Expected vs actual behavior
- Full error output

## Project Structure

```
axiom/
  include/axiom/          # Public headers
    tensor.hpp            # Tensor class
    operations.hpp        # OpType enum, Operation base, ops:: functions
    linalg.hpp            # Linear algebra
    fft.hpp               # FFT operations
    einops.hpp            # Einops
    random.hpp            # Random generation
    graph/                # Lazy evaluation graph
  src/
    tensor/               # Tensor implementation
    backends/
      cpu/                # CPU backend (SIMD kernels)
      metal/              # Metal GPU backend (MPSGraph)
    storage/              # Storage implementations
    io/                   # File I/O (FlatBuffers, NumPy)
  tests/                  # Test files
  docs/                   # Documentation (Sphinx)
  benchmarks/             # Benchmark suite
  cmake/                  # CMake modules
```
