# Getting Started

## Installation

### From source (CMake)

```bash
git clone https://github.com/Frikallo/axiom.git
cd axiom
make release
```

The build auto-detects platform features: Metal GPU on macOS, Accelerate/OpenBLAS for BLAS, and OpenMP for threading.

### CMake integration

Add Axiom as a subdirectory or use `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    axiom
    GIT_REPOSITORY https://github.com/Frikallo/axiom.git
    GIT_TAG main
)
FetchContent_MakeAvailable(axiom)
target_link_libraries(your_target PRIVATE axiom)
```

## Quickstart

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;
using namespace axiom::nn;

int main() {
    // Create tensors
    auto a = Tensor::randn({3, 4});           // 3x4 random normal
    auto b = Tensor::ones({4, 5});            // 4x5 ones

    // Matrix multiply
    auto c = ops::matmul(a, b);               // 3x5 result
    c.print();

    // Element-wise operations
    auto d = ops::relu(a) + Tensor::full({3, 4}, 2.0f);
    d.print();

    // Linear algebra
    auto [U, S, Vt] = linalg::svd(a);

    // FFT
    auto spectrum = fft::fft(a);

    // Neural network inference
    nn::Linear layer(true);
    std::map<std::string, Tensor> weights;
    weights["weight"] = Tensor::randn({5, 4});
    weights["bias"] = Tensor::zeros({5});
    layer.load_state_dict(weights);
    auto out = layer(a);                         // (3, 5)

    return 0;
}
```

## Next steps

- Browse the full [API Reference](api/index.md)
- See [Benchmarks](BENCHMARKS.md) for performance comparisons
