# File I/O

Axiom supports saving and loading tensors in two formats: the native Axiom FlatBuffers format (`.axfb`) and NumPy's `.npy` format.

## Saving a Single Tensor

```cpp
using namespace axiom;

auto tensor = Tensor::randn({100, 100});

// Save as Axiom FlatBuffers (recommended)
tensor.save("data.axfb");

// Or via the io namespace
io::save(tensor, "data.axfb");
```

## Loading a Single Tensor

```cpp
// Auto-detects format from magic bytes
auto tensor = Tensor::load("data.axfb");

// Load directly to GPU
auto gpu_tensor = Tensor::load("data.axfb", Device::GPU);

// Load NumPy files
auto np_tensor = Tensor::load("array.npy");
```

## Multi-Tensor Archives

Save and load collections of named tensors:

```cpp
// Save multiple tensors
std::map<std::string, Tensor> model_weights;
model_weights["layer1.weight"] = Tensor::randn({64, 32});
model_weights["layer1.bias"] = Tensor::randn({64});
model_weights["layer2.weight"] = Tensor::randn({10, 64});
model_weights["layer2.bias"] = Tensor::randn({10});

Tensor::save_tensors(model_weights, "model.axfb");

// Load all tensors
auto loaded = Tensor::load_tensors("model.axfb");
auto w1 = loaded["layer1.weight"];
auto b1 = loaded["layer1.bias"];
```

## Inspecting Archives

List tensor names without loading data:

```cpp
auto names = Tensor::list_tensors_in_archive("model.axfb");
for (const auto &name : names) {
    std::cout << name << std::endl;
}
// Output:
// layer1.bias
// layer1.weight
// layer2.bias
// layer2.weight

// Load a single tensor from an archive
auto w = Tensor::load_tensor_from_archive("model.axfb", "layer1.weight");
```

## File Formats

### Axiom FlatBuffers (`.axfb`)

The native format, based on Google FlatBuffers:

- Preserves all 14 dtypes
- Supports single and multi-tensor archives
- Fast serialization/deserialization (zero-copy reads where possible)
- Compact binary format

### NumPy (`.npy`)

Interoperability with Python/NumPy:

- Read `.npy` files written by `numpy.save()`
- Standard format understood by most scientific computing tools
- Single-tensor only

## Format Detection

Axiom auto-detects the format by examining magic bytes:

```cpp
auto format = io::detect_format("data.axfb");
// Returns io::FileFormat::Axiom

auto np_format = io::detect_format("array.npy");
// Returns io::FileFormat::NumPy

std::cout << io::format_name(format) << std::endl;
// Output: "Axiom"
```

## GPU Tensors

GPU tensors are automatically transferred to CPU before saving and can be loaded directly to GPU:

```cpp
auto gpu_tensor = Tensor::randn({256, 256}, DType::Float32, Device::GPU);

// Saves CPU copy (automatic transfer)
gpu_tensor.save("gpu_data.axfb");

// Load directly to GPU
auto loaded = Tensor::load("gpu_data.axfb", Device::GPU);
```

For complete function signatures, see [API Reference: I/O](../api/io).
