# Axiom Backend Architecture

## Overview

Axiom uses a modular backend system that allows easy extension to support different compute devices and memory types. Each backend implements the `Storage` interface and provides device-specific optimizations.

## Current Backends

### CPU Backend (`src/backends/cpu/`)
- **File**: `cpu_storage.hpp/cpp`
- **Device**: `Device::CPU`
- **Features**: 
  - System memory allocation
  - Direct pointer access
  - Memory views and zero-copy operations
  - Always available

### Metal Backend (`src/backends/metal/`)
- **File**: `metal_storage.hpp/mm` (Objective-C++)
- **Device**: `Device::GPU`
- **Features**:
  - Metal GPU buffer allocation
  - Automatic CPU ↔ GPU transfers
  - Shared memory mode for efficiency
  - Apple Silicon optimized
  - Only available on macOS

## Backend Interface

All backends implement the abstract `Storage` class:

```cpp
class Storage {
public:
    virtual void* data() = 0;                    // Raw data access (CPU only)
    virtual size_t size_bytes() const = 0;       // Memory size
    virtual Device device() const = 0;           // Backend device type
    virtual void copy_to(Storage& other) = 0;    // Cross-device copies
    virtual void copy_from(Storage& other) = 0;  // Cross-device copies
    virtual std::unique_ptr<Storage> clone() = 0; // Deep copy
    virtual bool is_view() const = 0;            // Memory sharing check
    virtual std::shared_ptr<Storage> base() = 0; // Base storage for views
};
```

## Adding New Backends

To add a new backend (e.g., CUDA, OpenCL, Vulkan):

1. **Create backend directory**: `src/backends/my_backend/`
2. **Implement Storage interface**: `my_storage.hpp/cpp` 
3. **Add Device enum**: Update `Device` enum in `storage.hpp`
4. **Update factory functions**: Modify `storage.cpp` dispatcher
5. **Add to CMake**: Include sources in `CMakeLists.txt`
6. **Register backend**: Update `backend_registry.cpp`

### Example Structure

```
src/backends/cuda/
├── cuda_storage.hpp      # CUDA storage interface
├── cuda_storage.cu       # CUDA implementation
└── cuda_utils.cu         # CUDA helper functions
```

## Usage Examples

### Basic Device Selection
```cpp
// Create tensors on different devices
auto cpu_tensor = zeros({100, 100}, DType::Float32, Device::CPU);
auto gpu_tensor = zeros({100, 100}, DType::Float32, Device::GPU);

// Cross-device transfers
auto moved_to_gpu = cpu_tensor.gpu();
auto moved_to_cpu = gpu_tensor.cpu();
```

### Backend Discovery
```cpp
#include "backends/backend_registry.hpp"

// List all available backends
auto backends = BackendRegistry::available_backends();
for (const auto& backend : backends) {
    std::cout << backend.name << ": " 
              << (backend.available ? "Available" : "Not Available")
              << std::endl;
}

// Check specific device
bool has_gpu = BackendRegistry::is_device_available(Device::GPU);
Device default_dev = BackendRegistry::default_device();
```

## Memory Management

### Zero-Copy Views
```cpp
auto base = zeros({1000, 1000}, DType::Float32, Device::CPU);
auto view = base.reshape({100, 10000});  // Zero-copy view
auto slice = base.squeeze();             // Zero-copy squeeze

// All share the same underlying Storage
assert(base.storage() == view.storage());
```

### Cross-Device Copies
```cpp
auto cpu_data = ones({512, 512}, DType::Float32, Device::CPU);
auto gpu_data = cpu_data.gpu();  // Triggers CPU → GPU copy

// Modify GPU data, copy back
gpu_data = process_on_gpu(gpu_data);
auto result = gpu_data.cpu();    // Triggers GPU → CPU copy
```

## Performance Considerations

### CPU Backend
- **Memory**: Standard system malloc/new
- **Access**: Direct pointer access
- **Views**: Zero-copy with stride calculations
- **Threading**: Single-threaded (operations layer will add threading)

### Metal Backend  
- **Memory**: `MTLBuffer` with shared storage mode
- **Access**: CPU-accessible GPU memory
- **Transfers**: Automatic via buffer contents
- **Optimization**: Unified memory on Apple Silicon

## Thread Safety

- **Storage objects**: Not thread-safe (by design)
- **Factory functions**: Thread-safe
- **Cross-device copies**: Synchronous and thread-safe
- **Memory views**: Share underlying data, require external synchronization

## Future Extensions

Planned backend additions:
- **CUDA**: NVIDIA GPU support
- **OpenCL**: Cross-platform GPU compute
- **Vulkan**: Modern graphics/compute API  
- **ROCm**: AMD GPU support
- **Memory Pool**: Custom allocators
- **Distributed**: Multi-node storage