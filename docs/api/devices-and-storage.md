# Devices & Storage

*For a tutorial introduction, see [User Guide: GPU Acceleration](../user-guide/gpu-acceleration).*

## Device Enum

```cpp
enum class Device {
    CPU,
    GPU  // Metal on Apple Silicon
};
```

## Storage Class

```cpp
class Storage {
public:
    virtual void *data() = 0;
    virtual const void *data() const = 0;
    virtual size_t size_bytes() const = 0;
    virtual Device device() const = 0;
    virtual void copy_to(Storage &other) const = 0;
    virtual void copy_from(const Storage &other) = 0;
    virtual std::unique_ptr<Storage> clone() const = 0;
};
```

Abstract base class for device-specific memory management.

### Implementations

- **CpuStorage** -- System memory with 64-byte alignment. Always available.
- **MetalStorage** -- Metal GPU buffer with shared storage mode. macOS only.

---

### make_storage

```cpp
std::unique_ptr<Storage> make_storage(size_t size_bytes,
                                       Device device = Device::CPU);
```

Factory function that creates the appropriate storage backend.

---

## Device Transfer Methods

### Tensor::to

```cpp
Tensor Tensor::to(Device device, MemoryOrder order = MemoryOrder::RowMajor) const;
```

Move tensor to a device. Returns a copy on the target device.

---

### Tensor::cpu

```cpp
Tensor Tensor::cpu() const;
```

Move to CPU. No-op if already on CPU.

---

### Tensor::gpu

```cpp
Tensor Tensor::gpu() const;
```

Move to GPU (Metal). Throws `DeviceError` if Metal is not available.

---

### Tensor::device

```cpp
Device Tensor::device() const;
```

Returns the device this tensor is on.

---

## Memory Order

```cpp
enum class MemoryOrder {
    RowMajor,    // C-contiguous (default)
    ColumnMajor  // Fortran-contiguous
};
```

### Tensor::ascontiguousarray

```cpp
Tensor Tensor::ascontiguousarray() const;
```

Returns a C-contiguous (row-major) copy if not already contiguous.

---

### Tensor::asfortranarray

```cpp
Tensor Tensor::asfortranarray() const;
```

Returns a Fortran-contiguous (column-major) copy.

---

## Memory Model

- **Shared storage:** Multiple tensors can share the same `Storage` via views (reshape, transpose, slice). Check with `tensor.shares_storage(other)`.
- **Unified memory:** On Apple Silicon, Metal uses shared storage mode for zero-copy CPU/GPU access.
- **Automatic transfers:** Operations between CPU and GPU tensors automatically transfer data as needed.

**See Also:** [Tensor Class](tensor-class), [System](system)
