# File I/O

*For a tutorial introduction, see [User Guide: File I/O](../user-guide/file-io).*

Save and load tensors in the `axiom::io` namespace. Also available as Tensor member/static methods.

## Tensor Methods

### Tensor::save

```cpp
void Tensor::save(const std::string &filename) const;
```

Save a single tensor to file (`.axfb` FlatBuffers format).

---

### Tensor::load

```cpp
static Tensor Tensor::load(const std::string &filename,
                            Device device = Device::CPU);
```

Load a single tensor. Auto-detects format (`.axfb` or `.npy`).

---

### Tensor::save_tensors

```cpp
static void Tensor::save_tensors(const std::map<std::string, Tensor> &tensors,
                                  const std::string &filename);
```

Save multiple named tensors to an archive.

---

### Tensor::load_tensors

```cpp
static std::map<std::string, Tensor>
Tensor::load_tensors(const std::string &filename, Device device = Device::CPU);
```

Load all tensors from an archive.

---

### Tensor::list_tensors_in_archive

```cpp
static std::vector<std::string>
Tensor::list_tensors_in_archive(const std::string &filename);
```

List tensor names in an archive without loading data.

---

### Tensor::load_tensor_from_archive

```cpp
static Tensor Tensor::load_tensor_from_archive(const std::string &filename,
                                                const std::string &tensor_name,
                                                Device device = Device::CPU);
```

Load a specific tensor from an archive by name.

---

## axiom::io Namespace

### io::detect_format

```cpp
io::FileFormat io::detect_format(const std::string &filename);
```

Detect file format by examining magic bytes. Returns `FileFormat::Axiom`, `FileFormat::NumPy`, or `FileFormat::Unknown`.

---

### io::save / io::load

```cpp
void io::save(const Tensor &tensor, const std::string &filename);
Tensor io::load(const std::string &filename, Device device = Device::CPU);
```

---

### io::save_archive / io::load_archive

```cpp
void io::save_archive(const std::map<std::string, Tensor> &tensors,
                      const std::string &filename);
std::map<std::string, Tensor> io::load_archive(const std::string &filename,
                                                Device device = Device::CPU);
```

---

### io::to_string

```cpp
std::string io::to_string(const Tensor &tensor);
```

NumPy-style string representation of a tensor.

---

## File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| Axiom FlatBuffers | `.axfb` | Native binary format. Supports single and multi-tensor archives. |
| NumPy | `.npy` | Read-only support. Compatible with `np.save()` output. |

## Exceptions

- `io::SerializationError` -- Base I/O exception.
- `io::FileFormatError` -- Invalid or unsupported file format.

**Example:**
```cpp
auto tensor = Tensor::randn({100, 100});
tensor.save("weights.axfb");

auto loaded = Tensor::load("weights.axfb");

// Multi-tensor archive
Tensor::save_tensors({{"W", W}, {"b", b}}, "model.axfb");
auto model = Tensor::load_tensors("model.axfb");
auto W_loaded = model["W"];
```

**See Also:** [Tensor Class](tensor-class)
