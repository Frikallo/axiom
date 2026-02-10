# Tensor Creation

*For a tutorial introduction, see [User Guide: Tensor Basics](../user-guide/tensor-basics).*

Static factory methods for creating tensors. All methods return a new `Tensor`. Unless otherwise noted, the result lives on `Device::CPU` with `MemoryOrder::RowMajor` layout.

---

## Tensor::zeros

```cpp
static Tensor Tensor::zeros(const Shape &shape,
                             DType dtype = DType::Float32,
                             Device device = Device::CPU,
                             MemoryOrder order = MemoryOrder::RowMajor);

static Tensor Tensor::zeros(std::initializer_list<size_t> shape,
                             DType dtype = DType::Float32,
                             Device device = Device::CPU,
                             MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor filled with zeros.

**Parameters:**
- `shape` (*Shape* or *initializer_list*) -- Dimensions of the output tensor.
- `dtype` (*DType*) -- Data type. Default: `DType::Float32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor filled with zeros.

**Example:**
```cpp
auto a = Tensor::zeros({3, 4});                          // 3x4 float32
auto b = Tensor::zeros({2, 3}, DType::Float64);          // 2x3 float64
auto c = Tensor::zeros({256, 256}, DType::Float32, Device::GPU);  // on GPU
```

**See Also:** [ones](#tensor-ones), [empty](#tensor-empty), [full](#tensor-full), [zeros_like](#tensor-zeros-like)

---

## Tensor::ones

```cpp
static Tensor Tensor::ones(const Shape &shape,
                            DType dtype = DType::Float32,
                            Device device = Device::CPU,
                            MemoryOrder order = MemoryOrder::RowMajor);

static Tensor Tensor::ones(std::initializer_list<size_t> shape,
                            DType dtype = DType::Float32,
                            Device device = Device::CPU,
                            MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor filled with ones.

**Parameters:**
- `shape` (*Shape* or *initializer_list*) -- Dimensions of the output tensor.
- `dtype` (*DType*) -- Data type. Default: `DType::Float32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor filled with ones.

**Example:**
```cpp
auto a = Tensor::ones({3, 4});
auto b = Tensor::ones({5}, DType::Int32);  // 1D integer tensor
```

**See Also:** [zeros](#tensor-zeros), [full](#tensor-full), [ones_like](#tensor-ones-like)

---

## Tensor::empty

```cpp
static Tensor Tensor::empty(const Shape &shape,
                             DType dtype = DType::Float32,
                             Device device = Device::CPU,
                             MemoryOrder order = MemoryOrder::RowMajor);

static Tensor Tensor::empty(std::initializer_list<size_t> shape,
                             DType dtype = DType::Float32,
                             Device device = Device::CPU,
                             MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor with uninitialized memory. The contents are indeterminate and should be written before being read.

**Parameters:**
- `shape` (*Shape* or *initializer_list*) -- Dimensions of the output tensor.
- `dtype` (*DType*) -- Data type. Default: `DType::Float32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor with allocated but uninitialized storage.

**Example:**
```cpp
auto buf = Tensor::empty({1024, 1024});  // fast allocation, no memset
buf.fill(0.0f);                          // fill later
```

**Notes:**
- Faster than `zeros` because it skips initialization.
- Reading from an empty tensor before writing is undefined behavior.

**See Also:** [zeros](#tensor-zeros), [empty_like](#tensor-empty-like)

---

## Tensor::full

```cpp
template <typename T>
static Tensor Tensor::full(const Shape &shape,
                            const T &value,
                            Device device = Device::CPU,
                            MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor filled with a scalar value. The dtype is inferred from the C++ type of `value` via `dtype_of_v<T>`.

**Parameters:**
- `shape` (*Shape*) -- Dimensions of the output tensor.
- `value` (*T*) -- Fill value. The tensor dtype is deduced from this type (e.g., `float` becomes `DType::Float32`, `int32_t` becomes `DType::Int32`).
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor filled with `value`.

**Example:**
```cpp
auto a = Tensor::full({3, 4}, 3.14f);     // float32 filled with 3.14
auto b = Tensor::full({2, 2}, 42);         // int32 filled with 42
auto c = Tensor::full({5}, 1.0, Device::GPU);  // double on GPU
```

**Notes:**
- The dtype is determined by the C++ type of `value`, not by an explicit dtype parameter. To control the dtype, cast the value: `Tensor::full({3}, static_cast<int64_t>(7))`.
- When `device` is `Device::GPU`, the tensor is created on CPU first and then transferred.

**See Also:** [zeros](#tensor-zeros), [ones](#tensor-ones), [full_like](#tensor-full-like)

---

## Tensor::eye

```cpp
static Tensor Tensor::eye(size_t n,
                           DType dtype = DType::Float32,
                           Device device = Device::CPU,
                           MemoryOrder order = MemoryOrder::RowMajor);
```

Create a 2D identity matrix of size `n x n`.

**Parameters:**
- `n` (*size_t*) -- Number of rows and columns.
- `dtype` (*DType*) -- Data type. Default: `DType::Float32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Square identity matrix with ones on the diagonal and zeros elsewhere.

**Example:**
```cpp
auto I = Tensor::eye(4);                     // 4x4 float32 identity
auto I64 = Tensor::eye(3, DType::Float64);   // 3x3 float64 identity
```

**See Also:** [identity](#tensor-identity), [diag](#tensor-diag)

---

## Tensor::identity

```cpp
static Tensor Tensor::identity(size_t n,
                                DType dtype = DType::Float32,
                                Device device = Device::CPU,
                                MemoryOrder order = MemoryOrder::RowMajor);
```

Alias for [`eye`](#tensor-eye). Creates a 2D identity matrix.

**Parameters:** Same as `eye`.

**Returns:** Same as `eye`.

**Example:**
```cpp
auto I = Tensor::identity(4);  // equivalent to Tensor::eye(4)
```

---

## Tensor::randn

```cpp
static Tensor Tensor::randn(const Shape &shape,
                             DType dtype = DType::Float32,
                             Device device = Device::CPU,
                             MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor filled with samples from the standard normal distribution (mean=0, std=1).

**Parameters:**
- `shape` (*Shape*) -- Dimensions of the output tensor.
- `dtype` (*DType*) -- Data type. Default: `DType::Float32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor with normally distributed random values.

**Example:**
```cpp
Tensor::manual_seed(42);                  // for reproducibility
auto x = Tensor::randn({3, 3});           // 3x3 standard normal
auto w = Tensor::randn({768, 512});       // weight initialization
```

**Notes:**
- Use `manual_seed` to set a deterministic seed before calling.
- The underlying generator is PCG64.

**See Also:** [rand](#tensor-rand), [uniform](#tensor-uniform), [randn_like](#tensor-randn-like), [manual_seed](#tensor-manual-seed)

---

## Tensor::rand

```cpp
static Tensor Tensor::rand(const Shape &shape,
                            DType dtype = DType::Float32,
                            Device device = Device::CPU,
                            MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor filled with uniform random values in [0, 1).

**Parameters:**
- `shape` (*Shape*) -- Dimensions of the output tensor.
- `dtype` (*DType*) -- Data type. Default: `DType::Float32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor with values uniformly distributed in [0, 1).

**Example:**
```cpp
auto mask = Tensor::rand({batch, seq_len});  // dropout mask
auto noise = Tensor::rand({3, 256, 256});    // random noise
```

**See Also:** [uniform](#tensor-uniform), [randn](#tensor-randn), [rand_like](#tensor-rand-like)

---

## Tensor::uniform

```cpp
static Tensor Tensor::uniform(double low, double high,
                               const Shape &shape,
                               DType dtype = DType::Float32,
                               Device device = Device::CPU,
                               MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor filled with uniform random values in [low, high).

**Parameters:**
- `low` (*double*) -- Lower bound (inclusive).
- `high` (*double*) -- Upper bound (exclusive).
- `shape` (*Shape*) -- Dimensions of the output tensor.
- `dtype` (*DType*) -- Data type. Default: `DType::Float32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor with values uniformly distributed in [low, high).

**Example:**
```cpp
auto x = Tensor::uniform(-1.0, 1.0, {3, 4});  // uniform in [-1, 1)
auto w = Tensor::uniform(-0.1, 0.1, {512, 256}, DType::Float32);
```

**See Also:** [rand](#tensor-rand), [randint](#tensor-randint)

---

## Tensor::randint

```cpp
static Tensor Tensor::randint(int64_t low, int64_t high,
                               const Shape &shape,
                               DType dtype = DType::Int64,
                               Device device = Device::CPU,
                               MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor filled with random integers in [low, high).

**Parameters:**
- `low` (*int64_t*) -- Lower bound (inclusive).
- `high` (*int64_t*) -- Upper bound (exclusive).
- `shape` (*Shape*) -- Dimensions of the output tensor.
- `dtype` (*DType*) -- Data type. Default: `DType::Int64`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor with random integer values in [low, high).

**Example:**
```cpp
auto labels = Tensor::randint(0, 10, {64});           // batch of labels
auto indices = Tensor::randint(0, 1000, {32, 128});   // random indices
```

**See Also:** [uniform](#tensor-uniform), [randint_like](#tensor-randint-like)

---

## Tensor::manual_seed

```cpp
static void Tensor::manual_seed(uint64_t seed);
```

Set the global random seed for reproducible tensor creation. Affects all subsequent calls to `rand`, `randn`, `uniform`, and `randint`.

**Parameters:**
- `seed` (*uint64_t*) -- Seed value.

**Returns:** None.

**Example:**
```cpp
Tensor::manual_seed(42);
auto a = Tensor::randn({3, 3});

Tensor::manual_seed(42);
auto b = Tensor::randn({3, 3});
// a and b are identical
```

**Notes:**
- Calling `manual_seed` resets the global PCG64 generator state.
- Applies process-wide; there is no per-thread seed.

**See Also:** [Random API reference](random)

---

## Tensor::arange

```cpp
static Tensor Tensor::arange(int64_t start, int64_t end,
                              int64_t step = 1,
                              DType dtype = DType::Int32,
                              Device device = Device::CPU);

static Tensor Tensor::arange(int64_t end,
                              DType dtype = DType::Int32,
                              Device device = Device::CPU);
```

Create a 1D tensor with evenly spaced values in a half-open interval.

**Parameters (two-argument form):**
- `start` (*int64_t*) -- Start of interval (inclusive).
- `end` (*int64_t*) -- End of interval (exclusive).
- `step` (*int64_t*) -- Spacing between values. Default: `1`.
- `dtype` (*DType*) -- Data type. Default: `DType::Int32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.

**Parameters (one-argument form):**
- `end` (*int64_t*) -- End of interval (exclusive). Start is implicitly `0`, step is `1`.
- `dtype` (*DType*) -- Data type. Default: `DType::Int32`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.

**Returns:** 1D tensor with values `[start, start+step, start+2*step, ...)`.

**Example:**
```cpp
auto a = Tensor::arange(5);              // [0, 1, 2, 3, 4]  int32
auto b = Tensor::arange(2, 10, 2);       // [2, 4, 6, 8]     int32
auto c = Tensor::arange(0, 100, 1, DType::Float32);  // float range
```

**Notes:**
- Analogous to `numpy.arange`. The number of elements is `ceil((end - start) / step)`.
- For floating-point ranges with exact endpoint control, prefer `linspace`.

**See Also:** [linspace](#tensor-linspace), [logspace](#tensor-logspace)

---

## Tensor::linspace

```cpp
static Tensor Tensor::linspace(double start, double stop,
                                size_t num = 50,
                                bool endpoint = true,
                                DType dtype = DType::Float64,
                                Device device = Device::CPU);
```

Create a 1D tensor with `num` evenly spaced values between `start` and `stop`.

**Parameters:**
- `start` (*double*) -- Start value.
- `stop` (*double*) -- End value.
- `num` (*size_t*) -- Number of values to generate. Default: `50`.
- `endpoint` (*bool*) -- Whether to include `stop` as the last value. Default: `true`.
- `dtype` (*DType*) -- Data type. Default: `DType::Float64`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.

**Returns:** 1D tensor of `num` evenly spaced values.

**Example:**
```cpp
auto a = Tensor::linspace(0.0, 1.0, 5);
// [0.0, 0.25, 0.5, 0.75, 1.0]

auto b = Tensor::linspace(0.0, 1.0, 5, false);
// [0.0, 0.2, 0.4, 0.6, 0.8]

auto t = Tensor::linspace(0.0, 2.0 * M_PI, 100);  // time axis
```

**Notes:**
- When `endpoint=true` (default), the spacing is `(stop - start) / (num - 1)`.
- When `endpoint=false`, the spacing is `(stop - start) / num`.

**See Also:** [arange](#tensor-arange), [logspace](#tensor-logspace), [geomspace](#tensor-geomspace)

---

## Tensor::logspace

```cpp
static Tensor Tensor::logspace(double start, double stop,
                                size_t num = 50,
                                bool endpoint = true,
                                double base = 10.0,
                                DType dtype = DType::Float64,
                                Device device = Device::CPU);
```

Create a 1D tensor with values spaced evenly on a log scale. Returns `base ^ linspace(start, stop, num)`.

**Parameters:**
- `start` (*double*) -- `base ^ start` is the first value.
- `stop` (*double*) -- `base ^ stop` is the last value (if `endpoint=true`).
- `num` (*size_t*) -- Number of values. Default: `50`.
- `endpoint` (*bool*) -- Include the endpoint. Default: `true`.
- `base` (*double*) -- Base of the logarithm. Default: `10.0`.
- `dtype` (*DType*) -- Data type. Default: `DType::Float64`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.

**Returns:** 1D tensor of `num` values spaced evenly on a log scale.

**Example:**
```cpp
auto a = Tensor::logspace(0.0, 3.0, 4);
// [1.0, 10.0, 100.0, 1000.0]  (10^0, 10^1, 10^2, 10^3)

auto b = Tensor::logspace(0.0, 1.0, 5, true, 2.0);
// [1.0, 1.189, 1.414, 1.682, 2.0]  (base 2)
```

**See Also:** [linspace](#tensor-linspace), [geomspace](#tensor-geomspace)

---

## Tensor::geomspace

```cpp
static Tensor Tensor::geomspace(double start, double stop,
                                 size_t num = 50,
                                 bool endpoint = true,
                                 DType dtype = DType::Float64,
                                 Device device = Device::CPU);
```

Create a 1D tensor with values spaced evenly on a geometric (multiplicative) scale. Each output value is a constant multiple of the previous one.

**Parameters:**
- `start` (*double*) -- Start of the sequence. Must be nonzero.
- `stop` (*double*) -- End of the sequence. Must be nonzero.
- `num` (*size_t*) -- Number of values. Default: `50`.
- `endpoint` (*bool*) -- Include the endpoint. Default: `true`.
- `dtype` (*DType*) -- Data type. Default: `DType::Float64`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.

**Returns:** 1D tensor of `num` geometrically spaced values.

**Example:**
```cpp
auto a = Tensor::geomspace(1.0, 1000.0, 4);
// [1.0, 10.0, 100.0, 1000.0]

auto lr = Tensor::geomspace(1e-4, 1e-1, 10);  // learning rate schedule
```

**Notes:**
- Unlike `logspace`, the `start` and `stop` arguments specify actual values, not exponents.
- Both `start` and `stop` must have the same sign.

**See Also:** [linspace](#tensor-linspace), [logspace](#tensor-logspace)

---

## Tensor::zeros_like

```cpp
static Tensor Tensor::zeros_like(const Tensor &prototype);
```

Create a tensor of zeros with the same shape, dtype, device, and memory order as `prototype`.

**Parameters:**
- `prototype` (*Tensor*) -- Tensor whose properties are copied.

**Returns:** Zero-filled tensor matching the prototype.

**Example:**
```cpp
auto x = Tensor::randn({3, 4}, DType::Float64);
auto grad = Tensor::zeros_like(x);  // same shape and dtype
```

**See Also:** [zeros](#tensor-zeros), [ones_like](#tensor-ones-like), [empty_like](#tensor-empty-like)

---

## Tensor::ones_like

```cpp
static Tensor Tensor::ones_like(const Tensor &prototype);
```

Create a tensor of ones with the same shape, dtype, device, and memory order as `prototype`.

**Parameters:**
- `prototype` (*Tensor*) -- Tensor whose properties are copied.

**Returns:** Ones-filled tensor matching the prototype.

**Example:**
```cpp
auto x = Tensor::randn({3, 4});
auto mask = Tensor::ones_like(x);
```

**See Also:** [ones](#tensor-ones), [zeros_like](#tensor-zeros-like)

---

## Tensor::empty_like

```cpp
static Tensor Tensor::empty_like(const Tensor &prototype);
```

Create a tensor with uninitialized memory matching the shape, dtype, device, and memory order of `prototype`.

**Parameters:**
- `prototype` (*Tensor*) -- Tensor whose properties are copied.

**Returns:** Uninitialized tensor matching the prototype.

**Example:**
```cpp
auto x = Tensor::randn({3, 4});
auto buf = Tensor::empty_like(x);  // same layout, uninitialized
```

**Notes:**
- Contents are indeterminate. Write before reading.

**See Also:** [empty](#tensor-empty), [zeros_like](#tensor-zeros-like)

---

## Tensor::full_like

```cpp
template <typename T>
static Tensor Tensor::full_like(const Tensor &prototype, const T &value);
```

Create a tensor filled with `value`, matching the shape, device, and memory order of `prototype`. The dtype is inferred from the C++ type of `value`.

**Parameters:**
- `prototype` (*Tensor*) -- Tensor whose shape, device, and memory order are copied.
- `value` (*T*) -- Fill value. Determines the output dtype.

**Returns:** Tensor filled with `value`, matching the prototype layout.

**Example:**
```cpp
auto x = Tensor::randn({3, 4});
auto filled = Tensor::full_like(x, -1.0f);  // float32 filled with -1.0
```

**Notes:**
- The output dtype comes from `value`, not from `prototype`. To match the prototype dtype exactly, cast the value: `Tensor::full_like(x, static_cast<double>(v))`.

**See Also:** [full](#tensor-full), [zeros_like](#tensor-zeros-like)

---

## Tensor::rand_like

```cpp
static Tensor Tensor::rand_like(const Tensor &prototype);
```

Create a tensor of uniform random values in [0, 1) with the same shape and dtype as `prototype`.

**Parameters:**
- `prototype` (*Tensor*) -- Tensor whose shape and dtype are copied.

**Returns:** Tensor with uniform random values matching the prototype.

**Example:**
```cpp
auto weights = Tensor::randn({512, 256});
auto noise = Tensor::rand_like(weights);
```

**See Also:** [rand](#tensor-rand), [randn_like](#tensor-randn-like)

---

## Tensor::randn_like

```cpp
static Tensor Tensor::randn_like(const Tensor &prototype);
```

Create a tensor of standard normal random values with the same shape and dtype as `prototype`.

**Parameters:**
- `prototype` (*Tensor*) -- Tensor whose shape and dtype are copied.

**Returns:** Tensor with normally distributed random values matching the prototype.

**Example:**
```cpp
auto x = Tensor::zeros({3, 4}, DType::Float64);
auto noise = Tensor::randn_like(x);  // float64 normal noise
```

**See Also:** [randn](#tensor-randn), [rand_like](#tensor-rand-like)

---

## Tensor::randint_like

```cpp
static Tensor Tensor::randint_like(const Tensor &prototype,
                                    int64_t low, int64_t high);
```

Create a tensor of random integers in [low, high) with the same shape and dtype as `prototype`.

**Parameters:**
- `prototype` (*Tensor*) -- Tensor whose shape and dtype are copied.
- `low` (*int64_t*) -- Lower bound (inclusive).
- `high` (*int64_t*) -- Upper bound (exclusive).

**Returns:** Tensor with random integers matching the prototype.

**Example:**
```cpp
auto idx = Tensor::randint(0, 100, {32});
auto idx2 = Tensor::randint_like(idx, 0, 50);  // same shape, different range
```

**See Also:** [randint](#tensor-randint), [rand_like](#tensor-rand-like)

---

## Tensor::from_data

```cpp
template <typename T>
static Tensor Tensor::from_data(const T *data,
                                 const Shape &shape,
                                 bool copy = true,
                                 MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor from a raw pointer to existing data. The dtype is inferred from the C++ type `T`.

**Parameters:**
- `data` (*const T\**) -- Pointer to source data. Must contain at least `product(shape)` elements.
- `shape` (*Shape*) -- Dimensions of the output tensor.
- `copy` (*bool*) -- If `true`, the data is copied into new storage. Default: `true`.
- `order` (*MemoryOrder*) -- Memory layout of the source data and output tensor. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor containing the data.

**Example:**
```cpp
float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
auto t = Tensor::from_data(data, {2, 3});
// [[1, 2, 3],
//  [4, 5, 6]]

double matrix[] = {1, 0, 0, 1};
auto eye = Tensor::from_data(matrix, {2, 2});
```

**Notes:**
- Only `copy=true` is currently supported. Passing `copy=false` throws `RuntimeError`.
- When `order` is `MemoryOrder::ColMajor`, the source data is assumed to be in row-major order and is transposed into column-major layout during the copy.

**See Also:** [from_array](#tensor-from-array), [asarray](#tensor-asarray)

---

## Tensor::from_array

```cpp
template <typename T, size_t N>
static Tensor Tensor::from_array(const T (&data)[N],
                                  const Shape &shape,
                                  DType target_dtype = dtype_of_v<T>,
                                  Device device = Device::CPU,
                                  MemoryOrder order = MemoryOrder::RowMajor);
```

Create a tensor from a C-style array with optional type conversion and device placement. The array size `N` is checked at runtime against the shape.

**Parameters:**
- `data` (*const T (&)[N]*) -- C-style array of `N` elements.
- `shape` (*Shape*) -- Dimensions of the output tensor. `product(shape)` must equal `N`.
- `target_dtype` (*DType*) -- Desired output dtype. Default: deduced from `T`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.
- `order` (*MemoryOrder*) -- Memory layout. Default: `MemoryOrder::RowMajor`.

**Returns:** Tensor containing the array data, optionally converted and placed on the target device.

**Example:**
```cpp
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
auto a = Tensor::from_array(data, {2, 2});

// Convert to float64 during creation
int vals[] = {1, 2, 3, 4, 5, 6};
auto b = Tensor::from_array(vals, {2, 3}, DType::Float64);

// Create directly on GPU
float weights[] = {0.1f, 0.2f, 0.3f};
auto c = Tensor::from_array(weights, {3}, DType::Float32, Device::GPU);
```

**Notes:**
- Throws `ShapeError` if `N != product(shape)`.
- If `target_dtype` differs from the source type, an `astype` conversion is applied.
- If `device` is not CPU, the tensor is first created on CPU and then transferred.

**See Also:** [from_data](#tensor-from-data), [asarray](#tensor-asarray)

---

## Tensor::asarray

```cpp
template <typename T>
static Tensor Tensor::asarray(const Tensor &tensor);

template <typename T>
static Tensor Tensor::asarray(const Tensor &tensor, Device device);
```

Convert a tensor to a specified C++ type (and optionally move to a device). This is a convenience wrapper around `astype` and `to`.

**Parameters:**
- `tensor` (*Tensor*) -- Input tensor.
- `device` (*Device*) -- Target device (second overload only).

**Returns:** Tensor with dtype matching `T`, optionally on the specified device.

**Example:**
```cpp
auto x = Tensor::arange(5);                       // int32
auto xf = Tensor::asarray<float>(x);              // float32
auto xg = Tensor::asarray<float>(x, Device::GPU); // float32 on GPU
```

**Notes:**
- The template parameter `T` determines the output dtype via `dtype_of_v<T>`.
- If the tensor already has the target dtype (and device), this may still create a copy.

**See Also:** [from_data](#tensor-from-data), [from_array](#tensor-from-array)

---

## Tensor::diag

```cpp
static Tensor Tensor::diag(const Tensor &v, int64_t k = 0);
```

Create a 2D diagonal matrix from a 1D vector, or extract the diagonal from a 2D matrix.

**Parameters:**
- `v` (*Tensor*) -- If 1D, used as the diagonal values. If 2D, the diagonal is extracted.
- `k` (*int64_t*) -- Diagonal offset. `0` is the main diagonal, positive values are above, negative values are below. Default: `0`.

**Returns:** If `v` is 1D, returns a 2D matrix with `v` on the k-th diagonal. If `v` is 2D, returns a 1D tensor of the k-th diagonal.

**Example:**
```cpp
float vals[] = {1.0f, 2.0f, 3.0f};
auto v = Tensor::from_data(vals, {3});

auto D = Tensor::diag(v);
// [[1, 0, 0],
//  [0, 2, 0],
//  [0, 0, 3]]

auto D_upper = Tensor::diag(v, 1);
// [[0, 1, 0, 0],
//  [0, 0, 2, 0],
//  [0, 0, 0, 3],
//  [0, 0, 0, 0]]

// Extract diagonal from matrix
auto d = Tensor::diag(D);   // [1, 2, 3]
```

**See Also:** [eye](#tensor-eye), [tri](#tensor-tri), [tril](#tensor-tril), [triu](#tensor-triu)

---

## Tensor::tri

```cpp
static Tensor Tensor::tri(size_t N, size_t M = 0,
                           int64_t k = 0,
                           DType dtype = DType::Float64,
                           Device device = Device::CPU);
```

Create a matrix with ones at and below the k-th diagonal and zeros elsewhere.

**Parameters:**
- `N` (*size_t*) -- Number of rows.
- `M` (*size_t*) -- Number of columns. Default: `0` (interpreted as `M = N`).
- `k` (*int64_t*) -- Diagonal offset. `0` is the main diagonal. Default: `0`.
- `dtype` (*DType*) -- Data type. Default: `DType::Float64`.
- `device` (*Device*) -- Target device. Default: `Device::CPU`.

**Returns:** NxM matrix with ones on and below the k-th diagonal.

**Example:**
```cpp
auto T = Tensor::tri(3);
// [[1, 0, 0],
//  [1, 1, 0],
//  [1, 1, 1]]

auto T2 = Tensor::tri(3, 4, 1);
// [[1, 1, 0, 0],
//  [1, 1, 1, 0],
//  [1, 1, 1, 1]]
```

**See Also:** [tril](#tensor-tril), [triu](#tensor-triu), [eye](#tensor-eye)

---

## Tensor::tril

```cpp
static Tensor Tensor::tril(const Tensor &m, int64_t k = 0);
```

Return the lower triangle of a matrix. Elements above the k-th diagonal are set to zero.

**Parameters:**
- `m` (*Tensor*) -- Input 2D tensor.
- `k` (*int64_t*) -- Diagonal offset. `0` is the main diagonal. Default: `0`.

**Returns:** Lower-triangular matrix.

**Example:**
```cpp
auto x = Tensor::ones({3, 3});
auto L = Tensor::tril(x);
// [[1, 0, 0],
//  [1, 1, 0],
//  [1, 1, 1]]

auto L1 = Tensor::tril(x, 1);
// [[1, 1, 0],
//  [1, 1, 1],
//  [1, 1, 1]]
```

**See Also:** [triu](#tensor-triu), [tri](#tensor-tri), [diag](#tensor-diag)

---

## Tensor::triu

```cpp
static Tensor Tensor::triu(const Tensor &m, int64_t k = 0);
```

Return the upper triangle of a matrix. Elements below the k-th diagonal are set to zero.

**Parameters:**
- `m` (*Tensor*) -- Input 2D tensor.
- `k` (*int64_t*) -- Diagonal offset. `0` is the main diagonal. Default: `0`.

**Returns:** Upper-triangular matrix.

**Example:**
```cpp
auto x = Tensor::ones({3, 3});
auto U = Tensor::triu(x);
// [[1, 1, 1],
//  [0, 1, 1],
//  [0, 0, 1]]

auto U_neg1 = Tensor::triu(x, -1);
// [[1, 1, 1],
//  [1, 1, 1],
//  [0, 1, 1]]
```

**See Also:** [tril](#tensor-tril), [tri](#tensor-tri), [diag](#tensor-diag)

---

## Tensor::result_type

```cpp
static DType Tensor::result_type(const Tensor &a, const Tensor &b);
```

Determine the result dtype when two tensors are combined in a binary operation. Follows NumPy-compatible type promotion rules.

**Parameters:**
- `a` (*Tensor*) -- First tensor.
- `b` (*Tensor*) -- Second tensor.

**Returns:** The promoted `DType`.

**Example:**
```cpp
auto a = Tensor::zeros({3}, DType::Float32);
auto b = Tensor::zeros({3}, DType::Float64);
DType dt = Tensor::result_type(a, b);  // DType::Float64

auto x = Tensor::zeros({3}, DType::Int32);
auto y = Tensor::zeros({3}, DType::Float32);
DType dt2 = Tensor::result_type(x, y);  // DType::Float32
```

**Notes:**
- Useful for pre-allocating output buffers with the correct dtype.
- Promotion rules: integer + float -> float; narrower + wider -> wider; signed + unsigned -> signed (widened if necessary).

**See Also:** [Data Types](dtypes)

---

## Summary Table

| Method | Description | Default dtype |
|--------|-------------|---------------|
| `zeros` | Fill with 0 | Float32 |
| `ones` | Fill with 1 | Float32 |
| `empty` | Uninitialized | Float32 |
| `full` | Fill with scalar | deduced from value |
| `eye` / `identity` | Identity matrix | Float32 |
| `randn` | Normal(0,1) | Float32 |
| `rand` | Uniform [0,1) | Float32 |
| `uniform` | Uniform [low,high) | Float32 |
| `randint` | Integer [low,high) | Int64 |
| `arange` | Integer range | Int32 |
| `linspace` | Linear spacing | Float64 |
| `logspace` | Log spacing | Float64 |
| `geomspace` | Geometric spacing | Float64 |
| `from_data` | From raw pointer | deduced from T |
| `from_array` | From C array | deduced from T |
| `diag` | Diagonal matrix | same as input |
| `tri` | Lower-tri ones | Float64 |
| `tril` | Lower triangle | same as input |
| `triu` | Upper triangle | same as input |

| CPU | GPU |
|-----|-----|
| All methods | All methods (created on CPU and transferred) |
