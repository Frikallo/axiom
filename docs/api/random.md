# Random

*For a tutorial introduction, see [User Guide: Random](../user-guide/random).*

Random number generation using the PCG64 algorithm.

## Tensor Factory Methods

### Tensor::randn

```cpp
static Tensor Tensor::randn(const Shape &shape, DType dtype = DType::Float32,
                             Device device = Device::CPU,
                             MemoryOrder order = MemoryOrder::RowMajor);
```

Random tensor from standard normal distribution (mean=0, std=1).

---

### Tensor::rand

```cpp
static Tensor Tensor::rand(const Shape &shape, DType dtype = DType::Float32,
                            Device device = Device::CPU,
                            MemoryOrder order = MemoryOrder::RowMajor);
```

Random tensor from uniform distribution in `[0, 1)`.

---

### Tensor::uniform

```cpp
static Tensor Tensor::uniform(double low, double high, const Shape &shape,
                               DType dtype = DType::Float32,
                               Device device = Device::CPU,
                               MemoryOrder order = MemoryOrder::RowMajor);
```

Random tensor from uniform distribution in `[low, high)`.

---

### Tensor::randint

```cpp
static Tensor Tensor::randint(int64_t low, int64_t high, const Shape &shape,
                               DType dtype = DType::Int64,
                               Device device = Device::CPU,
                               MemoryOrder order = MemoryOrder::RowMajor);
```

Random integers in `[low, high)`.

---

### Tensor::manual_seed

```cpp
static void Tensor::manual_seed(uint64_t seed);
```

Set the global random seed for reproducible results.

**Example:**
```cpp
Tensor::manual_seed(42);
auto a = Tensor::randn({3, 4});  // Reproducible
```

---

### Like Variants

```cpp
static Tensor Tensor::rand_like(const Tensor &prototype);
static Tensor Tensor::randn_like(const Tensor &prototype);
static Tensor Tensor::randint_like(const Tensor &prototype, int64_t low,
                                    int64_t high);
```

Create random tensors matching the shape and dtype of `prototype`.

---

## PCG64

```cpp
class PCG64 {
public:
    using result_type = uint64_t;

    PCG64();                                  // Default seed
    explicit PCG64(uint64_t seed);            // Seed only
    PCG64(uint64_t seed, uint64_t stream);    // Seed + stream

    void seed(uint64_t seed_val);
    void seed(uint64_t seed_val, uint64_t stream);

    uint64_t operator()();                    // Generate next value

    static constexpr uint64_t min();
    static constexpr uint64_t max();

    uint64_t state() const;
    uint64_t stream() const;
};
```

PCG-XSH-RR generator. Fast, small state (128 bits), excellent statistical properties. Same algorithm used by NumPy.

---

## RandomGenerator

```cpp
class RandomGenerator {
public:
    static RandomGenerator &instance();  // Singleton

    void seed(uint64_t seed_val);
    void seed(uint64_t seed_val, uint64_t stream);
    void seed_random();                  // Seed from system entropy

    uint64_t get_seed() const;
    PCG64 &generator();

    uint64_t random_uint64();
    double random_double();              // Uniform in [0, 1)

    template <typename T> T normal(T mean = 0, T stddev = 1);
    template <typename T> T uniform(T low = 0, T high = 1);
    int64_t randint(int64_t low, int64_t high);
};
```

Global singleton managing the random state.

---

## Free Functions

### axiom::manual_seed

```cpp
void axiom::manual_seed(uint64_t seed);
```

Set the global random seed.

---

### axiom::get_seed

```cpp
uint64_t axiom::get_seed();
```

Get the current seed value.

**See Also:** [Tensor Creation](tensor-creation)
