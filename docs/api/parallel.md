# Parallelism

Thread management and parallelization policies in the `axiom::parallel` namespace. Requires `AXIOM_USE_OPENMP` compile flag.

## Constants

```cpp
constexpr size_t parallel::DEFAULT_MIN_ELEMENTS = 65536;    // ~256KB for float32
constexpr size_t parallel::REDUCTION_MIN_ELEMENTS = 262144;  // Higher for reductions
constexpr size_t parallel::MATMUL_MIN_PRODUCT = 1000000;     // M*N*K threshold
```

Minimum element counts before parallelization is triggered.

---

## Thread Management

### parallel::get_num_threads

```cpp
size_t parallel::get_num_threads();
```

Returns the maximum number of threads available. Returns 1 if OpenMP is not enabled.

---

### parallel::set_num_threads

```cpp
void parallel::set_num_threads(size_t n);
```

Set the number of threads. Use `0` for all available.

---

### parallel::get_thread_id

```cpp
size_t parallel::get_thread_id();
```

Returns the current thread ID within a parallel region.

---

## Parallelization Policies

### parallel::should_parallelize

```cpp
bool parallel::should_parallelize(size_t elements,
                                  size_t min_elements = DEFAULT_MIN_ELEMENTS);
```

Returns `true` if parallelization is beneficial for the given element count.

---

### parallel::should_parallelize_reduction

```cpp
bool parallel::should_parallelize_reduction(size_t elements);
```

Uses a higher threshold than general operations (reductions have more overhead).

---

### parallel::should_parallelize_matmul

```cpp
bool parallel::should_parallelize_matmul(size_t M, size_t N, size_t K);
```

Checks if `M * N * K >= MATMUL_MIN_PRODUCT`.

---

## ThreadGuard

```cpp
class parallel::ThreadGuard {
public:
    explicit ThreadGuard(size_t n);  // Set thread count
    ~ThreadGuard();                  // Restore previous count
};
```

RAII guard for temporarily changing the thread count.

```cpp
{
    parallel::ThreadGuard guard(1);  // Force single-threaded
    // ... operations here run single-threaded ...
}  // Previous thread count restored
```

**See Also:** [System](system)
