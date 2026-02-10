# Lazy Evaluation

Most tensor libraries execute every operation the moment you call it.
Axiom supports a second mode -- **lazy evaluation** -- where operations
are recorded into a computation graph and only executed when their
results are actually needed. This lets the graph compiler fuse multiple
operations together, eliminate redundant memory traffic, and reuse
compiled execution plans across repeated calls.

## Eager vs Lazy Mode

By default, Axiom operations run in **lazy mode**: each call to
`ops::add`, `ops::relu`, `ops::sum`, and so on returns a *lazy tensor*
whose data has not yet been computed. The library infers the output
shape, dtype, and device immediately, but defers the actual kernel
launch until the result is consumed.

```cpp
#include <axiom/axiom.hpp>
using namespace axiom;

ops::OperationRegistry::initialize_builtin_operations();

auto a = Tensor::randn({1024, 1024});
auto b = Tensor::randn({1024, 1024});

// No kernel executes here -- c and d are lazy tensors
auto c = ops::add(a, b);
auto d = ops::multiply(c, c);

// Shape is known immediately
assert(d.shape() == Shape({1024, 1024}));
assert(d.dtype() == DType::Float32);
```

To force eager execution for an entire program, set the environment
variable `AXIOM_EAGER_MODE=1`. You can also switch to eager mode
within a specific scope using `EagerModeScope`:

```cpp
{
    graph::EagerModeScope eager;
    // All operations inside this scope execute immediately
    auto e = ops::add(a, b);   // computed right away
    auto f = ops::relu(e);     // computed right away
}
// Back to lazy mode here
```

## The Computation Graph

Each lazy operation creates a `GraphNode`. A node stores everything
needed to execute the operation later, but allocates no output memory:

| Field            | Purpose                                             |
|------------------|-----------------------------------------------------|
| `id`             | Unique node identifier (monotonically increasing)   |
| `op_type`        | The operation (`OpType::Add`, `OpType::ReLU`, etc.) |
| `inputs`         | Shared pointers to input `GraphNode`s               |
| `output_shape`   | Inferred output shape                               |
| `output_dtype`   | Inferred output dtype                               |
| `target_device`  | `Device::CPU` or `Device::GPU`                      |
| `params`         | Operation-specific parameters (axes, alpha, etc.)   |

Nodes form a **directed acyclic graph (DAG)**. When a materialized
(eager) tensor is used as input to a lazy operation, it is wrapped in a
*constant node* that holds a reference to the existing storage.

```cpp
auto a = Tensor::ones({256, 256});    // materialized
auto b = Tensor::ones({256, 256});    // materialized

// Graph: [const a] --\
//                      +--> [Add c] --> [ReLU d] --> [Sum e]
// Graph: [const b] --/
auto c = ops::add(a, b);
auto d = ops::relu(c);
auto e = ops::sum(d);
```

## Materialization

Materialization is the process of executing the computation graph to
produce actual tensor data. It is triggered automatically whenever you
access the underlying data of a lazy tensor:

- Calling `.data()` or `.typed_data<T>()`
- Printing the tensor (e.g. `std::cout << tensor`)
- Calling `.item<T>()` to extract a scalar
- Using the tensor as input to an eager operation
- Accessing `.storage()` directly

The method `materialize_if_needed()` is called internally before any
data access. If the node has already been materialized, this is a
no-op.

```cpp
auto a = Tensor::randn({3, 3});
auto b = Tensor::randn({3, 3});

auto c = ops::add(a, b);
assert(c.is_lazy() == true);    // no data yet

// Access data -- triggers materialization of the entire graph
float val = c.typed_data<float>()[0];

// Now the result is cached inside the graph node
assert(c.is_lazy() == true);    // still a lazy tensor object...
// ...but subsequent data accesses are free (cached)
float val2 = c.typed_data<float>()[1];  // no re-execution
```

Once a node is materialized, its result is **cached** in the node's
`cached_result_` storage. Any further data access on that tensor
returns the cached result without re-executing the graph.

## Introspection

Lazy tensors expose metadata without triggering materialization:

```cpp
auto a = Tensor::randn({4, 8});
auto b = ops::relu(a);

// These are all available without executing the graph
b.is_lazy();       // true -- not yet materialized
b.shape();         // {4, 8}
b.ndim();          // 2
b.size();          // 32
b.dtype();         // DType::Float32
b.device();        // Device::CPU
b.itemsize();      // 4
b.nbytes();        // 128
```

This makes it possible to build an entire computation graph, inspect
its shapes and types for correctness, and only pay the cost of
execution when results are needed.

## Operation Fusion

The key advantage of lazy evaluation is **operation fusion**. Instead
of executing each operation in isolation (allocating a temporary buffer,
reading inputs, writing output, and repeating for the next operation),
the graph compiler detects sequences of operations that can be merged
into a single pass over the data.

### Element-wise Chains

Consecutive unary and binary element-wise operations are fused into a
single loop. This avoids intermediate allocations and keeps data in CPU
registers or L1 cache:

```cpp
auto x = Tensor::randn({1000000});
auto y = Tensor::randn({1000000});

// Without fusion: 3 separate passes, 2 temporary buffers
// With fusion: 1 pass, 0 temporary buffers
auto result = ops::relu(ops::add(x, y));
```

### Recognized Fused Patterns

The compiler matches specific operation sequences to hand-optimized
SIMD kernels via the `FusedPattern` enum:

**Binary + Unary (2 inputs):**

| Pattern        | Computation       | Example use case       |
|----------------|-------------------|------------------------|
| `AddReLU`      | `relu(a + b)`     | Residual connection    |
| `SubAbs`       | `\|a - b\|`       | L1 distance            |
| `AddSquare`    | `(a + b)^2`       | Squared sum            |
| `MulReLU`      | `relu(a * b)`     | Gated activation       |
| `SubSquare`    | `(a - b)^2`       | L2 distance squared    |
| `AddSigmoid`   | `sigmoid(a + b)`  | Logistic pre-act       |
| `MulSigmoid`   | `sigmoid(a * b)`  | Gated sigmoid          |

**Ternary (3 inputs):**

| Pattern           | Computation              | Example use case       |
|-------------------|--------------------------|------------------------|
| `MulAdd`          | `a * b + c`              | FMA / linear layer     |
| `MulSub`          | `a * b - c`              | Scaled difference      |
| `ScaleShiftReLU`  | `relu(a * scale + bias)` | Batch norm + activate  |

When a recognized pattern is detected, the compiler emits a
`FusedKnown` execution step that dispatches directly to a specialized
SIMD kernel. Unrecognized element-wise chains still benefit from fusion
via the `FusedGeneric` path, which uses function-pointer dispatch in a
single fused loop.

### Fused Reduction

When an element-wise chain feeds directly into a full reduction
(all axes), the compiler merges them into a single `FusedReduction`
step. The element-wise operations and the reduction execute together in
a tiled loop, avoiding the intermediate allocation entirely:

```cpp
auto x = Tensor::randn({1024, 1024});
auto y = Tensor::randn({1024, 1024});

// Fused into one pass: element-wise add + square + sum-reduce
auto loss = ops::sum(ops::square(ops::subtract(x, y)));
```

### MatMul + Activation Fusion

When a matrix multiplication is immediately followed by a unary
activation function (and no other consumer uses the matmul result), the
compiler fuses them into a single `MatMulActivation` step:

```cpp
auto W = Tensor::randn({256, 512});
auto x = Tensor::randn({512, 128});

// matmul + relu fused into a single step
auto h = ops::relu(ops::matmul(W, x));
```

## The Compiled Graph

When a lazy tensor is materialized, its graph goes through a
multi-stage compilation pipeline:

1. **Topological sort** -- linearize the DAG into execution order
2. **Dead code elimination** -- remove nodes not reachable from the
   output
3. **Fusion analysis** -- group consecutive element-wise ops into fused
   chains; merge matmul+activation and element-wise+reduction pairs
4. **Memory planning** -- compute buffer lifetimes and assign reusable
   memory slots
5. **Loop parameter computation** -- determine tile sizes and classify
   input access patterns

The result is a `CompiledGraph` containing:

- **`steps`** -- A vector of `ExecutionStep` objects. Each step has a
  `Kind`:

  | Kind                | Description                                     |
  |---------------------|-------------------------------------------------|
  | `SingleOp`          | Dispatch via `OperationRegistry`                |
  | `FusedKnown`        | Matched SIMD pattern (e.g. `AddReLU`)           |
  | `FusedGeneric`      | Generic fused loop with function-pointer chain  |
  | `MatMulActivation`  | MatMul + activation in one step                 |
  | `FusedReduction`    | Element-wise chain + full reduction combined    |

- **`buffer_slots`** -- Metadata for each buffer the plan uses. Every
  slot tracks its `byte_size`, `dtype`, `shape`, `strides`, `device`,
  and liveness interval (`first_use` / `last_use` step indices).

- **`slot_to_allocation`** -- Maps buffer slots to physical memory
  allocations. Dead slots are reused by later slots to minimize peak
  memory.

- **Access patterns** -- Each input to a step is classified by its
  `AccessPattern`:

  | Pattern            | Meaning                                        |
  |--------------------|------------------------------------------------|
  | `Contiguous`       | Dense, row-major data -- fast sequential access|
  | `Strided`          | Non-unit strides -- needs offset calculation   |
  | `Broadcast`        | Smaller shape broadcast to match output        |
  | `ScalarBroadcast`  | Single-element tensor broadcast everywhere     |

## Graph Caching

Compiled graphs are cached by their **`GraphSignature`** -- a
structural hash of the graph's ops, shapes, dtypes, strides, and
parameters (but not data values). When the same computation pattern
appears again with different data, the cached plan is reused without
recompilation:

```cpp
for (int i = 0; i < 1000; ++i) {
    auto x = Tensor::randn({256, 256});
    auto y = Tensor::randn({256, 256});

    // First iteration: compile + execute
    // Remaining 999 iterations: cache hit, execute only
    auto z = ops::relu(ops::add(x, y));
    float val = z.typed_data<float>()[0];
}
```

The cache is an LRU map with a maximum capacity of 512 entries
(`GraphCache::MAX_SIZE`). It is thread-safe (protected by a mutex), and
concurrent compilations of different signatures are allowed in
parallel.

## Memory Optimization

The graph compiler applies two levels of memory optimization:

### Buffer Slot Reuse

During memory planning, each buffer slot records when it is first
produced (`first_use`) and when it is last read (`last_use`). Once a
slot is dead (all consumers have executed), its physical allocation can
be reused by a later slot. This is implemented as a linear-scan
allocator over the step sequence:

```
Step 0:  [alloc A] produces slot 0
Step 1:  reads slot 0, [alloc B] produces slot 1
Step 2:  reads slot 1  -- slot 0 is now dead
Step 3:  [reuse A] produces slot 2  -- same memory as slot 0
```

### Arena Pool

Each `CompiledGraph` maintains an **arena pool** of pre-allocated
buffer sets. When a compiled plan is executed, it first tries to
acquire an arena from the pool. If one is available, it reuses the
existing allocations instead of calling the system allocator. After
execution, the arena is returned to the pool.

The pool is capped at `MAX_FREE_ARENAS = 4` to prevent unbounded memory
growth under concurrent workloads. Excess arenas are dropped
automatically.

```
Execution 1: allocate arena (cold start)
Execution 2: reuse arena from pool (zero allocation)
Execution 3: reuse arena from pool (zero allocation)
```

## Performance Tips

Lazy evaluation with fusion provides the largest speedups when your
computation involves chains of element-wise operations, especially
when followed by a reduction. Here are common patterns that benefit:

**Scale-shift-activate (normalization output):**

```cpp
auto x = Tensor::randn({batch, features});
auto gamma = Tensor::ones({features});
auto beta = Tensor::zeros({features});

// Fused into ScaleShiftReLU: relu(x * gamma + beta)
auto out = ops::relu(ops::add(ops::multiply(x, gamma), beta));
```

**Residual connections:**

```cpp
auto x = Tensor::randn({batch, seq_len, hidden});
auto residual = Tensor::randn({batch, seq_len, hidden});

// Fused into AddReLU: relu(x + residual)
auto out = ops::relu(ops::add(x, residual));
```

**Loss computation (element-wise + reduction):**

```cpp
auto pred = Tensor::randn({batch, classes});
auto target = Tensor::randn({batch, classes});

// Fused into a single pass: sub -> square -> sum
auto mse = ops::sum(ops::square(ops::subtract(pred, target)));
```

**When lazy mode helps most:**

- Long chains of element-wise operations (3+ ops)
- Element-wise chains ending in a full reduction
- Matrix multiplications followed immediately by an activation
- Repeated execution of the same graph structure (benefits from caching)

**When to prefer eager mode:**

- Single isolated operations with no chain
- Debugging (easier to inspect intermediate values)
- Operations that require data access between steps (e.g. conditional
  branching on tensor values)

You can use `EagerModeScope` to switch selectively in hot paths:

```cpp
{
    graph::EagerModeScope eager;
    // Run a few ops eagerly for debugging
    auto debug_val = ops::sum(x);
    std::cout << debug_val << std::endl;
}
// Lazy mode resumes
auto fused_result = ops::relu(ops::add(x, y));
```

For complete API details, see [API Reference: Tensor Class](../api/tensor-class).
