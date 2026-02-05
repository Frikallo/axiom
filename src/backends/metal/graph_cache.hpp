#pragma once

#include "axiom/operations.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

// Forward-declare Objective-C types
#ifdef __OBJC__
@class MPSGraph;
@class MPSGraphTensor;
#else
typedef void MPSGraph;
typedef void MPSGraphTensor;
#endif

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// MPSGraph Cache Key
// ============================================================================

// Key for looking up cached MPSGraph instances
// Uniquely identifies a graph by operation type, input shapes/dtypes, and
// parameters
struct MPSGraphCacheKey {
    ops::OpType op_type;

    // Input shapes (up to 3 inputs)
    // For shape-agnostic caching, use -1 for dynamic dimensions
    std::vector<int64_t> input_shapes[3];
    size_t num_inputs;

    // Input data types (as integers for hashing)
    int input_dtypes[3];

    // Output data type
    int output_dtype;

    // MatMul-specific parameters
    bool transpose_a;
    bool transpose_b;

    // Reduction-specific parameters
    std::vector<int> reduction_axes;
    bool keep_dims;

    // Shape-agnostic mode: if true, only rank matters, not exact dimensions
    // This allows graphs to be reused for different batch sizes
    bool shape_agnostic;

    // Default constructor
    MPSGraphCacheKey()
        : op_type(ops::OpType::Add), num_inputs(0), input_dtypes{0, 0, 0},
          output_dtype(0), transpose_a(false), transpose_b(false),
          keep_dims(false), shape_agnostic(false) {}

    // Compute hash for the key
    size_t hash() const;

    // Equality comparison
    bool operator==(const MPSGraphCacheKey &other) const;
};

// Hash functor for use in unordered_map
struct MPSGraphCacheKeyHash {
    size_t operator()(const MPSGraphCacheKey &key) const { return key.hash(); }
};

// ============================================================================
// Cached MPSGraph Entry
// ============================================================================

// A cached MPSGraph with its placeholder tensors and output tensor reference
struct CachedMPSGraph {
    // The compiled graph (retained)
    void *graph; // MPSGraph*

    // Placeholder tensors for inputs (retained by graph)
    void *placeholders[3]; // MPSGraphTensor*
    size_t num_placeholders;

    // Output tensor reference (retained by graph)
    void *output; // MPSGraphTensor*

    // Default constructor
    CachedMPSGraph()
        : graph(nullptr), placeholders{nullptr, nullptr, nullptr},
          num_placeholders(0), output(nullptr) {}

    // Check if valid
    bool is_valid() const { return graph != nullptr && output != nullptr; }
};

// ============================================================================
// MPSGraph Cache
// ============================================================================

// Thread-safe LRU cache for compiled MPSGraph instances
// Eliminates per-operation graph compilation overhead
class MPSGraphCache {
  public:
    // Get singleton instance
    static MPSGraphCache &instance();

    // Get or create a cached graph
    // If not found, calls factory to create it, then caches and returns it
    // Factory signature: CachedMPSGraph factory()
    CachedMPSGraph *get_or_create(const MPSGraphCacheKey &key,
                                  std::function<CachedMPSGraph()> factory);

    // Clear all cached graphs (releases all MPSGraph objects)
    void clear();

    // Get current cache size
    size_t size() const;

    // Get cache statistics
    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }

    // Maximum cache size (configurable)
    static constexpr size_t MAX_SIZE = 1024;

  private:
    MPSGraphCache() = default;
    ~MPSGraphCache();

    // Prevent copying
    MPSGraphCache(const MPSGraphCache &) = delete;
    MPSGraphCache &operator=(const MPSGraphCache &) = delete;

    // LRU list: most recently used at front
    using LRUList = std::list<MPSGraphCacheKey>;
    LRUList lru_list_;

    // Map from key to (cached graph, iterator into LRU list)
    using CacheEntry = std::pair<CachedMPSGraph, LRUList::iterator>;
    std::unordered_map<MPSGraphCacheKey, CacheEntry, MPSGraphCacheKeyHash>
        cache_;

    // Thread safety
    mutable std::mutex mutex_;

    // Statistics
    size_t hits_ = 0;
    size_t misses_ = 0;

    // Evict least recently used entries until size <= MAX_SIZE
    void evict_if_needed();

    // Release a cached graph's resources
    void release_graph(CachedMPSGraph &entry);
};

// ============================================================================
// Helper Functions for Creating Cache Keys
// ============================================================================

// Create a cache key for binary operations
// shape_agnostic: if true, only rank matters for cache lookup (dynamic batch)
MPSGraphCacheKey make_binary_cache_key(ops::OpType op_type,
                                       const std::vector<int64_t> &lhs_shape,
                                       const std::vector<int64_t> &rhs_shape,
                                       int lhs_dtype, int rhs_dtype,
                                       int output_dtype,
                                       bool shape_agnostic = false);

// Create a cache key for unary operations
// shape_agnostic: if true, only rank matters for cache lookup (dynamic batch)
MPSGraphCacheKey make_unary_cache_key(ops::OpType op_type,
                                      const std::vector<int64_t> &input_shape,
                                      int input_dtype, int output_dtype,
                                      bool shape_agnostic = false);

// Create a cache key for ternary operations (where)
MPSGraphCacheKey make_ternary_cache_key(ops::OpType op_type,
                                        const std::vector<int64_t> &cond_shape,
                                        const std::vector<int64_t> &a_shape,
                                        const std::vector<int64_t> &b_shape,
                                        int cond_dtype, int a_dtype,
                                        int b_dtype, int output_dtype);

// Create a cache key for reduction operations
MPSGraphCacheKey
make_reduction_cache_key(ops::OpType op_type,
                         const std::vector<int64_t> &input_shape,
                         int input_dtype, int output_dtype,
                         const std::vector<int> &axes, bool keep_dims);

// Create a cache key for matmul operations
MPSGraphCacheKey make_matmul_cache_key(const std::vector<int64_t> &a_shape,
                                       const std::vector<int64_t> &b_shape,
                                       int a_dtype, int b_dtype,
                                       int output_dtype, bool transpose_a,
                                       bool transpose_b);

} // namespace metal
} // namespace backends
} // namespace axiom
