#import "graph_cache.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// MPSGraphCacheKey Implementation
// ============================================================================

size_t MPSGraphCacheKey::hash() const {
    size_t h = std::hash<int>{}(static_cast<int>(op_type));

    // Hash shape-agnostic flag
    h ^= std::hash<bool>{}(shape_agnostic) + 0x9e3779b9 + (h << 6) + (h >> 2);

    // Hash input shapes
    for (size_t i = 0; i < num_inputs; ++i) {
        if (shape_agnostic) {
            // Only hash rank for shape-agnostic mode
            h ^= std::hash<size_t>{}(input_shapes[i].size()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        } else {
            // Hash full shape for exact matching
            for (int64_t dim : input_shapes[i]) {
                h ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
        }
        h ^= std::hash<int>{}(input_dtypes[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }

    // Hash output dtype
    h ^= std::hash<int>{}(output_dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);

    // Hash matmul parameters
    h ^= std::hash<bool>{}(transpose_a) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(transpose_b) + 0x9e3779b9 + (h << 6) + (h >> 2);

    // Hash reduction parameters
    for (int axis : reduction_axes) {
        h ^= std::hash<int>{}(axis) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    h ^= std::hash<bool>{}(keep_dims) + 0x9e3779b9 + (h << 6) + (h >> 2);

    return h;
}

bool MPSGraphCacheKey::operator==(const MPSGraphCacheKey &other) const {
    if (op_type != other.op_type)
        return false;
    if (shape_agnostic != other.shape_agnostic)
        return false;
    if (num_inputs != other.num_inputs)
        return false;

    for (size_t i = 0; i < num_inputs; ++i) {
        if (shape_agnostic) {
            // Only compare rank for shape-agnostic mode
            if (input_shapes[i].size() != other.input_shapes[i].size())
                return false;
        } else {
            // Compare full shape for exact matching
            if (input_shapes[i] != other.input_shapes[i])
                return false;
        }
        if (input_dtypes[i] != other.input_dtypes[i])
            return false;
    }

    if (output_dtype != other.output_dtype)
        return false;
    if (transpose_a != other.transpose_a)
        return false;
    if (transpose_b != other.transpose_b)
        return false;
    if (reduction_axes != other.reduction_axes)
        return false;
    if (keep_dims != other.keep_dims)
        return false;

    return true;
}

// ============================================================================
// MPSGraphCache Implementation
// ============================================================================

MPSGraphCache &MPSGraphCache::instance() {
    static MPSGraphCache instance;
    return instance;
}

MPSGraphCache::~MPSGraphCache() { clear(); }

CachedMPSGraph *
MPSGraphCache::get_or_create(const MPSGraphCacheKey &key,
                             std::function<CachedMPSGraph()> factory) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already cached
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Move to front of LRU list (most recently used)
        lru_list_.erase(it->second.second);
        lru_list_.push_front(key);
        it->second.second = lru_list_.begin();
        ++hits_;
        return &it->second.first;
    }

    // Not in cache, create new entry
    ++misses_;

    CachedMPSGraph new_graph = factory();
    if (!new_graph.is_valid()) {
        return nullptr;
    }

    // Note: Graph is already retained by factory using CFBridgingRetain
    // We will release it in release_graph() when evicting

    // Evict if necessary
    evict_if_needed();

    // Add to LRU list
    lru_list_.push_front(key);

    // Add to cache
    auto result = cache_.emplace(key, std::make_pair(new_graph, lru_list_.begin()));
    return &result.first->second.first;
}

void MPSGraphCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &pair : cache_) {
        release_graph(pair.second.first);
    }

    cache_.clear();
    lru_list_.clear();
}

size_t MPSGraphCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

void MPSGraphCache::evict_if_needed() {
    // Called with mutex held
    while (cache_.size() >= MAX_SIZE && !lru_list_.empty()) {
        // Remove least recently used (back of list)
        const MPSGraphCacheKey &lru_key = lru_list_.back();
        auto it = cache_.find(lru_key);
        if (it != cache_.end()) {
            release_graph(it->second.first);
            cache_.erase(it);
        }
        lru_list_.pop_back();
    }
}

void MPSGraphCache::release_graph(CachedMPSGraph &entry) {
    if (entry.graph) {
        CFRelease(entry.graph);
        entry.graph = nullptr;
    }
    // Placeholders and output are owned by the graph, so they're released
    // automatically when the graph is released
    for (size_t i = 0; i < 3; ++i) {
        entry.placeholders[i] = nullptr;
    }
    entry.output = nullptr;
}

// ============================================================================
// Helper Functions Implementation
// ============================================================================

// Helper to convert shape to dynamic shape (-1 for all dims)
static std::vector<int64_t> to_dynamic_shape(const std::vector<int64_t> &shape) {
    return std::vector<int64_t>(shape.size(), -1);
}

MPSGraphCacheKey make_binary_cache_key(ops::OpType op_type,
                                       const std::vector<int64_t> &lhs_shape,
                                       const std::vector<int64_t> &rhs_shape,
                                       int lhs_dtype, int rhs_dtype,
                                       int output_dtype,
                                       bool shape_agnostic) {
    MPSGraphCacheKey key;
    key.op_type = op_type;
    key.num_inputs = 2;
    key.shape_agnostic = shape_agnostic;
    if (shape_agnostic) {
        key.input_shapes[0] = to_dynamic_shape(lhs_shape);
        key.input_shapes[1] = to_dynamic_shape(rhs_shape);
    } else {
        key.input_shapes[0] = lhs_shape;
        key.input_shapes[1] = rhs_shape;
    }
    key.input_dtypes[0] = lhs_dtype;
    key.input_dtypes[1] = rhs_dtype;
    key.output_dtype = output_dtype;
    return key;
}

MPSGraphCacheKey make_unary_cache_key(ops::OpType op_type,
                                      const std::vector<int64_t> &input_shape,
                                      int input_dtype, int output_dtype,
                                      bool shape_agnostic) {
    MPSGraphCacheKey key;
    key.op_type = op_type;
    key.num_inputs = 1;
    key.shape_agnostic = shape_agnostic;
    if (shape_agnostic) {
        key.input_shapes[0] = to_dynamic_shape(input_shape);
    } else {
        key.input_shapes[0] = input_shape;
    }
    key.input_dtypes[0] = input_dtype;
    key.output_dtype = output_dtype;
    return key;
}

MPSGraphCacheKey make_ternary_cache_key(ops::OpType op_type,
                                        const std::vector<int64_t> &cond_shape,
                                        const std::vector<int64_t> &a_shape,
                                        const std::vector<int64_t> &b_shape,
                                        int cond_dtype, int a_dtype, int b_dtype,
                                        int output_dtype) {
    MPSGraphCacheKey key;
    key.op_type = op_type;
    key.num_inputs = 3;
    key.input_shapes[0] = cond_shape;
    key.input_shapes[1] = a_shape;
    key.input_shapes[2] = b_shape;
    key.input_dtypes[0] = cond_dtype;
    key.input_dtypes[1] = a_dtype;
    key.input_dtypes[2] = b_dtype;
    key.output_dtype = output_dtype;
    return key;
}

MPSGraphCacheKey make_reduction_cache_key(ops::OpType op_type,
                                          const std::vector<int64_t> &input_shape,
                                          int input_dtype, int output_dtype,
                                          const std::vector<int> &axes,
                                          bool keep_dims) {
    MPSGraphCacheKey key;
    key.op_type = op_type;
    key.num_inputs = 1;
    key.input_shapes[0] = input_shape;
    key.input_dtypes[0] = input_dtype;
    key.output_dtype = output_dtype;
    key.reduction_axes = axes;
    key.keep_dims = keep_dims;
    return key;
}

MPSGraphCacheKey make_matmul_cache_key(const std::vector<int64_t> &a_shape,
                                       const std::vector<int64_t> &b_shape,
                                       int a_dtype, int b_dtype, int output_dtype,
                                       bool transpose_a, bool transpose_b) {
    MPSGraphCacheKey key;
    key.op_type = ops::OpType::MatMul;
    key.num_inputs = 2;
    key.input_shapes[0] = a_shape;
    key.input_shapes[1] = b_shape;
    key.input_dtypes[0] = a_dtype;
    key.input_dtypes[1] = b_dtype;
    key.output_dtype = output_dtype;
    key.transpose_a = transpose_a;
    key.transpose_b = transpose_b;
    return key;
}

} // namespace metal
} // namespace backends
} // namespace axiom
