#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "compiled_graph.hpp"
#include "graph_signature.hpp"

namespace axiom {
namespace graph {

struct GraphNode;

// Thread-safe LRU cache for compiled computation graphs.
// Pattern matches MPSGraphCache in src/backends/metal/graph_cache.hpp.
class GraphCache {
  public:
    static GraphCache &instance();

    // Look up a compiled graph by signature. On cache miss, compiles
    // the graph from `root` and inserts it. The compilation itself
    // happens outside the lock, so concurrent compilations of different
    // signatures are allowed.
    std::shared_ptr<CompiledGraph> get_or_compile(const GraphSignature &sig,
                                                  const GraphNode *root);

    void clear();
    size_t size() const;
    size_t hits() const;
    size_t misses() const;

    static constexpr size_t MAX_SIZE = 512;

  private:
    GraphCache() = default;
    ~GraphCache() = default;
    GraphCache(const GraphCache &) = delete;
    GraphCache &operator=(const GraphCache &) = delete;

    using LRUList = std::list<GraphSignature>;
    LRUList lru_list_;

    struct CacheEntry {
        std::shared_ptr<CompiledGraph> graph;
        LRUList::iterator lru_iter;
    };

    std::unordered_map<GraphSignature, CacheEntry, GraphSignatureHash> cache_;

    mutable std::mutex mutex_;
    size_t hits_ = 0;
    size_t misses_ = 0;

    void evict_if_needed();
};

} // namespace graph
} // namespace axiom
