#include "axiom/graph/graph_cache.hpp"
#include "axiom/graph/graph_compiler.hpp"

namespace axiom {
namespace graph {

GraphCache &GraphCache::instance() {
    static GraphCache cache;
    return cache;
}

std::shared_ptr<CompiledGraph>
GraphCache::get_or_compile(const GraphSignature &sig, const GraphNode *root) {
    // Phase 1: lookup under lock
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(sig);
        if (it != cache_.end()) {
            hits_++;
            // Move to front of LRU
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iter);
            return it->second.graph;
        }
        misses_++;
    }

    // Phase 2: compile without lock
    auto compiled = compile(sig, root);

    // Phase 3: insert under lock
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Check again in case another thread compiled the same signature
        auto it = cache_.find(sig);
        if (it != cache_.end()) {
            // Another thread beat us — use their result
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iter);
            return it->second.graph;
        }

        // Insert into LRU and cache
        lru_list_.push_front(sig);
        CacheEntry entry;
        entry.graph = compiled;
        entry.lru_iter = lru_list_.begin();
        cache_[sig] = entry;

        evict_if_needed();
    }

    return compiled;
}

void GraphCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
    hits_ = 0;
    misses_ = 0;
}

size_t GraphCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

size_t GraphCache::hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hits_;
}

size_t GraphCache::misses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return misses_;
}

void GraphCache::evict_if_needed() {
    while (cache_.size() > MAX_SIZE && !lru_list_.empty()) {
        auto &oldest = lru_list_.back();
        cache_.erase(oldest);
        lru_list_.pop_back();
    }
}

} // namespace graph
} // namespace axiom
