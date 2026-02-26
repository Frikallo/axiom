#pragma once

#ifdef AXIOM_CUDA_SUPPORT

#include "nvrtc_codegen.hpp"
#include "nvrtc_compiler.hpp"

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace axiom {
namespace backends {
namespace cuda {

// ============================================================================
// Cache key — identifies a unique fused kernel (shape-independent)
// ============================================================================

struct FusionCacheKey {
    std::vector<ops::OpType> op_chain;
    std::vector<DType> input_dtypes;
    DType output_dtype{};
    bool has_broadcast = false;
    int ndim = 0;
    // Per-input bitmask: bit d set iff input is broadcast along dim d.
    std::vector<uint32_t> broadcast_masks;

    bool operator==(const FusionCacheKey &o) const {
        return op_chain == o.op_chain && input_dtypes == o.input_dtypes &&
               output_dtype == o.output_dtype &&
               has_broadcast == o.has_broadcast && ndim == o.ndim &&
               broadcast_masks == o.broadcast_masks;
    }
};

struct FusionCacheKeyHash {
    size_t operator()(const FusionCacheKey &k) const {
        size_t h = std::hash<int>{}(static_cast<int>(k.output_dtype));
        h ^= std::hash<bool>{}(k.has_broadcast) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        h ^= std::hash<int>{}(k.ndim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (auto op : k.op_chain)
            h ^= std::hash<int>{}(static_cast<int>(op)) + 0x9e3779b9 +
                 (h << 6) + (h >> 2);
        for (auto dt : k.input_dtypes)
            h ^= std::hash<int>{}(static_cast<int>(dt)) + 0x9e3779b9 +
                 (h << 6) + (h >> 2);
        for (auto m : k.broadcast_masks)
            h ^= std::hash<uint32_t>{}(m) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ============================================================================
// Cached entry
// ============================================================================

struct CachedFusedKernel {
    CompiledKernel compiled;
    int block_size = 256;
};

// ============================================================================
// LRU cache
// ============================================================================

class FusedKernelCache {
  public:
    static FusedKernelCache &instance() {
        static FusedKernelCache cache;
        return cache;
    }

    // Look up a cached kernel, or compile one via `build_fn`.
    // Returns a pointer to the cached entry (valid until evicted).
    CachedFusedKernel *
    lookup_or_compile(const FusionCacheKey &key,
                      std::function<GeneratedKernel()> build_fn) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = map_.find(key);
            if (it != map_.end()) {
                // Move to front of LRU list
                lru_.splice(lru_.begin(), lru_, it->second.second);
                return &it->second.first;
            }
        }

        // Compile outside the lock (may take milliseconds)
        GeneratedKernel gen = build_fn();
        CompiledKernel compiled = nvrtc_compile(gen.source, gen.entry_point);
        CachedFusedKernel entry{compiled, 256};

        std::lock_guard<std::mutex> lock(mutex_);
        // Double-check: another thread may have inserted while we compiled
        auto it = map_.find(key);
        if (it != map_.end()) {
            nvrtc_release(compiled);
            lru_.splice(lru_.begin(), lru_, it->second.second);
            return &it->second.first;
        }

        // Evict if at capacity
        while (map_.size() >= MAX_CACHE_SIZE)
            evict_oldest();

        lru_.push_front(key);
        auto [ins, _] = map_.emplace(key, MapEntry{entry, lru_.begin()});
        return &ins->second.first;
    }

    static constexpr size_t MAX_CACHE_SIZE = 256;

  private:
    FusedKernelCache() = default;
    ~FusedKernelCache() {
        for (auto &[k, v] : map_)
            nvrtc_release(v.first.compiled);
    }

    FusedKernelCache(const FusedKernelCache &) = delete;
    FusedKernelCache &operator=(const FusedKernelCache &) = delete;

    using LruList = std::list<FusionCacheKey>;
    using MapEntry = std::pair<CachedFusedKernel, LruList::iterator>;

    std::unordered_map<FusionCacheKey, MapEntry, FusionCacheKeyHash> map_;
    LruList lru_;
    std::mutex mutex_;

    void evict_oldest() {
        if (lru_.empty())
            return;
        auto oldest_key = lru_.back();
        auto it = map_.find(oldest_key);
        if (it != map_.end()) {
            nvrtc_release(it->second.first.compiled);
            map_.erase(it);
        }
        lru_.pop_back();
    }
};

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif
