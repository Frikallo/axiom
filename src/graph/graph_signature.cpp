#include "axiom/graph/graph_signature.hpp"
#include "axiom/graph/graph_node.hpp"

#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace axiom {
namespace graph {

// FNV-1a streaming hash
static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
static constexpr uint64_t FNV_PRIME = 1099511628211ULL;

static inline uint64_t fnv_hash_byte(uint64_t h, uint8_t b) {
    h ^= b;
    h *= FNV_PRIME;
    return h;
}

static inline uint64_t fnv_hash_u64(uint64_t h, uint64_t val) {
    for (int i = 0; i < 8; ++i) {
        h = fnv_hash_byte(h, static_cast<uint8_t>(val & 0xFF));
        val >>= 8;
    }
    return h;
}

static inline uint64_t fnv_hash_i32(uint64_t h, int32_t val) {
    uint32_t u;
    std::memcpy(&u, &val, 4);
    for (int i = 0; i < 4; ++i) {
        h = fnv_hash_byte(h, static_cast<uint8_t>(u & 0xFF));
        u >>= 8;
    }
    return h;
}

static inline uint64_t fnv_hash_float(uint64_t h, float val) {
    uint32_t u;
    std::memcpy(&u, &val, 4);
    for (int i = 0; i < 4; ++i) {
        h = fnv_hash_byte(h, static_cast<uint8_t>(u & 0xFF));
        u >>= 8;
    }
    return h;
}

static inline uint64_t fnv_hash_bool(uint64_t h, bool val) {
    return fnv_hash_byte(h, val ? 1 : 0);
}

// Topological sort (non-recursive to avoid stack overflow on deep graphs)
static std::vector<const GraphNode *> topo_sort(const GraphNode *root) {
    std::vector<const GraphNode *> result;
    std::unordered_set<const GraphNode *> visited;
    std::vector<std::pair<const GraphNode *, size_t>> stack;

    stack.push_back({root, 0});

    while (!stack.empty()) {
        auto &[node, idx] = stack.back();

        if (visited.count(node)) {
            stack.pop_back();
            continue;
        }

        if (idx < node->inputs.size()) {
            const GraphNode *child = node->inputs[idx].get();
            idx++;
            if (!visited.count(child)) {
                stack.push_back({child, 0});
            }
        } else {
            visited.insert(node);
            result.push_back(node);
            stack.pop_back();
        }
    }

    return result;
}

GraphSignature compute_signature(const GraphNode *root) {
    auto sorted = topo_sort(root);

    // Assign a local index to each node for structural identity
    std::unordered_map<const GraphNode *, uint32_t> node_index;
    for (uint32_t i = 0; i < sorted.size(); ++i) {
        node_index[sorted[i]] = i;
    }

    uint64_t h = FNV_OFFSET;

    // Hash number of nodes as a prefix
    h = fnv_hash_u64(h, sorted.size());

    for (const auto *node : sorted) {
        // Hash op_type
        h = fnv_hash_u64(h, static_cast<uint64_t>(node->op_type));

        // Hash output shape
        h = fnv_hash_u64(h, node->output_shape.size());
        for (auto dim : node->output_shape) {
            h = fnv_hash_u64(h, dim);
        }

        // Hash output dtype
        h = fnv_hash_u64(h, static_cast<uint64_t>(node->output_dtype));

        // Hash target device
        h = fnv_hash_u64(h, static_cast<uint64_t>(node->target_device));

        // Hash is_constant flag
        h = fnv_hash_bool(h, node->is_constant);

        // For constant nodes: hash strides and contiguity (NOT data)
        if (node->is_constant) {
            h = fnv_hash_u64(h, node->constant_strides.size());
            for (auto s : node->constant_strides) {
                h = fnv_hash_u64(h, static_cast<uint64_t>(s));
            }
        }

        // Hash input connectivity (structural edges)
        h = fnv_hash_u64(h, node->inputs.size());
        for (const auto &inp : node->inputs) {
            auto it = node_index.find(inp.get());
            if (it != node_index.end()) {
                h = fnv_hash_u64(h, it->second);
            } else {
                // External node not in subgraph — use sentinel
                h = fnv_hash_u64(h, UINT64_MAX);
            }
        }

        // Hash params (visit the variant)
        h = fnv_hash_u64(h, node->params.index()); // variant index
        std::visit(
            [&](const auto &p) {
                using T = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<T, ReductionParams>) {
                    h = fnv_hash_u64(h, p.axes.size());
                    for (int ax : p.axes)
                        h = fnv_hash_i32(h, ax);
                    h = fnv_hash_bool(h, p.keep_dims);
                } else if constexpr (std::is_same_v<T, MatMulParams>) {
                    h = fnv_hash_bool(h, p.transpose_a);
                    h = fnv_hash_bool(h, p.transpose_b);
                } else if constexpr (std::is_same_v<T, ActivationParams>) {
                    h = fnv_hash_float(h, p.alpha);
                    h = fnv_hash_i32(h, p.axis);
                } else if constexpr (std::is_same_v<T, ReshapeParams>) {
                    h = fnv_hash_u64(h, p.new_shape.size());
                    for (auto dim : p.new_shape)
                        h = fnv_hash_u64(h, dim);
                }
                // NoParams: nothing to hash
            },
            node->params);
    }

    return GraphSignature{h};
}

} // namespace graph
} // namespace axiom
