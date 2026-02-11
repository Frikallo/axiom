#pragma once

#include <cstddef>
#include <cstdint>

namespace axiom {
namespace graph {

struct GraphNode;

struct GraphSignature {
    uint64_t hash;
    bool operator==(const GraphSignature &o) const { return hash == o.hash; }
    bool operator!=(const GraphSignature &o) const { return hash != o.hash; }
};

struct GraphSignatureHash {
    size_t operator()(const GraphSignature &s) const {
        return static_cast<size_t>(s.hash);
    }
};

// Compute a structural signature of the graph rooted at `root`.
// Two graphs with identical structure (ops, shapes, dtypes, strides,
// params) produce the same signature regardless of actual data values.
GraphSignature compute_signature(const GraphNode *root);

} // namespace graph
} // namespace axiom
