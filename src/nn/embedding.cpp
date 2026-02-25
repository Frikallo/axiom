#include "axiom/nn/embedding.hpp"

#include "axiom/error.hpp"
#include "axiom/operations.hpp"

namespace axiom::nn {

Embedding::Embedding() { register_parameter("weight", weight_); }

Tensor Embedding::forward(const Tensor &indices) const {
    if (!weight_.storage()) {
        throw RuntimeError("Embedding: weight not initialized (call "
                           "load_state_dict first)");
    }
    return ops::embedding(weight_, indices);
}

} // namespace axiom::nn
