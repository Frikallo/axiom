#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

namespace axiom {

// Forward declaration
class Tensor;

struct Slice {
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step;

    Slice(std::optional<int64_t> start_val = std::nullopt,
          std::optional<int64_t> stop_val = std::nullopt,
          std::optional<int64_t> step_val = std::nullopt)
        : start(start_val), stop(stop_val), step(step_val) {}
};

// Tensor-based indexing for boolean masks or integer indices
struct TensorIndex {
    std::shared_ptr<Tensor> indices; // Can be bool mask or integer indices

    // Constructors
    TensorIndex() = default;
    explicit TensorIndex(const Tensor &t);
    TensorIndex(const TensorIndex &other) = default;
    TensorIndex(TensorIndex &&other) noexcept = default;
    TensorIndex &operator=(const TensorIndex &other) = default;
    TensorIndex &operator=(TensorIndex &&other) noexcept = default;
};

// Represents a single index: integer, slice, or tensor-based
using Index = std::variant<int64_t, Slice, TensorIndex>;

// Proxy class for masked tensor access with read/write support
// Enables syntax like: tensor[mask] = value
class MaskedView {
  private:
    Tensor *parent_;
    std::shared_ptr<Tensor> mask_;
    bool is_const_;

  public:
    MaskedView(Tensor &parent, const Tensor &mask);
    MaskedView(const Tensor &parent, const Tensor &mask);

    // Read: convert to tensor (returns 1D tensor of selected elements)
    operator Tensor() const;

    // Write: set masked elements to scalar value
    MaskedView &operator=(float value);
    MaskedView &operator=(double value);
    MaskedView &operator=(int32_t value);
    MaskedView &operator=(int64_t value);

    // Write: set masked elements from tensor
    MaskedView &operator=(const Tensor &values);

    // Get the selected elements as 1D tensor
    Tensor select() const;

    // Get the mask
    const Tensor &mask() const;

    // Get the parent tensor
    const Tensor &parent() const;
};

} // namespace axiom