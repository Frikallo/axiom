#pragma once

#include <cstdint>
#include <optional>
#include <variant>

namespace axiom {

struct Slice {
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step;

    Slice(std::optional<int64_t> start_val = std::nullopt,
          std::optional<int64_t> stop_val = std::nullopt,
          std::optional<int64_t> step_val = std::nullopt)
        : start(start_val), stop(stop_val), step(step_val) {}
};

// Represents a single index, which can be a single integer or a slice.
using Index = std::variant<int64_t, Slice>;

} // namespace axiom