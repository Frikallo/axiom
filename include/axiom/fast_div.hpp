#pragma once

// Fast integer division using FXdiv
// Converts repeated divisions by the same divisor into multiplication + shifts
// Provides 2-5x speedup for index calculations in tensor operations

#include <cstddef>
#include <cstdint>
#include <vector>

#include <fxdiv.h>

namespace axiom {

// C++ wrapper for fxdiv_divisor_size_t with RAII semantics
class FastDivisor {
  public:
    FastDivisor() : divisor_{} { divisor_.value = 1; }

    explicit FastDivisor(size_t value) : divisor_(fxdiv_init_size_t(value)) {}

    // Get quotient only
    size_t divide(size_t n) const { return fxdiv_quotient_size_t(n, divisor_); }

    // Get remainder only
    size_t remainder(size_t n) const {
        return fxdiv_remainder_size_t(n, divisor_);
    }

    // Get both quotient and remainder (most efficient when both needed)
    struct DivResult {
        size_t quotient;
        size_t remainder;
    };

    DivResult divmod(size_t n) const {
        auto result = fxdiv_divide_size_t(n, divisor_);
        return {result.quotient, result.remainder};
    }

    size_t value() const { return divisor_.value; }

  private:
    fxdiv_divisor_size_t divisor_;
};

// Precomputed divisors for a shape - used for fast unravel_index
class ShapeDivisors {
  public:
    ShapeDivisors() = default;

    explicit ShapeDivisors(const std::vector<size_t> &shape) {
        divisors_.reserve(shape.size());
        for (size_t dim : shape) {
            divisors_.emplace_back(dim);
        }
    }

    // Fast unravel: convert linear index to multi-dimensional coordinates
    // Processes dimensions from last to first (row-major order)
    void unravel(size_t linear_idx, std::vector<size_t> &indices) const {
        size_t remaining = linear_idx;
        for (int i = static_cast<int>(divisors_.size()) - 1; i >= 0; --i) {
            auto result = divisors_[i].divmod(remaining);
            indices[i] = result.remainder;
            remaining = result.quotient;
        }
    }

    // Returns coordinates as a new vector
    std::vector<size_t> unravel(size_t linear_idx) const {
        std::vector<size_t> indices(divisors_.size());
        unravel(linear_idx, indices);
        return indices;
    }

    size_t ndim() const { return divisors_.size(); }

    const FastDivisor &operator[](size_t i) const { return divisors_[i]; }

  private:
    std::vector<FastDivisor> divisors_;
};

// Inline helper for single division when divisor is used only once
// (In this case, regular division may be faster due to init overhead)
inline size_t fast_div(size_t n, size_t d) {
    return fxdiv_quotient_size_t(n, fxdiv_init_size_t(d));
}

inline size_t fast_mod(size_t n, size_t d) {
    return fxdiv_remainder_size_t(n, fxdiv_init_size_t(d));
}

} // namespace axiom
