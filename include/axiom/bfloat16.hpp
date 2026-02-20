#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace axiom {

class bfloat16_t {
    uint16_t data_;

  public:
    bfloat16_t() = default;

    bfloat16_t(float f) {
        // Handle NaN: propagate any NaN (quiet or signaling)
        if (std::isnan(f)) {
            data_ = 0x7FC0; // Canonical BFloat16 quiet NaN
            return;
        }
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // Round-to-nearest-even: add rounding bias
        bits += 0x7FFF + ((bits >> 16) & 1);
        data_ = static_cast<uint16_t>(bits >> 16);
    }

    explicit bfloat16_t(double d) : bfloat16_t(static_cast<float>(d)) {}
    explicit bfloat16_t(bool b) : bfloat16_t(static_cast<float>(b)) {}
    explicit bfloat16_t(int8_t i) : bfloat16_t(static_cast<float>(i)) {}
    explicit bfloat16_t(int16_t i) : bfloat16_t(static_cast<float>(i)) {}
    explicit bfloat16_t(int32_t i) : bfloat16_t(static_cast<float>(i)) {}
    explicit bfloat16_t(int64_t i) : bfloat16_t(static_cast<float>(i)) {}
    explicit bfloat16_t(uint8_t i) : bfloat16_t(static_cast<float>(i)) {}
    explicit bfloat16_t(uint16_t i) : bfloat16_t(static_cast<float>(i)) {}
    explicit bfloat16_t(uint32_t i) : bfloat16_t(static_cast<float>(i)) {}
    explicit bfloat16_t(uint64_t i) : bfloat16_t(static_cast<float>(i)) {}

    operator float() const {
        uint32_t bits = static_cast<uint32_t>(data_) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    uint16_t bits() const { return data_; }
    static bfloat16_t from_bits(uint16_t b) {
        bfloat16_t h;
        h.data_ = b;
        return h;
    }

    // Arithmetic operators
    bfloat16_t operator+(bfloat16_t rhs) const {
        return bfloat16_t(float(*this) + float(rhs));
    }
    bfloat16_t operator-(bfloat16_t rhs) const {
        return bfloat16_t(float(*this) - float(rhs));
    }
    bfloat16_t operator*(bfloat16_t rhs) const {
        return bfloat16_t(float(*this) * float(rhs));
    }
    bfloat16_t operator/(bfloat16_t rhs) const {
        return bfloat16_t(float(*this) / float(rhs));
    }
    bfloat16_t operator-() const { return bfloat16_t(-float(*this)); }

    // Compound assignment
    bfloat16_t &operator+=(bfloat16_t rhs) {
        *this = *this + rhs;
        return *this;
    }
    bfloat16_t &operator-=(bfloat16_t rhs) {
        *this = *this - rhs;
        return *this;
    }
    bfloat16_t &operator*=(bfloat16_t rhs) {
        *this = *this * rhs;
        return *this;
    }
    bfloat16_t &operator/=(bfloat16_t rhs) {
        *this = *this / rhs;
        return *this;
    }

    // Comparisons
    bool operator==(bfloat16_t rhs) const { return data_ == rhs.data_; }
    bool operator!=(bfloat16_t rhs) const { return data_ != rhs.data_; }
    bool operator<(bfloat16_t rhs) const { return float(*this) < float(rhs); }
    bool operator<=(bfloat16_t rhs) const { return float(*this) <= float(rhs); }
    bool operator>(bfloat16_t rhs) const { return float(*this) > float(rhs); }
    bool operator>=(bfloat16_t rhs) const { return float(*this) >= float(rhs); }
};

} // namespace axiom

// std::numeric_limits specialization
namespace std {
template <> class numeric_limits<axiom::bfloat16_t> {
  public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2;

    static axiom::bfloat16_t min() noexcept {
        return axiom::bfloat16_t::from_bits(0x0080); // Smallest normal
    }
    static axiom::bfloat16_t max() noexcept {
        return axiom::bfloat16_t::from_bits(0x7F7F); // ~3.39e38
    }
    static axiom::bfloat16_t lowest() noexcept {
        return axiom::bfloat16_t::from_bits(0xFF7F); // -max
    }
    static axiom::bfloat16_t epsilon() noexcept {
        return axiom::bfloat16_t::from_bits(0x3C00); // 2^-7 = 0.0078125
    }
    static axiom::bfloat16_t infinity() noexcept {
        return axiom::bfloat16_t::from_bits(0x7F80);
    }
    static axiom::bfloat16_t quiet_NaN() noexcept {
        return axiom::bfloat16_t::from_bits(0x7FC0);
    }
    static axiom::bfloat16_t denorm_min() noexcept {
        return axiom::bfloat16_t::from_bits(0x0001);
    }
};
} // namespace std
