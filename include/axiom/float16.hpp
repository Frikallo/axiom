#pragma once

#include <cmath>
#include <cstdint>
#include <fp16.h>
#include <limits>

namespace axiom {

class float16_t {
    uint16_t data_;

  public:
    float16_t() = default;
    float16_t(float f) : data_(fp16_ieee_from_fp32_value(f)) {}
    explicit float16_t(double d) : float16_t(static_cast<float>(d)) {}
    explicit float16_t(bool b) : float16_t(static_cast<float>(b)) {}
    explicit float16_t(int8_t i) : float16_t(static_cast<float>(i)) {}
    explicit float16_t(int16_t i) : float16_t(static_cast<float>(i)) {}
    explicit float16_t(int32_t i) : float16_t(static_cast<float>(i)) {}
    explicit float16_t(int64_t i) : float16_t(static_cast<float>(i)) {}
    explicit float16_t(uint8_t i) : float16_t(static_cast<float>(i)) {}
    explicit float16_t(uint16_t i) : float16_t(static_cast<float>(i)) {}
    explicit float16_t(uint32_t i) : float16_t(static_cast<float>(i)) {}
    explicit float16_t(uint64_t i) : float16_t(static_cast<float>(i)) {}

    operator float() const { return fp16_ieee_to_fp32_value(data_); }

    uint16_t bits() const { return data_; }
    static float16_t from_bits(uint16_t b) {
        float16_t h;
        h.data_ = b;
        return h;
    }

    // Arithmetic operators
    float16_t operator+(float16_t rhs) const {
        return float16_t(float(*this) + float(rhs));
    }
    float16_t operator-(float16_t rhs) const {
        return float16_t(float(*this) - float(rhs));
    }
    float16_t operator*(float16_t rhs) const {
        return float16_t(float(*this) * float(rhs));
    }
    float16_t operator/(float16_t rhs) const {
        return float16_t(float(*this) / float(rhs));
    }
    float16_t operator-() const { return float16_t(-float(*this)); }

    // Compound assignment
    float16_t &operator+=(float16_t rhs) {
        *this = *this + rhs;
        return *this;
    }
    float16_t &operator-=(float16_t rhs) {
        *this = *this - rhs;
        return *this;
    }
    float16_t &operator*=(float16_t rhs) {
        *this = *this * rhs;
        return *this;
    }
    float16_t &operator/=(float16_t rhs) {
        *this = *this / rhs;
        return *this;
    }

    // Comparisons
    bool operator==(float16_t rhs) const { return data_ == rhs.data_; }
    bool operator!=(float16_t rhs) const { return data_ != rhs.data_; }
    bool operator<(float16_t rhs) const { return float(*this) < float(rhs); }
    bool operator<=(float16_t rhs) const { return float(*this) <= float(rhs); }
    bool operator>(float16_t rhs) const { return float(*this) > float(rhs); }
    bool operator>=(float16_t rhs) const { return float(*this) >= float(rhs); }
};

} // namespace axiom

// std::numeric_limits specialization
namespace std {
template <> class numeric_limits<axiom::float16_t> {
  public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;

    static axiom::float16_t min() noexcept {
        return axiom::float16_t::from_bits(0x0400);
    }
    static axiom::float16_t max() noexcept {
        return axiom::float16_t::from_bits(0x7BFF);
    }
    static axiom::float16_t lowest() noexcept {
        return axiom::float16_t::from_bits(0xFBFF);
    }
    static axiom::float16_t epsilon() noexcept {
        return axiom::float16_t::from_bits(0x1400);
    }
    static axiom::float16_t infinity() noexcept {
        return axiom::float16_t::from_bits(0x7C00);
    }
    static axiom::float16_t quiet_NaN() noexcept {
        return axiom::float16_t::from_bits(0x7FFF);
    }
    static axiom::float16_t denorm_min() noexcept {
        return axiom::float16_t::from_bits(0x0001);
    }
};
} // namespace std
