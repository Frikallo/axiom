#pragma once

#include <cstdint>
#include <limits>
#include <random>

namespace axiom {

// PCG-XSH-RR: Permuted Congruential Generator
// Fast, small state (128 bits), excellent statistical properties
// Used by NumPy, recommended for ML workloads
class PCG64 {
  public:
    using result_type = uint64_t;

    PCG64() : state_(0x853c49e6748fea9bULL), inc_(0xda3e39cb94b95bdbULL) {}

    explicit PCG64(uint64_t seed) { this->seed(seed); }

    PCG64(uint64_t seed, uint64_t stream) { this->seed(seed, stream); }

    void seed(uint64_t seed_val) { seed(seed_val, 1); }

    void seed(uint64_t seed_val, uint64_t stream) {
        state_ = 0;
        inc_ = (stream << 1) | 1;
        (*this)();
        state_ += seed_val;
        (*this)();
    }

    uint64_t operator()() {
        uint64_t old_state = state_;
        state_ = old_state * 6364136223846793005ULL + inc_;

        // PCG-XSH-RR output function
        uint32_t xorshifted =
            static_cast<uint32_t>(((old_state >> 18) ^ old_state) >> 27);
        uint32_t rot = static_cast<uint32_t>(old_state >> 59);
        uint32_t result32 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

        // Generate 64 bits from two 32-bit outputs
        uint64_t high = result32;

        old_state = state_;
        state_ = old_state * 6364136223846793005ULL + inc_;
        xorshifted =
            static_cast<uint32_t>(((old_state >> 18) ^ old_state) >> 27);
        rot = static_cast<uint32_t>(old_state >> 59);
        uint32_t low = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

        return (high << 32) | low;
    }

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() {
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t state() const { return state_; }
    uint64_t stream() const { return inc_ >> 1; }

  private:
    uint64_t state_;
    uint64_t inc_;
};

class RandomGenerator {
  public:
    static RandomGenerator &instance();

    void seed(uint64_t seed_val);
    void seed(uint64_t seed_val, uint64_t stream);

    void seed_random();

    uint64_t get_seed() const { return current_seed_; }

    PCG64 &generator();

    uint64_t random_uint64() { return generator()(); }

    double random_double() {
        return (generator()() >> 11) * (1.0 / 9007199254740992.0);
    }

    template <typename T> T normal(T mean = T(0), T stddev = T(1)) {
        std::normal_distribution<T> dist(mean, stddev);
        return dist(generator());
    }

    template <typename T> T uniform(T low = T(0), T high = T(1)) {
        std::uniform_real_distribution<T> dist(low, high);
        return dist(generator());
    }

    int64_t randint(int64_t low, int64_t high) {
        std::uniform_int_distribution<int64_t> dist(low, high - 1);
        return dist(generator());
    }

  private:
    RandomGenerator();
    RandomGenerator(const RandomGenerator &) = delete;
    RandomGenerator &operator=(const RandomGenerator &) = delete;

    PCG64 gen_;
    uint64_t current_seed_;
    bool seeded_;
};

void manual_seed(uint64_t seed);

uint64_t get_seed();

} // namespace axiom
