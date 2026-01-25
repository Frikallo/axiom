#include "axiom/random.hpp"

#include <random>

namespace axiom {

RandomGenerator::RandomGenerator() : current_seed_(0), seeded_(false) {}

RandomGenerator &RandomGenerator::instance() {
    static thread_local RandomGenerator gen;
    return gen;
}

void RandomGenerator::seed(uint64_t seed_val) {
    gen_.seed(seed_val);
    current_seed_ = seed_val;
    seeded_ = true;
}

void RandomGenerator::seed(uint64_t seed_val, uint64_t stream) {
    gen_.seed(seed_val, stream);
    current_seed_ = seed_val;
    seeded_ = true;
}

void RandomGenerator::seed_random() {
    std::random_device rd;
    uint64_t seed_val = (static_cast<uint64_t>(rd()) << 32) | rd();
    uint64_t stream = (static_cast<uint64_t>(rd()) << 32) | rd();
    gen_.seed(seed_val, stream);
    current_seed_ = seed_val;
    seeded_ = true;
}

PCG64 &RandomGenerator::generator() {
    if (!seeded_) {
        seed_random();
    }
    return gen_;
}

void manual_seed(uint64_t seed) { RandomGenerator::instance().seed(seed); }

uint64_t get_seed() { return RandomGenerator::instance().get_seed(); }

} // namespace axiom
