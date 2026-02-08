// Tests for random number generation

#include "axiom_test_utils.hpp"
#include <set>

using namespace axiom;

// Test randn creates tensors with expected shape
TEST(TensorRandom, RandnShape) {
    auto t = Tensor::randn({3, 4});
    ASSERT_TRUE(t.shape() == Shape({3, 4})) << "Shape mismatch";
    ASSERT_TRUE(t.dtype() == DType::Float32) << "DType should be Float32";
    ASSERT_TRUE(t.size() == 12) << "Size mismatch";
}

// Test randn creates different values
TEST(TensorRandom, RandnRandomness) {
    auto t1 = Tensor::randn({100});
    auto t2 = Tensor::randn({100});

    // Check that t1 and t2 are not identical
    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    int differences = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6) {
            differences++;
        }
    }

    ASSERT_TRUE(differences > 90) << "Random tensors should be different";
}

// Test manual_seed produces reproducible results
TEST(TensorRandom, ManualSeedReproducibility) {
    Tensor::manual_seed(42);
    auto t1 = Tensor::randn({10});

    Tensor::manual_seed(42);
    auto t2 = Tensor::randn({10});

    // Check that t1 and t2 are identical
    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(std::abs(data1[i] - data2[i]) < 1e-6)
            << "Seeded random should be reproducible";
    }
}

// Test manual_seed with different seeds produces different results
TEST(TensorRandom, ManualSeedDifferentSeeds) {
    Tensor::manual_seed(42);
    auto t1 = Tensor::randn({100});

    Tensor::manual_seed(123);
    auto t2 = Tensor::randn({100});

    // Check that t1 and t2 are different
    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    int differences = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6) {
            differences++;
        }
    }

    ASSERT_TRUE(differences > 90)
        << "Different seeds should produce different results";
}

// Test get_seed returns the current seed
TEST(TensorRandom, GetSeed) {
    uint64_t seed = 12345;
    manual_seed(seed);
    uint64_t retrieved_seed = get_seed();
    ASSERT_TRUE(retrieved_seed == seed)
        << "get_seed should return the set seed";
}

// Test randn statistical properties (rough check)
TEST(TensorRandom, RandnStatistics) {
    Tensor::manual_seed(42);
    auto t = Tensor::randn({10000});

    // Calculate mean and stddev
    const float *data = t.typed_data<float>();
    double sum = 0.0;
    for (size_t i = 0; i < 10000; ++i) {
        sum += data[i];
    }
    double mean = sum / 10000.0;

    double variance = 0.0;
    for (size_t i = 0; i < 10000; ++i) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= 10000.0;
    double stddev = std::sqrt(variance);

    // Check that mean is close to 0 and stddev is close to 1
    ASSERT_TRUE(std::abs(mean) < 0.05) << "Mean should be close to 0";
    ASSERT_TRUE(std::abs(stddev - 1.0) < 0.05) << "Stddev should be close to 1";
}

// Test randn on GPU if available
TEST(TensorRandom, RandnGpu) {
    SKIP_IF_NO_GPU();

    auto t = Tensor::randn({3, 4}, DType::Float32, Device::GPU);
    ASSERT_TRUE(t.device() == Device::GPU) << "Should be on GPU";
    ASSERT_TRUE(t.shape() == Shape({3, 4})) << "Shape mismatch";
}

// Test that free function manual_seed works
TEST(TensorRandom, FreeFunctionManualSeed) {
    manual_seed(999);
    auto t1 = Tensor::randn({10});

    manual_seed(999);
    auto t2 = Tensor::randn({10});

    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    for (size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(std::abs(data1[i] - data2[i]) < 1e-6)
            << "Free function manual_seed should work";
    }
}
