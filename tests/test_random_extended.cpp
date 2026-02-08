// Tests for expanded random module

#include "axiom_test_utils.hpp"

using namespace axiom;

TEST(RandomExtended, Rand) {
    auto t = Tensor::rand({3, 4});
    ASSERT_TRUE(t.shape() == Shape({3, 4})) << "Shape mismatch";
    ASSERT_TRUE(t.dtype() == DType::Float32) << "DType should be Float32";

    // Check values are in [0, 1)
    const float *data = t.typed_data<float>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= 0.0f && data[i] < 1.0f)
            << "rand values should be in [0, 1)";
    }
}

TEST(RandomExtended, RandFloat64) {
    auto t = Tensor::rand({5, 5}, DType::Float64);
    ASSERT_TRUE(t.dtype() == DType::Float64) << "DType should be Float64";

    const double *data = t.typed_data<double>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= 0.0 && data[i] < 1.0)
            << "rand float64 values should be in [0, 1)";
    }
}

TEST(RandomExtended, Uniform) {
    auto t = Tensor::uniform(5.0, 10.0, {100});
    const float *data = t.typed_data<float>();

    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= 5.0f && data[i] < 10.0f)
            << "uniform values should be in [5, 10)";
    }
}

TEST(RandomExtended, UniformNegative) {
    auto t = Tensor::uniform(-5.0, 5.0, {100});
    const float *data = t.typed_data<float>();

    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= -5.0f && data[i] < 5.0f)
            << "uniform values should be in [-5, 5)";
    }
}

TEST(RandomExtended, Randint) {
    auto t = Tensor::randint(0, 10, {100}, DType::Int64);
    const int64_t *data = t.typed_data<int64_t>();

    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= 0 && data[i] < 10)
            << "randint values should be in [0, 10)";
    }
}

TEST(RandomExtended, RandintInt32) {
    auto t = Tensor::randint(10, 20, {50}, DType::Int32);
    ASSERT_TRUE(t.dtype() == DType::Int32) << "DType should be Int32";

    const int32_t *data = t.typed_data<int32_t>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= 10 && data[i] < 20)
            << "randint int32 values should be in [10, 20)";
    }
}

TEST(RandomExtended, RandLike) {
    auto proto = Tensor::zeros({3, 4}, DType::Float64);
    auto t = Tensor::rand_like(proto);

    ASSERT_TRUE(t.shape() == proto.shape()) << "Shape should match prototype";
    ASSERT_TRUE(t.dtype() == proto.dtype()) << "DType should match prototype";

    const double *data = t.typed_data<double>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= 0.0 && data[i] < 1.0)
            << "rand_like values should be in [0, 1)";
    }
}

TEST(RandomExtended, RandnLike) {
    auto proto = Tensor::zeros({5, 5});
    auto t = Tensor::randn_like(proto);

    ASSERT_TRUE(t.shape() == proto.shape()) << "Shape should match prototype";
    ASSERT_TRUE(t.dtype() == proto.dtype()) << "DType should match prototype";
}

TEST(RandomExtended, RandintLike) {
    auto proto = Tensor::zeros({10}, DType::Int32);
    auto t = Tensor::randint_like(proto, 0, 100);

    ASSERT_TRUE(t.shape() == proto.shape()) << "Shape should match prototype";
    ASSERT_TRUE(t.dtype() == proto.dtype()) << "DType should match prototype";

    const int32_t *data = t.typed_data<int32_t>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(data[i] >= 0 && data[i] < 100)
            << "randint_like values should be in [0, 100)";
    }
}
