#include "axiom_test_utils.hpp"

#include <axiom/axiom.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>

using namespace axiom;

// ============================================================================
// BFloat16 type tests
// ============================================================================

TEST(BFloat16Type, ConversionRoundTrip) {
    // Normal values
    float values[] = {0.0f, 1.0f, -1.0f, 3.14f, -2.5f, 100.0f};
    for (float v : values) {
        bfloat16_t bf(v);
        float back = static_cast<float>(bf);
        // BFloat16 has 7-bit mantissa, so ~2 decimal digits of precision
        EXPECT_NEAR(back, v, std::abs(v) * 0.01f + 1e-3f)
            << "Round trip failed for " << v;
    }
}

TEST(BFloat16Type, SpecialValues) {
    // Zero
    bfloat16_t zero(0.0f);
    EXPECT_EQ(static_cast<float>(zero), 0.0f);
    EXPECT_EQ(zero.bits(), 0x0000);

    // Negative zero
    bfloat16_t neg_zero(-0.0f);
    EXPECT_EQ(neg_zero.bits(), 0x8000);

    // Infinity
    bfloat16_t inf(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(inf)));
    EXPECT_EQ(inf.bits(), 0x7F80);

    // Negative infinity
    bfloat16_t neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(neg_inf)));
    EXPECT_EQ(neg_inf.bits(), 0xFF80);

    // NaN
    bfloat16_t nan_val(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(nan_val)));
}

TEST(BFloat16Type, FromBits) {
    auto val = bfloat16_t::from_bits(0x3F80); // 1.0 in bfloat16
    EXPECT_EQ(static_cast<float>(val), 1.0f);

    auto val2 = bfloat16_t::from_bits(0x4000); // 2.0 in bfloat16
    EXPECT_EQ(static_cast<float>(val2), 2.0f);
}

TEST(BFloat16Type, Arithmetic) {
    bfloat16_t a(2.0f);
    bfloat16_t b(3.0f);

    EXPECT_NEAR(static_cast<float>(a + b), 5.0f, 0.01f);
    EXPECT_NEAR(static_cast<float>(a - b), -1.0f, 0.01f);
    EXPECT_NEAR(static_cast<float>(a * b), 6.0f, 0.01f);
    EXPECT_NEAR(static_cast<float>(a / b), 2.0f / 3.0f, 0.01f);
    EXPECT_NEAR(static_cast<float>(-a), -2.0f, 0.01f);
}

TEST(BFloat16Type, CompoundAssignment) {
    bfloat16_t a(2.0f);
    a += bfloat16_t(3.0f);
    EXPECT_NEAR(static_cast<float>(a), 5.0f, 0.01f);

    a -= bfloat16_t(1.0f);
    EXPECT_NEAR(static_cast<float>(a), 4.0f, 0.01f);

    a *= bfloat16_t(2.0f);
    EXPECT_NEAR(static_cast<float>(a), 8.0f, 0.01f);

    a /= bfloat16_t(4.0f);
    EXPECT_NEAR(static_cast<float>(a), 2.0f, 0.01f);
}

TEST(BFloat16Type, Comparisons) {
    bfloat16_t a(1.0f);
    bfloat16_t b(2.0f);
    bfloat16_t c(1.0f);

    EXPECT_TRUE(a == c);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a <= c);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(a >= c);
}

TEST(BFloat16Type, NumericLimits) {
    using limits = std::numeric_limits<bfloat16_t>;
    EXPECT_TRUE(limits::is_specialized);
    EXPECT_TRUE(limits::is_signed);
    EXPECT_FALSE(limits::is_integer);
    EXPECT_TRUE(limits::has_infinity);
    EXPECT_TRUE(limits::has_quiet_NaN);
    EXPECT_EQ(limits::digits, 8);

    EXPECT_TRUE(std::isinf(static_cast<float>(limits::infinity())));
    EXPECT_TRUE(std::isnan(static_cast<float>(limits::quiet_NaN())));
    EXPECT_GT(static_cast<float>(limits::max()), 1e38f);
    EXPECT_GT(static_cast<float>(limits::min()), 0.0f);
}

// ============================================================================
// DType infrastructure tests
// ============================================================================

TEST(BFloat16DType, BasicProperties) {
    EXPECT_EQ(dtype_size(DType::BFloat16), 2u);
    EXPECT_EQ(dtype_name(DType::BFloat16), "bfloat16");
    EXPECT_TRUE(is_floating_dtype(DType::BFloat16));
    EXPECT_FALSE(is_integer_dtype(DType::BFloat16));
    EXPECT_FALSE(is_complex_dtype(DType::BFloat16));
}

TEST(BFloat16DType, DTypeOf) {
    EXPECT_EQ(dtype_of_v<bfloat16_t>, DType::BFloat16);
}

// ============================================================================
// Tensor creation tests
// ============================================================================

TEST(BFloat16Tensor, Zeros) {
    auto t = Tensor::zeros({2, 3}, DType::BFloat16);
    EXPECT_EQ(t.dtype(), DType::BFloat16);
    EXPECT_EQ(t.shape(), Shape({2, 3}));
    // zeros uses memset(0), which correctly produces bf16 zero
    auto t_f32 = t.astype(DType::Float32);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(t_f32.typed_data<float>()[i], 0.0f);
    }
}

TEST(BFloat16Tensor, Ones) {
    auto t = Tensor::ones({2, 3}, DType::BFloat16);
    EXPECT_EQ(t.dtype(), DType::BFloat16);
    auto t_f32 = t.astype(DType::Float32);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_NEAR(t_f32.typed_data<float>()[i], 1.0f, 1e-6f);
    }
}

TEST(BFloat16Tensor, Eye) {
    auto t = Tensor::eye(3, DType::BFloat16);
    EXPECT_EQ(t.dtype(), DType::BFloat16);
    auto t_f32 = t.astype(DType::Float32);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(t_f32.item<float>({i, j}), expected, 1e-6f);
        }
    }
}

TEST(BFloat16Tensor, Randn) {
    auto t = Tensor::randn({100}, DType::BFloat16);
    EXPECT_EQ(t.dtype(), DType::BFloat16);
    EXPECT_EQ(t.size(), 100u);
    // Values should be roughly normally distributed
    auto t_f32 = t.astype(DType::Float32);
    float sum = 0;
    for (size_t i = 0; i < t.size(); ++i) {
        sum += t_f32.typed_data<float>()[i];
    }
    float mean = sum / 100.0f;
    EXPECT_NEAR(mean, 0.0f, 0.5f); // Should be roughly zero-mean
}

TEST(BFloat16Tensor, Uniform) {
    auto t = Tensor::uniform(0.0, 1.0, {100}, DType::BFloat16);
    EXPECT_EQ(t.dtype(), DType::BFloat16);
    auto t_f32 = t.astype(DType::Float32);
    for (size_t i = 0; i < t.size(); ++i) {
        float val = t_f32.typed_data<float>()[i];
        EXPECT_GE(val, 0.0f);
        // BFloat16 rounding can push values near 1.0 up to exactly 1.0
        EXPECT_LE(val, 1.0f);
    }
}

TEST(BFloat16Tensor, Full) {
    auto t = Tensor::full({2, 3}, bfloat16_t(3.5f));
    EXPECT_EQ(t.dtype(), DType::BFloat16);
    auto t_f32 = t.astype(DType::Float32);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_NEAR(t_f32.typed_data<float>()[i], 3.5f, 0.05f);
    }
}

// ============================================================================
// Type conversion tests
// ============================================================================

TEST(BFloat16Tensor, AsTypeFloat32) {
    auto t = Tensor::ones({2, 3}, DType::BFloat16);
    auto t_f32 = t.astype(DType::Float32);
    EXPECT_EQ(t_f32.dtype(), DType::Float32);
    for (size_t i = 0; i < t_f32.size(); ++i) {
        EXPECT_NEAR(t_f32.typed_data<float>()[i], 1.0f, 1e-6f);
    }
}

TEST(BFloat16Tensor, AsTypeFloat64) {
    auto t = Tensor::ones({2, 3}, DType::BFloat16);
    auto t_f64 = t.astype(DType::Float64);
    EXPECT_EQ(t_f64.dtype(), DType::Float64);
    for (size_t i = 0; i < t_f64.size(); ++i) {
        EXPECT_NEAR(t_f64.typed_data<double>()[i], 1.0, 1e-6);
    }
}

TEST(BFloat16Tensor, AsTypeFloat16) {
    auto t = Tensor::ones({2, 3}, DType::BFloat16);
    auto t_f16 = t.astype(DType::Float16);
    EXPECT_EQ(t_f16.dtype(), DType::Float16);
}

TEST(BFloat16Tensor, AsTypeInt32) {
    auto t_f32 = Tensor::full({3}, 42.0f);
    auto t_bf16 = t_f32.astype(DType::BFloat16);
    auto t_int = t_bf16.astype(DType::Int32);
    EXPECT_EQ(t_int.dtype(), DType::Int32);
    for (size_t i = 0; i < t_int.size(); ++i) {
        EXPECT_EQ(t_int.typed_data<int32_t>()[i], 42);
    }
}

TEST(BFloat16Tensor, AsTypeBool) {
    auto t = Tensor::ones({3}, DType::BFloat16);
    auto t_bool = t.astype(DType::Bool);
    EXPECT_EQ(t_bool.dtype(), DType::Bool);
    for (size_t i = 0; i < t_bool.size(); ++i) {
        EXPECT_TRUE(t_bool.typed_data<bool>()[i]);
    }
}

TEST(BFloat16Tensor, ConvenienceMethod) {
    auto t = Tensor::ones({2, 3}, DType::Float32);
    auto t_bf16 = t.bfloat16();
    EXPECT_EQ(t_bf16.dtype(), DType::BFloat16);
}

// ============================================================================
// Type promotion tests
// ============================================================================

TEST(BFloat16Tensor, PromoteBFloat16Float16) {
    // BFloat16 + Float16 → Float32
    DType result =
        type_conversion::promote_dtypes(DType::BFloat16, DType::Float16);
    EXPECT_EQ(result, DType::Float32);
}

TEST(BFloat16Tensor, PromoteBFloat16Float32) {
    DType result =
        type_conversion::promote_dtypes(DType::BFloat16, DType::Float32);
    EXPECT_EQ(result, DType::Float32);
}

TEST(BFloat16Tensor, PromoteBFloat16Int32) {
    DType result =
        type_conversion::promote_dtypes(DType::BFloat16, DType::Int32);
    EXPECT_EQ(result, DType::BFloat16);
}

TEST(BFloat16Tensor, PromoteBFloat16BFloat16) {
    DType result =
        type_conversion::promote_dtypes(DType::BFloat16, DType::BFloat16);
    EXPECT_EQ(result, DType::BFloat16);
}

// ============================================================================
// Binary operations tests
// ============================================================================

TEST(BFloat16Tensor, BinaryAdd) {
    auto a = Tensor::ones({2, 3}, DType::BFloat16);
    auto b = Tensor::ones({2, 3}, DType::BFloat16);
    auto c = a + b;
    EXPECT_EQ(c.dtype(), DType::BFloat16);
    auto c_f32 = c.astype(DType::Float32);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_NEAR(c_f32.typed_data<float>()[i], 2.0f, 0.01f);
    }
}

TEST(BFloat16Tensor, BinarySub) {
    auto a = Tensor::full({3}, bfloat16_t(5.0f));
    auto b = Tensor::full({3}, bfloat16_t(2.0f));
    auto c = a - b;
    EXPECT_EQ(c.dtype(), DType::BFloat16);
    auto c_f32 = c.astype(DType::Float32);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_NEAR(c_f32.typed_data<float>()[i], 3.0f, 0.01f);
    }
}

TEST(BFloat16Tensor, BinaryMul) {
    auto a = Tensor::full({3}, bfloat16_t(3.0f));
    auto b = Tensor::full({3}, bfloat16_t(4.0f));
    auto c = a * b;
    EXPECT_EQ(c.dtype(), DType::BFloat16);
    auto c_f32 = c.astype(DType::Float32);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_NEAR(c_f32.typed_data<float>()[i], 12.0f, 0.1f);
    }
}

TEST(BFloat16Tensor, BinaryDiv) {
    auto a = Tensor::full({3}, bfloat16_t(6.0f));
    auto b = Tensor::full({3}, bfloat16_t(2.0f));
    auto c = a / b;
    EXPECT_EQ(c.dtype(), DType::BFloat16);
    auto c_f32 = c.astype(DType::Float32);
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_NEAR(c_f32.typed_data<float>()[i], 3.0f, 0.01f);
    }
}

// ============================================================================
// Unary operations tests
// ============================================================================

TEST(BFloat16Tensor, UnaryAbs) {
    auto t = Tensor::full({3}, bfloat16_t(-2.0f));
    auto result = t.abs();
    EXPECT_EQ(result.dtype(), DType::BFloat16);
    auto r_f32 = result.astype(DType::Float32);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(r_f32.typed_data<float>()[i], 2.0f, 0.01f);
    }
}

TEST(BFloat16Tensor, UnarySqrt) {
    auto t = Tensor::full({3}, bfloat16_t(4.0f));
    auto result = t.sqrt();
    auto r_f32 = result.astype(DType::Float32);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(r_f32.typed_data<float>()[i], 2.0f, 0.05f);
    }
}

TEST(BFloat16Tensor, UnaryNeg) {
    auto t = Tensor::full({3}, bfloat16_t(3.0f));
    auto result = t.negative();
    auto r_f32 = result.astype(DType::Float32);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(r_f32.typed_data<float>()[i], -3.0f, 0.01f);
    }
}

// ============================================================================
// Reduction tests
// ============================================================================

TEST(BFloat16Tensor, Sum) {
    auto t = Tensor::ones({2, 3}, DType::BFloat16);
    auto s = t.sum();
    auto s_f32 = s.astype(DType::Float32);
    EXPECT_NEAR(s_f32.item<float>(), 6.0f, 0.1f);
}

TEST(BFloat16Tensor, Mean) {
    auto t = Tensor::ones({2, 3}, DType::BFloat16);
    auto m = t.mean();
    auto m_f32 = m.astype(DType::Float32);
    EXPECT_NEAR(m_f32.item<float>(), 1.0f, 0.01f);
}

TEST(BFloat16Tensor, Max) {
    float data[] = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t_f32 = Tensor::from_data(data, {4});
    auto t_bf16 = t_f32.astype(DType::BFloat16);
    auto max_val = t_bf16.max();
    auto max_f32 = max_val.astype(DType::Float32);
    EXPECT_NEAR(max_f32.item<float>(), 5.0f, 0.1f);
}

TEST(BFloat16Tensor, Min) {
    float data[] = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t_f32 = Tensor::from_data(data, {4});
    auto t_bf16 = t_f32.astype(DType::BFloat16);
    auto min_val = t_bf16.min();
    auto min_f32 = min_val.astype(DType::Float32);
    EXPECT_NEAR(min_f32.item<float>(), 1.0f, 0.1f);
}

// ============================================================================
// Safety checks
// ============================================================================

TEST(BFloat16Tensor, HasNan) {
    auto t = Tensor::ones({3}, DType::BFloat16);
    EXPECT_FALSE(t.has_nan());

    // Create a NaN value
    auto t_nan = Tensor::zeros({3}, DType::BFloat16);
    bfloat16_t *data = t_nan.typed_data<bfloat16_t>();
    data[1] = bfloat16_t(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(t_nan.has_nan());
}

TEST(BFloat16Tensor, HasInf) {
    auto t = Tensor::ones({3}, DType::BFloat16);
    EXPECT_FALSE(t.has_inf());

    auto t_inf = Tensor::zeros({3}, DType::BFloat16);
    bfloat16_t *data = t_inf.typed_data<bfloat16_t>();
    data[1] = bfloat16_t(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(t_inf.has_inf());
}

// ============================================================================
// I/O tests
// ============================================================================

TEST(BFloat16Tensor, FlatBuffersRoundTrip) {
    auto t = Tensor::ones({2, 3}, DType::BFloat16);
    std::string filename =
        (std::filesystem::temp_directory_path() / "test_bfloat16_tensor.axfb")
            .string();
    t.save(filename);

    auto loaded = Tensor::load(filename);
    EXPECT_EQ(loaded.dtype(), DType::BFloat16);
    EXPECT_EQ(loaded.shape(), Shape({2, 3}));

    auto loaded_f32 = loaded.astype(DType::Float32);
    for (size_t i = 0; i < loaded.size(); ++i) {
        EXPECT_NEAR(loaded_f32.typed_data<float>()[i], 1.0f, 1e-6f);
    }
}

// ============================================================================
// Repr / printing tests
// ============================================================================

TEST(BFloat16Tensor, Repr) {
    auto t = Tensor::zeros({2, 3}, DType::BFloat16);
    std::string repr = t.repr();
    EXPECT_NE(repr.find("bfloat16"), std::string::npos)
        << "repr should contain 'bfloat16', got: " << repr;
}

TEST(BFloat16Tensor, DtypeName) {
    auto t = Tensor::zeros({2}, DType::BFloat16);
    EXPECT_EQ(t.dtype_name(), "bfloat16");
}
