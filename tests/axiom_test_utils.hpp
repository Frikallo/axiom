#pragma once

#include <axiom/axiom.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

namespace axiom {
namespace testing {

// ============================================================================
// Global environment: initializes the operation registry once
// ============================================================================

class AxiomEnvironment : public ::testing::Environment {
  public:
    void SetUp() override {
        ops::OperationRegistry::initialize_builtin_operations();
    }
};

// ============================================================================
// GPU test helpers
// ============================================================================

#define SKIP_IF_NO_GPU()                                                       \
    do {                                                                        \
        if (!axiom::system::should_run_gpu_tests()) {                          \
            GTEST_SKIP() << "GPU tests disabled";                              \
        }                                                                      \
    } while (0)

// Fixture that auto-skips when no GPU is available
class GpuTest : public ::testing::Test {
  protected:
    void SetUp() override { SKIP_IF_NO_GPU(); }
};

// ============================================================================
// Tensor comparison utilities
// ============================================================================

template <typename T>
void ExpectTensorEquals(const Tensor &t,
                        const std::vector<T> &expected_data,
                        double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT_EQ(t_cpu.device(), Device::CPU) << "Tensor is not on CPU";
    ASSERT_EQ(t_cpu.size(), expected_data.size()) << "Tensor size mismatch";

    if (t_cpu.is_contiguous()) {
        const T *t_data = t_cpu.template typed_data<T>();
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                EXPECT_NEAR(static_cast<double>(t_data[i]),
                            static_cast<double>(expected_data[i]), epsilon)
                    << "Tensor data mismatch at index " << i;
            } else {
                EXPECT_EQ(t_data[i], expected_data[i])
                    << "Tensor data mismatch at index " << i;
            }
        }
    } else {
        // Non-contiguous tensors (views, slices, expanded): use item()
        std::vector<size_t> indices(t_cpu.ndim(), 0);
        for (size_t i = 0; i < expected_data.size(); ++i) {
            T val = t_cpu.template item<T>(indices);
            if constexpr (std::is_floating_point_v<T>) {
                EXPECT_NEAR(static_cast<double>(val),
                            static_cast<double>(expected_data[i]), epsilon)
                    << "Tensor data mismatch at index " << i;
            } else {
                EXPECT_EQ(val, expected_data[i])
                    << "Tensor data mismatch at index " << i;
            }
            for (int j = t_cpu.ndim() - 1; j >= 0; --j) {
                if (++indices[j] < t_cpu.shape()[j]) {
                    break;
                }
                indices[j] = 0;
            }
        }
    }
}

template <typename T>
void AssertTensorEquals(const Tensor &t,
                        const std::vector<T> &expected_data,
                        double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT_EQ(t_cpu.device(), Device::CPU) << "Tensor is not on CPU";
    ASSERT_EQ(t_cpu.size(), expected_data.size()) << "Tensor size mismatch";

    if (t_cpu.is_contiguous()) {
        const T *t_data = t_cpu.template typed_data<T>();
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if constexpr (std::is_floating_point_v<T>) {
                ASSERT_NEAR(static_cast<double>(t_data[i]),
                            static_cast<double>(expected_data[i]), epsilon)
                    << "Tensor data mismatch at index " << i;
            } else {
                ASSERT_EQ(t_data[i], expected_data[i])
                    << "Tensor data mismatch at index " << i;
            }
        }
    } else {
        // Non-contiguous tensors (views, slices, expanded): use item()
        std::vector<size_t> indices(t_cpu.ndim(), 0);
        for (size_t i = 0; i < expected_data.size(); ++i) {
            T val = t_cpu.template item<T>(indices);
            if constexpr (std::is_floating_point_v<T>) {
                ASSERT_NEAR(static_cast<double>(val),
                            static_cast<double>(expected_data[i]), epsilon)
                    << "Tensor data mismatch at index " << i;
            } else {
                ASSERT_EQ(val, expected_data[i])
                    << "Tensor data mismatch at index " << i;
            }
            for (int j = t_cpu.ndim() - 1; j >= 0; --j) {
                if (++indices[j] < t_cpu.shape()[j]) {
                    break;
                }
                indices[j] = 0;
            }
        }
    }
}

inline void ExpectTensorsClose(const Tensor &a, const Tensor &b,
                               double rtol = 1e-5, double atol = 1e-8) {
    EXPECT_TRUE(a.allclose(b, rtol, atol))
        << "Tensors not close (rtol=" << rtol << ", atol=" << atol << ")";
}

inline void AssertTensorsClose(const Tensor &a, const Tensor &b,
                               double rtol = 1e-5, double atol = 1e-8) {
    ASSERT_TRUE(a.allclose(b, rtol, atol))
        << "Tensors not close (rtol=" << rtol << ", atol=" << atol << ")";
}

// ============================================================================
// Device-parameterized test support
// ============================================================================

inline std::string DeviceName(
    const ::testing::TestParamInfo<Device> &info) {
    return info.param == Device::CPU ? "CPU" : "GPU";
}

// ============================================================================
// Typed test support — type lists mirroring dispatch.hpp categories
// ============================================================================

using AllFloatTypes = ::testing::Types<Float16, BFloat16, Float32, Float64>;
using StandardFloatTypes = ::testing::Types<Float32, Float64>;
using SignedIntTypes = ::testing::Types<Int8, Int16, Int32, Int64>;
using AllIntTypes = ::testing::Types<Bool, Int8, Int16, Int32, Int64, UInt8,
                                     UInt16, UInt32, UInt64>;
using NumericTypes =
    ::testing::Types<Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                     Float16, BFloat16, Float32, Float64>;
using AllTypes =
    ::testing::Types<Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32,
                     UInt64, Float16, BFloat16, Float32, Float64, Complex64,
                     Complex128>;

// Produces readable test names like "Float32" instead of mangled type names
struct AxiomTypeName {
    template <typename T> static std::string GetName(int) {
        return T::name();
    }
};

// Per-dtype absolute tolerance for floating-point comparisons
template <typename DT> constexpr double default_atol() {
    using T = typename DT::value_type;
    if constexpr (std::is_same_v<T, float16_t> ||
                  std::is_same_v<T, bfloat16_t>)
        return 1e-2;
    else if constexpr (std::is_same_v<T, float>)
        return 1e-5;
    else if constexpr (std::is_same_v<T, double>)
        return 1e-10;
    else
        return 0.0;
}

// Base fixture for dtype-parameterized tests
template <typename DTypeClass> class TypedTensorTest : public ::testing::Test {
  protected:
    using DT = DTypeClass;
    using value_type = typename DT::value_type;
    static constexpr DType dtype = dtype_of_v<value_type>;

    void assert_tensors_close(const Tensor &a, const Tensor &b,
                              double atol = default_atol<DTypeClass>()) {
        auto a32 = a.astype(DType::Float32);
        auto b32 = b.astype(DType::Float32);
        ASSERT_TRUE(a32.allclose(b32, atol, atol))
            << "Tensors not close for dtype " << DT::name();
    }
};

} // namespace testing
} // namespace axiom
