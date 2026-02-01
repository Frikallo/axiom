#pragma once

// SIMD Kernel Functors for Runtime Dispatch
//
// These kernels are written ONCE and compiled multiple times with different
// architecture flags (-msse2, -mavx2, etc.). The compiler generates optimal
// code for each target architecture from the same source.
//
// Pattern from xsimd docs:
// 1. Define functor with template<Arch, ...> operator()
// 2. Do NOT use in-class definitions (bypasses extern template)
// 3. Use extern template declarations to prevent inline instantiation
// 4. Explicit instantiation in separate .cpp files compiled with arch flags

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <xsimd/xsimd.hpp>

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Binary Operation Kernels
// ============================================================================

struct BinaryAdd {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinarySub {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryMul {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryDiv {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryMax {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

struct BinaryMin {
    template <class Arch, typename T>
    void operator()(Arch, const T *a, const T *b, T *result, size_t n);
};

// ============================================================================
// Unary Operation Kernels
// ============================================================================

struct UnaryNeg {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryAbs {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnarySqrt {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryExp {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryLog {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnarySin {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryCos {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct UnaryTanh {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

// ============================================================================
// Reduction Kernels
// ============================================================================

struct ReduceSum {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

struct ReduceMax {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

struct ReduceMin {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

struct ReduceProd {
    template <class Arch, typename T>
    T operator()(Arch, const T *data, size_t n);
};

// ============================================================================
// Activation Kernels
// ============================================================================

struct ActivationReLU {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct ActivationSigmoid {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

struct ActivationGELU {
    template <class Arch, typename T>
    void operator()(Arch, const T *input, T *output, size_t n);
};

// ============================================================================
// Kernel Implementations
// ============================================================================

// --- Binary Operations ---

template <class Arch, typename T>
void BinaryAdd::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va + vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

template <class Arch, typename T>
void BinarySub::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va - vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}

template <class Arch, typename T>
void BinaryMul::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va * vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] * b[i];
    }
}

template <class Arch, typename T>
void BinaryDiv::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        (va / vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = a[i] / b[i];
    }
}

template <class Arch, typename T>
void BinaryMax::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::max(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::max(a[i], b[i]);
    }
}

template <class Arch, typename T>
void BinaryMin::operator()(Arch, const T *a, const T *b, T *result, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto va = batch::load_unaligned(a + i);
        auto vb = batch::load_unaligned(b + i);
        xsimd::min(va, vb).store_unaligned(result + i);
    }
    for (; i < n; ++i) {
        result[i] = std::min(a[i], b[i]);
    }
}

// --- Unary Operations ---

template <class Arch, typename T>
void UnaryNeg::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        (-v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = -input[i];
    }
}

template <class Arch, typename T>
void UnaryAbs::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::abs(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::abs(input[i]);
    }
}

template <class Arch, typename T>
void UnarySqrt::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::sqrt(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::sqrt(input[i]);
    }
}

template <class Arch, typename T>
void UnaryExp::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::exp(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::exp(input[i]);
    }
}

template <class Arch, typename T>
void UnaryLog::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::log(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::log(input[i]);
    }
}

template <class Arch, typename T>
void UnarySin::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::sin(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::sin(input[i]);
    }
}

template <class Arch, typename T>
void UnaryCos::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::cos(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::cos(input[i]);
    }
}

template <class Arch, typename T>
void UnaryTanh::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::tanh(v).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

// --- Reductions ---

template <class Arch, typename T>
T ReduceSum::operator()(Arch, const T *data, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch acc(static_cast<T>(0));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        acc += batch::load_unaligned(data + i);
    }

    T result = xsimd::reduce_add(acc);
    for (; i < n; ++i) {
        result += data[i];
    }
    return result;
}

template <class Arch, typename T>
T ReduceMax::operator()(Arch, const T *data, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::lowest();

    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;
    T result;

    if (n >= width) {
        batch acc = batch::load_unaligned(data);
        for (i = width; i + width <= n; i += width) {
            acc = xsimd::max(acc, batch::load_unaligned(data + i));
        }
        result = xsimd::reduce_max(acc);
    } else {
        result = data[0];
        i = 1;
    }

    for (; i < n; ++i) {
        result = std::max(result, data[i]);
    }
    return result;
}

template <class Arch, typename T>
T ReduceMin::operator()(Arch, const T *data, size_t n) {
    if (n == 0)
        return std::numeric_limits<T>::max();

    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    size_t i = 0;
    T result;

    if (n >= width) {
        batch acc = batch::load_unaligned(data);
        for (i = width; i + width <= n; i += width) {
            acc = xsimd::min(acc, batch::load_unaligned(data + i));
        }
        result = xsimd::reduce_min(acc);
    } else {
        result = data[0];
        i = 1;
    }

    for (; i < n; ++i) {
        result = std::min(result, data[i]);
    }
    return result;
}

template <class Arch, typename T>
T ReduceProd::operator()(Arch, const T *data, size_t n) {
    if (n == 0)
        return static_cast<T>(1);

    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch acc(static_cast<T>(1));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        acc *= batch::load_unaligned(data + i);
    }

    // Horizontal product
    alignas(Arch::alignment()) T temp[width];
    acc.store_aligned(temp);
    T result = static_cast<T>(1);
    for (size_t j = 0; j < width; ++j) {
        result *= temp[j];
    }

    for (; i < n; ++i) {
        result *= data[i];
    }
    return result;
}

// --- Activations ---

template <class Arch, typename T>
void ActivationReLU::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch zero(static_cast<T>(0));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        xsimd::max(v, zero).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = input[i] > T{0} ? input[i] : T{0};
    }
}

template <class Arch, typename T>
void ActivationSigmoid::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    batch one(static_cast<T>(1));
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto v = batch::load_unaligned(input + i);
        (one / (one + xsimd::exp(-v))).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        output[i] = T{1} / (T{1} + std::exp(-input[i]));
    }
}

template <class Arch, typename T>
void ActivationGELU::operator()(Arch, const T *input, T *output, size_t n) {
    using batch = xsimd::batch<T, Arch>;
    constexpr size_t width = batch::size;
    constexpr T sqrt_2_over_pi = T{0.7978845608028654};
    constexpr T coeff = T{0.044715};
    batch half(static_cast<T>(0.5));
    batch one(static_cast<T>(1));
    batch sqrt2pi(sqrt_2_over_pi);
    batch c(coeff);
    size_t i = 0;

    for (; i + width <= n; i += width) {
        auto x = batch::load_unaligned(input + i);
        auto inner = sqrt2pi * (x + c * x * x * x);
        (half * x * (one + xsimd::tanh(inner))).store_unaligned(output + i);
    }
    for (; i < n; ++i) {
        T x = input[i];
        output[i] =
            T{0.5} * x *
            (T{1} + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
    }
}

// ============================================================================
// Extern Template Declarations
// ============================================================================
// These prevent the compiler from instantiating templates in every TU.
// Actual instantiations are in simd_kernels_*.cpp files compiled with
// architecture-specific flags.

#ifdef AXIOM_SIMD_MULTI_ARCH

// --- x86-64 architectures ---

// SSE2 (baseline)
extern template void BinaryAdd::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryAdd::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void BinarySub::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinarySub::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryMul::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryMul::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryDiv::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryDiv::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                           const double *, double *, size_t);
extern template void UnaryExp::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryExp::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnaryLog::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryLog::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnarySqrt::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                               const float *,
                                                               float *, size_t);
extern template void UnarySqrt::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                                const double *,
                                                                double *,
                                                                size_t);
extern template float
ReduceSum::operator()<xsimd::sse2, float>(xsimd::sse2, const float *, size_t);
extern template double
ReduceSum::operator()<xsimd::sse2, double>(xsimd::sse2, const double *, size_t);
extern template float
ReduceMax::operator()<xsimd::sse2, float>(xsimd::sse2, const float *, size_t);
extern template double
ReduceMax::operator()<xsimd::sse2, double>(xsimd::sse2, const double *, size_t);
extern template void
ActivationReLU::operator()<xsimd::sse2, float>(xsimd::sse2, const float *,
                                               float *, size_t);
extern template void
ActivationReLU::operator()<xsimd::sse2, double>(xsimd::sse2, const double *,
                                                double *, size_t);

// AVX2
extern template void BinaryAdd::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryAdd::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void BinarySub::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinarySub::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryMul::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryMul::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void BinaryDiv::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               const float *,
                                                               float *, size_t);
extern template void
BinaryDiv::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                           const double *, double *, size_t);
extern template void UnaryExp::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryExp::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnaryLog::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                              const float *,
                                                              float *, size_t);
extern template void UnaryLog::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                               const double *,
                                                               double *,
                                                               size_t);
extern template void UnarySqrt::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                               const float *,
                                                               float *, size_t);
extern template void UnarySqrt::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                                const double *,
                                                                double *,
                                                                size_t);
extern template float
ReduceSum::operator()<xsimd::avx2, float>(xsimd::avx2, const float *, size_t);
extern template double
ReduceSum::operator()<xsimd::avx2, double>(xsimd::avx2, const double *, size_t);
extern template float
ReduceMax::operator()<xsimd::avx2, float>(xsimd::avx2, const float *, size_t);
extern template double
ReduceMax::operator()<xsimd::avx2, double>(xsimd::avx2, const double *, size_t);
extern template void
ActivationReLU::operator()<xsimd::avx2, float>(xsimd::avx2, const float *,
                                               float *, size_t);
extern template void
ActivationReLU::operator()<xsimd::avx2, double>(xsimd::avx2, const double *,
                                                double *, size_t);

// AVX512BW
extern template void
BinaryAdd::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinaryAdd::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
BinarySub::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinarySub::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
BinaryMul::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinaryMul::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
BinaryDiv::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              const float *, float *, size_t);
extern template void BinaryDiv::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, const double *, double *, size_t);
extern template void
UnaryExp::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                             float *, size_t);
extern template void
UnaryExp::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                              double *, size_t);
extern template void
UnaryLog::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                             float *, size_t);
extern template void
UnaryLog::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                              double *, size_t);
extern template void
UnarySqrt::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              float *, size_t);
extern template void
UnarySqrt::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                               double *, size_t);
extern template float
ReduceSum::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              size_t);
extern template double
ReduceSum::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                               size_t);
extern template float
ReduceMax::operator()<xsimd::avx512bw, float>(xsimd::avx512bw, const float *,
                                              size_t);
extern template double
ReduceMax::operator()<xsimd::avx512bw, double>(xsimd::avx512bw, const double *,
                                               size_t);
extern template void ActivationReLU::operator()<xsimd::avx512bw, float>(
    xsimd::avx512bw, const float *, float *, size_t);
extern template void ActivationReLU::operator()<xsimd::avx512bw, double>(
    xsimd::avx512bw, const double *, double *, size_t);

#endif // AXIOM_SIMD_MULTI_ARCH

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
