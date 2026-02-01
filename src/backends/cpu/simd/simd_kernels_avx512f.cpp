// AVX-512F kernel instantiations (Foundation)
// Compile with: -mavx512f (Skylake-X 2017+)

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

template void BinaryAdd::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           const float *,
                                                           float *, size_t);
template void BinaryAdd::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            const double *,
                                                            double *, size_t);
template void BinarySub::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           const float *,
                                                           float *, size_t);
template void BinarySub::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            const double *,
                                                            double *, size_t);
template void BinaryMul::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           const float *,
                                                           float *, size_t);
template void BinaryMul::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            const double *,
                                                            double *, size_t);
template void BinaryDiv::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           const float *,
                                                           float *, size_t);
template void BinaryDiv::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            const double *,
                                                            double *, size_t);
template void BinaryMax::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           const float *,
                                                           float *, size_t);
template void BinaryMax::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            const double *,
                                                            double *, size_t);
template void BinaryMin::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           const float *,
                                                           float *, size_t);
template void BinaryMin::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            const double *,
                                                            double *, size_t);
template void UnaryNeg::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                          const float *,
                                                          float *, size_t);
template void UnaryNeg::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                           const double *,
                                                           double *, size_t);
template void UnaryAbs::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                          const float *,
                                                          float *, size_t);
template void UnaryAbs::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                           const double *,
                                                           double *, size_t);
template void UnarySqrt::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           float *, size_t);
template void UnarySqrt::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            double *, size_t);
template void UnaryExp::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                          const float *,
                                                          float *, size_t);
template void UnaryExp::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                           const double *,
                                                           double *, size_t);
template void UnaryLog::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                          const float *,
                                                          float *, size_t);
template void UnaryLog::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                           const double *,
                                                           double *, size_t);
template void UnarySin::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                          const float *,
                                                          float *, size_t);
template void UnarySin::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                           const double *,
                                                           double *, size_t);
template void UnaryCos::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                          const float *,
                                                          float *, size_t);
template void UnaryCos::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                           const double *,
                                                           double *, size_t);
template void UnaryTanh::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                           const float *,
                                                           float *, size_t);
template void UnaryTanh::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                            const double *,
                                                            double *, size_t);
template float ReduceSum::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                            const float *,
                                                            size_t);
template double ReduceSum::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                              const double *,
                                                              size_t);
template float ReduceMax::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                            const float *,
                                                            size_t);
template double ReduceMax::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                              const double *,
                                                              size_t);
template float ReduceMin::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                            const float *,
                                                            size_t);
template double ReduceMin::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                              const double *,
                                                              size_t);
template float ReduceProd::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                             const float *,
                                                             size_t);
template double ReduceProd::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                               const double *,
                                                               size_t);
template void ActivationReLU::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                                const float *,
                                                                float *,
                                                                size_t);
template void ActivationReLU::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                                 const double *,
                                                                 double *,
                                                                 size_t);
template void ActivationSigmoid::operator()<xsimd::avx512f, float>(
    xsimd::avx512f, const float *, float *, size_t);
template void ActivationSigmoid::operator()<xsimd::avx512f, double>(
    xsimd::avx512f, const double *, double *, size_t);
template void ActivationGELU::operator()<xsimd::avx512f, float>(xsimd::avx512f,
                                                                const float *,
                                                                float *,
                                                                size_t);
template void ActivationGELU::operator()<xsimd::avx512f, double>(xsimd::avx512f,
                                                                 const double *,
                                                                 double *,
                                                                 size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
