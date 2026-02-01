// AVX kernel instantiations
// Compile with: -mavx (Sandy Bridge 2011+)

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

template void BinaryAdd::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryAdd::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinarySub::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinarySub::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryMul::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryMul::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryDiv::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryDiv::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryMax::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryMax::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryMin::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryMin::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void UnaryNeg::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnaryNeg::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryAbs::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnaryAbs::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnarySqrt::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *, float *,
                                                       size_t);
template void UnarySqrt::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        double *, size_t);
template void UnaryExp::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnaryExp::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryLog::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnaryLog::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnarySin::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnarySin::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryCos::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnaryCos::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryTanh::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *, float *,
                                                       size_t);
template void UnaryTanh::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        double *, size_t);
template float ReduceSum::operator()<xsimd::avx, float>(xsimd::avx,
                                                        const float *, size_t);
template double
ReduceSum::operator()<xsimd::avx, double>(xsimd::avx, const double *, size_t);
template float ReduceMax::operator()<xsimd::avx, float>(xsimd::avx,
                                                        const float *, size_t);
template double
ReduceMax::operator()<xsimd::avx, double>(xsimd::avx, const double *, size_t);
template float ReduceMin::operator()<xsimd::avx, float>(xsimd::avx,
                                                        const float *, size_t);
template double
ReduceMin::operator()<xsimd::avx, double>(xsimd::avx, const double *, size_t);
template float ReduceProd::operator()<xsimd::avx, float>(xsimd::avx,
                                                         const float *, size_t);
template double
ReduceProd::operator()<xsimd::avx, double>(xsimd::avx, const double *, size_t);
template void ActivationReLU::operator()<xsimd::avx, float>(xsimd::avx,
                                                            const float *,
                                                            float *, size_t);
template void ActivationReLU::operator()<xsimd::avx, double>(xsimd::avx,
                                                             const double *,
                                                             double *, size_t);
template void ActivationSigmoid::operator()<xsimd::avx, float>(xsimd::avx,
                                                               const float *,
                                                               float *, size_t);
template void ActivationSigmoid::operator()<xsimd::avx, double>(xsimd::avx,
                                                                const double *,
                                                                double *,
                                                                size_t);
template void ActivationGELU::operator()<xsimd::avx, float>(xsimd::avx,
                                                            const float *,
                                                            float *, size_t);
template void ActivationGELU::operator()<xsimd::avx, double>(xsimd::avx,
                                                             const double *,
                                                             double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
