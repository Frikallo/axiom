// VSX kernel instantiations (PowerPC POWER7+)
// Compile with: -mvsx

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

template void BinaryAdd::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryAdd::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinarySub::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinarySub::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryMul::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryMul::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryDiv::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryDiv::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryMax::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryMax::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryMin::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryMin::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void UnaryNeg::operator()<xsimd::vsx, float>(xsimd::vsx, const float *,
                                                      float *, size_t);
template void UnaryNeg::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryAbs::operator()<xsimd::vsx, float>(xsimd::vsx, const float *,
                                                      float *, size_t);
template void UnaryAbs::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                       const double *, double *,
                                                       size_t);
template void UnarySqrt::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *, float *,
                                                       size_t);
template void UnarySqrt::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        double *, size_t);
template void UnaryExp::operator()<xsimd::vsx, float>(xsimd::vsx, const float *,
                                                      float *, size_t);
template void UnaryExp::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryLog::operator()<xsimd::vsx, float>(xsimd::vsx, const float *,
                                                      float *, size_t);
template void UnaryLog::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                       const double *, double *,
                                                       size_t);
template void UnarySin::operator()<xsimd::vsx, float>(xsimd::vsx, const float *,
                                                      float *, size_t);
template void UnarySin::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryCos::operator()<xsimd::vsx, float>(xsimd::vsx, const float *,
                                                      float *, size_t);
template void UnaryCos::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryTanh::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                       const float *, float *,
                                                       size_t);
template void UnaryTanh::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                        const double *,
                                                        double *, size_t);
template float ReduceSum::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                        const float *, size_t);
template double
ReduceSum::operator()<xsimd::vsx, double>(xsimd::vsx, const double *, size_t);
template float ReduceMax::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                        const float *, size_t);
template double
ReduceMax::operator()<xsimd::vsx, double>(xsimd::vsx, const double *, size_t);
template float ReduceMin::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                        const float *, size_t);
template double
ReduceMin::operator()<xsimd::vsx, double>(xsimd::vsx, const double *, size_t);
template float ReduceProd::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                         const float *, size_t);
template double
ReduceProd::operator()<xsimd::vsx, double>(xsimd::vsx, const double *, size_t);
template void ActivationReLU::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                            const float *,
                                                            float *, size_t);
template void ActivationReLU::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                             const double *,
                                                             double *, size_t);
template void ActivationSigmoid::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                               const float *,
                                                               float *, size_t);
template void ActivationSigmoid::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                                const double *,
                                                                double *,
                                                                size_t);
template void ActivationGELU::operator()<xsimd::vsx, float>(xsimd::vsx,
                                                            const float *,
                                                            float *, size_t);
template void ActivationGELU::operator()<xsimd::vsx, double>(xsimd::vsx,
                                                             const double *,
                                                             double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
