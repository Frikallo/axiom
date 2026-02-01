// FMA4 kernel instantiations (AMD Bulldozer/Piledriver)
// Compile with: -mfma4

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

template void BinaryAdd::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryAdd::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinarySub::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinarySub::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMul::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMul::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryDiv::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryDiv::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMax::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMax::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMin::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMin::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void UnaryNeg::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnaryNeg::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnaryAbs::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnaryAbs::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnarySqrt::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *, float *,
                                                        size_t);
template void UnarySqrt::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         double *, size_t);
template void UnaryExp::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnaryExp::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnaryLog::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnaryLog::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnarySin::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnarySin::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCos::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCos::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnaryTanh::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *, float *,
                                                        size_t);
template void UnaryTanh::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         double *, size_t);
template float ReduceSum::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                         const float *, size_t);
template double
ReduceSum::operator()<xsimd::fma4, double>(xsimd::fma4, const double *, size_t);
template float ReduceMax::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                         const float *, size_t);
template double
ReduceMax::operator()<xsimd::fma4, double>(xsimd::fma4, const double *, size_t);
template float ReduceMin::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                         const float *, size_t);
template double
ReduceMin::operator()<xsimd::fma4, double>(xsimd::fma4, const double *, size_t);
template float
ReduceProd::operator()<xsimd::fma4, float>(xsimd::fma4, const float *, size_t);
template double ReduceProd::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                            const double *,
                                                            size_t);
template void ActivationReLU::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                             const float *,
                                                             float *, size_t);
template void ActivationReLU::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                              const double *,
                                                              double *, size_t);
template void ActivationSigmoid::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                                const float *,
                                                                float *,
                                                                size_t);
template void ActivationSigmoid::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                                 const double *,
                                                                 double *,
                                                                 size_t);
template void ActivationGELU::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                             const float *,
                                                             float *, size_t);
template void ActivationGELU::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                              const double *,
                                                              double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
