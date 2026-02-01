// SSSE3 kernel instantiations
// Compile with: -mssse3 (Core 2 2006+)

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

template void BinaryAdd::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryAdd::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void BinarySub::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinarySub::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void BinaryMul::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryMul::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void BinaryDiv::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryDiv::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void BinaryMax::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryMax::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void BinaryMin::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryMin::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void UnaryNeg::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                        const float *, float *,
                                                        size_t);
template void UnaryNeg::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                         const double *,
                                                         double *, size_t);
template void UnaryAbs::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                        const float *, float *,
                                                        size_t);
template void UnaryAbs::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                         const double *,
                                                         double *, size_t);
template void UnarySqrt::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *, float *,
                                                         size_t);
template void UnarySqrt::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          double *, size_t);
template void UnaryExp::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                        const float *, float *,
                                                        size_t);
template void UnaryExp::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                         const double *,
                                                         double *, size_t);
template void UnaryLog::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                        const float *, float *,
                                                        size_t);
template void UnaryLog::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                         const double *,
                                                         double *, size_t);
template void UnarySin::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                        const float *, float *,
                                                        size_t);
template void UnarySin::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                         const double *,
                                                         double *, size_t);
template void UnaryCos::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCos::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                         const double *,
                                                         double *, size_t);
template void UnaryTanh::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                         const float *, float *,
                                                         size_t);
template void UnaryTanh::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                          const double *,
                                                          double *, size_t);
template float
ReduceSum::operator()<xsimd::ssse3, float>(xsimd::ssse3, const float *, size_t);
template double ReduceSum::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                            const double *,
                                                            size_t);
template float
ReduceMax::operator()<xsimd::ssse3, float>(xsimd::ssse3, const float *, size_t);
template double ReduceMax::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                            const double *,
                                                            size_t);
template float
ReduceMin::operator()<xsimd::ssse3, float>(xsimd::ssse3, const float *, size_t);
template double ReduceMin::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                            const double *,
                                                            size_t);
template float ReduceProd::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                           const float *,
                                                           size_t);
template double ReduceProd::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                             const double *,
                                                             size_t);
template void ActivationReLU::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                              const float *,
                                                              float *, size_t);
template void ActivationReLU::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                               const double *,
                                                               double *,
                                                               size_t);
template void ActivationSigmoid::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                                 const float *,
                                                                 float *,
                                                                 size_t);
template void ActivationSigmoid::operator()<xsimd::ssse3, double>(
    xsimd::ssse3, const double *, double *, size_t);
template void ActivationGELU::operator()<xsimd::ssse3, float>(xsimd::ssse3,
                                                              const float *,
                                                              float *, size_t);
template void ActivationGELU::operator()<xsimd::ssse3, double>(xsimd::ssse3,
                                                               const double *,
                                                               double *,
                                                               size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
