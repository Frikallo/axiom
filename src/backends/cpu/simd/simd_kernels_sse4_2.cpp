// SSE4.2 kernel instantiations
// Compile with: -msse4.2 (Nehalem 2008+)

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

template void BinaryAdd::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryAdd::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinarySub::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinarySub::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryMul::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryMul::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryDiv::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryDiv::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryMax::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryMax::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryMin::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryMin::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void UnaryNeg::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryNeg::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryAbs::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryAbs::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                          const double *,
                                                          double *, size_t);
template void UnarySqrt::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          float *, size_t);
template void UnarySqrt::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           double *, size_t);
template void UnaryExp::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryExp::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryLog::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryLog::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                          const double *,
                                                          double *, size_t);
template void UnarySin::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                         const float *, float *,
                                                         size_t);
template void UnarySin::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryCos::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryCos::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryTanh::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                          const float *,
                                                          float *, size_t);
template void UnaryTanh::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                           const double *,
                                                           double *, size_t);
template float ReduceSum::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                           const float *,
                                                           size_t);
template double ReduceSum::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                             const double *,
                                                             size_t);
template float ReduceMax::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                           const float *,
                                                           size_t);
template double ReduceMax::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                             const double *,
                                                             size_t);
template float ReduceMin::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                           const float *,
                                                           size_t);
template double ReduceMin::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                             const double *,
                                                             size_t);
template float ReduceProd::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                            const float *,
                                                            size_t);
template double ReduceProd::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                              const double *,
                                                              size_t);
template void ActivationReLU::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                               const float *,
                                                               float *, size_t);
template void ActivationReLU::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                                const double *,
                                                                double *,
                                                                size_t);
template void ActivationSigmoid::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                                  const float *,
                                                                  float *,
                                                                  size_t);
template void ActivationSigmoid::operator()<xsimd::sse4_2, double>(
    xsimd::sse4_2, const double *, double *, size_t);
template void ActivationGELU::operator()<xsimd::sse4_2, float>(xsimd::sse4_2,
                                                               const float *,
                                                               float *, size_t);
template void ActivationGELU::operator()<xsimd::sse4_2, double>(xsimd::sse4_2,
                                                                const double *,
                                                                double *,
                                                                size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
