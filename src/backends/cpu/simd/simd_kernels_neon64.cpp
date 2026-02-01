// NEON64 kernel instantiations (ARM64/AArch64)
// No special compile flags needed - NEON64 is always available on AArch64

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// Binary operations
template void BinaryAdd::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryAdd::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinarySub::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinarySub::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryMul::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryMul::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryDiv::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryDiv::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryMax::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryMax::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryMin::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryMin::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);

// Unary operations
template void UnaryNeg::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                         const float *, float *,
                                                         size_t);
template void UnaryNeg::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                          const double *,
                                                          double *, size_t);
template void UnaryAbs::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                         const float *, float *,
                                                         size_t);
template void UnaryAbs::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                          const double *,
                                                          double *, size_t);
template void UnarySqrt::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          float *, size_t);
template void UnarySqrt::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           double *, size_t);
template void UnaryExp::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                         const float *, float *,
                                                         size_t);
template void UnaryExp::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                          const double *,
                                                          double *, size_t);
template void UnaryLog::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                         const float *, float *,
                                                         size_t);
template void UnaryLog::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                          const double *,
                                                          double *, size_t);
template void UnarySin::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                         const float *, float *,
                                                         size_t);
template void UnarySin::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                          const double *,
                                                          double *, size_t);
template void UnaryCos::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                         const float *, float *,
                                                         size_t);
template void UnaryCos::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                          const double *,
                                                          double *, size_t);
template void UnaryTanh::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                          const float *,
                                                          float *, size_t);
template void UnaryTanh::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                           const double *,
                                                           double *, size_t);

// Reductions
template float ReduceSum::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                           const float *,
                                                           size_t);
template double ReduceSum::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                             const double *,
                                                             size_t);
template float ReduceMax::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                           const float *,
                                                           size_t);
template double ReduceMax::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                             const double *,
                                                             size_t);
template float ReduceMin::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                           const float *,
                                                           size_t);
template double ReduceMin::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                             const double *,
                                                             size_t);
template float ReduceProd::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                            const float *,
                                                            size_t);
template double ReduceProd::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                              const double *,
                                                              size_t);

// Activations
template void ActivationReLU::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                               const float *,
                                                               float *, size_t);
template void ActivationReLU::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                                const double *,
                                                                double *,
                                                                size_t);
template void ActivationSigmoid::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                                  const float *,
                                                                  float *,
                                                                  size_t);
template void ActivationSigmoid::operator()<xsimd::neon64, double>(
    xsimd::neon64, const double *, double *, size_t);
template void ActivationGELU::operator()<xsimd::neon64, float>(xsimd::neon64,
                                                               const float *,
                                                               float *, size_t);
template void ActivationGELU::operator()<xsimd::neon64, double>(xsimd::neon64,
                                                                const double *,
                                                                double *,
                                                                size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
