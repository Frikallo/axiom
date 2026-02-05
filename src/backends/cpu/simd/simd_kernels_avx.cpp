// AVX kernel instantiations
// Compile with: -mavx (Sandy Bridge 2011+)

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Binary Operations
// ============================================================================

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
template void BinaryPow::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *,
                                                       const float *, float *,
                                                       size_t);
template void BinaryPow::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        const double *,
                                                        double *, size_t);
template void BinaryAtan2::operator()<xsimd::avx, float>(xsimd::avx,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryAtan2::operator()<xsimd::avx, double>(xsimd::avx,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void BinaryHypot::operator()<xsimd::avx, float>(xsimd::avx,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryHypot::operator()<xsimd::avx, double>(xsimd::avx,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);
template void BinaryFmod::operator()<xsimd::avx, float>(xsimd::avx,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryFmod::operator()<xsimd::avx, double>(xsimd::avx,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);

// ============================================================================
// Unary Operations
// ============================================================================

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
template void UnaryTan::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnaryTan::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryErf::operator()<xsimd::avx, float>(xsimd::avx, const float *,
                                                      float *, size_t);
template void UnaryErf::operator()<xsimd::avx, double>(xsimd::avx,
                                                       const double *, double *,
                                                       size_t);
template void UnaryCbrt::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCbrt::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        double *, size_t);
template void UnarySquare::operator()<xsimd::avx, float>(xsimd::avx,
                                                         const float *, float *,
                                                         size_t);
template void UnarySquare::operator()<xsimd::avx, double>(xsimd::avx,
                                                          const double *,
                                                          double *, size_t);
template void UnaryReciprocal::operator()<xsimd::avx, float>(xsimd::avx,
                                                             const float *,
                                                             float *, size_t);
template void UnaryReciprocal::operator()<xsimd::avx, double>(xsimd::avx,
                                                              const double *,
                                                              double *, size_t);
template void UnarySign::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *, float *,
                                                       size_t);
template void UnarySign::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        double *, size_t);
template void UnaryFloor::operator()<xsimd::avx, float>(xsimd::avx,
                                                        const float *, float *,
                                                        size_t);
template void UnaryFloor::operator()<xsimd::avx, double>(xsimd::avx,
                                                         const double *,
                                                         double *, size_t);
template void UnaryCeil::operator()<xsimd::avx, float>(xsimd::avx,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCeil::operator()<xsimd::avx, double>(xsimd::avx,
                                                        const double *,
                                                        double *, size_t);
template void UnaryRound::operator()<xsimd::avx, float>(xsimd::avx,
                                                        const float *, float *,
                                                        size_t);
template void UnaryRound::operator()<xsimd::avx, double>(xsimd::avx,
                                                         const double *,
                                                         double *, size_t);
template void UnaryTrunc::operator()<xsimd::avx, float>(xsimd::avx,
                                                        const float *, float *,
                                                        size_t);
template void UnaryTrunc::operator()<xsimd::avx, double>(xsimd::avx,
                                                         const double *,
                                                         double *, size_t);

// ============================================================================
// Reductions
// ============================================================================

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

// ============================================================================
// Activations
// ============================================================================

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
template void ActivationSiLU::operator()<xsimd::avx, float>(xsimd::avx,
                                                            const float *,
                                                            float *, size_t);
template void ActivationSiLU::operator()<xsimd::avx, double>(xsimd::avx,
                                                             const double *,
                                                             double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
