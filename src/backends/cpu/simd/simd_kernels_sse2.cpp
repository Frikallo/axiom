// SSE2 kernel instantiations
// Compile with: -msse2 (baseline x86-64, always available)

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Binary Operations
// ============================================================================

template void BinaryAdd::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryAdd::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinarySub::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinarySub::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMul::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMul::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryDiv::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryDiv::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMax::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMax::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMin::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMin::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryPow::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryPow::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryAtan2::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryAtan2::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryHypot::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryHypot::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryFmod::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryFmod::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Unary Operations
// ============================================================================

template void UnaryNeg::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryNeg::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryAbs::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryAbs::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnarySqrt::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *, float *,
                                                        size_t);
template void UnarySqrt::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryExp::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryExp::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryLog::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryLog::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnarySin::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnarySin::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCos::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCos::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryTanh::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *, float *,
                                                        size_t);
template void UnaryTanh::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryTan::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryTan::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryErf::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryErf::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCbrt::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCbrt::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         double *, size_t);
template void UnarySquare::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                          const float *,
                                                          float *, size_t);
template void UnarySquare::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                           const double *,
                                                           double *, size_t);
template void UnaryReciprocal::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                              const float *,
                                                              float *, size_t);
template void UnaryReciprocal::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                               const double *,
                                                               double *,
                                                               size_t);
template void UnarySign::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *, float *,
                                                        size_t);
template void UnarySign::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryFloor::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryFloor::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryCeil::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCeil::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryRound::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryRound::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryTrunc::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryTrunc::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Reductions
// ============================================================================

template float ReduceSum::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                         const float *, size_t);
template double
ReduceSum::operator()<xsimd::sse2, double>(xsimd::sse2, const double *, size_t);
template float ReduceMax::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                         const float *, size_t);
template double
ReduceMax::operator()<xsimd::sse2, double>(xsimd::sse2, const double *, size_t);
template float ReduceMin::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                         const float *, size_t);
template double
ReduceMin::operator()<xsimd::sse2, double>(xsimd::sse2, const double *, size_t);
template float
ReduceProd::operator()<xsimd::sse2, float>(xsimd::sse2, const float *, size_t);
template double ReduceProd::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                            const double *,
                                                            size_t);

// ============================================================================
// Activations
// ============================================================================

template void ActivationReLU::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                             const float *,
                                                             float *, size_t);
template void ActivationReLU::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                              const double *,
                                                              double *, size_t);
template void ActivationSigmoid::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                                const float *,
                                                                float *,
                                                                size_t);
template void ActivationSigmoid::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                                 const double *,
                                                                 double *,
                                                                 size_t);
template void ActivationGELU::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                             const float *,
                                                             float *, size_t);
template void ActivationGELU::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                              const double *,
                                                              double *, size_t);
template void ActivationSiLU::operator()<xsimd::sse2, float>(xsimd::sse2,
                                                             const float *,
                                                             float *, size_t);
template void ActivationSiLU::operator()<xsimd::sse2, double>(xsimd::sse2,
                                                              const double *,
                                                              double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
