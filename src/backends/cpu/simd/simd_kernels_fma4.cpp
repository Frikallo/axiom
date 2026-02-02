// FMA4 kernel instantiations (AMD Bulldozer/Piledriver)
// Compile with: -mfma4

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Binary Operations
// ============================================================================

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
template void BinaryPow::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryPow::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryAtan2::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryAtan2::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryHypot::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryHypot::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryFmod::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryFmod::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Unary Operations
// ============================================================================

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
template void UnaryTan::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnaryTan::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnaryErf::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                       const float *, float *,
                                                       size_t);
template void UnaryErf::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCbrt::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCbrt::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         double *, size_t);
template void UnarySquare::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                          const float *,
                                                          float *, size_t);
template void UnarySquare::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                           const double *,
                                                           double *, size_t);
template void UnaryReciprocal::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                              const float *,
                                                              float *, size_t);
template void UnaryReciprocal::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                               const double *,
                                                               double *,
                                                               size_t);
template void UnarySign::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *, float *,
                                                        size_t);
template void UnarySign::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         double *, size_t);
template void UnaryFloor::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                         const float *, float *,
                                                         size_t);
template void UnaryFloor::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                          const double *,
                                                          double *, size_t);
template void UnaryCeil::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCeil::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                         const double *,
                                                         double *, size_t);
template void UnaryRound::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                         const float *, float *,
                                                         size_t);
template void UnaryRound::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                          const double *,
                                                          double *, size_t);
template void UnaryTrunc::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                         const float *, float *,
                                                         size_t);
template void UnaryTrunc::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Reductions
// ============================================================================

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

// ============================================================================
// Activations
// ============================================================================

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
template void ActivationSiLU::operator()<xsimd::fma4, float>(xsimd::fma4,
                                                             const float *,
                                                             float *, size_t);
template void ActivationSiLU::operator()<xsimd::fma4, double>(xsimd::fma4,
                                                              const double *,
                                                              double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
