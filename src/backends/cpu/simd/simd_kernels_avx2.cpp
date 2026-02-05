// AVX2 kernel instantiations
// Compile with: -mavx2 -mfma (Haswell+, 2013+, most common modern x86)

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Binary Operations
// ============================================================================

template void BinaryAdd::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryAdd::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinarySub::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinarySub::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMul::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMul::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryDiv::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryDiv::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMax::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMax::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMin::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMin::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryPow::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryPow::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryAtan2::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryAtan2::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryHypot::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryHypot::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryFmod::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryFmod::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Unary Operations
// ============================================================================

template void UnaryNeg::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryNeg::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryAbs::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryAbs::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnarySqrt::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *, float *,
                                                        size_t);
template void UnarySqrt::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryExp::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryExp::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryLog::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryLog::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnarySin::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnarySin::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCos::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCos::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryTanh::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *, float *,
                                                        size_t);
template void UnaryTanh::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryTan::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryTan::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryErf::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                       const float *, float *,
                                                       size_t);
template void UnaryErf::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCbrt::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCbrt::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         double *, size_t);
template void UnarySquare::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                          const float *,
                                                          float *, size_t);
template void UnarySquare::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                           const double *,
                                                           double *, size_t);
template void UnaryReciprocal::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                              const float *,
                                                              float *, size_t);
template void UnaryReciprocal::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                               const double *,
                                                               double *,
                                                               size_t);
template void UnarySign::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *, float *,
                                                        size_t);
template void UnarySign::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryFloor::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryFloor::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryCeil::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCeil::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                         const double *,
                                                         double *, size_t);
template void UnaryRound::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryRound::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                          const double *,
                                                          double *, size_t);
template void UnaryTrunc::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                         const float *, float *,
                                                         size_t);
template void UnaryTrunc::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Reductions
// ============================================================================

template float ReduceSum::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                         const float *, size_t);
template double
ReduceSum::operator()<xsimd::avx2, double>(xsimd::avx2, const double *, size_t);
template float ReduceMax::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                         const float *, size_t);
template double
ReduceMax::operator()<xsimd::avx2, double>(xsimd::avx2, const double *, size_t);
template float ReduceMin::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                         const float *, size_t);
template double
ReduceMin::operator()<xsimd::avx2, double>(xsimd::avx2, const double *, size_t);
template float
ReduceProd::operator()<xsimd::avx2, float>(xsimd::avx2, const float *, size_t);
template double ReduceProd::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                            const double *,
                                                            size_t);

// ============================================================================
// Activations
// ============================================================================

template void ActivationReLU::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                             const float *,
                                                             float *, size_t);
template void ActivationReLU::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                              const double *,
                                                              double *, size_t);
template void ActivationSigmoid::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                                const float *,
                                                                float *,
                                                                size_t);
template void ActivationSigmoid::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                                 const double *,
                                                                 double *,
                                                                 size_t);
template void ActivationGELU::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                             const float *,
                                                             float *, size_t);
template void ActivationGELU::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                              const double *,
                                                              double *, size_t);
template void ActivationSiLU::operator()<xsimd::avx2, float>(xsimd::avx2,
                                                             const float *,
                                                             float *, size_t);
template void ActivationSiLU::operator()<xsimd::avx2, double>(xsimd::avx2,
                                                              const double *,
                                                              double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
