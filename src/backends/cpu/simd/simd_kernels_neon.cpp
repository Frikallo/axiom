// NEON kernel instantiations (ARM32/ARMv7)
// Compile with: -mfpu=neon
// Note: ARMv7 NEON has limited double precision support - only float
// instantiated

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Binary Operations (float only)
// ============================================================================

template void BinaryAdd::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinarySub::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMul::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryDiv::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMax::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMin::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryPow::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryAtan2::operator()<xsimd::neon, float>(xsimd::neon,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryHypot::operator()<xsimd::neon, float>(xsimd::neon,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryFmod::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);

// ============================================================================
// Unary Operations (float only)
// ============================================================================

template void UnaryNeg::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnaryAbs::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnarySqrt::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *, float *,
                                                        size_t);
template void UnaryExp::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnaryLog::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnarySin::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCos::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnaryTanh::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *, float *,
                                                        size_t);
template void UnaryTan::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnaryErf::operator()<xsimd::neon, float>(xsimd::neon,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCbrt::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *, float *,
                                                        size_t);
template void UnarySquare::operator()<xsimd::neon, float>(xsimd::neon,
                                                          const float *,
                                                          float *, size_t);
template void UnaryReciprocal::operator()<xsimd::neon, float>(xsimd::neon,
                                                              const float *,
                                                              float *, size_t);
template void UnarySign::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *, float *,
                                                        size_t);
template void UnaryFloor::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, float *,
                                                         size_t);
template void UnaryCeil::operator()<xsimd::neon, float>(xsimd::neon,
                                                        const float *, float *,
                                                        size_t);
template void UnaryRound::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, float *,
                                                         size_t);
template void UnaryTrunc::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, float *,
                                                         size_t);

// ============================================================================
// Reductions (float only)
// ============================================================================

template float ReduceSum::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, size_t);
template float ReduceMax::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, size_t);
template float ReduceMin::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, size_t);
template float
ReduceProd::operator()<xsimd::neon, float>(xsimd::neon, const float *, size_t);

// ============================================================================
// Activations (float only)
// ============================================================================

template void ActivationReLU::operator()<xsimd::neon, float>(xsimd::neon,
                                                             const float *,
                                                             float *, size_t);
template void ActivationSigmoid::operator()<xsimd::neon, float>(xsimd::neon,
                                                                const float *,
                                                                float *,
                                                                size_t);
template void ActivationGELU::operator()<xsimd::neon, float>(xsimd::neon,
                                                             const float *,
                                                             float *, size_t);
template void ActivationSiLU::operator()<xsimd::neon, float>(xsimd::neon,
                                                             const float *,
                                                             float *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
