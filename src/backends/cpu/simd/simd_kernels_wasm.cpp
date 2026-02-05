// WASM SIMD128 kernel instantiations (WebAssembly)
// Compile with: -msimd128

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Binary Operations
// ============================================================================

template void BinaryAdd::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryAdd::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinarySub::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinarySub::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMul::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMul::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryDiv::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryDiv::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMax::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMax::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryMin::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryMin::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryPow::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *,
                                                        const float *, float *,
                                                        size_t);
template void BinaryPow::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         const double *,
                                                         double *, size_t);
template void BinaryAtan2::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryAtan2::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryHypot::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                          const float *,
                                                          const float *,
                                                          float *, size_t);
template void BinaryHypot::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                           const double *,
                                                           const double *,
                                                           double *, size_t);
template void BinaryFmod::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                         const float *,
                                                         const float *, float *,
                                                         size_t);
template void BinaryFmod::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                          const double *,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Unary Operations
// ============================================================================

template void UnaryNeg::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnaryNeg::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnaryAbs::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnaryAbs::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnarySqrt::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *, float *,
                                                        size_t);
template void UnarySqrt::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         double *, size_t);
template void UnaryExp::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnaryExp::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnaryLog::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnaryLog::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnarySin::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnarySin::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCos::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnaryCos::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnaryTanh::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *, float *,
                                                        size_t);
template void UnaryTanh::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         double *, size_t);
template void UnaryTan::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnaryTan::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnaryErf::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                       const float *, float *,
                                                       size_t);
template void UnaryErf::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                        const double *,
                                                        double *, size_t);
template void UnaryCbrt::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCbrt::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         double *, size_t);
template void UnarySquare::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                          const float *,
                                                          float *, size_t);
template void UnarySquare::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                           const double *,
                                                           double *, size_t);
template void UnaryReciprocal::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                              const float *,
                                                              float *, size_t);
template void UnaryReciprocal::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                               const double *,
                                                               double *,
                                                               size_t);
template void UnarySign::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *, float *,
                                                        size_t);
template void UnarySign::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         double *, size_t);
template void UnaryFloor::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                         const float *, float *,
                                                         size_t);
template void UnaryFloor::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                          const double *,
                                                          double *, size_t);
template void UnaryCeil::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                        const float *, float *,
                                                        size_t);
template void UnaryCeil::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                         const double *,
                                                         double *, size_t);
template void UnaryRound::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                         const float *, float *,
                                                         size_t);
template void UnaryRound::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                          const double *,
                                                          double *, size_t);
template void UnaryTrunc::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                         const float *, float *,
                                                         size_t);
template void UnaryTrunc::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                          const double *,
                                                          double *, size_t);

// ============================================================================
// Reductions
// ============================================================================

template float ReduceSum::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                         const float *, size_t);
template double
ReduceSum::operator()<xsimd::wasm, double>(xsimd::wasm, const double *, size_t);
template float ReduceMax::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                         const float *, size_t);
template double
ReduceMax::operator()<xsimd::wasm, double>(xsimd::wasm, const double *, size_t);
template float ReduceMin::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                         const float *, size_t);
template double
ReduceMin::operator()<xsimd::wasm, double>(xsimd::wasm, const double *, size_t);
template float
ReduceProd::operator()<xsimd::wasm, float>(xsimd::wasm, const float *, size_t);
template double ReduceProd::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                            const double *,
                                                            size_t);

// ============================================================================
// Activations
// ============================================================================

template void ActivationReLU::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                             const float *,
                                                             float *, size_t);
template void ActivationReLU::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                              const double *,
                                                              double *, size_t);
template void ActivationSigmoid::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                                const float *,
                                                                float *,
                                                                size_t);
template void ActivationSigmoid::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                                 const double *,
                                                                 double *,
                                                                 size_t);
template void ActivationGELU::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                             const float *,
                                                             float *, size_t);
template void ActivationGELU::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                              const double *,
                                                              double *, size_t);
template void ActivationSiLU::operator()<xsimd::wasm, float>(xsimd::wasm,
                                                             const float *,
                                                             float *, size_t);
template void ActivationSiLU::operator()<xsimd::wasm, double>(xsimd::wasm,
                                                              const double *,
                                                              double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
