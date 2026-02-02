// FMA3+AVX kernel instantiations
// Compile with: -mavx -mfma

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

using arch = xsimd::fma3<xsimd::avx>;

// ============================================================================
// Binary Operations
// ============================================================================

template void BinaryAdd::operator()<arch, float>(arch, const float *,
                                                 const float *, float *,
                                                 size_t);
template void BinaryAdd::operator()<arch, double>(arch, const double *,
                                                  const double *, double *,
                                                  size_t);
template void BinarySub::operator()<arch, float>(arch, const float *,
                                                 const float *, float *,
                                                 size_t);
template void BinarySub::operator()<arch, double>(arch, const double *,
                                                  const double *, double *,
                                                  size_t);
template void BinaryMul::operator()<arch, float>(arch, const float *,
                                                 const float *, float *,
                                                 size_t);
template void BinaryMul::operator()<arch, double>(arch, const double *,
                                                  const double *, double *,
                                                  size_t);
template void BinaryDiv::operator()<arch, float>(arch, const float *,
                                                 const float *, float *,
                                                 size_t);
template void BinaryDiv::operator()<arch, double>(arch, const double *,
                                                  const double *, double *,
                                                  size_t);
template void BinaryMax::operator()<arch, float>(arch, const float *,
                                                 const float *, float *,
                                                 size_t);
template void BinaryMax::operator()<arch, double>(arch, const double *,
                                                  const double *, double *,
                                                  size_t);
template void BinaryMin::operator()<arch, float>(arch, const float *,
                                                 const float *, float *,
                                                 size_t);
template void BinaryMin::operator()<arch, double>(arch, const double *,
                                                  const double *, double *,
                                                  size_t);
template void BinaryPow::operator()<arch, float>(arch, const float *,
                                                 const float *, float *,
                                                 size_t);
template void BinaryPow::operator()<arch, double>(arch, const double *,
                                                  const double *, double *,
                                                  size_t);
template void BinaryAtan2::operator()<arch, float>(arch, const float *,
                                                   const float *, float *,
                                                   size_t);
template void BinaryAtan2::operator()<arch, double>(arch, const double *,
                                                    const double *, double *,
                                                    size_t);
template void BinaryHypot::operator()<arch, float>(arch, const float *,
                                                   const float *, float *,
                                                   size_t);
template void BinaryHypot::operator()<arch, double>(arch, const double *,
                                                    const double *, double *,
                                                    size_t);
template void BinaryFmod::operator()<arch, float>(arch, const float *,
                                                  const float *, float *,
                                                  size_t);
template void BinaryFmod::operator()<arch, double>(arch, const double *,
                                                   const double *, double *,
                                                   size_t);

// ============================================================================
// Unary Operations
// ============================================================================

template void UnaryNeg::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnaryNeg::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnaryAbs::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnaryAbs::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnarySqrt::operator()<arch, float>(arch, const float *, float *,
                                                 size_t);
template void UnarySqrt::operator()<arch, double>(arch, const double *,
                                                  double *, size_t);
template void UnaryExp::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnaryExp::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnaryLog::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnaryLog::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnarySin::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnarySin::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnaryCos::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnaryCos::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnaryTanh::operator()<arch, float>(arch, const float *, float *,
                                                 size_t);
template void UnaryTanh::operator()<arch, double>(arch, const double *,
                                                  double *, size_t);
template void UnaryTan::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnaryTan::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnaryErf::operator()<arch, float>(arch, const float *, float *,
                                                size_t);
template void UnaryErf::operator()<arch, double>(arch, const double *, double *,
                                                 size_t);
template void UnaryCbrt::operator()<arch, float>(arch, const float *, float *,
                                                 size_t);
template void UnaryCbrt::operator()<arch, double>(arch, const double *,
                                                  double *, size_t);
template void UnarySquare::operator()<arch, float>(arch, const float *, float *,
                                                   size_t);
template void UnarySquare::operator()<arch, double>(arch, const double *,
                                                    double *, size_t);
template void UnaryReciprocal::operator()<arch, float>(arch, const float *,
                                                       float *, size_t);
template void UnaryReciprocal::operator()<arch, double>(arch, const double *,
                                                        double *, size_t);
template void UnarySign::operator()<arch, float>(arch, const float *, float *,
                                                 size_t);
template void UnarySign::operator()<arch, double>(arch, const double *,
                                                  double *, size_t);
template void UnaryFloor::operator()<arch, float>(arch, const float *, float *,
                                                  size_t);
template void UnaryFloor::operator()<arch, double>(arch, const double *,
                                                   double *, size_t);
template void UnaryCeil::operator()<arch, float>(arch, const float *, float *,
                                                 size_t);
template void UnaryCeil::operator()<arch, double>(arch, const double *,
                                                  double *, size_t);
template void UnaryRound::operator()<arch, float>(arch, const float *, float *,
                                                  size_t);
template void UnaryRound::operator()<arch, double>(arch, const double *,
                                                   double *, size_t);
template void UnaryTrunc::operator()<arch, float>(arch, const float *, float *,
                                                  size_t);
template void UnaryTrunc::operator()<arch, double>(arch, const double *,
                                                   double *, size_t);

// ============================================================================
// Reductions
// ============================================================================

template float ReduceSum::operator()<arch, float>(arch, const float *, size_t);
template double ReduceSum::operator()<arch, double>(arch, const double *,
                                                    size_t);
template float ReduceMax::operator()<arch, float>(arch, const float *, size_t);
template double ReduceMax::operator()<arch, double>(arch, const double *,
                                                    size_t);
template float ReduceMin::operator()<arch, float>(arch, const float *, size_t);
template double ReduceMin::operator()<arch, double>(arch, const double *,
                                                    size_t);
template float ReduceProd::operator()<arch, float>(arch, const float *, size_t);
template double ReduceProd::operator()<arch, double>(arch, const double *,
                                                     size_t);

// ============================================================================
// Activations
// ============================================================================

template void ActivationReLU::operator()<arch, float>(arch, const float *,
                                                      float *, size_t);
template void ActivationReLU::operator()<arch, double>(arch, const double *,
                                                       double *, size_t);
template void ActivationSigmoid::operator()<arch, float>(arch, const float *,
                                                         float *, size_t);
template void ActivationSigmoid::operator()<arch, double>(arch, const double *,
                                                          double *, size_t);
template void ActivationGELU::operator()<arch, float>(arch, const float *,
                                                      float *, size_t);
template void ActivationGELU::operator()<arch, double>(arch, const double *,
                                                       double *, size_t);
template void ActivationSiLU::operator()<arch, float>(arch, const float *,
                                                      float *, size_t);
template void ActivationSiLU::operator()<arch, double>(arch, const double *,
                                                       double *, size_t);

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
