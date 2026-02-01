// FMA3+AVX kernel instantiations
// Compile with: -mavx -mfma

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

using arch = xsimd::fma3<xsimd::avx>;

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

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
