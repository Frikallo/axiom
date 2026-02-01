// NEON kernel instantiations (ARM32/ARMv7)
// Compile with: -mfpu=neon

#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

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
template float ReduceSum::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, size_t);
template float ReduceMax::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, size_t);
template float ReduceMin::operator()<xsimd::neon, float>(xsimd::neon,
                                                         const float *, size_t);
template float
ReduceProd::operator()<xsimd::neon, float>(xsimd::neon, const float *, size_t);
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

// Note: ARMv7 NEON has limited double precision support

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
