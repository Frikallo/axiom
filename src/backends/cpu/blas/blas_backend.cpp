#include "blas_backend.hpp"
#include "blas_fallback.hpp"

#ifdef AXIOM_USE_ACCELERATE
#include "blas_accelerate.hpp"
#endif

#ifdef AXIOM_USE_OPENBLAS
#include "blas_openblas.hpp"
#endif

#include <atomic>
#include <mutex>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace cpu {
namespace blas {

namespace {

// Global backend state
std::atomic<BlasType> g_backend_type{BlasType::Auto};
std::unique_ptr<BlasBackend> g_backend{nullptr};
std::once_flag g_init_flag;

// Create backend instance based on type
std::unique_ptr<BlasBackend> create_backend(BlasType type) {
    switch (type) {
#ifdef AXIOM_USE_ACCELERATE
    case BlasType::Accelerate:
        return std::make_unique<AccelerateBlasBackend>();
#endif
#ifdef AXIOM_USE_OPENBLAS
    case BlasType::OpenBLAS:
        return std::make_unique<OpenBlasBackend>();
#endif
    case BlasType::Fallback:
        return std::make_unique<FallbackBlasBackend>();
    case BlasType::Auto:
        // Auto-detect: try backends in order of preference
#ifdef AXIOM_USE_ACCELERATE
        return std::make_unique<AccelerateBlasBackend>();
#elif defined(AXIOM_USE_OPENBLAS)
        return std::make_unique<OpenBlasBackend>();
#else
        return std::make_unique<FallbackBlasBackend>();
#endif
    default:
        throw std::runtime_error("Unknown BLAS backend type");
    }
}

void init_backend() {
    BlasType type = g_backend_type.load(std::memory_order_acquire);
    g_backend = create_backend(type);
}

} // anonymous namespace

BlasBackend &get_blas_backend() {
    std::call_once(g_init_flag, init_backend);
    return *g_backend;
}

void set_blas_backend(BlasType type) {
    // Check if backend is available
    if (!is_backend_available(type)) {
        throw std::runtime_error("Requested BLAS backend is not available on "
                                 "this platform");
    }
    g_backend_type.store(type, std::memory_order_release);
    // Note: If backend was already initialized, this won't take effect until
    // restart. In practice, this should be called at program start before any
    // BLAS ops.
}

BlasType get_blas_backend_type() {
    return g_backend_type.load(std::memory_order_acquire);
}

bool is_backend_available(BlasType type) {
    switch (type) {
    case BlasType::Auto:
        return true; // Auto always available (falls back to Fallback)
    case BlasType::Accelerate:
#ifdef AXIOM_USE_ACCELERATE
        return true;
#else
        return false;
#endif
    case BlasType::OpenBLAS:
#ifdef AXIOM_USE_OPENBLAS
        return true;
#else
        return false;
#endif
    case BlasType::Fallback:
        return true; // Fallback is always available
    default:
        return false;
    }
}

BlasType get_default_backend_type() {
#ifdef AXIOM_USE_ACCELERATE
    return BlasType::Accelerate;
#elif defined(AXIOM_USE_OPENBLAS)
    return BlasType::OpenBLAS;
#else
    return BlasType::Fallback;
#endif
}

} // namespace blas
} // namespace cpu
} // namespace backends
} // namespace axiom
