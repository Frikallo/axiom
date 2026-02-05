#include "lapack_backend.hpp"
#include "lapack_native.hpp"

#ifdef AXIOM_USE_ACCELERATE
#include "lapack_accelerate.hpp"
#endif

#ifdef AXIOM_USE_OPENBLAS
#include "lapack_openblas.hpp"
#endif

#include <atomic>
#include <mutex>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace cpu {
namespace lapack {

namespace {

// Global backend state
std::atomic<LapackType> g_backend_type{LapackType::Auto};
std::unique_ptr<LapackBackend> g_backend{nullptr};
std::once_flag g_init_flag;

// Create backend instance based on type
std::unique_ptr<LapackBackend> create_backend(LapackType type) {
    switch (type) {
#ifdef AXIOM_USE_ACCELERATE
    case LapackType::Accelerate:
        return std::make_unique<AccelerateLapackBackend>();
#endif
#ifdef AXIOM_USE_OPENBLAS
    case LapackType::OpenBLAS:
        return std::make_unique<OpenBlasLapackBackend>();
#endif
    case LapackType::Native:
        return std::make_unique<NativeLapackBackend>();
    case LapackType::Auto:
        // Auto-detect: try backends in order of preference
#ifdef AXIOM_USE_ACCELERATE
        return std::make_unique<AccelerateLapackBackend>();
#elif defined(AXIOM_USE_OPENBLAS)
        return std::make_unique<OpenBlasLapackBackend>();
#else
        return std::make_unique<NativeLapackBackend>();
#endif
    default:
        throw std::runtime_error("Unknown LAPACK backend type");
    }
}

void init_backend() {
    LapackType type = g_backend_type.load(std::memory_order_acquire);
    g_backend = create_backend(type);
}

} // anonymous namespace

LapackBackend &get_lapack_backend() {
    std::call_once(g_init_flag, init_backend);
    return *g_backend;
}

void set_lapack_backend(LapackType type) {
    // Check if backend is available
    if (!is_lapack_backend_available(type)) {
        throw std::runtime_error(
            "Requested LAPACK backend is not available on this platform");
    }
    g_backend_type.store(type, std::memory_order_release);
    // Note: If backend was already initialized, this won't take effect until
    // restart. In practice, this should be called at program start before any
    // LAPACK ops.
}

LapackType get_lapack_backend_type() {
    return g_backend_type.load(std::memory_order_acquire);
}

bool is_lapack_backend_available(LapackType type) {
    switch (type) {
    case LapackType::Auto:
        return true; // Auto always available (uses Native as fallback)
    case LapackType::Accelerate:
#ifdef AXIOM_USE_ACCELERATE
        return true;
#else
        return false;
#endif
    case LapackType::OpenBLAS:
#ifdef AXIOM_USE_OPENBLAS
        return true;
#else
        return false;
#endif
    case LapackType::Native:
        return true; // Native is always available (but throws on all ops)
    default:
        return false;
    }
}

LapackType get_default_lapack_backend_type() {
#ifdef AXIOM_USE_ACCELERATE
    return LapackType::Accelerate;
#elif defined(AXIOM_USE_OPENBLAS)
    return LapackType::OpenBLAS;
#else
    return LapackType::Native;
#endif
}

bool has_lapack() {
#if defined(AXIOM_USE_ACCELERATE) || defined(AXIOM_USE_OPENBLAS)
    return true;
#else
    return false;
#endif
}

} // namespace lapack
} // namespace cpu
} // namespace backends
} // namespace axiom
