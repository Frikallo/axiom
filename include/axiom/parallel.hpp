#pragma once

#include <cstddef>

#ifdef AXIOM_USE_OPENMP
#include <omp.h>
#endif

namespace axiom {
namespace parallel {

// ============================================================================
// Parallelization Thresholds
// ============================================================================

// Minimum elements to trigger parallelization (avoids overhead on small
// tensors)
constexpr size_t DEFAULT_MIN_ELEMENTS = 65536;    // 256KB for float32
constexpr size_t REDUCTION_MIN_ELEMENTS = 262144; // Higher for reductions
constexpr size_t MATMUL_MIN_PRODUCT = 1000000;    // M*N*K threshold

// ============================================================================
// Thread Management
// ============================================================================

/// Get the maximum number of threads available for parallel regions
inline size_t get_num_threads() {
#ifdef AXIOM_USE_OPENMP
    return static_cast<size_t>(omp_get_max_threads());
#else
    return 1;
#endif
}

/// Set the number of threads for parallel regions (0 = use all available)
inline void set_num_threads(size_t n) {
#ifdef AXIOM_USE_OPENMP
    omp_set_num_threads(static_cast<int>(n));
#endif
    (void)n;
}

/// Get the current thread ID within a parallel region
inline size_t get_thread_id() {
#ifdef AXIOM_USE_OPENMP
    return static_cast<size_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

// ============================================================================
// Parallelization Policies
// ============================================================================

/// Check if parallelization would be beneficial for the given element count
inline bool should_parallelize(size_t elements,
                               size_t min_elements = DEFAULT_MIN_ELEMENTS) {
#ifdef AXIOM_USE_OPENMP
    return elements >= min_elements && omp_get_max_threads() > 1;
#else
    (void)elements;
    (void)min_elements;
    return false;
#endif
}

/// Check if parallelization would be beneficial for reductions
inline bool should_parallelize_reduction(size_t elements) {
    return should_parallelize(elements, REDUCTION_MIN_ELEMENTS);
}

/// Check if parallelization would be beneficial for matrix multiplication
inline bool should_parallelize_matmul(size_t M, size_t N, size_t K) {
#ifdef AXIOM_USE_OPENMP
    return M * N * K >= MATMUL_MIN_PRODUCT && omp_get_max_threads() > 1;
#else
    (void)M;
    (void)N;
    (void)K;
    return false;
#endif
}

// ============================================================================
// RAII Thread Guard
// ============================================================================

/// RAII guard for temporarily changing the thread count
/// Restores the previous thread count when destroyed
class ThreadGuard {
#ifdef AXIOM_USE_OPENMP
    int prev_threads_;

  public:
    explicit ThreadGuard(size_t n) : prev_threads_(omp_get_max_threads()) {
        omp_set_num_threads(static_cast<int>(n));
    }
    ~ThreadGuard() { omp_set_num_threads(prev_threads_); }
#else
  public:
    explicit ThreadGuard(size_t) {}
#endif
    ThreadGuard(const ThreadGuard &) = delete;
    ThreadGuard &operator=(const ThreadGuard &) = delete;
};

} // namespace parallel
} // namespace axiom
