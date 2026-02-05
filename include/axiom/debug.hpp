#pragma once

#include <atomic>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace axiom {

// Forward declaration
class Tensor;

namespace trace {

struct TraceEvent {
    std::string op_name;
    std::string description;
    std::chrono::steady_clock::time_point timestamp;
    std::chrono::nanoseconds duration;
    size_t memory_bytes;
    bool materialized; // Did this op allocate new memory?
};

class Tracer {
  public:
    static Tracer &instance();

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
    }

    void record(const std::string &op_name, const std::string &desc,
                std::chrono::nanoseconds duration, size_t memory_bytes,
                bool materialized);

    std::string dump() const;

    const std::vector<TraceEvent> &events() const { return events_; }

  private:
    Tracer() = default;
    std::atomic<bool> enabled_{false};
    mutable std::mutex mutex_;
    std::vector<TraceEvent> events_;
};

inline void enable() { Tracer::instance().enable(); }
inline void disable() { Tracer::instance().disable(); }
inline void clear() { Tracer::instance().clear(); }
inline std::string dump() { return Tracer::instance().dump(); }
inline bool is_enabled() { return Tracer::instance().is_enabled(); }

class ScopedTrace {
  public:
    ScopedTrace(const std::string &op_name, const std::string &desc = "",
                size_t memory_bytes = 0, bool materialized = false);
    ~ScopedTrace();

  private:
    std::string op_name_;
    std::string desc_;
    std::chrono::steady_clock::time_point start_;
    size_t memory_bytes_;
    bool materialized_;
};

} // namespace trace

namespace profile {

struct OpProfile {
    std::string name;
    std::chrono::nanoseconds duration;
    size_t input_bytes;
    size_t output_bytes;
    std::string shape_info;
};

class Profiler {
  public:
    static Profiler &instance();

    void record_op(const OpProfile &profile);

    const OpProfile &last_op() const { return last_op_; }

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }

  private:
    Profiler() = default;
    std::atomic<bool> enabled_{false};
    OpProfile last_op_;
    mutable std::mutex mutex_;
};

inline void enable() { Profiler::instance().enable(); }
inline void disable() { Profiler::instance().disable(); }
inline const OpProfile &last_op() { return Profiler::instance().last_op(); }

} // namespace profile

// ============================================================================
// SIMD/Backend Diagnostics
// ============================================================================

namespace cpu_info {

// Print CPU SIMD architecture info (xsimd-detected) to stdout
void print_simd_info();

// Get SIMD architecture name (e.g., "neon64", "avx2", "sse4.2")
const char *simd_arch_name();

// Get SIMD info as a compact string
std::string simd_info_string();

} // namespace cpu_info

} // namespace axiom
