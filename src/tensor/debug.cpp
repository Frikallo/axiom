#include "axiom/debug.hpp"
#include "backends/cpu/cpu_simd.hpp"

namespace axiom {
namespace trace {

Tracer &Tracer::instance() {
    static Tracer tracer;
    return tracer;
}

void Tracer::record(const std::string &op_name, const std::string &desc,
                    std::chrono::nanoseconds duration, size_t memory_bytes,
                    bool materialized) {
    if (!enabled_)
        return;

    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back({op_name, desc, std::chrono::steady_clock::now(),
                       duration, memory_bytes, materialized});
}

std::string Tracer::dump() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "=== Axiom Trace (" << events_.size() << " events) ===\n";
    oss << std::left << std::setw(20) << "Operation" << std::setw(15)
        << "Duration(us)" << std::setw(15) << "Memory(KB)" << std::setw(12)
        << "Materialized"
        << "Description\n";
    oss << std::string(80, '-') << "\n";

    std::chrono::nanoseconds total_time{0};
    size_t total_memory = 0;
    size_t materialized_count = 0;

    for (const auto &event : events_) {
        double duration_us = event.duration.count() / 1000.0;
        double memory_kb = event.memory_bytes / 1024.0;

        oss << std::left << std::setw(20) << event.op_name << std::setw(15)
            << std::fixed << std::setprecision(2) << duration_us
            << std::setw(15) << std::fixed << std::setprecision(2) << memory_kb
            << std::setw(12) << (event.materialized ? "yes" : "no")
            << event.description << "\n";

        total_time += event.duration;
        total_memory += event.memory_bytes;
        if (event.materialized)
            materialized_count++;
    }

    oss << std::string(80, '-') << "\n";
    oss << "Total time: " << (total_time.count() / 1000.0) << " us\n";
    oss << "Total memory allocated: " << (total_memory / 1024.0) << " KB\n";
    oss << "Materialized ops: " << materialized_count << " / " << events_.size()
        << "\n";

    return oss.str();
}

ScopedTrace::ScopedTrace(const std::string &op_name, const std::string &desc,
                         size_t memory_bytes, bool materialized)
    : op_name_(op_name), desc_(desc), memory_bytes_(memory_bytes),
      materialized_(materialized) {
    if (Tracer::instance().is_enabled()) {
        start_ = std::chrono::steady_clock::now();
    }
}

ScopedTrace::~ScopedTrace() {
    if (Tracer::instance().is_enabled()) {
        auto end = std::chrono::steady_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
        Tracer::instance().record(op_name_, desc_, duration, memory_bytes_,
                                  materialized_);
    }
}

} // namespace trace

namespace profile {

Profiler &Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::record_op(const OpProfile &profile) {
    if (!enabled_)
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    last_op_ = profile;
}

} // namespace profile

namespace cpu_info {

void print_simd_info() { backends::cpu::simd::print_simd_info(); }

const char *simd_arch_name() {
    return backends::cpu::simd::get_simd_info().arch_name;
}

std::string simd_info_string() {
    return backends::cpu::simd::simd_info_string();
}

} // namespace cpu_info

} // namespace axiom
