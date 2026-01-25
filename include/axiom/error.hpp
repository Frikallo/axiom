#pragma once

#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace axiom {

// ============================================================================
// Base Axiom Exception
// ============================================================================

class AxiomError : public std::exception {
  public:
    explicit AxiomError(const std::string &message) : message_(message) {}

    const char *what() const noexcept override { return message_.c_str(); }

    const std::string &message() const { return message_; }

  protected:
    std::string message_;
};

// ============================================================================
// Shape-related errors
// ============================================================================

class ShapeError : public AxiomError {
  public:
    explicit ShapeError(const std::string &message)
        : AxiomError("ShapeError: " + message) {}

    template <typename Container>
    static ShapeError mismatch(const Container &expected,
                               const Container &got) {
        std::ostringstream oss;
        oss << "expected shape [";
        for (size_t i = 0; i < expected.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << expected[i];
        }
        oss << "] but got [";
        for (size_t i = 0; i < got.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << got[i];
        }
        oss << "]";
        return ShapeError(oss.str());
    }

    static ShapeError broadcast_incompatible(const std::string &details) {
        return ShapeError("shapes are not broadcastable: " + details);
    }

    static ShapeError invalid_axis(int axis, int ndim) {
        return ShapeError("axis " + std::to_string(axis) +
                          " out of bounds for tensor with " +
                          std::to_string(ndim) + " dimensions");
    }

    static ShapeError invalid_reshape(size_t from_size, size_t to_size) {
        return ShapeError("cannot reshape tensor of size " +
                          std::to_string(from_size) + " to size " +
                          std::to_string(to_size));
    }
};

// ============================================================================
// Type-related errors
// ============================================================================

class TypeError : public AxiomError {
  public:
    explicit TypeError(const std::string &message)
        : AxiomError("TypeError: " + message) {}

    static TypeError unsupported_dtype(const std::string &dtype,
                                       const std::string &operation) {
        return TypeError("unsupported dtype '" + dtype + "' for " + operation);
    }

    static TypeError dtype_mismatch(const std::string &expected,
                                    const std::string &got) {
        return TypeError("expected dtype " + expected + " but got " + got);
    }

    static TypeError conversion_not_safe(const std::string &from,
                                         const std::string &to) {
        return TypeError("conversion from " + from + " to " + to +
                         " may lose precision");
    }
};

// ============================================================================
// Device-related errors
// ============================================================================

class DeviceError : public AxiomError {
  public:
    explicit DeviceError(const std::string &message)
        : AxiomError("DeviceError: " + message) {}

    static DeviceError not_available(const std::string &device) {
        return DeviceError(device + " is not available on this system");
    }

    static DeviceError mismatch(const std::string &expected,
                                const std::string &got) {
        return DeviceError("expected tensor on " + expected + " but got " +
                           got);
    }

    static DeviceError cpu_only(const std::string &operation) {
        return DeviceError(operation + " is only available for CPU tensors");
    }
};

// ============================================================================
// Value-related errors (NaN, Inf, out of range)
// ============================================================================

class ValueError : public AxiomError {
  public:
    explicit ValueError(const std::string &message)
        : AxiomError("ValueError: " + message) {}

    static ValueError nan_detected(const std::string &context = "") {
        return ValueError("NaN detected" +
                          (context.empty() ? "" : " in " + context));
    }

    static ValueError inf_detected(const std::string &context = "") {
        return ValueError("Inf detected" +
                          (context.empty() ? "" : " in " + context));
    }

    static ValueError not_finite(const std::string &context = "") {
        return ValueError("non-finite value detected" +
                          (context.empty() ? "" : " in " + context));
    }

    static ValueError out_of_range(const std::string &what, double min,
                                   double max, double got) {
        std::ostringstream oss;
        oss << what << " must be in range [" << min << ", " << max
            << "] but got " << got;
        return ValueError(oss.str());
    }
};

// ============================================================================
// Index-related errors
// ============================================================================

class IndexError : public AxiomError {
  public:
    explicit IndexError(const std::string &message)
        : AxiomError("IndexError: " + message) {}

    static IndexError out_of_bounds(size_t index, size_t size, int dim = -1) {
        std::ostringstream oss;
        oss << "index " << index << " out of bounds for ";
        if (dim >= 0)
            oss << "dimension " << dim << " with ";
        oss << "size " << size;
        return IndexError(oss.str());
    }

    static IndexError invalid_slice(const std::string &details) {
        return IndexError("invalid slice: " + details);
    }
};

// ============================================================================
// Memory-related errors
// ============================================================================

class MemoryError : public AxiomError {
  public:
    explicit MemoryError(const std::string &message)
        : AxiomError("MemoryError: " + message) {}

    static MemoryError allocation_failed(size_t bytes) {
        return MemoryError("failed to allocate " + std::to_string(bytes) +
                           " bytes");
    }

    static MemoryError storage_too_small(size_t required, size_t available) {
        return MemoryError("storage has " + std::to_string(available) +
                           " bytes but " + std::to_string(required) +
                           " required");
    }

    static MemoryError not_contiguous(const std::string &operation) {
        return MemoryError(operation + " requires contiguous tensor");
    }
};

// ============================================================================
// Runtime/internal errors
// ============================================================================

class RuntimeError : public AxiomError {
  public:
    explicit RuntimeError(const std::string &message)
        : AxiomError("RuntimeError: " + message) {}

    static RuntimeError not_implemented(const std::string &feature) {
        return RuntimeError(feature + " is not yet implemented");
    }

    static RuntimeError internal(const std::string &details) {
        return RuntimeError("internal error: " + details);
    }
};

} // namespace axiom
