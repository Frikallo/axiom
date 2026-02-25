#include "axiom/sampling.hpp"
#include "axiom/dispatch.hpp"
#include "axiom/error.hpp"
#include "axiom/operations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace axiom {
namespace sampling {

Tensor temperature_scale(const Tensor &logits, float temperature) {
    if (temperature <= 0.0f) {
        throw ValueError("temperature_scale: temperature must be > 0, got " +
                         std::to_string(temperature));
    }
    if (temperature == 1.0f) {
        return logits;
    }
    auto temp_tensor = Tensor::full({1}, temperature, logits.device());
    return ops::divide(logits, temp_tensor);
}

Tensor top_k(const Tensor &logits, int k) {
    if (k <= 0) {
        throw ValueError("top_k: k must be > 0, got " + std::to_string(k));
    }

    // Work on CPU for nth_element
    Tensor cpu_logits = logits.device() == Device::CPU ? logits : logits.cpu();
    auto shape = cpu_logits.shape();
    size_t last_dim = shape[shape.size() - 1];

    if (static_cast<size_t>(k) > last_dim) {
        return logits; // k >= vocab_size, keep all
    }

    // Compute number of rows (all dims except last)
    size_t num_rows = cpu_logits.size() / last_dim;

    // Make contiguous copy for direct pointer access
    Tensor contiguous = cpu_logits.is_contiguous()
                            ? cpu_logits
                            : cpu_logits.ascontiguousarray();

    return dispatch_float(
        logits.dtype(), "top_k", [&]<typename DT>(DT) -> Tensor {
            using T = typename DT::value_type;
            const T *src = contiguous.typed_data<T>();
            Tensor result(shape, logits.dtype(), Device::CPU);
            T *dst = result.typed_data<T>();

            std::vector<T> row_copy(last_dim);
            const T neg_inf =
                static_cast<T>(-std::numeric_limits<double>::infinity());

            for (size_t r = 0; r < num_rows; ++r) {
                const T *row_src = src + r * last_dim;
                T *row_dst = dst + r * last_dim;

                // Copy and find k-th largest via nth_element
                std::copy(row_src, row_src + last_dim, row_copy.begin());
                std::nth_element(row_copy.begin(), row_copy.begin() + (k - 1),
                                 row_copy.end(), std::greater<T>());
                T threshold = row_copy[k - 1];

                // Mask below threshold to -inf
                for (size_t i = 0; i < last_dim; ++i) {
                    row_dst[i] =
                        (row_src[i] >= threshold) ? row_src[i] : neg_inf;
                }
            }

            return logits.device() == Device::GPU ? result.gpu() : result;
        });
}

Tensor top_p(const Tensor &logits, float p) {
    if (p <= 0.0f || p > 1.0f) {
        throw ValueError("top_p: p must be in (0, 1], got " +
                         std::to_string(p));
    }
    if (p == 1.0f) {
        return logits; // Keep all
    }

    // Work on CPU
    Tensor cpu_logits = logits.device() == Device::CPU ? logits : logits.cpu();
    auto shape = cpu_logits.shape();
    size_t last_dim = shape[shape.size() - 1];
    size_t num_rows = cpu_logits.size() / last_dim;

    Tensor contiguous = cpu_logits.is_contiguous()
                            ? cpu_logits
                            : cpu_logits.ascontiguousarray();

    return dispatch_float(
        logits.dtype(), "top_p", [&]<typename DT>(DT) -> Tensor {
            using T = typename DT::value_type;
            const T *src = contiguous.typed_data<T>();
            Tensor result(shape, logits.dtype(), Device::CPU);
            T *dst = result.typed_data<T>();

            std::vector<size_t> sorted_indices(last_dim);
            const T neg_inf =
                static_cast<T>(-std::numeric_limits<double>::infinity());

            for (size_t r = 0; r < num_rows; ++r) {
                const T *row_src = src + r * last_dim;
                T *row_dst = dst + r * last_dim;

                // Softmax for probabilities
                // Find max for numerical stability
                T max_val = row_src[0];
                for (size_t i = 1; i < last_dim; ++i) {
                    if (row_src[i] > max_val)
                        max_val = row_src[i];
                }

                // Compute exp and sum
                std::vector<double> probs(last_dim);
                double sum_exp = 0.0;
                for (size_t i = 0; i < last_dim; ++i) {
                    probs[i] = std::exp(static_cast<double>(row_src[i]) -
                                        static_cast<double>(max_val));
                    sum_exp += probs[i];
                }
                for (size_t i = 0; i < last_dim; ++i) {
                    probs[i] /= sum_exp;
                }

                // Sort indices by probability descending
                std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
                std::sort(sorted_indices.begin(), sorted_indices.end(),
                          [&probs](size_t a, size_t b) {
                              return probs[a] > probs[b];
                          });

                // Cumulative sum — find cutoff
                double cum_prob = 0.0;
                std::vector<bool> keep(last_dim, false);
                for (size_t i = 0; i < last_dim; ++i) {
                    size_t idx = sorted_indices[i];
                    keep[idx] = true;
                    cum_prob += probs[idx];
                    if (cum_prob >= static_cast<double>(p)) {
                        break;
                    }
                }

                // Apply mask
                for (size_t i = 0; i < last_dim; ++i) {
                    row_dst[i] = keep[i] ? row_src[i] : neg_inf;
                }
            }

            return logits.device() == Device::GPU ? result.gpu() : result;
        });
}

} // namespace sampling
} // namespace axiom
