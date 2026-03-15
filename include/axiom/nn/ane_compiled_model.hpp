#pragma once

#ifdef AXIOM_HAS_ANE

#include <memory>
#include <string>
#include <vector>

#include "axiom/tensor.hpp"

namespace axiom {

// Forward declarations
namespace nn {
class Module;
}

namespace backends {
namespace ane {

// Diagnostic info about a compiled ANE execution plan.
struct ANEPlanInfo {
    int ane_steps;       // Number of ANE dispatch steps
    int cpu_steps;       // Number of CPU fallback steps
    int compile_count;   // ANE compilations used
    size_t weight_bytes; // Total weight data in FP16
    std::string summary; // Human-readable summary
};

// A compiled model that runs NN module inference on Apple Neural Engine.
//
// Usage:
//   auto model = nn::Linear(512, 768);
//   model.load_state_dict(weights);
//   auto compiled = ANECompiledModel::compile(model, {1, 512});
//   auto output = compiled.forward(randn({1, 512}));
//
// The model is compiled once (including MIL generation, weight packing,
// and ANE hardware loading). Subsequent forward() calls reuse the compiled
// graph with zero overhead beyond IOSurface data transfer.
//
// Layout conversion is handled automatically:
//   Axiom: [batch, ..., features] (row-major FP32)
//   ANE:   [1, features, 1, spatial] (channel-first FP16)
class ANECompiledModel {
  public:
    ~ANECompiledModel();

    // Non-copyable, movable
    ANECompiledModel(const ANECompiledModel &) = delete;
    ANECompiledModel &operator=(const ANECompiledModel &) = delete;
    ANECompiledModel(ANECompiledModel &&) noexcept;
    ANECompiledModel &operator=(ANECompiledModel &&) noexcept;

    // Compile an NN module for ANE inference.
    // input_shape: the expected input tensor shape (e.g., {batch, features}
    //              or {batch, seq, features}).
    // quantize_weights: if true, quantize Linear weights to INT8 (1.88x
    //                   bandwidth savings; compute stays FP16).
    // Throws on failure (module unsupported or ANE unavailable).
    static ANECompiledModel compile(const nn::Module &module,
                                    const Shape &input_shape,
                                    bool quantize_weights = false);

    // Compile with dynamic weight staging.
    // Weights are packed into the input IOSurface on each forward() call,
    // so weight updates (load_state_dict) do NOT require recompilation.
    // Only Linear layers are supported in dynamic mode.
    static ANECompiledModel compile_dynamic(const nn::Module &module,
                                            const Shape &input_shape);

    // Re-stage weights from a module into the cached IOSurface.
    // Only valid for dynamically-compiled models.
    void update_weights(const nn::Module &module);

    // Run inference on ANE.
    // Input must match the shape declared at compile time.
    // Returns a CPU tensor with the module's output.
    Tensor forward(const Tensor &input) const;

    // Check if a module type is supported for ANE compilation.
    static bool is_supported(const nn::Module &module);

    // Get diagnostic info about the execution plan.
    ANEPlanInfo plan_info() const;

  private:
    ANECompiledModel();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ane
} // namespace backends
} // namespace axiom

#endif // AXIOM_HAS_ANE
