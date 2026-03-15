#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axiom/tensor.hpp"

#ifdef AXIOM_HAS_ANE
namespace axiom::backends::ane {
class ANECompiledModel;
}
#endif

namespace axiom::nn {

class Module {
  public:
    Module() = default;
    virtual ~Module() = default;

    Module(const Module &) = delete;
    Module &operator=(const Module &) = delete;
    Module(Module &&) = delete;
    Module &operator=(Module &&) = delete;

    // Single-tensor forward pass (override in modules that support it)
    // Modules with multi-arg forward (e.g. MHA) leave this as default.
    virtual Tensor forward(const Tensor &input) const;

    // Callable interface: dispatches to ANE if device is ANE, else forward().
    Tensor operator()(const Tensor &input) const;

    // Move all parameters and submodules to device.
    // Device::ANE is special: parameters stay on CPU, but forward() will
    // compile and run on the Neural Engine.
    virtual Module &to(Device device);

    // Cast all parameters and submodules to dtype
    virtual Module &to(DType dtype);

    // Get the device this module targets for inference
    Device device() const { return device_; }

    // Load weights from flat name->Tensor map with hierarchical prefix
    // resolution
    virtual void
    load_state_dict(const std::map<std::string, Tensor> &state_dict,
                    const std::string &prefix = "", bool strict = true);

    // Export all parameters as a flat name->Tensor map (inverse of
    // load_state_dict)
    std::map<std::string, Tensor>
    state_dict(const std::string &prefix = "") const;

    // Parameter introspection
    std::vector<std::pair<std::string, Tensor *>>
    named_parameters(const std::string &prefix = "") const;
    std::vector<Tensor *> parameters() const;

    // Submodule introspection
    const std::vector<std::pair<std::string, Module *>> &children() const {
        return submodules_;
    }

  protected:
    void register_parameter(const std::string &name, Tensor &param);
    void register_module(const std::string &name, Module &submodule);

  private:
    std::vector<std::pair<std::string, Tensor *>> params_;
    std::vector<std::pair<std::string, Module *>> submodules_;
    Device device_ = Device::CPU;

#ifdef AXIOM_HAS_ANE
    // Cached ANE compiled model (lazily compiled on first forward() call)
    mutable std::shared_ptr<backends::ane::ANECompiledModel> ane_model_;
    mutable Shape ane_compiled_shape_; // Input shape the cached model was
                                       // compiled for
#endif
};

// Single-item registration
#define AX_REGISTER_MODULE(m) register_module(#m, m)
#define AX_REGISTER_PARAMETER(p) register_parameter(#p, p)

// Variadic helpers — register many in one call
// Usage: AX_REGISTER_MODULES(ffn1_, attn_, conv_, ffn2_, final_norm_);
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define AX_DETAIL_APPLY(macro, x) macro(x)
#define AX_DETAIL_FE_1(macro, x) AX_DETAIL_APPLY(macro, x)
#define AX_DETAIL_FE_2(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_1(macro, __VA_ARGS__)
#define AX_DETAIL_FE_3(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_2(macro, __VA_ARGS__)
#define AX_DETAIL_FE_4(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_3(macro, __VA_ARGS__)
#define AX_DETAIL_FE_5(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_4(macro, __VA_ARGS__)
#define AX_DETAIL_FE_6(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_5(macro, __VA_ARGS__)
#define AX_DETAIL_FE_7(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_6(macro, __VA_ARGS__)
#define AX_DETAIL_FE_8(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_7(macro, __VA_ARGS__)
#define AX_DETAIL_FE_9(macro, x, ...)                                          \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_8(macro, __VA_ARGS__)
#define AX_DETAIL_FE_10(macro, x, ...)                                         \
    AX_DETAIL_APPLY(macro, x);                                                 \
    AX_DETAIL_FE_9(macro, __VA_ARGS__)

#define AX_DETAIL_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, NAME,     \
                            ...)                                               \
    NAME
#define AX_DETAIL_FOR_EACH(macro, ...)                                         \
    AX_DETAIL_GET_MACRO(__VA_ARGS__, AX_DETAIL_FE_10, AX_DETAIL_FE_9,          \
                        AX_DETAIL_FE_8, AX_DETAIL_FE_7, AX_DETAIL_FE_6,        \
                        AX_DETAIL_FE_5, AX_DETAIL_FE_4, AX_DETAIL_FE_3,        \
                        AX_DETAIL_FE_2, AX_DETAIL_FE_1)(macro, __VA_ARGS__)

#define AX_REGISTER_MODULES(...)                                               \
    AX_DETAIL_FOR_EACH(AX_REGISTER_MODULE, __VA_ARGS__)
#define AX_REGISTER_PARAMETERS(...)                                            \
    AX_DETAIL_FOR_EACH(AX_REGISTER_PARAMETER, __VA_ARGS__)
// NOLINTEND(cppcoreguidelines-macro-usage)

} // namespace axiom::nn
