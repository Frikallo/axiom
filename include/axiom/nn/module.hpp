#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "axiom/tensor.hpp"

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

    // Move all parameters and submodules to device
    virtual Module &to(Device device);

    // Load weights from flat name->Tensor map with hierarchical prefix
    // resolution
    virtual void
    load_state_dict(const std::map<std::string, Tensor> &state_dict,
                    const std::string &prefix = "", bool strict = true);

    // Parameter introspection
    std::vector<std::pair<std::string, Tensor *>>
    named_parameters(const std::string &prefix = "") const;
    std::vector<Tensor *> parameters() const;

  protected:
    void register_parameter(const std::string &name, Tensor &param);
    void register_module(const std::string &name, Module &submodule);

  private:
    std::vector<std::pair<std::string, Tensor *>> params_;
    std::vector<std::pair<std::string, Module *>> submodules_;
};

// Convenience macros: register_module(#name, name) / register_parameter(#name,
// name)
#define AX_REGISTER_MODULE(m) register_module(#m, m)
#define AX_REGISTER_PARAMETER(p) register_parameter(#p, p)

} // namespace axiom::nn
