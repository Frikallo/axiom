#include "axiom/nn/module.hpp"

#include "axiom/error.hpp"

#ifdef AXIOM_HAS_ANE
#include "axiom/nn/ane_compiled_model.hpp"
#endif

namespace axiom::nn {

Tensor Module::forward(const Tensor & /*input*/) const {
    throw RuntimeError(
        "forward(const Tensor&) not implemented for this module");
}

Tensor Module::operator()(const Tensor &input) const {
#ifdef AXIOM_HAS_ANE
    // Leaf-level ANE dispatch: only modules that ANE can directly
    // accelerate (Linear, Conv2d, etc.) compile and run on ANE.
    // Complex modules (conformer blocks, custom forward()) keep
    // device_==CPU and run their forward() normally — but their
    // child Linear/Conv2d modules dispatch to ANE individually.
    if (device_ == Device::ANE &&
        backends::ane::ANECompiledModel::is_supported(*this)) {
        if (!ane_model_ || ane_compiled_shape_ != input.shape()) {
            try {
                auto compiled = backends::ane::ANECompiledModel::compile(
                    *this, input.shape());
                ane_model_ = std::make_shared<backends::ane::ANECompiledModel>(
                    std::move(compiled));
                ane_compiled_shape_ = input.shape();
            } catch (const std::exception &) {
                // Compilation failed — fall back to CPU for this shape
                ane_model_.reset();
                ane_compiled_shape_ = input.shape();
            }
        }
        if (ane_model_) {
            return ane_model_->forward(input);
        }
        // Fallback: run on CPU
    }
#endif
    return forward(input);
}

Module &Module::to(Device device) {
#ifdef AXIOM_HAS_ANE
    if (device == Device::ANE) {
        // Propagate ANE to leaf modules that can be accelerated.
        // Complex modules with custom forward() keep device_==CPU
        // and run their control flow (residuals, inline ops, etc.)
        // on CPU. Their child Linear/Conv2d modules get device_==ANE
        // and dispatch individually to ANE hardware.
        if (backends::ane::ANECompiledModel::is_supported(*this)) {
            device_ = Device::ANE;
            ane_model_.reset();
            ane_compiled_shape_ = {};
        }
        // Always propagate to children so leaf modules get marked
        for (auto &[name, submodule] : submodules_) {
            submodule->to(device);
        }
        return *this;
    }

    // Moving away from ANE
    if (device_ == Device::ANE) {
        ane_model_.reset();
        ane_compiled_shape_ = {};
    }
#endif

    device_ = device;
    for (auto &[name, param] : params_) {
        if (param->storage()) {
            *param = param->to(device);
        }
    }
    for (auto &[name, submodule] : submodules_) {
        submodule->to(device);
    }
    return *this;
}

Module &Module::to(DType dtype) {
    for (auto &[name, param] : params_) {
        if (param->storage()) {
            *param = param->astype(dtype);
        }
    }
    for (auto &[name, submodule] : submodules_) {
        submodule->to(dtype);
    }
    return *this;
}

void Module::load_state_dict(const std::map<std::string, Tensor> &state_dict,
                             const std::string &prefix, bool strict) {
    for (auto &[name, param] : params_) {
        auto key = prefix + name;
        auto it = state_dict.find(key);
        if (it != state_dict.end()) {
            *param = it->second;
        } else if (strict) {
            throw ValueError("missing key '" + key + "' in state_dict");
        }
    }
    for (auto &[name, submodule] : submodules_) {
        submodule->load_state_dict(state_dict, prefix + name + ".", strict);
    }

#ifdef AXIOM_HAS_ANE
    // Invalidate ANE cache when weights change
    if (device_ == Device::ANE) {
        ane_model_.reset();
        ane_compiled_shape_ = {};
    }
#endif
}

std::map<std::string, Tensor>
Module::state_dict(const std::string &prefix) const {
    std::map<std::string, Tensor> result;
    for (auto &[name, param] : params_) {
        result[prefix + name] = *param;
    }
    for (auto &[name, submodule] : submodules_) {
        auto sub_dict = submodule->state_dict(prefix + name + ".");
        result.insert(sub_dict.begin(), sub_dict.end());
    }
    return result;
}

std::vector<std::pair<std::string, Tensor *>>
Module::named_parameters(const std::string &prefix) const {
    std::vector<std::pair<std::string, Tensor *>> result;
    for (auto &[name, param] : params_) {
        result.emplace_back(prefix + name, param);
    }
    for (auto &[name, submodule] : submodules_) {
        auto sub_params = submodule->named_parameters(prefix + name + ".");
        result.insert(result.end(), sub_params.begin(), sub_params.end());
    }
    return result;
}

std::vector<Tensor *> Module::parameters() const {
    std::vector<Tensor *> result;
    for (auto &[name, param] : params_) {
        result.push_back(param);
    }
    for (auto &[name, submodule] : submodules_) {
        auto sub_params = submodule->parameters();
        result.insert(result.end(), sub_params.begin(), sub_params.end());
    }
    return result;
}

void Module::register_parameter(const std::string &name, Tensor &param) {
    params_.emplace_back(name, &param);
}

void Module::register_module(const std::string &name, Module &submodule) {
    submodules_.emplace_back(name, &submodule);
}

} // namespace axiom::nn
