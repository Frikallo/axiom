#include "axiom/nn/module.hpp"

#include "axiom/error.hpp"

namespace axiom::nn {

Module &Module::to(Device device) {
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
