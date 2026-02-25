#include "axiom/nn/container.hpp"

#include "axiom/error.hpp"

namespace axiom::nn {

Module &ModuleList::push_back(std::unique_ptr<Module> module) {
    auto &ref = *module;
    register_module(std::to_string(modules_.size()), ref);
    modules_.push_back(std::move(module));
    return ref;
}

Module &ModuleList::operator[](size_t index) {
    if (index >= modules_.size()) {
        throw IndexError("ModuleList index " + std::to_string(index) +
                         " out of range for size " +
                         std::to_string(modules_.size()));
    }
    return *modules_[index];
}

const Module &ModuleList::operator[](size_t index) const {
    if (index >= modules_.size()) {
        throw IndexError("ModuleList index " + std::to_string(index) +
                         " out of range for size " +
                         std::to_string(modules_.size()));
    }
    return *modules_[index];
}

size_t ModuleList::size() const { return modules_.size(); }

// ============================================================================
// ModuleDict
// ============================================================================

Module &ModuleDict::insert(const std::string &key,
                           std::unique_ptr<Module> module) {
    auto &ref = *module;
    register_module(key, ref);
    modules_.emplace_back(key, std::move(module));
    return ref;
}

Module &ModuleDict::operator[](const std::string &key) {
    for (auto &[k, m] : modules_) {
        if (k == key) {
            return *m;
        }
    }
    throw IndexError("ModuleDict key '" + key + "' not found");
}

const Module &ModuleDict::operator[](const std::string &key) const {
    for (auto &[k, m] : modules_) {
        if (k == key) {
            return *m;
        }
    }
    throw IndexError("ModuleDict key '" + key + "' not found");
}

bool ModuleDict::contains(const std::string &key) const {
    for (auto &[k, m] : modules_) {
        if (k == key) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> ModuleDict::keys() const {
    std::vector<std::string> result;
    result.reserve(modules_.size());
    for (auto &[k, m] : modules_) {
        result.push_back(k);
    }
    return result;
}

size_t ModuleDict::size() const { return modules_.size(); }

// ============================================================================
// ParameterDict
// ============================================================================

void ParameterDict::insert(const std::string &key, const Tensor &param) {
    params_.emplace_back(key, param);
    register_parameter(key, params_.back().second);
}

Tensor &ParameterDict::operator[](const std::string &key) {
    for (auto &[k, p] : params_) {
        if (k == key) {
            return p;
        }
    }
    throw IndexError("ParameterDict key '" + key + "' not found");
}

const Tensor &ParameterDict::operator[](const std::string &key) const {
    for (auto &[k, p] : params_) {
        if (k == key) {
            return p;
        }
    }
    throw IndexError("ParameterDict key '" + key + "' not found");
}

bool ParameterDict::contains(const std::string &key) const {
    for (auto &[k, p] : params_) {
        if (k == key) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> ParameterDict::keys() const {
    std::vector<std::string> result;
    result.reserve(params_.size());
    for (auto &[k, p] : params_) {
        result.push_back(k);
    }
    return result;
}

size_t ParameterDict::size() const { return params_.size(); }

// ============================================================================
// Sequential
// ============================================================================

Tensor Sequential::forward(const Tensor &input) const {
    Tensor result = input;
    for (auto &m : modules_) {
        result = m->forward(result);
    }
    return result;
}

} // namespace axiom::nn
