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
