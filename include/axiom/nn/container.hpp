#pragma once

#include <memory>
#include <string>
#include <vector>

#include "axiom/nn/module.hpp"

namespace axiom::nn {

class ModuleList : public Module {
  public:
    ModuleList() = default;

    // Add a module constructed in-place, returns typed reference
    template <typename T, typename... Args> T &emplace_back(Args &&...args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        auto &ref = *ptr;
        register_module(std::to_string(modules_.size()), ref);
        modules_.push_back(std::move(ptr));
        return ref;
    }

    // Add a pre-constructed module
    Module &push_back(std::unique_ptr<Module> module);

    Module &operator[](size_t index);
    const Module &operator[](size_t index) const;
    size_t size() const;

    auto begin() { return modules_.begin(); }
    auto end() { return modules_.end(); }
    auto begin() const { return modules_.begin(); }
    auto end() const { return modules_.end(); }

  private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace axiom::nn
