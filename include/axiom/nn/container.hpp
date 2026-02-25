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

    // Typed iteration: for (auto &block : layers_.each<ConformerBlock>()) { ...
    // }
    template <typename T, bool Const> struct TypedModuleRangeImpl {
        using Vec = std::vector<std::unique_ptr<Module>>;
        using VecRef = std::conditional_t<Const, const Vec &, Vec &>;
        using Ref = std::conditional_t<Const, const T &, T &>;
        using It = std::conditional_t<Const, typename Vec::const_iterator,
                                      typename Vec::iterator>;

        struct Iterator {
            It it;
            Ref operator*() const { return static_cast<Ref>(**it); }
            Iterator &operator++() {
                ++it;
                return *this;
            }
            bool operator!=(const Iterator &o) const { return it != o.it; }
        };

        VecRef modules;
        Iterator begin() const { return {modules.begin()}; }
        Iterator end() const { return {modules.end()}; }
    };

    template <typename T> TypedModuleRangeImpl<T, true> each() const {
        return {modules_};
    }
    template <typename T> TypedModuleRangeImpl<T, false> each() {
        return {modules_};
    }

  private:
    std::vector<std::unique_ptr<Module>> modules_;
};

class ModuleDict : public Module {
  public:
    ModuleDict() = default;

    // Add a module constructed in-place, returns typed reference
    template <typename T, typename... Args>
    T &emplace(const std::string &key, Args &&...args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        auto &ref = *ptr;
        register_module(key, ref);
        modules_.emplace_back(key, std::move(ptr));
        return ref;
    }

    // Add a pre-constructed module
    Module &insert(const std::string &key, std::unique_ptr<Module> module);

    Module &operator[](const std::string &key);
    const Module &operator[](const std::string &key) const;

    bool contains(const std::string &key) const;
    std::vector<std::string> keys() const;
    size_t size() const;

    template <typename T> T &get(const std::string &key) {
        return static_cast<T &>((*this)[key]);
    }
    template <typename T> const T &get(const std::string &key) const {
        return static_cast<const T &>((*this)[key]);
    }

    auto begin() { return modules_.begin(); }
    auto end() { return modules_.end(); }
    auto begin() const { return modules_.begin(); }
    auto end() const { return modules_.end(); }

  private:
    std::vector<std::pair<std::string, std::unique_ptr<Module>>> modules_;
};

class ParameterDict : public Module {
  public:
    ParameterDict() = default;

    void insert(const std::string &key, const Tensor &param);

    Tensor &operator[](const std::string &key);
    const Tensor &operator[](const std::string &key) const;

    bool contains(const std::string &key) const;
    std::vector<std::string> keys() const;
    size_t size() const;

  private:
    std::vector<std::pair<std::string, Tensor>> params_;
};

class Sequential : public Module {
  public:
    Sequential() = default;

    template <typename T, typename... Args> T &emplace_back(Args &&...args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        auto &ref = *ptr;
        register_module(std::to_string(modules_.size()), ref);
        modules_.push_back(std::move(ptr));
        return ref;
    }

    Tensor forward(const Tensor &input) const override;
    Tensor operator()(const Tensor &input) const { return forward(input); }

    size_t size() const { return modules_.size(); }

  private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace axiom::nn
