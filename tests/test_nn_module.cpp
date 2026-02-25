#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;

// A simple test module with two parameters and a submodule
struct SimpleModule : Module {
    Tensor weight_;
    Tensor bias_;

    SimpleModule() {
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }
};

struct ParentModule : Module {
    SimpleModule child_;
    Tensor scale_;

    ParentModule() {
        register_module("child", child_);
        register_parameter("scale", scale_);
    }
};

// ============================================================================
// register_parameter / register_module
// ============================================================================

TEST(NNModule, RegisterParameter) {
    SimpleModule mod;
    auto params = mod.parameters();
    ASSERT_EQ(params.size(), 2u);
}

TEST(NNModule, NamedParameters) {
    SimpleModule mod;
    auto named = mod.named_parameters();
    ASSERT_EQ(named.size(), 2u);
    EXPECT_EQ(named[0].first, "weight");
    EXPECT_EQ(named[1].first, "bias");
}

TEST(NNModule, NamedParametersWithPrefix) {
    SimpleModule mod;
    auto named = mod.named_parameters("layer.");
    ASSERT_EQ(named.size(), 2u);
    EXPECT_EQ(named[0].first, "layer.weight");
    EXPECT_EQ(named[1].first, "layer.bias");
}

TEST(NNModule, SubmoduleParameters) {
    ParentModule parent;
    auto named = parent.named_parameters();
    ASSERT_EQ(named.size(), 3u);
    // Own parameter
    EXPECT_EQ(named[0].first, "scale");
    // Child parameters
    EXPECT_EQ(named[1].first, "child.weight");
    EXPECT_EQ(named[2].first, "child.bias");
}

// ============================================================================
// load_state_dict
// ============================================================================

TEST(NNModule, LoadStateDict) {
    SimpleModule mod;
    std::map<std::string, Tensor> state_dict;
    state_dict["weight"] = Tensor::ones({3, 4});
    state_dict["bias"] = Tensor::zeros({3});

    mod.load_state_dict(state_dict);

    ASSERT_TRUE(mod.weight_.storage() != nullptr);
    EXPECT_TRUE(mod.weight_.shape() == Shape({3, 4}));
    EXPECT_TRUE(mod.bias_.shape() == Shape({3}));
    EXPECT_FLOAT_EQ(mod.weight_.item<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(mod.bias_.item<float>({0}), 0.0f);
}

TEST(NNModule, LoadStateDictWithPrefix) {
    SimpleModule mod;
    std::map<std::string, Tensor> state_dict;
    state_dict["model.weight"] = Tensor::ones({2, 3});
    state_dict["model.bias"] = Tensor::zeros({2});

    mod.load_state_dict(state_dict, "model.");

    EXPECT_TRUE(mod.weight_.shape() == Shape({2, 3}));
    EXPECT_TRUE(mod.bias_.shape() == Shape({2}));
}

TEST(NNModule, LoadStateDictHierarchical) {
    ParentModule parent;
    std::map<std::string, Tensor> state_dict;
    state_dict["scale"] = Tensor::full({1}, 2.0f);
    state_dict["child.weight"] = Tensor::ones({4, 5});
    state_dict["child.bias"] = Tensor::zeros({4});

    parent.load_state_dict(state_dict);

    EXPECT_FLOAT_EQ(parent.scale_.item<float>({0}), 2.0f);
    EXPECT_TRUE(parent.child_.weight_.shape() == Shape({4, 5}));
}

TEST(NNModule, LoadStateDictStrictMissingKey) {
    SimpleModule mod;
    std::map<std::string, Tensor> state_dict;
    state_dict["weight"] = Tensor::ones({3, 4});
    // Missing "bias"

    EXPECT_THROW(mod.load_state_dict(state_dict, "", true), ValueError);
}

TEST(NNModule, LoadStateDictNonStrict) {
    SimpleModule mod;
    std::map<std::string, Tensor> state_dict;
    state_dict["weight"] = Tensor::ones({3, 4});
    // Missing "bias" but strict=false

    EXPECT_NO_THROW(mod.load_state_dict(state_dict, "", false));
    EXPECT_TRUE(mod.weight_.shape() == Shape({3, 4}));
}

// ============================================================================
// to(Device)
// ============================================================================

TEST(NNModule, ToDevice) {
    SimpleModule mod;
    mod.weight_ = Tensor::ones({2, 3});
    mod.bias_ = Tensor::zeros({2});
    EXPECT_EQ(mod.weight_.device(), Device::CPU);

    // to(CPU) is a no-op but should not crash
    mod.to(Device::CPU);
    EXPECT_EQ(mod.weight_.device(), Device::CPU);
}

TEST(NNModule, ToDeviceRecursive) {
    ParentModule parent;
    parent.scale_ = Tensor::ones({1});
    parent.child_.weight_ = Tensor::ones({3, 4});
    parent.child_.bias_ = Tensor::zeros({3});

    parent.to(Device::CPU);
    EXPECT_EQ(parent.scale_.device(), Device::CPU);
    EXPECT_EQ(parent.child_.weight_.device(), Device::CPU);
}

TEST(NNModule, ToDeviceWithUninitializedParams) {
    // to() should skip uninitialized (empty) tensors without crashing
    SimpleModule mod;
    EXPECT_NO_THROW(mod.to(Device::CPU));
}

// ============================================================================
// Extra keys in state_dict
// ============================================================================

TEST(NNModule, LoadStateDictExtraKeysIgnored) {
    // Extra keys in state_dict are silently ignored (not an error)
    SimpleModule mod;
    std::map<std::string, Tensor> state_dict;
    state_dict["weight"] = Tensor::ones({3, 4});
    state_dict["bias"] = Tensor::zeros({3});
    state_dict["unknown_key"] = Tensor::ones({5});

    EXPECT_NO_THROW(mod.load_state_dict(state_dict));
    EXPECT_TRUE(mod.weight_.shape() == Shape({3, 4}));
}
