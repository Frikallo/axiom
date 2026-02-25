#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;

// ============================================================================
// permute()
// ============================================================================

TEST(NNErgonomics, PermuteAlias) {
    auto t = Tensor::randn({2, 3, 4});
    auto p = t.permute({2, 0, 1});
    EXPECT_EQ(p.shape()[0], 4u);
    EXPECT_EQ(p.shape()[1], 2u);
    EXPECT_EQ(p.shape()[2], 3u);

    // Should be identical to transpose with same axes
    auto tr = t.transpose({2, 0, 1});
    EXPECT_TRUE(p.allclose(tr));
}

// ============================================================================
// glu()
// ============================================================================

TEST(NNErgonomics, GLUBasic) {
    // Input of shape (2, 6) — split dim=-1 into (2,3) and (2,3)
    float data[] = {1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0};
    auto input = Tensor::from_data(data, {2, 6});

    auto result = ops::glu(input, -1);
    EXPECT_EQ(result.shape()[0], 2u);
    EXPECT_EQ(result.shape()[1], 3u);

    // first_half * sigmoid(second_half)
    // sigmoid(0) = 0.5, so result = first_half * 0.5
    auto expected_ptr = result.typed_data<float>();
    EXPECT_NEAR(expected_ptr[0], 1.0f * 0.5f, 1e-5);
    EXPECT_NEAR(expected_ptr[1], 2.0f * 0.5f, 1e-5);
    EXPECT_NEAR(expected_ptr[2], 3.0f * 0.5f, 1e-5);
}

TEST(NNErgonomics, GLUMemberFunction) {
    auto input = Tensor::randn({4, 8});
    auto r1 = ops::glu(input, -1);
    auto r2 = input.glu(-1);
    EXPECT_TRUE(r1.allclose(r2));
}

TEST(NNErgonomics, GLUDim0) {
    auto input = Tensor::randn({6, 4});
    auto result = ops::glu(input, 0);
    EXPECT_EQ(result.shape()[0], 3u);
    EXPECT_EQ(result.shape()[1], 4u);
}

TEST(NNErgonomics, GLUOddDimThrows) {
    auto input = Tensor::randn({2, 5});
    EXPECT_THROW(ops::glu(input, -1), ValueError);
}

// ============================================================================
// BatchNorm1d
// ============================================================================

TEST(NNErgonomics, BatchNorm1d2D) {
    BatchNorm1d bn;

    size_t C = 3;
    // Simulate loading state dict with known values
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({C});
    state["bias"] = Tensor::zeros({C});
    state["running_mean"] = Tensor::zeros({C});
    state["running_var"] = Tensor::ones({C});
    state["num_batches_tracked"] = Tensor::zeros({1}, DType::Int64);
    bn.load_state_dict(state);

    // With mean=0, var=1, weight=1, bias=0 → output ≈ input
    auto input = Tensor::randn({4, C});
    auto output = bn(input);
    EXPECT_EQ(output.shape()[0], 4u);
    EXPECT_EQ(output.shape()[1], C);
    EXPECT_TRUE(output.allclose(input, 1e-4, 1e-4));
}

TEST(NNErgonomics, BatchNorm1d3D) {
    BatchNorm1d bn;
    size_t C = 4;

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({C});
    state["bias"] = Tensor::zeros({C});
    state["running_mean"] = Tensor::zeros({C});
    state["running_var"] = Tensor::ones({C});
    state["num_batches_tracked"] = Tensor::zeros({1}, DType::Int64);
    bn.load_state_dict(state);

    auto input = Tensor::randn({2, C, 8});
    auto output = bn(input);
    EXPECT_EQ(output.ndim(), 3u);
    EXPECT_EQ(output.shape()[1], C);
    EXPECT_TRUE(output.allclose(input, 1e-4, 1e-4));
}

TEST(NNErgonomics, BatchNorm1dNonTrivial) {
    BatchNorm1d bn;
    size_t C = 2;

    std::map<std::string, Tensor> state;
    float w[] = {2.0f, 3.0f};
    float b[] = {1.0f, -1.0f};
    float rm[] = {0.5f, -0.5f};
    float rv[] = {4.0f, 9.0f};
    state["weight"] = Tensor::from_data(w, {C});
    state["bias"] = Tensor::from_data(b, {C});
    state["running_mean"] = Tensor::from_data(rm, {C});
    state["running_var"] = Tensor::from_data(rv, {C});
    state["num_batches_tracked"] = Tensor::zeros({1}, DType::Int64);
    bn.load_state_dict(state);

    // input (1, 2): [0.5, -0.5]
    // channel 0: (0.5 - 0.5)/sqrt(4 + 1e-5) * 2 + 1 = 0 * 1 + 1 = 1
    // channel 1: (-0.5 - (-0.5))/sqrt(9 + 1e-5) * 3 + (-1) = 0 * 1 + (-1) =
    // -1
    float in[] = {0.5f, -0.5f};
    auto input = Tensor::from_data(in, {1, C});
    auto output = bn(input);
    auto ptr = output.typed_data<float>();
    EXPECT_NEAR(ptr[0], 1.0f, 1e-4);
    EXPECT_NEAR(ptr[1], -1.0f, 1e-4);
}

// ============================================================================
// BatchNorm2d
// ============================================================================

TEST(NNErgonomics, BatchNorm2d) {
    BatchNorm2d bn;
    size_t C = 3;

    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({C});
    state["bias"] = Tensor::zeros({C});
    state["running_mean"] = Tensor::zeros({C});
    state["running_var"] = Tensor::ones({C});
    state["num_batches_tracked"] = Tensor::zeros({1}, DType::Int64);
    bn.load_state_dict(state);

    auto input = Tensor::randn({2, C, 4, 4});
    auto output = bn(input);
    EXPECT_EQ(output.ndim(), 4u);
    EXPECT_EQ(output.shape()[1], C);
    EXPECT_TRUE(output.allclose(input, 1e-4, 1e-4));
}

TEST(NNErgonomics, BatchNorm2dShapeError) {
    BatchNorm2d bn;
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({3});
    state["bias"] = Tensor::zeros({3});
    state["running_mean"] = Tensor::zeros({3});
    state["running_var"] = Tensor::ones({3});
    state["num_batches_tracked"] = Tensor::zeros({1}, DType::Int64);
    bn.load_state_dict(state);

    auto input3d = Tensor::randn({2, 3, 4});
    EXPECT_THROW(bn(input3d), ShapeError);
}

// ============================================================================
// Conv config structs
// ============================================================================

TEST(NNErgonomics, Conv1dConfig) {
    Conv1dConfig cfg;
    cfg.stride = 2;
    cfg.padding = 1;
    cfg.groups = 1;

    Conv1d conv(cfg);
    // Just verify construction — forward needs loaded weights
    auto params = conv.parameters();
    EXPECT_GE(params.size(), 1u); // at least weight
}

TEST(NNErgonomics, Conv2dConfig) {
    Conv2dConfig cfg;
    cfg.stride = {2, 2};
    cfg.padding = {1, 1};

    Conv2d conv(cfg);
    auto params = conv.parameters();
    EXPECT_GE(params.size(), 1u);
}

// ============================================================================
// Registration macros
// ============================================================================

struct MacroTestModule : Module {
    Tensor weight_;
    Linear linear_;

    MacroTestModule() {
        AX_REGISTER_PARAMETER(weight_);
        AX_REGISTER_MODULE(linear_);
    }
};

TEST(NNErgonomics, RegisterMacros) {
    MacroTestModule mod;
    auto named = mod.named_parameters();
    // Should have "weight_" from our param + "linear_." submodule params
    bool found_weight = false;
    for (auto &[name, ptr] : named) {
        if (name == "weight_") {
            found_weight = true;
        }
    }
    EXPECT_TRUE(found_weight);
}

// ============================================================================
// ModuleList::each<T>()
// ============================================================================

struct DummyLayer : Module {
    int id;
    explicit DummyLayer(int id) : id(id) {}
    Tensor forward(const Tensor &input) const override { return input; }
};

TEST(NNErgonomics, ModuleListEach) {
    ModuleList list;
    list.emplace_back<DummyLayer>(10);
    list.emplace_back<DummyLayer>(20);
    list.emplace_back<DummyLayer>(30);

    std::vector<int> ids;
    for (const auto &layer : list.each<DummyLayer>()) {
        ids.push_back(layer.id);
    }
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 10);
    EXPECT_EQ(ids[1], 20);
    EXPECT_EQ(ids[2], 30);
}

// ============================================================================
// Sequential
// ============================================================================

struct DoubleModule : Module {
    Tensor forward(const Tensor &input) const override {
        return ops::add(input, input);
    }
};

TEST(NNErgonomics, SequentialForward) {
    Sequential seq;
    seq.emplace_back<DoubleModule>();
    seq.emplace_back<DoubleModule>();

    // Input x → 2x → 4x
    float data[] = {1.0f, 2.0f, 3.0f};
    auto input = Tensor::from_data(data, {3});
    auto output = seq(input);

    auto ptr = output.typed_data<float>();
    EXPECT_NEAR(ptr[0], 4.0f, 1e-5);
    EXPECT_NEAR(ptr[1], 8.0f, 1e-5);
    EXPECT_NEAR(ptr[2], 12.0f, 1e-5);
}

TEST(NNErgonomics, SequentialEmpty) {
    Sequential seq;
    auto input = Tensor::randn({3, 4});
    auto output = seq(input);
    EXPECT_TRUE(output.allclose(input));
}

TEST(NNErgonomics, SequentialSize) {
    Sequential seq;
    EXPECT_EQ(seq.size(), 0u);
    seq.emplace_back<DoubleModule>();
    EXPECT_EQ(seq.size(), 1u);
    seq.emplace_back<DoubleModule>();
    EXPECT_EQ(seq.size(), 2u);
}
