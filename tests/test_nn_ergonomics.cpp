#include "axiom_test_utils.hpp"

#include <axiom/nn.hpp>

using namespace axiom;
using namespace axiom::nn;
using namespace axiom::testing;

// ============================================================================
// Helper: create BatchNorm state dict with identity transform (mean=0, var=1,
// weight=1, bias=0) so output ≈ input.
// ============================================================================

static std::map<std::string, Tensor> identity_bn_state(size_t C, DType dtype) {
    std::map<std::string, Tensor> state;
    state["weight"] = Tensor::ones({C}, dtype);
    state["bias"] = Tensor::zeros({C}, dtype);
    state["running_mean"] = Tensor::zeros({C}, dtype);
    state["running_var"] = Tensor::ones({C}, dtype);
    state["num_batches_tracked"] = Tensor::zeros({1}, DType::Int64);
    return state;
}

// ============================================================================
// permute()
// ============================================================================

TEST(NNErgonomics, PermuteAlias) {
    auto t = Tensor::randn({2, 3, 4});
    auto p = t.permute({2, 0, 1});
    EXPECT_EQ(p.shape()[0], 4u);
    EXPECT_EQ(p.shape()[1], 2u);
    EXPECT_EQ(p.shape()[2], 3u);

    // Identical to transpose with same axes
    auto tr = t.transpose({2, 0, 1});
    EXPECT_TRUE(p.allclose(tr));
}

// ============================================================================
// glu() — correctness, member, dim variants, dtype, GPU
// ============================================================================

TEST(NNErgonomics, GLUBasic) {
    // sigmoid(0) = 0.5, so result = first_half * 0.5
    float data[] = {1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0};
    auto input = Tensor::from_data(data, {2, 6});
    auto result = ops::glu(input, -1);

    EXPECT_EQ(result.shape()[0], 2u);
    EXPECT_EQ(result.shape()[1], 3u);

    auto ptr = result.typed_data<float>();
    EXPECT_NEAR(ptr[0], 0.5f, 1e-5);
    EXPECT_NEAR(ptr[1], 1.0f, 1e-5);
    EXPECT_NEAR(ptr[2], 1.5f, 1e-5);
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

TEST(NNErgonomics, GLUFloat64) {
    auto input = Tensor::randn({4, 8}, DType::Float64);
    auto result = ops::glu(input, -1);
    EXPECT_EQ(result.dtype(), DType::Float64);
    EXPECT_EQ(result.shape()[1], 4u);
}

TEST(NNErgonomics, ChunkGPUParity) {
    SKIP_IF_NO_GPU();
    // Verify chunk() views transfer correctly to/from GPU
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto cpu_t = Tensor::from_data(data, {2, 4});
    auto gpu_t = cpu_t.gpu();

    auto cpu_chunks = cpu_t.chunk(2, -1);
    auto gpu_chunks = gpu_t.chunk(2, -1);

    auto gc0 = gpu_chunks[0].cpu();
    auto gc1 = gpu_chunks[1].cpu();

    // chunk[0] should be [[1,2],[5,6]], chunk[1] should be [[3,4],[7,8]]
    EXPECT_TRUE(cpu_chunks[0].allclose(gc0, 1e-5, 1e-5))
        << "chunk[0] GPU/CPU mismatch: CPU=["
        << cpu_chunks[0].item<float>({0, 0}) << ","
        << cpu_chunks[0].item<float>({0, 1}) << ";"
        << cpu_chunks[0].item<float>({1, 0}) << ","
        << cpu_chunks[0].item<float>({1, 1}) << "] vs GPU=["
        << gc0.item<float>({0, 0}) << "," << gc0.item<float>({0, 1}) << ";"
        << gc0.item<float>({1, 0}) << "," << gc0.item<float>({1, 1}) << "]";
    EXPECT_TRUE(cpu_chunks[1].allclose(gc1, 1e-5, 1e-5))
        << "chunk[1] GPU/CPU mismatch: CPU=["
        << cpu_chunks[1].item<float>({0, 0}) << ","
        << cpu_chunks[1].item<float>({0, 1}) << ";"
        << cpu_chunks[1].item<float>({1, 0}) << ","
        << cpu_chunks[1].item<float>({1, 1}) << "] vs GPU=["
        << gc1.item<float>({0, 0}) << "," << gc1.item<float>({0, 1}) << ";"
        << gc1.item<float>({1, 0}) << "," << gc1.item<float>({1, 1}) << "]";
}

TEST(NNErgonomics, SigmoidOnGPUSlice) {
    SKIP_IF_NO_GPU();
    // Use distinct values so sigmoid results differ per-element
    float data[] = {-2, -1, 0, 1, 2, 3, 4, 5};
    auto cpu_t = Tensor::from_data(data, {2, 4});
    auto gpu_t = cpu_t.gpu();

    auto cpu_chunks = cpu_t.chunk(2, -1);
    auto gpu_chunks = gpu_t.chunk(2, -1);

    auto cpu_sig = ops::sigmoid(cpu_chunks[1]);
    auto gpu_sig = ops::sigmoid(gpu_chunks[1]).cpu();

    EXPECT_TRUE(cpu_sig.allclose(gpu_sig, 1e-5, 1e-5))
        << "sigmoid on GPU slice doesn't match CPU: CPU=["
        << cpu_sig.item<float>({0, 0}) << "," << cpu_sig.item<float>({0, 1})
        << ";" << cpu_sig.item<float>({1, 0}) << ","
        << cpu_sig.item<float>({1, 1}) << "] vs GPU=["
        << gpu_sig.item<float>({0, 0}) << "," << gpu_sig.item<float>({0, 1})
        << ";" << gpu_sig.item<float>({1, 0}) << ","
        << gpu_sig.item<float>({1, 1}) << "]";
}

TEST(NNErgonomics, MultiplyGPUSlices) {
    SKIP_IF_NO_GPU();
    // Test multiply on two non-contiguous GPU slices (the core of GLU)
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto cpu_t = Tensor::from_data(data, {2, 4});
    auto gpu_t = cpu_t.gpu();

    auto cpu_chunks = cpu_t.chunk(2, -1);
    auto gpu_chunks = gpu_t.chunk(2, -1);

    // multiply(chunk[0], chunk[1]) — both non-contiguous
    auto cpu_mul = ops::multiply(cpu_chunks[0], cpu_chunks[1]);
    auto gpu_mul = ops::multiply(gpu_chunks[0], gpu_chunks[1]).cpu();

    EXPECT_TRUE(cpu_mul.allclose(gpu_mul, 1e-5, 1e-5))
        << "multiply on GPU slices doesn't match CPU: CPU=["
        << cpu_mul.item<float>({0, 0}) << "," << cpu_mul.item<float>({0, 1})
        << ";" << cpu_mul.item<float>({1, 0}) << ","
        << cpu_mul.item<float>({1, 1}) << "] vs GPU=["
        << gpu_mul.item<float>({0, 0}) << "," << gpu_mul.item<float>({0, 1})
        << ";" << gpu_mul.item<float>({1, 0}) << ","
        << gpu_mul.item<float>({1, 1}) << "]";
}

TEST(NNErgonomics, GLUGPU) {
    SKIP_IF_NO_GPU();
    auto cpu_input = Tensor::uniform(-2.0, 2.0, {4, 8}, DType::Float32);
    auto gpu_input = cpu_input.gpu();

    auto gpu_result = ops::glu(gpu_input, -1);
    EXPECT_EQ(gpu_result.device(), Device::GPU);
    EXPECT_EQ(gpu_result.shape()[1], 4u);

    auto cpu_result = ops::glu(cpu_input, -1);
    EXPECT_TRUE(cpu_result.allclose(gpu_result.cpu(), 1e-4, 1e-4))
        << "GLU GPU/CPU mismatch";
}

// ============================================================================
// BatchNorm1d — 2D, 3D, non-trivial values, dtype preservation, GPU
// ============================================================================

TEST(NNErgonomics, BatchNorm1d2D) {
    BatchNorm1d bn;
    size_t C = 3;
    bn.load_state_dict(identity_bn_state(C, DType::Float32));

    auto input = Tensor::randn({4, C});
    auto output = bn(input);
    EXPECT_EQ(output.shape()[0], 4u);
    EXPECT_EQ(output.shape()[1], C);
    EXPECT_EQ(output.dtype(), DType::Float32);
    EXPECT_TRUE(output.allclose(input, 1e-4, 1e-4));
}

TEST(NNErgonomics, BatchNorm1d3D) {
    BatchNorm1d bn;
    size_t C = 4;
    bn.load_state_dict(identity_bn_state(C, DType::Float32));

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

    // channel 0: (0.5 - 0.5)/sqrt(4 + 1e-5) * 2 + 1 = 1
    // channel 1: (-0.5 - (-0.5))/sqrt(9 + 1e-5) * 3 + (-1) = -1
    float in[] = {0.5f, -0.5f};
    auto input = Tensor::from_data(in, {1, C});
    auto output = bn(input);
    auto ptr = output.typed_data<float>();
    EXPECT_NEAR(ptr[0], 1.0f, 1e-4);
    EXPECT_NEAR(ptr[1], -1.0f, 1e-4);
}

TEST(NNErgonomics, BatchNorm1dFloat64) {
    BatchNorm1d bn;
    size_t C = 3;
    bn.load_state_dict(identity_bn_state(C, DType::Float64));

    auto input = Tensor::randn({4, C}, DType::Float64);
    auto output = bn(input);
    EXPECT_EQ(output.dtype(), DType::Float64);
    // Identity BN: x / sqrt(1 + eps) ≈ x with ~5e-6 error from eps
    EXPECT_TRUE(output.allclose(input, 1e-4, 1e-4));
}

TEST(NNErgonomics, BatchNorm1dDtypePreservation) {
    // Input Float64 with Float32 stats — output must still be Float64
    BatchNorm1d bn;
    size_t C = 2;
    bn.load_state_dict(identity_bn_state(C, DType::Float32));

    auto input = Tensor::randn({2, C}, DType::Float64);
    auto output = bn(input);
    EXPECT_EQ(output.dtype(), DType::Float64)
        << "Output dtype should match input dtype";
}

TEST(NNErgonomics, BatchNorm1dGPU) {
    SKIP_IF_NO_GPU();
    BatchNorm1d bn;
    size_t C = 3;
    bn.load_state_dict(identity_bn_state(C, DType::Float32));

    auto input = Tensor::randn({4, C}).gpu();
    auto output = bn(input);
    EXPECT_EQ(output.device(), Device::GPU);
    EXPECT_EQ(output.dtype(), DType::Float32);

    // Cross-check against CPU
    auto cpu_output = bn(input.cpu());
    ExpectTensorsClose(output.cpu(), cpu_output, 1e-4, 1e-4);
}

// ============================================================================
// BatchNorm2d — basic, shape error, dtype, GPU
// ============================================================================

TEST(NNErgonomics, BatchNorm2d) {
    BatchNorm2d bn;
    size_t C = 3;
    bn.load_state_dict(identity_bn_state(C, DType::Float32));

    auto input = Tensor::randn({2, C, 4, 4});
    auto output = bn(input);
    EXPECT_EQ(output.ndim(), 4u);
    EXPECT_EQ(output.shape()[1], C);
    EXPECT_TRUE(output.allclose(input, 1e-4, 1e-4));
}

TEST(NNErgonomics, BatchNorm2dShapeError) {
    BatchNorm2d bn;
    bn.load_state_dict(identity_bn_state(3, DType::Float32));
    EXPECT_THROW(bn(Tensor::randn({2, 3, 4})), ShapeError);
}

TEST(NNErgonomics, BatchNorm2dFloat64) {
    BatchNorm2d bn;
    size_t C = 3;
    bn.load_state_dict(identity_bn_state(C, DType::Float64));

    auto input = Tensor::randn({2, C, 4, 4}, DType::Float64);
    auto output = bn(input);
    EXPECT_EQ(output.dtype(), DType::Float64);
    EXPECT_TRUE(output.allclose(input, 1e-4, 1e-4));
}

TEST(NNErgonomics, BatchNorm2dGPU) {
    SKIP_IF_NO_GPU();
    BatchNorm2d bn;
    size_t C = 3;
    bn.load_state_dict(identity_bn_state(C, DType::Float32));

    auto input = Tensor::randn({2, C, 4, 4}).gpu();
    auto output = bn(input);
    EXPECT_EQ(output.device(), Device::GPU);

    auto cpu_output = bn(input.cpu());
    ExpectTensorsClose(output.cpu(), cpu_output, 1e-4, 1e-4);
}

// ============================================================================
// Conv config structs
// ============================================================================

TEST(NNErgonomics, Conv1dConfig) {
    Conv1dConfig cfg;
    cfg.stride = 2;
    cfg.padding = 1;

    Conv1d conv(cfg);
    auto params = conv.parameters();
    EXPECT_GE(params.size(), 1u);
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
// Registration macros — single and variadic
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
    bool found_weight = false;
    for (auto &[name, ptr] : named) {
        if (name == "weight_") {
            found_weight = true;
        }
    }
    EXPECT_TRUE(found_weight);
}

struct VariadicMacroModule : Module {
    Tensor w1_;
    Tensor w2_;
    Tensor w3_;
    Linear lin1_;
    Linear lin2_;

    VariadicMacroModule() {
        AX_REGISTER_PARAMETERS(w1_, w2_, w3_);
        AX_REGISTER_MODULES(lin1_, lin2_);
    }
};

TEST(NNErgonomics, RegisterVariadicMacros) {
    VariadicMacroModule mod;
    auto named = mod.named_parameters();
    EXPECT_EQ(named.size(), 7u);

    std::vector<std::string> names;
    for (auto &[name, ptr] : named) {
        names.push_back(name);
    }
    EXPECT_EQ(names[0], "w1_");
    EXPECT_EQ(names[1], "w2_");
    EXPECT_EQ(names[2], "w3_");
    EXPECT_EQ(names[3], "lin1_.weight");
    EXPECT_EQ(names[4], "lin1_.bias");
    EXPECT_EQ(names[5], "lin2_.weight");
    EXPECT_EQ(names[6], "lin2_.bias");
}

// ============================================================================
// ModuleList::each<T>() — const and mutable
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

TEST(NNErgonomics, ModuleListEachMutable) {
    ModuleList list;
    list.emplace_back<DummyLayer>(1);
    list.emplace_back<DummyLayer>(2);

    for (auto &layer : list.each<DummyLayer>()) {
        layer.id *= 100;
    }

    std::vector<int> ids;
    for (const auto &layer : list.each<DummyLayer>()) {
        ids.push_back(layer.id);
    }
    EXPECT_EQ(ids[0], 100);
    EXPECT_EQ(ids[1], 200);
}

// ============================================================================
// Sequential — forward chaining, empty, size, GPU
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

    // x → 2x → 4x
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

TEST(NNErgonomics, SequentialGPU) {
    SKIP_IF_NO_GPU();
    Sequential seq;
    seq.emplace_back<DoubleModule>();

    auto input = Tensor::randn({4}).gpu();
    auto output = seq(input);
    EXPECT_EQ(output.device(), Device::GPU);

    auto cpu_output = seq(input.cpu());
    ExpectTensorsClose(output.cpu(), cpu_output, 1e-5, 1e-5);
}

// ============================================================================
// ModuleDict
// ============================================================================

TEST(NNErgonomics, ModuleDictInsertAndAccess) {
    ModuleDict dict;
    dict.emplace<Linear>("linear");
    dict.emplace<LayerNorm>("norm");

    EXPECT_EQ(dict.size(), 2u);
    EXPECT_TRUE(dict.contains("linear"));
    EXPECT_TRUE(dict.contains("norm"));
    EXPECT_FALSE(dict.contains("missing"));
}

TEST(NNErgonomics, ModuleDictKeys) {
    ModuleDict dict;
    dict.emplace<Linear>("b_linear");
    dict.emplace<Linear>("a_linear");

    auto keys = dict.keys();
    ASSERT_EQ(keys.size(), 2u);
    // Insertion order preserved
    EXPECT_EQ(keys[0], "b_linear");
    EXPECT_EQ(keys[1], "a_linear");
}

TEST(NNErgonomics, ModuleDictTypedGet) {
    ModuleDict dict;
    dict.emplace<Linear>("lin");

    auto &lin = dict.get<Linear>("lin");
    auto params = lin.parameters();
    EXPECT_GE(params.size(), 1u);
}

TEST(NNErgonomics, ModuleDictStateDictIntegration) {
    ModuleDict dict;
    dict.emplace<Linear>("encoder");

    auto named = dict.named_parameters();
    bool found = false;
    for (auto &[name, ptr] : named) {
        if (name.find("encoder") != std::string::npos) {
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST(NNErgonomics, ModuleDictToDevice) {
    ModuleDict dict;
    dict.emplace<Linear>("lin");

    // Load state
    std::map<std::string, Tensor> state;
    state["lin.weight"] = Tensor::randn({4, 3});
    state["lin.bias"] = Tensor::randn({4});
    dict.load_state_dict(state);

    dict.to(Device::CPU); // Should not crash
    auto &lin = dict.get<Linear>("lin");
    auto params = lin.parameters();
    for (auto *p : params) {
        if (p->storage()) {
            EXPECT_EQ(p->device(), Device::CPU);
        }
    }
}

TEST(NNErgonomics, ModuleDictMissing) {
    ModuleDict dict;
    EXPECT_THROW(dict["missing"], IndexError);
}

// ============================================================================
// ParameterDict
// ============================================================================

TEST(NNErgonomics, ParameterDictInsertAndAccess) {
    ParameterDict pdict;
    pdict.insert("weight", Tensor::ones({3, 4}));
    pdict.insert("bias", Tensor::zeros({3}));

    EXPECT_EQ(pdict.size(), 2u);
    EXPECT_TRUE(pdict.contains("weight"));
    EXPECT_TRUE(pdict.contains("bias"));
    EXPECT_TRUE(pdict["weight"].shape() == Shape({3, 4}));
}

TEST(NNErgonomics, ParameterDictKeys) {
    ParameterDict pdict;
    pdict.insert("beta", Tensor::ones({2}));
    pdict.insert("alpha", Tensor::ones({3}));

    auto keys = pdict.keys();
    ASSERT_EQ(keys.size(), 2u);
    EXPECT_EQ(keys[0], "beta");
    EXPECT_EQ(keys[1], "alpha");
}

TEST(NNErgonomics, ParameterDictStateDictIntegration) {
    ParameterDict pdict;
    pdict.insert("weight", Tensor::ones({3, 4}));

    // state_dict should include the parameter
    auto sd = pdict.state_dict();
    EXPECT_TRUE(sd.count("weight") > 0);
}

TEST(NNErgonomics, ParameterDictLoadStateDict) {
    ParameterDict pdict;
    pdict.insert("w", Tensor());

    std::map<std::string, Tensor> state;
    state["w"] = Tensor::randn({5, 5});
    pdict.load_state_dict(state);

    EXPECT_TRUE(pdict["w"].shape() == Shape({5, 5}));
}

TEST(NNErgonomics, ParameterDictToDevice) {
    ParameterDict pdict;
    pdict.insert("w", Tensor::ones({3}));
    pdict.to(Device::CPU);

    EXPECT_EQ(pdict["w"].device(), Device::CPU);
}

TEST(NNErgonomics, ParameterDictMissing) {
    ParameterDict pdict;
    EXPECT_THROW(pdict["missing"], IndexError);
}
