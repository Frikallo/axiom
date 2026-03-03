#include "axiom_test_utils.hpp"
#include <axiom/axiom.hpp>

using namespace axiom;
using namespace axiom::nn;

TEST(NnRnn, LSTMCellConstructs) {
    LSTMCell cell;
    auto params = cell.named_parameters();
    ASSERT_FALSE(params.empty());
}

TEST(NnRnn, LSTMCellForward) {
    int input_size = 8;
    int hidden_size = 4;

    LSTMCell cell;
    // input_proj: (4*hidden, input) weight + (4*hidden) bias
    // hidden_proj: (4*hidden, hidden) weight
    std::map<std::string, Tensor> weights;
    weights["input_proj_.weight"] =
        Tensor::randn({static_cast<size_t>(4 * hidden_size),
                       static_cast<size_t>(input_size)});
    weights["input_proj_.bias"] =
        Tensor::zeros({static_cast<size_t>(4 * hidden_size)});
    weights["hidden_proj_.weight"] =
        Tensor::randn({static_cast<size_t>(4 * hidden_size),
                       static_cast<size_t>(hidden_size)});
    cell.load_state_dict(weights);

    // Forward pass
    auto x = Tensor::randn({2, static_cast<size_t>(input_size)});
    auto h = Tensor::zeros({2, static_cast<size_t>(hidden_size)});
    auto c = Tensor::zeros({2, static_cast<size_t>(hidden_size)});

    auto [h_new, c_new] = cell.forward(x, {h, c});
    ASSERT_EQ(h_new.shape(), h.shape());
    ASSERT_EQ(c_new.shape(), c.shape());
}

TEST(NnRnn, LSTMConstruction) {
    LSTM lstm(2);
    ASSERT_EQ(lstm.num_layers(), 2);
    auto params = lstm.named_parameters();
    ASSERT_FALSE(params.empty());
}

TEST(NnRnn, LSTMStep) {
    int input_size = 8;
    int hidden_size = 4;
    int num_layers = 2;

    LSTM lstm(num_layers);

    std::map<std::string, Tensor> weights;
    // Layer 0: input_size → hidden_size
    weights["cells_.0.input_proj_.weight"] =
        Tensor::randn({static_cast<size_t>(4 * hidden_size),
                       static_cast<size_t>(input_size)});
    weights["cells_.0.input_proj_.bias"] =
        Tensor::zeros({static_cast<size_t>(4 * hidden_size)});
    weights["cells_.0.hidden_proj_.weight"] =
        Tensor::randn({static_cast<size_t>(4 * hidden_size),
                       static_cast<size_t>(hidden_size)});
    // Layer 1: hidden_size → hidden_size
    weights["cells_.1.input_proj_.weight"] =
        Tensor::randn({static_cast<size_t>(4 * hidden_size),
                       static_cast<size_t>(hidden_size)});
    weights["cells_.1.input_proj_.bias"] =
        Tensor::zeros({static_cast<size_t>(4 * hidden_size)});
    weights["cells_.1.hidden_proj_.weight"] =
        Tensor::randn({static_cast<size_t>(4 * hidden_size),
                       static_cast<size_t>(hidden_size)});
    lstm.load_state_dict(weights);

    // Step
    auto x = Tensor::randn({2, static_cast<size_t>(input_size)});
    std::vector<LSTMState> states;
    for (int i = 0; i < num_layers; ++i) {
        states.push_back(
            {Tensor::zeros({2, static_cast<size_t>(hidden_size)}),
             Tensor::zeros({2, static_cast<size_t>(hidden_size)})});
    }

    auto out = lstm.step(x, states);
    ASSERT_EQ(out.shape()[0], 2);
    ASSERT_EQ(out.shape()[1], static_cast<size_t>(hidden_size));
}
