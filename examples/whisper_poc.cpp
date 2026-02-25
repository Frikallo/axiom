// Whisper Inference POC
//
// Full encoder-decoder transformer matching HuggingFace whisper-tiny.
// Loads weights from model.safetensors and runs greedy decoding.
//
// Usage:
//   ./whisper_poc <path-to-model.safetensors>
//
// Download model:
//   curl -L
//   https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors \
//        -o whisper-tiny.safetensors

#include <axiom/axiom.hpp>
#include <axiom/nn.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

using namespace axiom;
using namespace axiom::nn;

// ============================================================================
// Whisper config (whisper-tiny defaults)
// ============================================================================

struct WhisperConfig {
    int d_model = 384;
    int encoder_layers = 4;
    int decoder_layers = 4;
    int encoder_heads = 6;
    int decoder_heads = 6;
    int encoder_ffn_dim = 1536;
    int decoder_ffn_dim = 1536;
    int vocab_size = 51865;
    int max_source_positions = 1500;
    int max_target_positions = 448;
    int n_mels = 80;
    float sample_rate = 16000.0f;
    int n_fft = 400;
    int hop_length = 160;

    // Special tokens
    int sot_token = 50258;           // <|startoftranscript|>
    int eot_token = 50257;           // <|endoftext|>
    int lang_en_token = 50259;       // <|en|>
    int transcribe_token = 50359;    // <|transcribe|>
    int no_timestamps_token = 50363; // <|notimestamps|>
};

// ============================================================================
// Encoder layer: self-attention + FFN with pre-norm
// ============================================================================

struct WhisperEncoderLayer : Module {
    MultiHeadAttention self_attn_;
    LayerNorm self_attn_layer_norm_;
    Linear fc1_{true};
    Linear fc2_{true};
    LayerNorm final_layer_norm_;

    explicit WhisperEncoderLayer(int num_heads) : self_attn_(num_heads) {
        register_module("self_attn", self_attn_);
        register_module("self_attn_layer_norm", self_attn_layer_norm_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("final_layer_norm", final_layer_norm_);
    }

    Tensor forward(const Tensor &x) const {
        // Pre-norm self attention with residual
        auto residual = x;
        auto h = self_attn_layer_norm_(x);
        h = self_attn_.forward(h, h, h);
        h = ops::add(residual, h);

        // Pre-norm FFN with residual
        residual = h;
        h = final_layer_norm_(h);
        h = fc1_(h).gelu();
        h = fc2_(h);
        return ops::add(residual, h);
    }
};

// Helper: get device from a module's first parameter
inline Device model_device(const Module &mod) {
    auto params = mod.parameters();
    for (auto *p : params) {
        if (p->storage())
            return p->device();
    }
    return Device::CPU;
}

// ============================================================================
// Encoder: conv feature extractor + positional embedding + transformer layers
// ============================================================================

struct WhisperEncoder : Module {
    Conv1d conv1_{1, 1}; // stride=1, padding=1
    Conv1d conv2_{2, 1}; // stride=2, padding=1
    Embedding embed_positions_;
    ModuleList layers_;
    LayerNorm layer_norm_;

    WhisperEncoder(const WhisperConfig &cfg) {
        register_module("conv1", conv1_);
        register_module("conv2", conv2_);
        register_module("embed_positions", embed_positions_);
        register_module("layers", layers_);
        register_module("layer_norm", layer_norm_);

        for (int i = 0; i < cfg.encoder_layers; ++i) {
            layers_.emplace_back<WhisperEncoderLayer>(cfg.encoder_heads);
        }
    }

    Tensor forward(const Tensor &mel) const {
        // mel: (batch, n_mels, n_frames)
        auto x = conv1_(mel).gelu();
        x = conv2_(x).gelu();

        // x: (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose({0, 2, 1});

        // Add sinusoidal positional embeddings
        auto dev = model_device(*this);
        auto seq_len = static_cast<int64_t>(x.shape()[1]);
        auto positions = Tensor::arange(0, seq_len).to(dev);
        x = ops::add(x, embed_positions_(positions));

        for (size_t i = 0; i < layers_.size(); ++i) {
            auto &layer = static_cast<const WhisperEncoderLayer &>(layers_[i]);
            x = layer.forward(x);
        }
        return layer_norm_(x);
    }
};

// ============================================================================
// Decoder layer: self-attention + cross-attention + FFN with pre-norm
// ============================================================================

struct WhisperDecoderLayer : Module {
    using Module::forward;

    MultiHeadAttention self_attn_;
    LayerNorm self_attn_layer_norm_;
    MultiHeadAttention encoder_attn_;
    LayerNorm encoder_attn_layer_norm_;
    Linear fc1_{true};
    Linear fc2_{true};
    LayerNorm final_layer_norm_;

    explicit WhisperDecoderLayer(int num_heads)
        : self_attn_(num_heads), encoder_attn_(num_heads) {
        register_module("self_attn", self_attn_);
        register_module("self_attn_layer_norm", self_attn_layer_norm_);
        register_module("encoder_attn", encoder_attn_);
        register_module("encoder_attn_layer_norm", encoder_attn_layer_norm_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("final_layer_norm", final_layer_norm_);
    }

    Tensor forward(const Tensor &x, const Tensor &encoder_out,
                   const Tensor &causal_mask) const {
        // Self attention with causal mask
        auto residual = x;
        auto h = self_attn_layer_norm_(x);
        h = self_attn_.forward(h, h, h, causal_mask);
        h = ops::add(residual, h);

        // Cross attention to encoder output
        residual = h;
        h = encoder_attn_layer_norm_(h);
        h = encoder_attn_.forward(h, encoder_out, encoder_out);
        h = ops::add(residual, h);

        // FFN
        residual = h;
        h = final_layer_norm_(h);
        h = fc1_(h).gelu();
        h = fc2_(h);
        return ops::add(residual, h);
    }
};

// ============================================================================
// Decoder: token embedding + positional embedding + transformer layers
// ============================================================================

struct WhisperDecoder : Module {
    using Module::forward; // multi-arg forward hides base; re-expose

    Embedding embed_tokens_;
    Embedding embed_positions_;
    ModuleList layers_;
    LayerNorm layer_norm_;

    WhisperDecoder(const WhisperConfig &cfg) {
        register_module("embed_tokens", embed_tokens_);
        register_module("embed_positions", embed_positions_);
        register_module("layers", layers_);
        register_module("layer_norm", layer_norm_);

        for (int i = 0; i < cfg.decoder_layers; ++i) {
            layers_.emplace_back<WhisperDecoderLayer>(cfg.decoder_heads);
        }
    }

    Tensor forward(const Tensor &token_ids, const Tensor &encoder_out) const {
        // token_ids: (batch, seq_len) Int64
        auto dev = model_device(*this);
        auto seq_len = static_cast<int64_t>(token_ids.shape()[1]);
        auto positions = Tensor::arange(0, seq_len).to(dev);

        auto x =
            ops::add(embed_tokens_(token_ids), embed_positions_(positions));

        // Build causal mask on CPU then move (need typed_data access)
        auto causal_mask = Tensor::zeros(
            {1, 1, static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)},
            DType::Bool);
        bool *mask_ptr = causal_mask.typed_data<bool>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = i + 1; j < seq_len; ++j) {
                mask_ptr[i * seq_len + j] = true;
            }
        }
        if (dev != Device::CPU) {
            causal_mask = causal_mask.to(dev);
        }

        for (size_t i = 0; i < layers_.size(); ++i) {
            auto &layer = static_cast<const WhisperDecoderLayer &>(layers_[i]);
            x = layer.forward(x, encoder_out, causal_mask);
        }
        return layer_norm_(x);
    }
};

// ============================================================================
// Full Whisper model
// ============================================================================

struct Whisper : Module {
    WhisperEncoder encoder_;
    WhisperDecoder decoder_;
    Linear proj_out_{false}; // Final projection to vocab logits (no bias)
    WhisperConfig cfg_;

    explicit Whisper(const WhisperConfig &cfg)
        : encoder_(cfg), decoder_(cfg), cfg_(cfg) {
        register_module("model.encoder", encoder_);
        register_module("model.decoder", decoder_);
        register_module("proj_out", proj_out_);
    }

    Tensor encode(const Tensor &mel) const { return encoder_.forward(mel); }

    Tensor decode(const Tensor &token_ids, const Tensor &encoder_out) const {
        auto hidden = decoder_.forward(token_ids, encoder_out);
        return proj_out_(hidden); // (batch, seq, vocab_size)
    }

    // Greedy decode: generate tokens one at a time
    std::vector<int64_t> greedy_decode(const Tensor &mel,
                                       int max_tokens = 100) const {
        auto encoder_out = encode(mel);
        auto dev = model_device(*this);
        std::vector<int64_t> tokens = {
            static_cast<int64_t>(cfg_.sot_token),
            static_cast<int64_t>(cfg_.lang_en_token),
            static_cast<int64_t>(cfg_.transcribe_token),
            static_cast<int64_t>(cfg_.no_timestamps_token),
        };

        for (int step = 0; step < max_tokens; ++step) {
            auto token_tensor =
                Tensor::from_data(tokens.data(), {1, tokens.size()}, true);
            token_tensor = token_tensor.astype(DType::Int64).to(dev);

            auto logits = decode(token_tensor, encoder_out);

            // Take last token's logits: (1, seq, vocab) -> (vocab,)
            auto last_logits =
                logits.slice({Slice(0, 1),
                              Slice(static_cast<int64_t>(tokens.size()) - 1,
                                    static_cast<int64_t>(tokens.size())),
                              Slice()});
            last_logits =
                last_logits.reshape({static_cast<size_t>(cfg_.vocab_size)});

            auto next_token = ops::argmax(last_logits).cpu();
            auto token_id = next_token.ndim() == 0
                                ? next_token.item<int64_t>({})
                                : next_token.item<int64_t>({0});

            if (token_id == cfg_.eot_token) {
                break;
            }
            tokens.push_back(token_id);
        }
        return tokens;
    }
};

// ============================================================================
// Utilities
// ============================================================================

void print_model_info(const Whisper &model) {
    auto params = model.parameters();
    size_t total_params = 0;
    for (auto *p : params) {
        if (p->storage()) {
            total_params += p->size();
        }
    }
    std::cout << "  Parameters: " << params.size() << std::endl;
    std::cout << "  Total values: " << total_params << " ("
              << (total_params * 4 / 1024 / 1024) << " MB float32)"
              << std::endl;
}

void print_section(const std::string &title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    WhisperConfig cfg;
    bool has_weights = false;
    std::string model_path;

    if (argc >= 2) {
        model_path = argv[1];
        has_weights = std::filesystem::exists(model_path);
    }

    // ---- Build model ----
    print_section("Model Construction");
    auto t0 = std::chrono::high_resolution_clock::now();
    Whisper model(cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto build_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "  Built Whisper-tiny in " << build_ms << " ms" << std::endl;

    // ---- Verify structure ----
    print_section("Module Structure");
    auto named = model.named_parameters();
    std::cout << "  Registered parameters: " << named.size() << std::endl;

    // Print first/last few param names to verify hierarchy
    std::cout << "  First 5 keys:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, named.size()); ++i) {
        std::cout << "    " << named[i].first << std::endl;
    }
    std::cout << "  Last 5 keys:" << std::endl;
    for (size_t i = named.size() > 5 ? named.size() - 5 : 0; i < named.size();
         ++i) {
        std::cout << "    " << named[i].first << std::endl;
    }

    // ---- Load weights ----
    if (has_weights) {
        print_section("Weight Loading");
        auto t2 = std::chrono::high_resolution_clock::now();
        auto state_dict = io::safetensors::load(model_path);
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << "  Loaded " << state_dict.size() << " tensors from "
                  << model_path << " in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 -
                                                                           t2)
                         .count()
                  << " ms" << std::endl;

        model.load_state_dict(state_dict, "", false);
        auto t4 = std::chrono::high_resolution_clock::now();
        std::cout << "  load_state_dict() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t4 -
                                                                           t3)
                         .count()
                  << " ms" << std::endl;

        print_model_info(model);

        // Move model to GPU
        print_section("Move to GPU");
        auto t_gpu0 = std::chrono::high_resolution_clock::now();
        model.to(Device::GPU);
        auto t_gpu1 = std::chrono::high_resolution_clock::now();
        std::cout << "  model.to(GPU) took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         t_gpu1 - t_gpu0)
                         .count()
                  << " ms" << std::endl;
    } else {
        print_section("Random Weights (no model file provided)");
        // Build a synthetic state dict with correct shapes
        auto d = static_cast<size_t>(cfg.d_model);
        auto ffn = static_cast<size_t>(cfg.encoder_ffn_dim);
        auto mels = static_cast<size_t>(cfg.n_mels);
        auto vocab = static_cast<size_t>(cfg.vocab_size);
        auto max_src = static_cast<size_t>(cfg.max_source_positions);
        auto max_tgt = static_cast<size_t>(cfg.max_target_positions);

        std::map<std::string, Tensor> state;
        // Encoder convs
        state["model.encoder.conv1.weight"] = Tensor::randn({d, mels, 3});
        state["model.encoder.conv1.bias"] = Tensor::zeros({d});
        state["model.encoder.conv2.weight"] = Tensor::randn({d, d, 3});
        state["model.encoder.conv2.bias"] = Tensor::zeros({d});
        state["model.encoder.embed_positions.weight"] =
            Tensor::randn({max_src, d});
        state["model.encoder.layer_norm.weight"] = Tensor::ones({d});
        state["model.encoder.layer_norm.bias"] = Tensor::zeros({d});

        // Encoder layers
        for (int i = 0; i < cfg.encoder_layers; ++i) {
            auto pfx = "model.encoder.layers." + std::to_string(i) + ".";
            for (auto proj : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
                state[pfx + "self_attn." + proj + ".weight"] =
                    Tensor::randn({d, d}) * 0.02f;
                state[pfx + "self_attn." + proj + ".bias"] = Tensor::zeros({d});
            }
            state[pfx + "self_attn_layer_norm.weight"] = Tensor::ones({d});
            state[pfx + "self_attn_layer_norm.bias"] = Tensor::zeros({d});
            state[pfx + "fc1.weight"] = Tensor::randn({ffn, d}) * 0.02f;
            state[pfx + "fc1.bias"] = Tensor::zeros({ffn});
            state[pfx + "fc2.weight"] = Tensor::randn({d, ffn}) * 0.02f;
            state[pfx + "fc2.bias"] = Tensor::zeros({d});
            state[pfx + "final_layer_norm.weight"] = Tensor::ones({d});
            state[pfx + "final_layer_norm.bias"] = Tensor::zeros({d});
        }

        // Decoder embeddings
        state["model.decoder.embed_tokens.weight"] =
            Tensor::randn({vocab, d}) * 0.02f;
        state["model.decoder.embed_positions.weight"] =
            Tensor::randn({max_tgt, d});
        state["model.decoder.layer_norm.weight"] = Tensor::ones({d});
        state["model.decoder.layer_norm.bias"] = Tensor::zeros({d});

        // Decoder layers
        for (int i = 0; i < cfg.decoder_layers; ++i) {
            auto pfx = "model.decoder.layers." + std::to_string(i) + ".";
            for (auto attn : {"self_attn", "encoder_attn"}) {
                for (auto proj : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
                    state[pfx + attn + "." + proj + ".weight"] =
                        Tensor::randn({d, d}) * 0.02f;
                    state[pfx + attn + "." + proj + ".bias"] =
                        Tensor::zeros({d});
                }
            }
            state[pfx + "self_attn_layer_norm.weight"] = Tensor::ones({d});
            state[pfx + "self_attn_layer_norm.bias"] = Tensor::zeros({d});
            state[pfx + "encoder_attn_layer_norm.weight"] = Tensor::ones({d});
            state[pfx + "encoder_attn_layer_norm.bias"] = Tensor::zeros({d});
            state[pfx + "fc1.weight"] = Tensor::randn({ffn, d}) * 0.02f;
            state[pfx + "fc1.bias"] = Tensor::zeros({ffn});
            state[pfx + "fc2.weight"] = Tensor::randn({d, ffn}) * 0.02f;
            state[pfx + "fc2.bias"] = Tensor::zeros({d});
            state[pfx + "final_layer_norm.weight"] = Tensor::ones({d});
            state[pfx + "final_layer_norm.bias"] = Tensor::zeros({d});
        }

        // Output projection
        state["proj_out.weight"] = Tensor::randn({vocab, d}) * 0.02f;

        model.load_state_dict(state, "", false);
        print_model_info(model);
        std::cout << "  (provide .safetensors path for real inference)"
                  << std::endl;

        // Move model to GPU
        print_section("Move to GPU");
        auto t_gpu0 = std::chrono::high_resolution_clock::now();
        model.to(Device::GPU);
        auto t_gpu1 = std::chrono::high_resolution_clock::now();
        std::cout << "  model.to(GPU) took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         t_gpu1 - t_gpu0)
                         .count()
                  << " ms" << std::endl;
    }

    // ---- Encoder forward pass ----
    print_section("Encoder Forward Pass");
    {
        // Simulate 30s audio: mel spectrogram (1, 80, 3000)
        auto mel = Tensor::randn({1, static_cast<size_t>(cfg.n_mels), 3000},
                                 DType::Float32, Device::GPU);
        std::cout << "  Input mel: (1, " << cfg.n_mels << ", 3000)"
                  << std::endl;

        auto t5 = std::chrono::high_resolution_clock::now();
        auto encoder_out = model.encode(mel);
        auto t6 = std::chrono::high_resolution_clock::now();

        std::cout << "  Encoder output: (" << encoder_out.shape()[0] << ", "
                  << encoder_out.shape()[1] << ", " << encoder_out.shape()[2]
                  << ") on "
                  << (encoder_out.device() == Device::GPU ? "GPU" : "CPU")
                  << std::endl;
        std::cout << "  Encoder time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t6 -
                                                                           t5)
                         .count()
                  << " ms" << std::endl;
    }

    // ---- Decoder forward pass ----
    print_section("Decoder Forward Pass");
    {
        // Encode a short mel
        auto mel = Tensor::randn({1, static_cast<size_t>(cfg.n_mels), 3000},
                                 DType::Float32, Device::GPU);
        auto encoder_out = model.encode(mel);

        // Decoder prompt tokens
        std::vector<int64_t> prompt = {
            static_cast<int64_t>(cfg.sot_token),
            static_cast<int64_t>(cfg.lang_en_token),
            static_cast<int64_t>(cfg.transcribe_token),
            static_cast<int64_t>(cfg.no_timestamps_token),
        };
        auto tokens = Tensor::from_data(prompt.data(), {1, prompt.size()}, true)
                          .astype(DType::Int64)
                          .to(Device::GPU);
        std::cout << "  Prompt tokens: " << prompt.size() << std::endl;

        auto t7 = std::chrono::high_resolution_clock::now();
        auto logits = model.decode(tokens, encoder_out);
        auto t8 = std::chrono::high_resolution_clock::now();

        std::cout << "  Logits shape: (" << logits.shape()[0] << ", "
                  << logits.shape()[1] << ", " << logits.shape()[2] << ") on "
                  << (logits.device() == Device::GPU ? "GPU" : "CPU")
                  << std::endl;
        std::cout << "  Decoder time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t8 -
                                                                           t7)
                         .count()
                  << " ms" << std::endl;

        // Argmax on last position
        auto last_logits =
            logits.slice({Slice(0, 1),
                          Slice(static_cast<int64_t>(prompt.size()) - 1,
                                static_cast<int64_t>(prompt.size())),
                          Slice()});
        last_logits =
            last_logits.reshape({static_cast<size_t>(cfg.vocab_size)});
        auto predicted = ops::argmax(last_logits).cpu();
        auto token_id = predicted.ndim() == 0 ? predicted.item<int64_t>({})
                                              : predicted.item<int64_t>({0});
        std::cout << "  Predicted next token ID: " << token_id << std::endl;
    }

    // ---- Greedy decode (only with real weights) ----
    if (has_weights) {
        print_section("Greedy Decoding (10 tokens max)");
        auto mel = Tensor::randn({1, static_cast<size_t>(cfg.n_mels), 3000},
                                 DType::Float32, Device::GPU);

        auto t9 = std::chrono::high_resolution_clock::now();
        auto tokens = model.greedy_decode(mel, 10);
        auto t10 = std::chrono::high_resolution_clock::now();

        std::cout << "  Generated " << tokens.size() << " tokens in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t10 -
                                                                           t9)
                         .count()
                  << " ms" << std::endl;
        std::cout << "  Token IDs:";
        for (auto tok : tokens) {
            std::cout << " " << tok;
        }
        std::cout << std::endl;
    }

    // ---- Audio pipeline (mel spectrogram) ----
    print_section("Audio Pipeline");
    {
        // Simulate 1 second of audio at 16kHz
        auto waveform = Tensor::randn({1, 16000});
        std::cout << "  Waveform: (1, 16000) @ 16kHz" << std::endl;

        auto t11 = std::chrono::high_resolution_clock::now();
        auto mel = audio::mel_spectrogram(waveform, cfg.sample_rate, cfg.n_fft,
                                          cfg.hop_length, cfg.n_mels);
        auto t12 = std::chrono::high_resolution_clock::now();

        std::cout << "  Mel spectrogram: (";
        for (size_t i = 0; i < mel.ndim(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << mel.shape()[i];
        }
        std::cout << ")" << std::endl;
        std::cout << "  Mel computation: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t12 -
                                                                           t11)
                         .count()
                  << " ms" << std::endl;
    }

    print_section("POC Complete");
    std::cout << "  Axiom nn module system is inference-ready." << std::endl;

    return 0;
}
