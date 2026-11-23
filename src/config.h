// port of config.rs (model hyperparameters read from GGUF metadata, plus the per-architecture table).
//
// Deviations forced by C++ (all listed in MODIFICATIONS.md): `&'static [(&str, &str)]` alias tables
// become std::vector<std::pair<std::string, std::string>>, and `Option<[usize; 4]>` becomes
// std::optional<std::array<size_t, 4>>.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gguf.h"

namespace strata {

// port of config.rs: LayerKind
enum class LayerKind { Delta, Attn };

// port of config.rs: Eagle3Config (standalone EAGLE3 1-layer draft head parameters)
struct Eagle3Config {
    std::array<size_t, 3> extract_layers{};
    size_t n_ff = 0;
    size_t n_head = 0;
    size_t n_head_kv = 0;
    size_t head_dim = 0;
    size_t n_rot = 0;
    float rope_base = 0.0f;
    float eps = 0.0f;
    size_t draft_vocab = 0;
    bool norm_before_residual = false;
    bool rope_neox = false;
    std::vector<int32_t> d2t;
};

// port of config.rs: Arch - what differs between the supported architectures.
struct Arch {
    // DeltaNet layers with full attention every `full_attention_interval` layers; otherwise all attention.
    bool hybrid = false;
    // `attn_q` is [query | sigmoid output gate] per head (2 * head_dim wide).
    bool q_gate = false;
    // DeltaNet value head h reads key head h / (nv / nk) (llama.cpp qwen3next), not h % nk.
    bool k_interleave = false;
    // Canonical tensor suffix to this architecture's suffix.
    std::vector<std::pair<std::string, std::string>> aliases;
    // StrataFreet addition (INKLING2_NOTES.md): the architecture family the forward dispatches on.
    // Inkling2 is not a qwen variant: its loop, shortconv, rel-bias attention and NVFP4 experts have
    // no qwen3next/qwen4exp counterpart.
    enum class Kind { Qwen, Flint, Inkling2 };
    Kind kind = Kind::Qwen;
};

// port of config.rs: arch(name). Returns nullptr for an unsupported architecture.
const Arch* arch(const std::string& name);

// port of config.rs: ModelConfig
struct ModelConfig {
    std::string arch;  // Rust: `pub arch: String`
    size_t n_layer = 0;
    size_t n_nextn = 0;
    size_t n_embd = 0;
    size_t n_head = 0;
    size_t n_head_kv = 0;
    size_t head_dim = 0;
    size_t n_rot = 0;
    // Qwen3-VL/3.5 M-RoPE section sizes (sum = n_rot/2); nullopt uses 1-D RoPE.
    std::optional<std::array<size_t, 4>> mrope_sections;
    float rope_base = 0.0f;
    float eps = 0.0f;
    size_t n_vocab = 0;
    // Dense FFN size (0 for MoE).
    size_t n_ff = 0;
    size_t n_expert = 0;
    size_t n_expert_used = 0;
    size_t n_ff_exp = 0;
    size_t n_ff_shexp = 0;
    // Leading layers with a dense FFN; the rest are MoE (0 when the whole model is MoE).
    size_t n_dense = 0;
    // 0 = no extra scale.
    float expert_weights_scale = 0.0f;
    // Router activation is sigmoid (Flint) instead of softmax.
    bool router_sigmoid = false;
    // Sliding-window width for the layers flagged in `swa` (0 = all layers are dense).
    size_t n_swa = 0;
    // Per-layer sliding-window attention (Flint). Empty for architectures without it.
    std::vector<bool> swa;
    // Per-layer RoPE; Flint's full-attention layers use no positional encoding.
    std::vector<bool> rope_layer;
    // Per-head learned attention sinks (`blk.N.attn_sinks.weight`).
    bool attn_sinks = false;
    // DeltaNet sizes, 0 without DeltaNet layers.
    size_t ssm_d_inner = 0;
    size_t ssm_d_state = 0;
    size_t ssm_n_group = 0;
    size_t ssm_dt_rank = 0;
    size_t ssm_conv = 0;
    // Hyper-connection residual streams and their low rank (qwen4exp; 0 = plain residual).
    size_t hc_n = 0;
    size_t hc_rank = 0;
    // Per-layer (n-gram) embedding width (qwen4exp; 0 = none).
    size_t n_embd_ple = 0;
    // QSA sparse attention indexer (qwen4exp): heads, head size, and the number of cells kept per row.
    size_t idx_n_head = 0;
    size_t idx_head_dim = 0;
    size_t idx_top_k = 0;
    // StrataFreet addition (INKLING2_NOTES.md): the looped-core geometry of inkling2. `n_layer` stays
    // the slot count (block_count in the GGUF); the physical blocks are n_layer - loop_n_extra().
    // Slots 0..n_layer-1 share the weights of loop_layer_of(c, il); each slot keeps its own KV and
    // shortconv state. 0 n_loop_passes (or 1) = no loop, and every field below is unused.
    size_t n_loop_passes = 0;
    size_t loop_start = 0;
    size_t loop_end = 0;
    size_t loop_gate_rank = 0;
    // Physical blocks holding weights: n_layer when there is no loop.
    size_t n_layer_phys = 0;
    // Inkling2 attention/normalization scalars.
    float situ_beta = 0.0f;
    float situ_linear_beta = 0.0f;
    size_t n_rel = 0;
    size_t n_rel_extent = 0;
    float f_rel_clip = 0.0f;
    float f_qk_clip = 0.0f;
    size_t n_log_scale_floor = 0;
    float f_log_scale_alpha = 0.0f;
    size_t n_shortconv = 0;
    float f_final_softcap = 0.0f;
    size_t n_attn_res_block = 0;
    // Padded/unpadded vocabulary: n_vocab is the padded row count of token_embd, the unpadding is the
    // router-free tail past the real tokens (logits are masked to n_vocab_unpadded).
    size_t n_vocab_padded = 0;
    size_t n_vocab_unpadded = 0;
    // Mixer of each main layer (MTP layers are attention).
    std::vector<LayerKind> layers;
    bool q_gate = false;
    bool k_interleave = false;
    std::vector<std::pair<std::string, std::string>> aliases;
    // StrataFreet addition (INKLING2_NOTES.md): the architecture family the forward dispatches on.
    Arch::Kind kind = Arch::Kind::Qwen;
    bool has_eagle3 = false;
    Eagle3Config eagle3;

    // port of config.rs: ModelConfig::from_gguf
    bool from_gguf(const Gguf& g, std::string& err);

    // port of config.rs: ModelConfig::is_moe
    bool is_moe() const { return n_expert > 0; }
    // port of config.rs: ModelConfig::is_moe_layer
    bool is_moe_layer(size_t il) const { return n_expert > 0 && il >= n_dense; }
    // port of config.rs: ModelConfig::ssm_dv
    size_t ssm_dv() const { return ssm_d_inner / (ssm_dt_rank ? ssm_dt_rank : 1); }
    // StrataFreet addition (INKLING2_NOTES.md): the model repeats its loop core (n_loop_passes > 1).
    bool is_looped() const { return n_loop_passes > 1; }
};

// StrataFreet addition (INKLING2_NOTES.md): the cache slot `il` plays (0..n_layer-1) mapped to the
// physical block whose weights it shares, for the looped architectures (n_loop_passes > 1). The
// formula is llama.cpp PR #389's hparams::loop_layer: slots below loop_start are themselves; the
// loop passes repeat the core; the post-loop blocks follow. Identity when the model has no loop.
size_t loop_layer_of(const ModelConfig& c, size_t il);

}  // namespace strata
