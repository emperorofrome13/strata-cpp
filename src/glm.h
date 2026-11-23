// port of glm/mod.rs (GLM-5.3-Flash hyperparameters, tensor-table validation, gpu_defaults) and glm/forward.rs (the
// CPU reference forward with GPU dense matvecs and a VRAM expert cache, chunked layer-major prefill, the k-pool DSA
// indexer). GGUF architecture `glm5-next` (RustingStrata's spelling) or `glm5next` (llama.cpp's, as the unsloth
// GGUFs carry it).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md). The model structure mirrors llama.cpp (MIT License, ggml authors)
// src/models/glm5-next.cpp, as the Rust module does.
//
// Not ported from glm/mod.rs: `generate`, `chat`, `serve`, `chat_prompt*`, `parse_calls`, `decode_greedy`, `stop_ids`
// (tokenization, chat templates, tool calls and the HTTP server stay in the host application, as for every other
// architecture of this DLL). The NextN/MTP block is loaded by neither the Rust code nor this port.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ds_mat.h"
#include "ds_ops.h"
#include "gguf.h"

namespace strata {

// port of glm/mod.rs: LayerKind. Token mixer of a layer: Kimi Delta Attention (linear) or DSA (MLA with a k-pool indexer).
enum class GlmKind : uint8_t { Kda, Dsa };

// Whether `arch` (general.architecture) is GLM-5.3-Flash: RustingStrata's `glm5-next` or llama.cpp's `glm5next`.
bool is_glm_arch(const std::string& arch);

// port of glm/mod.rs: GlmConfig
struct GlmConfig {
    std::string arch;          // the GGUF's spelling; its metadata keys are `<arch>.<key>`
    size_t n_layer_file = 0;   // all blocks in the file, including the trailing NextN (MTP) block
    size_t n_layer = 0;        // blocks used for normal decoding (file blocks minus NextN blocks)
    size_t n_embd = 0;
    size_t n_head = 0;
    size_t n_vocab = 0;
    size_t n_ff = 0;
    size_t n_dense = 0;
    float eps = 0;
    float kda_eps = 0;
    std::vector<GlmKind> kind;
    size_t conv_k = 0;
    size_t kda_head_dim = 0;
    float gate_lower_bound = 0;
    size_t q_lora = 0;
    size_t kv_lora = 0;
    size_t k_mla = 0;          // MLA per-head widths before absorption (k = v = 256)
    size_t v_mla = 0;
    size_t idx_n_head = 0;
    size_t idx_head_dim = 0;
    size_t idx_top_k = 0;
    size_t n_ctx_train = 0;    // trained context (`<arch>.context_length`), the default request limit
    size_t kpool = 0;
    size_t hc_n = 0;
    float hc_eps = 0;
    size_t hc_iters = 0;
    size_t n_expert = 0;
    size_t n_used = 0;
    size_t n_ff_exp = 0;
    size_t n_ff_shexp = 0;
    size_t n_shared = 0;
    bool weights_norm = false;
    float weights_scale = 0;
    std::vector<float> clamp_exp;
    std::vector<float> clamp_shexp;
    size_t n_nextn = 0;

    // port of glm/mod.rs: GlmConfig::from_gguf
    static bool from_gguf(const Gguf& g, GlmConfig& out, std::string& err);
    // port of glm/mod.rs: GlmConfig::n_kda / n_dsa
    size_t n_kda() const;
    size_t n_dsa() const { return n_layer - n_kda(); }
    // port of glm/mod.rs: GlmConfig::expected_tensors (GGUF dims order: fastest first)
    std::vector<std::pair<std::string, std::vector<uint64_t>>> expected_tensors() const;
    // port of glm/mod.rs: GlmConfig::validate. Every expected tensor exists with the expected dims and a supported
    // quant type, and nothing else is in the file. `err` lists up to 40 problems.
    bool validate(const Gguf& g, std::string& err) const;
};

// port of glm/mod.rs: gpu_defaults, plus the STRATA_DS_* reads of deepseek/mat.rs dev::get. Rust sets the GLM defaults
// as environment variables (STRATA_DS_GPU=1, STRATA_DS_GPU_FRACTION=0.9, STRATA_GLM_PREFETCH=2) unless STRATA_GLM_CPU
// is set; a DLL must not change its host's environment, so the same defaults are applied to the values read here.
// `device` is the default for STRATA_DS_DEVICES (main.rs hands `--device` to that variable).
DsGpuConfig glm_gpu_config(int device);
// STRATA_GLM_PREFETCH with the gpu_defaults value (2, or the forward's own default 1 under STRATA_GLM_CPU).
size_t glm_prefetch_default();
// Rust `std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok())`.
std::optional<size_t> env_usize(const char* k);

// What sc_load passes to the GLM model.
struct GlmLoad {
    DsGpuConfig gpu;
    BackendOpen open_backend;  // capi.cpp: CudaBackend::create; empty in the CPU test DLL
    size_t threads = 0;        // 0: default_threads()
    size_t ctx = 0;            // 0: max_ctx (STRATA_GLM_CTX or the trained context)
};

// port of glm/forward.rs: GlmModel
class GlmModel {
public:
    GlmConfig cfg;
    std::vector<uint32_t> toks;
    bool trace = false;

    ~GlmModel();

    // port of glm/forward.rs: GlmModel::new
    static bool create(const Gguf& g, const GlmLoad& opt, std::unique_ptr<GlmModel>& out, std::string& err);
    // port of glm/forward.rs: GlmModel::reset (forget the whole sequence)
    void reset();
    // port of glm/forward.rs: GlmModel::step (feed one token at the next position; logits out)
    bool step(uint32_t tok, std::vector<float>& logits, std::string& err);
    // port of glm/forward.rs: GlmModel::prefill (feed prompt tokens; only the last one computes logits)
    bool prefill(const std::vector<uint32_t>& ids, std::vector<float>& logits, std::string& err);

    // The context limit of this load (port of glm/forward.rs: max_ctx, or sc_load's ctx).
    size_t ctx_limit() const { return ctx_; }
    // Device side, for sc_stats (backend "cpu" when the GPU path is off).
    std::string backend_name() const;
    std::string device_name() const;
    std::string gpu_note() const { return gpu_note_; }
    size_t threads() const;
    size_t resident_experts() const;
    size_t gpu_bytes() const;
    uint64_t hits() const;
    uint64_t misses() const;
    // Total routed-expert tensors (3 per expert of every MoE layer), as main.rs big_expert_stats reports them.
    size_t expert_tensors_total() const { return 3 * (cfg.n_layer - cfg.n_dense) * cfg.n_expert; }

    struct Impl;

private:
    GlmModel() = default;
    std::unique_ptr<Impl> m_;
    size_t ctx_ = 0;
    std::string gpu_note_;
    bool broken_ = false;
};

// port of glm/forward.rs: max_ctx (STRATA_GLM_CTX, else the trained context)
size_t glm_max_ctx(const GlmConfig& cfg);

// port of glm/forward.rs: route_sigmoid. probs = sigmoid(logit); pick top n_used by probs+bias (ties to the lower
// index); weights = probs[sel], normalised by their sum (clamped like ggml), then scaled.
std::vector<std::pair<size_t, float>> route_sigmoid(const float* logits, const float* bias, size_t n, size_t n_used,
                                                    bool norm, float scale);

// port of glm/forward.rs: l2_norm. x / max(||x||, eps), in place (ggml l2_norm).
void glm_l2_norm(float* x, size_t n, float eps);

}  // namespace strata
