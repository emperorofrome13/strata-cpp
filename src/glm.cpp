// port of glm/mod.rs: GlmConfig (from_gguf, n_kda, expected_tensors, validate) and gpu_defaults. See glm.h.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "glm.h"

#include <cstdlib>
#include <set>

namespace strata {

namespace {

// Rust `std::env::var(k).ok().and_then(|v| v.parse().ok())` for f64 / usize.
std::optional<double> env_f64(const char* k) {
    const char* v = std::getenv(k);
    if (!v || !*v) return std::nullopt;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    if (end == v || *end) return std::nullopt;
    return d;
}

std::string dims_text(const std::vector<uint64_t>& d) {
    std::string s = "[";
    for (size_t i = 0; i < d.size(); i++) s += (i ? ", " : "") + std::to_string(d[i]);
    return s + "]";
}

// Per-expert and 2-D conv tensors may carry trailing 1s; compare ignoring them.
std::vector<uint64_t> trim(std::vector<uint64_t> d) {
    while (d.size() > 1 && d.back() == 1) d.pop_back();
    return d;
}

}  // namespace

std::optional<size_t> env_usize(const char* k) {
    const char* v = std::getenv(k);
    if (!v || !*v) return std::nullopt;
    char* end = nullptr;
    const unsigned long long u = std::strtoull(v, &end, 10);
    if (end == v || *end || *v == '-') return std::nullopt;
    return (size_t)u;
}

bool is_glm_arch(const std::string& arch) { return arch == "glm5-next" || arch == "glm5next"; }

// port of glm/mod.rs: GlmConfig::from_gguf. StrataFreet addition: the `glm5next` spelling of llama.cpp and the
// unsloth GGUFs is accepted next to RustingStrata's `glm5-next`; the metadata keys use whichever the file carries.
bool GlmConfig::from_gguf(const Gguf& g, GlmConfig& c, std::string& err) {
    const Value* av = g.get("general.architecture");
    const std::string* as = av ? av->as_str() : nullptr;
    const std::string arch = as ? *as : std::string();
    if (!is_glm_arch(arch)) {
        err = "architecture '" + arch + "' is not glm5-next";
        return false;
    }
    c = GlmConfig();
    c.arch = arch;
    auto key = [&](const char* s) { return arch + "." + s; };
    auto u = [&](const char* s, size_t& out) -> bool {
        const Value* v = g.get(key(s));
        if (!v) {
            err = "missing metadata key " + key(s);
            return false;
        }
        const auto x = v->as_u64();
        if (!x) {
            err = key(s);
            return false;
        }
        out = (size_t)*x;
        return true;
    };
    auto f = [&](const char* s, float& out) -> bool {
        const Value* v = g.get(key(s));
        if (!v) {
            err = "missing metadata key " + key(s);
            return false;
        }
        const auto x = v->as_f32();
        if (!x) {
            err = key(s);
            return false;
        }
        out = *x;
        return true;
    };
    auto arr_f = [&](const char* s, std::vector<float>& out) -> bool {
        const Value* v = g.get(key(s));
        if (!v) {
            err = "missing metadata key " + key(s);
            return false;
        }
        const std::vector<Value>* a = v->as_arr();
        if (!a) {
            err = key(s);
            return false;
        }
        out.clear();
        for (const Value& x : *a) {
            const auto y = x.as_f32();
            if (!y) {
                err = key(s);
                return false;
            }
            out.push_back(*y);
        }
        return true;
    };
    if (!u("block_count", c.n_layer_file) || !u("nextn_predict_layers", c.n_nextn)) return false;
    if (c.n_nextn > c.n_layer_file) {
        err = "nextn_predict_layers > block_count";
        return false;
    }
    const Value* kvh = g.get(key("attention.head_count_kv"));
    const std::vector<Value>* kva = kvh ? kvh->as_arr() : nullptr;
    if (!kvh) {
        err = "missing metadata key " + key("attention.head_count_kv");
        return false;
    }
    if (!kva) {
        err = "head_count_kv";
        return false;
    }
    std::vector<size_t> kv_heads;
    for (const Value& x : *kva) {
        const auto y = x.as_u64();
        if (!y) {
            err = "head_count_kv entry";
            return false;
        }
        kv_heads.push_back((size_t)*y);
    }
    if (kv_heads.size() < c.n_layer_file) {
        err = "head_count_kv has " + std::to_string(kv_heads.size()) + " entries for " + std::to_string(c.n_layer_file) +
              " layers";
        return false;
    }
    for (size_t i = 0; i < c.n_layer_file; i++) c.kind.push_back(kv_heads[i] == 0 ? GlmKind::Kda : GlmKind::Dsa);
    const auto emb = g.tensors().find("token_embd.weight");
    if (emb == g.tensors().end()) {
        err = "missing token_embd.weight";
        return false;
    }
    if (emb->second.dims.size() < 2) {
        err = "token_embd.weight is not 2-D";
        return false;
    }
    c.n_layer = c.n_layer_file - c.n_nextn;
    c.n_vocab = (size_t)emb->second.dims[1];
    size_t gating = 0;
    if (!u("embedding_length", c.n_embd) || !u("attention.head_count", c.n_head) || !u("feed_forward_length", c.n_ff) ||
        !u("leading_dense_block_count", c.n_dense) || !f("attention.layer_norm_rms_epsilon", c.eps) ||
        !f("attention.layer_norm_epsilon", c.kda_eps) || !u("ssm.conv_kernel", c.conv_k) ||
        !u("kda.head_dim", c.kda_head_dim) || !f("kda.gate_lower_bound", c.gate_lower_bound) ||
        !u("attention.q_lora_rank", c.q_lora) || !u("attention.kv_lora_rank", c.kv_lora) ||
        !u("attention.key_length_mla", c.k_mla) || !u("attention.value_length_mla", c.v_mla) ||
        !u("attention.indexer.head_count", c.idx_n_head) || !u("attention.indexer.key_length", c.idx_head_dim) ||
        !u("attention.indexer.top_k", c.idx_top_k) || !u("context_length", c.n_ctx_train) ||
        !u("attention.indexer.kpool", c.kpool) || !u("hyper_connection.count", c.hc_n) ||
        !f("hyper_connection.epsilon", c.hc_eps) || !u("hyper_connection.sinkhorn_iterations", c.hc_iters) ||
        !u("expert_count", c.n_expert) || !u("expert_used_count", c.n_used) ||
        !u("expert_feed_forward_length", c.n_ff_exp) || !u("expert_shared_feed_forward_length", c.n_ff_shexp) ||
        !u("expert_shared_count", c.n_shared))
        return false;
    const Value* wn = g.get(key("expert_weights_norm"));
    if (!wn) {
        err = "missing metadata key " + key("expert_weights_norm");
        return false;
    }
    c.weights_norm = wn->as_u64().value_or(0) != 0;
    if (!f("expert_weights_scale", c.weights_scale) || !arr_f("swiglu_clamp_exp", c.clamp_exp) ||
        !arr_f("swiglu_clamp_shexp", c.clamp_shexp) || !u("expert_gating_func", gating))
        return false;
    if (gating != 2) {
        err = "only sigmoid expert gating (2) is supported";
        return false;
    }
    if (c.hc_n != 4) {
        err = "only 4 hyper-connection streams are supported";
        return false;
    }
    if (!(c.n_shared == 1 && c.n_ff_shexp == c.n_ff_exp)) {
        err = "unexpected shared-expert shape";
        return false;
    }
    if (!(c.n_head * c.kda_head_dim == 8192 && c.kda_head_dim == 128)) {
        err = "unexpected KDA dimensions";
        return false;
    }
    if (!(c.clamp_exp.size() >= c.n_layer_file && c.clamp_shexp.size() >= c.n_layer_file)) {
        err = "swiglu clamp arrays too short";
        return false;
    }
    if (c.n_dense > c.n_layer) {
        err = "leading_dense_block_count out of range";
        return false;
    }
    // StrataFreet additions: values the Rust code divides by or indexes with, refused here instead of a panic.
    if (c.conv_k < 2 || c.kpool == 0 || c.n_used == 0 || c.n_used > c.n_expert || c.n_head == 0 || c.n_embd == 0) {
        err = "glm5-next: conv_kernel, kpool, expert counts or head counts out of range";
        return false;
    }
    return true;
}

size_t GlmConfig::n_kda() const {
    size_t n = 0;
    for (size_t i = 0; i < n_layer; i++) n += kind[i] == GlmKind::Kda;
    return n;
}

// port of glm/mod.rs: GlmConfig::expected_tensors
std::vector<std::pair<std::string, std::vector<uint64_t>>> GlmConfig::expected_tensors() const {
    const uint64_t e = n_embd, h = hc_n, nh = n_head;
    const uint64_t ff = n_ff, ffe = n_ff_exp, ne = n_expert;
    const uint64_t ql = q_lora, kl = kv_lora, kd = k_mla;
    const uint64_t vd = v_mla, di = nh * kda_head_dim;
    const uint64_t ni = idx_head_dim, nih = idx_n_head, hd = kda_head_dim;
    const uint64_t mix = (2 + h) * h;
    std::vector<std::pair<std::string, std::vector<uint64_t>>> v = {
        {"token_embd.weight", {e, (uint64_t)n_vocab}},
        {"output_norm.weight", {e}},
        {"output.weight", {e, (uint64_t)n_vocab}},
    };
    for (size_t il = 0; il < n_layer_file; il++) {
        const std::string b = "blk." + std::to_string(il) + ".";
        auto t = [&](const std::string& n, std::vector<uint64_t> d) { v.emplace_back(b + n, std::move(d)); };
        t("attn_norm.weight", {e});
        t("ffn_norm.weight", {e});
        if (il < n_layer) {  // the NextN block has no hyper-connections
            for (const char* s : {"attn", "ffn"}) {
                t(std::string("hc_") + s + "_fn.weight", {h * e, mix});
                t(std::string("hc_") + s + "_base.weight", {mix});
                t(std::string("hc_") + s + "_scale.weight", {3});
            }
        }
        if (kind[il] == GlmKind::Kda) {
            t("attn_q.weight", {e, di});
            t("attn_k.weight", {e, di});
            t("attn_v.weight", {e, di});
            t("attn_output.weight", {di, e});
            t("ssm_a", {nh});
            t("ssm_beta.weight", {e, nh});
            for (const char* s : {"q", "k", "v"}) t(std::string("ssm_conv1d_") + s + ".weight", {(uint64_t)conv_k, 1, di});
            t("ssm_dt.bias", {di});
            t("ssm_f_a.weight", {e, hd});
            t("ssm_f_b.weight", {hd, di});
            t("ssm_g_a.weight", {e, hd});
            t("ssm_g_b.weight", {hd, di});
            t("ssm_norm.weight", {hd});
        } else {
            t("attn_q_a.weight", {e, ql});
            t("attn_q_a_norm.weight", {ql});
            t("attn_q_b.weight", {ql, nh * kd});
            t("attn_kv_a_mqa.weight", {e, kl});
            t("attn_kv_a_norm.weight", {kl});
            t("attn_k_b.weight", {kd, kl, nh});
            t("attn_v_b.weight", {kl, vd, nh});
            t("attn_output.weight", {nh * vd, e});
            t("indexer.attn_k.weight", {e, ni});
            t("indexer.attn_q_b.weight", {ql, nih * ni});
            t("indexer.k_norm.weight", {ni});
            t("indexer.k_norm.bias", {ni});
            t("indexer.proj.weight", {e, nih});
            t("indexer_compressor_ape.weight", {ni, (uint64_t)kpool});
            t("indexer_compressor_gate.weight", {e, ni});
        }
        if (il < n_dense) {
            t("ffn_gate.weight", {e, ff});
            t("ffn_up.weight", {e, ff});
            t("ffn_down.weight", {ff, e});
        } else {
            t("ffn_gate_inp.weight", {e, ne});
            t("exp_probs_b.bias", {ne});
            t("ffn_gate_exps.weight", {e, ffe, ne});
            t("ffn_up_exps.weight", {e, ffe, ne});
            t("ffn_down_exps.weight", {ffe, e, ne});
            const uint64_t fs = ffe * n_shared;
            t("ffn_gate_shexp.weight", {e, fs});
            t("ffn_up_shexp.weight", {e, fs});
            t("ffn_down_shexp.weight", {fs, e});
        }
        if (il >= n_layer) {
            t("nextn.eh_proj.weight", {2 * e, e});
            t("nextn.enorm.weight", {e});
            t("nextn.hnorm.weight", {e});
            t("nextn.shared_head_norm.weight", {e});
        }
    }
    return v;
}

// port of glm/mod.rs: GlmConfig::validate
bool GlmConfig::validate(const Gguf& g, std::string& err) const {
    std::vector<std::string> errs;
    const auto exp = expected_tensors();
    for (const auto& [name, dims] : exp) {
        const auto it = g.tensors().find(name);
        if (it == g.tensors().end()) {
            errs.push_back("missing tensor " + name);
            continue;
        }
        const TensorInfo& t = it->second;
        if (trim(t.dims) != trim(dims))
            errs.push_back(name + ": dims " + dims_text(t.dims) + ", expected " + dims_text(dims));
        if (!ty_from_u32(t.ty)) errs.push_back(name + ": unsupported GGML type " + std::to_string(t.ty));
    }
    if (g.tensors().size() != exp.size()) {
        std::set<std::string> known;
        for (const auto& p : exp) known.insert(p.first);
        for (const auto& [n, t] : g.tensors())
            if (!known.count(n)) errs.push_back("unexpected tensor " + n);
    }
    if (errs.empty()) return true;
    err = std::to_string(errs.size()) + " tensor problems:";
    for (size_t i = 0; i < errs.size() && i < 40; i++) err += "\n" + errs[i];
    return false;
}

// port of glm/mod.rs: gpu_defaults + the reads of deepseek/mat.rs dev::get
DsGpuConfig glm_gpu_config(int device) {
    const bool glm_cpu = std::getenv("STRATA_GLM_CPU") != nullptr;
    DsGpuConfig c;
    c.on = !glm_cpu || std::getenv("STRATA_DS_GPU") != nullptr;
    if (const char* b = std::getenv("STRATA_DS_BACKEND")) c.backend = b;
    c.devices = {device};
    if (const char* d = std::getenv("STRATA_DS_DEVICES")) {
        std::vector<int> ids;
        std::string s = d, tok;
        for (size_t i = 0; i <= s.size(); i++) {
            if (i == s.size() || s[i] == ',') {
                size_t a = tok.find_first_not_of(" \t"), z = tok.find_last_not_of(" \t");
                if (a != std::string::npos) {
                    const std::string t = tok.substr(a, z - a + 1);
                    char* end = nullptr;
                    const long v = std::strtol(t.c_str(), &end, 10);
                    if (end != t.c_str() && !*end && v >= 0) ids.push_back((int)v);
                }
                tok.clear();
            } else {
                tok += s[i];
            }
        }
        if (!ids.empty()) c.devices = ids;
    }
    c.gb = env_f64("STRATA_DS_GPU_EXPERTS");
    // GLM keeps its tuned 90% share of free VRAM (dev::get's own default is 80%).
    c.frac = env_f64("STRATA_DS_GPU_FRACTION").value_or(glm_cpu ? 0.8 : 0.9);
    return c;
}

size_t glm_prefetch_default() {
    if (const auto v = env_usize("STRATA_GLM_PREFETCH")) return *v;
    return std::getenv("STRATA_GLM_CPU") ? 1 : 2;
}

}  // namespace strata
