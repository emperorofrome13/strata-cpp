// port of config.rs. Function and field names match the Rust ones.
#include "config.h"

#include <cmath>
#include <cstring>

namespace strata {
namespace {

// port of config.rs: const FFN_NORM
const std::pair<const char*, const char*> FFN_NORM[] = {{"post_attention_norm.weight", "ffn_norm.weight"}};

Arch make_arch(bool hybrid, bool q_gate, bool k_interleave, const std::pair<const char*, const char*>* al,
               size_t n_al) {
    Arch a;
    a.hybrid = hybrid;
    a.q_gate = q_gate;
    a.k_interleave = k_interleave;
    for (size_t i = 0; i < n_al; i++) a.aliases.emplace_back(al[i].first, al[i].second);
    return a;
}

}  // namespace

// port of config.rs: arch(name)
const Arch* arch(const std::string& name) {
    static const Arch a_qwen35 = make_arch(true, true, false, nullptr, 0);
    static const Arch a_qwen4exp = make_arch(true, true, false, nullptr, 0);
    static const Arch a_qwen3next = make_arch(true, true, true, nullptr, 0);
    static const Arch a_flint = make_arch(false, false, false, FFN_NORM, 1);
    static const Arch a_qwen3 = make_arch(false, false, false, FFN_NORM, 1);
    // Inkling2 (StrataFreet addition, INKLING2_NOTES.md): attention with sliding windows, shortconv,
    // rel-bias attention, NVFP4 experts, and a looped core. Not a qwen variant: no aliases.
    static const Arch a_inkling2 = make_arch(false, false, false, nullptr, 0);
    if (name == "qwen35" || name == "qwen35moe") return &a_qwen35;
    // Qwen3.8-Flash-Next: qwen35-style DeltaNet pairing, plus hyper-connections and PLE from metadata.
    if (name == "qwen4exp") return &a_qwen4exp;
    if (name == "qwen3next") return &a_qwen3next;
    // Recursive Flint: Qwen-style attention with sliding windows, per-head QK norms and attention sinks.
    if (name == "flint") return &a_flint;
    if (name == "inkling2") return &a_inkling2;
    if (name == "qwen3" || name == "qwen3moe") return &a_qwen3;
    return nullptr;
}

// StrataFreet addition (INKLING2_NOTES.md): the cache slot il plays, mapped to the physical block
// whose weights it shares. PR #389's hparams::loop_layer: slots below loop_start are themselves; the
// loop passes (n_passes - 1 extra) repeat the core [loop_start, loop_end); the post-loop blocks follow.
size_t loop_layer_of(const ModelConfig& c, size_t il) {
    if (!c.is_looped()) return il;
    const size_t n_core = c.loop_end - c.loop_start;
    if (il < c.loop_start) return il;
    if (il < c.loop_start + c.n_loop_passes * n_core) return c.loop_start + (il - c.loop_start) % n_core;
    return il - (c.n_loop_passes - 1) * n_core;
}

// port of config.rs: ModelConfig::from_gguf
bool ModelConfig::from_gguf(const Gguf& g, std::string& err) {
    const Value* av = g.get("general.architecture");
    const std::string* astr = av ? av->as_str() : nullptr;
    if (!astr) {
        err = "general.architecture";
        return false;
    }
    const std::string aname = *astr;
    // (qualified: the member `arch` shadows the free function inside a member)
    const Arch* a = strata::arch(aname);
    if (!a) {
        err = "unsupported architecture '" + aname +
              "' (supported: qwen3, qwen3moe, qwen3next, qwen35, qwen35moe, qwen4exp, flint, inkling2)";
        return false;
    }

    // port of config.rs: let opt = |k| g.meta.get(&format!("{arch}.{k}")).and_then(as_u64).unwrap_or(0)
    auto opt = [&](const std::string& k) -> size_t {
        const Value* v = g.get(aname + "." + k);
        if (!v) return 0;
        auto u = v->as_u64();
        return u ? (size_t)*u : 0;
    };
    // port of config.rs: let req = |k| g.get(&key)?.as_u64()
    auto req = [&](const std::string& k, size_t& out) -> bool {
        const std::string key = aname + "." + k;
        const Value* v = g.get(key);
        if (!v) {
            err = "missing " + key;
            return false;
        }
        auto u = v->as_u64();
        if (!u) {
            err = key + ": not an integer";
            return false;
        }
        out = (size_t)*u;
        return true;
    };
    // port of config.rs: let f = |k| g.meta.get(&format!("{arch}.{k}")).and_then(as_f32)
    auto f = [&](const std::string& k, std::optional<float>& out) {
        const Value* v = g.get(aname + "." + k);
        if (!v) {
            out.reset();
            return;
        }
        out = v->as_f32();
    };
    // StrataFreet addition: like `req`, but the value may be a uniform array (inkling2 stores
    // head_count / head_count_kv per slot; all entries are equal). A scalar works as before.
    auto req_arr_uniform = [&](const std::string& k, size_t& out) -> bool {
        const std::string key = aname + "." + k;
        const Value* v = g.get(key);
        if (!v) {
            err = "missing " + key;
            return false;
        }
        if (const std::vector<Value>* arr = v->as_arr()) {
            if (arr->empty()) {
                err = key + ": empty array";
                return false;
            }
            auto first = (*arr)[0].as_u64();
            if (!first) {
                err = key + ": not an integer array";
                return false;
            }
            for (size_t i = 1; i < arr->size(); i++) {
                if ((*arr)[i].as_u64() != first) {
                    err = key + ": per-layer heads differ (unsupported)";
                    return false;
                }
            }
            out = (size_t)*first;
            return true;
        }
        return req(k, out);
    };

    n_nextn = opt("nextn_predict_layers");
    size_t block_count = 0;
    if (!req("block_count", block_count)) return false;
    if (block_count < n_nextn) {
        err = "block_count " + std::to_string(block_count) + " below nextn_predict_layers " +
              std::to_string(n_nextn);
        return false;
    }
    n_layer = block_count - n_nextn;

    layers.clear();
    if (a->hybrid) {
        size_t every = opt("full_attention_interval");
        if (every == 0) every = 4;
        layers.reserve(n_layer);
        for (size_t il = 0; il < n_layer; il++)
            layers.push_back((il + 1) % every == 0 ? LayerKind::Attn : LayerKind::Delta);
    } else {
        layers.assign(n_layer, LayerKind::Attn);
    }
    bool has_delta = false;
    for (LayerKind k : layers)
        if (k == LayerKind::Delta) has_delta = true;

    // port of config.rs: let ssm = |k| if has_delta { req(k) } else { Ok(opt(k)) }
    auto ssm = [&](const std::string& k, size_t& out) -> bool {
        if (has_delta) return req(k, out);
        out = opt(k);
        return true;
    };

    if (!req("embedding_length", n_embd)) return false;
    if (!req_arr_uniform("attention.head_count", n_head)) return false;
    const size_t key_length = opt("attention.key_length");
    head_dim = key_length == 0 ? (n_head ? n_embd / n_head : 0) : key_length;
    n_expert = opt("expert_count");
    if (n_expert > 512) {
        err = std::to_string(n_expert) + " experts (at most 512 supported)";
        return false;
    }
    n_dense = opt("leading_dense_block_count");
    if (n_dense > n_layer) {
        err = std::to_string(n_dense) + " leading dense blocks for " + std::to_string(n_layer) + " layers";
        return false;
    }

    // port of config.rs: let flags = |k| -> Result<Option<Vec<bool>>>
    auto flags = [&](const std::string& k, std::optional<std::vector<bool>>& out) -> bool {
        const std::string key = aname + "." + k;
        const Value* v = g.get(key);
        if (!v) {
            out.reset();
            return true;
        }
        const std::vector<Value>* arr = v->as_arr();
        if (!arr) {
            err = key + ": not an array";
            return false;
        }
        if (arr->size() < n_layer) {
            err = key + " has " + std::to_string(arr->size()) + " entries for " + std::to_string(n_layer) +
                  " layers";
            return false;
        }
        std::vector<bool> v2(n_layer, false);
        for (size_t i = 0; i < n_layer; i++) {
            auto u = (*arr)[i].as_u64();
            v2[i] = u.value_or(0) != 0;
        }
        out = std::move(v2);
        return true;
    };

    std::optional<std::vector<bool>> swa_opt, rope_opt;
    if (!flags("attention.sliding_window_pattern", swa_opt)) return false;
    if (!flags("attention.rope_pattern", rope_opt)) return false;
    swa = swa_opt ? *swa_opt : std::vector<bool>(n_layer, false);
    rope_layer = rope_opt ? *rope_opt : std::vector<bool>(n_layer, true);
    n_swa = opt("attention.sliding_window");
    bool any_swa = false;
    for (bool b : swa)
        if (b) any_swa = true;
    if (any_swa && n_swa == 0) {
        err = "sliding-window layers without attention.sliding_window";
        return false;
    }

    auto emb = g.tensors().find("token_embd.weight");
    if (emb == g.tensors().end()) {
        err = "missing tensor token_embd.weight";
        return false;
    }
    if (emb->second.dims.size() < 2) {
        err = "token_embd.weight: expected 2 dims";
        return false;
    }

    arch = aname;
    size_t head_count_kv = 0;
    if (!req_arr_uniform("attention.head_count_kv", head_count_kv)) return false;
    n_head_kv = head_count_kv;
    size_t dim_count = opt("rope.dimension_count");
    n_rot = dim_count == 0 ? head_dim : dim_count;

    // port of config.rs: mrope_sections from `{arch}.rope.dimension_sections` (>= 4 entries, all integers)
    mrope_sections.reset();
    if (const Value* mv = g.get(aname + ".rope.dimension_sections")) {
        if (const std::vector<Value>* arr = mv->as_arr()) {
            if (arr->size() >= 4) {
                std::array<size_t, 4> sec{};
                bool ok = true;
                for (size_t i = 0; i < 4; i++) {
                    auto u = (*arr)[i].as_u64();
                    if (!u) {
                        ok = false;
                        break;
                    }
                    sec[i] = (size_t)*u;
                }
                if (ok) mrope_sections = sec;
            }
        }
    }

    std::optional<float> rf;
    f("rope.freq_base", rf);
    rope_base = rf ? *rf : 10000.0f;
    std::optional<float> ef;
    f("attention.layer_norm_rms_epsilon", ef);
    if (!ef) {
        err = "missing rms epsilon";
        return false;
    }
    eps = *ef;
    n_vocab = (size_t)emb->second.dims[1];
    n_ff = opt("feed_forward_length");
    n_expert_used = opt("expert_used_count");
    n_ff_exp = opt("expert_feed_forward_length");
    n_ff_shexp = opt("expert_shared_feed_forward_length");
    std::optional<float> ews;
    f("expert_weights_scale", ews);
    expert_weights_scale = ews ? *ews : 0.0f;
    router_sigmoid = aname == "flint" && opt("expert_gating_func") == 2;
    attn_sinks = g.tensors().count("blk.0.attn_sinks.weight") != 0;
    if (!ssm("ssm.inner_size", ssm_d_inner)) return false;
    if (!ssm("ssm.state_size", ssm_d_state)) return false;
    if (!ssm("ssm.group_count", ssm_n_group)) return false;
    if (!ssm("ssm.time_step_rank", ssm_dt_rank)) return false;
    if (!ssm("ssm.conv_kernel", ssm_conv)) return false;
    hc_n = opt("hyper_connection.count");
    hc_rank = opt("hyper_connection.low_rank");
    n_embd_ple = opt("embedding_length_per_layer_input");
    idx_n_head = opt("attention.indexer.head_count");
    idx_head_dim = opt("attention.indexer.key_length");
    idx_top_k = opt("attention.indexer.top_k");
    q_gate = a->q_gate;
    k_interleave = a->k_interleave;
    aliases = a->aliases;
    kind = a->kind;
    has_eagle3 = false;

    // StrataFreet addition (INKLING2_NOTES.md): inkling2's loop geometry, attention scalars and the
    // padded vocabulary. block_count is the SLOT count (48); the physical blocks are fewer. The
    // reference validates the same invariants (inkling2.cpp load_arch_hparams).
    if (aname == "inkling2") {
        kind = Arch::Kind::Inkling2;
        // The router is always sigmoid (LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID in the reference).
        router_sigmoid = true;
        // No RoPE: the relative-position bias carries the positions.
        rope_layer.assign(n_layer, false);

        if (!req("vocab_size", n_vocab_padded)) return false;
        if (!req("vocab_size_unpadded", n_vocab_unpadded)) return false;
        if (n_vocab_unpadded == 0 || n_vocab_unpadded > n_vocab_padded) {
            err = "inkling2: invalid padded/unpadded vocabulary";
            return false;
        }
        // The padded row count of token_embd must agree with the metadata: n_vocab drives the logits
        // row count, n_vocab_padded the unpadding mask — a disagreement would read past one of them.
        if (n_vocab_padded != n_vocab) {
            err = "inkling2: vocab_size " + std::to_string(n_vocab_padded) + " does not match the token_embd rows " +
                  std::to_string(n_vocab);
            return false;
        }
        if (!req("attention.relative_length", n_rel)) return false;
        if (!req("attention.relative_extent", n_rel_extent)) return false;
        if (!req("attention.sliding_window", n_swa)) return false;
        if (!req("attn_res.block_size", n_attn_res_block)) return false;
        if (!req("shortconv.l_cache", n_shortconv)) return false;
        if (!n_rel || !n_rel_extent || !n_swa || !n_attn_res_block || n_shortconv < 2 || n_shortconv > 64) {
            err = "inkling2: invalid dimensions, vocabulary, or layer pattern";
            return false;
        }
        // The reference requires qk_clip, relative_clip, situ_beta, situ_linear_beta finite positive.
        auto reqf = [&](const std::string& k, float& out) -> bool {
            std::optional<float> v;
            f(k, v);
            if (!v) {
                err = "missing " + aname + "." + k;
                return false;
            }
            out = *v;
            return true;
        };
        if (!reqf("attention.qk_clip", f_qk_clip)) return false;
        if (!reqf("attention.relative_clip", f_rel_clip)) return false;
        if (!reqf("activation.situ_beta", situ_beta)) return false;
        if (!reqf("activation.situ_linear_beta", situ_linear_beta)) return false;
        for (float value : {eps, f_qk_clip, f_rel_clip, situ_beta, situ_linear_beta}) {
            if (!(value > 0.0f) || std::isfinite(value) == false) {
                err = "inkling2: clipping and scaling parameters must be finite positive";
                return false;
            }
        }
        if (!req("attention.log_scale_floor", n_log_scale_floor)) return false;
        if (!reqf("attention.log_scale_alpha", f_log_scale_alpha)) return false;
        if (!reqf("final_logit_softcapping", f_final_softcap)) return false;

        // The looped core (num_loops >= 2; slots 0..5, 42..47 are outside it for this model).
        n_loop_passes = opt("num_loops");
        if (n_loop_passes > 1) {
            if (!req("loop.core_start", loop_start)) return false;
            if (!req("loop.core_end", loop_end)) return false;
            if (!req("loop.gate_rank", loop_gate_rank)) return false;
            if (!loop_gate_rank || !loop_start || loop_start >= loop_end ||
                loop_start % n_attn_res_block || loop_end % n_attn_res_block) {
                err = "inkling2: the looped core must start after the first block and span whole blocks";
                return false;
            }
            const size_t n_extra = (n_loop_passes - 1) * (loop_end - loop_start);
            if (n_extra >= n_layer) {
                err = "inkling2: loop passes exceed the physical blocks";
                return false;
            }
            n_layer_phys = n_layer - n_extra;
            if (loop_end > n_layer_phys) {
                err = "inkling2: loop core end beyond the physical blocks";
                return false;
            }
        } else {
            n_loop_passes = 0;
            n_layer_phys = n_layer;
        }
        if (!n_head_kv || n_head % n_head_kv != 0 || !n_expert_used || n_expert_used > n_expert) {
            err = "inkling2: invalid attention or expert dimensions";
            return false;
        }
    }

    // Kernels divide by these and index value heads by nv / nk key heads.
    const size_t nk = ssm_n_group, nv = ssm_dt_rank;
    if (has_delta && !(nk > 0 && nv > 0 && nv % nk == 0 && ssm_d_inner % nv == 0)) {
        err = "bad ssm dims: group_count " + std::to_string(nk) + ", time_step_rank " + std::to_string(nv) +
              ", inner_size " + std::to_string(ssm_d_inner);
        return false;
    }
    return true;
}

}  // namespace strata
