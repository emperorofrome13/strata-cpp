// port of gpu_model.rs: GpuModel::load (layers, output head, scratch, expert cache and CPU worker), reset, step,
// step_greedy, forward, layer (the decode forward of one token, with the hyper-connection streams, PLE and QSA of
// qwen4exp), finish, hc_read, ple, after_experts and save_profile, with the graph key of GpuModel::run.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "gpu_model.h"

#include "quant.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <utility>

namespace strata {

bool has_indexer(const Model& m, size_t il) {
    Tensor t;
    std::string e;
    return m.w.layer(il, "indexer.k_proj.weight", t, e);
}

bool qsa_active(const Model& m) {
    if (m.cfg.idx_top_k == 0) return false;
    for (size_t il = 0; il < m.cfg.n_layer; il++) {
        if (has_indexer(m, il)) return true;
    }
    return false;
}

namespace {

// Rows of every scratch activation: load_part sizes each one TMAX rows (the widest verify the backend takes).
constexpr size_t kTmax = TMAX;

// port of gpu_model.rs: gq_min (the decode attention regime for `nt` tokens from position `pos`)
int32_t gq_min(size_t pos, size_t nt) {
    const size_t m = static_cast<size_t>(ATT_GQA_MIN);
    if (pos + nt < m) return INT32_MAX;
    if (pos + 1 >= m) return 0;
    return ATT_GQA_MIN;
}

// port of gpu_model.rs: the STRATA_ADAPT_EVERY / STRATA_ADAPT_SWAPS environment knobs (defaults 8 and 64)
size_t env_size(const char* name, size_t dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    const unsigned long long x = std::strtoull(v, &end, 10);
    return end == v ? dflt : static_cast<size_t>(x);
}

// A boolean environment switch: set and non-zero enables, so STRATA_INK_DEBUG=0 / STRATA_INK_XQ=0 turn the feature
// off (the debug sentinel upload breaks graph capture, so an explicit 0 must not enable it).
bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::atoi(v) != 0;
}

// Runs `f` as the batch of independent ops (port of Backend::batch as gpu_model.rs calls it).
template <class F>
bool batch(Backend& g, F&& f, std::string& err) {
    return g.batch([&](std::string& e) { return f(e); }, err);
}

// port of gpu_model.rs: Model::hc_read's `mv` / GpuModel::hmv: y[t] = W x[t] for t < nt, a dense GEMM for `a.gemm`
bool hmv(Backend& g, const HcAt& a, const GpuTensor& w, uint64_t x, size_t xs, uint64_t y, size_t ys, size_t nt,
         std::string& err) {
    if (a.gemm) return g.gemm_dense(w, nt, x, xs, y, ys, 0, err);
    if (nt == 1) return g.gemv(w, x, y, err);
    return g.gemvn(w, x, xs, y, ys, nt, err);
}

// port of gpu_model.rs: load_part's scratch activations, (name, floats a row) for TMAX rows each. The QSA and
// hyper-connection rows are there when the model has them.
std::vector<std::pair<const char*, size_t>> scratch_sizes(const ModelConfig& c, bool qsa_on) {
    const size_t ne = c.n_embd;
    const size_t ke = std::max<size_t>(c.n_expert_used, 1);
    const size_t ff = std::max(ke * c.n_ff_exp, c.n_ff);
    const size_t nh = c.n_head, hd = c.head_dim;
    const size_t kv = c.n_head_kv * hd;
    const size_t channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
    const size_t nv = c.ssm_dt_rank;
    std::vector<std::pair<const char*, size_t>> v = {
        {"h", ne}, {"hq", ne * 9 / 32}, {"mix", 3 * ne}, {"qkv", channels}, {"z", c.ssm_d_inner}, {"beta", nv}, {"g", nv},
        {"y", channels}, {"o", c.ssm_d_inner}, {"qg", nh * 2 * hd}, {"q", nh * hd}, {"kraw", kv}, {"k", kv}, {"v", kv},
        {"att", nh * hd}, {"rl", c.n_expert}, {"wts", ke}, {"ge", ff}, {"ue", ff}, {"ye", ke * ne}, {"shg", c.n_ff_shexp},
        {"shu", c.n_ff_shexp}, {"shy", ne}, {"sg", 1}, {"wh", ke}, {"attp", nh * ATT_SPLITS * (hd + 2)},
        {"ffq", ff * 9 / 32}, {"shq", c.n_ff_shexp * 9 / 32}, {"attq", nh * hd * 9 / 32},
        {"oq", c.ssm_d_inner * 9 / 32},
    };
    if (qsa_on) v.insert(v.end(), {{"ik", QSA_D}, {"iq", QSA_H * QSA_D}, {"iqn", QSA_H * QSA_D}});
    if (c.hc_n > 0) {
        const size_t d = c.hc_n * ne;
        v.insert(v.end(), {{"xn", d}, {"lo", c.hc_rank}, {"hg", d}, {"inj", c.hc_n}, {"mo", ne}, {"pk", d}, {"pq", d},
                           {"pg", d}, {"pn", d}, {"pv", ne}, {"emb", ne}});
    }
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode scratch. bks holds the block-boundary residual
    // streams (row 0 = the prelude's b0, rows 1-2 the current pass's pushes, row 3 the tail's); bkl the previous
    // pass's last output, pre the prelude output every pass restarts from; mat/mkv the two cross-pass message pairs;
    // gat/g64/cg/cf the three loop-gate control gates, down outputs, gains and forgets (the FFN control reuses row
    // 0); gu the fused gate/up expert rows (k rows of 2 * n_ff), tbl the rel-bias table, moe the zeroed combine
    // target, ao the attention output through its post conv.
    if (c.kind == Arch::Kind::Inkling2) {
        const size_t nkv = c.n_head_kv * c.head_dim;
        v.insert(v.end(), {
            {"gu", 2 * ff},
            {"tbl", (c.n_rel_extent + 1) * nh},
            {"rel", c.n_rel * nh},
            {"sink", nh},
            {"kpj", c.n_shortconv * nkv},
            {"vpj", c.n_shortconv * nkv},
            {"bks", 4 * ne},
            {"bnk", 2 * ne},   // the message bank snapshot (the rows past the first push of the previous pass)
            {"bkl", ne},
            {"pre", ne},
            {"inpb", 3 * ne},  // the three attention inputs (distinct rows: the controls/norms write in place)
            {"mat", 2 * ne},
            {"mkv", 2 * ne},
            {"gat", 3 * 2 * ne},
            {"g64", 3 * c.loop_gate_rank},
            {"cg", 3 * ne},
            {"cf", 3 * ne},
            {"moe", ne},
            {"ao", ne},
        });
    }
    return v;
}

// The device bytes Backend::upload takes for `t` (backend.cpp: F16 widened to F32, Q4_0 and Q5_0 kept as Q8_0).
size_t upload_bytes(const Tensor& t) {
    if (t.ty == Ty::F16) return 4 * t.rows * t.cols;
    if (t.ty == Ty::Q4_0 || t.ty == Ty::Q5_0) return (t.len / block_bytes(t.ty)) * 34;
    return t.len;
}

// The device memory GpuModel::load allocates before the expert cache, counted as the allocations themselves round
// (256-byte units; see CudaBackend::reserve_arena): the layers' weights and state, the hyper-connection reads, the PLE
// blocks, the QSA indexers, the KV cache, the output head, the scratch activations and the prefill scratch for `chunk`
// tokens (0: none). Mirrors the allocations of load.
bool plan_bytes(const Model& m, Backend& g, size_t ctx, size_t chunk, bool qsa_on, size_t& total, std::string& err) {
    const ModelConfig& c = m.cfg;
    total = 0;
    auto add = [&](size_t b) { total += ((b ? b : 1) + 255) & ~size_t(255); };
    auto up = [&](const std::string& name) -> bool {
        Tensor t;
        if (!m.w.get(name, t, err)) return false;
        add(upload_bytes(t));
        return true;
    };
    // StrataFreet addition (INKLING2_NOTES.md): through Weights::layer, so a looped slot plans its
    // physical block's tensors.
    auto lt = [&](size_t il, const char* s) -> bool {
        Tensor t;
        if (!m.w.layer(il, s, t, err)) return false;
        add(upload_bytes(t));
        return true;
    };
    // An optional tensor: its bytes when present (the error of its absence is cleared).
    auto opt = [&](const std::string& name) {
        Tensor t;
        std::string e;
        if (m.w.get(name, t, e)) add(upload_bytes(t));
    };
    auto optl = [&](size_t il, const char* s) {
        Tensor t;
        std::string e;
        if (m.w.layer(il, s, t, e)) add(upload_bytes(t));
    };
    const size_t dk = c.ssm_d_state, nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    const size_t channels = c.ssm_d_inner + 2 * nk * dk;
    const size_t kv = c.n_head_kv * c.head_dim;
    const size_t hcn = std::max<size_t>(c.hc_n, 1);
    const size_t ne = c.n_embd;
    if (c.kind == Arch::Kind::Inkling2) {
        // StrataFreet addition (INKLING2_NOTES.md): the inkling2 allocations — the shared and extra
        // weights per slot (looped slots plan their physical block), the per-slot KV cache and
        // shortconv state, the loop gates of the core blocks, the global queries, the norms and the
        // output head.
        const size_t n_sconv = 2 * kv + 2 * ne;
        if (c.is_looped()) {
            if (!up("loop_res_query.weight") || !up("loop_kv_res_query.weight")) return false;
        }
        if (!up("token_embd_norm.weight") || !up("output_res_query.weight")) return false;
        // StrataFreet addition (INKLING2_NOTES.md): the plan mirrors the load — each physical block's
        // tensors are counted at the first slot that plays it (the identity slots cover the prelude
        // and the core, the post-loop slots 42..47 cover blocks 18..23), and every slot's own KV
        // cache and shortconv state are counted each time.
        std::vector<char> phys_seen(c.n_layer_phys, 0);
        for (size_t il = 0; il < c.n_layer; il++) {
            const size_t phys = c.is_looped() ? loop_layer_of(c, il) : il;
            if (phys_seen[phys]) {
                add(2 * g.kv_words(ctx * kv));
                add(2 * g.kv_words(ctx * kv));
                add(4 * (c.n_shortconv - 1) * n_sconv);  // the shortconv history
                continue;
            }
            phys_seen[phys] = 1;
            for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                  "attn_gate.weight", "attn_rel.weight", "attn_q_clip.weight",
                                  "attn_rel_clip.weight", "attn_sink_proj.weight", "attn_k_conv.weight",
                                  "attn_v_conv.weight", "attn_post_conv.weight", "ffn_post_conv.weight",
                                  "ffn_res_query.weight", "ffn_norm.weight", "ffn_gate_inp.weight"}) {
                if (!lt(il, s)) return false;
            }
            {
                // the rel_bias uploads padded to (n_extent + 1) rows of n_rel (see the load branch)
                Tensor t;
                if (!m.w.layer(il, "attn_rel_bias.weight", t, err)) return false;
                add(upload_bytes(t) + 4 * t.cols);
            }
            optl(il, "attn_norm.weight");
            optl(il, "attn_q_inp_norm.weight");
            optl(il, "attn_k_inp_norm.weight");
            optl(il, "attn_v_inp_norm.weight");
            optl(il, "attn_k_conv_proj.weight");
            optl(il, "attn_v_conv_proj.weight");
            optl(il, "attn_res_query.weight");
            optl(il, "exp_probs_b.bias");
            optl(il, "ffn_gate_up_exps.scale");
            optl(il, "ffn_down_exps.scale");
            for (const char* s : {"loop_attn_down.weight", "loop_attn_gain.weight", "loop_attn_forget.weight",
                                  "loop_k_down.weight", "loop_k_gain.weight", "loop_k_forget.weight",
                                  "loop_v_down.weight", "loop_v_gain.weight", "loop_v_forget.weight",
                                  "loop_mlp_down.weight", "loop_mlp_gain.weight", "loop_mlp_forget.weight"}) {
                optl(il, s);
            }
            add(2 * g.kv_words(ctx * kv));
            add(2 * g.kv_words(ctx * kv));
            add(4 * (c.n_shortconv - 1) * n_sconv);  // the shortconv history
        }
        Tensor t;
        if (m.w.get("output.weight", t, err)) {
            add(upload_bytes(t));
        } else {
            err.clear();
            if (!up("token_embd.weight")) return false;
        }
        // The scratch activations (scratch_sizes) and the fixed buffers, as the load allocates them.
        const size_t ke = std::max<size_t>(c.n_expert_used, 1);
        add(4 * kTmax * hcn * ne);
        for (const auto& [name, n] : scratch_sizes(c, qsa_on)) {
            (void)name;
            add(4 * kTmax * n);
        }
        add(4 * kTmax * c.n_vocab);
        add(4 * kTmax * ke);
        add(4 * kTmax * ke);
        add(4 * 2 * kTmax);
        add(4 * 2 * kTmax);
        add(4 * kTmax);
        add(4 * ne);                        // StrataFreet addition: the ones gamma of the unweighted rms_norms
        add(4 * c.n_head);                  // the zero sink vector
        if (chunk > 0) {
            PrefillSizes z;
            if (!prefill_sizes(m, chunk, z, err)) return false;
            add(4 * z.slab_floats);
            add(4 * z.ints);
            add(8 * z.tiles);
            if (z.staging > 0) add(z.staging);
        }
        return true;
    }
    for (size_t il = 0; il < c.n_layer; il++) {
        if (c.layers[il] == LayerKind::Delta) {
            for (const char* s : {"attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_dt.bias",
                                  "ssm_a", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight"}) {
                if (!lt(il, s)) return false;
            }
            add(4 * (c.ssm_conv - 1) * channels);
            add(4 * nv * c.ssm_dv() * dk);
        } else {
            for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                                  "attn_output.weight"}) {
                if (!lt(il, s)) return false;
            }
            if (c.attn_sinks && !lt(il, "attn_sinks.weight")) return false;
            add(2 * g.kv_words(ctx * kv));
            add(2 * g.kv_words(ctx * kv));
            if (qsa_on && has_indexer(m, il)) {
                for (const char* s : {"indexer.k_proj.weight", "indexer.q_proj.weight", "indexer.k_norm.weight",
                                      "indexer.q_norm.weight"}) {
                    if (!lt(il, s)) return false;
                }
                add(4 * ctx * QSA_D / QSA_POOL);  // the pooled keys
            }
        }
        if (c.hc_n > 0) {
            for (const char* p : {"hc_attn_", "hc_ffn_"}) {
                const std::string pre = "blk." + std::to_string(il) + "." + p;
                if (!up(pre + "norm.weight") || !up(pre + "down.weight") || !up(pre + "up.weight")) return false;
                opt(pre + "inject.weight");
            }
        } else if (!lt(il, "attn_norm.weight") || !lt(il, "post_attention_norm.weight")) {
            return false;
        }
        Tensor pk;
        std::string pe;
        if (m.w.layer(il, "ple_key.weight", pk, pe)) {
            Tensor kw;
            if (!m.w.get("blk." + std::to_string(il) + ".ple_conv1d.weight", kw, err)) return false;
            add(4 * kw.rows * kw.cols);
            for (const char* s : {"ple_key.weight", "ple_value.weight", "ple_norm_key.weight", "ple_norm_query.weight",
                                  "ple_norm_conv.weight"}) {
                if (!lt(il, s)) return false;
            }
        }
        if (c.is_moe_layer(il)) {
            if (!lt(il, "ffn_gate_inp.weight")) return false;
            Tensor sh;
            const bool shared = m.w.layer(il, "ffn_up_shexp.weight", sh, err);
            err.clear();
            if (shared) {
                const bool has_gi = m.w.layer(il, "ffn_gate_inp_shexp.weight", sh, err);
                err.clear();
                if (has_gi && !lt(il, "ffn_gate_inp_shexp.weight")) return false;
                if (!lt(il, "ffn_gate_shexp.weight") || !lt(il, "ffn_up_shexp.weight") || !lt(il, "ffn_down_shexp.weight"))
                    return false;
            }
        } else if (!lt(il, "ffn_gate.weight") || !lt(il, "ffn_up.weight") || !lt(il, "ffn_down.weight")) {
            return false;
        }
    }
    if (c.hc_n > 0) {
        if (!up("output_hc_norm.weight") || !up("output_hc_down.weight") || !up("output_hc_up.weight")) return false;
        opt("output_hc_inject.weight");
    } else if (!up("output_norm.weight")) {
        return false;
    }
    Tensor t;
    if (m.w.get("output.weight", t, err)) {
        add(upload_bytes(t));
    } else {
        err.clear();
        if (!up("token_embd.weight")) return false;
    }
    if (qsa_on) add(4 * c.n_layer * QSA_RING);                 // the raw indexer key ring
    if (c.n_embd_ple > 0) add(4 * 16 * hcn * ne);              // the PLE ring
    const size_t ke = std::max<size_t>(c.n_expert_used, 1);
    add(4 * kTmax * hcn * ne);                                 // x: the residual stack
    for (const auto& [name, n] : scratch_sizes(c, qsa_on)) {
        (void)name;
        add(4 * kTmax * n);
    }
    add(4 * kTmax * c.n_vocab);
    add(4 * kTmax * ke);
    add(4 * kTmax * ke);
    add(4 * 2 * kTmax);
    add(4 * 2 * kTmax);
    add(4 * kTmax);
    if (qsa_on) {
        const size_t nbs = ctx / QSA_POOL;
        const size_t sst = 1 + c.idx_top_k + QSA_POOL;
        const size_t mw = (nbs + 1 + 31) / 32;
        const size_t rows = chunk > 0 ? (chunk + 63) / 64 * 64 : 0;
        add(4 * std::max(QSA_ROWS, kTmax) * nbs);  // score
        add(4 * kTmax * sst);                      // sel
        add(4 * std::max<size_t>(rows * mw, 1));   // mask
    }
    if (chunk > 0) {
        PrefillSizes z;
        if (!prefill_sizes(m, chunk, z, err)) return false;
        add(4 * z.slab_floats);
        add(4 * z.ints);
        add(8 * z.tiles);
        if (z.staging > 0) add(z.staging);
    }
    return true;
}

}  // namespace

// port of gpu_model.rs: GpuModel::load
bool GpuModel::load(const Model& m, std::unique_ptr<Backend> gpu, const LoadOptions& o, std::unique_ptr<GpuModel>& out,
                    std::string& err) {
    const ModelConfig& c = m.cfg;
    const size_t ctx = o.ctx;
    if (!c.is_moe()) {
        err = "dense (non-MoE) models are not ported yet: this decode path covers the qwen35moe and qwen4exp families";
        return false;
    }
    if (c.n_embd_ple > 0 && c.hc_n == 0) {
        err = "per-layer embeddings without hyper-connections are not supported";
        return false;
    }
    if (ctx == 0) {
        err = "ctx must be at least 1";
        return false;
    }
    auto gm = std::unique_ptr<GpuModel>(new GpuModel(m));
    gm->gpu_ = std::move(gpu);
    gm->ctx_ = ctx;
    gm->adapt_every = env_size("STRATA_ADAPT_EVERY", 8);
    gm->adapt_swaps = env_size("STRATA_ADAPT_SWAPS", 64);
    gm->threads_ = o.threads ? o.threads : default_threads();
    // StrataFreet addition (INKLING2_NOTES.md): inkling2 has no batched prefill yet (W3); forcing the
    // chunk to 0 routes prefill through step(), whose layer() guard reports the missing forward.
    gm->chunk_ = c.kind == Arch::Kind::Inkling2 ? 0 : (o.chunk ? std::min(o.chunk, ctx) : 0);  // port of gpu_model.rs: prefill::Scratch::new(m, gpu, chunk.min(ctx))
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 batched prefill (W3). One chunk is at most 16 tokens
    // (the per-token mailbox regions are TMAX = 17 rows); chunk 0 keeps the token-by-token path. No PrefillScratch is
    // made for it: the chunk path reuses the decode buffers' spare rows.
    gm->ink_chunk_ = c.kind == Arch::Kind::Inkling2
                         ? std::min<size_t>(o.chunk ? std::min<size_t>(o.chunk, ctx) : 0, 16)
                         : 0;
    gm->short_len_ = o.short_len;
    gm->hcn_ = std::max<size_t>(c.hc_n, 1);
    gm->reset_counts_after_prefill_ = std::getenv("STRATA_RESET_COUNTS") != nullptr;
    gm->has_ple_ = c.n_embd_ple > 0;
    gm->has_output_hc_ = c.hc_n > 0;
    const bool qsa_on = qsa_active(m);
    gm->has_qsa_ = qsa_on;
    if (qsa_on && (c.idx_head_dim != QSA_D || c.idx_n_head != QSA_H)) {
        err = "QSA indexer: " + std::to_string(c.idx_n_head) + " heads of " + std::to_string(c.idx_head_dim) +
              " (the GPU kernels take " + std::to_string(QSA_H) + " of " + std::to_string(QSA_D) + ")";
        return false;
    }
    Backend& g = *gm->gpu_;
    // The free VRAM before the weights (the plan line's "weights+state+scratch" is the difference to the end of load).
    size_t free0 = 0, total0 = 0;
    if (!g.mem_info(free0, total0, err)) return false;
    // The device memory of the load, reserved as one exact block (see CudaBackend::reserve_arena).
    size_t planned = 0;
    if (!plan_bytes(m, g, ctx, gm->chunk_, qsa_on, planned, err)) return false;
    if (!g.reserve_arena(planned, err)) return false;
    // Uploads: the layer weights and state, as load_part's load_layer does.
    auto up = [&](const std::string& name, GpuTensor& o_) -> bool {
        Tensor t;
        if (!m.w.get(name, t, err)) return false;
        return g.upload(t, o_, err);
    };
    // StrataFreet addition (INKLING2_NOTES.md): the layer lookup goes through Weights::layer, whose
    // loop_layer_of remap sends a looped slot to its physical block's tensors.
    auto lt = [&](size_t il, const char* s, GpuTensor& o_) -> bool {
        Tensor t;
        if (!m.w.layer(il, s, t, err)) return false;
        return g.upload(t, o_, err);
    };
    auto lbuf = [&](size_t floats, Buf& o_) -> bool { return g.alloc(4 * floats, o_, err); };
    // port of gpu_model.rs: hc_read (the norm, down and up projections, and the inject when the file has one)
    auto hc_load = [&](const std::string& pre, HcRead& h) -> bool {
        if (!up(pre + "norm.weight", h.norm) || !up(pre + "down.weight", h.down) || !up(pre + "up.weight", h.up)) return false;
        Tensor t;
        std::string e;
        h.has_inject = m.w.get(pre + "inject.weight", t, e);
        if (h.has_inject && !up(pre + "inject.weight", h.inject)) return false;
        return true;
    };
    const size_t dk = c.ssm_d_state, nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    const size_t channels = c.ssm_d_inner + 2 * nk * dk;
    const size_t kv = c.n_head_kv * c.head_dim;
    const size_t ne = c.n_embd;
    // StrataFreet addition (INKLING2_NOTES.md): the widest shortconv state, 2*n_embd_k_gqa + 2*n_embd
    // (the K and V conv states, the attention output's and the FFN output's).
    const size_t n_sconv = 2 * kv + 2 * ne;
    if (qsa_on && !lbuf(c.n_layer * QSA_RING, gm->qsa_.ring)) return false;
    if (gm->has_ple_ && !lbuf(16 * gm->hcn_ * ne, gm->ple_ring_)) return false;
    if (c.kind == Arch::Kind::Inkling2) {
        // The global cross-pass message queries of the looped core, the embedding norm and the output
        // residual-mix query.
        if (c.is_looped()) {
            if (!up("loop_res_query.weight", gm->loop_res_query_) ||
                !up("loop_kv_res_query.weight", gm->loop_kv_res_query_))
                return false;
        }
        if (!up("token_embd_norm.weight", gm->tok_norm_) || !up("output_res_query.weight", gm->output_res_query_))
            return false;
    }
    // StrataFreet addition (INKLING2_NOTES.md): the first slot to play each physical block uploads its
    // weights; the other slots of that block (the loop's second and third passes) copy the GpuTensors.
    // The key is the PHYSICAL block, not the slot: for this model no slot has il == phys in 18..23
    // (slot 42 plays block 18), so a `phys != il` test would leave blocks 18..23 unuploaded.
    std::vector<size_t> phys_layer(c.n_layer_phys, SIZE_MAX);
    for (size_t il = 0; il < c.n_layer; il++) {
        GpuLayer L;
        L.delta = c.layers[il] == LayerKind::Delta;
        L.hc = c.hc_n > 0;
        if (c.kind == Arch::Kind::Inkling2) {
            // StrataFreet addition (INKLING2_NOTES.md): the inkling2 block. The shared q/k/v/o/gate/
            // router fields of GpuLayer carry its big weights; the SWA norm lands in attn_norm (the
            // full-attention blocks use the three input norms instead), and every slot keeps its own
            // KV cache and shortconv state.
            L.inkling2 = true;
            const size_t phys = c.is_looped() ? loop_layer_of(c, il) : il;
            if (phys_layer[phys] != SIZE_MAX) {
                // A later slot of the same physical block shares its weight tensors (the recorded
                // design: "two/three GpuLayer entries, same addresses"); only the KV cache and the
                // shortconv state below are the slot's own.
                const GpuLayer& src = gm->layers_[phys_layer[phys]];
                L.q = src.q;
                L.k = src.k;
                L.v = src.v;
                L.o = src.o;
                L.dn_gate = src.dn_gate;
                L.router = src.router;
                L.attn_norm = src.attn_norm;
                L.ink = src.ink;
                L.ink.conv_state = Buf{};  // the slot's own state is allocated below
            } else {
                phys_layer[phys] = gm->layers_.size();  // the index this layer is pushed at
            if (!lt(il, "attn_q.weight", L.q) || !lt(il, "attn_k.weight", L.k) || !lt(il, "attn_v.weight", L.v) ||
                !lt(il, "attn_output.weight", L.o) || !lt(il, "attn_gate.weight", L.dn_gate) ||
                !lt(il, "ffn_gate_inp.weight", L.router))
                return false;
            if (!lt(il, "attn_rel.weight", L.ink.rel) || !lt(il, "attn_q_clip.weight", L.ink.q_clip) ||
                !lt(il, "attn_rel_clip.weight", L.ink.rel_clip) || !lt(il, "attn_sink_proj.weight", L.ink.sink_proj))
                return false;
            {
                // attn_rel_bias {n_rel, n_extent} uploads PADDED to {n_rel, n_extent + 1}: the reference pads the
                // bias with a zero row before the table projection, so a distance >= extent gathers -sink[h].
                Tensor tb;
                if (!m.w.layer(il, "attn_rel_bias.weight", tb, err)) return false;
                const size_t nr = tb.cols, next_ = tb.rows;
                const std::vector<float> b = tb.to_f32();
                std::vector<float> pad((next_ + 1) * nr, 0.0f);
                std::copy(b.begin(), b.end(), pad.begin());
                Buf bb;
                if (!g.alloc(4 * pad.size(), bb, err) || !put(g, pad.data(), pad.size(), bb, 0, err)) return false;
                L.ink.rel_bias = GpuTensor(Ty::F32, next_ + 1, nr, std::move(bb));
            }
            if (!lt(il, "attn_k_conv.weight", L.ink.k_conv) || !lt(il, "attn_v_conv.weight", L.ink.v_conv) ||
                !lt(il, "attn_post_conv.weight", L.ink.post_conv) || !lt(il, "ffn_post_conv.weight", L.ink.ffn_post_conv))
                return false;
            if (!lt(il, "ffn_res_query.weight", L.ink.ffn_res_q) || !lt(il, "ffn_norm.weight", L.ink.ffn_norm)) return false;
            {
                Tensor t;
                std::string probe;
                // The attention norm of the SWA blocks; the input norms and the conv tap projections
                // of the full-attention blocks (one block in attn_res.block_size).
                if (m.w.layer(il, "attn_norm.weight", t, probe)) {
                    err.clear();
                    if (!lt(il, "attn_norm.weight", L.attn_norm)) return false;
                } else {
                    err.clear();
                    if (!lt(il, "attn_q_inp_norm.weight", L.ink.q_inp_norm) ||
                        !lt(il, "attn_k_inp_norm.weight", L.ink.k_inp_norm) ||
                        !lt(il, "attn_v_inp_norm.weight", L.ink.v_inp_norm) ||
                        !lt(il, "attn_k_conv_proj.weight", L.ink.k_conv_proj) ||
                        !lt(il, "attn_v_conv_proj.weight", L.ink.v_conv_proj))
                        return false;
                }
                // The residual-mix query (absent on the first block) and the router bias (optional).
                if (m.w.layer(il, "attn_res_query.weight", t, probe)) {
                    err.clear();
                    if (!lt(il, "attn_res_query.weight", L.ink.res_q)) return false;
                } else {
                    err.clear();
                }
                if (m.w.layer(il, "exp_probs_b.bias", t, probe)) {
                    err.clear();
                    if (!lt(il, "exp_probs_b.bias", L.ink.exp_probs_b)) return false;
                } else {
                    err.clear();
                }
                // The per-expert f32 scales (the NVFP4 blocks; absent on the IQ4_NL ones).
                if (m.w.layer(il, "ffn_gate_up_exps.scale", t, probe)) {
                    err.clear();
                    if (!lt(il, "ffn_gate_up_exps.scale", L.ink.gu_scale)) return false;
                } else {
                    err.clear();
                }
                if (m.w.layer(il, "ffn_down_exps.scale", t, probe)) {
                    err.clear();
                    if (!lt(il, "ffn_down_exps.scale", L.ink.dn_scale)) return false;
                } else {
                    err.clear();
                }
            }
            // The loop gates of the core blocks (loop_start <= phys < loop_end); a shared slot copied
            // them with the rest of its physical block above.
            if (c.is_looped() && phys >= c.loop_start && phys < c.loop_end) {
                static const char* down[4] = {"loop_attn_down.weight", "loop_k_down.weight", "loop_v_down.weight",
                                              "loop_mlp_down.weight"};
                static const char* gain[4] = {"loop_attn_gain.weight", "loop_k_gain.weight", "loop_v_gain.weight",
                                              "loop_mlp_gain.weight"};
                static const char* forget[4] = {"loop_attn_forget.weight", "loop_k_forget.weight",
                                                "loop_v_forget.weight", "loop_mlp_forget.weight"};
                for (int gate = 0; gate < 4; gate++) {
                    if (!lt(il, down[gate], L.ink.loop_down[gate]) || !lt(il, gain[gate], L.ink.loop_gain[gate]) ||
                        !lt(il, forget[gate], L.ink.loop_forget[gate]))
                        return false;
                }
            }
            }  // the per-physical-block upload branch (a shared slot only copies the tensors)
            if (!g.alloc(2 * g.kv_words(ctx * kv), L.kc, err) || !g.alloc(2 * g.kv_words(ctx * kv), L.vc, err))
                return false;
            if (!lbuf((c.n_shortconv - 1) * n_sconv, L.ink.conv_state)) return false;
            gm->layers_.push_back(std::move(L));
            continue;
        }
        if (L.delta) {
            if (!lt(il, "attn_qkv.weight", L.qkv) || !lt(il, "attn_gate.weight", L.dn_gate) ||
                !lt(il, "ssm_beta.weight", L.beta) || !lt(il, "ssm_alpha.weight", L.alpha) ||
                !lt(il, "ssm_dt.bias", L.dt_bias) || !lt(il, "ssm_a", L.a) || !lt(il, "ssm_conv1d.weight", L.conv_w) ||
                !lt(il, "ssm_norm.weight", L.norm) || !lt(il, "ssm_out.weight", L.out))
                return false;
            if (!lbuf((c.ssm_conv - 1) * channels, L.conv) || !lbuf(nv * c.ssm_dv() * dk, L.s)) return false;
        } else {
            if (!lt(il, "attn_q.weight", L.q) || !lt(il, "attn_k.weight", L.k) || !lt(il, "attn_v.weight", L.v) ||
                !lt(il, "attn_q_norm.weight", L.qn) || !lt(il, "attn_k_norm.weight", L.kn) ||
                !lt(il, "attn_output.weight", L.o))
                return false;
            if (c.attn_sinks && !lt(il, "attn_sinks.weight", L.sink)) return false;
            if (!g.alloc(2 * g.kv_words(ctx * kv), L.kc, err) || !g.alloc(2 * g.kv_words(ctx * kv), L.vc, err)) return false;
            if (qsa_on && has_indexer(m, il)) {
                L.has_qsa = true;
                if (!lt(il, "indexer.k_proj.weight", L.qsa.kp) || !lt(il, "indexer.q_proj.weight", L.qsa.qp) ||
                    !lt(il, "indexer.k_norm.weight", L.qsa.kn) || !lt(il, "indexer.q_norm.weight", L.qsa.qn))
                    return false;
                // Rows of QSA_D floats per position, as the KV rows: ctx / QSA_POOL blocks of QSA_POOL positions.
                if (!lbuf(ctx * QSA_D / QSA_POOL, L.qsa.pool)) return false;
                L.qsa.ring = gm->qsa_.ring.addr() + 4 * il * QSA_RING;
            }
        }
        if (L.hc) {
            if (!hc_load("blk." + std::to_string(il) + ".hc_attn_", L.hcr[0]) ||
                !hc_load("blk." + std::to_string(il) + ".hc_ffn_", L.hcr[1]))
                return false;
        } else if (!lt(il, "attn_norm.weight", L.attn_norm) || !lt(il, "post_attention_norm.weight", L.post_norm)) {
            return false;
        }
        Tensor probe;
        if (m.w.layer(il, "ple_key.weight", probe, err)) {
            // port of gpu_model.rs: the PLE block; ple_conv1d as f32 [channel][tap] on the device
            L.has_ple = true;
            Tensor kw;
            if (!m.w.get("blk." + std::to_string(il) + ".ple_conv1d.weight", kw, err)) return false;
            const std::vector<float> conv = kw.to_f32();
            if (!g.alloc(4 * conv.size(), L.ple.conv, err) || !put(g, conv.data(), conv.size(), L.ple.conv, 0, err)) return false;
            if (!lt(il, "ple_key.weight", L.ple.key) || !lt(il, "ple_value.weight", L.ple.value) ||
                !lt(il, "ple_norm_key.weight", L.ple.norm_key) || !lt(il, "ple_norm_query.weight", L.ple.norm_query) ||
                !lt(il, "ple_norm_conv.weight", L.ple.norm_conv))
                return false;
        } else {
            err.clear();
        }
        L.moe = c.is_moe_layer(il);
        if (L.moe) {
            if (!lt(il, "ffn_gate_inp.weight", L.router)) return false;
            Tensor sh;
            L.shared = m.w.layer(il, "ffn_up_shexp.weight", sh, err);
            err.clear();
            if (L.shared) {
                L.has_shared_gate_inp = m.w.layer(il, "ffn_gate_inp_shexp.weight", sh, err);
                err.clear();
                if (L.has_shared_gate_inp && !lt(il, "ffn_gate_inp_shexp.weight", L.sh_gate_inp)) return false;
                if (!lt(il, "ffn_gate_shexp.weight", L.sh_gate) || !lt(il, "ffn_up_shexp.weight", L.sh_up) ||
                    !lt(il, "ffn_down_shexp.weight", L.sh_down))
                    return false;
            }
        } else {
            if (!lt(il, "ffn_gate.weight", L.gate) || !lt(il, "ffn_up.weight", L.up) || !lt(il, "ffn_down.weight", L.down))
                return false;
        }
        gm->layers_.push_back(std::move(L));
    }
    if (c.hc_n > 0) {
        if (!hc_load("output_hc_", gm->output_hc_)) return false;
    } else if (!up("output_norm.weight", gm->output_norm_)) {
        return false;
    }
    {
        Tensor t;
        if (m.w.get("output.weight", t, err)) {
            if (!g.upload(t, gm->output_, err)) return false;
        } else {
            err.clear();
            if (!up("token_embd.weight", gm->output_)) return false;
        }
    }

    // Scratch activations (load_part's `sizes`, TMAX rows each; widths in floats).
    const size_t ke = std::max<size_t>(c.n_expert_used, 1);
    for (const auto& [name, n] : scratch_sizes(c, qsa_on)) {
        Buf b;
        if (!lbuf(kTmax * n, b)) return false;
        gm->bufs_[name] = std::move(b);
    }
    if (!lbuf(kTmax * gm->hcn_ * ne, gm->x_) || !lbuf(kTmax * c.n_vocab, gm->logits_)) return false;
    if (!g.alloc(4 * kTmax * ke, gm->ids_, err) || !g.alloc(4 * kTmax * ke, gm->slots_, err)) return false;
    if (!g.alloc(4 * 2 * kTmax, gm->step_buf_, err) || !g.alloc(4 * 2 * kTmax, gm->rope_buf_, err)) return false;
    if (!g.alloc(4 * kTmax, gm->amax_, err)) return false;
    if (c.kind == Arch::Kind::Inkling2) {
        // StrataFreet addition (INKLING2_NOTES.md): the ones gamma of the unweighted rms_norms (the V heads) and the
        // zero sink vector the attention takes (the sink logit lives in the rel-bias table; the softmax's extra
        // term is exp(0) = 1).
        if (!lbuf(ne, gm->ones_) || !lbuf(c.n_head, gm->sinkz_)) return false;
        std::vector<float> one(ne, 1.0f);
        if (!put(g, one.data(), ne, gm->ones_, 0, err)) return false;
        // The batched-prefill expert rows of one ink_chunk_ (t * ke slots of 2 * n_ff) and situ_q's quantized form
        // (rows * n * 9 / 32 floats). Only made when the chunk path is on.
        if (gm->ink_chunk_ > 0) {
            const size_t ke_ = std::max<size_t>(c.n_expert_used, 1);
            if (!lbuf(gm->ink_chunk_ * ke_ * 2 * c.n_ff_exp, gm->bufs_["gu16"]) ||
                !lbuf(gm->ink_chunk_ * ke_ * c.n_ff_exp * 9 / 32, gm->bufs_["ffq16"]))
                return false;
        }
        // Debug grid: 2304-float rows; the night-2 dump layout reaches row 72 (ctrl stages cg/cf) and the slot-0
        // multi-row bands end at row 49, so 200*1024 floats (88 rows) leaves headroom (the old 160*1024 = 71.1 rows
        // overran on the row-71 ne write).
        if (env_flag("STRATA_INK_DEBUG") && !lbuf(200 * 1024, gm->ink_dbg_)) return false;
        if (std::getenv("STRATA_INK_DUMP") && !lbuf(16 * ne + 1024, gm->ink_dump_)) return false;
    }
    // The prefill scratch (load_part: prefill::Scratch::new before the free-VRAM check, so the expert slots leave it room).
    const bool trace = std::getenv("STRATA_LOAD_TRACE") != nullptr;
    auto show = [&](const char* what) {
        if (!trace) return;
        size_t f = 0, t = 0;
        if (g.mem_info(f, t, err)) std::fprintf(stderr, "load trace: %-22s used %8.1f MB\n", what, (double(free0) - double(f)) / 1048576.0);
    };
    show("after uploads+bufs");
    if (gm->chunk_ > 0 && !gm->make_prefill(gm->chunk_, err)) return false;
    show("after prefill scratch");
    if (qsa_on) {
        // port of gpu_model.rs: QsaBufs (the scores of the query rows, the selection of a decode step, the prefill mask)
        QsaBufs& q = gm->qsa_;
        q.nbs = ctx / QSA_POOL;
        q.top = c.idx_top_k / QSA_POOL;
        q.sst = 1 + c.idx_top_k + QSA_POOL;
        q.mw = (q.nbs + 1 + 31) / 32;
        const size_t rows = gm->pf_ ? (gm->pf_->chunk + 63) / 64 * 64 : 0;
        if (!lbuf(std::max(QSA_ROWS, kTmax) * q.nbs, q.score) || !g.alloc(4 * kTmax * q.sst, q.sel, err) ||
            !g.alloc(4 * std::max<size_t>(rows * q.mw, 1), q.mask, err))
            return false;
    }

    // VRAM: the same check as load_part (the model, the KV cache and the scratch must leave the margin free, at most
    // 512 MB), then the expert slots take what is left above the margin (STRATA_EXPERT_MB caps them, as in Rust).
    size_t free1 = 0, total1 = 0;
    if (!g.mem_info(free1, total1, err)) return false;
    show("before expert cache");
    const size_t floor_bytes = std::min<size_t>(o.vram_margin, size_t(512) << 20);
    if (free1 < floor_bytes) {
        err = "the model, KV cache and scratch leave " + std::to_string(free1 >> 20) + " MB of VRAM free, under the " +
              std::to_string(floor_bytes >> 20) + " MB margin: lower ctx (or pass vram_margin_mb 0)";
        return false;
    }
    size_t budget = free1 > o.vram_margin ? free1 - o.vram_margin : 0;
    if (const char* cap = std::getenv("STRATA_EXPERT_MB"); cap && *cap) budget = std::min(budget, env_size("STRATA_EXPERT_MB", 0) << 20);
    // StrataFreet addition (INKLING2_NOTES.md): the cache is keyed by the physical blocks (the loop
    // passes read the same table); the mailbox and the CPU worker cover every slot (each pass routes
    // independently, and the worker's expert lookup remaps slot -> phys through Weights::layer).
    const size_t cache_layers = c.is_looped() ? c.n_layer_phys : c.n_layer;
    if (!ExpertCache::create(m, g, budget, 0, cache_layers, gm->cache_, err)) return false;
    // StrataFreet addition (INKLING2_NOTES.md): the batched inkling2 prefill is a win when the GPU computes every
    // expert (IQ3_XXS pp512 110.5 vs 73.7 tok/s token-by-token at ctx 8192). With experts on the CPU worker the
    // per-chunk request granularity measured slower than the token steps (Q8 pp512 9.2 vs 13.9), so a partly
    // resident cache keeps the token path. The gu16/ffq16 buffers stay allocated (1.3 MB).
    if (c.kind == Arch::Kind::Inkling2 && gm->ink_chunk_ > 0 && !gm->cache_->fully_resident()) gm->ink_chunk_ = 0;
    gm->layout_ = Mailbox::make(c.n_expert_used, c.n_embd);

    // The CPU expert worker over the host-mapped mailboxes of the MoE layers (Rust: CpuWorker::new in load_part).
    std::vector<size_t> moe_layers;
    for (size_t il = 0; il < c.n_layer; il++) {
        if (c.is_moe_layer(il)) moe_layers.push_back(il);
    }
    if (!moe_layers.empty()) {
        const size_t words = gm->layout_.stride * (moe_layers.back() + 1);
        uint32_t* host = nullptr;
        if (!g.host_mailbox(words, gm->mailbox_, host, err)) return false;
        if (!CpuWorker::create(m.w, host, 4 * words, gm->mailbox_.addr(), moe_layers, gm->threads_, gm->worker_, err))
            return false;
    }
    // port of gpu_model.rs: the `plan:` line of load_part (weights+state+scratch = free VRAM before minus after; KV for
    // ctx tokens over the attention layers; prefill scratch; expert slots and the resident count), then the
    // `expert traffic:` line of the expert cache.
    {
        const double mb_ = 1048576.0;
        size_t n_attn = 0;
        for (size_t il = 0; il < c.n_layer; il++) n_attn += c.layers[il] == LayerKind::Attn ? 1 : 0;
        const size_t kv_bytes = 2 * 2 * g.kv_words(ctx * kv) * n_attn;
        const size_t used = free0 > free1 ? free0 - free1 : 0;
        const size_t slots = gm->cache_ ? gm->cache_->slot_bytes_total() : 0;
        const size_t resident = gm->cache_ ? gm->cache_->resident_count() : 0;
        const size_t total = gm->cache_ ? gm->cache_->total_experts() : 0;
        char line[512];
        std::snprintf(line, sizeof line,
                      "plan: weights+state+scratch %.0f MB (KV %.0f MB for %zu tokens, prefill %.0f MB), expert slots %.0f MB, "
                      "resident experts %zu/%zu\n",
                      used / mb_, kv_bytes / mb_, ctx, gm->prefill_bytes() / mb_, slots / mb_, resident, total);
        std::fputs(line, stderr);
        if (gm->cache_) {
            const size_t k = c.n_expert_used;
            size_t per = 0;
            for (size_t il = 0; il < gm->cache_->layers_len(); il++) per = std::max(per, gm->cache_->active_set_bytes(il, k));
            std::snprintf(line, sizeof line,
                          "expert traffic: %.1f MB per layer active set (top-%zu), %.1f MB per token over %zu MoE layers\n",
                          per / mb_, k, gm->cache_->token_bytes(k) / mb_, total / std::max<size_t>(c.n_expert, 1));
            std::fputs(line, stderr);
        }
    }
    out = std::move(gm);
    return true;
}

GpuModel::~GpuModel() = default;

uint64_t GpuModel::buf(const char* name) const { return bufs_.at(name).addr(); }

// port of gpu_model.rs: GpuModel::hc_at (the decode scratch rows; absent rows read as 0)
HcAt GpuModel::hc_at() const {
    auto p = [&](const char* n) -> uint64_t {
        auto it = bufs_.find(n);
        return it == bufs_.end() ? 0 : it->second.addr();
    };
    HcAt a;
    a.xn = p("xn");
    a.lo = p("lo");
    a.hg = p("hg");
    a.inj = p("inj");
    a.h = p("h");
    a.hq = p("hq");
    a.emb = p("emb");
    a.pk = p("pk");
    a.pq = p("pq");
    a.pg = p("pg");
    a.pn = p("pn");
    a.pv = p("pv");
    a.gemm = false;
    return a;
}

// port of gpu_model.rs: GpuModel::gnorm (per-stream RMSNorm of the hc_n rows of n_embd at `x`, nt stacks)
bool GpuModel::gnorm(uint64_t x, uint64_t w, uint64_t out, size_t nt, std::string& err) {
    const ModelConfig& c = m_.cfg;
    return gpu_->rms_norm_streams(x, w, out, c.n_embd, c.hc_n, nt, c.eps, err);
}

// port of gpu_model.rs: GpuModel::hc_read (see Model::hc_read): the mixed vector into a.h (and a.hq), the inject weights
// into a.inj, from the stack at `r` of `nt` tokens.
bool GpuModel::hc_read(const HcRead& hr, uint64_t r, size_t nt, const HcAt& a, std::string& err) {
    Backend& g = *gpu_;
    const ModelConfig& c = m_.cfg;
    const size_t ne = c.n_embd, hc = c.hc_n, d = c.hc_n * c.n_embd;
    if (!gnorm(r, hr.norm.ptr, a.xn, nt, err)) return false;
    if (hr.has_inject && a.gemm) {
        if (!g.gemm_dense(hr.down, nt, a.xn, d, a.lo, c.hc_rank, 0, err)) return false;
        if (!g.gemm_dense(hr.inject, nt, a.xn, d, a.inj, hc, 0, err)) return false;
    } else if (hr.has_inject) {
        if (!g.gemvq2(hr.down, hr.inject, a.xn, d, 0, a.lo, a.inj, nt, err)) return false;
    } else if (!hmv(g, a, hr.down, a.xn, d, a.lo, c.hc_rank, nt, err)) {
        return false;
    }
    if (!g.hc_silu(a.lo, 1.0f / static_cast<float>(hc), nt * c.hc_rank, err)) return false;
    if (!hmv(g, a, hr.up, a.lo, c.hc_rank, a.hg, d, nt, err)) return false;
    if (!g.hc_mix(a.xn, a.hg, a.h, ne, hc, nt, err)) return false;
    if (a.hq != 0 && !g.quant_xq(a.h, ne, ne, nt, a.hq, err)) return false;
    return true;
}

// port of gpu_model.rs: GpuModel::ple (the PLE block on the stack at `x`, PLE input in a.emb): a per-stream gate from
// key . query scales the value; the normalized gated rows feed the causal conv over the PLE ring.
bool GpuModel::ple_block(const PleBlock& pl, uint64_t x, uint64_t step, size_t nt, const HcAt& a, std::string& err) {
    Backend& g = *gpu_;
    const ModelConfig& c = m_.cfg;
    const size_t ne = c.n_embd, hc = c.hc_n, d = c.hc_n * c.n_embd;
    if (!hmv(g, a, pl.key, a.emb, ne, a.xn, d, nt, err)) return false;
    if (!gnorm(a.xn, pl.norm_key.ptr, a.pk, nt, err)) return false;
    if (!gnorm(x, pl.norm_query.ptr, a.pq, nt, err)) return false;
    if (!hmv(g, a, pl.value, a.emb, ne, a.pv, ne, nt, err)) return false;
    if (!g.ple_gate(a.pk, a.pq, a.pv, a.pg, ne, hc, nt, err)) return false;
    if (!gnorm(a.pg, pl.norm_conv.ptr, a.pn, nt, err)) return false;
    return g.ple_conv(x, a.pg, a.pn, pl.conv.addr(), ple_ring_.addr(), step, d, nt, err);
}

// port of gpu_model.rs: GpuModel::reset
bool GpuModel::reset(std::string& err) {
    Backend& g = *gpu_;
    for (const GpuLayer& l : layers_) {
        if (l.delta) {
            if (!g.memset(l.conv, err) || !g.memset(l.s, err)) return false;
        } else {
            if (!g.memset(l.kc, err) || !g.memset(l.vc, err)) return false;
        }
        // StrataFreet addition (INKLING2_NOTES.md): the shortconv history is recurrent state too; a
        // reset that leaves it stale poisons the next conversation's convolutions.
        if (l.inkling2 && !g.memset(l.ink.conv_state, err)) return false;
    }
    pos_ = 0;
    ple_prev_ = {-1, -1};
    return true;
}

// port of gpu_model.rs: GpuModel::step
bool GpuModel::step(uint32_t token, std::vector<float>& logits, std::string& err) {
    return run_token(token, true, &logits, nullptr, err);
}

// port of gpu_model.rs: GpuModel::step_greedy (the argmax on the device, first index on ties)
bool GpuModel::step_greedy(uint32_t token, int32_t& out, std::string& err) {
    return run_token(token, false, nullptr, &out, err);
}

// port of gpu_model.rs: GpuModel::run_token (GpuModel::step: the embedding (and PLE input), the per-token values, the
// graph, finish, the logits or argmax, then after_forward's position and expert counters)
bool GpuModel::run_token(uint32_t token, bool want_logits, std::vector<float>* logits, int32_t* argmax, std::string& err) {
    if (pos_ >= ctx_) {
        err = "context full (" + std::to_string(ctx_) + " tokens; raise ctx)";
        return false;
    }
    Backend& g = *gpu_;
    std::vector<float> emb;
    if (!m_.embed(token, emb, err)) return false;
    // port of gpu_model.rs: put_embed (the embedding repeated per hyper-connection stream; the PLE input into "emb")
    std::vector<float> stack;
    stack.reserve(emb.size() * hcn_);
    for (size_t s = 0; s < hcn_; s++) stack.insert(stack.end(), emb.begin(), emb.end());
    if (!put(g, stack.data(), stack.size(), x_, 0, err)) return false;
    if (env_flag("STRATA_INK_DEBUG")) {  // verify the embedding landed intact
        std::vector<float> back(stack.size());
        if (!get(g, x_, 0, back.size(), back, err)) return false;
        int bad = 0;
        for (size_t i = 0; i < back.size(); i++) if (back[i] != stack[i]) bad++;
        std::fprintf(stderr, "  [ink] embed verify: %d/%zu mismatches\n", bad, back.size());
    }
    if (has_ple_) {
        std::vector<float> pe;
        if (!m_.ple_embed(ple_prev_, token, pe, err)) return false;
        if (!put(g, pe.data(), pe.size(), bufs_.at("emb"), 0, err)) return false;
    }
    // Per-token values the graph reads from device memory: [position, worker sequence] (sequence 0: the fully
    // resident path asks the CPU worker nothing) and [RoPE position, sequence].
    const bool fast = expert_fast_path();
    const int32_t seq = (worker_ && !fast) ? static_cast<int32_t>(worker_->begin()) : 0;
    const int32_t sb[2] = {static_cast<int32_t>(pos_), seq};
    if (!put(g, sb, 2, step_buf_, 0, err)) return false;
    const int32_t rb[2] = {static_cast<int32_t>(pos_), seq};  // rope_offset is 0 for text
    if (!put(g, rb, 2, rope_buf_, 0, err)) return false;

    // port of gpu_model.rs: GpuModel::run. The graph key of the step: nt | gq << 8 | cur << 40 | rope << 48.
    const int32_t gq = gq_min(pos_, 1);
    const uint64_t key = uint64_t(1) | (uint64_t(static_cast<uint32_t>(gq)) << 8);
    if (!g.graph(key, [this, gq](std::string& e) { return forward(gq, -1, e); }, err)) return false;
    // port of gpu_model.rs: finish (after the forward's wait, resume while a combine gave up on the worker)
    if (!finish(gq, err)) return false;
    if (const char* df = std::getenv("STRATA_INK_DUMP"); df && m_.cfg.kind == Arch::Kind::Inkling2 && ink_dump_.len() > 0) {
        // The slot-1 MoE record the graph staged into ink_dump_: h, moe, ye heads, w, ids, slots, then the pre-MoE h
        // and mix the graph writes at 8*ne+352 and 8*ne+352+4*ne BYTES: a full record is 4*ne+88 floats.
        std::vector<float> rec(4 * m_.cfg.n_embd + 88);
        if (!get(g, ink_dump_, 0, rec.size(), rec, err)) return false;
        if (FILE* f = std::fopen(df, "ab")) {
            std::fwrite(rec.data(), 4, rec.size(), f);
            std::fclose(f);
        }
    }
    if (env_flag("STRATA_INK_DEBUG") && m_.cfg.kind == Arch::Kind::Inkling2 && ink_dbg_.len() > 0) {
        if (const char* df = std::getenv("STRATA_INK_DEBUG_FILE")) {
            std::vector<float> all(200 * 1024);
            if (!get(g, ink_dbg_, 0, all.size(), all, err)) return false;
            if (FILE* f = std::fopen(df, "wb")) {
                std::fwrite(all.data(), 4, all.size(), f);
                std::fclose(f);
            }
        }
        std::vector<float> v(200 * 1024);
        if (!get(g, ink_dbg_, 0, v.size(), v, err)) return false;
        // the debug grid: 2304-float rows (row r at float 2304*r): 0-1 the embeddings, 2-13 slot 0's single-row
        // stages, 33-49 its multi-row stages (tbl 33-43, att 44-45, qg 46-47, q 48-49), 16-19 the bank rows at slot
        // 18, 20 bkl, 21-22 m_attn, 23-24 m_kv, 25 bnk, 26-28 the post-control Q/K/V inputs,
        // 50-51 the head stages, the per-slot rms at float 60*2304 + 8*slot
        for (size_t r : {size_t(21), size_t(23), size_t(25), size_t(26), size_t(50), size_t(51)}) {
            float s0 = 0.0f;
            const float* row = v.data() + 2304 * r;
            for (size_t i = 0; i < 2304; i++) s0 += row[i] * row[i];
            std::fprintf(stderr, "  [ink] row %zu rms=%.4g head:", r, std::sqrt(s0 / 2304));
            for (size_t i = 0; i < 6; i++) std::fprintf(stderr, " %.4g", row[i]);
            std::fputc('\n', stderr);
        }
        std::fprintf(stderr, "  [ink] slot rms:");
        for (size_t r = 0; r < 48; r++) {
            float s0 = 0.0f;
            const float* row = v.data() + 60 * 2304 + 8 * r;
            for (size_t i = 0; i < 8; i++) s0 += row[i] * row[i];
            std::fprintf(stderr, " %.3f", std::sqrt(s0 / 8));
        }
        std::fputc('\n', stderr);
    }
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 logits are the padded vocabulary on the device; the
    // reference strips the padded tail before anything reads them, so the argmax and the copy cover the unpadded
    // count only (the padded rows would win the argmax with garbage).
    const size_t nv = m_.cfg.kind == Arch::Kind::Inkling2 && m_.cfg.n_vocab_unpadded > 0 ? m_.cfg.n_vocab_unpadded
                                                                                        : m_.cfg.n_vocab;
    if (want_logits) {
        if (!get(g, logits_, 0, nv, *logits, err)) return false;
    } else {
        if (!g.argmax_rows(logits_.addr(), nv, 1, amax_.addr(), err)) return false;
        std::vector<int32_t> am;
        if (!get(g, amax_, 0, 1, am, err)) return false;
        *argmax = am[0];
    }
    // port of gpu_model.rs: after_forward (position, then after_experts with the position after the token)
    pos_ += 1;
    tokens += 1;
    return after_experts(pos_, err);
}

// port of gpu_model.rs: GpuModel::forward (one token; the graph body). `from` >= 0 resumes at that layer's combine.
bool GpuModel::forward(int32_t gq, int64_t from, std::string& err) {
    // StrataFreet addition (INKLING2_NOTES.md): inkling2 is not a qwen variant; its decode runs through the W2 port.
    if (m_.cfg.kind == Arch::Kind::Inkling2) return forward_inkling2(gq, from, err);
    const ModelConfig& c = m_.cfg;
    Backend& g = *gpu_;
    const size_t ne = c.n_embd;
    const uint64_t x = x_.addr();
    const uint64_t step = step_buf_.addr();
    uint64_t res = 0;
    for (size_t il = from >= 0 ? static_cast<size_t>(from) : 0; il < layers_.size(); il++) {
        if (!layer(il, layers_[il], x, step, gq, res, from >= 0 && static_cast<int64_t>(il) == from, err)) return false;
    }
    // The head: the hyper-connection read of the output stack, or the output norm (which takes the pending residual).
    if (has_output_hc_) {
        if (!hc_read(output_hc_, x, 1, hc_at(), err)) return false;
    } else if (!g.rms_norm_q(x, ne, output_norm_.ptr, buf("h"), ne, ne, 1, c.eps, buf("hq"), res, err)) {
        return false;
    }
    return g.gemvq(output_, buf("h"), ne, buf("hq"), logits_.addr(), c.n_vocab, 1, err);
}

// StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode forward (W2), the port of PR #389's graph for one
// token. The slots run in order 0..n_layer-1 and slot il plays physical block loop_layer_of(il) (the prelude 0..5,
// the loop core pass 0 = 6..17, the loop passes 18..41, the tail 42..47). The block bank `bks` holds the residual
// streams at the attn_res boundaries: row 0 is b0 (the normed embedding), rows 1-2 the current pass's pushes, row 3
// the tail's push. A loop pass past the first resets the residual to the prelude output (`pre`) and reads the
// previous pass's blocks (the bank rows past its first push) and last output (`bkl`) through the messages: two depth
// mixes (m_attn on loop_res_query, m_kv on loop_kv_res_query, two query rows each) whose columns feed control()
// (Q, K, V inputs before the input norms, and the FFN input after the ffn_res_query mix). The layer body ports
// inkling2.cpp's build_layer: the clip()ped Q/K/V/R, the shortconv with the state regions at 3 * channel offset,
// the rel-bias table (rel_table) the attention gathers by distance with a zero sink, the sigmoid output gate, the
// post conv, the block residual rule on the PHYSICAL block, and the fused gate/up MoE with situ, the per-expert
// scales and the router bias. `from` >= 0 resumes at that slot's MoE combine: the device state of the aborted
// forward is intact, only the row counts below are replayed on the host.
bool GpuModel::forward_inkling2(int32_t gq, int64_t from, std::string& err) {
    const ModelConfig& c = m_.cfg;
    Backend& g = *gpu_;
    const size_t ne = c.n_embd, nh = c.n_head, nkv = c.n_head_kv, hd = c.head_dim, kv = nkv * hd;
    const size_t nc = c.n_shortconv, nrel = c.n_rel, next1 = c.n_rel_extent + 1, rank = c.loop_gate_rank;
    const size_t core_start = c.loop_start, core_end = c.loop_end, n_core = core_end - core_start;
    const bool looped = c.is_looped();
    const size_t blk = c.n_attn_res_block, ke = std::max<size_t>(c.n_expert_used, 1);
    const uint64_t x = x_.addr(), step = step_buf_.addr(), ones = ones_.addr(), sinkz = sinkz_.addr();
    const uint64_t bks = buf("bks"), bkl = buf("bkl"), mat = buf("mat"), mkv = buf("mkv"),
                   gat = buf("gat"), g64 = buf("g64"), cg = buf("cg"), cf = buf("cf"), mixb = buf("mix"),
                   h = buf("h"), hq = buf("hq"), qb = buf("q"), kb = buf("kraw"), kout = buf("k"), vout = buf("v"),
                   rel = buf("rel"), snk = buf("sink"), kpj = buf("kpj"), vpj = buf("vpj"), tbl = buf("tbl"),
                   att = buf("att"), attp = buf("attp"), qg = buf("qg"), rl = buf("rl"), wts = buf("wts"),
                   wh = buf("wh"), gu = buf("gu"), ffq = buf("ffq"), ye = buf("ye"), moe = buf("moe"), ao = buf("ao");
    const uint64_t slots = slots_.addr(), ids = ids_.addr(), logits = logits_.addr();
    const float rfloor = 1.0f / (float)c.n_log_scale_floor;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float conv_scale = 1.0f / std::sqrt((float)nc + 1.0f);
    // STRATA_INK_XQ=1: the int8-quantized activation path (the llama.cpp CUDA mmvq analogue) for the inkling2
    // router, routed experts and head; default 0 uses the exact f32 activations.
    const bool ink_xq = env_flag("STRATA_INK_XQ");

    // The loop state (the block row counts of the reference's `blocks` vector), replayed on the host up to the
    // resume entry: the capture at the pass-0 core start, the push at every attn_res boundary, and the reset of the
    // later passes (bank = the previous pass's rows past its first push, nb back to the prelude's).
    size_t nb = 0, prelude_nb = 0, bank_row = 0, bank_nb = 0;
    const size_t from_l = from >= 0 ? (size_t)from : 0;
    for (size_t il = 0; il < from_l; il++) {
        const size_t ip = looped ? loop_layer_of(c, il) : il;
        const int pass = ip >= core_start && ip < core_end ? (int)((il - core_start) / n_core) : 0;
        if (looped && ip == core_start && pass > 0) {
            bank_row = prelude_nb + 1;
            bank_nb = nb - prelude_nb - 1;
            nb = prelude_nb;
        }
        if (ip % blk == 0) {
            if (looped && ip == core_start && pass == 0) prelude_nb = nb;
            nb++;
        }
    }

    // The embedding norm (tok_norm), in place: the reference normalizes the raw embeddings before layer 0.
    // STRATA_INK_DEBUG: capture-safe device-to-device copies of slot 0's stages into ink_dbg_ (named rows, full
    // vectors); run_token prints the rms/heads after the graph.
    const bool ink_dbg = env_flag("STRATA_INK_DEBUG");
    Buf mo_dbg = ink_dbg_;
    // A 2304-float row of the debug grid. A stage wider than 2304 floats must be given a start row whose band is
    // large enough and free for its whole width (the slot-0 stages use rows 33-49; see the dump calls), otherwise
    // its dump runs into the next stage's row.
    auto dump_row = [&](const Buf& b, size_t row, size_t floats) {
        if (!ink_dbg || 4 * (row * 2304 + floats) > mo_dbg.len()) return true;
        return g.dtod(b, 0, mo_dbg, 4 * row * 2304, 4 * floats, err);
    };
    if (ink_dbg) {  // sentinel-fill the debug buffer: row r gets the value r (so unwritten rows read back as ints)
        std::vector<float> sents(mo_dbg.len() / 4);
        for (size_t i = 0; i < sents.size(); i++) sents[i] = (float)(i / 2304) + 0.5f;
        if (!put(g, sents.data(), sents.size(), mo_dbg, 0, err)) return false;
    }
    if (ink_dbg) dump_row(x_, 0, ne);  // the raw embedding (pre-tok_norm)
    if (from_l == 0 && !g.rms_norm(x, ne, tok_norm_.ptr, x, ne, ne, 1, c.eps, err)) return false;
    if (ink_dbg) dump_row(x_, 1, ne);  // the post-tok_norm embedding

    for (size_t il = from_l; il < c.n_layer; il++) {
        const GpuLayer& l = layers_[il];
        const size_t ip = looped ? loop_layer_of(c, il) : il;
        const bool is_swa = il < c.swa.size() && c.swa[il];
        const int pass = ip >= core_start && ip < core_end ? (int)((il - core_start) / n_core) : 0;
        const bool resume = from >= 0 && (int64_t)il == from;
        // The shortconv state regions: the channel offset o of a conv keeps its (nc - 1) inputs at float 3 * o.
        const uint64_t cs = l.ink.conv_state.addr();
        const uint64_t kstate = cs, vstate = cs + 12 * kv, astate = cs + 24 * kv, fstate = cs + 12 * (2 * kv + ne);
        // The loop-state mechanics at a core start (PR #389's graph): pass 0 captures the prelude (the block
        // count and the residual) — every later pass reads the PREVIOUS pass's blocks past its first push (the
        // bank, snapshotted to bnk so the pass's own pushes cannot overwrite it) and the previous pass's last
        // output (bkl), then restarts from the prelude output. A resume skips it: the aborted attempt already
        // applied the reset, and the device state is intact.
        if (looped && ip == core_start && !resume) {
            if (pass == 0) {
                prelude_nb = nb;
                if (!g.dtod(x_, 0, bufs_.at("pre"), 0, 4 * ne, err)) return false;
            } else {
                bank_row = prelude_nb + 1;
                bank_nb = nb - prelude_nb - 1;
                if (ink_dbg)
                    std::fprintf(stderr, "[ink] slot %zu reset: nb %zu prelude_nb %zu bank_row %zu bank_nb %zu\n", il, nb,
                                 prelude_nb, bank_row, bank_nb);
                if (!g.dtod(bufs_.at("bks"), 4 * bank_row * ne, bufs_.at("bnk"), 0, 4 * bank_nb * ne, err))
                    return false;
                if (!g.dtod(x_, 0, bufs_.at("bkl"), 0, 4 * ne, err)) return false;
                if (!g.dtod(bufs_.at("pre"), 0, x_, 0, 4 * ne, err)) return false;
                nb = prelude_nb;
            }
        }
        // The cross-pass messages: two depth mixes over the bank rows and bkl with this block's two query rows;
        // m_attn's columns are the Q and FFN messages, m_kv's the K and V ones.
        uint64_t msgp[4] = {0, 0, 0, 0};
        if (looped && pass > 0 && ip >= core_start && ip < core_end) {
            const size_t row = 2 * (ip - core_start);
            if (!g.mix_dec(bufs_.at("bnk").addr(), (int)bank_nb, bkl, loop_res_query_.ptr + 4 * row * ne, 2, mat, ne, err))
                return false;
            if (!g.mix_dec(bufs_.at("bnk").addr(), (int)bank_nb, bkl, loop_kv_res_query_.ptr + 4 * row * ne, 2, mkv, ne, err))
                return false;
            msgp[0] = mat;              // Q
            msgp[1] = mkv;              // K
            msgp[2] = mkv + 4 * ne;     // V   (BYTE offset: mix_dec packs its rows ne floats = 4*ne bytes apart)
            msgp[3] = mat + 4 * ne;     // FFN (ditto)
            if (ink_dbg && il == 18) {  // the first loop slot's message stages into rows 16..25 (bks, bkl, bnk)
                if (!g.dtod(bufs_.at("bks"), 0, mo_dbg, 4 * 16 * 2304, 4 * 4 * ne, err) ||
                    !g.dtod(bufs_.at("bkl"), 0, mo_dbg, 4 * 20 * 2304, 4 * ne, err) ||
                    !g.dtod(bufs_.at("bnk"), 0, mo_dbg, 4 * 25 * 2304, 4 * bank_nb * ne, err) ||
                    !g.dtod(bufs_.at("mat"), 0, mo_dbg, 4 * 21 * 2304, 4 * 2 * ne, err) ||
                    !g.dtod(bufs_.at("mkv"), 0, mo_dbg, 4 * 23 * 2304, 4 * 2 * ne, err))
                    return false;
            }
        }
        if (!resume) {
            // The attention inputs: the residual mix (attn_res_query) of the block boundaries and cur; the SWA
            // layers mix one query (all three inputs read it), the full-attention ones one per Q/K/V. The inputs
            // land in DISTINCT rows of inpb: the controls and the input norms write in place, and the reference
            // gives each of Q/K/V its own tensor (the residual stream and the bank rows must not be clobbered).
            const uint64_t inpb = buf("inpb");
            if (l.ink.res_q.ptr != 0) {
                const size_t nq = is_swa ? 1 : 3;
                if (!g.mix_dec(bks, (int)nb, x, l.ink.res_q.ptr, nq, mixb, ne, err)) return false;
                if (nq == 1) {
                    // the SWA mix produces one row; the reference feeds that same vector to Q, K and V
                    if (!g.dtod(bufs_.at("mix"), 0, bufs_.at("inpb"), 0, 4 * ne, err) ||
                        !g.dtod(bufs_.at("mix"), 0, bufs_.at("inpb"), 4 * ne, 4 * ne, err) ||
                        !g.dtod(bufs_.at("mix"), 0, bufs_.at("inpb"), 8 * ne, 4 * ne, err))
                        return false;
                } else {
                    for (size_t j = 0; j < 3; j++)
                        if (!g.dtod(bufs_.at("mix"), 4 * j * ne, bufs_.at("inpb"), 4 * j * ne, 4 * ne, err))
                            return false;
                }
            } else {
                // no res query (block 0): Q, K and V all read cur; x_ holds ONE row, so replicate it
                for (size_t j = 0; j < 3; j++)
                    if (!g.dtod(x_, 0, bufs_.at("inpb"), 4 * j * ne, 4 * ne, err)) return false;
            }
            uint64_t inp[3] = {inpb, inpb + 4 * ne, inpb + 8 * ne};
            if (ink_dbg && il == 18) {  // the pre-control Q/K/V inputs into rows 30..32
                if (!g.dtod(bufs_.at("inpb"), 0, mo_dbg, 4 * 30 * 2304, 4 * 3 * ne, err)) return false;
            }
            // The messages enter before the input norms: control() on each of Q, K and V (their gates, downs, gains
            // and forgets are independent, so each stage batches the three).
            if (msgp[0] != 0) {
                if (!batch(g,
                           [&](std::string& e) {
                               // ALL offsets here are BYTES (Buf addresses): one packed gate is 2*ne floats =
                               // 8*ne bytes, a down output rank floats = 4*rank bytes, a gain/forget ne floats
                               // = 4*ne bytes. Each buffer holds three controller slots (gat 6*ne, g64 3*rank,
                               // cg/cf 3*ne floats).
                               return g.gate_pack(inp[0], msgp[0], gat, ne, 1e-6f, e) &&
                                      g.gate_pack(inp[1], msgp[1], gat + 8 * ne, ne, 1e-6f, e) &&
                                      g.gate_pack(inp[2], msgp[2], gat + 16 * ne, ne, 1e-6f, e);
                           },
                           err))
                    return false;
                if (!batch(g,
                           [&](std::string& e) {
                               return g.gemv(l.ink.loop_down[0], gat, g64, e) &&
                                      g.gemv(l.ink.loop_down[1], gat + 8 * ne, g64 + 4 * rank, e) &&
                                      g.gemv(l.ink.loop_down[2], gat + 16 * ne, g64 + 8 * rank, e);
                           },
                           err))
                    return false;
                if (!batch(g,
                           [&](std::string& e) {
                               return g.gemv(l.ink.loop_gain[0], g64, cg, e) &&
                                      g.gemv(l.ink.loop_gain[1], g64 + 4 * rank, cg + 4 * ne, e) &&
                                      g.gemv(l.ink.loop_gain[2], g64 + 8 * rank, cg + 8 * ne, e) &&
                                      g.gemv(l.ink.loop_forget[0], g64, cf, e) &&
                                      g.gemv(l.ink.loop_forget[1], g64 + 4 * rank, cf + 4 * ne, e) &&
                                      g.gemv(l.ink.loop_forget[2], g64 + 8 * rank, cf + 8 * ne, e);
                           },
                           err))
                    return false;
                if (!batch(g,
                           [&](std::string& e) {
                               return g.control_mix(inp[0], msgp[0], cg, cf, inp[0], ne, e) &&
                                      g.control_mix(inp[1], msgp[1], cg + 4 * ne, cf + 4 * ne, inp[1], ne, e) &&
                                      g.control_mix(inp[2], msgp[2], cg + 8 * ne, cf + 8 * ne, inp[2], ne, e);
                           },
                           err))
                    return false;
            }
            if (ink_dbg && il == 18 && msgp[0] != 0) {
                // controller 0's stages: gat (2*ne, rows 68-69), g64 (3*rank, row 70), cg / cf of ctrl 0 (rows 71/72)
                if (!g.dtod(bufs_.at("gat"), 0, mo_dbg, 4 * 68 * 2304, 4 * 2 * ne, err) ||
                    !g.dtod(bufs_.at("g64"), 0, mo_dbg, 4 * 70 * 2304, 4 * 3 * rank, err) ||
                    !g.dtod(bufs_.at("cg"), 0, mo_dbg, 4 * 71 * 2304, 4 * ne, err) ||
                    !g.dtod(bufs_.at("cf"), 0, mo_dbg, 4 * 72 * 2304, 4 * ne, err))
                    return false;
            }
            if (ink_dbg && il == 18) {  // the post-control Q/K/V inputs (before the norms rewrite them) into rows 26..28
                if (!g.dtod(bufs_.at("inpb"), 0, mo_dbg, 4 * 26 * 2304, 4 * 3 * ne, err)) return false;
            }
            // The input norms: the SWA norm on one shared input (or on each of the three when messages made them
            // differ), else the three separate input norms of the full-attention blocks.
            if (is_swa) {
                if (msgp[0] != 0) {
                    if (!batch(g,
                               [&](std::string& e) {
                                   return g.rms_norm(inp[0], ne, l.attn_norm.ptr, inp[0], ne, ne, 1, c.eps, e) &&
                                          g.rms_norm(inp[1], ne, l.attn_norm.ptr, inp[1], ne, ne, 1, c.eps, e) &&
                                          g.rms_norm(inp[2], ne, l.attn_norm.ptr, inp[2], ne, ne, 1, c.eps, e);
                               },
                               err))
                        return false;
                } else if (!batch(g,
                                  [&](std::string& e) {
                                      // the three rows hold identical copies; norming all three with the same
                                      // gamma gives the reference's shared inputs
                                      return g.rms_norm(inp[0], ne, l.attn_norm.ptr, inp[0], ne, ne, 1, c.eps, e) &&
                                             g.rms_norm(inp[1], ne, l.attn_norm.ptr, inp[1], ne, ne, 1, c.eps, e) &&
                                             g.rms_norm(inp[2], ne, l.attn_norm.ptr, inp[2], ne, ne, 1, c.eps, e);
                                  },
                                  err)) {
                    return false;
                }
            } else if (!batch(g,
                              [&](std::string& e) {
                                  return g.rms_norm(inp[0], ne, l.ink.q_inp_norm.ptr, inp[0], ne, ne, 1, c.eps, e) &&
                                         g.rms_norm(inp[1], ne, l.ink.k_inp_norm.ptr, inp[1], ne, ne, 1, c.eps, e) &&
                                         g.rms_norm(inp[2], ne, l.ink.v_inp_norm.ptr, inp[2], ne, ne, 1, c.eps, e);
                              },
                              err)) {
                return false;
            }
            if (ink_dbg && il == 0) dump_row(x_, 2, ne);  // the normed input (the norms are in place)
            // Q, K, V, the relative projection, the sink projection, and (full-attention blocks) the conv taps.
            if (!batch(g,
                       [&](std::string& e) {
                           return g.gemv(l.q, inp[0], qb, e) && g.gemv(l.k, inp[1], kb, e) && g.gemv(l.v, inp[2], vout, e) &&
                                  g.gemv(l.ink.rel, inp[0], rel, e) && g.gemv(l.ink.sink_proj, inp[0], snk, e) &&
                                  (is_swa || (g.gemv(l.ink.k_conv_proj, inp[1], kpj, e) &&
                                              g.gemv(l.ink.v_conv_proj, inp[2], vpj, e)));
                       },
                       err))
                return false;

            if (ink_dbg && il == 0) {
                dump_row(bufs_.at("q"), 48, nh * hd);  // 3072 floats: rows 48-49 (the band's start is below)
                dump_row(bufs_.at("kraw"), 4, kv);
                dump_row(bufs_.at("v"), 5, kv);
                dump_row(bufs_.at("rel"), 6, nrel * nh);
                dump_row(bufs_.at("sink"), 7, nh);
            }
            // The shortconv of K and V (dynamic taps on the full-attention blocks), the clips of Q and R (with the
            // log-scaled positions of the full-attention blocks), then the clip of K and the V norm.
            const float cscale = 1.0f / std::sqrt((float)(is_swa ? 1 : 2) * (float)nc + 1.0f);
            if (!batch(g,
                       [&](std::string& e) {
                           return g.conv_ink(kb, kstate, l.ink.k_conv.ptr, is_swa ? 0 : kpj, kout, kv, nc, hd, cscale, e) &&
                                  g.conv_ink(vout, vstate, l.ink.v_conv.ptr, is_swa ? 0 : vpj, vout, kv, nc, hd, cscale, e) &&
                                  g.clip_rows(qb, l.ink.q_clip.ptr, hd, nh, c.f_qk_clip, is_swa ? 0.0f : c.f_log_scale_alpha,
                                              rfloor, step, e) &&
                                  g.clip_rows(rel, l.ink.rel_clip.ptr, nrel, nh, c.f_rel_clip,
                                              is_swa ? 0.0f : c.f_log_scale_alpha, rfloor, step, e);
                       },
                       err))
                return false;
            if (!batch(g,
                       [&](std::string& e) {
                           return g.clip_rows(kout, 0, hd, nkv, c.f_qk_clip, 0.0f, rfloor, step, e) &&
                                  g.rms_norm(vout, hd, ones, vout, hd, hd, nkv, c.eps, e);
                       },
                       err))
                return false;
            if (!g.kv_store(kout, vout, l.kc.addr(), l.vc.addr(), kv, 1, step, hd, err)) return false;

            if (ink_dbg && il == 0) {
                dump_row(bufs_.at("k"), 8, kv);
                dump_row(bufs_.at("rel"), 9, nrel * nh);
            }
            // The rel-bias table of this token (with the sink logit folded in), and the output gate projection.
            if (!batch(g,
                       [&](std::string& e) {
                           return g.rel_table(l.ink.rel_bias.ptr, rel, snk, tbl, nrel, next1, nh, e) &&
                                  g.gemv(l.dn_gate, inp[0], qg, e);
                       },
                       err))
                return false;
            // The attention: the gathered bias enters after the scale, the zero sink adds exp(0) = 1 to the
            // denominator (the sink logit is already inside every gathered entry). No output quantization: the
            // sigmoid gate scales the attention output before the wo projection. attn_decode_gq (not _q) carries
            // the bias/extent pair the inkling2 table needs.
            uint64_t aq = 0;
            if (!g.attn_decode_gq(qb, l.kc.addr(), l.vc.addr(), 0, att, attp, step, nh, nkv, hd, attn_scale, gq, 1,
                                  sinkz, is_swa ? c.n_swa : 0, tbl, c.n_rel_extent, err))
                return false;
            if (!g.sig_mul(att, qg, nh * hd, err)) return false;
            if (ink_dbg && il == 0) {
                dump_row(bufs_.at("tbl"), 33, next1 * nh);  // 24600 floats: rows 33-43 (11 rows)
                dump_row(bufs_.at("att"), 44, nh * hd);     // 3072 floats: rows 44-45
                dump_row(bufs_.at("qg"), 46, nh * hd);      // 3072 floats: rows 46-47
            }
            if (!g.gemv(l.o, att, ao, err)) return false;
            if (ink_dbg && il == 0) dump_row(bufs_.at("ao"), 13, ne);
            // The attention output's post conv (static), then the block residual rule on the PHYSICAL block: a
            // boundary pushes inpSA into the bank and starts a new residual stream (the attention output), an inner
            // one adds the attention output to it.
            if (!g.conv_ink(ao, astate, l.ink.post_conv.ptr, 0, ao, ne, nc, 0, conv_scale, err)) return false;
            if (ip % blk == 0) {
                if (!g.dtod(x_, 0, bufs_.at("bks"), 4 * nb * ne, 4 * ne, err)) return false;
                nb++;
                if (!g.dtod(bufs_.at("ao"), 0, x_, 0, 4 * ne, err)) return false;
            } else if (!g.add(x, ao, ne, err)) {
                return false;
            }

            // The FFN: the depth mix over the bank rows and the residual, the FFN message's control, the norm, the
            // fused gate/up MoE with situ, the per-expert scales and the router bias, then the post conv and the
            // residual add.
            if (!g.mix_dec(bks, (int)nb, x, l.ink.ffn_res_q.ptr, 1, mixb, ne, err)) return false;
            if (msgp[3] != 0) {
                if (!g.gate_pack(mixb, msgp[3], gat, ne, 1e-6f, err)) return false;
                if (!g.gemv(l.ink.loop_down[3], gat, g64, err)) return false;
                if (!batch(g,
                           [&](std::string& e) {
                               return g.gemv(l.ink.loop_gain[3], g64, cg, e) && g.gemv(l.ink.loop_forget[3], g64, cf, e);
                           },
                           err))
                    return false;
                if (!g.control_mix(mixb, msgp[3], cg, cf, mixb, ne, err)) return false;
            }
            if (!g.rms_norm_q(mixb, ne, l.ink.ffn_norm.ptr, h, ne, ne, 1, c.eps, hq, 0, err)) return false;
            if (ink_dump_.len() > 0 && il == 1) {
                // STRATA_INK_DUMP: h and the FFN mix input captured BEFORE the MoE, past the record's fields.
                if (!g.dtod(bufs_.at("h"), 0, ink_dump_, 8 * ne + 352, 4 * ne, err) ||
                    !g.dtod(bufs_.at("mix"), 0, ink_dump_, 8 * ne + 352 + 4 * ne, 4 * ne, err))
                    return false;
            }
        }
        // The MoE (Ffn::Moe with the fused gate/up): the routed experts in VRAM here, the missed ones on the CPU
        // worker (the mailbox is per slot; each pass routes independently), the per-expert scales of the pool slots.
        if (!g.memset(bufs_.at("moe"), err)) return false;  // the combine adds into it
        const size_t phys = looped ? loop_layer_of(c, il) : il;
        if (!cache_) {
            err = "MoE layer without an expert cache";
            return false;
        }
        const LayerExperts* lx = cache_->layer(phys);
        if (!lx) {
            err = "MoE layer " + std::to_string(il) + " has no expert cache entry";
            return false;
        }
        const bool fast = cache_->fully_resident() && g.moe_mailbox_optional();
        if (!fast && !worker_) {
            err = "routed experts are not all resident in VRAM and the model has no CPU expert worker";
            return false;
        }
        const uint64_t mb = fast ? 0 : worker_->dev() + 4 * il * layout_.stride;
        const auto [base, stride] = cache_->base(*lx);
        const ExpertMat& mg = lx->mats[0];
        const ExpertMat& md = lx->mats[1];
        const size_t nff = md.cols;
        uint64_t dq = 0;
        if (!resume) {
            if (!fast) mailbox_writes++;
            // f32 activation (exact routing) or the quantized hq (STRATA_INK_XQ, the llama.cpp mmvq analogue)
            if (!(ink_xq ? g.gemvq(l.router, h, ne, hq, rl, c.n_expert, 1, err) : g.gemv(l.router, h, rl, err)))
                return false;
            if (ink_dbg && il == 1) {  // slot 1's router logits (row 66) and the normed FFN input h (row 67)
                if (!g.dtod(bufs_.at("rl"), 0, mo_dbg, 4 * 66 * 2304, 4 * c.n_expert, err) ||
                    !g.dtod(bufs_.at("h"), 0, mo_dbg, 4 * 67 * 2304, 4 * ne, err))
                    return false;
            }
            // The router bias (exp_probs_b) applies to the selection only; the weights normalize with the additive
            // epsilon (weights_norm_eps) before the weights scale.
            if (!g.moe_route(rl, c.n_expert, ke, c.expert_weights_scale, lx->table_dev.addr(), ids, wts, slots, wh, h, mb,
                             ne, layout_.h, step, 1, 0, true, l.ink.exp_probs_b.ptr, 1e-20f, err))
                return false;
            // The fused gate/up of the k routed experts (rows 0..n_ff-1 gate, the rest up per slot), the situ
            // activation with the slot's gate/up scale, quantized for the down GEMV.
            if (!g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off, stride, slots, ke, h, ne, ke, gu, std::nullopt,
                            ink_xq ? hq : 0, err))
                return false;
            if (!g.situ_q(gu, nff, ke, l.ink.gu_scale.ptr, ids, c.situ_beta, c.situ_linear_beta, ffq, dq, err))
                return false;
        } else if (md.ty != Ty::F32 && md.ty != Ty::BF16) {
            dq = ffq;  // resuming: the gate/up GEMVs and the situ are already in the buffers
        }
        // The down GEMV reads slot t's situ row (the fused row stride), then the weighted combine into the zeroed
        // MoE target (the down scale of a resident slot folds into its weight).
        if (!g.moe_gemv(md.ty, md.rows, md.cols, base + md.off, stride, slots, ke, gu, 2 * nff, 1, ye, std::nullopt,
                        ink_xq ? dq : 0, err))
            return false;
        if (!g.moe_combine(moe, ye, wh, ke, 0, 0, mb, layout_.y, step, slots, ne, 1, l.ink.dn_scale.ptr, ids, err))
            return false;
        if (ink_dump_.len() > 0 && il == 1) {
            // STRATA_INK_DUMP: the slot-1 MoE record into the dedicated buffer (h, moe, w, ids, slots), read by
            // run_token after the graph (later layers would clobber the live buffers).
            if (!g.dtod(bufs_.at("h"), 0, ink_dump_, 0, 4 * ne, err) ||
                !g.dtod(bufs_.at("moe"), 0, ink_dump_, 4 * ne, 4 * ne, err) ||
                !g.dtod(bufs_.at("ye"), 0, ink_dump_, 8 * ne, 32 * ke, err) ||
                !g.dtod(bufs_.at("wh"), 0, ink_dump_, 8 * ne + 32 * ke, 32, err) ||
                !g.dtod(ids_, 0, ink_dump_, 8 * ne + 32 * ke + 32, 32, err) ||
                !g.dtod(slots_, 0, ink_dump_, 8 * ne + 32 * ke + 64, 32, err))
                return false;
        }
        if (ink_dbg) {  // the per-slot residual rms into the row-60 band (8 floats per slot)
            if (!g.dtod(x_, 0, mo_dbg, 4 * (60 * 2304 + 8 * (il % 48)), 32, err)) return false;
            // the routed expert ids (int32 bits) and their combine weights into the row-64 / row-65 bands
            if (!g.dtod(ids_, 0, mo_dbg, 4 * (64 * 2304 + 8 * (il % 48)), 4 * ke, err) ||
                !g.dtod(bufs_.at("wts"), 0, mo_dbg, 4 * (65 * 2304 + 8 * (il % 48)), 4 * ke, err))
                return false;
        }
        if (ink_dbg && il == 1 && mailbox_.len() > 0) {
            // the CPU answer of this slot's mailbox (the words moe_combine reads) into rows 61..62; also
            // the h the mailbox carried (the CPU's input), row 62's tail is unused; wh/slots/ids into row 63
            if (!g.dtod(mailbox_, 4 * ((size_t)il * layout_.stride + layout_.y), mo_dbg, 4 * 61 * 2304, 4 * ne, err) ||
                !g.dtod(mailbox_, 4 * ((size_t)il * layout_.stride + layout_.h), mo_dbg, 4 * 62 * 2304, 4 * ne, err) ||
                !g.dtod(bufs_.at("wh"), 0, mo_dbg, 4 * 63 * 2304, 128, err) ||
                !g.dtod(slots_, 0, mo_dbg, 4 * 63 * 2304 + 128, 128, err) ||
                !g.dtod(ids_, 0, mo_dbg, 4 * 63 * 2304 + 256, 128, err) ||
                !g.dtod(bufs_.at("h"), 0, bufs_.at("ye"), 0, 4 * ne, err))
                return false;
        }
        // The FFN output's post conv (static) over the raw MoE output, then the residual add.
        if (!g.conv_ink(moe, fstate, l.ink.ffn_post_conv.ptr, 0, moe, ne, nc, 0, conv_scale, err)) return false;
        if (!g.add(x, moe, ne, err)) return false;

    }
    // The output head: the depth mix over the bank rows and the residual, the output norm, the (padded) lm_head, the
    // softcap.
    if (!g.mix_dec(bks, (int)nb, x, output_res_query_.ptr, 1, mixb, ne, err)) return false;
    if (!g.rms_norm_q(mixb, ne, output_norm_.ptr, h, ne, ne, 1, c.eps, hq, 0, err)) return false;
    if (ink_dbg) {  // the head stages into rows 50..51 (the mix, the normed head input)
        if (!g.dtod(bufs_.at("mix"), 0, mo_dbg, 4 * 50 * 2304, 4 * ne, err) ||
            !g.dtod(bufs_.at("h"), 0, mo_dbg, 4 * 51 * 2304, 4 * ne, err))
            return false;
    }
    // exact f32 activation by default (the reference's Wo @ h); STRATA_INK_XQ=1 uses the int8-quantized hq
    if (!(ink_xq ? g.gemvq(output_, h, ne, hq, logits, c.n_vocab, 1, err) : g.gemv(output_, h, logits, err)))
        return false;
    return g.softcap(logits, c.n_vocab, c.f_final_softcap, err);
}

// StrataFreet addition (INKLING2_NOTES.md): the inkling2 chunked prefill (W3). The slot graph of forward_inkling2
// over t tokens (1..16; the per-token mailbox regions are TMAX = 17 rows). The per-token work runs token by token
// through the same launch sequence, while the routed-expert pass is batched over the chunk:
//   one router gemm_dense over t rows, one moe_route with t rows (the CPU worker answers all t tokens' misses in
//   one experts_multi pass), one fused gate/up moe_gemv with ke slots per token (xg = ke groups the slots by token),
//   one situ_q over t*ke rows, one down moe_gemv and one moe_combine with t tokens.
// The per-token state that spans slots (x, bks, bnk, bkl, pre, h, moe) lives in per-token regions of the decode
// buffers (bks keeps 4 rows a token: 64 of the 68, bnk 2 rows a token: 32 of 34); the slot-local scratch (messages,
// controller gates, attention rows) is reused token by token. The head runs for the chunk's last token.
bool GpuModel::forward_inkling2_chunk(const uint32_t* toks, size_t t, std::string& err) {
    const ModelConfig& c = m_.cfg;
    Backend& g = *gpu_;
    if (t == 0 || t > 16) {
        err = "inkling2 chunk: 1..16 tokens (got " + std::to_string(t) + ")";
        return false;
    }
    if (hcn_ != 1) {
        err = "inkling2 chunk: hyper-connections are not supported";
        return false;
    }
    const size_t ne = c.n_embd, nh = c.n_head, nkv = c.n_head_kv, hd = c.head_dim, kv = nkv * hd;
    const size_t nc = c.n_shortconv, nrel = c.n_rel, next1 = c.n_rel_extent + 1, rank = c.loop_gate_rank;
    const size_t core_start = c.loop_start, core_end = c.loop_end, n_core = core_end - core_start;
    const bool looped = c.is_looped();
    const size_t blk = c.n_attn_res_block, ke = std::max<size_t>(c.n_expert_used, 1);
    const size_t nff = c.n_ff_exp;
    const uint64_t x = x_.addr(), stepb = step_buf_.addr(), ones = ones_.addr(), sinkz = sinkz_.addr();
    const uint64_t bks = buf("bks"), bkl = buf("bkl"), bnk = buf("bnk"),
                   mat = buf("mat"), mkv = buf("mkv"), gat = buf("gat"), g64 = buf("g64"), cg = buf("cg"),
                   cf = buf("cf"), mixb = buf("mix"), h = buf("h"), hq = buf("hq"), qb = buf("q"), kb = buf("kraw"),
                   kout = buf("k"), vout = buf("v"), rel = buf("rel"), snk = buf("sink"), kpj = buf("kpj"),
                   vpj = buf("vpj"), tbl = buf("tbl"), att = buf("att"), attp = buf("attp"), qg = buf("qg"),
                   rl = buf("rl"), wts = buf("wts"), wh = buf("wh"), gu = buf("gu16"), guq = buf("ffq16"),
                   ye = buf("ye"), moe = buf("moe"), ao = buf("ao"), inpb = buf("inpb");
    const uint64_t slots = slots_.addr(), ids = ids_.addr(), logits = logits_.addr();
    const float rfloor = 1.0f / (float)c.n_log_scale_floor;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float conv_scale = 1.0f / std::sqrt((float)nc + 1.0f);
    const bool ink_xq = env_flag("STRATA_INK_XQ");
    if (std::getenv("STRATA_INK_CHUNK_TRACE")) {
        std::fprintf(stderr, "[chunk] t=%zu pos=%zu x=%llx h=%llx bks=%llx inpb=%llx v=%llx ones=%llx tok_norm=%llx "
                             "attn_norm=%llx q_inp=%llx ffn_norm=%llx out_norm=%llx gu16=%llx ffq16=%llx ne=%zu\n",
                     t, pos_, (unsigned long long)x_.addr(), (unsigned long long)buf("h"),
                     (unsigned long long)buf("bks"), (unsigned long long)buf("inpb"), (unsigned long long)buf("v"),
                     (unsigned long long)ones_.addr(), (unsigned long long)tok_norm_.ptr,
                     layers_.empty() ? 0ULL : (unsigned long long)layers_[0].attn_norm.ptr,
                     layers_.empty() ? 0ULL : (unsigned long long)layers_[0].ink.q_inp_norm.ptr,
                     layers_.empty() ? 0ULL : (unsigned long long)layers_[0].ink.ffn_norm.ptr,
                     (unsigned long long)output_norm_.ptr, (unsigned long long)buf("gu16"),
                     (unsigned long long)buf("ffq16"), ne);
    }
    const bool trace_ck = std::getenv("STRATA_INK_CHUNK_TRACE") != nullptr;
    // Per-token regions (BYTE offsets: Buf addresses).
    auto xat = [&](size_t i) { return x + 4 * ne * i; };
    auto bksat = [&](size_t i) { return bks + 16 * ne * i; };  // 4 rows of ne floats a token
    auto bnkat = [&](size_t i) { return bnk + 8 * ne * i; };   // 2 rows
    auto rowat = [&](uint64_t base, size_t i) { return base + 4 * ne * i; };
    auto stepat = [&](size_t i) { return stepb + 8 * i; };
    // Buf views for the copies (put / dtod / rms_norm take Bufs, the kernels take raw addresses).
    auto xbuf = [&](size_t i) { return x_.slice(4 * ne * i, 4 * ne); };
    auto bksbuf = [&](size_t i) { return bufs_.at("bks").slice(16 * ne * i, 16 * ne); };
    auto bnkbuf = [&](size_t i) { return bufs_.at("bnk").slice(8 * ne * i, 8 * ne); };
    auto rowbuf = [&](const char* name, size_t i) { return bufs_.at(name).slice(4 * ne * i, 4 * ne); };

    // The embeddings of the chunk, then tok_norm over the whole chunk in place.
    {
        std::vector<float> row;
        for (size_t i = 0; i < t; i++) {
            if (!m_.embed(toks[i], row, err)) return false;
            if (!put(g, row.data(), row.size(), xbuf(i), 0, err)) return false;
        }
        if (!g.rms_norm(x_.addr(), ne, tok_norm_.ptr, x_.addr(), ne, ne, t, c.eps, err)) return false;
        if (trace_ck) std::fprintf(stderr, "[chunk] embed norm ok\n");
    }
    // One worker sequence for the whole chunk: every layer's mailbox request carries all t tokens (mb[1] = t) and
    // the worker answers them with one experts_multi pass over the missed experts.
    const bool fast = expert_fast_path();
    const int32_t seq = (worker_ && !fast) ? static_cast<int32_t>(worker_->begin()) : 0;
    std::vector<int32_t> sb(2 * t);
    for (size_t i = 0; i < t; i++) {
        sb[2 * i] = static_cast<int32_t>(pos_ + i);
        sb[2 * i + 1] = seq;
    }
    if (!put(g, sb.data(), sb.size(), step_buf_, 0, err)) return false;

    // The chunk graph: the slot loop's host control flow is fixed by t (and the gq regime), so one captured graph per
    // (t, regime) replays the two thousand launches of a chunk in one launch. A chunk that straddles the
    // ATT_GQA_MIN position (gq_min flips from INT32_MAX to 0 there) runs eagerly; that is once per prefill.
    const bool small_gq = pos_ + t - 1 < (size_t)ATT_GQA_MIN;
    const bool large_gq = pos_ >= (size_t)ATT_GQA_MIN;
    if (small_gq || large_gq) {
        const uint64_t key = (uint64_t(1) << 40) | uint64_t(t) | (small_gq ? 0 : (uint64_t(1) << 8));
        if (!gpu_->graph(key, [&](std::string& e) { return ink_chunk_body(t, e); }, err)) return false;
    } else if (!ink_chunk_body(t, err)) {
        return false;
    }
    pos_ += t;
    tokens += t;
    return after_experts(pos_, err);
}

// The captured body of forward_inkling2_chunk: the slot graph over the chunk's per-token regions and the batched
// routed-expert pass (the host control flow depends only on t and the layer indices, so a graph can replay it).
bool GpuModel::ink_chunk_body(size_t t, std::string& err) {
    const ModelConfig& c = m_.cfg;
    Backend& g = *gpu_;
    const bool fast = expert_fast_path();
    const size_t ne = c.n_embd, nh = c.n_head, nkv = c.n_head_kv, hd = c.head_dim, kv = nkv * hd;
    const size_t nc = c.n_shortconv, nrel = c.n_rel, next1 = c.n_rel_extent + 1, rank = c.loop_gate_rank;
    const size_t core_start = c.loop_start, core_end = c.loop_end, n_core = core_end - core_start;
    const bool looped = c.is_looped();
    const size_t blk = c.n_attn_res_block, ke = std::max<size_t>(c.n_expert_used, 1);
    const size_t nff = c.n_ff_exp;
    const uint64_t x = x_.addr(), stepb = step_buf_.addr(), ones = ones_.addr(), sinkz = sinkz_.addr();
    const uint64_t bks = buf("bks"), bkl = buf("bkl"), bnk = buf("bnk"),
                   mat = buf("mat"), mkv = buf("mkv"), gat = buf("gat"), g64 = buf("g64"), cg = buf("cg"),
                   cf = buf("cf"), mixb = buf("mix"), h = buf("h"), hq = buf("hq"), qb = buf("q"), kb = buf("kraw"),
                   kout = buf("k"), vout = buf("v"), rel = buf("rel"), snk = buf("sink"), kpj = buf("kpj"),
                   vpj = buf("vpj"), tbl = buf("tbl"), att = buf("att"), attp = buf("attp"), qg = buf("qg"),
                   rl = buf("rl"), wts = buf("wts"), wh = buf("wh"), gu = buf("gu16"), guq = buf("ffq16"),
                   ye = buf("ye"), moe = buf("moe"), ao = buf("ao"), inpb = buf("inpb");
    const uint64_t slots = slots_.addr(), ids = ids_.addr(), logits = logits_.addr();
    const float rfloor = 1.0f / (float)c.n_log_scale_floor;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float conv_scale = 1.0f / std::sqrt((float)nc + 1.0f);
    const bool ink_xq = env_flag("STRATA_INK_XQ");
    if (std::getenv("STRATA_INK_CHUNK_TRACE")) {
        std::fprintf(stderr, "[chunk] t=%zu pos=%zu x=%llx h=%llx bks=%llx inpb=%llx v=%llx ones=%llx tok_norm=%llx "
                             "attn_norm=%llx q_inp=%llx ffn_norm=%llx out_norm=%llx gu16=%llx ffq16=%llx ne=%zu\n",
                     t, pos_, (unsigned long long)x_.addr(), (unsigned long long)buf("h"),
                     (unsigned long long)buf("bks"), (unsigned long long)buf("inpb"), (unsigned long long)buf("v"),
                     (unsigned long long)ones_.addr(), (unsigned long long)tok_norm_.ptr,
                     layers_.empty() ? 0ULL : (unsigned long long)layers_[0].attn_norm.ptr,
                     layers_.empty() ? 0ULL : (unsigned long long)layers_[0].ink.q_inp_norm.ptr,
                     layers_.empty() ? 0ULL : (unsigned long long)layers_[0].ink.ffn_norm.ptr,
                     (unsigned long long)output_norm_.ptr, (unsigned long long)buf("gu16"),
                     (unsigned long long)buf("ffq16"), ne);
    }
    const bool trace_ck = std::getenv("STRATA_INK_CHUNK_TRACE") != nullptr;
    // Per-token regions (BYTE offsets: Buf addresses).
    auto xat = [&](size_t i) { return x + 4 * ne * i; };
    auto bksat = [&](size_t i) { return bks + 16 * ne * i; };  // 4 rows of ne floats a token
    auto bnkat = [&](size_t i) { return bnk + 8 * ne * i; };   // 2 rows
    auto rowat = [&](uint64_t base, size_t i) { return base + 4 * ne * i; };
    auto stepat = [&](size_t i) { return stepb + 8 * i; };
    // Buf views for the copies (put / dtod / rms_norm take Bufs, the kernels take raw addresses).
    auto xbuf = [&](size_t i) { return x_.slice(4 * ne * i, 4 * ne); };
    auto bksbuf = [&](size_t i) { return bufs_.at("bks").slice(16 * ne * i, 16 * ne); };
    auto bnkbuf = [&](size_t i) { return bufs_.at("bnk").slice(8 * ne * i, 8 * ne); };
    auto rowbuf = [&](const char* name, size_t i) { return bufs_.at(name).slice(4 * ne * i, 4 * ne); };

    // The loop state (the block row counts of the reference's `blocks` vector): the capture at the pass-0 core
    // start, the push at every attn_res boundary, and the reset of the later passes (bank = the previous pass's
    // rows past its first push, nb back to the prelude's). Routing-independent, so one state for the chunk.
    size_t nb = 0, prelude_nb = 0, bank_row = 0, bank_nb = 0;
    for (size_t il = 0; il < c.n_layer; il++) {
        const GpuLayer& l = layers_[il];
        const size_t ip = looped ? loop_layer_of(c, il) : il;
        const bool is_swa = il < c.swa.size() && c.swa[il];
        const int pass = ip >= core_start && ip < core_end ? (int)((il - core_start) / n_core) : 0;
        const uint64_t cs = l.ink.conv_state.addr();
        const uint64_t kstate = cs, vstate = cs + 12 * kv, astate = cs + 24 * kv, fstate = cs + 12 * (2 * kv + ne);
        // The cross-pass reset, per token: snapshot the bank rows past the prelude's first push, keep the previous
        // pass's last output as bkl, and restart from the prelude output.
        if (looped && ip == core_start) {
            if (pass == 0) {
                prelude_nb = nb;
                for (size_t i = 0; i < t; i++)
                    if (!g.dtod(xbuf(i), 0, rowbuf("pre", i), 0, 4 * ne, err)) return false;
            } else {
                bank_row = prelude_nb + 1;
                bank_nb = nb - prelude_nb - 1;
                for (size_t i = 0; i < t; i++) {
                    if (!g.dtod(bksbuf(i), 4 * bank_row * ne, bnkbuf(i), 0, 4 * bank_nb * ne, err)) return false;
                    if (!g.dtod(xbuf(i), 0, rowbuf("bkl", i), 0, 4 * ne, err)) return false;
                    if (!g.dtod(rowbuf("pre", i), 0, xbuf(i), 0, 4 * ne, err)) return false;
                }
                nb = prelude_nb;
            }
        }
        const bool boundary = ip % blk == 0;
        const size_t nb_ffn = nb + (boundary ? 1 : 0);  // the FFN mix reads the bank AFTER this slot's push
        // The per-token slot head: attention (with the message controls), the block residual rule, the FFN mix and
        // the norm into this token's row of h. The slot-local scratch is shared; tokens run in order (the conv
        // states advance along the sequence).
        for (size_t i = 0; i < t; i++) {
            const uint64_t xi = xat(i), bksi = bksat(i), stepi = stepat(i);
            uint64_t msgQ = 0, msgK = 0, msgV = 0, msgF = 0;
            if (looped && pass > 0 && ip >= core_start && ip < core_end) {
                const size_t row = 2 * (ip - core_start);
                if (!g.mix_dec(bnkat(i), (int)bank_nb, rowat(bkl, i), loop_res_query_.ptr + 4 * row * ne, 2, mat,
                               ne, err))
                    return false;
                if (!g.mix_dec(bnkat(i), (int)bank_nb, rowat(bkl, i), loop_kv_res_query_.ptr + 4 * row * ne, 2,
                               mkv, ne, err))
                    return false;
                msgQ = mat;
                msgK = mkv;
                msgV = mkv + 4 * ne;  // BYTE offset (mix_dec packs its rows ne floats apart), as the decode
                msgF = mat + 4 * ne;
            }
            // The attention inputs: the residual mix over the bank rows, one query for SWA / three for full attention;
            // block 0 has no res query, so Q/K/V all read the residual stream.
            if (l.ink.res_q.ptr != 0) {
                const size_t nq = is_swa ? 1 : 3;
                if (!g.mix_dec(bksi, (int)nb, xi, l.ink.res_q.ptr, nq, mixb, ne, err)) return false;
                if (nq == 1) {
                    if (!g.dtod(bufs_.at("mix"), 0, bufs_.at("inpb"), 0, 4 * ne, err) ||
                        !g.dtod(bufs_.at("mix"), 0, bufs_.at("inpb"), 4 * ne, 4 * ne, err) ||
                        !g.dtod(bufs_.at("mix"), 0, bufs_.at("inpb"), 8 * ne, 4 * ne, err))
                        return false;
                } else {
                    for (size_t j = 0; j < 3; j++)
                        if (!g.dtod(bufs_.at("mix"), 4 * j * ne, bufs_.at("inpb"), 4 * j * ne, 4 * ne, err))
                            return false;
                }
            } else {
                for (size_t j = 0; j < 3; j++)
                    if (!g.dtod(xbuf(i), 0, bufs_.at("inpb"), 4 * j * ne, 4 * ne, err)) return false;
            }
            uint64_t inp[3] = {inpb, inpb + 4 * ne, inpb + 8 * ne};
            // The message controls on Q, K and V (each stage batches the three, as the decode does).
            if (msgQ != 0) {
                if (!batch(g, [&](std::string& e) {
                        return g.gate_pack(inp[0], msgQ, gat, ne, 1e-6f, e) &&
                               g.gate_pack(inp[1], msgK, gat + 8 * ne, ne, 1e-6f, e) &&
                               g.gate_pack(inp[2], msgV, gat + 16 * ne, ne, 1e-6f, e);
                    }, err))
                    return false;
                if (!batch(g, [&](std::string& e) {
                        return g.gemv(l.ink.loop_down[0], gat, g64, e) &&
                               g.gemv(l.ink.loop_down[1], gat + 8 * ne, g64 + 4 * rank, e) &&
                               g.gemv(l.ink.loop_down[2], gat + 16 * ne, g64 + 8 * rank, e);
                    }, err))
                    return false;
                if (!batch(g, [&](std::string& e) {
                        return g.gemv(l.ink.loop_gain[0], g64, cg, e) &&
                               g.gemv(l.ink.loop_gain[1], g64 + 4 * rank, cg + 4 * ne, e) &&
                               g.gemv(l.ink.loop_gain[2], g64 + 8 * rank, cg + 8 * ne, e) &&
                               g.gemv(l.ink.loop_forget[0], g64, cf, e) &&
                               g.gemv(l.ink.loop_forget[1], g64 + 4 * rank, cf + 4 * ne, e) &&
                               g.gemv(l.ink.loop_forget[2], g64 + 8 * rank, cf + 8 * ne, e);
                    }, err))
                    return false;
                if (!batch(g, [&](std::string& e) {
                        return g.control_mix(inp[0], msgQ, cg, cf, inp[0], ne, e) &&
                               g.control_mix(inp[1], msgK, cg + 4 * ne, cf + 4 * ne, inp[1], ne, e) &&
                               g.control_mix(inp[2], msgV, cg + 8 * ne, cf + 8 * ne, inp[2], ne, e);
                    }, err))
                    return false;
            }
            // The input norms (the SWA norm on the shared input, else the three separate full-attention norms).
            if (trace_ck) std::fprintf(stderr, "[chunk] il=%zu i=%zu in-norm swa=%d msg=%d\n", il, i, (int)is_swa,
                                       (int)(msgQ != 0));
            if (is_swa) {
                if (!batch(g, [&](std::string& e) {
                        return g.rms_norm(inp[0], ne, l.attn_norm.ptr, inp[0], ne, ne, 1, c.eps, e) &&
                               g.rms_norm(inp[1], ne, l.attn_norm.ptr, inp[1], ne, ne, 1, c.eps, e) &&
                               g.rms_norm(inp[2], ne, l.attn_norm.ptr, inp[2], ne, ne, 1, c.eps, e);
                    }, err))
                    return false;
            } else if (!batch(g, [&](std::string& e) {
                           return g.rms_norm(inp[0], ne, l.ink.q_inp_norm.ptr, inp[0], ne, ne, 1, c.eps, e) &&
                                  g.rms_norm(inp[1], ne, l.ink.k_inp_norm.ptr, inp[1], ne, ne, 1, c.eps, e) &&
                                  g.rms_norm(inp[2], ne, l.ink.v_inp_norm.ptr, inp[2], ne, ne, 1, c.eps, e);
                       }, err))
                return false;
            // Q, K, V, the relative projection, the sink projection, and (full attention) the conv taps.
            if (!batch(g, [&](std::string& e) {
                    return g.gemv(l.q, inp[0], qb, e) && g.gemv(l.k, inp[1], kb, e) &&
                           g.gemv(l.v, inp[2], vout, e) && g.gemv(l.ink.rel, inp[0], rel, e) &&
                           g.gemv(l.ink.sink_proj, inp[0], snk, e) &&
                           (is_swa || (g.gemv(l.ink.k_conv_proj, inp[1], kpj, e) &&
                                       g.gemv(l.ink.v_conv_proj, inp[2], vpj, e)));
                }, err))
                return false;
            // The shortconvs (the state advances token by token), the clips, the K clip and V norm.
            const float cscale = 1.0f / std::sqrt((float)(is_swa ? 1 : 2) * (float)nc + 1.0f);
            if (!batch(g, [&](std::string& e) {
                    return g.conv_ink(kb, kstate, l.ink.k_conv.ptr, is_swa ? 0 : kpj, kout, kv, nc, hd, cscale, e) &&
                           g.conv_ink(vout, vstate, l.ink.v_conv.ptr, is_swa ? 0 : vpj, vout, kv, nc, hd, cscale, e) &&
                           g.clip_rows(qb, l.ink.q_clip.ptr, hd, nh, c.f_qk_clip,
                                       is_swa ? 0.0f : c.f_log_scale_alpha, rfloor, stepi, e) &&
                           g.clip_rows(rel, l.ink.rel_clip.ptr, nrel, nh, c.f_rel_clip,
                                       is_swa ? 0.0f : c.f_log_scale_alpha, rfloor, stepi, e);
                }, err))
                return false;
            if (!batch(g, [&](std::string& e) {
                    return g.clip_rows(kout, 0, hd, nkv, c.f_qk_clip, 0.0f, rfloor, stepi, e) &&
                           g.rms_norm(vout, hd, ones, vout, hd, hd, nkv, c.eps, e);
                }, err))
                return false;
            if (trace_ck) std::fprintf(stderr, "[chunk] il=%zu i=%zu v-norm ok\n", il, i);
            if (!g.kv_store(kout, vout, l.kc.addr(), l.vc.addr(), kv, 1, stepi, hd, err)) return false;
            // The rel-bias table of this token and the output-gate projection, then the attention.
            if (!batch(g, [&](std::string& e) {
                    return g.rel_table(l.ink.rel_bias.ptr, rel, snk, tbl, nrel, next1, nh, e) &&
                           g.gemv(l.dn_gate, inp[0], qg, e);
                }, err))
                return false;
            if (!g.attn_decode_gq(qb, l.kc.addr(), l.vc.addr(), 0, att, attp, stepi, nh, nkv, hd, attn_scale,
                                  gq_min(pos_ + i, 1), 1, sinkz, is_swa ? c.n_swa : 0, tbl, c.n_rel_extent, err))
                return false;
            if (!g.sig_mul(att, qg, nh * hd, err)) return false;
            if (!g.gemv(l.o, att, ao, err)) return false;
            if (!g.conv_ink(ao, astate, l.ink.post_conv.ptr, 0, ao, ne, nc, 0, conv_scale, err)) return false;
            // The block residual rule on the PHYSICAL block.
            if (boundary) {
                if (!g.dtod(xbuf(i), 0, bksbuf(i), 4 * nb * ne, 4 * ne, err)) return false;
                if (!g.dtod(bufs_.at("ao"), 0, xbuf(i), 0, 4 * ne, err)) return false;
            } else if (!g.add(xi, ao, ne, err)) {
                return false;
            }
            // The FFN input: the depth mix over the bank rows (this slot's push included) and the residual, the FFN
            // message's control, the norm into this token's h row.
            if (!g.mix_dec(bksi, (int)nb_ffn, xi, l.ink.ffn_res_q.ptr, 1, mixb, ne, err)) return false;
            if (msgF != 0) {
                if (!g.gate_pack(mixb, msgF, gat, ne, 1e-6f, err)) return false;
                if (!g.gemv(l.ink.loop_down[3], gat, g64, err)) return false;
                if (!batch(g, [&](std::string& e) {
                        return g.gemv(l.ink.loop_gain[3], g64, cg, e) &&
                               g.gemv(l.ink.loop_forget[3], g64, cf, e);
                    }, err))
                    return false;
                if (!g.control_mix(mixb, msgF, cg, cf, mixb, ne, err)) return false;
            }
            if (!g.rms_norm_q(mixb, ne, l.ink.ffn_norm.ptr, rowbuf("h", i).addr(), ne, ne, 1, c.eps, hq, 0, err))
                return false;
            if (trace_ck) std::fprintf(stderr, "[chunk] il=%zu i=%zu ffn-norm ok\n", il, i);
        }
        if (boundary) nb++;
        // The batched MoE over the chunk's h rows.
        if (trace_ck) std::fprintf(stderr, "[chunk] il=%zu moe kt=%zu\n", il, t * ke);
        if (!g.memset(bufs_.at("moe"), err)) return false;
        const size_t phys = looped ? loop_layer_of(c, il) : il;
        if (!cache_) {
            err = "MoE layer without an expert cache";
            return false;
        }
        const LayerExperts* lx = cache_->layer(phys);
        if (!lx) {
            err = "MoE layer " + std::to_string(il) + " has no expert cache entry";
            return false;
        }
        if (!fast && !worker_) {
            err = "routed experts are not all resident in VRAM and the model has no CPU expert worker";
            return false;
        }
        const uint64_t mb = fast ? 0 : worker_->dev() + 4 * il * layout_.stride;
        const auto [base, stride] = cache_->base(*lx);
        const ExpertMat& mg = lx->mats[0];
        const ExpertMat& md = lx->mats[1];
        const size_t kt = t * ke;
        // The router per token (the decode's exact f32 gemv; a batched GEMM changes the logits' rounding and can flip a
        // near-tie in the top-8), its rows contiguous for the chunk's moe_route.
        for (size_t i = 0; i < t; i++)
            if (!g.gemv(l.router, rowat(h, i), rl + 4 * c.n_expert * i, err)) return false;
        if (!g.moe_route(rl, c.n_expert, ke, c.expert_weights_scale, lx->table_dev.addr(), ids, wts, slots, wh, h,
                         mb, ne, layout_.h, stepb, t, 0, true, l.ink.exp_probs_b.ptr, 1e-20f, err))
            return false;
        if (!g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off, stride, slots, kt, h, ne, ke, gu, std::nullopt,
                        0, err))
            return false;
        uint64_t dq = guq;
        if (!g.situ_q(gu, nff, kt, l.ink.gu_scale.ptr, ids, c.situ_beta, c.situ_linear_beta, guq, dq, err))
            return false;
        if (!g.moe_gemv(md.ty, md.rows, md.cols, base + md.off, stride, slots, kt, gu, 2 * nff, 1, ye,
                        std::nullopt, ink_xq ? dq : 0, err))
            return false;
        if (!g.moe_combine(moe, ye, wh, ke, 0, 0, mb, layout_.y, stepb, slots, ne, t, l.ink.dn_scale.ptr, ids, err))
            return false;
        // The FFN output's post conv (its state advances token by token) and the residual add.
        for (size_t i = 0; i < t; i++) {
            if (!g.conv_ink(rowat(moe, i), fstate, l.ink.ffn_post_conv.ptr, 0, rowat(moe, i), ne, nc, 0,
                            conv_scale, err))
                return false;
            if (!g.add(xat(i), rowat(moe, i), ne, err)) return false;
        }
    }
    // The head for the chunk's last token: the output depth mix, the output norm, the lm_head and the softcap.
    {
        const size_t last = t - 1;
        if (!g.mix_dec(bksat(last), (int)nb, xat(last), output_res_query_.ptr, 1, mixb, ne, err)) return false;
        if (!g.rms_norm_q(mixb, ne, output_norm_.ptr, rowat(h, last), ne, ne, 1, c.eps, hq, 0, err)) return false;
        // exact f32 head in the chunk path (the XQ path's quantized hq is a single-row buffer)
        if (!g.gemv(output_, rowat(h, last), logits, err)) return false;
        if (!g.softcap(logits, c.n_vocab, c.f_final_softcap, err)) return false;
    }


    return true;
}

// port of gpu_model.rs: GpuModel::finish. While a MoE combine gave up waiting for the CPU worker (the GPU reports it
// through stalled), wait for that layer's answer and run the rest of the forward from its combine.
bool GpuModel::finish(int32_t gq, std::string& err) {
    Backend& g = *gpu_;
    int64_t last = -1;
    while (g.stalled()) {
        if (!worker_) {
            err = "a MoE combine gave up waiting, but the model has no CPU expert worker";
            return false;
        }
        const std::optional<size_t> il = worker_->stalled_layer();
        if (!il) {
            err = "a MoE combine gave up waiting, but no layer says so";
            return false;
        }
        // The worker had answered the last time: the GPU does not see the mailbox.
        if (last == static_cast<int64_t>(*il)) {
            err = "moe_combine: the GPU does not see the CPU expert worker's answers. On an AMD RX 9000 (RDNA4) GPU set "
                  "radv_device_coherent_memory=true in the environment";
            return false;
        }
        last = static_cast<int64_t>(*il);
        if (!worker_->wait_done(*il, -1, err)) return false;
        if (!forward(gq, last, err)) return false;
        if (!g.sync(err)) return false;
        resumed++;
    }
    return true;
}

// port of gpu_model.rs: GpuModel::layer. The PLE block, the mixer (DeltaNet or attention, with the QSA indexer) between
// the hyper-connection reads and writes (or the plain norms), then the FFN (dense, or MoE: the router, the routed
// experts in VRAM and on the CPU worker, the shared expert) into the streams. `resume`: only the MoE combine (what
// `finish` runs after a combine gave up waiting on the worker).
bool GpuModel::layer(size_t il, const GpuLayer& l, uint64_t x, uint64_t step, int32_t gq, uint64_t& res, bool resume,
                     std::string& err) {
    const ModelConfig& c = m_.cfg;
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode forward is W2. Without this guard
    // a decode falls into the generic qwen attention path, whose first op reads the empty q_norm of an
    // inkling2 block — a null device pointer (or, past it, silently wrong logits).
    if (l.inkling2) {
        err = "inkling2 decodes through forward_inkling2 (W2); layer() is the qwen path";
        return false;
    }
    Backend& g = *gpu_;
    const size_t ne = c.n_embd;
    const float eps = c.eps;
    const size_t dk = c.ssm_d_state, nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    const size_t dv = c.ssm_dv();
    const size_t channels = c.ssm_d_inner + 2 * nk * dk;
    const size_t hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv;
    const size_t kv = nkv * hd;
    const size_t qs = c.q_gate ? 2 * hd : hd;  // per-head row of attn_q: query, then the gate if any
    const bool use_rope = il < c.rope_layer.size() ? c.rope_layer[il] : true;
    const size_t win = (il < c.swa.size() && c.swa[il]) ? c.n_swa : 0;
    const uint64_t rope = rope_buf_.addr();

    if (!resume) {
        const HcAt ha = hc_at();
        // port of gpu_model.rs: the PLE block adds the per-layer embedding to the stack first
        if (l.has_ple && !ple_block(l.ple, x, step, 1, ha, err)) return false;
        if (l.hc) {
            if (!hc_read(l.hcr[0], x, 1, ha, err)) return false;
        } else if (!g.rms_norm_q(x, ne, l.attn_norm.ptr, buf("h"), ne, ne, 1, eps, buf("hq"), res, err)) {
            return false;
        }
        if (l.delta) {
            // port of gpu_model.rs: Mixer::Delta (nt == 1: one conv_silu, one deltanet launch; 128-wide heads get the
            // q/k l2 norms inside conv_silu)
            if (!batch(g,
                       [&](std::string& e) {
                           return g.gemvq2(l.qkv, l.dn_gate, buf("h"), ne, buf("hq"), buf("qkv"), buf("z"), 1, e) &&
                                  g.gemvq2(l.beta, l.alpha, buf("h"), ne, buf("hq"), buf("beta"), buf("g"), 1, e);
                       },
                       err))
                return false;
            const size_t nn = dk == 128 ? 2 * nk : 0;
            if (!g.conv_silu(buf("qkv"), l.conv.addr(), l.conv_w.ptr, buf("y"), channels, c.ssm_conv, nn, eps, err)) return false;
            if (nn == 0 && !g.l2_norm(buf("y"), dk, 2 * nk, channels, 2 * nk, eps, err)) return false;
            const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
            if (!g.deltanet(buf("y"), l.s.addr(), buf("beta"), buf("g"), l.dt_bias.ptr, l.a.ptr, buf("o"), nk, nv,
                            c.k_interleave, dk, dv, scale, 1, channels, 0, 0, err))
                return false;
            // The output gate: sigmoid(z) in the hyper-connection models (norm_gate_sig), silu(z) otherwise.
            uint64_t oq = 0;
            const bool gated = l.hc ? g.norm_gate_sig(buf("o"), l.norm.ptr, buf("z"), dv, nv, eps, buf("oq"), oq, err)
                                    : g.norm_gate_q(buf("o"), l.norm.ptr, buf("z"), dv, nv, eps, buf("oq"), oq, err);
            if (!gated) return false;
            if (oq != 0) {
                if (!g.gemvq(l.out, buf("o"), c.ssm_d_inner, oq, buf("mix"), ne, 1, err)) return false;
            } else if (!g.gemv(l.out, buf("o"), buf("mix"), err)) {
                return false;
            }
        } else {
            // port of gpu_model.rs: Mixer::Attn (nt == 1, no image rows)
            if (!batch(g,
                       [&](std::string& e) {
                           return g.gemvq2(l.q, l.k, buf("h"), ne, buf("hq"), buf("qg"), buf("kraw"), 1, e) &&
                                  g.gemvq(l.v, buf("h"), ne, buf("hq"), buf("v"), kv, 1, e);
                       },
                       err))
                return false;
            if (use_rope) {
                // Fused: q and k norms and RoPE, and the K and V rows into the cache (rope_offset is 0, not an image token).
                if (!g.rms_norm_rope_qk(buf("qg"), qs, l.qn.ptr, buf("q"), nh, nh, buf("kraw"), l.kn.ptr, buf("k"), nkv, nkv, hd,
                                        eps, c.n_rot, step, c.rope_base, buf("v"), l.kc.addr(), l.vc.addr(), err))
                    return false;
            } else {
                if (!g.rms_norm(buf("qg"), qs, l.qn.ptr, buf("q"), hd, hd, nh, eps, err)) return false;
                if (!g.rms_norm(buf("kraw"), hd, l.kn.ptr, buf("k"), hd, hd, nkv, eps, err)) return false;
                if (!g.kv_store(buf("k"), buf("v"), l.kc.addr(), l.vc.addr(), kv, 1, step, hd, err)) return false;
            }
            // port of gpu_model.rs: the QSA indexer (pool the raw keys, select the blocks of the query); the selection goes
            // to the attention as its sparse cells.
            uint64_t sel = 0;
            size_t sst = 0;
            if (l.has_qsa) {
                const QsaBlock& a = l.qsa;
                if (!g.gemvq2(a.qp, a.kp, buf("h"), ne, buf("hq"), buf("iq"), buf("ik"), 1, err)) return false;
                if (!g.qsa_pool(buf("ik"), a.ring, a.pool.addr(), a.kn.ptr, step, 1, c.n_rot, c.rope_base, eps, err)) return false;
                if (!g.rms_norm(buf("iq"), QSA_D, a.qn.ptr, buf("iqn"), QSA_D, QSA_D, QSA_H, eps, err)) return false;
                if (!g.rope_neox(buf("iqn"), QSA_D, QSA_H, QSA_H, c.n_rot, rope, c.rope_base, err)) return false;
                if (!g.qsa_select(buf("iqn"), a.pool.addr(), qsa_.score.addr(), qsa_.sel.addr(), 0, step, 0, 1, qsa_.nbs, qsa_.top,
                                  qsa_.sst, 0, err))
                    return false;
                sel = qsa_.sel.addr();
                sst = qsa_.sst;
            }
            const uint64_t sink = l.sink.ptr;
            uint64_t aq = 0;
            if (!g.attn_decode_q(buf("q"), l.kc.addr(), l.vc.addr(), c.q_gate ? buf("qg") : 0, buf("att"), buf("attp"), step, nh,
                                 nkv, hd, 1.0f / std::sqrt(static_cast<float>(hd)), gq, 1, buf("attq"), sel, sst, sink, win, aq, err))
                return false;
            if (aq != 0) {
                if (!g.gemvq(l.o, buf("att"), nh * hd, aq, buf("mix"), ne, 1, err)) return false;
            } else if (!g.gemv(l.o, buf("att"), buf("mix"), err)) {
                return false;
            }
        }
        // The mixer's output joins the residual (in the streams, with hyper-connections), and h = the FFN's input.
        if (!l.hc) {
            if (!g.rms_norm_q(x, ne, l.post_norm.ptr, buf("h"), ne, ne, 1, eps, buf("hq"), buf("mix"), err)) return false;
        } else {
            if (!g.hc_write(x, buf("mix"), buf("inj"), ne, c.hc_n, 1, false, err)) return false;
            if (!hc_read(l.hcr[1], x, 1, ha, err)) return false;
        }
    }

    if (!l.moe) {
        // port of gpu_model.rs: Ffn::Dense. Its output stays a residual for the next norm.
        if (!batch(g, [&](std::string& e) { return g.gemvq2(l.gate, l.up, buf("h"), ne, buf("hq"), buf("ge"), buf("ue"), 1, e); }, err))
            return false;
        uint64_t q = 0;
        const size_t n = l.down.cols;
        if (l.down.ty == Ty::F32 || l.down.ty == Ty::BF16) {
            if (!g.swiglu(buf("ge"), buf("ue"), n, err)) return false;
        } else if (!g.swiglu_q(buf("ge"), buf("ue"), n, 1, buf("ffq"), q, err)) {
            return false;
        }
        if (!g.gemvq(l.down, buf("ge"), n, q, buf("ye"), ne, 1, err)) return false;
        res = buf("ye");
        return true;
    }

    // port of gpu_model.rs: Ffn::Moe. The routed experts in VRAM run here; the missed ones on the CPU worker, whose
    // answer the combine adds (mb = the layer's mailbox). A fully resident cache takes mb = 0: no mailbox, no wait.
    // StrataFreet addition (INKLING2_NOTES.md): the cache is keyed by the slot's physical block; the
    // mailbox stays per slot (each loop pass routes independently).
    const size_t k = c.n_expert_used;
    const size_t phys = c.is_looped() ? loop_layer_of(c, il) : il;
    if (!cache_) {
        err = "MoE layer without an expert cache";
        return false;
    }
    const LayerExperts* lx = cache_->layer(phys);
    if (!lx) {
        err = "MoE layer " + std::to_string(il) + " has no expert cache entry";
        return false;
    }
    const bool fast = cache_->fully_resident() && g.moe_mailbox_optional();
    if (!fast && !worker_) {
        err = "routed experts are not all resident in VRAM and the model has no CPU expert worker";
        return false;
    }
    const uint64_t mb = fast ? 0 : worker_->dev() + 4 * il * layout_.stride;
    const auto [base, stride] = cache_->base(*lx);
    const ExpertMat& mg = lx->mats[0];
    const ExpertMat& mu = lx->mats[1];
    const ExpertMat& md = lx->mats[2];
    const uint64_t slots = slots_.addr();
    const uint64_t ids = ids_.addr();
    uint64_t sq = 0;
    uint64_t dq = 0;
    if (!resume) {
        if (!fast) mailbox_writes++;
        // Router, and the shared expert's gate and up (their gate is a sigmoid when ffn_gate_inp_shexp exists).
        if (l.shared) {
            if (l.has_shared_gate_inp) {
                if (!g.gemvq2(l.router, l.sh_gate_inp, buf("h"), ne, buf("hq"), buf("rl"), buf("sg"), 1, err)) return false;
            } else if (!g.gemvq(l.router, buf("h"), ne, buf("hq"), buf("rl"), c.n_expert, 1, err)) {
                return false;
            }
            if (!g.gemvq2(l.sh_gate, l.sh_up, buf("h"), ne, buf("hq"), buf("shg"), buf("shu"), 1, err)) return false;
        } else if (!g.gemvq(l.router, buf("h"), ne, buf("hq"), buf("rl"), c.n_expert, 1, err)) {
            return false;
        }
        if (!g.moe_route(buf("rl"), c.n_expert, k, c.expert_weights_scale, lx->table_dev.addr(), ids, buf("wts"), slots,
                         buf("wh"), buf("h"), mb, ne, layout_.h, step, 1, 0, c.router_sigmoid, 0, 0.0f, err))
            return false;
        // The routed gate and up, paired when they share a type and shape.
        if (mg.ty == mu.ty && mg.rows == mu.rows && mg.cols == mu.cols) {
            std::optional<std::pair<int64_t, int64_t>> pair = std::make_pair(
                static_cast<int64_t>(mu.off) - static_cast<int64_t>(mg.off),
                static_cast<int64_t>(buf("ue")) - static_cast<int64_t>(buf("ge")));
            if (!g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off, stride, slots, k, buf("h"), ne, k, buf("ge"), pair,
                            buf("hq"), err))
                return false;
        } else {
            if (!g.moe_gemv(mg.ty, mg.rows, mg.cols, base + mg.off, stride, slots, k, buf("h"), ne, k, buf("ge"), std::nullopt,
                            buf("hq"), err))
                return false;
            if (!g.moe_gemv(mu.ty, mu.rows, mu.cols, base + mu.off, stride, slots, k, buf("h"), ne, k, buf("ue"), std::nullopt,
                            buf("hq"), err))
                return false;
        }
        if (l.shared) {
            // The shared expert's SwiGLU, quantized for its down projection when that is quantized.
            if (l.sh_down.ty == Ty::F32 || l.sh_down.ty == Ty::BF16) {
                if (!g.swiglu(buf("shg"), buf("shu"), c.n_ff_shexp, err)) return false;
            } else if (!g.swiglu_q(buf("shg"), buf("shu"), c.n_ff_shexp, 1, buf("shq"), sq, err)) {
                return false;
            }
        }
        if (md.ty == Ty::F32 || md.ty == Ty::BF16) {
            if (!g.swiglu(buf("ge"), buf("ue"), c.n_ff_exp * k, err)) return false;
        } else if (!g.swiglu_q(buf("ge"), buf("ue"), c.n_ff_exp, k, buf("ffq"), dq, err)) {
            return false;
        }
        if (l.shared && !g.gemvq(l.sh_down, buf("shg"), c.n_ff_shexp, sq, buf("shy"), ne, 1, err)) return false;
    } else {
        // Resuming: the routing, the expert GEMVs and the shared expert are already in the buffers; only the down
        // quantization the skipped part left remains (see the Rust resume branch).
        if (!(md.ty == Ty::F32 || md.ty == Ty::BF16)) dq = buf("ffq");
    }
    const uint64_t sh = l.shared ? buf("shy") : 0;
    const uint64_t sgate = (l.shared && l.has_shared_gate_inp) ? buf("sg") : 0;
    // Hyper-connections: the FFN output collects in "mo" (zero: hc_write clears it after use) and goes to the streams.
    const uint64_t fx = l.hc ? buf("mo") : x;
    // The down projection of the k routed experts and the weighted sum into fx (the residual).
    if (!g.moe_down_combine(md.ty, md.rows, md.cols, base + md.off, stride, slots, k, buf("ge"), buf("ye"), dq, fx, buf("wh"), sh,
                            sgate, mb, layout_.y, step, 1, err))
        return false;
    if (l.hc && !g.hc_write(x, buf("mo"), buf("inj"), ne, c.hc_n, 1, true, err)) return false;
    res = 0;
    return true;
}

// port of expert_cache.rs / gpu_model.rs: after_experts. Routing counters from the mailbox (a fully resident part reads
// none: every routed expert is a hit), the record of the routing, and the placement update every adapt_every tokens
// (never with no miss: the fast path).
bool GpuModel::after_experts(size_t count, std::string& err) {
    if (!cache_) return true;
    const bool fast = expert_fast_path();
    const size_t k = m_.cfg.n_expert_used;
    const ModelConfig& c = m_.cfg;
    for (size_t il = 0; il < layers_.size(); il++) {
        // StrataFreet addition (INKLING2_NOTES.md): the routing counts land on the slot's physical
        // block (the loop passes share the resident set and the profile).
        const size_t phys = c.is_looped() ? loop_layer_of(c, il) : il;
        if (!cache_->layer(phys)) continue;
        if (fast) {
            hits += k;
            continue;
        }
        const auto r = worker_->routing(il);
        mailbox_reads++;
        std::vector<int32_t> ids;
        ids.reserve(r.size());
        for (const auto& [id, w, slot] : r) {
            (void)w;
            if (slot >= 0) {
                hits++;
            } else {
                misses++;
            }
            ids.push_back(id);
        }
        cache_->record(phys, ids);
    }
    if (!fast && adapt_every > 0 && (count - 1) / adapt_every < count / adapt_every) {
        if (!cache_->adapt(m_, *gpu_, adapt_swaps, err)) return false;
    }
    return true;
}

// port of expert_cache.rs: ExpertCache::save_profile, through the model
bool GpuModel::save_profile(std::string& err) {
    return cache_ ? cache_->save_profile(m_, err) : true;
}

// port of gpu_model.rs: expert_fast_path (every expert of the cache in VRAM and the backend takes a null mailbox)
bool GpuModel::expert_fast_path() const {
    return cache_ && cache_->fully_resident() && gpu_->moe_mailbox_optional();
}

}  // namespace strata
