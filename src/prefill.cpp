// port of gpu_model/prefill.rs: the chunked batched prefill of one part (Scratch::new, prefill / prefill_with,
// adapt_experts, prefill_chunk with its hyper-connection streams, PLE and QSA, and its MoE FFN). The routed experts run
// as grouped GEMMs over the chunk's (token, expert) pairs sorted by expert; experts not resident in the cache are copied
// into a small staging area, a group of them at a time, each group's copy beside the previous group's GEMMs.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Scope (see MODIFICATIONS.md, sections W3-E3 and W3-E1b): one part, no MTP draft head, no EAGLE3 head, no LoRA. The
// hyper-connections, PLE and QSA of qwen4exp are ported in W3-E1b.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "expert_cache.h"
#include "gpu_model.h"
#include "weights.h"

namespace strata {

namespace {

// port of prefill.rs: STAGE (experts staged per group; staging holds three groups)
constexpr size_t kStage = 16;

// One routed expert: its id and the device addresses of its gate, up and down matrices.
using Item = std::pair<size_t, std::array<uint64_t, 3>>;
using Group = std::vector<Item>;

size_t env_size(const char* name, size_t dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    const unsigned long long x = std::strtoull(v, &end, 10);
    return end == v ? dflt : static_cast<size_t>(x);
}

// Runs `f` as the batch of independent ops (port of Backend::batch as prefill.rs calls it).
template <class F>
bool batch(Backend& g, F&& f, std::string& err) {
    return g.batch([&](std::string& e) { return f(e); }, err);
}

// STRATA_TRACE_PREFILL (opt-in, not in the official binary): host wall time per phase of the batched prefill, summed
// over the chunks and layers of one prefill call, and printed to stderr. "=1" times the host side (the GPU runs ahead, so
// a phase's time is the host's enqueue plus any wait on the device it makes); "=2" also drains the device at the end of
// the phases marked as fenced, so each one's time is its own device time too (serialized, not the real overlap).
// Unset: one branch on a cached flag per phase, nothing else.
enum PfPhase { kPhEmbed, kPhPle, kPhHc, kPhMix, kPhRoute, kPhIds, kPhBuild, kPhEnq, kPhCopy, kPhJoin, kPhDense, kPhNum };
const char* const kPhNames[kPhNum] = {"embed+put", "ple", "hc read/write", "mixer", "router+shared", "ids sync",
                                      "build lists", "beside enqueue", "  of which copy", "join+combine", "dense ffn"};
struct PfTrace {
    bool on = false;
    bool fence = false;
    double ms[kPhNum] = {};
    double total_ms = 0;
    uint64_t chunks = 0, moe_layers = 0, copy_bytes = 0, missed = 0, resident = 0;
};
PfTrace& pft() {
    static PfTrace t;
    static bool init = false;
    if (!init) {
        init = true;
        const char* v = std::getenv("STRATA_TRACE_PREFILL");
        t.on = v && *v && std::strcmp(v, "0") != 0;
        t.fence = t.on && std::strcmp(v, "2") == 0;
    }
    return t;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// A phase's start mark (0 when tracing is off) and its end: adds the elapsed time to phase `p`.
double pf_t() { return pft().on ? now_ms() : 0.0; }
void pf_add(int p, double t0) {
    if (pft().on) pft().ms[p] += now_ms() - t0;
}
// Drains the device before a fenced phase ends (trace mode 2 only).
bool pf_fence(Backend& g, std::string& err) { return !pft().fence || g.sync(err); }

}  // namespace

// port of gpu_model/prefill.rs: Scratch::new's sizes (the slab layout, the ints, the tiles and the staging)
bool prefill_sizes(const Model& m, size_t chunk, PrefillSizes& out, std::string& err) {
    const ModelConfig& c = m.cfg;
    // Bytes of one staged group's slot: the experts' matrices laid out as ExpertCache lays a slot out.
    size_t stage_bytes = 0;
    for (size_t il = 0; il < c.n_layer; il++) {
        if (!c.is_moe_layer(il)) continue;
        size_t b = 0;
        for (const char* name : kExpertMats) {
            Tensor t;
            if (!m.w.layer(il, name, t, err)) return false;
            b = (b + t.expert(0, c.n_expert).len + 15) / 16 * 16;
        }
        stage_bytes = std::max(stage_bytes, b);
    }
    const size_t ne = c.n_embd;
    const size_t k = std::max<size_t>(c.n_expert_used, 1);
    const size_t nv = c.ssm_dt_rank;
    const size_t kv = c.n_head_kv * c.head_dim;
    const size_t channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
    const size_t q = c.n_head * c.head_dim;
    const size_t hcn = std::max<size_t>(c.hc_n, 1);
    const size_t hd = hcn * ne;  // one token's stack of hc_n rows (x)
    const size_t qsa_w = c.idx_top_k > 0 ? (1 + 2 * QSA_H) * QSA_D : 0;
    // x (one stack of hd floats a token) at 0, then h, then the mixer regions a..e and the FFN regions, as Scratch::new.
    size_t at = (hd + ne) * chunk;
    const size_t base = at;
    auto take = [&](size_t n) {
        const size_t o = at;
        at += n * chunk;
        return o;
    };
    PrefillLay& L = out.lay;
    L.h = hd * chunk;
    L.a = take(std::max({2 * q, channels, 2 * ne}));
    L.b = take(std::max(q, c.ssm_d_inner));
    L.c = take(std::max({channels, 3 * kv, qsa_w}));
    L.d = take(std::max(q, c.ssm_d_inner));
    L.e = take(2 * nv);
    const size_t mix_end = at;
    at = base;
    L.rl = take(c.n_expert);
    L.wts = take(k);
    L.ge = take(std::max(k * c.n_ff_exp, c.n_ff));
    L.ye = take(k * ne);
    L.shg = take(c.n_ff_shexp);
    L.shy = take(ne);
    L.sg = take(1);
    size_t total = std::max(at, mix_end);
    if (c.hc_n > 0) {
        // The hyper-connection regions: the reads' rows (they alias the mixer's and FFN's), then the write's inject, the
        // mixer's output, the FFN's collector and the PLE input (all after the others).
        at = base;
        L.xn = take(hd);
        L.lo = take(c.hc_rank);
        L.hg = take(hd);
        L.pk = take(hd);
        L.pq = take(hd);
        L.pg = take(hd);
        L.pn = take(hd);
        L.pv = take(ne);
        at = std::max(at, total);
        L.inj = take(c.hc_n);
        L.mix = take(ne);
        L.mo = take(ne);
        L.emb = take(ne);
        total = at;
    }
    out.slab_floats = total;
    out.ints = 3 * chunk * k;
    out.tiles = 2 * 3 * (chunk * k / GEMM_BM + 2 * c.n_expert);
    out.staging = c.is_moe() ? 3 * kStage * stage_bytes : 0;
    return true;
}

// port of gpu_model/prefill.rs: Scratch::new (allocates the sizes of prefill_sizes)
bool GpuModel::make_prefill(size_t chunk, std::string& err) {
    Backend& g = *gpu_;
    PrefillSizes z;
    if (!prefill_sizes(m_, chunk, z, err)) return false;
    PrefillScratch s;
    s.chunk = chunk;
    s.lay = z.lay;
    if (!g.alloc(4 * z.slab_floats, s.slab, err)) return false;
    if (!g.alloc(4 * z.ints, s.ints, err)) return false;
    if (!g.alloc(8 * z.tiles, s.tiles, err)) return false;
    if (z.staging > 0 && !g.alloc(z.staging, s.staging, err)) return false;
    pf_ = std::make_unique<PrefillScratch>(std::move(s));
    return true;
}

// port of gpu_model/prefill.rs: GpuModel::prefill (prefill_with with no between hook, one part). Runs `tokens` from the
// current position; returns the logits after the last one.
bool GpuModel::prefill(const std::vector<uint32_t>& tokens, std::vector<float>& logits, std::string& err) {
    if (tokens.empty()) {
        err = "prefill: empty prompt";
        return false;
    }
    // StrataFreet addition (INKLING2_NOTES.md): inkling2 has its own chunked prefill (W3), built on the decode
    // buffers; it does not use the qwen PrefillScratch (chunk_ is 0 for it).
    if (m_.cfg.kind == Arch::Kind::Inkling2) return prefill_inkling2(tokens, logits, err);
    // Short prompts (and a model loaded without a chunk) go through the decode step, token by token (prefill.rs:
    // step_all for a single part).
    if (!pf_ || tokens.size() < short_len_) {
        for (uint32_t t : tokens) {
            if (!step(t, logits, err)) return false;
        }
        return true;
    }
    if (pos_ + tokens.size() > ctx_) {
        err = "context full (" + std::to_string(ctx_) + " tokens; raise ctx)";
        return false;
    }
    // STRATA_PF_ADAPT=N: place the hot experts after every N-th chunk of a long prompt (0 = off, the default).
    const size_t every = env_size("STRATA_PF_ADAPT", 0);
    const size_t chunk = pf_->chunk;
    const size_t nchunks = (tokens.size() + chunk - 1) / chunk;
    PfTrace& tr = pft();
    if (tr.on) tr = PfTrace{tr.on, tr.fence};
    double h_mc = 0, h_wait = 0, h_dma = 0;
    uint64_t h_chunks = 0, h_bytes = 0;
    if (tr.on) htod_trace_take(h_mc, h_wait, h_dma, h_chunks, h_bytes);
    const double t_start = tr.on ? now_ms() : 0.0;
    for (size_t i = 0; i < nchunks; i++) {
        const size_t off = i * chunk;
        const size_t t = std::min(chunk, tokens.size() - off);
        if (!prefill_chunk(tokens.data() + off, t, err)) return false;
        if (every > 0 && (i + 1) % every == 0 && i + 1 < nchunks && !adapt_experts(err)) return false;
    }
    if (tr.on) {
        // The chunk loop's wall time, with the device drained; then one summary line per prefill call.
        if (!gpu_->sync(err)) return false;
        tr.total_ms = now_ms() - t_start;
        std::fprintf(stderr, "prefill trace: %zu tokens in %llu chunk(s), %.1f ms (%.1f tok/s), %llu MoE layer runs\n",
                     tokens.size(), static_cast<unsigned long long>(tr.chunks), tr.total_ms,
                     1000.0 * static_cast<double>(tokens.size()) / tr.total_ms, static_cast<unsigned long long>(tr.moe_layers));
        for (int p = 0; p < kPhNum; p++) {
            std::fprintf(stderr, "  %-16s %9.1f ms  %5.1f%%\n", kPhNames[p], tr.ms[p], 100.0 * tr.ms[p] / tr.total_ms);
        }
        std::fprintf(stderr, "  copies: %.1f MB staged for %llu missed expert(s)\n", static_cast<double>(tr.copy_bytes) / 1e6,
                     static_cast<unsigned long long>(tr.missed));
        htod_trace_take(h_mc, h_wait, h_dma, h_chunks, h_bytes);
        std::fprintf(stderr, "  pinned ring: %llu chunk(s), %.1f MB: memcpy %.1f ms, waits %.1f ms, DMA enqueue %.1f ms\n",
                     static_cast<unsigned long long>(h_chunks), static_cast<double>(h_bytes) / 1e6, h_mc, h_wait, h_dma);
    }
    // The prompt's routing predicts the reply's: place its hot experts now instead of 64 per adapt during decode.
    if (!adapt_experts(err)) return false;
    // Serve: place the prompt's experts for the prefill, then let decode follow the reply's own routing.
    if (reset_counts_after_prefill_) reset_counts();
    // prefill_chunk leaves the last hidden state (the stack, hc_n rows) in row 0 of x, as after `step`.
    Backend& g = *gpu_;
    const ModelConfig& c = m_.cfg;
    if (has_output_hc_) {
        if (!hc_read(output_hc_, x_.addr(), 1, hc_at(), err)) return false;
    } else if (!g.rms_norm(x_.addr(), 0, output_norm_.ptr, buf("h"), 0, c.n_embd, 1, c.eps, err)) {
        return false;
    }
    if (!g.gemv(output_, buf("h"), logits_.addr(), err)) return false;
    return get(g, logits_, 0, c.n_vocab, logits, err);
}

// StrataFreet addition (INKLING2_NOTES.md): the inkling2 prefill (W3). Chunks of ink_chunk_ (<= 16) tokens through
// forward_inkling2_chunk, which leaves the last token's logits on the device; the debug dumps and the slot-1 MoE
// record are only written by the token-by-token path, so an enabled STRATA_INK_DEBUG / STRATA_INK_DUMP keeps it.
bool GpuModel::prefill_inkling2(const std::vector<uint32_t>& tokens, std::vector<float>& logits, std::string& err) {
    if (pos_ + tokens.size() > ctx_) {
        err = "context full (" + std::to_string(ctx_) + " tokens; raise ctx)";
        return false;
    }
    const bool dbg = std::getenv("STRATA_INK_DEBUG") != nullptr || std::getenv("STRATA_INK_DUMP") != nullptr;
    if (ink_chunk_ == 0 || dbg) {
        for (uint32_t t : tokens) {
            if (!step(t, logits, err)) return false;
        }
        return true;
    }
    for (size_t off = 0; off < tokens.size(); off += ink_chunk_) {
        const size_t t = std::min(ink_chunk_, tokens.size() - off);
        if (!forward_inkling2_chunk(tokens.data() + off, t, err)) return false;
    }
    if (!gpu_->sync(err)) return false;
    // The unpadded vocabulary only: the padded tail is garbage and the decode path strips it too.
    const ModelConfig& c = m_.cfg;
    const size_t nv = c.n_vocab_unpadded > 0 ? c.n_vocab_unpadded : c.n_vocab;
    if (!get(*gpu_, logits_, 0, nv, logits, err)) return false;
    if (!adapt_experts(err)) return false;
    if (reset_counts_after_prefill_) reset_counts();
    return true;
}

// port of gpu_model/prefill.rs: GpuModel::adapt_experts (a fully resident cache has no hot missed expert to place)
bool GpuModel::adapt_experts(std::string& err) {
    if (adapt_every > 0 && cache_ && !cache_->fully_resident()) return cache_->settle(m_, *gpu_, err);
    return true;
}

// port of gpu_model/prefill.rs: GpuModel::prefill_chunk (one part; `t` tokens from pos_)
bool GpuModel::prefill_chunk(const uint32_t* toks, size_t t, std::string& err) {
    const ModelConfig& c = m_.cfg;
    Backend& g = *gpu_;
    PrefillScratch& pf = *pf_;
    const size_t ne = c.n_embd;
    const size_t n = pf.chunk;
    const size_t dstk = hcn_ * ne;  // one token's stack (hc_n rows of ne; ne without hyper-connections)
    // Missed experts are read from the mapped weights in the order the staging copies take them.
    std::unique_ptr<Prefetch> pre;
    if (cache_) {
        pre = std::make_unique<Prefetch>();
        pre->start();
    }
    // The embeddings of the chunk into the x region (each token's stack of hc_n rows, repeated from the embedding), and
    // the PLE inputs of the chunk into the emb region, as put_embed does for one token.
    const double tm_embed = pf_t();
    if (pft().on) pft().chunks++;
    std::vector<float> stack;
    stack.reserve(t * dstk);
    std::vector<float> row;
    for (size_t r = 0; r < t; r++) {
        if (!m_.embed(toks[r], row, err)) return false;
        for (size_t s = 0; s < hcn_; s++) stack.insert(stack.end(), row.begin(), row.end());
    }
    if (!put(g, stack.data(), stack.size(), pf.slab, 0, err)) return false;
    if (has_ple_) {
        std::vector<float> pe, one;
        for (size_t r = 0; r < t; r++) {
            if (!m_.ple_embed(ple_prev_, toks[r], one, err)) return false;
            pe.insert(pe.end(), one.begin(), one.end());
        }
        if (!put(g, pe.data(), pe.size(), pf.slab, pf.lay.emb, err)) return false;
    }
    // Per-token values of the chunk's first position: [position, 0] and the RoPE position (rope_offset is 0 for text).
    const int32_t sb[2] = {static_cast<int32_t>(pos_), 0};
    if (!put(g, sb, 2, step_buf_, 0, err)) return false;
    if (!put(g, sb, 2, rope_buf_, 0, err)) return false;
    pf_add(kPhEmbed, tm_embed);
    const uint64_t step = step_buf_.addr();
    const uint64_t rope = rope_buf_.addr();
    const uint64_t base = pf.slab.addr();
    auto at = [&](size_t o) { return base + 4 * uint64_t(o); };
    const PrefillLay& L = pf.lay;
    const uint64_t x = at(0);
    const uint64_t h = at(L.h);
    // port of gpu_model/prefill.rs: the hyper-connection scratch rows (gemm: the reads are batched GEMMs)
    HcAt ha;
    ha.xn = at(L.xn);
    ha.lo = at(L.lo);
    ha.hg = at(L.hg);
    ha.inj = at(L.inj);
    ha.h = h;
    ha.hq = 0;
    ha.emb = at(L.emb);
    ha.pk = at(L.pk);
    ha.pq = at(L.pq);
    ha.pg = at(L.pg);
    ha.pn = at(L.pn);
    ha.pv = at(L.pv);
    ha.gemm = true;
    // The mixer's output goes to "mix" with hyper-connections (then into the streams), else straight into x (residual).
    const bool hc = c.hc_n > 0;
    const uint64_t mx = hc ? at(L.mix) : x;
    const int32_t mact = hc ? 0 : 2;
    const size_t dk = c.ssm_d_state, nk = c.ssm_n_group, nv = c.ssm_dt_rank;
    const size_t dv = c.ssm_dv();
    const size_t channels = c.ssm_d_inner + 2 * nk * dk;
    const size_t hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv;
    const size_t kv = nkv * hd;
    const size_t qs = c.q_gate ? 2 * hd : hd;
    const float eps = c.eps;
    for (size_t il = 0; il < layers_.size(); il++) {
        const GpuLayer& ly = layers_[il];
        // port of gpu_model/prefill.rs: the PLE block first, then the mixer input (the plain norm, or the hc read)
        const double tm_ple = pf_t();
        if (ly.has_ple && !ple_block(ly.ple, x, step, t, ha, err)) return false;
        pf_add(kPhPle, tm_ple);
        const double tm_hc = pf_t();
        if (ly.hc) {
            if (!hc_read(ly.hcr[0], x, t, ha, err)) return false;
        } else if (!g.rms_norm(x, ne, ly.attn_norm.ptr, h, ne, ne, t, eps, err)) {
            return false;
        }
        pf_add(kPhHc, tm_hc);
        const double tm_mix = pf_t();
        if (ly.delta) {
            // port of prefill.rs: Mixer::Delta
            const uint64_t eb = at(L.e), eg = at(L.e + n * nv);
            if (!g.gemm_dense(ly.qkv, t, h, ne, at(L.a), channels, 0, err)) return false;
            // The qkv GEMM above already quantized h; prepare it now when it did not, so beta, alpha and the gate reuse it.
            if (!g.prequant_gemm_x(ly.dn_gate, t, h, ne, err)) return false;
            if (!g.conv_seq(at(L.a), ly.conv.addr(), ly.conv_w.ptr, at(L.c), channels, c.ssm_conv, t, 0, 0, err)) return false;
            // The beta and alpha GEMMs run beside the q/k l2 norms.
            if (!batch(g,
                       [&](std::string& e) {
                           return g.gemm_dense(ly.beta, t, h, ne, eb, nv, SAME_X, e) &&
                                  g.l2_norm(at(L.c), dk, 2 * nk, channels, t * 2 * nk, eps, e) &&
                                  g.gemm_dense(ly.alpha, t, h, ne, eg, nv, SAME_X, e);
                       },
                       err))
                return false;
            if (!g.ssm_gates(eb, eg, ly.dt_bias.ptr, ly.a.ptr, t * nv, nv, err)) return false;
            // The z GEMM runs beside the scan.
            if (!batch(g,
                       [&](std::string& e) {
                           return g.deltanet(at(L.c), ly.s.addr(), eb, eg, 0, 0, at(L.d), nk, nv, c.k_interleave, dk, dv,
                                             1.0f / std::sqrt(static_cast<float>(dk)), t, channels, 0, 0, e) &&
                                  g.gemm_dense(ly.dn_gate, t, h, ne, at(L.b), c.ssm_d_inner, SAME_X, e);
                       },
                       err))
                return false;
            // The output gate: sigmoid(z) in the hyper-connection models (norm_gate_sig), silu(z) otherwise.
            if (hc) {
                uint64_t xq_unused = 0;
                if (!g.norm_gate_sig(at(L.d), ly.norm.ptr, at(L.b), dv, t * nv, eps, 0, xq_unused, err)) return false;
            } else if (!g.norm_gate(at(L.d), ly.norm.ptr, at(L.b), dv, t * nv, eps, err)) {
                return false;
            }
            if (!g.gemm_dense(ly.out, t, at(L.d), c.ssm_d_inner, mx, ne, mact, err)) return false;
        } else {
            // port of prefill.rs: Mixer::Attn
            const uint64_t kraw = at(L.c), kk = at(L.c + n * kv), vv = at(L.c + 2 * n * kv);
            const bool use_rope = il < c.rope_layer.size() ? c.rope_layer[il] : true;
            if (!g.gemm_dense(ly.q, t, h, ne, at(L.a), nh * qs, 0, err)) return false;
            if (!g.rms_norm(at(L.a), qs, ly.qn.ptr, at(L.b), hd, hd, t * nh, eps, err)) return false;
            if (use_rope && !g.rope_neox(at(L.b), hd, t * nh, nh, c.n_rot, rope, c.rope_base, err)) return false;
            if (!g.gemm_dense(ly.k, t, h, ne, kraw, kv, SAME_X, err)) return false;
            if (!g.gemm_dense(ly.v, t, h, ne, vv, kv, SAME_X, err)) return false;
            if (!g.rms_norm(kraw, hd, ly.kn.ptr, kk, hd, hd, t * nkv, eps, err)) return false;
            if (use_rope && !g.rope_neox(kk, hd, t * nkv, nkv, c.n_rot, rope, c.rope_base, err)) return false;
            if (!g.kv_store(kk, vv, ly.kc.addr(), ly.vc.addr(), kv, t, step, hd, err)) return false;
            // port of prefill.rs: the QSA indexer. Region c (free after kv_store) holds the raw indexer keys, the raw
            // queries and the normed queries. Rows with at most `top` complete blocks attend densely (no mask).
            uint64_t qm = 0;
            size_t mw = 0;
            if (ly.has_qsa) {
                const QsaBlock& a = ly.qsa;
                const uint64_t ik = at(L.c), iq = at(L.c + n * QSA_D), iqn = at(L.c + n * (1 + QSA_H) * QSA_D);
                const size_t dq = QSA_D, nq = QSA_H * QSA_D;
                if (!g.gemm_dense(a.kp, t, h, ne, ik, dq, 0, err)) return false;
                if (!g.gemm_dense(a.qp, t, h, ne, iq, nq, SAME_X, err)) return false;
                if (!g.qsa_pool(ik, a.ring, a.pool.addr(), a.kn.ptr, step, t, c.n_rot, c.rope_base, eps, err)) return false;
                if (pos_ + t >= QSA_POOL * (qsa_.top + 1)) {
                    if (!g.rms_norm(iq, dq, a.qn.ptr, iqn, dq, dq, t * QSA_H, eps, err)) return false;
                    if (!g.rope_neox(iqn, dq, t * QSA_H, QSA_H, c.n_rot, rope, c.rope_base, err)) return false;
                    for (size_t r0 = 0; r0 < t; r0 += QSA_ROWS) {
                        if (!g.qsa_select(iqn + 4 * uint64_t(r0 * nq), a.pool.addr(), qsa_.score.addr(), 0,
                                          qsa_.mask.addr() + 4 * uint64_t(r0 * qsa_.mw), step, r0, std::min(QSA_ROWS, t - r0),
                                          qsa_.nbs, qsa_.top, 0, qsa_.mw, err))
                            return false;
                    }
                    qm = qsa_.mask.addr();
                    mw = qsa_.mw;
                }
            }
            const size_t win = (il < c.swa.size() && c.swa[il]) ? c.n_swa : 0;
            if (!g.attn_prefill_m(at(L.b), ly.kc.addr(), ly.vc.addr(), c.q_gate ? at(L.a) : 0, at(L.d), step, t, nh, nkv, hd,
                                  1.0f / std::sqrt(static_cast<float>(hd)), qm, mw, ly.sink.ptr, win, err))
                return false;
            if (!g.gemm_dense(ly.o, t, at(L.d), nh * hd, mx, ne, mact, err)) return false;
        }
        if (!pf_fence(g, err)) return false;
        pf_add(kPhMix, tm_mix);
        // port of prefill.rs: the mixer's output joins the residual (in the streams, with hyper-connections); h = the FFN input
        const double tm_hc2 = pf_t();
        if (!ly.hc) {
            if (!g.rms_norm(x, ne, ly.post_norm.ptr, h, ne, ne, t, eps, err)) return false;
        } else {
            if (!g.hc_write(x, mx, at(L.inj), ne, c.hc_n, t, false, err)) return false;
            if (!hc_read(ly.hcr[1], x, t, ha, err)) return false;
        }
        pf_add(kPhHc, tm_hc2);
        if (ly.moe) {
            // The FFN output collects in "mo" (zero: hc_write clears it) with hyper-connections, else into x.
            const uint64_t fx = ly.hc ? at(L.mo) : x;
            if (!prefill_moe(il, ly, t, base, fx, pre.get(), err)) return false;
            if (ly.hc && !g.hc_write(x, at(L.mo), at(L.inj), ne, c.hc_n, t, true, err)) return false;
        } else {
            // port of prefill.rs: Ffn::Dense
            const double tm_dense = pf_t();
            if (!g.gemm_dense(ly.gate, t, h, ne, at(L.ge), c.n_ff, 0, err)) return false;
            if (!g.gemm_dense(ly.up, t, h, ne, at(L.ge), c.n_ff, 1 | SAME_X, err)) return false;
            if (!g.gemm_dense(ly.down, t, at(L.ge), c.n_ff, x, ne, 2, err)) return false;
            if (!pf_fence(g, err)) return false;
            pf_add(kPhDense, tm_dense);
        }
        if (!g.kick(err)) return false;
    }
    // Row 0 of x gets the chunk's last stack (as after `step`).
    if (!g.dtod(pf.slab, 4 * (t - 1) * dstk, x_, 0, 4 * dstk, err)) return false;
    pos_ += t;
    return true;
}

// port of gpu_model/prefill.rs: prefill_chunk's Ffn::Moe branch for layer `il`. The routed experts' sum (and the shared
// expert's) goes into `fx`: x itself, or the FFN collector "mo" with hyper-connections.
bool GpuModel::prefill_moe(size_t il, const GpuLayer& ly, size_t t, uint64_t base, uint64_t fx, Prefetch* pre, std::string& err) {
    const ModelConfig& c = m_.cfg;
    Backend& g = *gpu_;
    PrefillScratch& pf = *pf_;
    const PrefillLay& L = pf.lay;
    const size_t n = pf.chunk, ne = c.n_embd, k = c.n_expert_used;
    auto at = [&](size_t o) { return base + 4 * uint64_t(o); };
    const uint64_t h = at(L.h);
    const uint64_t ids_d = pf.ints.addr();
    if (!cache_) {
        err = "MoE layer without an expert cache";
        return false;
    }
    // The router, and the shared expert's gate_inp (a 1-row GEMM), beside each other.
    if (pft().on) {
        pft().moe_layers++;
    }
    const double tm_route = pf_t();
    if (!batch(g,
               [&](std::string& e) {
                   if (!g.gemm_dense(ly.router, t, h, ne, at(L.rl), c.n_expert, 0, e)) return false;
                   return !(ly.shared && ly.has_shared_gate_inp) ||
                          g.gemm_dense(ly.sh_gate_inp, t, h, ne, at(L.sg), 1, SAME_X, e);
               },
               err))
        return false;
    if (!g.router_topk(at(L.rl), c.n_expert, k, c.expert_weights_scale, ids_d, at(L.wts), t, c.router_sigmoid, 0, 0.0f, err)) return false;
    uint64_t sh = 0, sg = 0;
    if (ly.shared) {
        if (!g.gemm_dense(ly.sh_gate, t, h, ne, at(L.shg), c.n_ff_shexp, SAME_X, err)) return false;
        if (!g.gemm_dense(ly.sh_up, t, h, ne, at(L.shg), c.n_ff_shexp, 1 | SAME_X, err)) return false;
        sh = at(L.shy);
        if (ly.has_shared_gate_inp) sg = at(L.sg);
    }
    if (!pf_fence(g, err)) return false;
    pf_add(kPhRoute, tm_route);
    // The routed ids of the chunk (t rows of k). Ids outside 0..n_expert mean the router ran on corrupt logits.
    const double tm_ids = pf_t();
    std::vector<int32_t> ids;
    if (!get(g, pf.ints, 0, t * k, ids, err)) return false;
    for (int32_t e : ids) {
        if (e < 0 || static_cast<size_t>(e) >= c.n_expert) {
            err = "prefill: router gave expert id " + std::to_string(e) + " at layer " + std::to_string(il) + ", chunk of " +
                  std::to_string(t) + " tokens from position " + std::to_string(pos_) + "; lower the chunk or the ctx";
            return false;
        }
    }
    cache_->record_prompt(il, ids);
    pf_add(kPhIds, tm_ids);
    const double tm_build = pf_t();
    // Sort the (token, j) pairs by expert: slot s holds token xrow[s]; inv maps t * k + j to its slot.
    std::vector<size_t> off(c.n_expert + 1, 0);
    for (int32_t e : ids) off[static_cast<size_t>(e) + 1]++;
    for (size_t e = 0; e < c.n_expert; e++) off[e + 1] += off[e];
    std::vector<size_t> fill(off);
    std::vector<int32_t> xrow(t * k, 0), inv(t * k, 0);
    for (size_t i = 0; i < ids.size(); i++) {
        size_t& s = fill[static_cast<size_t>(ids[i])];
        xrow[s] = static_cast<int32_t>(i / k);
        inv[i] = static_cast<int32_t>(s);
        s++;
    }
    if (!put(g, xrow.data(), xrow.size(), pf.ints, n * k, err)) return false;
    if (!put(g, inv.data(), inv.size(), pf.ints, 2 * n * k, err)) return false;

    const LayerExperts* lx = cache_->layer(il);
    if (!lx) {
        err = "MoE layer " + std::to_string(il) + " has no expert cache entry";
        return false;
    }
    const auto pool_stride = cache_->base(*lx);
    const uint64_t pool = pool_stride.first;
    const size_t stride = pool_stride.second;
    auto used = [&](size_t e) { return off[e + 1] > off[e]; };
    std::vector<size_t> missed;
    for (size_t e = 0; e < c.n_expert; e++) {
        if (used(e) && cache_->slot(il, e) < 0) missed.push_back(e);
    }
    if (pft().on) {
        pft().missed += missed.size();
        pft().copy_bytes += missed.size() * (lx->mats[0].bytes + lx->mats[1].bytes + lx->mats[2].bytes);
    }
    if (pre) {
        // port of prefill.rs: the Prefetch request (the mapped weights of the staged groups, in copy order)
        std::vector<std::pair<size_t, size_t>> rs;
        Tensor w[3];
        for (size_t mi = 0; mi < 3; mi++) {
            if (!m_.w.layer(il, kExpertMats[mi], w[mi], err)) return false;
        }
        for (size_t g0 = 0; g0 < missed.size(); g0 += kStage) {
            const size_t g1 = std::min(missed.size(), g0 + kStage);
            for (size_t mi = 0; mi < 3; mi++) {
                const ExpertMat& mat = lx->mats[mi];
                size_t a = g0;
                while (a < g1) {
                    size_t b = a + 1;
                    while (b < g1 && missed[b] == missed[b - 1] + 1) b++;
                    const uintptr_t addr = reinterpret_cast<uintptr_t>(w[mi].data + missed[a] * mat.bytes);
                    rs.push_back({static_cast<size_t>(addr), (b - a) * mat.bytes});
                    a = b;
                }
            }
        }
        pre->send(std::move(rs));
    }
    // Staged groups of at most kStage missed experts, group j in staging third j % 3. Staging holds each matrix in its
    // own section (all gates, then ups, then downs), so experts with consecutive ids are contiguous there.
    const uint64_t sp = pf.staging.addr();
    const size_t third = pf.staging.len() / 3;
    auto half = [&](size_t j) { return (j % 3) * third; };
    std::vector<Group> staged;
    for (size_t j = 0; j * kStage < missed.size(); j++) {
        Group grp;
        const size_t cnt = std::min(kStage, missed.size() - j * kStage);
        for (size_t i = 0; i < cnt; i++) {
            const size_t e = missed[j * kStage + i];
            std::array<uint64_t, 3> a{};
            for (size_t mi = 0; mi < 3; mi++) {
                const ExpertMat& mat = lx->mats[mi];
                a[mi] = sp + uint64_t(half(j) + kStage * mat.off + i * mat.bytes);
            }
            grp.push_back({e, a});
        }
        staged.push_back(std::move(grp));
    }
    // The resident experts split into one chunk per staged group, of about equal positions, so the copies run beside
    // them too: step s runs group s - 2 and chunk s beside the copy of group s (into the third group s - 3 used, done in
    // step s - 1). Without overlap one chunk.
    const size_t ns = g.beside_overlaps() ? std::max<size_t>(staged.size(), 1) : 1;
    Group res;
    for (size_t e = 0; e < c.n_expert; e++) {
        if (!used(e)) continue;
        const int32_t s = cache_->slot(il, e);
        if (s < 0) continue;
        std::array<uint64_t, 3> a{};
        for (size_t mi = 0; mi < 3; mi++) a[mi] = pool + uint64_t(static_cast<size_t>(s) * stride + lx->mats[mi].off);
        res.push_back({e, a});
    }
    size_t total = 0;
    for (const Item& r : res) total += off[r.first + 1] - off[r.first];
    total = std::max<size_t>(total, 1);
    std::vector<Group> chunks(ns);
    size_t acc = 0;
    for (const Item& r : res) {
        chunks[std::min(acc * ns / total, ns - 1)].push_back(r);
        acc += off[r.first + 1] - off[r.first];
    }
    // Per step: its groups (a range of `groups`) and the staged group it copies.
    struct Step {
        size_t g0, g1;
        std::optional<size_t> copy;
    };
    std::vector<Group> groups;
    std::vector<Step> steps;
    size_t next_chunk = 0;
    for (size_t s = 0; s < staged.size() + 2; s++) {
        const size_t g0 = groups.size();
        if (s >= 2 && s - 2 < staged.size()) groups.push_back(staged[s - 2]);
        if (next_chunk < chunks.size()) groups.push_back(chunks[next_chunk++]);
        steps.push_back({g0, groups.size(), s < staged.size() ? std::optional<size_t>(s) : std::nullopt});
    }
    // The GEMM tile list: [w, m0 | count << 32] pairs. Per group a matrix's tiles run in expert order.
    std::vector<uint64_t> tiles;
    auto push = [&](const Group& grp, size_t mi, size_t lo) -> std::pair<size_t, size_t> {
        const size_t start = tiles.size() / 2;
        for (const Item& it : grp) {
            const size_t e = it.first;
            for (size_t m0 = off[e]; m0 < off[e + 1]; m0 += GEMM_BM) {
                tiles.push_back(it.second[mi]);
                tiles.push_back(uint64_t(m0 - lo) | (uint64_t(std::min(GEMM_BM, off[e + 1] - m0)) << 32));
            }
        }
        return {start, tiles.size() / 2 - start};
    };
    // Per step: the gate and up tiles of all its groups (one launch each), then per group its down tiles and the slot
    // rows lo..hi it covers. The down GEMM reads rows lo..hi of ge only.
    struct Down {
        std::pair<size_t, size_t> tile;
        size_t lo, hi;
    };
    struct Span {
        std::pair<size_t, size_t> gu[2];
        std::vector<Down> downs;
    };
    std::vector<Span> spans;
    for (const Step& st : steps) {
        Span sp_;
        for (size_t mi = 0; mi < 2; mi++) {
            const size_t s0 = tiles.size() / 2;
            for (size_t gi = st.g0; gi < st.g1; gi++) push(groups[gi], mi, 0);
            sp_.gu[mi] = {s0, tiles.size() / 2 - s0};
        }
        for (size_t gi = st.g0; gi < st.g1; gi++) {
            const Group& grp = groups[gi];
            if (grp.empty()) continue;
            const size_t lo = off[grp.front().first];
            const size_t hi = off[grp.back().first + 1];
            sp_.downs.push_back({push(grp, 2, lo), lo, hi});
        }
        spans.push_back(std::move(sp_));
    }
    if (8 * tiles.size() > pf.tiles.len()) {
        err = "prefill tile list overflow";
        return false;
    }
    if (!tiles.empty() && !put(g, tiles.data(), tiles.size(), pf.tiles, 0, err)) return false;
    const uint64_t tp_base = pf.tiles.addr();
    auto tp = [&](size_t s) { return tp_base + 16 * uint64_t(s); };
    const uint64_t xrow_d = pf.ints.addr() + 4 * uint64_t(n * k);
    const ExpertMat& mg = lx->mats[0];
    const ExpertMat& mu = lx->mats[1];
    const ExpertMat& md = lx->mats[2];
    // The copies of staged group j: each matrix's runs of consecutive ids, from the mapped weights.
    auto copy = [&](size_t j, std::string& e) -> bool {
        const Group& grp = staged[j];
        for (size_t mi = 0; mi < 3; mi++) {
            Tensor w;
            if (!m_.w.layer(il, kExpertMats[mi], w, e)) return false;
            const ExpertMat& mat = lx->mats[mi];
            size_t i = 0;
            size_t a = 0;
            while (a < grp.size()) {
                size_t b = a + 1;
                while (b < grp.size() && grp[b].first == grp[b - 1].first + 1) b++;
                const size_t ex = grp[a].first;
                if (!g.htod(w.data + ex * mat.bytes, (b - a) * mat.bytes, pf.staging,
                            static_cast<size_t>(half(j) + kStage * mat.off + i * mat.bytes), e))
                    return false;
                i += b - a;
                a = b;
            }
        }
        return true;
    };
    pf_add(kPhBuild, tm_build);
    // `beside` starts a step's copies after the GEMMs before it (they queue last, so the GEMMs need not wait for them).
    const double tm_enq = pf_t();
    for (size_t si = 0; si < steps.size(); si++) {
        const Span& sp_ = spans[si];
        const std::optional<size_t> cp = steps[si].copy;
        if (!g.beside(
                [&](std::string& e) {
                    if (sp_.gu[0].second > 0) {
                        if (!g.gemm(mg.ty, mg.rows, mg.cols, tp(sp_.gu[0].first), sp_.gu[0].second, h, ne, t, xrow_d, at(L.ge),
                                    c.n_ff_exp, SAME_X, e))
                            return false;
                        if (!g.gemm(mu.ty, mu.rows, mu.cols, tp(sp_.gu[1].first), sp_.gu[1].second, h, ne, t, xrow_d, at(L.ge),
                                    c.n_ff_exp, 1 | SAME_X, e))
                            return false;
                    }
                    for (const Down& d : sp_.downs) {
                        if (!g.gemm(md.ty, md.rows, md.cols, tp(d.tile.first), d.tile.second,
                                    at(L.ge) + 4 * uint64_t(d.lo * c.n_ff_exp), c.n_ff_exp, d.hi - d.lo, 0,
                                    at(L.ye) + 4 * uint64_t(d.lo * ne), ne, 0, e))
                            return false;
                    }
                    if (cp) {
                        const double tm_copy = pf_t();
                        const bool ok_copy = copy(*cp, e);
                        pf_add(kPhCopy, tm_copy);
                        if (!ok_copy) return false;
                    }
                    return true;
                },
                err))
            return false;
    }
    pf_add(kPhEnq, tm_enq);
    const double tm_join = pf_t();
    if (!g.beside_join(err)) return false;
    if (!pf_fence(g, err)) return false;
    pf_add(kPhJoin, tm_join);
    // The shared down runs last, so the routed gate and up reuse the int8 h of the shared up.
    if (ly.shared && !g.gemm_dense(ly.sh_down, t, at(L.shg), c.n_ff_shexp, at(L.shy), ne, 0, err)) return false;
    // Expert rows are slots (token, j) sorted by expert: the weighted sum over each token's k slots goes into fx.
    return g.moe_combine_seq(fx, at(L.ye), at(L.wts), ids_d + 4 * uint64_t(2 * n * k), k, sh, sg, ne, t, err);
}

}  // namespace strata
