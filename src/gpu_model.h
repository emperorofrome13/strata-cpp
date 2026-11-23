// port of gpu_model.rs: the GPU decode forward for the qwen35moe and qwen4exp (Flash-Next) families (DeltaNet and
// full-attention layers, hyper-connection streams, per-layer embeddings (PLE), QSA sparse attention, MoE with the
// routed experts split between the VRAM expert cache and the CPU expert worker, the shared expert), its weights, the
// KV cache and the recurrent state, and the CUDA graph of the decode step.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Scope (see MODIFICATIONS.md, sections W3-E1a, W3-E3 and W3-E1b): one part, one slot, no MTP draft head, no EAGLE3
// head, no LoRA. Hyper-connections, PLE and QSA are ported in W3-E1b.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "cpu_worker.h"
#include "expert_cache.h"
#include "model.h"

namespace strata {

class Prefetch;  // weights.h

// port of gpu_model.rs: QSA_POOL (cells per pooled indexer key), QSA_D (indexer head size), QSA_H (indexer heads),
// QSA_RING (raw indexer key floats a layer keeps: the last 32 positions), QSA_ROWS (score rows per prefill launch)
constexpr size_t QSA_POOL = 4;
constexpr size_t QSA_D = 128;
constexpr size_t QSA_H = 4;
constexpr size_t QSA_RING = 32 * QSA_D;
constexpr size_t QSA_ROWS = 64;

// port of gpu_model.rs: has_indexer (layer `il` has an indexer.k_proj.weight) and qsa_on (the config has a top_k and some
// layer has an indexer)
bool has_indexer(const Model& m, size_t il);
bool qsa_active(const Model& m);

// port of gpu_model.rs: HcRead (one hyper-connection read: the norm, down and up projections and the optional inject)
struct HcRead {
    GpuTensor norm, down, up, inject;
    bool has_inject = false;
};

// port of gpu_model.rs: the qwen4exp PLE block of a layer (key, value, the three norms and the conv, as f32 [channel][tap])
struct PleBlock {
    GpuTensor key, value, norm_key, norm_query, norm_conv;
    Buf conv;
};

// port of gpu_model.rs: Qsa (the QSA indexer of an attention layer: projections and norms, the pooled keys and this
// layer's rows of the raw indexer key ring)
struct QsaBlock {
    GpuTensor kp, qp, kn, qn;
    Buf pool;
    uint64_t ring = 0;
};

// StrataFreet addition (INKLING2_NOTES.md): the inkling2 block's weights beyond the shared q/k/v/o/gate/router
// fields of GpuLayer, and its per-slot shortconv state. The loop gates exist on the core blocks only.
struct Inkling2Block {
    GpuTensor rel;         // attn_rel {n_embd, n_rel * n_head}
    GpuTensor rel_bias;    // attn_rel_bias {n_rel, n_extent} (padded to n_extent + 1 on the device)
    GpuTensor q_clip;      // attn_q_clip {head_dim, n_head}
    GpuTensor rel_clip;    // attn_rel_clip {n_rel, n_head}
    GpuTensor sink_proj;   // attn_sink_proj {n_embd, n_head} (a projection, not a per-head bias)
    GpuTensor k_conv;      // attn_k_conv {n_shortconv, n_embd_k_gqa}
    GpuTensor v_conv;      // attn_v_conv {n_shortconv, n_embd_k_gqa}
    GpuTensor k_conv_proj; // attn_k_conv_proj {n_embd, n_shortconv * n_head_kv} (full-attention blocks)
    GpuTensor v_conv_proj; // attn_v_conv_proj {n_embd, n_shortconv * n_head_kv} (full-attention blocks)
    GpuTensor post_conv;   // attn_post_conv {n_shortconv, n_embd}
    GpuTensor ffn_post_conv;  // ffn_post_conv {n_shortconv, n_embd}
    GpuTensor q_inp_norm, k_inp_norm, v_inp_norm;  // full-attention blocks only
    GpuTensor res_q;       // attn_res_query {n_embd, 1 or 3}
    GpuTensor ffn_res_q;   // ffn_res_query {n_embd}
    GpuTensor ffn_norm;    // ffn_norm {n_embd} (W2: the MoE input norm; missed by the W1 loader)
    GpuTensor exp_probs_b; // ffn_exp_probs_b {n_expert} (router bias, optional)
    // StrataFreet addition (W2): the per-expert f32 scales as DEDICATED device tensors ({n_expert} each), indexed
    // by expert id like the CPU worker reads them (the pool-slot copy stays unused; the W1 pool placement's scale
    // region read garbage in the decode).
    GpuTensor gu_scale, dn_scale;
    // Loop gates of the core blocks: (attn, k, v, mlp) x (down {2*n_embd, rank}, gain/forget {rank, n_embd}).
    GpuTensor loop_down[4], loop_gain[4], loop_forget[4];
    Buf conv_state;        // (n_shortconv - 1) * n_sconv_channels floats of shortconv history
};

// port of gpu_model.rs: GpuLayer. The device weights and state of one decoder layer; fields a layer kind does not use
// stay empty.
struct GpuLayer {
    bool delta = false;                     // a DeltaNet (recurrent) layer; false: full attention
    // StrataFreet addition (INKLING2_NOTES.md): an inkling2 block; the q/k/v/o/gate/router fields of
    // GpuLayer carry its shared weights, attn_norm its SWA norm (empty on the full-attention blocks).
    bool inkling2 = false;
    Inkling2Block ink;
    GpuTensor attn_norm, post_norm;         // attn_norm, post_attention_norm (empty with hyper-connections)
    // DeltaNet: attn_qkv, attn_gate (`dn_gate`), ssm_beta, ssm_alpha, ssm_dt.bias, ssm_a, ssm_conv1d, ssm_norm, ssm_out
    GpuTensor qkv, dn_gate, beta, alpha, dt_bias, a, conv_w, norm, out;
    Buf conv;  // (ssm_conv - 1) * channels floats of conv state
    Buf s;     // dt_rank * head_v * head_k floats of DeltaNet state
    // Attention: attn_q (with the output gate when q_gate), attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output
    GpuTensor q, k, v, qn, kn, o;
    GpuTensor sink;  // attn_sinks (empty unless the model has them)
    Buf kc, vc;      // f16 KV cache, ctx rows of n_head_kv * head_dim
    // FFN: a MoE layer has the router, and a shared expert when ffn_up_shexp exists; a dense layer has gate/up/down.
    bool moe = false;
    GpuTensor gate, up, down;               // dense FFN (ffn_gate, ffn_up, ffn_down)
    GpuTensor router;                       // ffn_gate_inp
    bool shared = false;                    // ffn_up_shexp exists
    bool has_shared_gate_inp = false;       // ffn_gate_inp_shexp exists (sigmoid gate of the shared expert)
    GpuTensor sh_gate_inp, sh_gate, sh_up, sh_down;
    // qwen4exp: the hyper-connection reads of the mixer (hc[0]) and the FFN (hc[1]); the PLE block; the QSA indexer.
    bool hc = false;
    HcRead hcr[2];
    bool has_ple = false;
    PleBlock ple;
    bool has_qsa = false;
    QsaBlock qsa;
};

// port of gpu_model.rs: the load options that come from the command line (and the environment, read at load as Rust
// does): ctx, the VRAM margin, the CPU threads of the expert passes (0: the physical cores, as Strata's default), the
// prefill chunk in tokens (0: no batched prefill) and the short-prompt threshold (prompts shorter than it run token by
// token, as prefill.rs's `short`).
struct LoadOptions {
    size_t ctx = 4096;
    size_t vram_margin = size_t(512) << 20;
    size_t threads = 0;
    size_t chunk = 4096;
    size_t short_len = 32;
};

// port of gpu_model.rs: HcAt (the scratch rows of hc_read and ple: the decode buffers, or the prefill slab where gemm is
// true and the projections are batched GEMMs)
struct HcAt {
    uint64_t xn = 0, lo = 0, hg = 0, inj = 0, h = 0, hq = 0, emb = 0, pk = 0, pq = 0, pg = 0, pn = 0, pv = 0;
    bool gemm = false;
};

// port of gpu_model/prefill.rs: Lay. Float offsets of the regions of the prefill slab (each region holds `chunk` rows;
// the mixer and FFN regions alias). The hyper-connection regions (xn..emb) are zero when the model has none.
struct PrefillLay {
    size_t h = 0, a = 0, b = 0, c = 0, d = 0, e = 0, rl = 0, wts = 0, ge = 0, ye = 0, shg = 0, shy = 0, sg = 0;
    size_t xn = 0, lo = 0, hg = 0, pk = 0, pq = 0, pg = 0, pn = 0, pv = 0, inj = 0, mix = 0, mo = 0, emb = 0;
};

// port of gpu_model/prefill.rs: Scratch::new's sizes (the slab layout and the element counts of the other buffers), so
// the load plan can count them before any allocation.
struct PrefillSizes {
    PrefillLay lay;
    size_t slab_floats = 0;
    size_t ints = 0;     // int32 elements
    size_t tiles = 0;    // u64 elements
    size_t staging = 0;  // bytes
};
bool prefill_sizes(const Model& m, size_t chunk, PrefillSizes& out, std::string& err);

// port of gpu_model/prefill.rs: Scratch. The slab of activations, the id / slot-row / inverse-slot ints, the GEMM
// tile list and the staging of the experts that are not resident.
struct PrefillScratch {
    size_t chunk = 0;
    PrefillLay lay;
    Buf slab;     // floats: x (chunk rows of hc * n_embd) at 0, then the regions of `lay`
    Buf ints;     // int32: ids (chunk * k), xrow (chunk * k), inv (chunk * k)
    Buf tiles;    // u64 pairs (weight address, m0 | count << 32), 16 bytes a tile
    Buf staging;  // three staging groups of STAGE missed experts each
    size_t bytes() const { return slab.len() + ints.len() + tiles.len() + staging.len(); }
};

// port of gpu_model.rs: QsaBufs (the QSA scratch shared by the layers): the raw indexer keys of each layer's last 32
// positions (QSA_RING floats a layer), the selection of the decode step, the scores and the prefill mask.
struct QsaBufs {
    Buf ring, score, sel, mask;
    size_t top = 0, nbs = 0, sst = 0, mw = 0;
};

// port of gpu_model.rs: GpuModel (one part, one slot).
class GpuModel {
public:
    // port of gpu_model.rs: GpuModel::load (one part, one slot, no MTP draft head). `gpu` is owned by the model and is
    // freed after every device buffer it made. The expert cache takes the VRAM beyond `vram_margin` (the Rust
    // budget: free - margin, capped by STRATA_EXPERT_MB); experts that do not fit run on the CPU worker.
    static bool load(const Model& m, std::unique_ptr<Backend> gpu, const LoadOptions& o, std::unique_ptr<GpuModel>& out,
                     std::string& err);

    GpuModel(const GpuModel&) = delete;
    GpuModel& operator=(const GpuModel&) = delete;
    ~GpuModel();

    // port of gpu_model.rs: GpuModel::step. Runs `token` at the current position and returns the logits of the next
    // token (`cfg.n_vocab` floats).
    bool step(uint32_t token, std::vector<float>& logits, std::string& err);
    // port of gpu_model.rs: GpuModel::step_greedy. The same step, returning only the argmax of the logits (taken on
    // the device: the logits never leave it).
    bool step_greedy(uint32_t token, int32_t& out, std::string& err);
    // port of gpu_model/prefill.rs: GpuModel::prefill (prefill_with, one part, no between hook). Runs `tokens` from the
    // current position and returns the logits after the last one. Prompts shorter than `short_len` (and models loaded
    // without a chunk) go through `step` token by token; the rest through the chunked batched prefill.
    bool prefill(const std::vector<uint32_t>& tokens, std::vector<float>& logits, std::string& err);
    // port of expert_cache.rs: ExpertCache::reset_counts (through the model, as main.rs's reset_counts)
    void reset_counts() {
        if (cache_) cache_->reset_counts();
    }
    // port of gpu_model.rs: GpuModel::reset. Clears the KV cache and the recurrent state; the position goes back to 0.
    bool reset(std::string& err);
    // port of expert_cache.rs: ExpertCache::save_profile, as main.rs calls it at the end of a run.
    bool save_profile(std::string& err);

    // Positions run so far (the KV rows in use) and the context size in tokens.
    size_t pos() const { return pos_; }
    size_t ctx() const { return ctx_; }
    // The backend's name ("cuda") and device name.
    Backend& backend() { return *gpu_; }
    // port of gpu_model.rs: GpuModel::after_experts counters (hits = routed experts of a token served from VRAM).
    size_t hits = 0;
    size_t misses = 0;
    size_t tokens = 0;
    // port of gpu_model.rs: mailbox_writes / mailbox_reads / cpu_time / resumed
    size_t mailbox_writes = 0;
    size_t mailbox_reads = 0;
    size_t resumed = 0;
    uint64_t cpu_busy_ns() const { return worker_ ? worker_->busy_ns() : 0; }
    uint64_t cpu_experts() const { return worker_ ? worker_->experts_done() : 0; }
    // Routed-expert residency of this model.
    const ExpertCache* experts() const { return cache_.get(); }
    // port of gpu_model.rs: expert_bytes_per_token (routed-expert bytes one decode token reads; 0 without a cache)
    size_t expert_bytes_per_token() const { return cache_ ? cache_->token_bytes(m_.cfg.n_expert_used) : 0; }
    // The prefill chunk in tokens (0: no batched prefill) and the short-prompt threshold; the prefill scratch in bytes.
    // For inkling2 this is the chunked-prefill size (ink_chunk_), which has no PrefillScratch.
    size_t chunk() const { return ink_chunk_ ? ink_chunk_ : chunk_; }
    size_t short_len() const { return short_len_; }
    size_t prefill_bytes() const { return pf_ ? pf_->bytes() : 0; }
    // The CPU worker (null for a model without MoE layers).
    const CpuWorker* worker() const { return worker_.get(); }
    // port of gpu_model.rs: expert_fast_path (every expert in VRAM and the backend takes a null mailbox)
    bool expert_fast_path() const;
    size_t adapt_every = 8;
    size_t adapt_swaps = 64;

private:
    GpuModel(const Model& m) : m_(m) {}

    // One decode step; `want_logits` copies the logits to `logits`, else only the device argmax to `argmax`.
    bool run_token(uint32_t token, bool want_logits, std::vector<float>* logits, int32_t* argmax, std::string& err);
    // port of gpu_model.rs: GpuModel::forward (one token; the graph body). `gq` is the attention regime (see gq_min).
    // `from` >= 0 resumes at that layer's MoE combine (see finish).
    bool forward(int32_t gq, int64_t from, std::string& err);
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode forward (W2, the PR #389 graph): the prelude,
    // the loop core pass 0, the loop passes (each restarts from the prelude output and reads the previous pass's
    // blocks through messages), the tail, and the output head. `from` >= 0 resumes at that slot's MoE combine.
    bool forward_inkling2(int32_t gq, int64_t from, std::string& err);
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 chunked prefill (W3). The same slot graph over t <= 16
    // tokens: the per-token work runs token by token through the decode launches, while the routed-expert pass is
    // batched over the chunk (one router GEMM, one moe_route with t rows - the CPU worker answers all t tokens'
    // misses in one experts_multi pass - one fused gate/up/situ/down/combine pass). Per-token state that spans slots
    // lives in per-token regions of the decode buffers; the slot-local scratch is reused token by token.
    bool forward_inkling2_chunk(const uint32_t* toks, size_t t, std::string& err);
    // The captured body of forward_inkling2_chunk: the slot graph over the chunk's per-token regions and the
    // batched routed-expert pass; the host control flow depends only on t and the layer indices, so a graph
    // (one per chunk size and gq regime) can replay it.
    bool ink_chunk_body(size_t t, std::string& err);
    // The inkling2 prefill entry: chunks of ink_chunk_ tokens; the debug dumps fall back to the token-by-token path.
    bool prefill_inkling2(const std::vector<uint32_t>& tokens, std::vector<float>& logits, std::string& err);
    // port of gpu_model.rs: GpuModel::layer for one token. `x` is the residual stream (hc_n rows of n_embd with
    // hyper-connections), `res` the pending residual of the previous FFN (0 for none), updated to the one this layer
    // leaves. `resume`: only the MoE combine (see finish).
    bool layer(size_t il, const GpuLayer& l, uint64_t x, uint64_t step, int32_t gq, uint64_t& res, bool resume,
               std::string& err);
    // port of gpu_model.rs: GpuModel::finish. While a MoE combine gave up waiting for the CPU worker, wait for the
    // worker and run the rest of the forward from that combine.
    bool finish(int32_t gq, std::string& err);
    // port of gpu_model.rs: GpuModel::after_experts (routing counters, record, adapt every adapt_every tokens)
    bool after_experts(size_t count, std::string& err);
    // port of gpu_model.rs: GpuModel::hc_read (see Model::hc_read): the mixed vector into a.h (and a.hq), the inject
    // weights into a.inj. `r` is the stack of hc_n rows of n_embd at `nt` tokens.
    bool hc_read(const HcRead& hr, uint64_t r, size_t nt, const HcAt& a, std::string& err);
    // port of gpu_model.rs: GpuModel::ple (the PLE block on the stack at `x`, its input in a.emb)
    bool ple_block(const PleBlock& pl, uint64_t x, uint64_t step, size_t nt, const HcAt& a, std::string& err);
    // port of gpu_model.rs: the per-stream RMSNorm of nt stacks (`gnorm`)
    bool gnorm(uint64_t x, uint64_t w, uint64_t out, size_t nt, std::string& err);
    // port of gpu_model.rs: GpuModel::hc_at (the decode scratch rows)
    HcAt hc_at() const;

    // port of gpu_model/prefill.rs: Scratch::new, called by load (chunk tokens a chunk, at most ctx)
    bool make_prefill(size_t chunk, std::string& err);
    // port of gpu_model/prefill.rs: GpuModel::prefill_chunk (`t` tokens from the current position, one part, no MTP
    // draft head, no EAGLE3 head). Leaves the last hidden state (hc_n rows) in row 0 of x_.
    bool prefill_chunk(const uint32_t* toks, size_t t, std::string& err);
    // port of gpu_model/prefill.rs: the MoE FFN of prefill_chunk's layer `il` (router, routed experts grouped by expert
    // in the GEMM tiles, experts not resident staged from the mapped weights, shared expert, combine into `fx`). `base`
    // is the slab; `pre` the prefetch of the mapped weights (null: none).
    bool prefill_moe(size_t il, const GpuLayer& ly, size_t t, uint64_t base, uint64_t fx, Prefetch* pre, std::string& err);
    // port of gpu_model/prefill.rs: adapt_experts (settle the prompt's routing into the placement, when not resident)
    bool adapt_experts(std::string& err);

    uint64_t buf(const char* name) const;

    // Declared first so that it is destroyed last: the buffers below free through it.
    std::unique_ptr<Backend> gpu_;
    const Model& m_;
    size_t ctx_ = 0;
    size_t pos_ = 0;
    size_t chunk_ = 0;
    size_t ink_chunk_ = 0;  // the inkling2 batched-prefill chunk (0: the token-by-token path)
    size_t short_len_ = 32;
    size_t hcn_ = 1;                           // port of gpu_model.rs: hc_n.max(1) (streams a token)
    bool reset_counts_after_prefill_ = false;  // port of gpu_model.rs: reset_counts_after_prefill (STRATA_RESET_COUNTS)
    std::vector<GpuLayer> layers_;
    GpuTensor output_norm_;
    GpuTensor output_;
    // StrataFreet addition (INKLING2_NOTES.md): the global cross-pass message queries of the looped
    // core (loop_res_query {n_embd, 2 * core span}, loop_kv_res_query same shape), empty without a loop.
    GpuTensor loop_res_query_, loop_kv_res_query_;
    // The inkling2 embedding norm and the output residual-mix query.
    GpuTensor tok_norm_, output_res_query_;
    // StrataFreet addition: the all-ones gamma of the unweighted rms_norms (the V heads) and the zero sink vector the
    // attention takes (the sink logit is folded into the rel-bias table, so the softmax's extra term is exp(0) = 1).
    Buf ones_, sinkz_;
    // StrataFreet addition: the STRATA_INK_DEBUG staging rows (30 rows of 8 floats; null without the env flag).
    Buf ink_dbg_;
    // StrataFreet addition: the STRATA_INK_DUMP record (h 2304, moe 2304, w 8, ids 8, slots 8 floats; null without
    // the env flag).
    Buf ink_dump_;
    HcRead output_hc_;                         // port of gpu_model.rs: output_hc (hc models; output_norm_ is empty then)
    bool has_output_hc_ = false;
    Buf ple_ring_;                             // port of gpu_model.rs: ple_ring (16 rows of hc_n * n_embd floats)
    bool has_ple_ = false;
    std::array<int64_t, 2> ple_prev_{-1, -1};  // port of gpu_model.rs: ple_prev (the last two tokens, -1 for none)
    QsaBufs qsa_;
    bool has_qsa_ = false;
    std::unordered_map<std::string, Buf> bufs_;  // scratch activations by name, one token each (TMAX rows)
    Buf x_, logits_, ids_, slots_, step_buf_, rope_buf_, amax_;
    std::unique_ptr<PrefillScratch> pf_;          // null when the model has no prefill chunk
    Buf mailbox_;                                 // the host-mapped mailbox memory of the CPU worker
    Mailbox layout_;  // the CPU worker's mailbox layout (its offsets are read by moe_route / moe_down_combine)
    // Destroyed first: the worker joins its thread before the cache and the mailbox are freed.
    std::unique_ptr<ExpertCache> cache_;
    std::unique_ptr<CpuWorker> worker_;
    size_t threads_ = 0;
};

}  // namespace strata
