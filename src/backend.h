// port of backend.rs: the Backend seam (device memory as Buf, and the kernel ops the model runs).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Deviations forced by C++ (also in MODIFICATIONS.md):
//  - anyhow::Result<T> becomes `bool` plus a `std::string& err` out-parameter, the convention the rest
//    of strata-cpp uses (Weights::get, Gguf::open, ...). A Rust return value becomes an out-parameter
//    (`attn_decode_q` returns its xq through `xq_out`).
//  - `&mut dyn FnMut() -> Result<()>` becomes `const OpFn&` (std::function).
//  - Rust trait default methods keep their bodies here; required methods are pure virtual.
//  - Rust's `Box<dyn Any>` owner of a Buf becomes `std::shared_ptr<void>`, which keeps the deleter.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "quant.h"
#include "weights.h"

namespace strata {

// port of backend.rs: ATT_SPLITS (blocks per head in decode attention)
constexpr size_t ATT_SPLITS = 32;
// port of backend.rs: ATT_SPLITS_1 (splits of the one-head-per-block decode attention below ATT_GQA_MIN)
constexpr size_t ATT_SPLITS_1 = 16;
// port of backend.rs: ATT_GQA_MIN
constexpr int32_t ATT_GQA_MIN = 2048;
static_assert(ATT_SPLITS <= 256, "attn_merge keeps one weight per split in a 256-thread block");
// port of backend.rs: GEMM_BM (positions per GEMM tile)
constexpr size_t GEMM_BM = 128;
// port of backend.rs: SAME_X (gemm `act` flag: x unchanged since the previous gemm call)
constexpr int32_t SAME_X = 4;

// port of cuda/mod.rs: Gpu::mmq (the type policy: int8 prefill GEMM unless the type is float or listed in
// STRATA_MMQ). Pure policy; a backend's `mmq` also requires the arch to have int8 tensor cores.
bool cuda_mmq_policy(Ty ty);

// port of backend.rs: Buf. `len` bytes of device memory at device address `addr`, freed when the last
// copy of `owner` goes away.
class Buf {
public:
    Buf() = default;
    Buf(uint64_t addr, size_t len, std::shared_ptr<void> owner)
        : addr_(addr), len_(len), owner_(std::move(owner)) {}
    // port of backend.rs: Buf::addr
    uint64_t addr() const { return addr_; }
    // port of backend.rs: Buf::len
    size_t len() const { return len_; }
    // StrataFreet addition: a view of `n` bytes at `off` (the per-token regions of the inkling2 chunk prefill; the
    // owner keeps the memory alive).
    Buf slice(size_t off, size_t n) const { return Buf(addr_ + off, n, owner_); }

private:
    uint64_t addr_ = 0;
    size_t len_ = 0;
    std::shared_ptr<void> owner_;
};

// port of backend.rs: GpuTensor
struct GpuTensor {
    Ty ty = Ty::F32;
    size_t rows = 0;
    size_t cols = 0;
    uint64_t ptr = 0;
    Buf buf;

    // An empty tensor (a struct member the load fills in later; the GPU model's optional weights stay empty).
    GpuTensor() = default;
    // port of backend.rs: GpuTensor::new - `rows` x `cols` of `ty` stored in `buf` from its start.
    GpuTensor(Ty t, size_t r, size_t c, Buf b) : ty(t), rows(r), cols(c), ptr(b.addr()), buf(std::move(b)) {}
    // port of backend.rs: GpuTensor::bytes
    size_t bytes() const { return row_bytes(ty, cols) * rows; }
};

// port of backend.rs: OpFn (`&mut dyn FnMut() -> anyhow::Result<()>`)
using OpFn = std::function<bool(std::string&)>;

// port of backend.rs: upload_type (F16 is stored as F32, Q4_0/Q5_0 as Q8_0)
Ty upload_type(Ty ty);

// port of backend.rs: the Backend trait
class Backend {
public:
    virtual ~Backend() = default;

    // "cuda" or "vulkan".
    virtual const char* name() const = 0;
    // The GPU's name as the driver reports it.
    virtual std::string device() const = 0;
    // Zeroed buffer of `bytes` (at least 1).
    virtual bool alloc(size_t bytes, Buf& out, std::string& err) = 0;
    // Reserves device memory that the next allocations carve from exactly (CudaBackend::reserve_arena); a backend
    // without one allocates each buffer on its own, as this default does nothing.
    virtual bool reserve_arena(size_t bytes, std::string& err) {
        (void)bytes;
        (void)err;
        return true;
    }
    virtual bool htod(const void* src, size_t n, const Buf& dst, size_t off, std::string& err) = 0;
    virtual bool dtoh(const Buf& src, size_t off, void* dst, size_t n, std::string& err) = 0;
    virtual bool dtod(const Buf& src, size_t src_off, const Buf& dst, size_t dst_off, size_t bytes, std::string& err) = 0;
    // Zero all of `dst`.
    virtual bool memset(const Buf& dst, std::string& err) = 0;
    // Free and total device memory in bytes.
    virtual bool mem_info(size_t& free_bytes, size_t& total_bytes, std::string& err) = 0;
    // Wait for all queued device work.
    virtual bool sync(std::string& err) = 0;
    // Run the ops `f` queues as the graph recorded under `key`, recording it on first use. `f` runs only when
    // recording, so its ops must read per-call values from device memory.
    virtual bool graph(uint64_t key, const OpFn& f, std::string& err) = 0;
    // Queue the ops of `f`, none of which reads what another writes; ops after the batch see all results.
    virtual bool batch(const OpFn& f, std::string& err) { return f(err); }
    // Queue the ops of `f` in order, its host copies beside its other ops (CUDA: on a side stream).
    virtual bool beside(const OpFn& f, std::string& err) { return f(err); }
    // Ops after this see the copies of every `beside` so far.
    virtual bool beside_join(std::string& err) {
        (void)err;
        return true;
    }
    // Submit the ops queued so far without waiting.
    virtual bool kick(std::string& err) {
        (void)err;
        return true;
    }
    // Whether `beside` really overlaps its copies with its other ops (else it queues them in order).
    virtual bool beside_overlaps() const { return false; }
    // Whether a MoE combine gave up waiting for the CPU worker since the last call. Clears the flag.
    virtual bool stalled() { return false; }
    // Whether the prefill GEMM for `ty` runs on int8 tensor cores on this backend.
    virtual bool mmq(Ty ty) const { return cuda_mmq_policy(ty); }
    // Whether this backend runs the Volta (sm_70) CUDA fallbacks.
    virtual bool volta() const { return false; }
    // Zeroed host memory of `words` u32 that kernels read and write at the `buf` address. The host pointer is
    // valid while `buf` lives.
    virtual bool host_mailbox(size_t words, Buf& buf, uint32_t*& host, std::string& err) = 0;
    // Refuse, before any weight is uploaded, if a pipeline these weight types need exceeds the shared-memory limit.
    virtual bool preflight(const std::vector<Ty>& types, bool verbose, std::string& err) {
        (void)types;
        (void)verbose;
        (void)err;
        return true;
    }
    // Upload a tensor as `upload_type` stores it.
    virtual bool upload(const Tensor& t, GpuTensor& out, std::string& err);

    // u16 words of a KV cache for `n` values.
    virtual size_t kv_words(size_t n) const = 0;
    // GEMM over a tile list (see backend.rs). `tiles` is a device array of [w, m0 | cnt << 32]. `act`: 0 stores,
    // 1 computes y = silu(y) * result, 2 adds to y; `| SAME_X` reuses the int8 copy of x.
    virtual bool gemm(Ty ty, size_t rows, size_t cols, uint64_t tiles, size_t n_tiles, uint64_t x, size_t x_stride,
                      size_t n_x, uint64_t xrow, uint64_t y, size_t y_stride, int32_t act, std::string& err) = 0;
    // Whether `gemm` and `gemm_dense` run on int8 tensor cores for `ty`.
    virtual bool int8_gemm(Ty ty) const = 0;
    // y[m] (op) W x[m] for positions m < n.
    virtual bool gemm_dense(const GpuTensor& w, size_t n, uint64_t x, size_t x_stride, uint64_t y, size_t y_stride,
                            int32_t act, std::string& err) = 0;
    virtual bool prequant_gemm_x(const GpuTensor& w, size_t n, uint64_t x, size_t x_stride, std::string& err) {
        (void)w;
        (void)n;
        (void)x;
        (void)x_stride;
        (void)err;
        return true;
    }
    // y = W x (all rows of `w`).
    virtual bool gemv(const GpuTensor& w, uint64_t x, uint64_t y, std::string& err) = 0;
    // y[t] = W x[t] for t < nt (1 to 4; x[t] at x + 4 * t * x_stride bytes, y[t] at y + 4 * t * y_stride bytes).
    virtual bool gemvn(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t y, size_t y_stride, size_t nt,
                       std::string& err) = 0;
    // `gemvn` that reads x pre-quantized by `quant_xq` at `xq` (0 = quantize x itself; float weights read x).
    virtual bool gemvq(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t xq, uint64_t y, size_t y_stride,
                       size_t nt, std::string& err) = 0;
    // `gemvq` of `a` into `ya` and of `b` into `yb` on the same x.
    virtual bool gemvq2(const GpuTensor& a, const GpuTensor& b, uint64_t x, size_t x_stride, uint64_t xq, uint64_t ya,
                        uint64_t yb, size_t nt, std::string& err) {
        return gemvq(a, x, x_stride, xq, ya, a.rows, nt, err) && gemvq(b, x, x_stride, xq, yb, b.rows, nt, err);
    }
    // For j < k with ids[j] >= 0: y[j] = W_j x[j / xg], where W_j is the `rows` x `cols` matrix at w + ids[j] * stride
    // bytes. `pair` = (w2, y2) also computes the same product for the matrices at byte offset w2 into the output at
    // byte offset y2. `xq` (not 0): x pre-quantized by `quant_xq` (k / xg rows), read in place of x.
    virtual bool moe_gemv(Ty ty, size_t rows, size_t cols, uint64_t w, size_t stride, uint64_t ids, size_t k, uint64_t x,
                          size_t x_stride, size_t xg, uint64_t y, std::optional<std::pair<int64_t, int64_t>> pair,
                          uint64_t xq, std::string& err) = 0;
    // out = the first `rows` rows of `w` as floats (a test helper).
    virtual bool dequant(const GpuTensor& w, size_t rows, uint64_t out, std::string& err) = 0;
    // Quantize `nt` rows of n floats (row t at x + 4 * t * x_stride bytes) for gemvq / moe_gemv into `out`:
    // the nt rows of n int8 values, then their f32 scales (one per 32 values).
    virtual bool quant_xq(uint64_t x, size_t x_stride, size_t n, size_t nt, uint64_t out, std::string& err) = 0;
    // `rows` rows: out[r] = rmsnorm(x[r]) * w.
    virtual bool rms_norm(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t out_stride, size_t n,
                          size_t rows, float eps, std::string& err) = 0;
    // `rms_norm` that also writes its output quantized as `quant_xq` does to `xq` (when not 0), and first adds
    // `res` (rows `x_stride` apart, when not 0) into x in place.
    virtual bool rms_norm_q(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t out_stride, size_t n,
                            size_t rows, float eps, uint64_t xq, uint64_t res, std::string& err) = 0;
    // Per-stream RMSNorm of nt stacks of hc rows of n (contiguous) with the hc x n gamma `w`.
    virtual bool rms_norm_streams(uint64_t x, uint64_t w, uint64_t out, size_t n, size_t hc, size_t nt, float eps,
                                  std::string& err) {
        for (size_t s = 0; s < hc; s++) {
            const uint64_t o = 4 * (uint64_t)(s * n);
            if (!rms_norm(x + o, hc * n, w + o, out + o, hc * n, n, nt, eps, err)) return false;
        }
        return true;
    }
    // out[r] = argmax of row r of x (`rows` rows of n floats), first index on ties.
    virtual bool argmax_rows(uint64_t x, size_t n, size_t rows, uint64_t out, std::string& err) = 0;
    // Row r: vector r % per (n floats) of token r / per, tokens `stride` floats apart.
    virtual bool l2_norm(uint64_t x, size_t n, size_t per, size_t stride, size_t rows, float eps, std::string& err) = 0;
    // `rope_neox` of pairs `lo..hi` only (M-RoPE: one call per section).
    virtual bool rope_pairs(uint64_t x, size_t stride, size_t heads, size_t per, size_t n_rot, uint64_t step, float base,
                            size_t lo, size_t hi, std::string& err) = 0;
    // `heads` rows of `stride` floats, `per` per token; token t is at position step[0] + t.
    virtual bool rope_neox(uint64_t x, size_t stride, size_t heads, size_t per, size_t n_rot, uint64_t step, float base,
                           std::string& err) {
        return rope_pairs(x, stride, heads, per, n_rot, step, base, 0, n_rot / 2, err);
    }
    // M-RoPE (`ops::rope_mrope`): section s of the pairs turns by position `pos[2 * s]`; pairs past the sections
    // use the last position.
    virtual bool rope_mrope(uint64_t x, size_t stride, size_t heads, size_t per, size_t n_rot, uint64_t pos, float base,
                            const std::array<size_t, 4>& sections, std::string& err) {
        const size_t half = n_rot / 2;
        size_t lo = 0;
        for (size_t s = 0; s < 4; s++) {
            const size_t hi = s == 3 ? half : std::min(lo + sections[s], half);
            if (hi > lo && !rope_pairs(x, stride, heads, per, n_rot, pos + 8 * s, base, lo, hi, err)) return false;
            lo = hi;
        }
        return true;
    }
    // `rms_norm` of `rows` heads of n (output rows of n), then `rope_neox` on them.
    virtual bool rms_norm_rope(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t n, size_t rows, float eps,
                               size_t per, size_t n_rot, uint64_t step, float base, std::string& err) {
        return rms_norm(x, x_stride, w, out, n, n, rows, eps, err) && rope_neox(out, n, rows, per, n_rot, step, base, err);
    }
    // `rms_norm_rope` of the k heads (`per` per token), then `kv_store` of them and v (rows of n, like out).
    virtual bool rms_norm_rope_kv(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t n, size_t rows, float eps,
                                  size_t per, size_t n_rot, uint64_t step, float base, uint64_t v, uint64_t kc, uint64_t vc,
                                  std::string& err) = 0;
    // `rms_norm_rope` of the q heads (`qrows` rows, `qper` per token), then `rms_norm_rope_kv` of the k heads.
    virtual bool rms_norm_rope_qk(uint64_t q, size_t q_stride, uint64_t qw, uint64_t qo, size_t qrows, size_t qper,
                                  uint64_t k, uint64_t kw, uint64_t ko, size_t krows, size_t kper, size_t n, float eps,
                                  size_t n_rot, uint64_t step, float base, uint64_t v, uint64_t kc, uint64_t vc,
                                  std::string& err) {
        return rms_norm_rope(q, q_stride, qw, qo, n, qrows, eps, qper, n_rot, step, base, err) &&
               rms_norm_rope_kv(k, n, kw, ko, n, krows, eps, kper, n_rot, step, base, v, kc, vc, err);
    }
    // x += y.
    virtual bool add(uint64_t x, uint64_t y, size_t n, std::string& err) = 0;
    // g = silu(g) * u.
    virtual bool swiglu(uint64_t g, uint64_t u, size_t n, std::string& err) = 0;
    // The SwiGLU of `rows` rows of n for a GEMV with quantized weights: either g = silu(g) * u (`xq_out` 0), or
    // silu(g) * u quantized into `xq` as `quant_xq` does (`xq_out` = xq; g is then left as is).
    virtual bool swiglu_q(uint64_t g, uint64_t u, size_t n, size_t rows, uint64_t xq, uint64_t& xq_out, std::string& err) {
        (void)xq;
        xq_out = 0;
        return swiglu(g, u, rows * n, err);
    }
    virtual bool ssm_gates(uint64_t beta, uint64_t g, uint64_t dt_bias, uint64_t a, size_t n, size_t nv, std::string& err) = 0;
    // One token's causal conv and SiLU; then the first `nn` groups of 128 channels are l2-normalized as `l2_norm` does.
    virtual bool conv_silu(uint64_t qkv, uint64_t state, uint64_t w, uint64_t y, size_t ch, size_t kw, size_t nn, float eps,
                           std::string& err) = 0;
    // Scan `t` tokens (rows of y `ych` floats apart) through the DeltaNet state `s`. `snap`: room for `nsnap` states.
    // `interleave`: value head h reads key head h / (nv / nk), else h % nk. `dt_bias` not 0: beta and g are raw.
    virtual bool deltanet(uint64_t y, uint64_t s, uint64_t beta, uint64_t g, uint64_t dt_bias, uint64_t a, uint64_t o,
                          size_t nk, size_t nv, bool interleave, size_t dk, size_t dv, float scale, size_t t, size_t ych,
                          uint64_t snap, size_t nsnap, std::string& err) = 0;
    // Causal conv + SiLU over `t` tokens of `ch` channels, then roll the conv state forward (snapshots as in `deltanet`).
    virtual bool conv_seq(uint64_t qkv, uint64_t state, uint64_t w, uint64_t y, size_t ch, size_t kw, size_t t,
                          uint64_t snap, size_t nsnap, std::string& err) = 0;
    // `qg` 0: no output gate.
    virtual bool attn_prefill(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t step, size_t t,
                              size_t nh, size_t nkv, size_t hd, float scale, std::string& err) {
        return attn_prefill_m(q, kc, vc, qg, out, step, t, nh, nkv, hd, scale, 0, 0, 0, 0, err);
    }
    // `attn_prefill` where query row i sees only the keys whose 4-cell block has its bit set in row i of `qm`
    // (`mw` words a row). `qm` 0: no mask. `sink` (0 = none): per-head attention-sink logits. `win` (0 = none): window.
    virtual bool attn_prefill_m(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t step, size_t t,
                                size_t nh, size_t nkv, size_t hd, float scale, uint64_t qm, size_t mw, uint64_t sink,
                                size_t win, std::string& err) = 0;
    // QSA indexer keys (see `Model::qsa`): pool the blocks of 4 positions into `pool`. Not implemented by default.
    virtual bool qsa_pool(uint64_t ik, uint64_t ring, uint64_t pool, uint64_t w, uint64_t step, size_t t, size_t n_rot,
                          float base, float eps, std::string& err) {
        (void)ik; (void)ring; (void)pool; (void)w; (void)step; (void)t; (void)n_rot; (void)base; (void)eps;
        err = std::string(name()) + ": QSA sparse attention not implemented";
        return false;
    }
    // QSA block selection for `rows` query rows (see backend.rs). Not implemented by default.
    virtual bool qsa_select(uint64_t q, uint64_t pool, uint64_t score, uint64_t sel, uint64_t mask, uint64_t step,
                            size_t r0, size_t rows, size_t nbs, size_t top, size_t sst, size_t mw, std::string& err) {
        (void)q; (void)pool; (void)score; (void)sel; (void)mask; (void)step; (void)r0; (void)rows; (void)nbs;
        (void)top; (void)sst; (void)mw;
        err = std::string(name()) + ": QSA sparse attention not implemented";
        return false;
    }
    virtual bool moe_combine_seq(uint64_t x, uint64_t ye, uint64_t wts, uint64_t inv, size_t k, uint64_t sh, uint64_t sg,
                                 size_t n, size_t t, std::string& err) = 0;
    virtual bool norm_gate(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, std::string& err) = 0;
    // `norm_gate`, then either sets `xq_out` to 0, or also writes o quantized as `quant_xq` does to `xq`.
    virtual bool norm_gate_q(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, uint64_t xq,
                             uint64_t& xq_out, std::string& err) {
        (void)xq;
        xq_out = 0;
        return norm_gate(o, w, z, n, heads, eps, err);
    }
    // `norm_gate_q` gating with sigmoid(z) in place of silu(z) (qwen4exp). Not implemented by default.
    virtual bool norm_gate_sig(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, uint64_t xq,
                               uint64_t& xq_out, std::string& err) {
        (void)o; (void)w; (void)z; (void)n; (void)heads; (void)eps; (void)xq; (void)xq_out;
        err = std::string(name()) + ": sigmoid-gated norm not implemented";
        return false;
    }
    // qwen4exp hyper-connections (see `Model::hc_read`); the residual of a token is hc rows of n. Not implemented by default.
    virtual bool hc_silu(uint64_t x, float sc, size_t n, std::string& err) {
        (void)x; (void)sc; (void)n;
        err = std::string(name()) + ": hyper-connections not implemented";
        return false;
    }
    virtual bool hc_mix(uint64_t xn, uint64_t g, uint64_t out, size_t n, size_t hc, size_t nt, std::string& err) {
        (void)xn; (void)g; (void)out; (void)n; (void)hc; (void)nt;
        err = std::string(name()) + ": hyper-connections not implemented";
        return false;
    }
    virtual bool hc_write(uint64_t r, uint64_t out, uint64_t inj, size_t n, size_t hc, size_t nt, bool clear, std::string& err) {
        (void)r; (void)out; (void)inj; (void)n; (void)hc; (void)nt; (void)clear;
        err = std::string(name()) + ": hyper-connections not implemented";
        return false;
    }
    // PLE gate (see `Model::ple`). Not implemented by default.
    virtual bool ple_gate(uint64_t key, uint64_t query, uint64_t value, uint64_t gated, size_t n, size_t hc, size_t nt,
                          std::string& err) {
        (void)key; (void)query; (void)value; (void)gated; (void)n; (void)hc; (void)nt;
        err = std::string(name()) + ": PLE not implemented";
        return false;
    }
    // PLE conv and residual add for nt tokens at positions step[0] ... Not implemented by default.
    virtual bool ple_conv(uint64_t r, uint64_t gated, uint64_t norm, uint64_t kw, uint64_t ring, uint64_t step, size_t d,
                          size_t nt, std::string& err) {
        (void)r; (void)gated; (void)norm; (void)kw; (void)ring; (void)step; (void)d; (void)nt;
        err = std::string(name()) + ": PLE not implemented";
        return false;
    }
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode ops.
    // One token's causal shortconv with residual and optional per-head dynamic taps: y = (cur + conv) * scale, and
    // the state (kw - 1 slots per channel, interleaved, oldest first) rolls forward. `hch` = the channels of one head
    // for the taps (dyn 0: none).
    virtual bool conv_ink(uint64_t cur, uint64_t state, uint64_t w, uint64_t dyn, uint64_t y, size_t ch, size_t kw,
                          size_t hch, float scale, std::string& err) {
        (void)cur; (void)state; (void)w; (void)dyn; (void)y; (void)ch; (void)kw; (void)hch; (void)scale;
        err = std::string(name()) + ": inkling2 shortconv not implemented";
        return false;
    }
    // The depth mix of the `nb` bank rows and cur with the `nq` query rows of `q` (rows of ne): per query row the
    // RMSNormed states (eps 1e-6) score against the query (scale 1/sqrt(ne)), softmax over the states, and the
    // weighted sum goes to out (nq rows of ne).
    virtual bool mix_dec(uint64_t bank, size_t nb, uint64_t cur, uint64_t q, size_t nq, uint64_t out, size_t ne,
                         std::string& err) {
        (void)bank; (void)nb; (void)cur; (void)q; (void)nq; (void)out; (void)ne;
        err = std::string(name()) + ": inkling2 depth mix not implemented";
        return false;
    }
    // The L2 clip of `rows` rows of n in place (factor = clamp(threshold / (sqrt(sum x^2 * sc^2) + 1e-6), 0, 1)), then
    // tau = 1 + alpha * log(max(1, (pos + 1) * rfloor)) (alpha 0 gives exactly 1; rfloor = 1/floor). `sc` 0: no scale.
    virtual bool clip_rows(uint64_t x, uint64_t sc, size_t n, size_t rows, float threshold, float alpha, float rfloor,
                           uint64_t step, std::string& err) {
        (void)x; (void)sc; (void)n; (void)rows; (void)threshold; (void)alpha; (void)rfloor; (void)step;
        err = std::string(name()) + ": inkling2 clip not implemented";
        return false;
    }
    // out = [rms_norm(cur, eps), rms_norm(msg, eps)] without gamma (the loop-gate control's gate input).
    virtual bool gate_pack(uint64_t cur, uint64_t msg, uint64_t out, size_t ne, float eps, std::string& err) {
        (void)cur; (void)msg; (void)out; (void)ne; (void)eps;
        err = std::string(name()) + ": inkling2 loop gate not implemented";
        return false;
    }
    // out[i] = cur[i] * (1 - tanh(forget[i])) + tanh(gain[i]) * msg[i] (in place safe).
    virtual bool control_mix(uint64_t cur, uint64_t msg, uint64_t gain, uint64_t forget, uint64_t out, size_t n,
                             std::string& err) {
        (void)cur; (void)msg; (void)gain; (void)forget; (void)out; (void)n;
        err = std::string(name()) + ": inkling2 loop gate not implemented";
        return false;
    }
    // x[i] *= sigmoid(g[i]) (the attention output gate).
    virtual bool sig_mul(uint64_t x, uint64_t g, size_t n, std::string& err) {
        (void)x; (void)g; (void)n;
        err = std::string(name()) + ": inkling2 gate not implemented";
        return false;
    }
    // The Situ activation of the fused gate/up rows (slot t: gate at gu[t * 2n + i], up at gu[t * 2n + n + i]), the
    // per-expert f32 scale ds[ids[t]] on both when ds is set, quantized as quant_xq does into xq (xq_out = xq; 0
    // when the backend quantizes nowhere). gu's gate rows are left as they are.
    virtual bool situ_q(uint64_t gu, size_t n, size_t rows, uint64_t ds, uint64_t ids, float beta, float lb, uint64_t xq,
                        uint64_t& xq_out, std::string& err) {
        (void)gu; (void)n; (void)rows; (void)ds; (void)ids; (void)beta; (void)lb; (void)xq;
        xq_out = 0;
        err = std::string(name()) + ": situ activation not implemented";
        return false;
    }
    // The relative-bias table of one token: table[d * nh + h] = sum_r bias[r * next1 + d] * rc[r * nh + h] - sink[h]
    // (the attention gathers table[min(distance, extent) * nh + h]; the sink logit is folded in here).
    virtual bool rel_table(uint64_t bias, uint64_t rc, uint64_t sink, uint64_t table, size_t nr, size_t next1, size_t nh,
                           std::string& err) {
        (void)bias; (void)rc; (void)sink; (void)table; (void)nr; (void)next1; (void)nh;
        err = std::string(name()) + ": inkling2 rel table not implemented";
        return false;
    }
    // x[i] = cap * tanh(x[i] / cap) (the final logit softcap; cap <= 0: a no-op).
    virtual bool softcap(uint64_t x, size_t n, float cap, std::string& err) {
        (void)x; (void)n; (void)cap;
        err = std::string(name()) + ": logit softcap not implemented";
        return false;
    }
    // Store `t` tokens of `n` values at positions step[0] ...
    virtual bool kv_store(uint64_t k, uint64_t v, uint64_t kc, uint64_t vc, size_t n, size_t t, uint64_t step, size_t hd,
                          std::string& err) = 0;
    // Decode attention for `nt` query tokens, token t over positions 0 ..= step[2 t]. `qg` 0: no output gate.
    virtual bool attn_decode(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                             size_t nh, size_t nkv, size_t hd, float scale, size_t nt, std::string& err) {
        return attn_decode_gq(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, ATT_GQA_MIN, nt, 0, 0, 0, 0, err);
    }
    // `attn_decode_gq`, also writing `out` quantized for `gemvq` into `xq` where supported; `xq_out` is the xq to
    // use (0: none). `sel` not 0: QSA cell lists. `sink` (0 = none), `win` (0 = none).
    virtual bool attn_decode_q(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                               size_t nh, size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t xq,
                               uint64_t sel, size_t sst, uint64_t sink, size_t win, uint64_t& xq_out, std::string& err) {
        (void)xq;
        (void)sst;
        xq_out = 0;
        if (sel != 0) {
            err = std::string(name()) + ": QSA sparse attention not implemented";
            return false;
        }
        return attn_decode_gq(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, sink, win, 0, 0, err);
    }
    // Whether `rms_norm_rope_qk_rows` and `attn_rows` exist here (`GpuModel::step_batch` needs them).
    virtual bool kv_rows() const { return false; }
    // `rms_norm_rope_qk` for `nt` tokens of different sequences: token t is at position `step[2t]` and stores its K
    // and V in the caches whose addresses are `kvt[2t]` and `kvt[2t + 1]`. Not implemented by default.
    virtual bool rms_norm_rope_qk_rows(uint64_t q, size_t q_stride, uint64_t qw, uint64_t qo, size_t nh, uint64_t k,
                                       uint64_t kw, uint64_t ko, size_t nkv, size_t hd, float eps, size_t n_rot, uint64_t step,
                                       float base, uint64_t v, uint64_t kvt, size_t nt, std::string& err) {
        (void)q; (void)q_stride; (void)qw; (void)qo; (void)nh; (void)k; (void)kw; (void)ko; (void)nkv; (void)hd;
        (void)eps; (void)n_rot; (void)step; (void)base; (void)v; (void)kvt; (void)nt;
        err = std::string(name()) + ": no per-row KV kernels";
        return false;
    }
    // `attn_decode_gq` with `gq_min` 0 for `nt` tokens of different sequences. Not implemented by default.
    virtual bool attn_rows(uint64_t q, uint64_t qg, uint64_t out, uint64_t part, uint64_t step, size_t nh, size_t nkv,
                           size_t hd, float scale, size_t nt, uint64_t sink, size_t win, uint64_t kvt, size_t splits,
                           std::string& err) {
        (void)q; (void)qg; (void)out; (void)part; (void)step; (void)nh; (void)nkv; (void)hd; (void)scale; (void)nt;
        (void)sink; (void)win; (void)kvt; (void)splits;
        err = std::string(name()) + ": no per-row KV kernels";
        return false;
    }
    // `attn_decode` that shares K/V loads across a GQA group from `gq_min` positions on (tests use 0 to force it).
    // `bias` not 0: the Inkling2 relative-bias table {extent + 1, nh} of the query token — the score of cache row i
    // gets table[min(distance, extent) * nh + h] added before the scale.
    virtual bool attn_decode_gq(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                                size_t nh, size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t sink,
                                size_t win, uint64_t bias, size_t extent, std::string& err) = 0;
    // Route `t` tokens (logit rows of n, ids / wts rows of k). `sigmoid` selects sigmoid scores instead of softmax.
    // `bias` not 0: the Inkling2 router bias (the selection runs on probs + bias, the weights on the unbiased probs).
    // `eps_add` > 0: the weight normalization divides by sum + eps_add, else the F16 clamp as before.
    virtual bool router_topk(uint64_t p, size_t n, size_t k, float extra_scale, uint64_t ids, uint64_t wts, size_t t,
                             bool sigmoid, uint64_t bias, float eps_add, std::string& err) = 0;
    // Decode routing: `router_topk`, the expert cache lookup, then the routing and (on a miss) `h` written to the
    // layer mailbox at `mb` with its request flag raised. `mb` = 0 skips every mailbox store where
    // `moe_mailbox_optional` holds. `pid` (0 = none) receives the last token's ids. `bias` / `eps_add`: the Inkling2
    // router bias and weight epsilon (see router_topk).
    virtual bool moe_route(uint64_t p, size_t n, size_t k, float extra_scale, uint64_t table, uint64_t ids, uint64_t wts,
                           uint64_t slots, uint64_t wh, uint64_t h, uint64_t mb, size_t nh, size_t h_off, uint64_t step,
                           size_t t, uint64_t pid, bool sigmoid, uint64_t bias, float eps_add, std::string& err) = 0;
    // `mb` = 0: no CPU part. Otherwise adds the CPU result at word `y_off` of the mailbox once its done flag is `seq`.
    // `ds` not 0: the Inkling2 per-expert down scale — the weight of slot j multiplies by ds[ids[j]] (the expert id;
    // the mailbox part arrives pre-scaled and a missed slot's weight is 0).
    virtual bool moe_combine(uint64_t x, uint64_t ye, uint64_t w, size_t k, uint64_t sh, uint64_t sg, uint64_t mb,
                             size_t y_off, uint64_t step, uint64_t slots, size_t n, size_t t, uint64_t ds, uint64_t ids,
                             std::string& err) = 0;
    // Most tokens of one gemvn / verify forward: 4 unless the backend's kernels take cpu_worker::TMAX.
    virtual size_t verify_rows() const { return 4; }
    // Whether moe_route / moe_combine accept mb = 0, i.e. a fully resident expert cache costs the CPU worker nothing.
    virtual bool moe_mailbox_optional() const { return false; }
    // Whether `moe_prefetch` does anything here.
    virtual bool moe_prefetch_supported() const { return false; }
    // Hint into L2 the cached slots of the experts `ids` of a layer (a performance knob only). A no-op by default.
    virtual bool moe_prefetch(uint64_t ids, size_t k, uint64_t table, uint64_t base, size_t stride, size_t bytes,
                              std::string& err) {
        (void)ids; (void)k; (void)table; (void)base; (void)stride; (void)bytes; (void)err;
        return true;
    }
    // `moe_gemv` of the down experts (`k` slots per token, x pre-quantized at `xq` when not 0) into `ye`, then
    // `moe_combine` of ye into x.
    virtual bool moe_down_combine(Ty ty, size_t rows, size_t cols, uint64_t w, size_t stride, uint64_t slots, size_t k,
                                  uint64_t h, uint64_t ye, uint64_t xq, uint64_t x, uint64_t wh, uint64_t sh, uint64_t sg,
                                  uint64_t mb, size_t y_off, uint64_t step, size_t t, std::string& err) {
        return moe_gemv(ty, rows, cols, w, stride, slots, t * k, h, cols, 1, ye, std::nullopt, xq, err) &&
               moe_combine(x, ye, wh, k, sh, sg, mb, y_off, step, slots, rows, t, 0, 0, err);
    }
};

// Typed copies (port of backend.rs: `impl dyn Backend`: zeros, put, get). Counts are in elements of T.
template <class T>
bool zeros(Backend& b, size_t n, Buf& out, std::string& err) {
    return b.alloc(n * sizeof(T), out, err);
}

// STRATA_TRACE_PREFILL (CUDA only): reads and resets the pinned-ring counters of htod (host ms of the memcpy into the
// ring, of the waits for ring slots, and of the DMA enqueue; the number of ring chunks and their bytes).
void htod_trace_take(double& memcpy_ms, double& wait_ms, double& dma_ms, uint64_t& chunks, uint64_t& bytes);

template <class T>
bool put(Backend& b, const T* src, size_t n, const Buf& dst, size_t at, std::string& err) {
    return b.htod(src, n * sizeof(T), dst, at * sizeof(T), err);
}

template <class T>
bool get(Backend& b, const Buf& src, size_t at, size_t n, std::vector<T>& out, std::string& err) {
    out.assign(n, T{});
    return b.dtoh(src, at * sizeof(T), out.data(), n * sizeof(T), err);
}

}  // namespace strata
