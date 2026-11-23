// Test ABI for the CPU expert path (cpu_moe.cpp), built into build\strata_cpu_test.dll and driven from
// tests\test_cpu_moe.py through ctypes. Not part of the product ABI (capi.cpp).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Every function returns 0 (or a non-null handle) on success; sct_last_error() gives the text of the last
// failure on the calling thread.
#include <malloc.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config.h"
#include "cpu_moe.h"
#include "cpu_worker.h"
#include "ds_mat.h"
#include "ds_ops.h"
#include "gguf.h"
#include "glm.h"
#include "weights.h"

#ifdef _WIN32
#define SC_EXPORT extern "C" __declspec(dllexport)
#else
#define SC_EXPORT extern "C"
#endif

using namespace strata;

namespace {

thread_local std::string g_err;

// One GGUF with its Weights and a thread pool, alive until sct_close. The Weights point into the Gguf, so the
// object is heap-allocated and never moved.
struct Handle {
    Gguf g;
    Weights w;
    std::unique_ptr<ThreadPool> pool;
    size_t pool_threads = 0;

    // Pool of `threads` workers (0 = default_threads()), rebuilt when the count changes.
    ThreadPool& pool_for(size_t threads) {
        const size_t want = threads ? threads : default_threads();
        if (!pool || pool_threads != want) {
            pool.reset();   // join the old workers before the new ones start
            pool = std::make_unique<ThreadPool>(want);
            pool_threads = want;
        }
        return *pool;
    }
};

Handle* as_handle(void* h) { return static_cast<Handle*>(h); }

}  // namespace

// Opens a GGUF (for a split model, the -00001-of-0000N shard) and builds its Weights. NULL on failure.
SC_EXPORT void* sct_open(const char* path) {
    g_err.clear();
    std::unique_ptr<Handle> hd(new Handle());
    if (!hd->g.open(path, g_err)) return nullptr;
    if (!hd->w.init(&hd->g, nullptr, g_err)) return nullptr;
    return hd.release();
}

SC_EXPORT void sct_close(void* h) { delete as_handle(h); }

SC_EXPORT const char* sct_last_error(void) { return g_err.c_str(); }

// out[0..5] = n_layer, n_embd, n_expert, n_expert_used, n_ff_exp.
SC_EXPORT int sct_info(void* h, int64_t* out) {
    Handle* hd = as_handle(h);
    if (!hd) { g_err = "null handle"; return -1; }
    const ModelConfig& c = hd->w.config();
    out[0] = (int64_t)c.n_layer;
    out[1] = (int64_t)c.n_embd;
    out[2] = (int64_t)c.n_expert;
    out[3] = (int64_t)c.n_expert_used;
    out[4] = (int64_t)c.n_ff_exp;
    return 0;
}

// port of cpu_moe.rs: experts for one token. ids and w hold n (expert, routing weight) pairs in router order; x holds
// n_embd values; y receives n_embd values. threads = 0 uses default_threads().
SC_EXPORT int sct_experts(void* h, int layer, const int32_t* ids, const float* w, int n, const float* x, float* y,
                          int threads) {
    Handle* hd = as_handle(h);
    if (!hd) { g_err = "null handle"; return -1; }
    g_err.clear();
    ExpertLayer L;
    if (!load_expert_layer(hd->w, hd->w.config(), (size_t)layer, L, g_err)) return -1;
    std::vector<std::pair<size_t, float>> sel;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0) { g_err = "negative expert id"; return -1; }
        sel.push_back({(size_t)ids[i], w[i]});
    }
    std::vector<float> xv(x, x + L.n_embd);
    std::vector<float> yv(L.n_embd);
    if (!experts(L, xv.data(), sel, yv.data(), hd->pool_for((size_t)threads), g_err)) return -1;
    for (size_t i = 0; i < L.n_embd; i++) y[i] = yv[i];
    return 0;
}

// port of cpu_moe.rs: experts_multi. nt tokens, each with n (expert, weight) pairs (ids and w are nt * n long);
// xs is nt * n_embd, ys receives nt * n_embd.
SC_EXPORT int sct_experts_multi(void* h, int layer, int nt, const int32_t* ids, const float* w, int n, const float* xs,
                                float* ys, int threads) {
    Handle* hd = as_handle(h);
    if (!hd) { g_err = "null handle"; return -1; }
    g_err.clear();
    ExpertLayer L;
    if (!load_expert_layer(hd->w, hd->w.config(), (size_t)layer, L, g_err)) return -1;
    std::vector<std::vector<float>> hs((size_t)nt, std::vector<float>(L.n_embd));
    std::vector<std::vector<std::pair<size_t, float>>> sels((size_t)nt);
    for (int t = 0; t < nt; t++) {
        for (size_t k = 0; k < L.n_embd; k++) hs[t][k] = xs[(size_t)t * L.n_embd + k];
        for (int i = 0; i < n; i++) {
            const size_t at = (size_t)t * n + i;
            if (ids[at] < 0) { g_err = "negative expert id"; return -1; }
            sels[t].push_back({(size_t)ids[at], w[at]});
        }
    }
    std::vector<std::vector<float>> out;
    if (!experts_multi(L, hs, sels, out, hd->pool_for((size_t)threads), g_err)) return -1;
    for (int t = 0; t < nt; t++)
        for (size_t k = 0; k < L.n_embd; k++) ys[(size_t)t * L.n_embd + k] = out[t][k];
    return 0;
}

// Every integer kernel (scalar and AVX2) against the dequantized dot product (the Rust unit test). 0 when all pass.
SC_EXPORT int sct_dot_selftest(void) {
    g_err.clear();
    return int_dots_selftest(g_err) ? 0 : -1;
}

// StrataFreet test hook (cpu_moe.h int_dot_rows): y[r] = row r . x for `rows` rows of `rb` bytes of GGML type `ty`,
// through the scalar kernel (scalar != 0) or the AVX2 one. Returns 0, or -1 with sct_last_error set.
SC_EXPORT int sct_dot_rows(unsigned int ty, int scalar, const uint8_t* w, int rows, long long rb, const float* x,
                           long long n, float* y) {
    g_err.clear();
    const std::optional<Ty> t = ty_from_u32(ty);
    if (!t || rows < 0 || rb <= 0 || n <= 0) {
        g_err = "sct_dot_rows: bad arguments";
        return -1;
    }
    return int_dot_rows(*t, scalar != 0, w, (size_t)rows, (size_t)rb, x, (size_t)n, y, g_err) ? 0 : -1;
}

// default_threads() (port of main.rs physical_cores, see cpu_moe.h), for the tests and the timing run.
SC_EXPORT int sct_default_threads(void) { return (int)default_threads(); }

// ---------------------------------------------------------------- the CPU mailbox worker (cpu_worker.cpp)
//
// The Python test plays the GPU's role: it writes the request words of each layer into the mailbox buffer that
// sctw_create allocates (64-byte aligned, zeroed), raises REQ/SKIP, and reads DONE and y. sctw_layout gives the
// buffer address and the layout, so the test can view the memory through ctypes/numpy.

namespace {

// The worker, and the mailbox buffer it runs over. The buffer is test-owned: in production it is
// CudaBackend::host_mailbox's memory. The worker is destroyed before the buffer is freed.
struct WorkerHandle {
    void* mem = nullptr;
    size_t mem_bytes = 0;
    std::unique_ptr<CpuWorker> w;
    ~WorkerHandle() {
        w.reset();
        if (mem) _aligned_free(mem);
    }
};

WorkerHandle* as_worker(void* wh) { return static_cast<WorkerHandle*>(wh); }

}  // namespace

// port of cpu_worker.rs: CpuWorker::new over a test buffer for layers[0..n_layers) of the model behind `h`. Returns
// NULL on failure (sct_last_error). threads = 0 uses default_threads(). Layer indices are absolute.
SC_EXPORT void* sctw_create(void* h, const int32_t* layers, int n_layers, int threads) {
    g_err.clear();
    Handle* hd = as_handle(h);
    if (!hd) { g_err = "null handle"; return nullptr; }
    if (n_layers <= 0) { g_err = "no worker layers"; return nullptr; }
    std::vector<size_t> moe;
    size_t max_il = 0;
    for (int i = 0; i < n_layers; i++) {
        if (layers[i] < 0) { g_err = "negative layer index"; return nullptr; }
        moe.push_back((size_t)layers[i]);
        max_il = std::max(max_il, (size_t)layers[i]);
    }
    const ModelConfig& c = hd->w.config();
    const Mailbox lay = Mailbox::make(c.n_expert_used, c.n_embd);
    size_t bytes = lay.stride * (max_il + 1) * sizeof(uint32_t);
    bytes = (bytes + 63) / 64 * 64;
    std::unique_ptr<WorkerHandle> wh(new WorkerHandle());
    wh->mem = _aligned_malloc(bytes, 64);
    if (!wh->mem) { g_err = "out of memory for the mailbox"; return nullptr; }
    std::memset(wh->mem, 0, bytes);
    wh->mem_bytes = bytes;
    if (!CpuWorker::create(hd->w, static_cast<uint32_t*>(wh->mem), bytes, 0, moe, (size_t)threads, wh->w, g_err))
        return nullptr;
    return wh.release();
}

// Stops and joins the worker, then frees its buffer.
SC_EXPORT void sctw_free(void* wh) { delete as_worker(wh); }

// Stop and join only (the Rust Drop without the free). Idempotent.
SC_EXPORT void sctw_stop(void* wh) {
    WorkerHandle* p = as_worker(wh);
    if (p && p->w) p->w->stop();
}

// out[0..6] = k, n, h, y, stride (words), buffer address, buffer bytes. Returns 0.
SC_EXPORT int sctw_layout(void* wh, int64_t* out) {
    WorkerHandle* p = as_worker(wh);
    if (!p || !p->w) { g_err = "null worker"; return -1; }
    const Mailbox& L = p->w->layout;
    out[0] = (int64_t)L.k;
    out[1] = (int64_t)L.n;
    out[2] = (int64_t)L.h;
    out[3] = (int64_t)L.y;
    out[4] = (int64_t)L.stride;
    out[5] = (int64_t)(uintptr_t)p->mem;
    out[6] = (int64_t)p->mem_bytes;
    return 0;
}

// port of cpu_worker.rs: CpuWorker::begin. Returns the sequence number to publish with.
SC_EXPORT uint32_t sctw_begin(void* wh) {
    WorkerHandle* p = as_worker(wh);
    return p && p->w ? p->w->begin() : 0;
}

// port of cpu_worker.rs: CpuWorker::routing. Writes up to `cap` entries; returns the count, or -1.
SC_EXPORT int sctw_routing(void* wh, int il, int32_t* ids, float* ws, int32_t* slots, int cap) {
    WorkerHandle* p = as_worker(wh);
    if (!p || !p->w) { g_err = "null worker"; return -1; }
    if (il < 0 || !p->w->has_layer((size_t)il)) { g_err = "not a worker layer"; return -1; }
    const auto r = p->w->routing((size_t)il);
    if ((int)r.size() > cap) { g_err = "routing buffer too small"; return -1; }
    for (size_t j = 0; j < r.size(); j++) {
        ids[j] = std::get<0>(r[j]);
        ws[j] = std::get<1>(r[j]);
        slots[j] = std::get<2>(r[j]);
    }
    return (int)r.size();
}

// port of cpu_worker.rs: CpuWorker::stalled_layer. The layer, or -1.
SC_EXPORT int sctw_stalled_layer(void* wh) {
    WorkerHandle* p = as_worker(wh);
    if (!p || !p->w) return -1;
    const auto il = p->w->stalled_layer();
    return il ? (int)*il : -1;
}

// port of cpu_worker.rs: CpuWorker::wait_done. 0 when layer `il` is answered (its verdict is reset), -1 with
// sct_last_error on a timeout (timeout_ms >= 0), a stopped worker, or a bad layer.
SC_EXPORT int sctw_wait_done(void* wh, int il, int timeout_ms) {
    WorkerHandle* p = as_worker(wh);
    if (!p || !p->w) { g_err = "null worker"; return -1; }
    g_err.clear();
    return p->w->wait_done((size_t)il, timeout_ms, g_err) ? 0 : -1;
}

// port of cpu_worker.rs: CpuWorker::busy, in nanoseconds.
SC_EXPORT uint64_t sctw_busy_ns(void* wh) {
    WorkerHandle* p = as_worker(wh);
    return p && p->w ? p->w->busy_ns() : 0;
}

// ---------------------------------------------------------------- GLM-5.3-Flash (glm5-next), CPU only
//
// Test hooks for the port of glm/forward.rs, deepseek/mat.rs and deepseek/ops.rs, driven from tests\test_glm_cpu.py.
// The model is always opened with the GPU path off (no backend factory, DsGpuConfig::on false), so nothing here
// creates a CUDA context.

namespace {

struct GlmHandle {
    Gguf g;
    std::unique_ptr<GlmModel> m;
    std::vector<float> logits;
};

GlmHandle* as_glm(void* h) { return static_cast<GlmHandle*>(h); }

}  // namespace

// Opens a glm5-next GGUF on the CPU path (threads 0 = default_threads(); ctx 0 = max_ctx). Null with sct_last_error.
SC_EXPORT void* sctg_open(const char* path, int threads, long long ctx) {
    auto h = std::make_unique<GlmHandle>();
    std::string err;
    if (!h->g.open(path, err)) {
        g_err = err;
        return nullptr;
    }
    GlmLoad opt;
    opt.gpu.on = false;
    opt.threads = threads > 0 ? (size_t)threads : 0;
    opt.ctx = ctx > 0 ? (size_t)ctx : 0;
    if (!GlmModel::create(h->g, opt, h->m, err)) {
        g_err = err;
        return nullptr;
    }
    return h.release();
}

SC_EXPORT void sctg_close(void* h) { delete as_glm(h); }

// out[0..8]: n_layer, n_kda, n_dsa, n_vocab, n_embd, n_expert, n_used, n_nextn, ctx limit.
SC_EXPORT int sctg_info(void* h, int64_t* out) {
    GlmHandle* p = as_glm(h);
    if (!p) { g_err = "null handle"; return 1; }
    const GlmConfig& c = p->m->cfg;
    const int64_t v[9] = {(int64_t)c.n_layer, (int64_t)c.n_kda(), (int64_t)c.n_dsa(), (int64_t)c.n_vocab,
                          (int64_t)c.n_embd, (int64_t)c.n_expert, (int64_t)c.n_used, (int64_t)c.n_nextn,
                          (int64_t)p->m->ctx_limit()};
    std::memcpy(out, v, sizeof v);
    return 0;
}

// port of glm/mod.rs: GlmConfig::validate on the open file. 0, or 1 with the problems in sct_last_error.
SC_EXPORT int sctg_validate(void* h) {
    GlmHandle* p = as_glm(h);
    if (!p) { g_err = "null handle"; return 1; }
    std::string err;
    if (!p->m->cfg.validate(p->g, err)) { g_err = err; return 1; }
    return 0;
}

// GlmConfig::from_gguf + validate on a file without building the model (any size: only the header is read).
SC_EXPORT int sctg_check(const char* path, int64_t* out) {
    Gguf g;
    std::string err;
    if (!g.open(path, err)) { g_err = err; return 1; }
    GlmConfig c;
    if (!GlmConfig::from_gguf(g, c, err)) { g_err = err; return 1; }
    out[0] = (int64_t)c.n_layer;
    out[1] = (int64_t)c.n_kda();
    out[2] = (int64_t)c.n_dsa();
    out[3] = (int64_t)c.n_nextn;
    out[4] = (int64_t)c.n_expert;
    out[5] = (int64_t)c.n_used;
    out[6] = (int64_t)g.tensors().size();
    if (!c.validate(g, err)) { g_err = err; return 2; }
    return 0;
}

// port of glm/forward.rs: GlmModel::step. `logits` receives n_vocab floats.
SC_EXPORT int sctg_step(void* h, int32_t tok, float* logits) {
    GlmHandle* p = as_glm(h);
    if (!p) { g_err = "null handle"; return 1; }
    std::string err;
    if (!p->m->step((uint32_t)tok, p->logits, err)) { g_err = err; return 1; }
    std::memcpy(logits, p->logits.data(), p->logits.size() * sizeof(float));
    return 0;
}

// port of glm/forward.rs: GlmModel::prefill (chunked layer-major for all but the last token; STRATA_GLM_CHUNK at open).
SC_EXPORT int sctg_prefill(void* h, const int32_t* ids, int n, float* logits) {
    GlmHandle* p = as_glm(h);
    if (!p || n <= 0) { g_err = "null handle or empty prompt"; return 1; }
    std::vector<uint32_t> t(ids, ids + n);
    std::string err;
    if (!p->m->prefill(t, p->logits, err)) { g_err = err; return 1; }
    std::memcpy(logits, p->logits.data(), p->logits.size() * sizeof(float));
    return 0;
}

// port of glm/forward.rs: GlmModel::reset
SC_EXPORT int sctg_reset(void* h) {
    GlmHandle* p = as_glm(h);
    if (!p) { g_err = "null handle"; return 1; }
    p->m->reset();
    return 0;
}

// port of deepseek/ops.rs: hc_coeffs. out: pre[4], post[4], comb[16].
SC_EXPORT void sctg_hc_coeffs(const float* mixes, const float* scale, const float* base, float eps, int iters, float* out) {
    const Hc hc = hc_coeffs(mixes, scale, base, eps, (size_t)iters);
    std::memcpy(out, hc.pre.data(), 16);
    std::memcpy(out + 4, hc.post.data(), 16);
    std::memcpy(out + 8, hc.comb.data(), 64);
}

// port of deepseek/ops.rs: hc_collapse / hc_post (n values per stream).
SC_EXPORT void sctg_hc_post(const float* f, const float* residual, const float* coeffs, int n, float* out) {
    Hc hc;
    std::memcpy(hc.pre.data(), coeffs, 16);
    std::memcpy(hc.post.data(), coeffs + 4, 16);
    std::memcpy(hc.comb.data(), coeffs + 8, 64);
    hc_post(f, residual, hc, out, (size_t)n);
}

// port of glm/forward.rs: route_sigmoid. ids/weights receive n_used entries.
SC_EXPORT void sctg_route(const float* logits, const float* bias, int n, int n_used, int norm, float scale, int32_t* ids,
                          float* weights) {
    const auto r = route_sigmoid(logits, bias, (size_t)n, (size_t)n_used, norm != 0, scale);
    for (size_t i = 0; i < r.size(); i++) {
        ids[i] = (int32_t)r[i].first;
        weights[i] = r[i].second;
    }
}

// port of glm/forward.rs: l2_norm (in place).
SC_EXPORT void sctg_l2_norm(float* x, int n, float eps) { glm_l2_norm(x, (size_t)n, eps); }

// port of deepseek/ops.rs: swiglu_clamp, over n pairs.
SC_EXPORT void sctg_swiglu_clamp(const float* g, const float* u, int n, float limit, float* out) {
    for (int i = 0; i < n; i++) out[i] = swiglu_clamp(g[i], u[i], limit);
}

// port of deepseek/mat.rs: q8_roundtrip and the f16 rounding it uses.
SC_EXPORT void sctg_q8_roundtrip(unsigned int ty, int cpu, const float* x, int n, float* out) {
    const auto t = ty_from_u32(ty);
    const std::vector<float> r = q8_roundtrip(t ? *t : Ty::F32, cpu != 0, x, (size_t)n);
    std::memcpy(out, r.data(), r.size() * sizeof(float));
}
SC_EXPORT float sctg_f16_round(float v) { return f16_round(v); }

// port of deepseek/mat.rs: Mat::matvec on the CPU (no device): `rows` x `cols` weights of type `ty` in `w`,
// `cpu` = routed-expert flag, `cpu_ref` = STRATA_CPU_REF.
SC_EXPORT int sctg_matvec(unsigned int ty, const uint8_t* w, long long len, int rows, int cols, int cpu, int cpu_ref,
                          int threads, const float* x, float* y) {
    const auto t = ty_from_u32(ty);
    if (!t || (size_t)len != row_bytes(*t, (uint64_t)cols) * (size_t)rows) { g_err = "bad type or size"; return 1; }
    ThreadPool pool(threads > 0 ? (size_t)threads : default_threads());
    DsEnv env;
    env.pool = &pool;
    env.batch = true;
    env.cpu_ref = cpu_ref != 0;
    Mat m;
    m.data = w;
    m.len = (size_t)len;
    m.ty = *t;
    m.rows = (size_t)rows;
    m.cols = (size_t)cols;
    m.cpu = cpu != 0;
    m.env = &env;
    try {
        m.matvec(x, y);
    } catch (const std::exception& e) {
        g_err = e.what();
        return 1;
    }
    return 0;
}
