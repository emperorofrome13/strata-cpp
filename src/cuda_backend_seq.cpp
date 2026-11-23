// port of cuda/mod.rs: the CUDA launch wrappers of W2-C part 2b. ssm_gates, conv_silu, deltanet and conv_seq (the
// Gated DeltaNet and its causal conv), qsa_pool and qsa_select (QSA sparse attention), hc_silu / hc_mix / hc_write
// (qwen4exp hyper-connections), ple_gate / ple_conv (per-layer embeddings), rms_norm_rope_qk_rows and attn_rows (the
// per-row KV kernels of GpuModel::step_batch), moe_prefetch, and the graph machinery of Gpu::graph, Gpu::beside and
// Gpu::beside_join.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Every launch keeps the grid, block, dynamic shared memory and argument order of the Rust wrapper it ports, and
// every integer argument is passed as the 32-bit int the kernel declares (see launch in cuda_backend.h).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "cuda_backend.h"

namespace strata {

namespace {

// port of cuda/mod.rs: BESIDE (nb at or above it: launches stay on the main stream, copies go to the side stream)
constexpr size_t kBeside = SIZE_MAX - 1;
// port of cuda/mod.rs: GRAPHS_MAX (graphs kept by Gpu::graph; past the cap a forward runs uncaptured)
constexpr size_t kGraphsMax = 32;
// port of kernels.cu QSA_D (the indexer head dim, one thread per dim in qsa_pool)
constexpr size_t kQsaD = 128;

// port of cuda/mod.rs: Gpu::grid (one thread per value, 256 a block)
LaunchCfg grid_n(size_t n) { return cfg((unsigned)cuda_detail::div_ceil(n, 256), 1, 256, 0); }

bool env_set(const char* k) { return std::getenv(k) != nullptr; }

}  // namespace

// port of cuda/mod.rs: Gpu::ssm_gates (beta = sigmoid(beta); g = softplus(g + dt_bias[h]) * a[h], h = i % nv)
bool CudaBackend::ssm_gates(uint64_t beta, uint64_t g, uint64_t dt_bias, uint64_t a, size_t n, size_t nv, std::string& err) {
    return launch("ssm_gates", grid_n(n), err, beta, g, dt_bias, a, (int32_t)n, (int32_t)nv);
}

// port of cuda/mod.rs: Gpu::conv_silu (one token: causal conv over the state and the input, SiLU, state rolled forward;
// the first nn blocks of 128 channels are then l2-normalized)
bool CudaBackend::conv_silu(uint64_t qkv, uint64_t state, uint64_t w, uint64_t y, size_t ch, size_t kw, size_t nn, float eps,
                            std::string& err) {
    return launch("conv_silu", cfg((unsigned)cuda_detail::div_ceil(ch, 128), 1, 128, 0), err, qkv, state, w, y, (int32_t)ch,
                  (int32_t)kw, (int32_t)nn, eps);
}

// port of cuda/mod.rs: Gpu::conv_seq (conv_seq over t * ch values, then conv_state rolls the state past the chunk and
// writes the snapshots of the first min(nsnap, t - 1) tokens)
bool CudaBackend::conv_seq(uint64_t qkv, uint64_t state, uint64_t w, uint64_t y, size_t ch, size_t kw, size_t t, uint64_t snap,
                           size_t nsnap, std::string& err) {
    if (!launch("conv_seq", grid_n(t * ch), err, qkv, state, w, y, (int32_t)ch, (int32_t)kw, (int32_t)t)) return false;
    return launch("conv_state", grid_n(ch), err, qkv, state, (int32_t)ch, (int32_t)kw, (int32_t)t, snap, (int32_t)nsnap);
}

// port of cuda/mod.rs: Gpu::deltanet (one warp per 32 / L state rows of a value head; grid (value heads, dv / (8 R)),
// block 256, R = 512 / dk rows per warp group)
bool CudaBackend::deltanet(uint64_t y, uint64_t s, uint64_t beta, uint64_t g, uint64_t dt_bias, uint64_t a, uint64_t o,
                           size_t nk, size_t nv, bool interleave, size_t dk, size_t dv, float scale, size_t t, size_t ych,
                           uint64_t snap, size_t nsnap, std::string& err) {
    const size_t rows = 512 / std::max<size_t>(dk, 1);
    if (!(dk == 32 || dk == 64 || dk == 128 || dk == 256) || dv % rows != 0) {
        err = "deltanet: head dims " + std::to_string(dk) + "/" + std::to_string(dv) + " unsupported";
        return false;
    }
    return launch("deltanet_seq", cfg((unsigned)nv, (unsigned)cuda_detail::div_ceil(dv, 8 * rows), 256, 0), err, y, s, beta, g,
                  o, (int32_t)nk, (int32_t)nv, (int32_t)interleave, (int32_t)dk, (int32_t)dv, scale, (int32_t)t, (int32_t)ych,
                  snap, (int32_t)nsnap, dt_bias, a);
}

// port of cuda/mod.rs: Gpu::qsa_pool (CTA i of the grid pools block p0 / 4 + i of the t raw indexer keys; the ring holds
// the raw keys of the last 32 positions)
bool CudaBackend::qsa_pool(uint64_t ik, uint64_t ring, uint64_t pool, uint64_t w, uint64_t step, size_t t, size_t n_rot,
                           float base, float eps, std::string& err) {
    return launch("qsa_pool", cfg((unsigned)(t / 4 + 1), 1, (unsigned)kQsaD, 0), err, ik, ring, pool, w, step, (int32_t)t,
                  (int32_t)n_rot, base, eps);
}

// port of cuda/mod.rs: Gpu::qsa_select (qsa_score scores the blocks of query rows r0 ... into score; qsa_select keeps the
// top blocks of each row into sel, or into the mask when mw is not 0)
bool CudaBackend::qsa_select(uint64_t q, uint64_t pool, uint64_t score, uint64_t sel, uint64_t mask, uint64_t step, size_t r0,
                             size_t rows, size_t nbs, size_t top, size_t sst, size_t mw, std::string& err) {
    if (!launch("qsa_score", cfg(64, (unsigned)cuda_detail::div_ceil(rows, 8), 256, 0), err, q, pool, score, step, (int32_t)r0,
                (int32_t)rows, (int32_t)nbs, (int32_t)top))
        return false;
    return launch("qsa_select", cfg((unsigned)rows, 1, 1024, 0), err, score, sel, mask, step, (int32_t)r0, (int32_t)nbs,
                  (int32_t)top, (int32_t)sst, (int32_t)mw);
}

// port of cuda/mod.rs: Gpu::hc_silu (x = silu(x * sc) over n values)
bool CudaBackend::hc_silu(uint64_t x, float sc, size_t n, std::string& err) {
    return launch("hc_silu", grid_n(n), err, x, sc, (int32_t)n);
}

// port of cuda/mod.rs: Gpu::hc_mix (grid.y = tokens: out[t][i] = sum over the hc streams of xn * sigmoid(g) / hc)
bool CudaBackend::hc_mix(uint64_t xn, uint64_t g, uint64_t out, size_t n, size_t hc, size_t nt, std::string& err) {
    return launch("hc_mix", cfg((unsigned)cuda_detail::div_ceil(n, 256), (unsigned)nt, 256, 0), err, xn, g, out, (int32_t)n,
                  (int32_t)hc);
}

// port of cuda/mod.rs: Gpu::hc_write (grid.y = tokens: stream c of r += out * 2 sigmoid(inj / hc); clear zeroes out)
bool CudaBackend::hc_write(uint64_t r, uint64_t out, uint64_t inj, size_t n, size_t hc, size_t nt, bool clear, std::string& err) {
    return launch("hc_write", cfg((unsigned)cuda_detail::div_ceil(n, 256), (unsigned)nt, 256, 0), err, r, out, inj, (int32_t)n,
                  (int32_t)hc, (int32_t)clear);
}

// port of cuda/mod.rs: Gpu::ple_gate (block (c, t): the key . query of stream c, gate and value per stream)
bool CudaBackend::ple_gate(uint64_t key, uint64_t query, uint64_t value, uint64_t gated, size_t n, size_t hc, size_t nt,
                           std::string& err) {
    return launch("ple_gate", cfg((unsigned)hc, (unsigned)nt, 256, 0), err, key, query, value, gated, (int32_t)n);
}

// port of cuda/mod.rs: Gpu::ple_conv (the causal conv of d channels over nt tokens at positions step[0] ...)
bool CudaBackend::ple_conv(uint64_t r, uint64_t gated, uint64_t norm, uint64_t kw, uint64_t ring, uint64_t step, size_t d,
                           size_t nt, std::string& err) {
    return launch("ple_conv", grid_n(d), err, r, gated, norm, kw, ring, step, (int32_t)d, (int32_t)nt);
}

// port of cuda/mod.rs: Gpu::rms_norm_rope_qk_rows (the q heads of nt tokens, then their k heads and v, each token into
// its own caches kvt[2t] and kvt[2t + 1] at position step[2t]; one launch)
bool CudaBackend::rms_norm_rope_qk_rows(uint64_t q, size_t q_stride, uint64_t qw, uint64_t qo, size_t nh, uint64_t k,
                                        uint64_t kw, uint64_t ko, size_t nkv, size_t hd, float eps, size_t n_rot, uint64_t step,
                                        float base, uint64_t v, uint64_t kvt, size_t nt, std::string& err) {
    if (!(n_rot <= std::min<size_t>(hd, 512) && hd % 32 == 0)) {
        err = "rms_norm_rope: head of " + std::to_string(hd) + ", n_rot " + std::to_string(n_rot);
        return false;
    }
    uint64_t fr = 0;
    if (!rope_fr(base, n_rot, fr, err)) return false;
    return launch("rms_norm_rope", cfg((unsigned)(nt * (nh + nkv)), 1, 256, 0), err, k, (int32_t)hd, kw, ko, (int32_t)hd, eps,
                  (int32_t)nkv, (int32_t)n_rot, step, base, v, (uint64_t)0, (uint64_t)0, (int32_t)(nt * nh), q, (int32_t)q_stride,
                  qw, qo, (int32_t)nh, fr, kvt);
}

// port of cuda/mod.rs: Gpu::attn_rows (decode attention for nt tokens of different sequences: gq_min 0, so the GQA
// kernel; each token reads its own caches kvt[2t], kvt[2t + 1] over positions 0 ..= step[2t])
bool CudaBackend::attn_rows(uint64_t q, uint64_t qg, uint64_t out, uint64_t part, uint64_t step, size_t nh, size_t nkv,
                            size_t hd, float scale, size_t nt, uint64_t sink, size_t win, uint64_t kvt, size_t splits,
                            std::string& err) {
    return attn(q, 0, 0, qg, out, part, step, nh, nkv, hd, scale, 0, nt, 0, 0, 0, sink, win, kvt, splits, 0, 0, err);
}

// port of cuda/mod.rs: Gpu::moe_prefetch (hint the cached slots of the k experts into L2; grid.x = line ranges of 128
// bytes, grid.y = experts; a performance hint only)
bool CudaBackend::moe_prefetch(uint64_t ids, size_t k, uint64_t table, uint64_t base, size_t stride, size_t bytes,
                               std::string& err) {
    const size_t lines = bytes >> 7;
    const unsigned gx = (unsigned)std::max<size_t>(cuda_detail::div_ceil(lines, 256), 1);
    return launch("moe_prefetch", cfg(gx, (unsigned)k, 256, 0), err, ids, (int32_t)k, table, base, (int32_t)stride,
                  (int32_t)bytes);
}

// port of cuda/mod.rs: Gpu::graph. The first call for a key captures what `f` queues on the main stream and instantiates
// it, then launches it; later calls launch the instantiated graph without running `f`. STRATA_CUDA_SYNC and
// STRATA_NO_GRAPH run `f` directly, as does a call past kGraphsMax graphs. A rope table the capture lacked is made
// after the capture, and the graph is not kept (its next call captures again, with the table in place).
bool CudaBackend::graph(uint64_t key, const OpFn& f, std::string& err) {
    if (!bind(err)) return false;
    size_t kept = 0;
    {
        std::lock_guard<std::mutex> g(graph_mu_);
        auto it = graphs_.find(key);
        if (it != graphs_.end()) {
            const CUresult r = cuGraphLaunch(it->second, stream_);
            if (r != CUDA_SUCCESS) {
                err = cuda_detail::cu_error("cuGraphLaunch", r);
                return false;
            }
            return true;
        }
        kept = graphs_.size();
    }
    if (env_set("STRATA_CUDA_SYNC") || env_set("STRATA_NO_GRAPH") || kept >= kGraphsMax) return f(err);

    // Capture what `f` queues on the main stream (its side-stream work joins through the events of `batch` / `beside`).
    CUresult r = cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    if (r != CUDA_SUCCESS) {
        err = cuda_detail::cu_error("cuStreamBeginCapture", r);
        return false;
    }
    const bool ok = f(err);
    CUgraph captured = nullptr;
    const CUresult end = cuStreamEndCapture(stream_, &captured);
    if (end != CUDA_SUCCESS) {
        if (captured) cuGraphDestroy(captured);
        err = cuda_detail::cu_error("cuStreamEndCapture", end);
        return false;
    }
    if (!ok) {
        if (captured) cuGraphDestroy(captured);
        return false;
    }
    if (!captured) {
        err = "empty CUDA graph";
        return false;
    }
    CUgraphExec exec = nullptr;
    r = cuGraphInstantiateWithFlags(&exec, captured, CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
    cuGraphDestroy(captured);
    if (r != CUDA_SUCCESS) {
        err = cuda_detail::cu_error("cuGraphInstantiate", r);
        return false;
    }
    if ((r = cuGraphLaunch(exec, stream_)) != CUDA_SUCCESS) {
        cuGraphExecDestroy(exec);
        err = cuda_detail::cu_error("cuGraphLaunch", r);
        return false;
    }
    // A rope table the capture lacked: make it now (the frequencies are computed by the kernel in this graph), and do
    // not keep the graph, so the next call captures again with the table in place.
    std::vector<std::pair<float, size_t>> want;
    {
        std::lock_guard<std::mutex> g(fr_mu_);
        want.swap(fr_want_);
    }
    if (!want.empty()) {
        cuGraphExecDestroy(exec);
        for (const auto& [base, n_rot] : want) {
            uint64_t unused = 0;
            if (!rope_fr(base, n_rot, unused, err)) return false;
        }
        return true;
    }
    std::lock_guard<std::mutex> g(graph_mu_);
    graphs_.emplace(key, exec);
    return true;
}

// port of cuda/mod.rs: Gpu::beside (`f` queues its host copies on the side stream, beside the main stream's launches;
// the side stream first waits for the main stream, and the main stream waits for the side stream's previous beside)
bool CudaBackend::beside(const OpFn& f, std::string& err) {
    if (!bind(err) || !join(side_, stream_, err)) return false;
    nb_.store(kBeside);
    const bool ok = f(err);
    nb_.store(SIZE_MAX);
    CUevent e = nullptr;
    CUresult r = cuEventCreate(&e, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) r = cuEventRecord(e, side_);
    if (r != CUDA_SUCCESS) {
        if (e) cuEventDestroy(e);
        err = cuda_detail::cu_error("beside", r);
        return false;
    }
    if (lag_) {
        const CUresult w = cuStreamWaitEvent(stream_, lag_, 0);
        cuEventDestroy(lag_);
        lag_ = nullptr;
        if (w != CUDA_SUCCESS) {
            cuEventDestroy(e);
            err = cuda_detail::cu_error("beside wait", w);
            return false;
        }
    }
    lag_ = e;
    return ok;
}

// port of cuda/mod.rs: Gpu::beside_join (the main stream waits for every side-stream op so far)
bool CudaBackend::beside_join(std::string& err) {
    if (!bind(err)) return false;
    if (lag_) {
        cuEventDestroy(lag_);
        lag_ = nullptr;
    }
    return join(stream_, side_, err);
}

}  // namespace strata
