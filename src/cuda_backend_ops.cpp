// port of cuda/mod.rs: the CUDA launch wrappers of W2-C part 2a. gemm / gemm_dense (gemm_raw, q8_x, the int8
// path), prequant_gemm_x, rms_norm / rms_norm_q / rms_norm_streams, argmax_rows, l2_norm, rope_pairs and the
// rms_norm_rope family (rope_fr), add, swiglu / swiglu_q, norm_gate / norm_gate_q / norm_gate_sig, kv_store,
// the decode attention (attn, attn_decode_gq, attn_decode_q), attn_prefill_m, router_topk, moe_route,
// moe_combine and moe_combine_seq.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Every launch keeps the grid, block, dynamic shared memory and argument order of the Rust wrapper it ports, and
// every integer argument is passed as the 32-bit int the kernel declares (see launch in cuda_backend.h). Deviations
// are the ones of cuda_backend.h (out-parameters for Rust's Result<u64>) and the device-side ops that this part
// the part 2b ops (cuda_backend_seq.cpp) cover the per-row KV kernels, the QSA indexer and the graph capture.
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "cuda_backend.h"

namespace strata {

namespace {

// port of cuda/mod.rs: llama_numerics (STRATA_LLAMA_NUMERICS: one scale per 256 values, not per 32)
// port of cuda/mod.rs: BESIDE (nb at or above it: launches stay on the main stream, no side-stream alternation)
constexpr size_t kBeside = SIZE_MAX - 1;

bool llama_numerics() { return std::getenv("STRATA_LLAMA_NUMERICS") != nullptr; }

// port of cuda/mod.rs: Gpu::grid (one thread per value, 256 a block)
LaunchCfg grid_of(size_t n) { return cfg((unsigned)cuda_detail::div_ceil(n, 256), 1, 256, 0); }

// port of cuda/mod.rs: quant_xq layout of `rows` rows of n values: n int8 values per row, then n / 32 f32 scales
// per row (rows * (n + n / 8) bytes), which the xq consumers read.
size_t xq_bytes(size_t rows, size_t n) { return rows * (n + n / 8); }

}  // namespace

// port of cuda/mod.rs: Gpu::rope_fr. The table is made by rope_freq on the main stream, then synchronized, as the
// Rust does (the launch is not routed through `st`, so it never lands on the side stream).
bool CudaBackend::rope_fr(float base, size_t n_rot, uint64_t& out, std::string& err) {
    std::lock_guard<std::mutex> g(fr_mu_);
    const std::pair<uint32_t, size_t> key{std::bit_cast<uint32_t>(base), n_rot};
    if (auto it = fr_.find(key); it != fr_.end()) {
        out = it->second.addr();
        return true;
    }
    // While a graph is captured a missing table gives 0 (the kernel computes the frequencies itself) and is made
    // after the capture, which `graph` then repeats (cuda/mod.rs: rope_fr checks the capture status first).
    if (!bind(err)) return false;
    CUstreamCaptureStatus cap = CU_STREAM_CAPTURE_STATUS_NONE;
    if (const CUresult c = cuStreamIsCapturing(stream_, &cap); c != CUDA_SUCCESS) {
        err = cuda_detail::cu_error("cuStreamIsCapturing", c);
        return false;
    }
    if (cap != CU_STREAM_CAPTURE_STATUS_NONE) {
        fr_want_.emplace_back(base, n_rot);
        out = 0;
        return true;
    }
    Buf t;
    if (!alloc(8 * std::max<size_t>(n_rot / 2, 1), t, err)) return false;
    uint64_t p = t.addr();
    int32_t nr = (int32_t)n_rot;
    void* args[] = {&p, &base, &nr};
    if (!bind(err)) return false;
    auto it = fns_.find("rope_freq");
    if (it == fns_.end()) {
        err = "kernel rope_freq is not loaded";
        return false;
    }
    const CUresult r = cuLaunchKernel(it->second, 1, 1, 1, 256, 1, 1, 0, stream_, args, nullptr);
    if (r != CUDA_SUCCESS) {
        err = cuda_detail::cu_error("launch rope_freq", r);
        return false;
    }
    if (!wait(err)) return false;
    fr_.emplace(key, std::move(t));
    out = p;
    return true;
}

// port of cuda/mod.rs: Gpu::q8_x (the caller holds q8_mu_)
bool CudaBackend::q8_x(Ty ty, size_t cols, size_t n_x, uint64_t x, size_t x_stride, std::optional<uint64_t>& xq,
                       std::string& err) {
    xq.reset();
    if (!int8_mma_ || !cuda_mmq_policy(ty) || cols % 128 != 0) return true;
    const std::array<uint64_t, 4> key{x, x_stride, n_x, cols};
    if (q8_key_ && *q8_key_ == key) {
        xq = q8_.addr();
        return true;
    }
    const size_t need = n_x * (cols + cols / 8 + cols / 4);
    if (q8_.len() < need && !alloc(need, q8_, err)) return false;
    xq = q8_.addr();
    if (!launch("quant_q8", cfg((unsigned)n_x, 1, 256, 0), err, x, (int32_t)x_stride, *xq, (int32_t)cols)) return false;
    q8_key_ = key;
    return true;
}

// port of cuda/mod.rs: Gpu::gemm_raw. A tile list (tiles != 0) runs gemm_* and gemms_* (each takes the tiles of its
// size); a dense matrix (tiles == 0) runs the sm variant when it has few blocks or at most 32 positions.
bool CudaBackend::gemm_raw(Ty ty, size_t rows, size_t cols, uint64_t tiles, size_t n_tiles, uint64_t wd, size_t nd,
                           uint64_t x, size_t x_stride, size_t n_x, uint64_t xrow, uint64_t y, size_t y_stride, int32_t act,
                           std::string& err) {
    // Partial super-blocks (Q2_0's 640 columns, BF16's 320) are fine: K steps are 64 deep.
    if (cols % 64 != 0) {
        err = std::string(ty_name(ty)) + " matrix with " + std::to_string(cols) + " columns: GEMM needs a multiple of 64";
        return false;
    }
    if (n_tiles == 0) return true;
    const size_t gx = cuda_detail::div_ceil(rows, 64);
    const bool sm = tiles == 0 && (gx * n_tiles < 112 || nd <= 32);
    const unsigned gy = sm ? (unsigned)cuda_detail::div_ceil(nd, 32) : (unsigned)n_tiles;
    const LaunchCfg c = cfg((unsigned)gx, gy, (unsigned)(2 * GEMM_BM), 0);
    const bool same = (act & SAME_X) != 0;
    act &= ~SAME_X;
    std::lock_guard<std::mutex> g(q8_mu_);
    const std::array<uint64_t, 4> key{x, x_stride, n_x, cols};
    const bool reuse = same && q8_key_ && *q8_key_ == key;
    // The int8 path loads q8 rows 16 bytes at a time: its stride (cols * 11 / 8) is aligned only for cols % 128 == 0.
    // Volta and Turing have no int8 tensor-core mma, so their kernels are not loaded and they take the f16 path.
    const bool int8 = i8_gemm_ && int8_mma_ && cuda_mmq_policy(ty) && cols % 128 == 0;
    if (!int8 || (same && !reuse && nb_.load() < kBeside)) {
        if (!same) q8_key_.reset();
        const std::string kn = cuda_detail::kname(sm ? "gemms" : "gemm", ty);
        if (!launch(kn.c_str(), c, err, tiles, wd, (int32_t)nd, x, (int32_t)x_stride, xrow, y, (int32_t)y_stride,
                    (int32_t)rows, (int32_t)(cols / 64), act))
            return false;
        // Tile lists: gemm_* skips the tiles of at most 32 positions, which gemms_* takes.
        if (tiles != 0) {
            const std::string kn2 = cuda_detail::kname("gemms", ty);
            if (!launch(kn2.c_str(), c, err, tiles, wd, (int32_t)nd, x, (int32_t)x_stride, xrow, y, (int32_t)y_stride,
                        (int32_t)rows, (int32_t)(cols / 64), act))
                return false;
        }
        return true;
    }
    std::optional<uint64_t> xq;
    if (!q8_x(ty, cols, n_x, x, x_stride, xq, err)) return false;
    if (!xq) {
        err = "int8 GEMM without a quantized x";
        return false;
    }
    const std::string kn = cuda_detail::kname(sm ? "gemmis" : "gemmi", ty);
    if (!launch(kn.c_str(), c, err, tiles, wd, (int32_t)nd, *xq, xrow, y, (int32_t)y_stride, (int32_t)rows,
                (int32_t)(cols / 64), act))
        return false;
    if (tiles != 0) {
        const std::string kn2 = cuda_detail::kname("gemmis", ty);
        if (!launch(kn2.c_str(), c, err, tiles, wd, (int32_t)nd, *xq, xrow, y, (int32_t)y_stride, (int32_t)rows,
                    (int32_t)(cols / 64), act))
            return false;
    }
    return true;
}

// port of cuda/mod.rs: Gpu::gemm (a tile list)
bool CudaBackend::gemm(Ty ty, size_t rows, size_t cols, uint64_t tiles, size_t n_tiles, uint64_t x, size_t x_stride,
                       size_t n_x, uint64_t xrow, uint64_t y, size_t y_stride, int32_t act, std::string& err) {
    return gemm_raw(ty, rows, cols, tiles, n_tiles, 0, 0, x, x_stride, n_x, xrow, y, y_stride, act, err);
}

// port of cuda/mod.rs: Gpu::gemm_dense (y[m] (op) W x[m] for the n positions of x; one tile per 128 positions)
bool CudaBackend::gemm_dense(const GpuTensor& w, size_t n, uint64_t x, size_t x_stride, uint64_t y, size_t y_stride,
                             int32_t act, std::string& err) {
    return gemm_raw(w.ty, w.rows, w.cols, 0, cuda_detail::div_ceil(n, GEMM_BM), w.ptr, n, x, x_stride, n, 0, y, y_stride,
                    act, err);
}

// port of cuda/mod.rs: Gpu::prequant_gemm_x (quantizes x for the int8 GEMM ahead of a gemm that reuses it)
bool CudaBackend::prequant_gemm_x(const GpuTensor& w, size_t n, uint64_t x, size_t x_stride, std::string& err) {
    std::lock_guard<std::mutex> g(q8_mu_);
    std::optional<uint64_t> xq;
    return q8_x(w.ty, w.cols, n, x, x_stride, xq, err);
}

// port of cuda/mod.rs: Gpu::rms_norm_q (rows of n: out = rmsnorm(x) * w; with res, x += res first; xq as quant_xq)
bool CudaBackend::rms_norm_q(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t out_stride, size_t n, size_t rows,
                             float eps, uint64_t xq, uint64_t res, std::string& err) {
    return launch("rms_norm", cfg((unsigned)rows, 1, 512, 0), err, x, (int32_t)x_stride, w, out, (int32_t)out_stride,
                  (int32_t)n, eps, xq, res, (int32_t)1);
}

// port of cuda/mod.rs: Gpu::rms_norm
bool CudaBackend::rms_norm(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t out_stride, size_t n, size_t rows,
                           float eps, std::string& err) {
    return rms_norm_q(x, x_stride, w, out, out_stride, n, rows, eps, 0, 0, err);
}

// port of cuda/mod.rs: Gpu::rms_norm_streams (nt stacks of hc rows, one block per row, gamma shared by the stacks)
bool CudaBackend::rms_norm_streams(uint64_t x, uint64_t w, uint64_t out, size_t n, size_t hc, size_t nt, float eps,
                                   std::string& err) {
    return launch("rms_norm", cfg((unsigned)(nt * hc), 1, 512, 0), err, x, (int32_t)n, w, out, (int32_t)n, (int32_t)n, eps,
                  (uint64_t)0, (uint64_t)0, (int32_t)hc);
}

// port of cuda/mod.rs: Gpu::argmax_rows (AMAX_PARTS blocks per row share the row; one block reads ~1 MB of logits)
bool CudaBackend::argmax_rows(uint64_t x, size_t n, size_t rows, uint64_t out, std::string& err) {
    if (rows * cuda_detail::kAmaxParts > am_.len() / 8) {
        err = "argmax_rows: " + std::to_string(rows) + " rows";
        return false;
    }
    const uint64_t part = am_.addr();
    if (!launch("argmax_part", cfg((unsigned)cuda_detail::kAmaxParts, (unsigned)rows, 256, 0), err, x, (int32_t)n, part))
        return false;
    return launch("argmax_rows", cfg((unsigned)rows, 1, (unsigned)cuda_detail::kAmaxParts, 0), err, part, (int32_t)n, out);
}

// port of cuda/mod.rs: Gpu::l2_norm (in place; row r is vector r % per of token r / per, tokens `stride` apart)
bool CudaBackend::l2_norm(uint64_t x, size_t n, size_t per, size_t stride, size_t rows, float eps, std::string& err) {
    return launch("l2_norm", cfg((unsigned)rows, 1, 128, 0), err, x, (int32_t)n, (int32_t)per, (int32_t)stride, eps);
}

// port of cuda/mod.rs: Gpu::rope_pairs (NEOX pairs lo..hi; token t sits at position step[0] + t)
bool CudaBackend::rope_pairs(uint64_t x, size_t stride, size_t heads, size_t per, size_t n_rot, uint64_t step, float base,
                             size_t lo, size_t hi, std::string& err) {
    return launch("rope_neox", cfg((unsigned)(heads / per), 1, (unsigned)(n_rot / 2), 0), err, x, (int32_t)stride,
                  (int32_t)per, (int32_t)n_rot, step, base, (int32_t)lo, (int32_t)hi);
}

// port of cuda/mod.rs: Gpu::rms_norm_rope_kv (rows heads of n: rmsnorm, NEOX rope of the first n_rot dims, then
// kv_store of the head and of v when kc is not 0)
bool CudaBackend::rms_norm_rope_kv(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t n, size_t rows, float eps,
                                   size_t per, size_t n_rot, uint64_t step, float base, uint64_t v, uint64_t kc, uint64_t vc,
                                   std::string& err) {
    if (!(n_rot <= std::min<size_t>(n, 512) && n % 32 == 0)) {
        err = "rms_norm_rope: head of " + std::to_string(n) + ", n_rot " + std::to_string(n_rot);
        return false;
    }
    uint64_t fr = 0;
    if (!rope_fr(base, n_rot, fr, err)) return false;
    return launch("rms_norm_rope", cfg((unsigned)rows, 1, 256, 0), err, x, (int32_t)x_stride, w, out, (int32_t)n, eps,
                  (int32_t)per, (int32_t)n_rot, step, base, v, kc, vc, (int32_t)0, (uint64_t)0, (int32_t)0, (uint64_t)0,
                  (uint64_t)0, (int32_t)0, fr, (uint64_t)0);
}

// port of cuda/mod.rs: Gpu::rms_norm_rope (no cache store)
bool CudaBackend::rms_norm_rope(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t n, size_t rows, float eps,
                                size_t per, size_t n_rot, uint64_t step, float base, std::string& err) {
    return rms_norm_rope_kv(x, x_stride, w, out, n, rows, eps, per, n_rot, step, base, 0, 0, 0, err);
}

// port of cuda/mod.rs: Gpu::rms_norm_rope_qk (the q heads first, then the k heads and their v, one launch)
bool CudaBackend::rms_norm_rope_qk(uint64_t q, size_t q_stride, uint64_t qw, uint64_t qo, size_t qrows, size_t qper,
                                   uint64_t k, uint64_t kw, uint64_t ko, size_t krows, size_t kper, size_t n, float eps,
                                   size_t n_rot, uint64_t step, float base, uint64_t v, uint64_t kc, uint64_t vc,
                                   std::string& err) {
    if (!(n_rot <= std::min<size_t>(n, 512) && n % 32 == 0)) {
        err = "rms_norm_rope: head of " + std::to_string(n) + ", n_rot " + std::to_string(n_rot);
        return false;
    }
    uint64_t fr = 0;
    if (!rope_fr(base, n_rot, fr, err)) return false;
    return launch("rms_norm_rope", cfg((unsigned)(qrows + krows), 1, 256, 0), err, k, (int32_t)n, kw, ko, (int32_t)n, eps,
                  (int32_t)kper, (int32_t)n_rot, step, base, v, kc, vc, (int32_t)qrows, q, (int32_t)q_stride, qw, qo,
                  (int32_t)qper, fr, (uint64_t)0);
}

// port of cuda/mod.rs: Gpu::add (x += y)
bool CudaBackend::add(uint64_t x, uint64_t y, size_t n, std::string& err) {
    return launch("add", grid_of(n), err, x, y, (int32_t)n);
}

// port of cuda/mod.rs: Gpu::swiglu (g = silu(g) * u)
bool CudaBackend::swiglu(uint64_t g, uint64_t u, size_t n, std::string& err) {
    return launch("swiglu", grid_of(n), err, g, u, (int32_t)n);
}

// port of cuda/mod.rs: Gpu::swiglu_q (silu(g) * u of rows of n, quantized into xq as quant_xq does; g is left as is)
bool CudaBackend::swiglu_q(uint64_t g, uint64_t u, size_t n, size_t rows, uint64_t xq, uint64_t& xq_out, std::string& err) {
    xq_out = xq;
    return launch("swiglu_q", cfg((unsigned)cuda_detail::div_ceil(n, 256), (unsigned)rows, 32, 0), err, g, u, (int32_t)n, xq);
}

// port of cuda/mod.rs: Gpu::norm_gate_q (heads of n: out = rmsnorm(o) * w * silu(z), in place; xq as quant_xq does).
// llama numerics take one scale per 256 values, more than a head, and n not a multiple of 32 has no quant layout.
bool CudaBackend::norm_gate_q(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, uint64_t xq,
                              uint64_t& xq_out, std::string& err) {
    xq_out = (llama_numerics() || n % 32 != 0) ? 0 : xq;
    return launch("norm_gate", cfg((unsigned)heads, 1, 128, 0), err, o, w, z, (int32_t)n, eps, xq_out, (int32_t)0);
}

// port of cuda/mod.rs: Gpu::norm_gate (no xq)
bool CudaBackend::norm_gate(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, std::string& err) {
    uint64_t unused = 0;
    return norm_gate_q(o, w, z, n, heads, eps, 0, unused, err);
}

// port of cuda/mod.rs: Gpu::norm_gate_sig (sigmoid(z) in place of silu(z), qwen4exp)
bool CudaBackend::norm_gate_sig(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, uint64_t xq,
                                uint64_t& xq_out, std::string& err) {
    xq_out = (llama_numerics() || n % 32 != 0) ? 0 : xq;
    return launch("norm_gate", cfg((unsigned)heads, 1, 128, 0), err, o, w, z, (int32_t)n, eps, xq_out, (int32_t)1);
}

// port of cuda/mod.rs: Gpu::kv_store (t tokens of n values at positions step[0] ...; q8 when the cache is q8)
bool CudaBackend::kv_store(uint64_t k, uint64_t v, uint64_t kc, uint64_t vc, size_t n, size_t t, uint64_t step, size_t hd,
                           std::string& err) {
    if (kv_q8_ && !(hd % 32 == 0 && n % hd == 0)) {
        err = "q8 KV cache: head_dim " + std::to_string(hd) + " is not a multiple of 32";
        return false;
    }
    return launch("kv_store", grid_of(n * t), err, k, v, kc, vc, (int32_t)n, (int32_t)(n * t), step, (int32_t)hd);
}

// port of cuda/mod.rs: Gpu::attn. One launch serves all nt tokens (block x serves token x % nt). attn_decode_<hd>
// (ns1 splits, one head per block) runs when gq_min != 0; attn_decode_gqa* (one block per KV head group, ns splits)
// runs when gq_min != i32::MAX; attn_merge then combines the splits and applies the gate and sink. ns is ATT_SPLITS
// unless kvt selects per-row caches.
bool CudaBackend::attn(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                       size_t nh, size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t xq, uint64_t sel,
                       size_t sst, uint64_t sink, size_t win, uint64_t kvt, size_t ns1, uint64_t bias, size_t extent,
                       std::string& err) {
    if (hd != 64 && hd != 128 && hd != 256 && hd != 512) {
        err = "attention head_dim " + std::to_string(hd) + " unsupported (64, 128, 256 or 512)";
        return false;
    }
    if (nkv == 0 || nh % nkv != 0) {
        err = "attention: " + std::to_string(nh) + " query heads over " + std::to_string(nkv) + " KV heads";
        return false;
    }
    const size_t group = nh / nkv;  // query heads per KV head, served by one block (up to 8, G * hd <= 2048)
    size_t g = 0;
    for (size_t cand : {8, 4, 2, 1}) {
        if (group % cand == 0 && cand * hd <= 2048) {
            g = cand;
            break;
        }
    }
    if (g == 0) {
        err = "attention: no query-head group for " + std::to_string(group) + " heads of " + std::to_string(hd);
        return false;
    }
    const size_t ns = kvt != 0 ? ns1 : ATT_SPLITS;
    const char* ink_dbg = std::getenv("STRATA_INK_DEBUG");  // set and non-zero, as gpu_model.cpp's env_flag
    if (ink_dbg && *ink_dbg && std::atoi(ink_dbg) != 0)
        std::fprintf(stderr, "  [attn] bias=%llu extent=%zu hd=%zu nh=%zu\n", (unsigned long long)bias, extent, hd, nh);
    const int32_t pz = (int32_t)(nh * ns * (hd + 2));  // a token's partials, as attn_merge strides them
    // gq_min 0 or i32::MAX fixes the regime, so the kernel that would exit at once is not launched.
    if (gq_min != 0) {
        const std::string kn = "attn_decode_" + std::to_string(hd);
        if (!launch(kn.c_str(), cfg((unsigned)(nh * nt), (unsigned)ns1, 256, (unsigned)(8 * hd * 4)), err, q, kc, vc, part,
                    step, (int32_t)nh, (int32_t)nkv, (int32_t)hd, scale, gq_min, pz, (int32_t)nt, sel, (int32_t)sst, sink,
                    (int32_t)win, kvt, bias, (int32_t)extent))
            return false;
    }
    // attn_decode_gqa with the f16 cache at head_dim 128 or 256 also holds 2 chunks of K and V for its cp.async,
    // attn_decode_gqa4_128 one. Turing has 64 KB of shared memory a block: head_dim 256 chunks do not fit there, so it
    // loads them through registers instead (ATT_RUN_CP in kernels.cu).
    const bool cp = !kv_q8_ && (hd == 128 || (hd == 256 && int8_mma_));
    const bool g4 = cp && g == 4 && hd == 128;
    std::string name;
    size_t cp_bytes;
    if (g4) {
        name = "attn_decode_gqa4_128";
        cp_bytes = 2 * 32 * hd * 2;
    } else {
        name = "attn_decode_gqa_" + std::to_string(hd);
        cp_bytes = cp ? 2 * 2 * 32 * hd * 2 : 0;
    }
    if (gq_min != std::numeric_limits<int32_t>::max()) {
        if (!launch(name.c_str(), cfg((unsigned)(nh / g * nt), (unsigned)ns, 256, (unsigned)(8 * hd * 4 + cp_bytes)), err, q,
                    kc, vc, part, step, (int32_t)nh, (int32_t)nkv, (int32_t)hd, scale, gq_min, pz, (int32_t)nt, sel,
                    (int32_t)sst, sink, (int32_t)win, kvt, bias, (int32_t)extent))
            return false;
    }
    LaunchCfg m;
    m.gx = (unsigned)nh;
    m.gy = 1;
    m.gz = (unsigned)nt;
    m.bx = 256;
    m.by = 1;
    m.bz = 1;
    m.smem = 0;
    return launch("attn_merge", m, err, part, qg, out, (int32_t)hd, (int32_t)ns, step, (int32_t)ns1, gq_min, xq, sink);
}

// port of cuda/mod.rs: Gpu::attn_decode_gq (decode attention, token t over positions 0 ..= step[2 t]; bias/extent:
// the Inkling2 relative-bias table)
bool CudaBackend::attn_decode_gq(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                                 size_t nh, size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t sink,
                                 size_t win, uint64_t bias, size_t extent, std::string& err) {
    return attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, 0, 0, 0, sink, win, 0, ATT_SPLITS_1, bias,
                extent, err);
}

// port of cuda/mod.rs: Gpu::attn_decode_q (also writes out quantized for gemvq into xq; QSA cell lists through sel)
bool CudaBackend::attn_decode_q(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                                size_t nh, size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t xq,
                                uint64_t sel, size_t sst, uint64_t sink, size_t win, uint64_t& xq_out, std::string& err) {
    xq_out = llama_numerics() ? 0 : xq;
    return attn(q, kc, vc, qg, out, part, step, nh, nkv, hd, scale, gq_min, nt, xq_out, sel, sst, sink, win, 0, ATT_SPLITS_1,
                0, 0, err);
}

// port of cuda/mod.rs: Gpu::attn_prefill_m (flash attention over T query rows; query t sits at position step[0] + t and
// sees the keys 0 ..= step[0] + t, masked by the bits of qm when it is not 0)
bool CudaBackend::attn_prefill_m(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t step, size_t t,
                                 size_t nh, size_t nkv, size_t hd, float scale, uint64_t qm, size_t mw, uint64_t sink,
                                 size_t win, std::string& err) {
    if (hd != 64 && hd != 128 && hd != 256) {
        err = "attention prefill: head_dim " + std::to_string(hd) + " unsupported (64, 128 or 256)";
        return false;
    }
    // Turing keeps 16 keys a tile for head_dim 256 (FA_BK_FOR in kernels.cu).
    const size_t bk = (turing_ && hd == 256) ? 16 : cuda_detail::kFaBk;
    const unsigned smem = (unsigned)cuda_detail::attn_prefill_smem_bk(hd, bk);
    const std::string kn = "attn_prefill_" + std::to_string(hd);
    return launch(kn.c_str(), cfg((unsigned)cuda_detail::div_ceil(t, 64), (unsigned)nh, 128, smem), err, q, kc, vc, qg, out,
                  step, (int32_t)t, (int32_t)nh, (int32_t)nkv, scale, qm, (int32_t)mw, sink, (int32_t)win);
}

// port of cuda/mod.rs: Gpu::router_topk (one warp per token: softmax or sigmoid over n logits, the top k, weights
// renormalized over the k picks, ties to the lower index; bias/eps_add: the Inkling2 router bias and epsilon)
bool CudaBackend::router_topk(uint64_t p, size_t n, size_t k, float extra_scale, uint64_t ids, uint64_t wts, size_t t,
                              bool sigmoid, uint64_t bias, float eps_add, std::string& err) {
    if (n > 512 || k > 32) {
        err = "router_topk: " + std::to_string(n) + " experts, top " + std::to_string(k);
        return false;
    }
    return launch("router_topk", cfg((unsigned)t, 1, 32, 0), err, p, (int32_t)n, (int32_t)k, extra_scale, ids, wts,
                  (int32_t)sigmoid, bias, eps_add);
}

// port of cuda/mod.rs: Gpu::moe_route (decode routing in one block: router_topk, the expert cache lookup, and on a miss
// the routing and h copied to the mailbox mb, whose request flag is then raised; mb 0 skips the mailbox)
bool CudaBackend::moe_route(uint64_t p, size_t n, size_t k, float extra_scale, uint64_t table, uint64_t ids, uint64_t wts,
                            uint64_t slots, uint64_t wh, uint64_t h, uint64_t mb, size_t nh, size_t h_off, uint64_t step,
                            size_t t, uint64_t pid, bool sigmoid, uint64_t bias, float eps_add, std::string& err) {
    if (n > 512 || k > 32) {
        err = "moe_route: " + std::to_string(n) + " experts, top " + std::to_string(k);
        return false;
    }
    return launch("moe_route", cfg(1, 1, 256, 0), err, p, (int32_t)n, (int32_t)k, extra_scale, table, ids, wts, slots, wh, h,
                  mb, (int32_t)nh, (int32_t)h_off, step, (int32_t)t, pid, (int32_t)sigmoid, bias, eps_add);
}

// port of cuda/mod.rs: Gpu::moe_combine (grid.y = token: x += sum_j w ye + sigmoid(sg) sh + the CPU result when a slot
// of the token missed; mb 0: no CPU part; ds/ids: the Inkling2 per-expert down scale)
bool CudaBackend::moe_combine(uint64_t x, uint64_t ye, uint64_t w, size_t k, uint64_t sh, uint64_t sg, uint64_t mb,
                              size_t y_off, uint64_t step, uint64_t slots, size_t n, size_t t, uint64_t ds, uint64_t ids,
                              std::string& err) {
    LaunchCfg c = grid_of(n);
    c.gy = (unsigned)t;
    return launch("moe_combine", c, err, x, ye, w, (int32_t)k, sh, sg, mb, (int32_t)y_off, step, slots, (int32_t)n, ds,
                  ids);
}

// port of cuda/mod.rs: Gpu::moe_combine_seq (prefill: x[t] += sum_j wts[t, j] ye[inv[t, j]] + sigmoid(sg[t]) sh[t])
bool CudaBackend::moe_combine_seq(uint64_t x, uint64_t ye, uint64_t wts, uint64_t inv, size_t k, uint64_t sh, uint64_t sg,
                                  size_t n, size_t t, std::string& err) {
    return launch("moe_combine_seq", grid_of(n * t), err, x, ye, wts, inv, (int32_t)k, sh, sg, (int32_t)n, (int32_t)t);
}

// StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode ops (see backend.h for the math each mirrors).

bool CudaBackend::conv_ink(uint64_t cur, uint64_t state, uint64_t w, uint64_t dyn, uint64_t y, size_t ch, size_t kw,
                           size_t hch, float scale, std::string& err) {
    return launch("conv_ink", cfg((unsigned)cuda_detail::div_ceil(ch, 128), 1, 128, 0), err, cur, state, w, dyn, y,
                  (int32_t)ch, (int32_t)kw, (int32_t)hch, scale);
}

bool CudaBackend::mix_dec(uint64_t bank, size_t nb, uint64_t cur, uint64_t q, size_t nq, uint64_t out, size_t ne,
                          std::string& err) {
    return launch("mix_dec", cfg((unsigned)nq, 1, 256, 0), err, bank, (int32_t)nb, cur, q, out, (int32_t)ne);
}

bool CudaBackend::clip_rows(uint64_t x, uint64_t sc, size_t n, size_t rows, float threshold, float alpha, float rfloor,
                            uint64_t step, std::string& err) {
    return launch("clip_rows", cfg((unsigned)rows, 1, 256, 0), err, x, sc, (int32_t)n, threshold, alpha, rfloor, step);
}

bool CudaBackend::gate_pack(uint64_t cur, uint64_t msg, uint64_t out, size_t ne, float eps, std::string& err) {
    return launch("gate_pack", cfg(1, 1, 256, 0), err, cur, msg, out, (int32_t)ne, eps);
}

bool CudaBackend::control_mix(uint64_t cur, uint64_t msg, uint64_t gain, uint64_t forget, uint64_t out, size_t n,
                              std::string& err) {
    return launch("control_mix", grid_of(n), err, cur, msg, gain, forget, out, (int32_t)n);
}

bool CudaBackend::sig_mul(uint64_t x, uint64_t g, size_t n, std::string& err) {
    return launch("sig_mul", grid_of(n), err, x, g, (int32_t)n);
}

bool CudaBackend::situ_q(uint64_t gu, size_t n, size_t rows, uint64_t ds, uint64_t ids, float beta, float lb, uint64_t xq,
                         uint64_t& xq_out, std::string& err) {
    xq_out = xq;
    return launch("situ_q", cfg((unsigned)cuda_detail::div_ceil(n, 256), (unsigned)rows, 32, 0), err, gu, (int32_t)n, ds,
                  ids, beta, lb, xq);
}

bool CudaBackend::rel_table(uint64_t bias, uint64_t rc, uint64_t sink, uint64_t table, size_t nr, size_t next1, size_t nh,
                            std::string& err) {
    return launch("rel_table", grid_of(next1 * nh), err, bias, rc, sink, table, (int32_t)nr, (int32_t)next1, (int32_t)nh);
}

bool CudaBackend::softcap(uint64_t x, size_t n, float cap, std::string& err) {
    if (cap <= 0.0f) return true;
    return launch("softcap", grid_of(n), err, x, (int32_t)n, cap);
}

}  // namespace strata
