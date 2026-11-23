// port of cuda/mod.rs: the CUDA backend (primary context, streams, device buffers, host-mapped mailbox memory,
// the NVRTC module built from kernels.cu with Strata's prepended macros and tables, and the launch wrappers).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Part 1 of W2-C (cuda_backend.cpp) implements: context and streams, alloc/htod/dtoh/dtod/memset/mem_info/sync,
// batch, host_mailbox, the cubin cache, dyn_shmem_need, and the gemv / gemvn / gemvq / gemvq2 / moe_gemv / dequant /
// quant_xq launch wrappers for all quant types. Part 2a (cuda_backend_ops.cpp) implements gemm, gemm_dense, the
// norms, argmax, rope, add, swiglu, norm_gate, kv_store, the decode and prefill attention, and the MoE routing and
// combine launches. Part 2b (cuda_backend_seq.cpp) implements the DeltaNet / conv ops, the QSA indexer, the
// hyper-connection and PLE ops, the per-row KV kernels, moe_prefetch, the CUDA graph capture of `graph` and the
// side-stream `beside` / `beside_join`.
#pragma once

#include <cuda.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend.h"

namespace strata {

// port of cuda/mod.rs: LaunchConfig (grid, block, dynamic shared memory) as cuda/mod.rs `cfg` builds it.
struct LaunchCfg {
    unsigned gx = 1, gy = 1, gz = 1;
    unsigned bx = 1, by = 1, bz = 1;
    unsigned smem = 0;
};

// port of cuda/mod.rs: cfg(grid, block, smem) (a 2-D grid and a 1-D block)
LaunchCfg cfg(unsigned gx, unsigned gy, unsigned block, unsigned smem);

// Helpers shared by cuda_backend.cpp (which defines them over its own anonymous-namespace versions) and
// cuda_backend_ops.cpp.
namespace cuda_detail {
// port of cuda/mod.rs: FA_BQ, FA_BK and ATTN_SMEM_PAD (attention prefill tile sizes)
constexpr size_t kFaBq = 64;
constexpr size_t kFaBk = 32;
constexpr size_t kAttnSmemPad = 8;
// port of cuda/mod.rs: AMAX_PARTS (argmax partials per row)
constexpr size_t kAmaxParts = 64;
// port of cuda/mod.rs: kname (prefix + "_" + the type name lower-cased)
std::string kname(const std::string& prefix, Ty ty);
// port of cuda/mod.rs: div_ceil
size_t div_ceil(size_t a, size_t b);
// port of cuda/mod.rs: Gpu::float_ty (F32 and BF16 read x as float)
bool float_ty(Ty ty);
// port of cuda/mod.rs: attn_prefill_smem_bk (shared memory of one attn_prefill launch at a K/V depth of bk)
size_t attn_prefill_smem_bk(size_t hd, size_t bk);
// The driver error text of a failed call `what` (cuda/mod.rs: the cudarc error context).
std::string cu_error(const char* what, CUresult r);
}  // namespace cuda_detail

class CudaBackend final : public Backend {
public:
    // port of cuda/mod.rs: Gpu::new. Returns null and sets `err` when the device, the NVRTC compile or the
    // module load fails. Device memory returned by this backend must be dropped before the backend.
    // `kv_q8` (StrataFreet addition): the session asked for the int8 KV cache ("kv": "q8" in sc_load); the
    // STRATA_KV_Q8 environment switch still enables it for the tooling.
    static std::unique_ptr<CudaBackend> create(int device, std::string& err, bool kv_q8 = false);
    ~CudaBackend() override;

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    const char* name() const override { return "cuda"; }
    std::string device() const override { return name_; }
    bool alloc(size_t bytes, Buf& out, std::string& err) override;
    // Reserves one device block of `bytes` (rounded up to 2 MB) that `alloc` carves its buffers from, 256-byte aligned,
    // until it is full; a buffer that does not fit is allocated on its own. The driver rounds each cuMemAlloc up, and
    // the ~700 tensors and buffers of a model load would take ~150 MB more VRAM than Strata's exact accounting (the
    // plan's "weights+state+scratch" and the expert slots that follow it). Carved buffers live until the backend goes.
    bool reserve_arena(size_t bytes, std::string& err) override;
    bool htod(const void* src, size_t n, const Buf& dst, size_t off, std::string& err) override;
    // StrataFreet addition (sc_load "pin"): cuMemHostRegister of a host range (the RAM copy of the weights). An htod
    // whose source lies wholly in a registered range is a direct DMA from it, with no copy through the pinned ring.
    // Registration happens once, before the first copy; unregister_host runs in the destructor.
    bool register_host(const void* p, size_t n, std::string& err);
    void unregister_host();
    bool host_registered(const void* p, size_t n) const;
    // Bytes copied straight from registered ranges (sc_stats "ram_direct_bytes").
    uint64_t host_direct_bytes() const { return host_direct_bytes_.load(); }
    bool dtoh(const Buf& src, size_t off, void* dst, size_t n, std::string& err) override;
    bool dtod(const Buf& src, size_t src_off, const Buf& dst, size_t dst_off, size_t bytes, std::string& err) override;
    bool memset(const Buf& dst, std::string& err) override;
    bool mem_info(size_t& free_bytes, size_t& total_bytes, std::string& err) override;
    bool sync(std::string& err) override;
    bool graph(uint64_t key, const OpFn& f, std::string& err) override;
    bool batch(const OpFn& f, std::string& err) override;
    bool mmq(Ty ty) const override { return i8_gemm_ && cuda_mmq_policy(ty); }
    bool volta() const override { return !i8_gemm_; }
    bool host_mailbox(size_t words, Buf& buf, uint32_t*& host, std::string& err) override;
    size_t kv_words(size_t n) const override { return kv_q8_ ? n * 17 / 32 : n; }
    bool int8_gemm(Ty ty) const override { return int8_mma_ && cuda_mmq_policy(ty); }
    bool moe_mailbox_optional() const override { return true; }

    bool gemv(const GpuTensor& w, uint64_t x, uint64_t y, std::string& err) override;
    bool gemvn(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t y, size_t y_stride, size_t nt, std::string& err) override;
    bool gemvq(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t xq, uint64_t y, size_t y_stride, size_t nt,
               std::string& err) override;
    bool gemvq2(const GpuTensor& a, const GpuTensor& b, uint64_t x, size_t x_stride, uint64_t xq, uint64_t ya, uint64_t yb,
                size_t nt, std::string& err) override;
    bool moe_gemv(Ty ty, size_t rows, size_t cols, uint64_t w, size_t stride, uint64_t ids, size_t k, uint64_t x,
                  size_t x_stride, size_t xg, uint64_t y, std::optional<std::pair<int64_t, int64_t>> pair, uint64_t xq,
                  std::string& err) override;
    bool dequant(const GpuTensor& w, size_t rows, uint64_t out, std::string& err) override;
    bool quant_xq(uint64_t x, size_t x_stride, size_t n, size_t nt, uint64_t out, std::string& err) override;

    // Part 2a of W2-C (cuda_backend_ops.cpp).
    bool gemm(Ty ty, size_t rows, size_t cols, uint64_t tiles, size_t n_tiles, uint64_t x, size_t x_stride, size_t n_x,
              uint64_t xrow, uint64_t y, size_t y_stride, int32_t act, std::string& err) override;
    bool gemm_dense(const GpuTensor& w, size_t n, uint64_t x, size_t x_stride, uint64_t y, size_t y_stride, int32_t act,
                    std::string& err) override;
    bool prequant_gemm_x(const GpuTensor& w, size_t n, uint64_t x, size_t x_stride, std::string& err) override;
    bool rms_norm_streams(uint64_t x, uint64_t w, uint64_t out, size_t n, size_t hc, size_t nt, float eps,
                          std::string& err) override;
    bool rms_norm_rope(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t n, size_t rows, float eps, size_t per,
                       size_t n_rot, uint64_t step, float base, std::string& err) override;
    bool rms_norm_rope_qk(uint64_t q, size_t q_stride, uint64_t qw, uint64_t qo, size_t qrows, size_t qper, uint64_t k,
                          uint64_t kw, uint64_t ko, size_t krows, size_t kper, size_t n, float eps, size_t n_rot, uint64_t step,
                          float base, uint64_t v, uint64_t kc, uint64_t vc, std::string& err) override;
    bool swiglu_q(uint64_t g, uint64_t u, size_t n, size_t rows, uint64_t xq, uint64_t& xq_out, std::string& err) override;
    bool norm_gate_q(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, uint64_t xq, uint64_t& xq_out,
                     std::string& err) override;
    bool norm_gate_sig(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, uint64_t xq, uint64_t& xq_out,
                       std::string& err) override;
    bool attn_decode_q(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                       size_t nh, size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t xq, uint64_t sel,
                       size_t sst, uint64_t sink, size_t win, uint64_t& xq_out, std::string& err) override;

    // Part 2a: the norms, argmax, rope, elementwise and attention launches of cuda_backend_ops.cpp.
    bool rms_norm(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t out_stride, size_t n, size_t rows, float eps,
                  std::string& err) override;
    bool rms_norm_q(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t out_stride, size_t n, size_t rows,
                    float eps, uint64_t xq, uint64_t res, std::string& err) override;
    bool argmax_rows(uint64_t x, size_t n, size_t rows, uint64_t out, std::string& err) override;
    bool l2_norm(uint64_t x, size_t n, size_t per, size_t stride, size_t rows, float eps, std::string& err) override;
    bool rope_pairs(uint64_t x, size_t stride, size_t heads, size_t per, size_t n_rot, uint64_t step, float base, size_t lo,
                    size_t hi, std::string& err) override;
    bool rms_norm_rope_kv(uint64_t x, size_t x_stride, uint64_t w, uint64_t out, size_t n, size_t rows, float eps, size_t per,
                          size_t n_rot, uint64_t step, float base, uint64_t v, uint64_t kc, uint64_t vc, std::string& err) override;
    bool add(uint64_t x, uint64_t y, size_t n, std::string& err) override;
    bool swiglu(uint64_t g, uint64_t u, size_t n, std::string& err) override;
    // Part 2b (cuda_backend_seq.cpp).
    bool ssm_gates(uint64_t beta, uint64_t g, uint64_t dt_bias, uint64_t a, size_t n, size_t nv, std::string& err) override;
    bool conv_silu(uint64_t qkv, uint64_t state, uint64_t w, uint64_t y, size_t ch, size_t kw, size_t nn, float eps,
                   std::string& err) override;
    bool deltanet(uint64_t y, uint64_t s, uint64_t beta, uint64_t g, uint64_t dt_bias, uint64_t a, uint64_t o, size_t nk,
                  size_t nv, bool interleave, size_t dk, size_t dv, float scale, size_t t, size_t ych, uint64_t snap,
                  size_t nsnap, std::string& err) override;
    bool conv_seq(uint64_t qkv, uint64_t state, uint64_t w, uint64_t y, size_t ch, size_t kw, size_t t, uint64_t snap,
                  size_t nsnap, std::string& err) override;
    bool attn_prefill_m(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t step, size_t t, size_t nh,
                        size_t nkv, size_t hd, float scale, uint64_t qm, size_t mw, uint64_t sink, size_t win,
                        std::string& err) override;
    bool moe_combine_seq(uint64_t x, uint64_t ye, uint64_t wts, uint64_t inv, size_t k, uint64_t sh, uint64_t sg, size_t n,
                         size_t t, std::string& err) override;
    bool norm_gate(uint64_t o, uint64_t w, uint64_t z, size_t n, size_t heads, float eps, std::string& err) override;
    bool kv_store(uint64_t k, uint64_t v, uint64_t kc, uint64_t vc, size_t n, size_t t, uint64_t step, size_t hd,
                  std::string& err) override;
    bool attn_decode_gq(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step,
                        size_t nh, size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t sink, size_t win,
                        uint64_t bias, size_t extent, std::string& err) override;
    bool router_topk(uint64_t p, size_t n, size_t k, float extra_scale, uint64_t ids, uint64_t wts, size_t t, bool sigmoid,
                     uint64_t bias, float eps_add, std::string& err) override;
    bool moe_route(uint64_t p, size_t n, size_t k, float extra_scale, uint64_t table, uint64_t ids, uint64_t wts, uint64_t slots,
                   uint64_t wh, uint64_t h, uint64_t mb, size_t nh, size_t h_off, uint64_t step, size_t t, uint64_t pid,
                   bool sigmoid, uint64_t bias, float eps_add, std::string& err) override;
    bool moe_combine(uint64_t x, uint64_t ye, uint64_t w, size_t k, uint64_t sh, uint64_t sg, uint64_t mb, size_t y_off,
                     uint64_t step, uint64_t slots, size_t n, size_t t, uint64_t ds, uint64_t ids,
                     std::string& err) override;

    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode ops (cuda_backend_ops.cpp).
    bool conv_ink(uint64_t cur, uint64_t state, uint64_t w, uint64_t dyn, uint64_t y, size_t ch, size_t kw, size_t hch,
                  float scale, std::string& err) override;
    bool mix_dec(uint64_t bank, size_t nb, uint64_t cur, uint64_t q, size_t nq, uint64_t out, size_t ne,
                 std::string& err) override;
    bool clip_rows(uint64_t x, uint64_t sc, size_t n, size_t rows, float threshold, float alpha, float rfloor,
                   uint64_t step, std::string& err) override;
    bool gate_pack(uint64_t cur, uint64_t msg, uint64_t out, size_t ne, float eps, std::string& err) override;
    bool control_mix(uint64_t cur, uint64_t msg, uint64_t gain, uint64_t forget, uint64_t out, size_t n,
                     std::string& err) override;
    bool sig_mul(uint64_t x, uint64_t g, size_t n, std::string& err) override;
    bool situ_q(uint64_t gu, size_t n, size_t rows, uint64_t ds, uint64_t ids, float beta, float lb, uint64_t xq,
                uint64_t& xq_out, std::string& err) override;
    bool rel_table(uint64_t bias, uint64_t rc, uint64_t sink, uint64_t table, size_t nr, size_t next1, size_t nh,
                   std::string& err) override;
    bool softcap(uint64_t x, size_t n, float cap, std::string& err) override;

    // Part 2b (cuda_backend_seq.cpp): graph capture, the side stream, the per-row KV kernels, QSA, hyper-connections,
    // PLE and moe_prefetch.
    bool beside(const OpFn& f, std::string& err) override;
    bool beside_join(std::string& err) override;
    bool beside_overlaps() const override { return true; }
    bool kv_rows() const override { return true; }
    bool rms_norm_rope_qk_rows(uint64_t q, size_t q_stride, uint64_t qw, uint64_t qo, size_t nh, uint64_t k, uint64_t kw,
                               uint64_t ko, size_t nkv, size_t hd, float eps, size_t n_rot, uint64_t step, float base,
                               uint64_t v, uint64_t kvt, size_t nt, std::string& err) override;
    bool attn_rows(uint64_t q, uint64_t qg, uint64_t out, uint64_t part, uint64_t step, size_t nh, size_t nkv, size_t hd,
                   float scale, size_t nt, uint64_t sink, size_t win, uint64_t kvt, size_t splits, std::string& err) override;
    bool qsa_pool(uint64_t ik, uint64_t ring, uint64_t pool, uint64_t w, uint64_t step, size_t t, size_t n_rot, float base,
                  float eps, std::string& err) override;
    bool qsa_select(uint64_t q, uint64_t pool, uint64_t score, uint64_t sel, uint64_t mask, uint64_t step, size_t r0,
                    size_t rows, size_t nbs, size_t top, size_t sst, size_t mw, std::string& err) override;
    bool hc_silu(uint64_t x, float sc, size_t n, std::string& err) override;
    bool hc_mix(uint64_t xn, uint64_t g, uint64_t out, size_t n, size_t hc, size_t nt, std::string& err) override;
    bool hc_write(uint64_t r, uint64_t out, uint64_t inj, size_t n, size_t hc, size_t nt, bool clear, std::string& err) override;
    bool ple_gate(uint64_t key, uint64_t query, uint64_t value, uint64_t gated, size_t n, size_t hc, size_t nt,
                  std::string& err) override;
    bool ple_conv(uint64_t r, uint64_t gated, uint64_t norm, uint64_t kw, uint64_t ring, uint64_t step, size_t d, size_t nt,
                  std::string& err) override;
    bool moe_prefetch_supported() const override { return true; }
    bool moe_prefetch(uint64_t ids, size_t k, uint64_t table, uint64_t base, size_t stride, size_t bytes,
                      std::string& err) override;

private:
    CudaBackend() = default;

    // The block reserve_arena made (0: none) and how much of it alloc has carved.
    CUdeviceptr arena_mem_ = 0;
    size_t arena_size_ = 0;
    size_t arena_used_ = 0;

    // A region of the pinned staging ring that a queued copy still reads, and the event that marks its end.
    struct RingBusy {
        size_t a, b;
        CUevent ev;
    };

    bool bind(std::string& err);
    bool wait(std::string& err);
    CUstream st();
    bool launch_raw(const char* kname, const LaunchCfg& c, void** args, std::string& err);
    bool join(CUstream waiter, CUstream waited, std::string& err);
    // port of cuda/mod.rs: Gpu::gemvq_pair (z = 2 also runs the second matrix at `pair`).
    struct Pair {
        int64_t w2 = 0;
        int64_t y2 = 0;
        int32_t rows2 = 0;
    };
    bool gemvq_pair(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t xq, uint64_t y, size_t y_stride, size_t nt,
                    Pair pair, unsigned z, std::string& err);

    // port of cuda/mod.rs: Gpu::gemm_raw (tile lists or a dense matrix; the int8 path when the arch and type allow it)
    bool gemm_raw(Ty ty, size_t rows, size_t cols, uint64_t tiles, size_t n_tiles, uint64_t wd, size_t nd, uint64_t x,
                  size_t x_stride, size_t n_x, uint64_t xrow, uint64_t y, size_t y_stride, int32_t act, std::string& err);
    // port of cuda/mod.rs: Gpu::q8_x (quantizes x for the int8 GEMM unless the cached copy is the same rows; `xq`
    // is empty when the f16 path is taken). The caller holds q8_mu_.
    bool q8_x(Ty ty, size_t cols, size_t n_x, uint64_t x, size_t x_stride, std::optional<uint64_t>& xq, std::string& err);
    // port of cuda/mod.rs: Gpu::rope_fr (the rope frequencies of base and n_rot, made once by rope_freq)
    bool rope_fr(float base, size_t n_rot, uint64_t& out, std::string& err);
    // port of cuda/mod.rs: Gpu::attn (attn_decode_<hd> when gq_min != 0, attn_decode_gqa* when gq_min != i32::MAX,
    // then attn_merge). `kvt` != 0 selects the per-row caches (ns1 splits). `bias` != 0: the Inkling2 relative-bias
    // table the decode kernels gather (see Backend::attn_decode_gq).
    bool attn(uint64_t q, uint64_t kc, uint64_t vc, uint64_t qg, uint64_t out, uint64_t part, uint64_t step, size_t nh,
              size_t nkv, size_t hd, float scale, int32_t gq_min, size_t nt, uint64_t xq, uint64_t sel, size_t sst,
              uint64_t sink, size_t win, uint64_t kvt, size_t ns1, uint64_t bias, size_t extent, std::string& err);

    template <class... A>
    bool launch(const char* kname, const LaunchCfg& c, std::string& err, A... args) {
        void* ptrs[sizeof...(A)] = {static_cast<void*>(&args)...};
        return launch_raw(kname, c, ptrs, err);
    }

    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
    std::string name_;
    int cc_major_ = 0;
    int cc_minor_ = 0;
    // The int8 prefill GEMM is compiled in and its tensor cores exist: sm_80+ (`int8_gemm_arch`).
    bool i8_gemm_ = false;
    // int8 tensor-core mma: sm_80+.
    bool int8_mma_ = false;
    // The kernels were compiled for a q8 KV cache (STRATA_KV_Q8).
    bool kv_q8_ = false;
    // The device's opt-in shared memory per block.
    size_t max_smem_ = 0;
    CUstream stream_ = nullptr;
    // Takes every second launch inside `batch`, so a captured graph runs them as parallel branches.
    CUstream side_ = nullptr;
    // Launches so far in the current `batch` (SIZE_MAX outside one).
    std::atomic<size_t> nb_{SIZE_MAX};
    CUmodule module_ = nullptr;
    std::map<std::string, CUfunction> fns_;
    // Pinned staging for large htod (a ring) and its busy regions.
    void* ring_ = nullptr;
    size_t ring_head_ = 0;
    std::deque<RingBusy> busy_;
    std::mutex ring_mu_;
    // Recorded by `wait`.
    CUevent done_ = nullptr;
    // port of cuda/mod.rs: Gpu::q8 (the int8 copy of x and the key of the rows it holds), under q8_mu_.
    std::mutex q8_mu_;
    Buf q8_;
    std::optional<std::array<uint64_t, 4>> q8_key_;
    // port of cuda/mod.rs: Gpu::am (argmax partials, kAmaxParts int2 per row)
    Buf am_;
    // port of cuda/mod.rs: Gpu::fr (rope frequency tables by (base bits, n_rot)), under fr_mu_
    std::mutex fr_mu_;
    std::map<std::pair<uint32_t, size_t>, Buf> fr_;
    // port of cuda/mod.rs: Gpu::fr.1 (rope tables a graph capture asked for; made after the capture), under fr_mu_
    std::vector<std::pair<float, size_t>> fr_want_;
    // port of cuda/mod.rs: Gpu::graphs (instantiated graphs by key), under graph_mu_
    std::mutex graph_mu_;
    std::map<uint64_t, CUgraphExec> graphs_;
    // port of cuda/mod.rs: Gpu::lag (the side stream's event after the last `beside`), null when none is pending
    CUevent lag_ = nullptr;
    // port of cuda/mod.rs: Gpu::turing (sm_75 to sm_79: no int8 tensor cores, 64 KB shared memory)
    bool turing_ = false;
    // StrataFreet addition: host ranges registered by register_host (ptr, bytes); the copy counter of host_direct_bytes.
    std::vector<std::pair<const char*, size_t>> host_regs_;
    std::atomic<uint64_t> host_direct_bytes_{0};
};

}  // namespace strata
