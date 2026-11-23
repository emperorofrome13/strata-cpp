// port of cpu_moe.rs (CPU routed-expert path: Q8_K activations, integer dot products per weight type, the
// weighted sum of the routed experts) and of the thread pool that rayon provides to it.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md). The integer kernels are llama.cpp's ggml-quants.c / arch/x86/quants.c
// arithmetic (MIT License, ggml authors), reached through RustingStrata 1.2.5.
//
// Deviations forced by C++ (listed in MODIFICATIONS.md): rayon's global pool becomes an explicit
// ThreadPool passed to the functions; `Result` becomes a bool plus an error string; LoRA deltas are out of
// scope (STRATA_CPP_PLAN.md section 4), so the `m.lora_delta` calls of cpu_moe.rs are absent.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "quant.h"
#include "weights.h"

namespace strata {

// port of rayon's thread pool (ThreadPoolBuilder::num_threads(n) and par_iter().for_each): N persistent
// worker threads that sleep between jobs. The calling thread waits, as it does in rayon when it is not a
// pool thread. One caller at a time (the expert paths are driven from one thread).
class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t size() const { return th_.size(); }
    // STRATA_TRACE_WORKER only: per-job counters (not a port). Start latency = time from the job's publication to a
    // pool thread's first item; busy = time pool threads spend in items; the caller's total and wait per job.
    struct Stats {
        uint64_t jobs = 0, start_lat_ns = 0, busy_ns = 0, items = 0, caller_total_ns = 0, caller_wait_ns = 0,
                 caller_items = 0, caller_busy_ns = 0, caller_calls = 0;
    };
    void set_trace(bool on) { trace_ = on; }
    Stats stats() const;
    // port of rayon's `(0..n).into_par_iter().for_each(f)`: f(i) runs once for each i on some worker;
    // returns when all of them have finished.
    void for_each(size_t n, const std::function<void(size_t)>& f);

private:
    void worker();

    std::vector<std::thread> th_;
    std::mutex mu_;                  // guards the sleep of idle pool threads (generation changes take it too)
    std::condition_variable work_cv_;
    const std::function<void(size_t)>* job_ = nullptr;  // written by for_each before the generation is published
    size_t n_ = 0;
    std::atomic<size_t> next_{0};    // next item to claim (pool threads and the caller)
    std::atomic<size_t> finished_{0};
    std::atomic<size_t> acked_{0};   // pool threads that have left the current job
    std::atomic<uint64_t> gen_{0};
    std::atomic<bool> stop_{false};
    std::atomic<size_t> sleepers_{0};
    bool trace_ = false;
    std::atomic<int64_t> pub_ns_{0};
    std::atomic<uint64_t> s_jobs_{0}, s_lat_{0}, s_busy_{0}, s_items_{0}, c_total_{0}, c_wait_{0}, c_items_{0}, c_busy_{0},
        c_calls_{0};
};

// port of main.rs: physical_cores. Distinct physical cores from the processor topology; 0 when unknown.
size_t physical_cores();
// port of main.rs: the thread count used when --threads is not given. Strata's Windows branch of
// physical_cores returns None (rayon then uses every logical core); this port follows the plan and takes
// the physical core count. The difference is recorded in MODIFICATIONS.md.
size_t default_threads();

// port of cpu_moe.rs: Q8K. `bs` holds the sums of the 16 groups of 16 `qs`, for the min terms.
struct Q8K {
    float d;
    int8_t qs[256];
    int16_t bs[16];
};

// port of cpu_moe.rs: quantize_q8k. Quantizes `n` values in blocks of 256; a short last block is zero-padded.
std::vector<Q8K> quantize_q8k(const float* x, size_t n);

// port of cpu_moe.rs: int_dot, as the public function deepseek/mat.rs calls (`crate::cpu_moe::int_dot(ty)`): the
// integer dot kernel of `ty` for one row against Q8K blocks, or nullptr for a type without one. `n` is the row length
// in bytes, `nx` the number of Q8K blocks. Reads STRATA_SCALAR on each call, as the Rust function does.
using IntDotFn = float (*)(const uint8_t* row, size_t n, const Q8K* xq, size_t nx);
IntDotFn int_dot_kernel(Ty ty);

// port of model.rs: the routed-expert tensors of one layer, as `Model::w.layer(il, ...)` resolves them
// (`blk.<il>.ffn_gate_exps.weight`, `ffn_up_exps`, `ffn_down_exps`). The tensors point into the mapping of
// the Gguf the Weights were built from; that Gguf and Weights must outlive this struct.
struct ExpertLayer {
    Tensor gate;   // [n_embd, n_ff_exp, n_expert]: cols = n_embd, rows = n_ff_exp * n_expert
    Tensor up;     // same shape as gate
    Tensor down;   // [n_ff_exp, n_embd, n_expert]: cols = n_ff_exp, rows = n_embd * n_expert
    // StrataFreet addition (INKLING2_NOTES.md): inkling2 fuses gate and up in one [n_embd, 2*n_ff_exp,
    // n_expert] tensor (gate rows 0..n_ff-1, up rows n_ff..2*n_ff-1). `gate_up` is set, `gate`/`up` are
    // then per-expert views handed out by experts() itself; the per-expert f32 scales (one float per
    // expert per matrix, when the file has them) multiply the experts() output.
    Tensor gate_up = {};
    Tensor gate_scale = {};
    Tensor down_scale = {};
    bool fused_gu = false;
    bool has_scales = false;
    size_t n_embd = 0;
    size_t n_expert = 0;
    size_t n_ff = 0;   // n_ff_exp
    // StrataFreet addition (W2): the Situ activation betas of inkling2 (0 = the SwiGLU of the qwen families).
    float situ_beta = 0.0f;
    float situ_linear_beta = 0.0f;
};

// port of model.rs: the expert part of Model's weight map (Weights::layer for the three expert tensors),
// with the shape checks the Rust code leaves to its indexing.
bool load_expert_layer(const Weights& w, const ModelConfig& cfg, size_t il, ExpertLayer& out, std::string& err);

// port of cpu_moe.rs: experts. y = sum_j w_j * down_{e_j}(silu(gate_{e_j} h) * up_{e_j} h), over `sel` (expert,
// routing weight) in router order. `h` has n_embd values, `y` receives n_embd values. No router, no shared expert.
bool experts(const ExpertLayer& L, const float* h, const std::vector<std::pair<size_t, float>>& sel, float* y,
             ThreadPool& pool, std::string& err);

// STRATA_TRACE_WORKER only (not a port: a measurement hook). When g_experts_trace is set on the calling thread, experts()
// stores the steady-clock ns of its pass boundaries: t[0] start, t[1] after gate/up, t[2] after silu, t[3] after down,
// t[4] end. The pointer is null otherwise, so the cost when tracing is off is one thread-local load per call.
struct ExpertsTrace {
    int64_t t[5] = {0, 0, 0, 0, 0};
};
extern thread_local ExpertsTrace* g_experts_trace;

// port of cpu_moe.rs: experts_multi. The same sums for several tokens: each routed expert reads its weights once
// and applies them to every token that routed to it. Output row t equals experts() for token t (same kernels,
// same summation order), so the results are bit-identical.
bool experts_multi(const ExpertLayer& L, const std::vector<std::vector<float>>& hs,
                   const std::vector<std::vector<std::pair<size_t, float>>>& sels,
                   std::vector<std::vector<float>>& ys, ThreadPool& pool, std::string& err);

// port of cpu_moe.rs: int_dots_match_dequant (the Rust unit test) for every integer kernel, the scalar and the
// AVX2 one, against the dequantized dot product. Returns false and the first mismatch in `err`.
bool int_dots_selftest(std::string& err);

// StrataFreet test hook (not in cpu_moe.rs): y[r] = row r . x through the integer kernel of `ty` (scalar when
// `scalar`, else AVX2), for `rows` rows of `rb` bytes, x of `n` values quantized as Q8_K. False with `err` when the
// type has no integer kernel.
bool int_dot_rows(Ty ty, bool scalar, const uint8_t* w, size_t rows, size_t rb, const float* x, size_t n, float* y,
                  std::string& err);

}  // namespace strata
