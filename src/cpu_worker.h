// port of cpu_worker.rs: the CPU side of the expert split (the mailbox layout, the worker thread that spins on the
// mailbox flags and computes the missed experts with cpu_moe, and the accessors the GPU side uses).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Deviations forced by C++ (listed in MODIFICATIONS.md): the mailbox memory is a raw host pointer handed in by the
// caller (CudaBackend::host_mailbox in production, a test buffer in the tests); `Result` becomes a bool + an error
// string; rayon's pool becomes an explicit ThreadPool; the mpsc channel becomes a mutex/condvar queue; the flags are
// read and written with volatile accesses between seq_cst fences (see the MODIFICATIONS.md entry for the decision).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "cpu_moe.h"
#include "weights.h"

namespace strata {

// port of cpu_worker.rs: word offsets inside one layer's mailbox. Flags sit on separate cache lines.
constexpr size_t REQ = 0;      // the GPU asks the worker to compute this layer's missed experts
constexpr size_t NTOK = 1;     // tokens in the request (1 for decode, up to TMAX for a verify)
constexpr size_t SKIP = 2;     // raised instead of REQ, with no data, when no expert of the layer missed
constexpr size_t VERDICT = 8;  // Vulkan only in Rust; here the value wait_done resets to 0
constexpr size_t DONE = 16;    // the worker answered
constexpr size_t IDS = 32;     // start of the data area: ids, weights and slots for NTOK tokens

// port of cpu_worker.rs: TMAX, the most tokens one request may carry (1 current + up to 16 draft tokens).
constexpr size_t TMAX = 17;

// port of cpu_worker.rs: Mailbox. Per layer: ids, weights and slots for up to TMAX tokens of `k` entries (token t
// at t * k; the three arrays are each NTOK * k long), then TMAX activations of `n` floats at `h` and TMAX results
// at `y`. All offsets are in 32-bit words.
struct Mailbox {
    size_t k = 0, n = 0, h = 0, y = 0, stride = 0;
    // port of cpu_worker.rs: Mailbox::new
    static Mailbox make(size_t k, size_t n);
};

// STRATA_TRACE_WORKER (measurement only, not in Rust): one row per layer the worker looked at. Times are steady-clock
// ns relative to the worker's epoch; gu/act/down are the pass times of experts() summed over the layer's calls.
struct TraceRow {
    uint32_t seq = 0, il = 0, nt = 0, missed = 0;
    uint32_t kind = 0;                 // 0 computed, 1 skipped (no request for this seq: SKIP or a newer seq)
    int64_t t_top = 0, t_req = 0, t_read = 0, t_exp0 = 0, t_exp1 = 0, t_end = 0, t_done = 0;
    int64_t gu = 0, act = 0, down = 0, tail = 0;   // experts() passes; tail = accumulate after down
    uint64_t pf = 0;                   // process page faults between REQ and DONE (whole process)
    uint64_t user_ns = 0, kern_ns = 0; // this thread's CPU time between REQ and DONE
    uint32_t e0 = 0;                   // first missed expert id of the layer
    uint64_t ws_kb = 0;                // process working set at DONE, KB
};

// port of cpu_worker.rs: CpuWorker. The worker thread runs the loop of the Rust closure on its own thread; the
// expert passes run on a ThreadPool of `threads` workers (rayon's pool in Rust). Dropping the worker stops and
// joins the thread. The Weights and the Gguf behind them must outlive the worker.
class CpuWorker {
public:
    // port of cpu_worker.rs: CpuWorker::new. `host` is the host-mapped mailbox memory (64-byte aligned, at least
    // stride * (max layer + 1) words, `host_bytes` long); `dev` is its device address (0 when there is none).
    // `threads` = 0 uses default_threads().
    static bool create(const Weights& w, uint32_t* host, size_t host_bytes, uint64_t dev,
                       const std::vector<size_t>& moe_layers, size_t threads, std::unique_ptr<CpuWorker>& out,
                       std::string& err);
    // port of cpu_worker.rs: impl Drop for CpuWorker (stop flag, close the queue, join).
    ~CpuWorker();
    CpuWorker(const CpuWorker&) = delete;
    CpuWorker& operator=(const CpuWorker&) = delete;

    // Device address of the mailboxes (layer `il` at dev + 4 * il * layout.stride).
    uint64_t dev() const { return dev_; }
    Mailbox layout;

    // port of cpu_worker.rs: CpuWorker::begin. Starts a new token: returns the sequence number the GPU must
    // publish with.
    uint32_t begin();
    // port of cpu_worker.rs: CpuWorker::routing. Routing of layer `il` from the last finished request, all its
    // tokens: (expert id, routed weight, slot or -1). Empty when `il` is not a worker layer.
    std::vector<std::tuple<int32_t, float, int32_t>> routing(size_t il) const;
    // port of cpu_worker.rs: CpuWorker::stalled_layer. The first layer whose verdict is 2 (none in the CUDA path).
    std::optional<size_t> stalled_layer() const;
    // port of cpu_worker.rs: CpuWorker::wait_done. Waits until the worker has answered layer `il` of the current
    // request, then resets the layer's verdict. `timeout_ms` < 0 waits as long as Rust does; otherwise it gives up
    // with `err` set. Returns false when the worker stopped, the layer is not a worker layer, or the wait timed out.
    bool wait_done(size_t il, int64_t timeout_ms, std::string& err);
    // port of cpu_worker.rs: CpuWorker::busy. Total nanoseconds the worker spent computing experts.
    uint64_t busy_ns() const { return busy_ns_.load(std::memory_order_relaxed); }
    // Missed experts the worker has computed (over all layers and requests).
    uint64_t experts_done() const { return experts_done_.load(std::memory_order_relaxed); }
    // True when `il` is one of the worker's layers.
    bool has_layer(size_t il) const;
    // Sets the stop flag, closes the queue and joins the worker thread (the Rust Drop). Idempotent.
    void stop();

private:
    CpuWorker() = default;
    void run();
    void thread_main();

    uint32_t* host_ = nullptr;
    size_t host_bytes_ = 0;
    uint64_t dev_ = 0;
    std::vector<size_t> layers_;
    std::vector<ExpertLayer> experts_;   // experts_[i] belongs to layers_[i]
    std::unique_ptr<ThreadPool> pool_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<uint32_t> queue_;   // the mpsc channel of Rust: sequence numbers in send order
    bool closed_ = false;          // the sender was dropped
    uint32_t seq_ = 0;             // the sequence number of the current token (main thread only)

    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> busy_ns_{0};
    std::atomic<uint64_t> experts_done_{0};
    std::atomic<bool> dead_{false};   // the thread has exited
    std::string dead_err_;            // why it exited on an error (guarded by mu_)
    std::optional<std::chrono::milliseconds> delay_;   // STRATA_TEST_WORKER_DELAY

    // STRATA_TRACE_WORKER: rows are appended by the worker thread and written by stop() after the join.
    bool trace_ = false;
    std::string trace_path_;
    int64_t trace_epoch_ = 0;
    std::vector<TraceRow> trace_rows_;
    void dump_trace() const;

    std::thread thread_;
};

}  // namespace strata
