// port of cpu_worker.rs: the CPU side of the expert split. See cpu_worker.h.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md). Each function names the Rust function it ports.
#include "cpu_worker.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace strata {

namespace {

// STRATA_TRACE_WORKER helpers (measurement only). steady_clock is QueryPerformanceCounter on MSVC.
inline int64_t tick_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// process page-fault count; `ws` receives the working set in KB
inline uint64_t page_faults(uint64_t* ws = nullptr) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof pmc;
    K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    if (ws) *ws = pmc.WorkingSetSize >> 10;
    return pmc.PageFaultCount;
#else
    if (ws) *ws = 0;
    return 0;
#endif
}
// user and kernel CPU time of the calling thread, in ns (out params)
inline void thread_times(uint64_t& user, uint64_t& kern) {
#ifdef _WIN32
    FILETIME c, e, k, u;
    GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u);
    auto ft = [](const FILETIME& f) { return ((uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime) * 100ull; };
    user = ft(u);
    kern = ft(k);
#else
    user = kern = 0;
#endif
}

// Rust's `next_multiple_of(16)` and `next_multiple_of(4)` for usize.
inline size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

// port of cpu_worker.rs: the read_volatile / write_volatile of the flags and the mailbox words. The fences around
// them are seq_cst, as in Rust (see MODIFICATIONS.md).
inline uint32_t vread(const uint32_t* p) { return *static_cast<const volatile uint32_t*>(p); }
inline void vwrite(uint32_t* p, uint32_t v) { *static_cast<volatile uint32_t*>(p) = v; }

inline float bits_to_f(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof f);
    return f;
}

// port of cpu_worker.rs: the `ahead` closure. A flag is ahead of `seq` when it is set and not older than seq
// (wrapping comparison).
inline bool ahead(uint32_t v, uint32_t seq) { return v != 0 && static_cast<int32_t>(v - seq) >= 0; }

// port of cpu_worker.rs: read_wc. Copies `n` words (a multiple of 4) from 16-byte aligned `p`. On write-combining
// memory streaming loads fetch a whole line per read instead of one uncached read per load.
inline void read_wc(const uint32_t* p, uint32_t* out, size_t n) {
#if defined(__AVX2__) || defined(__SSE4_1__)
    for (size_t i = 0; i + 4 <= n; i += 4) {
        const __m128i x = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(p + i)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), x);
    }
#else
    std::memcpy(out, p, n * sizeof(uint32_t));
#endif
}

}  // namespace

// port of cpu_worker.rs: Mailbox::new
Mailbox Mailbox::make(size_t k, size_t n) {
    Mailbox m;
    m.k = k;
    m.n = n;
    m.h = round_up(IDS + 3 * TMAX * k, 16);
    m.y = m.h + TMAX * n;
    m.stride = round_up(m.h + 2 * TMAX * n, 16);
    return m;
}

// port of cpu_worker.rs: CpuWorker::new
bool CpuWorker::create(const Weights& w, uint32_t* host, size_t host_bytes, uint64_t dev,
                       const std::vector<size_t>& moe_layers, size_t threads, std::unique_ptr<CpuWorker>& out,
                       std::string& err) {
    const ModelConfig& c = w.config();
    std::unique_ptr<CpuWorker> cw(new CpuWorker());
    cw->layout = Mailbox::make(c.n_expert_used, c.n_embd);
    // read_wc copies whole 4-word chunks; Rust would leave a tail unread. Every row is a multiple of 4 words.
    if (c.n_embd % 4 != 0) {
        err = "cpu worker: n_embd must be a multiple of 4";
        return false;
    }
    size_t max_il = 0;
    for (size_t il : moe_layers) max_il = std::max(max_il, il);
    const size_t len_bytes = moe_layers.empty() ? 0 : cw->layout.stride * (max_il + 1) * sizeof(uint32_t);
    if (host_bytes < len_bytes) {
        err = "cpu worker: mailbox memory is smaller than the layers need";
        return false;
    }
    if (len_bytes > 0 && (reinterpret_cast<uintptr_t>(host) % 64) != 0) {
        err = "cpu worker: mailbox memory must be 64-byte aligned";
        return false;
    }
    try {
        // port of cpu_worker.rs: `m.w.layer(il, ..)` in the loop, resolved once here so that a bad layer fails at
        // creation (the Rust code would fail on the first request that reaches it).
        for (size_t il : moe_layers) {
            ExpertLayer L;
            if (!load_expert_layer(w, c, il, L, err)) return false;
            cw->experts_.push_back(L);
        }
        cw->host_ = host;
        cw->host_bytes_ = host_bytes;
        cw->dev_ = dev;
        cw->layers_ = moe_layers;
        // The worker loop thread takes part in every pass (for_each runs items on the caller too), so the pool holds one
        // thread less than `threads`: `threads` threads compute, as rayon's `threads` workers do under pool.install.
        const size_t nt = threads ? threads : default_threads();
        cw->pool_ = std::make_unique<ThreadPool>(nt > 1 ? nt - 1 : 1);
        if (const char* tv = std::getenv("STRATA_TRACE_WORKER"); tv && *tv && std::strcmp(tv, "0") != 0) cw->pool_->set_trace(true);
        // STRATA_TRACE_WORKER=1 writes strata_worker_trace.csv in the working directory; any other value is the path.
        if (const char* tv = std::getenv("STRATA_TRACE_WORKER"); tv && *tv && std::strcmp(tv, "0") != 0) {
            cw->trace_ = true;
            cw->trace_path_ = std::strcmp(tv, "1") == 0 ? std::string("strata_worker_trace.csv") : std::string(tv);
            cw->trace_epoch_ = tick_ns();
            cw->trace_rows_.reserve(1 << 16);
        }
        // port of cpu_worker.rs: STRATA_TEST_WORKER_DELAY. Parsed as u64 milliseconds; anything else is ignored.
        if (const char* d = std::getenv("STRATA_TEST_WORKER_DELAY")) {
            char* end = nullptr;
            const unsigned long long ms = std::strtoull(d, &end, 10);
            if (end != d && *end == '\0') cw->delay_ = std::chrono::milliseconds(static_cast<long long>(ms));
        }
        CpuWorker* self = cw.get();
        cw->thread_ = std::thread([self] { self->thread_main(); });
    } catch (const std::exception& e) {
        err = std::string("cpu worker: ") + e.what();
        return false;
    }
    out = std::move(cw);
    return true;
}

CpuWorker::~CpuWorker() { stop(); }

// port of cpu_worker.rs: the thread closure. A worker that exits (stop, or an error in the expert pass) is dead;
// wait_done reports that instead of waiting for an answer that will never come.
void CpuWorker::thread_main() {
    run();
    dead_.store(true, std::memory_order_release);
}

// port of cpu_worker.rs: the body of `pool.install(move || { ... })`: `while let Ok(seq) = rx.recv()` and the layer
// loop inside it.
void CpuWorker::run() {
    // port of cpu_worker.rs: `let pre = crate::weights::Prefetch::start();`
    Prefetch pre;
    pre.start();
    const size_t stride = layout.stride;
    const size_t n = layout.n;
    const size_t k = layout.k;
    auto fail = [this](const std::string& e) {
        std::lock_guard<std::mutex> lk(mu_);
        dead_err_ = e;
    };

    for (;;) {
        uint32_t seq;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return closed_ || !queue_.empty(); });
            if (queue_.empty()) return;   // the sender was dropped: `rx.recv()` returns Err
            seq = queue_.front();
            queue_.pop_front();
        }
        for (size_t li = 0; li < layers_.size(); li++) {
            const size_t il = layers_[li];
            const ExpertLayer& ex = experts_[li];
            uint32_t* base = host_ + il * stride;
            auto at = [base](size_t o) { return base + o; };
            TraceRow tr;
            const bool tracing = trace_;
            if (tracing) {
                tr.seq = seq;
                tr.il = static_cast<uint32_t>(il);
                tr.t_top = tick_ns();
            }

            // Skip the layer on its skip flag, or on a newer seq anywhere (the GPU already moved on to the next
            // token, so it did not wait for this one). Waiting for the old seq would hang.
            uint32_t r;
            for (;;) {
                r = vread(at(REQ));
                if (r == seq) break;
                if (stop_.load(std::memory_order_relaxed)) return;
                if (ahead(r, seq) || ahead(vread(at(SKIP)), seq)) break;
                _mm_pause();
            }
            if (r != seq) {
                if (tracing) {
                    tr.kind = 1;
                    tr.t_req = tr.t_done = tick_ns();
                    trace_rows_.push_back(tr);
                }
                continue;
            }

            // port of cpu_worker.rs: fence(SeqCst) before reading the request (mfence on x86; the loads of
            // write-combining memory are weakly ordered).
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t pf0 = 0, user0 = 0, kern0 = 0;
            if (tracing) {
                tr.t_req = tick_ns();
                pf0 = page_faults();
                thread_times(user0, kern0);
            }
            const size_t nt = std::clamp<size_t>(vread(at(NTOK)), 1, TMAX);
            const size_t kt = k * nt;

            // Bulk streaming reads, as in `routing`.
            std::vector<uint32_t> v(round_up(3 * kt, 4));
            read_wc(at(IDS), v.data(), v.size());
            std::vector<size_t> missed;
            for (size_t j = 0; j < kt; j++) {
                if (static_cast<int32_t>(v[2 * kt + j]) < 0) missed.push_back(v[j]);
            }
            if (tracing) {
                tr.nt = static_cast<uint32_t>(nt);
                tr.missed = static_cast<uint32_t>(missed.size());
                tr.t_read = tick_ns();
            }
            if (!missed.empty()) {
                // Gate and up first: they are read first. One range per (tensor, missed expert).
                // StrataFreet addition: the fused gate_up of inkling2 covers both (gate rows first,
                // then up), so the prefetch lists it in place of the two separate tensors.
                std::vector<std::pair<size_t, size_t>> rs;
                for (const Tensor* t : (ex.fused_gu ? std::vector<const Tensor*>{&ex.gate_up, &ex.down}
                                                    : std::vector<const Tensor*>{&ex.gate, &ex.up, &ex.down})) {
                    for (size_t e : missed) {
                        const Tensor d = t->expert(e, ex.n_expert);
                        rs.push_back({reinterpret_cast<uintptr_t>(d.data), d.len});
                    }
                }
                pre.send(std::move(rs));
            }

            // port of cpu_worker.rs: `sel` of token t (its missed entries, in router order).
            auto sel_of = [&](size_t t) {
                std::vector<std::pair<size_t, float>> sel;
                for (size_t j = t * k; j < (t + 1) * k; j++) {
                    if (static_cast<int32_t>(v[2 * kt + j]) < 0) sel.push_back({v[j], bits_to_f(v[kt + j])});
                }
                return sel;
            };
            // port of cpu_worker.rs: the activation of token t, read in bulk.
            auto read_h = [&](size_t t) {
                std::vector<uint32_t> hw(n);
                read_wc(at(layout.h + t * n), hw.data(), n);
                std::vector<float> h(n);
                std::memcpy(h.data(), hw.data(), n * sizeof(float));
                return h;
            };

            std::string e;
            if (nt > 1) {
                // A verify batch: one pass over each missed expert's weights for all of its tokens.
                std::vector<std::vector<float>> hs;
                std::vector<std::vector<std::pair<size_t, float>>> sels;
                std::vector<size_t> rows;
                for (size_t t = 0; t < nt; t++) {
                    std::vector<std::pair<size_t, float>> sel = sel_of(t);
                    if (sel.empty()) continue;
                    hs.push_back(read_h(t));
                    sels.push_back(std::move(sel));
                    rows.push_back(t);
                }
                if (!hs.empty()) {
                    std::vector<std::vector<float>> ys;
                    if (!experts_multi(ex, hs, sels, ys, *pool_, e)) return fail(e);
                    for (size_t i = 0; i < ys.size(); i++) {
                        std::memcpy(at(layout.y + rows[i] * n), ys[i].data(), n * sizeof(float));
                    }
                }
            } else {
                for (size_t t = 0; t < nt; t++) {
                    const std::vector<std::pair<size_t, float>> sel = sel_of(t);
                    if (sel.empty()) continue;
                    const std::vector<float> h = read_h(t);
                    std::vector<float> y(n);
                    ExpertsTrace et;
                    if (tracing) g_experts_trace = &et;
                    const bool ok = experts(ex, h.data(), sel, y.data(), *pool_, e);
                    if (tracing) {
                        g_experts_trace = nullptr;
                        if (tr.t_exp0 == 0) tr.t_exp0 = et.t[0];
                        tr.t_exp1 = et.t[4];
                        tr.gu += et.t[1] - et.t[0];
                        tr.act += et.t[2] - et.t[1];
                        tr.down += et.t[3] - et.t[2];
                        tr.tail += et.t[4] - et.t[3];
                    }
                    if (!ok) return fail(e);
                    std::memcpy(at(layout.y + t * n), y.data(), n * sizeof(float));
                }
            }
            if (tracing) tr.t_end = tick_ns();

            // port of cpu_worker.rs: STRATA_TEST_WORKER_DELAY. Answers for some (seq, layer) pairs late.
            if (delay_ && (static_cast<size_t>(seq) * 31 + il) % 17 == 0) std::this_thread::sleep_for(*delay_);
            // SeqCst fences: y before DONE, and DONE out now.
            std::atomic_thread_fence(std::memory_order_seq_cst);
            vwrite(at(DONE), seq);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0);
            busy_ns_.fetch_add(static_cast<uint64_t>(ns.count()), std::memory_order_relaxed);
            experts_done_.fetch_add(missed.size(), std::memory_order_relaxed);
            if (tracing) {
                tr.t_done = tick_ns();
                uint64_t user1 = 0, kern1 = 0;
                thread_times(user1, kern1);
                tr.pf = page_faults(&tr.ws_kb) - pf0;
                if (!missed.empty()) tr.e0 = static_cast<uint32_t>(missed[0]);
                tr.user_ns = user1 - user0;
                tr.kern_ns = kern1 - kern0;
                trace_rows_.push_back(tr);
            }
        }
    }
}

// port of cpu_worker.rs: CpuWorker::begin
uint32_t CpuWorker::begin() {
    // self.seq.wrapping_add(1).max(1)
    uint32_t s = seq_ + 1;
    if (s == 0) s = 1;
    seq_ = s;
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(s);
    }
    cv_.notify_one();
    return s;
}

// port of cpu_worker.rs: CpuWorker::routing
std::vector<std::tuple<int32_t, float, int32_t>> CpuWorker::routing(size_t il) const {
    std::vector<std::tuple<int32_t, float, int32_t>> out;
    if (!has_layer(il)) return out;
    const uint32_t* base = host_ + il * layout.stride;
    const size_t nt = std::clamp<size_t>(vread(base + NTOK), 1, TMAX);
    const size_t k = layout.k * nt;
    // One bulk read: on Vulkan the mailbox is BAR memory, where each host read is a slow uncached access.
    std::vector<uint32_t> v(round_up(3 * k, 4));
    read_wc(base + IDS, v.data(), v.size());
    out.reserve(k);
    for (size_t j = 0; j < k; j++) {
        out.emplace_back(static_cast<int32_t>(v[j]), bits_to_f(v[k + j]), static_cast<int32_t>(v[2 * k + j]));
    }
    return out;
}

// port of cpu_worker.rs: CpuWorker::stalled_layer
std::optional<size_t> CpuWorker::stalled_layer() const {
    for (size_t il : layers_) {
        if (vread(host_ + il * layout.stride + VERDICT) == 2) return il;
    }
    return std::nullopt;
}

// port of cpu_worker.rs: CpuWorker::wait_done
bool CpuWorker::wait_done(size_t il, int64_t timeout_ms, std::string& err) {
    if (!has_layer(il)) {
        err = "cpu worker: layer " + std::to_string(il) + " is not a worker layer";
        return false;
    }
    uint32_t* base = host_ + il * layout.stride;
    const uint32_t seq = seq_;
    const auto t0 = std::chrono::steady_clock::now();
    while (vread(base + DONE) != seq) {
        if (dead_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lk(mu_);
            err = "cpu worker: the CPU expert worker stopped" + (dead_err_.empty() ? std::string() : ": " + dead_err_);
            return false;
        }
        if (timeout_ms >= 0) {
            const auto waited = std::chrono::steady_clock::now() - t0;
            if (waited >= std::chrono::milliseconds(timeout_ms)) {
                err = "cpu worker: timed out waiting for layer " + std::to_string(il);
                return false;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    vwrite(base + VERDICT, 0);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return true;
}

bool CpuWorker::has_layer(size_t il) const {
    return std::find(layers_.begin(), layers_.end(), il) != layers_.end();
}

// port of cpu_worker.rs: impl Drop for CpuWorker
void CpuWorker::stop() {
    stop_.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (trace_ && !trace_rows_.empty()) {
        dump_trace();
        trace_rows_.clear();
    }
    if (trace_ && pool_) {
        const ThreadPool::Stats st = pool_->stats();
        if (FILE* f = std::fopen((trace_path_ + ".pool.txt").c_str(), "w")) {
            const double jobs = double(std::max<uint64_t>(st.jobs, 1));
            const double calls = double(std::max<uint64_t>(st.caller_calls, 1));
            std::fprintf(f, "pool threads %zu, for_each calls %llu\n", pool_->size(),
                         static_cast<unsigned long long>(st.caller_calls));
            std::fprintf(f, "pool thread jobs %llu, items %llu\n", static_cast<unsigned long long>(st.jobs),
                         static_cast<unsigned long long>(st.items));
            std::fprintf(f, "pool thread start latency per job (us): %.2f\n", double(st.start_lat_ns) / jobs / 1000.0);
            std::fprintf(f, "pool thread busy per job (us): %.2f, per item (us): %.2f\n",
                         double(st.busy_ns) / jobs / 1000.0, double(st.busy_ns) / double(std::max<uint64_t>(st.items, 1)) / 1000.0);
            std::fprintf(f, "caller per for_each (us): total %.2f, waiting for finish+acks %.2f, own items busy %.2f (own items %.2f per call)\n",
                         double(st.caller_total_ns) / calls / 1000.0, double(st.caller_wait_ns) / calls / 1000.0,
                         double(st.caller_busy_ns) / calls / 1000.0, double(st.caller_items) / calls);
            std::fclose(f);
        }
    }
}

// STRATA_TRACE_WORKER: one CSV row per layer the worker looked at (times in us from the worker's epoch). Written after
// the worker thread has joined, so the rows are not shared.
void CpuWorker::dump_trace() const {
    FILE* f = std::fopen(trace_path_.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "seq,il,nt,missed,kind,t_top,t_req,t_read,t_exp0,t_exp1,t_end,t_done,gu_us,act_us,down_us,tail_us,"
                    "pf,user_us,kern_us,e0,ws_kb\n");
    auto us = [this](int64_t t) { return t == 0 ? 0.0 : double(t - trace_epoch_) / 1000.0; };
    auto d = [](int64_t x) { return double(x) / 1000.0; };
    for (const TraceRow& r : trace_rows_) {
        std::fprintf(f, "%u,%u,%u,%u,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%.3f,%.3f,%u,%llu\n",
                     r.seq, r.il, r.nt, r.missed, r.kind, us(r.t_top), us(r.t_req), us(r.t_read), us(r.t_exp0),
                     us(r.t_exp1), us(r.t_end), us(r.t_done), d(r.gu), d(r.act), d(r.down), d(r.tail),
                     static_cast<unsigned long long>(r.pf), double(r.user_ns) / 1000.0, double(r.kern_ns) / 1000.0,
                     r.e0, static_cast<unsigned long long>(r.ws_kb));
    }
    std::fclose(f);
}

}  // namespace strata
