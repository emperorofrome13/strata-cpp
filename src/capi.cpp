// C ABI of strata-cpp (implementation of src/capi.h).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Every function that can fail returns 0 on success; the text of the last failure is returned by
// sc_last_error(). The strings this file hands out are thread-local and stay valid until the next call
// on the same thread (sc_stats: until the next call with the same handle).
#include "capi.h"

#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "cuda_backend.h"
#include "gguf.h"
#include "gpu_model.h"
#include "model.h"
#include "nvrtc.h"
#include "sampling.h"

namespace {

thread_local std::string g_err;

void set_error(const std::string& s) { g_err = s; }

// The kernel sc_selftest_kernel compiles and launches. Strata builds its decode kernel set the same way
// (src/cuda/mod.rs source_with() prepends macros, then NVRTC compiles); this is the smallest working form.
const char* kSelftestSrc =
    "extern \"C\" __global__ void sc_fill(float* y, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) y[i] = 2.0f * i;\n"
    "}\n";

std::string cu_error(const char* what, CUresult r) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &text);
    return std::string(what) + " failed: " + (name ? name : "?") + " (" + (text ? text : "?") + ")";
}

// A number under `key` in a flat JSON object. False when the key is absent or not a number.
bool json_num(const std::string& js, const char* key, double& out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return false;
    p = js.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < js.size() && (js[p] == ' ' || js[p] == '\t' || js[p] == '\n' || js[p] == '\r')) p++;
    char* end = nullptr;
    const double v = std::strtod(js.c_str() + p, &end);
    if (end == js.c_str() + p) return false;
    out = v;
    return true;
}

// A boolean under `key` in a flat JSON object. False when the key is absent or the value is not true/false.
bool json_bool(const std::string& js, const char* key, bool& out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return false;
    p = js.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p++;
    while (p < js.size() && (js[p] == ' ' || js[p] == '\t' || js[p] == '\n' || js[p] == '\r')) p++;
    if (js.compare(p, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (js.compare(p, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

// A string under `key` in a flat JSON object (no escapes). False when the key is absent or not a string.
bool json_str(const std::string& js, const char* key, std::string& out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return false;
    p = js.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p = js.find('"', p + 1);
    if (p == std::string::npos) return false;
    const size_t q = js.find('"', p + 1);
    if (q == std::string::npos) return false;
    out = js.substr(p + 1, q - p - 1);
    return true;
}

// One loaded model: the GGUF mapping, the model (which points into it) and the GPU model (which owns the device
// memory and the backend). Member order is the destruction order's reverse: the GPU model goes first.
struct Session {
    strata::Gguf g;
    strata::Model model;
    std::unique_ptr<strata::GpuModel> gm;
    std::vector<float> logits;
    std::string stats;
    size_t threads = 0;
    size_t vram_margin = 0;
    uint64_t seed = 1;
    uint64_t rng = 1;
    // The calibration file (strata-<stem>.json): -1 when absent or the key is missing.
    double calib_chunk = -1, calib_mtp = -1, calib_short = -1;
    std::string calib_path;
    // StrataFreet addition: sc_load "ram" / "pin" and what they did (sc_stats reports them).
    std::string ram_mode = "off";
    strata::RamLoad ram;
    bool ram_pinned = false;
    double load_seconds = 0;
};

std::string json_esc(const std::string& in);

// A JSON array body of strings (the ram warnings).
std::string warnings_json(const std::vector<std::string>& ws) {
    std::string out;
    for (size_t i = 0; i < ws.size(); i++) {
        if (i) out += ",";
        out += "\"" + json_esc(ws[i]) + "\"";
    }
    return out;
}

// JSON string body for a path (backslashes and quotes escaped).
std::string json_esc(const std::string& in) {
    std::string out;
    for (char c : in) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

Session* session(void* h, const char* fn) {
    if (!h) {
        set_error(std::string(fn) + ": null model handle");
        return nullptr;
    }
    return static_cast<Session*>(h);
}

}  // namespace

SC_EXPORT const char* sc_version(void) {
    return "strata-cpp 0.0.1 (port of RustingStrata 1.2.5 by Vasyl Trefilov)";
}

SC_EXPORT const char* sc_last_error(void) { return g_err.c_str(); }

SC_EXPORT const char* sc_nvrtc_version(void) {
    static thread_local std::string out;
    if (auto v = strata::nvrtc::version())
        out = std::to_string(v->first) + "." + std::to_string(v->second);
    else
        out = strata::nvrtc::missing_message();
    return out.c_str();
}

SC_EXPORT int sc_selftest_kernel(float* out, int n) {
    if (!out || n <= 0) {
        set_error("sc_selftest_kernel: out must be non-null and n must be positive");
        return 1;
    }

    CUresult cr = cuInit(0);
    if (cr != CUDA_SUCCESS) {
        set_error(cu_error("cuInit", cr));
        return 1;
    }
    CUdevice dev = 0;
    cr = cuDeviceGet(&dev, 0);
    if (cr != CUDA_SUCCESS) {
        set_error(cu_error("cuDeviceGet", cr));
        return 1;
    }
    int major = 0, minor = 0;
    cr = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (cr == CUDA_SUCCESS)
        cr = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (cr != CUDA_SUCCESS) {
        set_error(cu_error("cuDeviceGetAttribute(compute capability)", cr));
        return 1;
    }

    // A CUBIN runs only on the arch it was built for, so compile for this device's own sm_XY.
    std::vector<uint8_t> cubin;
    std::string nerr;
    if (!strata::nvrtc::cubin(kSelftestSrc, strata::nvrtc::sm_arch(major, minor), cubin, nerr)) {
        set_error(nerr);
        return 1;
    }

    // The primary context is the one the CUDA runtime uses, so this does not fork a second context
    // next to StrataFreet's own work in the same process.
    CUcontext ctx = nullptr;
    cr = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (cr != CUDA_SUCCESS) {
        set_error(cu_error("cuDevicePrimaryCtxRetain", cr));
        return 1;
    }
    cr = cuCtxSetCurrent(ctx);
    if (cr != CUDA_SUCCESS) {
        set_error(cu_error("cuCtxSetCurrent", cr));
        cuDevicePrimaryCtxRelease(dev);
        return 1;
    }

    CUmodule mod = nullptr;
    CUdeviceptr d = 0;
    CUfunction fn = nullptr;
    cr = cuModuleLoadData(&mod, cubin.data());
    if (cr != CUDA_SUCCESS) {
        set_error(cu_error("cuModuleLoadData", cr));
    } else if ((cr = cuModuleGetFunction(&fn, mod, "sc_fill")) != CUDA_SUCCESS) {
        set_error(cu_error("cuModuleGetFunction(sc_fill)", cr));
    } else if ((cr = cuMemAlloc(&d, (size_t)n * sizeof(float))) != CUDA_SUCCESS) {
        set_error(cu_error("cuMemAlloc", cr));
    } else {
        void* args[] = {&d, &n};
        const unsigned block = 256;
        const unsigned grid = ((unsigned)n + block - 1) / block;
        cr = cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr);
        if (cr == CUDA_SUCCESS) cr = cuCtxSynchronize();
        if (cr == CUDA_SUCCESS) cr = cuMemcpyDtoH(out, d, (size_t)n * sizeof(float));
        if (cr != CUDA_SUCCESS) set_error(cu_error("cuLaunchKernel/copy", cr));
    }

    if (d) cuMemFree(d);
    if (mod) cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return cr == CUDA_SUCCESS ? 0 : 1;
}

SC_EXPORT void* sc_load(const char* gguf_path, const char* options_json) {
    if (!gguf_path) {
        set_error("sc_load: gguf_path is null");
        return nullptr;
    }
    const std::string opts = options_json ? options_json : "{}";
    double ctx = 4096, margin_mb = 512, threads = 0, seed = 1, device = 0;
    json_num(opts, "ctx", ctx);
    json_num(opts, "vram_margin_mb", margin_mb);
    json_num(opts, "threads", threads);
    json_num(opts, "seed", seed);
    json_num(opts, "device", device);
    std::string kv = "f16";
    json_str(opts, "kv", kv);
    if (kv != "f16" && kv != "q8") {
        set_error("sc_load: kv \"" + kv + "\" is not f16 or q8");
        return nullptr;
    }
    if (ctx < 1 || margin_mb < 0 || threads < 0) {
        set_error("sc_load: ctx must be at least 1, vram_margin_mb and threads not negative");
        return nullptr;
    }
    // StrataFreet addition. "ram": off (file mapping, the default), copy (private RAM copy of the tensor data),
    // lock (copy, resident). "pin": the RAM copy is also registered with CUDA so expert copies DMA from it.
    std::string ram = "off";
    json_str(opts, "ram", ram);
    bool pin = false;
    json_bool(opts, "pin", pin);
    if (ram != "off" && ram != "copy" && ram != "lock") {
        set_error("sc_load: ram \"" + ram + "\" is not off, copy or lock");
        return nullptr;
    }
    if (pin && ram == "off") {
        set_error("sc_load: pin needs ram copy or lock (the pinned memory is the RAM copy)");
        return nullptr;
    }
    const auto t_load0 = std::chrono::steady_clock::now();

    auto s = std::make_unique<Session>();
    std::string err;
    if (!s->g.open(gguf_path, err)) {
        set_error(err);
        return nullptr;
    }
    s->ram_mode = ram;
    const strata::RamMode mode = ram == "lock" ? strata::RamMode::Lock
                                 : ram == "copy" ? strata::RamMode::Copy
                                                 : strata::RamMode::Off;
    if (!s->g.load_to_ram(mode, s->ram, err)) {
        set_error(err);
        return nullptr;
    }
    if (!s->model.load(s->g, err)) {
        set_error(err);
        return nullptr;
    }
    // port of main.rs: the calibration file of this model (calibration_path: cache_dir()/strata-<stem>.json, the
    // CUDA name has no suffix). Its chunk and short settings are the defaults of the load (sc_load below); mtp is only
    // reported (MTP is not ported).
    {
        const std::string gp = gguf_path;
        const size_t slash = gp.find_last_of("/\\");
        std::string stem = slash == std::string::npos ? gp : gp.substr(slash + 1);
        const size_t dot = stem.find_last_of('.');
        if (dot != std::string::npos && dot > 0) stem = stem.substr(0, dot);
        s->calib_path = strata::cache_dir() + "\\strata-" + stem + ".json";
        std::ifstream f(s->calib_path, std::ios::binary);
        if (f) {
            const std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            double v = 0;
            if (json_num(js, "chunk", v)) s->calib_chunk = v;
            if (json_num(js, "mtp", v)) s->calib_mtp = v;
            if (json_num(js, "short", v)) s->calib_short = v;
        }
    }
    std::unique_ptr<strata::CudaBackend> be = strata::CudaBackend::create(static_cast<int>(device), err, kv == "q8");
    if (!be) {
        set_error(err);
        return nullptr;
    }
    if (pin) {
        // All or nothing: a registration the driver refuses (size or pinned-memory limits) is a warning and the
        // loads go through the pinned ring as before.
        std::string perr;
        bool ok = true;
        for (const strata::RamShard& r : s->g.ram()) {
            if (r.len == 0) continue;
            if (!be->register_host(r.base, r.len, perr)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            s->ram_pinned = true;
        } else {
            be->unregister_host();
            const std::string w = "pin failed: " + perr + "; continuing unpinned";
            s->ram.warnings.push_back(w);
            std::fprintf(stderr, "[strata-cpp] WARNING %s\n", w.c_str());
        }
    }
    s->vram_margin = static_cast<size_t>(margin_mb) << 20;
    s->threads = static_cast<size_t>(threads);
    s->seed = seed >= 1 ? static_cast<uint64_t>(seed) : 1;
    s->rng = s->seed;
    // port of main.rs: Engine::fill_from_calibration / `get("chunk").unwrap_or(4096)`, `get("short").unwrap_or(32)`: the
    // options win, then the calibration file, then the defaults.
    double chunk = s->calib_chunk >= 0 ? s->calib_chunk : 4096;
    double short_len = s->calib_short >= 0 ? s->calib_short : 32;
    json_num(opts, "chunk", chunk);
    json_num(opts, "short", short_len);
    if (chunk < 0 || short_len < 0) {
        set_error("sc_load: chunk and short must not be negative");
        return nullptr;
    }
    strata::LoadOptions lo;
    lo.ctx = static_cast<size_t>(ctx);
    lo.vram_margin = s->vram_margin;
    lo.threads = s->threads;
    lo.chunk = static_cast<size_t>(chunk);
    lo.short_len = static_cast<size_t>(short_len);
    if (!strata::GpuModel::load(s->model, std::move(be), lo, s->gm, err)) {
        set_error(err);
        return nullptr;
    }
    s->load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    return s.release();
}

SC_EXPORT int sc_n_vocab(void* h) {
    Session* s = session(h, "sc_n_vocab");
    if (!s) return 0;
    return static_cast<int>(s->model.cfg.n_vocab);
}

SC_EXPORT int sc_decode(void* h, int32_t tok, float* logits_out) {
    Session* s = session(h, "sc_decode");
    if (!s) return 1;
    if (tok < 0 || static_cast<size_t>(tok) >= s->model.cfg.n_vocab) {
        set_error("sc_decode: token " + std::to_string(tok) + " is outside the vocabulary (" +
                  std::to_string(s->model.cfg.n_vocab) + ")");
        return 1;
    }
    std::string err;
    if (!s->gm->step(static_cast<uint32_t>(tok), s->logits, err)) {
        set_error(err);
        return 1;
    }
    if (logits_out) std::memcpy(logits_out, s->logits.data(), s->logits.size() * sizeof(float));
    return 0;
}

SC_EXPORT int sc_decode_sample(void* h, int32_t tok, const char* sampler_json, int32_t* out) {
    Session* s = session(h, "sc_decode_sample");
    if (!s) return 1;
    if (!out) {
        set_error("sc_decode_sample: out must be non-null");
        return 1;
    }
    if (tok < 0 || static_cast<size_t>(tok) >= s->model.cfg.n_vocab) {
        set_error("sc_decode_sample: token " + std::to_string(tok) + " is outside the vocabulary");
        return 1;
    }
    const std::string sj = sampler_json ? sampler_json : "{}";
    strata::Sampler smp;
    std::vector<int32_t> recent;
    double seed = 0;
    bool has_seed = false;
    std::string err;
    if (!strata::sampler_from_json(sj, smp, recent, seed, has_seed, err)) {
        set_error("sc_decode_sample: " + err);
        return 1;
    }
    if (has_seed) s->rng = seed >= 1 ? static_cast<uint64_t>(seed) : 1;
    smp.recent = recent.data();
    smp.recent_n = recent.size();
    smp.rng = s->rng;
    if (smp.temp == 0 && !smp.has_penalties()) {
        // Greedy: the argmax is taken on the device (no logits copy), first index on ties (as ops.rs argmax).
        // With penalties active the host path below applies them before the argmax (engine.py parity).
        int32_t am = 0;
        if (!s->gm->step_greedy(static_cast<uint32_t>(tok), am, err)) {
            set_error(err);
            return 1;
        }
        *out = am;
        return 0;
    }
    if (!s->gm->step(static_cast<uint32_t>(tok), s->logits, err)) {
        set_error(err);
        return 1;
    }
    size_t pick = 0;
    if (!smp.pick(s->logits.data(), s->logits.size(), pick, err)) {
        set_error(err);
        return 1;
    }
    s->rng = smp.rng;
    *out = static_cast<int32_t>(pick);
    return 0;
}

// The prompt through the batched prefill (GpuModel::prefill); `logits` receives the last position's logits.
static int prefill_run(Session* s, const char* fn, const int32_t* ids, int n, std::vector<float>& logits) {
    if (!ids || n <= 0) {
        set_error(std::string(fn) + ": ids must be non-null and n positive");
        return 1;
    }
    const size_t nv = s->model.cfg.n_vocab;
    std::vector<uint32_t> toks(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= nv) {
            set_error(std::string(fn) + ": token " + std::to_string(ids[i]) + " is outside the vocabulary (" +
                      std::to_string(nv) + ")");
            return 1;
        }
        toks[static_cast<size_t>(i)] = static_cast<uint32_t>(ids[i]);
    }
    std::string err;
    if (!s->gm->prefill(toks, logits, err)) {
        set_error(err);
        return 1;
    }
    return 0;
}

SC_EXPORT int sc_prefill(void* h, const int32_t* ids, int n) {
    Session* s = session(h, "sc_prefill");
    if (!s) return 1;
    return prefill_run(s, "sc_prefill", ids, n, s->logits);
}

SC_EXPORT int sc_prefill_logits(void* h, const int32_t* ids, int n, float* logits_out) {
    Session* s = session(h, "sc_prefill_logits");
    if (!s) return 1;
    if (!logits_out) {
        set_error("sc_prefill_logits: logits_out must be non-null");
        return 1;
    }
    if (prefill_run(s, "sc_prefill_logits", ids, n, s->logits) != 0) return 1;
    std::memcpy(logits_out, s->logits.data(), s->logits.size() * sizeof(float));
    return 0;
}

SC_EXPORT int sc_prefill_sample(void* h, const int32_t* ids, int n, const char* sampler_json, int32_t* out) {
    Session* s = session(h, "sc_prefill_sample");
    if (!s) return 1;
    if (!out) {
        set_error("sc_prefill_sample: out must be non-null");
        return 1;
    }
    if (prefill_run(s, "sc_prefill_sample", ids, n, s->logits) != 0) return 1;
    const std::string sj = sampler_json ? sampler_json : "{}";
    strata::Sampler smp;
    std::vector<int32_t> recent;
    double seed = 0;
    bool has_seed = false;
    std::string err;
    if (!strata::sampler_from_json(sj, smp, recent, seed, has_seed, err)) {
        set_error("sc_prefill_sample: " + err);
        return 1;
    }
    if (has_seed) s->rng = seed >= 1 ? static_cast<uint64_t>(seed) : 1;
    smp.recent = recent.data();
    smp.recent_n = recent.size();
    smp.rng = s->rng;
    if (smp.temp == 0 && !smp.has_penalties()) {
        // Greedy: the first index of the largest logit (ops.rs argmax). With penalties active the path below
        // applies them before the argmax (engine.py parity).
        size_t best = 0;
        for (size_t i = 1; i < s->logits.size(); i++) {
            if (s->logits[i] > s->logits[best]) best = i;
        }
        *out = static_cast<int32_t>(best);
        return 0;
    }
    size_t pick = 0;
    if (!smp.pick(s->logits.data(), s->logits.size(), pick, err)) {
        set_error(err);
        return 1;
    }
    s->rng = smp.rng;
    *out = static_cast<int32_t>(pick);
    return 0;
}

SC_EXPORT int sc_reset(void* h, int32_t keep_tokens) {
    Session* s = session(h, "sc_reset");
    if (!s) return 1;
    if (keep_tokens != 0) {
        set_error("sc_reset: keep_tokens must be 0 (a partial reset needs the DeltaNet state snapshots, not ported yet)");
        return 1;
    }
    std::string err;
    if (!s->gm->reset(err)) {
        set_error(err);
        return 1;
    }
    return 0;
}

SC_EXPORT int sc_save_profile(void* h) {
    Session* s = session(h, "sc_save_profile");
    if (!s) return 1;
    std::string err;
    if (!s->gm->save_profile(err)) {
        set_error(err);
        return 1;
    }
    return 0;
}

SC_EXPORT const char* sc_stats(void* h) {
    Session* s = session(h, "sc_stats");
    if (!s) return "";
    const strata::GpuModel& gm = *s->gm;
    const strata::ExpertCache* ec = gm.experts();
    const double busy_ms = double(gm.cpu_busy_ns()) / 1e6;
    const double total = double(gm.hits + gm.misses);
    char rate[32];
    std::snprintf(rate, sizeof rate, "%.4f", total > 0 ? double(gm.hits) / total : 1.0);
    char busy_tok[32];
    std::snprintf(busy_tok, sizeof busy_tok, "%.4f", gm.tokens ? busy_ms / double(gm.tokens) : 0.0);
    const auto* cuda = dynamic_cast<const strata::CudaBackend*>(&s->gm->backend());
    const uint64_t direct = cuda ? cuda->host_direct_bytes() : 0;
    s->stats = "{\"arch\":\"" + s->model.cfg.arch + "\"" +
               ",\"n_layer\":" + std::to_string(s->model.cfg.n_layer) +
               ",\"n_vocab\":" + std::to_string(s->model.cfg.n_vocab) +
               ",\"ctx\":" + std::to_string(gm.ctx()) +
               ",\"pos\":" + std::to_string(gm.pos()) +
               ",\"tokens\":" + std::to_string(gm.tokens) +
               ",\"hits\":" + std::to_string(gm.hits) +
               ",\"misses\":" + std::to_string(gm.misses) +
               ",\"hit_rate\":" + rate +
               ",\"swaps\":" + std::to_string(ec ? ec->swaps : 0) +
               ",\"experts_resident\":" + std::to_string(ec ? ec->resident_count() : 0) +
               ",\"experts_total\":" + std::to_string(ec ? ec->total_experts() : 0) +
               ",\"expert_mb\":" + std::to_string((ec ? ec->slot_bytes_total() : 0) >> 20) +
               ",\"fully_resident\":" + (ec && ec->fully_resident() ? "true" : "false") +
               ",\"worker_busy_ms\":" + std::to_string(busy_ms) +
               ",\"worker_busy_ms_per_token\":" + busy_tok +
               ",\"worker_experts\":" + std::to_string(gm.cpu_experts()) +
               ",\"mailbox_writes\":" + std::to_string(gm.mailbox_writes) +
               ",\"mailbox_reads\":" + std::to_string(gm.mailbox_reads) +
               ",\"resumed\":" + std::to_string(gm.resumed) +
               ",\"adapt_every\":" + std::to_string(gm.adapt_every) +
               ",\"adapt_swaps\":" + std::to_string(gm.adapt_swaps) +
               ",\"profile\":\"" + json_esc(ec ? ec->profile_path() : std::string()) + "\"" +
               ",\"calibration\":{\"path\":\"" + json_esc(s->calib_path) + "\",\"chunk\":" + std::to_string(s->calib_chunk) +
               ",\"mtp\":" + std::to_string(s->calib_mtp) + ",\"short\":" + std::to_string(s->calib_short) + "}" +
               ",\"expert_bytes_per_token\":" + std::to_string(gm.expert_bytes_per_token()) +
               ",\"prefill_chunk\":" + std::to_string(gm.chunk()) +
               ",\"prefill_short\":" + std::to_string(gm.short_len()) +
               ",\"prefill_mb\":" + std::to_string(gm.prefill_bytes() >> 20) +
               ",\"n_expert\":" + std::to_string(s->model.cfg.n_expert) +
               ",\"n_expert_used\":" + std::to_string(s->model.cfg.n_expert_used) +
               ",\"device\":\"" + s->gm->backend().device() + "\"" +
               ",\"threads\":" + std::to_string(s->threads) +
               ",\"vram_margin_mb\":" + std::to_string(s->vram_margin >> 20) +
               ",\"ram_mode\":\"" + s->ram_mode + "\"" +
               ",\"ram_bytes\":" + std::to_string(s->ram.bytes) +
               ",\"ram_locked_bytes\":" + std::to_string(s->ram.locked_bytes) +
               ",\"ram_pinned\":" + (s->ram_pinned ? "true" : "false") +
               ",\"ram_direct_bytes\":" + std::to_string(direct) +
               ",\"ram_load_seconds\":" + std::to_string(s->ram.seconds) +
               ",\"load_seconds\":" + std::to_string(s->load_seconds) +
               ",\"ram_warnings\":[" + warnings_json(s->ram.warnings) + "]}";
    return s->stats.c_str();
}

SC_EXPORT void sc_free(void* h) {
    if (!h) return;
    Session* s = static_cast<Session*>(h);
    std::string err;
    if (s->gm && !s->gm->save_profile(err)) set_error(err);
    delete s;
}
