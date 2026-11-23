// port of deepseek/mat.rs: a quantised matrix view over the GGUF mapping with an f32 matvec (the reference path the
// GLM-5.3-Flash forward runs on), and the optional GPU offload of its matvecs (`mod dev`): weights resident in VRAM,
// an LRU expert cache keyed by the expert's bytes, a 16 MB staging slot for experts that miss it, and batched gemvs.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Deviations forced by C++ or by the DLL (also in MODIFICATIONS.md, section "GLM-5.3-Flash"):
//  - Rust keeps the device state in a process-wide `OnceLock` (`dev::get()`) and runs on rayon's global pool. Here
//    both live in a `DsEnv` owned by the model and every `Mat` points at it, so `sc_free` releases the VRAM and two
//    loaded models do not share one budget.
//  - The environment switches the Rust code reads at each call (STRATA_DS_BATCH once per process, STRATA_CPU_REF,
//    STRATA_DS_NOEVICT, STRATA_DS_LFU per call) are read once when the `DsEnv` is created.
//  - The Rust `.unwrap()`s on device calls panic; here they throw `DsError`, which the GLM model's public functions
//    catch and turn into an error string (`sc_last_error`). The model must then be reset.
//  - The backend is opened through a factory the caller passes (capi.cpp passes `CudaBackend::create`), so this file
//    builds into the CPU-only test DLL too. STRATA_DS_BACKEND=vulkan is refused: there is no Vulkan backend.
//  - `prefetch_expert` is the Rust non-Linux body (a no-op): the Rust code only advises the kernel on Linux.
//  - The Rust `Mutex` around the device state is not needed: one thread drives a model.
//  - `Mat::load_groups` and `Mat::host_only` have no GLM caller and are not ported.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "cpu_moe.h"
#include "gguf.h"
#include "quant.h"

namespace strata {

// A failed device call inside a matvec (the Rust code's `.unwrap()` panic).
struct DsError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The STRATA_DS_* settings `dev::get` reads when it opens the GPUs (after glm::gpu_defaults filled the GLM defaults).
struct DsGpuConfig {
    bool on = false;                 // STRATA_DS_GPU set
    std::string backend;             // STRATA_DS_BACKEND (empty: cuda)
    std::vector<int> devices{0};     // STRATA_DS_DEVICES
    std::optional<double> gb;        // STRATA_DS_GPU_EXPERTS
    double frac = 0.8;               // STRATA_DS_GPU_FRACTION
};

struct Mat;

// Opens backend `device` (capi.cpp: CudaBackend::create). Returns null and sets `err` on failure.
using BackendOpen = std::function<std::unique_ptr<Backend>(int device, std::string& err)>;

// port of deepseek/mat.rs: mod dev. The GPU side of the matvecs.
class Dev {
public:
    // port of deepseek/mat.rs: dev::STAGE (scratch slot for experts that missed the VRAM cache)
    static constexpr size_t STAGE = 16u << 20;
    // port of deepseek/mat.rs: dev::STAGED
    static constexpr uint32_t STAGED = UINT32_MAX;

    // port of deepseek/mat.rs: dev::get (the OnceLock initialiser). Null when the GPU path is off or a GPU could not
    // be opened; `note` then says why (the Rust code prints "GPU d: ...; running without it" and runs on the CPU).
    static std::unique_ptr<Dev> open(const DsGpuConfig& c, const BackendOpen& open_backend, std::string& note);

    // port of deepseek/mat.rs: Dev::bytes
    size_t bytes() const;
    // port of deepseek/mat.rs: Dev::info - (backend name, resident expert slots, bytes of all uploaded weights)
    const char* backend_name() const;
    size_t resident() const { return ex_.size(); }
    // port of deepseek/mat.rs: Dev::budget_left
    size_t budget_left() const;
    // port of deepseek/mat.rs: Dev::pick
    size_t pick(size_t len);
    // port of deepseek/mat.rs: Dev::upload_on
    std::optional<uint32_t> upload_on(size_t d, const uint8_t* data, size_t len, Ty ty, size_t rows, size_t cols);
    // port of deepseek/mat.rs: Dev::upload
    std::optional<uint32_t> upload(const uint8_t* data, size_t len, Ty ty, size_t rows, size_t cols);
    // port of deepseek/mat.rs: Dev::expert (VRAM copy of one routed expert; LRU over slots of the same shape)
    std::optional<uint32_t> expert(const uint8_t* data, size_t len, Ty ty, size_t rows, size_t cols);
    // port of deepseek/mat.rs: Dev::gemv. `staged` carries the host bytes when i == STAGED.
    void gemv(uint32_t i, const float* x, size_t nx, float* y, size_t ny, const uint8_t* staged, size_t staged_len,
              Ty ty, size_t rows, size_t cols);

    // port of deepseek/mat.rs: EX_LOOKUPS (hits, misses since the last PROF print) and running totals for sc_stats.
    std::atomic<uint64_t> lookups_hit{0}, lookups_miss{0};
    uint64_t total_hit = 0, total_miss = 0;
    std::string device_name() const;

    // port of deepseek/mat.rs: Dev::batch_mats. False (Rust None) when the matrices cannot share one GPU batch.
    bool batch_mats(const std::vector<Mat>& ms, const float* x, size_t x_len, size_t xs, std::vector<float>& y);
    // port of deepseek/mat.rs: Dev::rows. False (Rust None) when the weight is not on a GPU or a call failed.
    bool rows(const Mat& m, const std::vector<const float*>& xs, std::vector<float>& y);

private:
    // port of deepseek/mat.rs: Part
    struct Part {
        std::unique_ptr<Backend> be;
        std::vector<GpuTensor> ts;
        Buf x, y, stage;
        size_t bytes = 0, budget = 0, room = 0;
    };
    // port of deepseek/mat.rs: dev::gpu_of
    static size_t gpu_of(uint32_t id) { return (size_t)(id >> SHIFT); }
    static constexpr uint32_t SHIFT = 24;
    const GpuTensor& tensor(uint32_t i) const;
    std::optional<size_t> part_of(const Mat& m) const;

    std::vector<Part> parts_;
    // ex: expert data address -> index in exs; exs: (tensor id, data address, last use, frequency)
    struct Slot {
        uint32_t id;
        uintptr_t key;
        uint64_t last;
        float freq;
    };
    std::unordered_map<uintptr_t, size_t> ex_;
    std::vector<Slot> exs_;
    uint64_t now_ = 0;
    size_t cur_ = 0;
    uint64_t lookups_ = 0;
    bool noevict_ = false;  // STRATA_DS_NOEVICT
    bool lfu_ = false;      // STRATA_DS_LFU
};

// What Rust reads from globals in deepseek/mat.rs: the device (`dev::get()`), rayon's pool, and the switches.
struct DsEnv {
    std::unique_ptr<Dev> dev;
    ThreadPool* pool = nullptr;
    bool batch = true;     // STRATA_DS_BATCH != "0" (matvec_many_opt's OnceLock)
    bool cpu_ref = false;  // STRATA_CPU_REF
    // Reads STRATA_DS_BATCH and STRATA_CPU_REF.
    void read_switches();
};

// port of deepseek/mat.rs: Mat. Row-major weight: `rows` rows of `cols` elements (ggml dims[0] = cols, dims[1] = rows).
// A 3-D expert tensor is addressed with `expert(e)`.
struct Mat {
    const uint8_t* data = nullptr;
    size_t len = 0;
    Ty ty = Ty::F32;
    size_t rows = 0;
    size_t cols = 0;
    // Routed-expert tensor: llama.cpp runs these on the CPU (Q8_K activations); the rest runs on CUDA.
    bool cpu = false;
    // Index of the VRAM copy (STRATA_DS_GPU).
    std::optional<uint32_t> gpu;
    DsEnv* env = nullptr;

    // port of deepseek/mat.rs: Mat::load
    static bool load(const Gguf& g, DsEnv* env, const std::string& name, Mat& out, std::string& err);
    // port of deepseek/mat.rs: Mat::group (row group i of n)
    Mat group(size_t i, size_t n) const;
    // port of deepseek/mat.rs: Mat::expert (slice of a [cols, rows, n_expert] tensor for expert e)
    Mat expert(size_t e, size_t n_expert) const;
    // port of deepseek/mat.rs: Mat::prefetch_expert (the non-Linux body: nothing)
    void prefetch_expert(size_t e, size_t n_expert) const {
        (void)e;
        (void)n_expert;
    }
    // port of deepseek/mat.rs: Mat::row_into
    void row_into(size_t r, float* out) const;
    // port of deepseek/mat.rs: Mat::matvec (y = W x). x has `cols` values, y `rows`.
    void matvec(const float* x, float* y) const;
};

// port of deepseek/mat.rs: matvec_many. Matvecs of several matrices on one activation (xs = 0) or on consecutive
// slices x[i*xs..] (xs > 0); outputs concatenated.
std::vector<float> matvec_many(const std::vector<Mat>& ms, const float* x, size_t x_len, size_t xs);
// port of deepseek/mat.rs: matvec_rows. W x[t] for each row of `xs` (all rows of one matrix).
std::vector<float> matvec_rows(const Mat& m, const std::vector<const float*>& xs);
// port of deepseek/mat.rs: vec_f32 (small F32/BF16 tensor fully dequantised)
bool vec_f32(const Gguf& g, DsEnv* env, const std::string& name, std::vector<float>& out, std::string& err);
// port of deepseek/mat.rs: q8_roundtrip (ggml's Q8_K / Q8_0 activation rounding before a quantised dot)
std::vector<float> q8_roundtrip(Ty ty, bool cpu, const float* x, size_t n);
// half::f16::from_f32(v).to_f32(): round to the nearest f16 (ties to even), back to f32.
float f16_round(float v);

}  // namespace strata
