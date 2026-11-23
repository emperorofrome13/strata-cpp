// port of weights.rs (named tensor views over the mmapped GGUF, page-cache advice for them, and a
// reference parallel matvec). Function and type names match the Rust ones.
//
// Deviations forced by C++ (listed in MODIFICATIONS.md):
//  - `data: &'a [u8]` carries its length (`len`) because a C++ slice does not.
//  - rayon's `par_chunks_mut` becomes a fixed-chunk split over std::thread workers; the chunk
//    boundaries and every per-element summation order are identical, so the results are bit-identical.
//  - `Box::leak` of the synthesized ssm_beta/ssm_alpha bytes becomes ownership in `owned_`.
//  - `bake_lora` is out of scope (STRATA_CPP_PLAN.md section 4: LoRA out); `clear_baked` is ported.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "gguf.h"
#include "quant.h"

namespace strata {

// port of weights.rs: Tensor
struct Tensor {
    Ty ty = Ty::F32;
    // Elements per row (GGUF dims[0]).
    size_t cols = 0;
    // Product of the remaining dims; for 3-D expert tensors this is out_rows * n_expert.
    size_t rows = 0;
    const uint8_t* data = nullptr;
    // Rust's `&[u8]` length.
    size_t len = 0;

    // port of weights.rs: Tensor::row_bytes
    size_t row_bytes() const;
    // port of weights.rs: Tensor::row (y = row_r dequantized, `cols` floats)
    bool row(size_t r, float* out) const;
    // port of weights.rs: Tensor::matvec (y[r] = row_r . x)
    void matvec(const float* x, float* y) const;
    // port of weights.rs: Tensor::matmul (out[t][r] = row_r . x[t])
    void matmul(const float* x, size_t n_tokens, float* out) const;
    // port of weights.rs: Tensor::expert (rows belonging to expert `e` of a [cols, out, n_expert] tensor)
    Tensor expert(size_t e, size_t n_expert) const;
    // port of weights.rs: Tensor::to_f32
    std::vector<float> to_f32() const;
};

// port of weights.rs: Weights
class Weights {
public:
    // port of weights.rs: Weights::new / Weights::new_with_aux
    bool init(const Gguf* g, const Gguf* aux_g, std::string& err);
    // port of weights.rs: Weights::get
    bool get(const std::string& name, Tensor& out, std::string& err) const;
    // port of weights.rs: Weights::layer
    bool layer(size_t il, const std::string& suffix, Tensor& out, std::string& err) const;
    // StrataFreet addition (INKLING2_NOTES.md): the same lookup WITHOUT the looped slot remap, for the
    // callers that address physical blocks directly (the expert cache is keyed by them).
    bool layer_phys(size_t il, const std::string& suffix, Tensor& out, std::string& err) const;
    // port of weights.rs: Weights::validate
    bool validate(size_t n_layer, bool& nextn, std::string& err) const;
    // port of weights.rs: Weights::clear_baked
    void clear_baked();

    const ModelConfig& config() const { return cfg_; }
    const Gguf* g = nullptr;
    const Gguf* aux_g = nullptr;
    // Loader-owned overrides: synthesized ssm_beta/ssm_alpha (LoRA-baked tensors are out of scope).
    std::map<std::string, Tensor> built;

private:
    // Announces the tensor names in `built` to `get`; a faithful stand-in for the borrow of
    // `ModelConfig::aliases` the Rust version takes from a `&'static` table.
    ModelConfig cfg_;
    // Heap storage the tensors in `built` point into (the Rust version leaks these with Box::leak).
    std::vector<std::unique_ptr<uint8_t[]>> owned_;
    std::set<std::string> lora_baked_;
};

// port of weights.rs: will_need. The Rust function is a no-op off Linux (`#[cfg(not(target_os =
// "linux"))] fn madvise(..) {}`), so on Windows the reference performs no readahead; the port keeps
// that exact behaviour rather than inventing one.
void will_need(size_t p, size_t n);

// port of weights.rs: cold. Also a no-op on Windows for the same reason (MADV_DONTNEED/MADV_COLD).
void cold(const uint8_t* b, size_t len);

// port of weights.rs: Prefetch - one background thread that reads ranges of mapped weights in the
// order sent, so a cold read keeps many drive reads in flight instead of one page fault at a time.
// Dropping it (or destroy()) joins the thread.
class Prefetch {
public:
    Prefetch() = default;
    ~Prefetch();
    Prefetch(const Prefetch&) = delete;
    Prefetch& operator=(const Prefetch&) = delete;

    // port of weights.rs: Prefetch::start
    void start();
    // port of weights.rs: Prefetch::send - (address, bytes) ranges of mapped weights.
    void send(std::vector<std::pair<size_t, size_t>> rs);

private:
    void destroy();
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::vector<std::pair<size_t, size_t>>> queue_;
    bool stop_ = false;
    bool running_ = false;
};

// port of weights.rs: dot - 16 independent partial sums, so the compiler vectorizes it (a single
// running f32 sum cannot be). 32 and 64 sums measured slower.
float dot(const float* a, const float* b, size_t n);

}  // namespace strata
