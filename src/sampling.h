// port of ops.rs: Sampler, argmax, top_k (the token choice after the logits; main.rs and chat.rs use it for every
// decode and the score command uses argmax and top_k directly).
//
// StrataFreet additions (not in ops.rs; see INKLING2_NOTES.md): min_p, presence_penalty, repeat_penalty and the
// recent-token window the penalties see, with the same semantics as stratafreet/engine.py _sample (the torch
// oracle): the penalties land on the raw logits first (repeat: v>0 ? v/rp : v*rp, then presence: v -= pp, on the
// unique recent ids), then the greedy/sampling choice; min_p keeps candidates with p >= min_p * p_max, applied
// after the top_k trim and before the top_p cut.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata {

// port of ops.rs: argmax (the largest value; the first one on ties).
size_t argmax(const float* x, size_t n);

// port of ops.rs: top_k (indices of the k largest values, descending; ties go to the lower index).
std::vector<size_t> top_k(const float* p, size_t n, size_t k);

// port of ops.rs: Sampler. Greedy when `temp` is 0, else softmax(logits / temp) cut to the `top_k` most likely tokens
// (0 = no cut), then to the smallest set whose probability reaches `top_p`. `rng` is the xorshift64 state (not 0).
// The StrataFreet additions: `min_p` (0 = off; keeps candidates with p >= min_p * p_max), `presence_penalty` (0 = off;
// subtracted from the logits of the tokens in `recent` — a positive penalty lowers the logit of tokens that already
// appeared, as OpenAI), `repeat_penalty` (1 = off; logits of the recent tokens are divided
// by it when positive, multiplied when not). `recent` points at the last tokens fed to the model, most recent last
// (the caller's storage; engine.py's window is the last 64 of prompt + generated). The penalties apply before the
// greedy argmax too, as in engine.py.
struct Sampler {
    float temp = 0.0f;
    size_t top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float repeat_penalty = 1.0f;
    const int32_t* recent = nullptr;
    size_t recent_n = 0;
    uint64_t rng = 1;

    // port of ops.rs: Sampler::greedy
    static Sampler greedy() { return Sampler{}; }

    // port of ops.rs: Sampler::pick. Fails on non-finite logits.
    bool pick(const float* logits, size_t n, size_t& out, std::string& err);

    // A penalty is active: repeat_penalty not 1 or 0 (engine.py's truthiness treats 0 as off), or presence_penalty
    // != 0, with recent tokens to apply them to.
    bool has_penalties() const {
        return ((repeat_penalty != 1.0f && repeat_penalty != 0.0f) || presence_penalty != 0.0f) && recent_n > 0;
    }

private:
    // engine.py _sample: the penalties on a copy of the logits (the caller's buffer is not touched).
    void penalize(std::vector<float>& logits) const;
    std::vector<std::pair<size_t, float>> cands(const float* logits, size_t n) const;
    bool draw(const std::vector<std::pair<size_t, float>>& c, size_t& out, std::string& err);
    float rand();
};

// The sampler JSON of the C ABI (capi.h): "temp", "top_k", "top_p", "min_p", "presence_penalty",
// "repeat_penalty", "recent" (an array of token ids), "seed". Absent keys keep the Sampler defaults. Returns false
// with `err` on values outside their ranges (temp/top_k not negative, top_p in (0, 1], min_p in [0, 1],
// repeat_penalty not negative, penalties finite, recent no longer than 4096 ids). `seed` is -1 when the key is
// absent, else the parsed value (the caller applies it to its generator state).
bool sampler_from_json(const std::string& sj, Sampler& out, std::vector<int32_t>& recent, double& seed,
                       bool& has_seed, std::string& err);

}  // namespace strata
