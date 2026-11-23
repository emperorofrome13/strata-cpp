// port of ops.rs: argmax, top_k, Sampler::pick / cands / draw / rand.
//
// StrataFreet additions (not in ops.rs): the penalties and min_p of stratafreet/engine.py _sample (the torch
// oracle), and sampler_from_json (the C ABI's sampler JSON, shared by capi.cpp and the dev surface).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace strata {

namespace {
const char* kNonFinite = "the forward pass produced NaN or infinite logits";
}  // namespace

// port of ops.rs: argmax. `total_cmp` order: NaN sorts above every number, as in Rust.
size_t argmax(const float* x, size_t n) {
    size_t b = 0;
    for (size_t i = 1; i < n; i++) {
        if (std::isgreater(x[i], x[b]) || (std::isnan(x[i]) && !std::isnan(x[b]))) b = i;
    }
    return b;
}

// port of ops.rs: top_k
std::vector<size_t> top_k(const float* p, size_t n, size_t k) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t(0));
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (p[a] != p[b]) return p[a] > p[b];
        return a < b;
    });
    if (idx.size() > k) idx.resize(k);
    return idx;
}

// engine.py _sample: on the unique recent ids, first the repeat penalty (v > 0 ? v / rp : v * rp), then the
// presence penalty (v -= pp). The order matters (v/rp - pp != (v - pp)/rp in general); this is engine.py's order.
// Out-of-range ids are skipped (engine.py would crash on them; the DLL is defensive).
void Sampler::penalize(std::vector<float>& logits) const {
    std::vector<int32_t> seen;
    seen.reserve(recent_n);
    for (size_t i = 0; i < recent_n; i++) {
        const int32_t id = recent[i];
        if (id < 0 || static_cast<size_t>(id) >= logits.size()) continue;
        bool dup = false;
        for (const int32_t s : seen) {
            if (s == id) { dup = true; break; }
        }
        if (dup) continue;
        seen.push_back(id);
        float& v = logits[static_cast<size_t>(id)];
        // engine.py's truthiness treats repeat_penalty 0 as off; a 0 here would divide by zero.
        if (repeat_penalty != 1.0f && repeat_penalty != 0.0f)
            v = v > 0.0f ? v / repeat_penalty : v * repeat_penalty;
        if (presence_penalty != 0.0f) v -= presence_penalty;
    }
}

// port of ops.rs: Sampler::pick
bool Sampler::pick(const float* logits, size_t n, size_t& out, std::string& err) {
    std::vector<float> buf;
    const float* x = logits;
    if (has_penalties()) {
        // engine.py _sample: the penalties land on the raw logits, before the greedy/sampling choice.
        buf.assign(logits, logits + n);
        penalize(buf);
        x = buf.data();
    }
    if (temp <= 0.0f) {
        out = argmax(x, n);
        if (!std::isfinite(x[out])) {
            err = kNonFinite;
            return false;
        }
        return true;
    }
    return draw(cands(x, n), out, err);
}

// port of ops.rs: Sampler::cands, with the StrataFreet min_p cut (engine.py: keep probs >= min_p * probs[0] on
// the softmax probs — the same ratio on the raw exps, applied after the top_k trim and before the top_p cut).
std::vector<std::pair<size_t, float>> Sampler::cands(const float* logits, size_t n) const {
    float m = -INFINITY;
    for (size_t i = 0; i < n; i++) m = std::max(m, logits[i]);
    const float lo = m - 21.0f * temp;
    std::vector<std::pair<size_t, float>> c;
    for (size_t i = 0; i < n; i++) {
        if (!(logits[i] >= lo)) continue;
        const float x = (logits[i] - m) / temp;
        if (!(x > -20.0f)) continue;
        c.push_back({i, std::exp(x)});
    }
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    if (top_k > 0 && c.size() > top_k) c.resize(top_k);
    if (min_p > 0.0f && !c.empty()) {
        // Sorted descending by probability, so the survivors are a prefix. min_p <= 1 keeps at least the first.
        const float cut = min_p * c.front().second;
        size_t keep = 0;
        while (keep < c.size() && c[keep].second >= cut) keep++;
        c.resize(keep);
    }
    float total = 0.0f;
    for (const auto& x : c) total += x.second;
    size_t keep = c.size();
    float acc = 0.0f;
    for (size_t j = 0; j < c.size(); j++) {
        acc += c[j].second;
        if (acc >= top_p * total) {
            keep = j + 1;
            break;
        }
    }
    c.resize(keep);
    return c;
}

// port of ops.rs: Sampler::draw
bool Sampler::draw(const std::vector<std::pair<size_t, float>>& c, size_t& out, std::string& err) {
    if (c.empty()) {
        err = kNonFinite;
        return false;
    }
    float total = 0.0f;
    for (const auto& x : c) total += x.second;
    float r = rand() * total;
    for (const auto& [i, p] : c) {
        r -= p;
        if (r < 0.0f) {
            out = i;
            return true;
        }
    }
    out = c.back().first;
    return true;
}

// port of ops.rs: Sampler::rand (uniform in [0, 1))
float Sampler::rand() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return static_cast<float>(rng >> 40) / static_cast<float>(uint64_t(1) << 24);
}

// A number under `key` in a flat JSON object. False when the key is absent or not a number.
static bool json_num(const std::string& js, const char* key, double& out) {
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

// An integer array under `key` in a flat JSON object (e.g. "recent": [12, 7, 40]). False when the key is absent
// or the array is malformed. The caller bounds the length.
static bool json_ints(const std::string& js, const char* key, std::vector<int32_t>& out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return false;
    p = js.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p = js.find('[', p + 1);
    if (p == std::string::npos) return false;
    p++;
    out.clear();
    while (p < js.size()) {
        while (p < js.size() && (js[p] == ' ' || js[p] == '\t' || js[p] == '\n' || js[p] == '\r' || js[p] == ',')) p++;
        if (p < js.size() && js[p] == ']') return true;
        char* end = nullptr;
        const long v = std::strtol(js.c_str() + p, &end, 10);
        if (end == js.c_str() + p) return false;
        out.push_back(static_cast<int32_t>(v));
        p = static_cast<size_t>(end - js.c_str());
    }
    return false;
}

// The sampler JSON of the C ABI (capi.h). Absent keys keep the Sampler defaults.
bool sampler_from_json(const std::string& sj, Sampler& out, std::vector<int32_t>& recent, double& seed,
                       bool& has_seed, std::string& err) {
    out = Sampler{};
    recent.clear();
    seed = -1.0;
    has_seed = false;
    double temp = 0, top_k = 0, top_p = 1, min_p = 0, presence = 0, repeat = 1;
    json_num(sj, "temp", temp);
    json_num(sj, "top_k", top_k);
    json_num(sj, "top_p", top_p);
    json_num(sj, "min_p", min_p);
    json_num(sj, "presence_penalty", presence);
    json_num(sj, "repeat_penalty", repeat);
    double sd = 0;
    if (json_num(sj, "seed", sd)) {
        seed = sd;
        has_seed = true;
    }
    if (temp < 0 || top_k < 0 || top_p <= 0 || top_p > 1) {
        err = "the sampler JSON is invalid: temp and top_k not negative, top_p in (0, 1]";
        return false;
    }
    if (min_p < 0 || min_p > 1) {
        err = "the sampler JSON is invalid: min_p in [0, 1]";
        return false;
    }
    if (!std::isfinite(presence) || !std::isfinite(repeat) || repeat < 0) {
        err = "the sampler JSON is invalid: penalties must be finite and repeat_penalty not negative";
        return false;
    }
    if (json_ints(sj, "recent", recent)) {
        if (recent.size() > 4096) {
            err = "the sampler JSON is invalid: recent is longer than 4096 ids";
            return false;
        }
    } else if (sj.find("\"recent\"") != std::string::npos) {
        err = "the sampler JSON is invalid: recent must be an array of integers";
        return false;
    }
    out.temp = static_cast<float>(temp);
    out.top_k = static_cast<size_t>(top_k);
    out.top_p = static_cast<float>(top_p);
    out.min_p = static_cast<float>(min_p);
    out.presence_penalty = static_cast<float>(presence);
    out.repeat_penalty = static_cast<float>(repeat);
    return true;
}

}  // namespace strata
