// port of deepseek/ops.rs (the parts glm/forward.rs uses). See ds_ops.h.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Summation order follows the Rust iterators (`iter().sum()` is a left fold), so the f32 results match.
#include "ds_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strata {

namespace {

// port of deepseek/ops.rs: sigmoid
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

// port of deepseek/ops.rs: rms_norm
void ds_rms_norm(const float* x, size_t n, float eps, float* out) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float ms = ss / (float)n;
    const float s = 1.0f / std::sqrt(ms + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * s;
}

// port of deepseek/ops.rs: rms_norm_w
void ds_rms_norm_w(const float* x, const float* w, size_t n, float eps, float* out) {
    ds_rms_norm(x, n, eps, out);
    for (size_t i = 0; i < n; i++) out[i] *= w[i];
}

// port of deepseek/ops.rs: hc_coeffs
Hc hc_coeffs(const float* mixes, const float* scale, const float* base, float hc_eps, size_t iters) {
    Hc h;
    for (size_t i = 0; i < 4; i++) {
        h.pre[i] = sigmoid(mixes[i] * scale[0] + base[i]) + hc_eps;
        h.post[i] = 2.0f * sigmoid(mixes[4 + i] * scale[1] + base[4 + i]);
    }
    std::array<float, 16>& m = h.comb;
    for (size_t i = 0; i < 16; i++) m[i] = mixes[8 + i] * scale[2] + base[8 + i];
    // softmax over d for each s, then eps
    for (size_t s = 0; s < 4; s++) {
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t d = 0; d < 4; d++) mx = std::max(mx, m[d + 4 * s]);
        float sum = 0.0f;
        for (size_t d = 0; d < 4; d++) {
            m[d + 4 * s] = std::exp(m[d + 4 * s] - mx);
            sum += m[d + 4 * s];
        }
        for (size_t d = 0; d < 4; d++) m[d + 4 * s] = m[d + 4 * s] / sum + hc_eps;
    }
    // norm_cols: M[d][s] /= sum_s M[d][s] + eps
    auto norm_cols = [&]() {
        for (size_t d = 0; d < 4; d++) {
            float t = 0.0f;
            for (size_t s = 0; s < 4; s++) t += m[d + 4 * s];
            t += hc_eps;
            for (size_t s = 0; s < 4; s++) m[d + 4 * s] /= t;
        }
    };
    auto norm_rows = [&]() {
        for (size_t s = 0; s < 4; s++) {
            float t = 0.0f;
            for (size_t d = 0; d < 4; d++) t += m[d + 4 * s];
            t += hc_eps;
            for (size_t d = 0; d < 4; d++) m[d + 4 * s] /= t;
        }
    };
    norm_cols();
    for (size_t it = 1; it < iters; it++) {
        norm_rows();
        norm_cols();
    }
    return h;
}

// port of deepseek/ops.rs: hc_collapse
void hc_collapse(const float* x, const std::array<float, 4>& w, float* out, size_t n) {
    for (size_t j = 0; j < n; j++) {
        float a = 0.0f;
        for (size_t s = 0; s < 4; s++) a += x[s * n + j] * w[s];
        out[j] = a;
    }
}

// port of deepseek/ops.rs: hc_post
void hc_post(const float* f, const float* residual, const Hc& hc, float* out, size_t n) {
    for (size_t d = 0; d < 4; d++) {
        for (size_t j = 0; j < n; j++) {
            float a = f[j] * hc.post[d];
            for (size_t s = 0; s < 4; s++) a += residual[s * n + j] * hc.comb[d + 4 * s];
            out[d * n + j] = a;
        }
    }
}

// port of deepseek/ops.rs: swiglu_clamp
float swiglu_clamp(float gate, float up, float limit) {
    const float g = std::min(gate, limit);
    return g / (1.0f + std::exp(-g)) * std::clamp(up, -limit, limit);
}

}  // namespace strata
