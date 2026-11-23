// port of deepseek/ops.rs: the small CPU f32 building blocks the GLM-5.3-Flash forward (glm/forward.rs) uses -
// RMS norm, the hyper-connection coefficients (Sinkhorn), collapse and post, and ggml's swiglu_clamp.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md). The formulas mirror llama.cpp (MIT License, ggml authors) deepseek41.cpp /
// glm5-next.cpp, as the Rust module does.
//
// Not ported (DeepSeek-V4.1 only, no GLM caller): `Rope` and `route` (the sqrt-softplus router).
#pragma once

#include <array>
#include <cstddef>

namespace strata {

// port of deepseek/ops.rs: rms_norm. RMS norm without weight: x / sqrt(mean(x^2) + eps). `n` values.
void ds_rms_norm(const float* x, size_t n, float eps, float* out);

// port of deepseek/ops.rs: rms_norm_w. RMS norm then scale by weight.
void ds_rms_norm_w(const float* x, const float* w, size_t n, float eps, float* out);

// port of deepseek/ops.rs: Hc (hyper-connection coefficients for one sublayer). comb[d + 4*s].
struct Hc {
    std::array<float, 4> pre{};
    std::array<float, 4> post{};
    std::array<float, 16> comb{};
};

// port of deepseek/ops.rs: hc_coeffs. mixes: 24 values (hc_fn @ rms(flat x)); scale: 3; base: 24.
Hc hc_coeffs(const float* mixes, const float* scale, const float* base, float hc_eps, size_t iters);

// port of deepseek/ops.rs: hc_collapse. out[j] = sum_s x[s * n + j] * w[s]; x is 4 streams of n each.
void hc_collapse(const float* x, const std::array<float, 4>& w, float* out, size_t n);

// port of deepseek/ops.rs: hc_post. out[d * n + j] = f[j] * post[d] + sum_s residual[s * n + j] * comb[d + 4s].
void hc_post(const float* f, const float* residual, const Hc& hc, float* out, size_t n);

// port of deepseek/ops.rs: swiglu_clamp (ggml swiglu_clamp).
float swiglu_clamp(float gate, float up, float limit);

}  // namespace strata
