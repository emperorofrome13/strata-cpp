// port of glm/forward.rs: the GLM-5.3-Flash reference forward (Kimi Delta Attention, nope MLA with the k-pool DSA
// indexer, four-stream hyper-connections, sigmoid-routed MoE with a shared expert), with the dense matvecs and a VRAM
// expert cache on the GPU through ds_mat.cpp. Mirrors llama.cpp glm5-next.cpp, as the Rust module does.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Deviations (also in MODIFICATIONS.md, section "GLM-5.3-Flash"):
//  - rayon's `par_*` run on the model's ThreadPool; each parallel item computes the same values in the same order.
//  - The STRATA_GLM_* switches the Rust code reads per call (CHUNK, BATCH, PREFETCH, ROUTE, PROF, DUMP, PRELOAD,
//    PRELOAD_HOST) are read once at load.
//  - The context checks (`assert!` in prefill_chunk / step_opt) refuse before any state changes and report the error;
//    the limit is sc_load's `ctx` when given, else max_ctx.
//  - The tensor dims the forward reads are checked at load (the Rust matvec asserts them at run time).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>

#include "glm.h"

namespace strata {

namespace {

float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void par_for(ThreadPool* pool, size_t n, const std::function<void(size_t)>& f) {
    if (pool && n > 1) {
        pool->for_each(n, f);
    } else {
        for (size_t i = 0; i < n; i++) f(i);
    }
}

using Vec = std::vector<float>;

// port of glm/forward.rs: Hcw
struct Hcw {
    Mat f;
    Vec base;
    Vec scale;
};

// port of glm/forward.rs: Kda
struct Kda {
    Mat wq, wk, wv, wo;
    Vec conv[3];  // depthwise conv taps, [channel * conv_k + tap]
    Vec dt_b, a;
    Mat beta, f_a, f_b, g_a, g_b;
    Vec norm;
};

// port of glm/forward.rs: Dsa
struct Dsa {
    Mat q_a;
    Vec q_a_norm;
    Mat q_b, kv_a;
    Vec kv_a_norm;
    Mat k_b, v_b, wo;
    // k-pool indexer
    Mat ix_k, ix_qb, ix_proj, ix_gate;
    Vec ix_kn_w, ix_kn_b;
    Vec ix_ape;  // [kpool][head_dim]
};

// port of glm/forward.rs: Ffn (Dense / Moe)
struct Ffn {
    bool moe = false;
    Mat gate, up, down;                              // Dense
    Mat gate_inp;                                    // Moe
    Vec bias;
    Mat g, u, d, gsh, ush, dsh;
};

// port of glm/forward.rs: Layer (Mixer is the kind plus the matching struct)
struct Layer {
    Vec attn_norm, ffn_norm;
    Hcw hc_attn, hc_ffn;
    GlmKind kind = GlmKind::Kda;
    Kda kda;
    Dsa dsa;
    Ffn ffn;
};

// port of glm/forward.rs: State (recurrent / cached state of one layer)
struct State {
    Vec conv;                     // KDA: last conv_k-1 projected inputs for q, k, v; [3][tap][channel]
    Vec s;                        // KDA: [head][key channel][value channel]
    std::vector<Vec> lat;         // DSA: latent rows
    Vec ik, ig, pooled;           // DSA indexer: normed key and pool gate per position, pooled keys per pool
};

std::string env_s(const char* k) {
    const char* v = std::getenv(k);
    return v ? v : "";
}

}  // namespace

// port of glm/forward.rs: max_ctx
size_t glm_max_ctx(const GlmConfig& cfg) { return env_usize("STRATA_GLM_CTX").value_or(cfg.n_ctx_train); }

// port of glm/forward.rs: l2_norm
void glm_l2_norm(float* x, size_t n, float eps) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float s = 1.0f / std::max(std::sqrt(ss), eps);
    for (size_t i = 0; i < n; i++) x[i] *= s;
}

// port of glm/forward.rs: route_sigmoid
std::vector<std::pair<size_t, float>> route_sigmoid(const float* logits, const float* bias, size_t n, size_t n_used,
                                                    bool norm, float scale) {
    Vec probs(n);
    for (size_t i = 0; i < n; i++) probs[i] = sigmoid(logits[i]);
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        const float ka = probs[a] + bias[a], kb = probs[b] + bias[b];
        if (ka != kb) return ka > kb;
        return a < b;
    });
    idx.resize(std::min(n_used, n));
    float sum = 0.0f;
    for (size_t i : idx) sum += probs[i];
    sum = std::max(sum, 6.103515625e-5f);
    std::vector<std::pair<size_t, float>> out;
    for (size_t i : idx) out.emplace_back(i, (norm ? probs[i] / sum : probs[i]) * scale);
    return out;
}

struct GlmModel::Impl {
    GlmModel* self = nullptr;
    std::unique_ptr<ThreadPool> pool;
    DsEnv env;
    Mat tok_embd;
    Vec out_norm;
    Mat out_w;
    std::vector<Layer> layers;
    std::vector<State> state;
    // Experts routed in the previous token, per layer, to read the next layer's likely pages while this one computes.
    std::vector<std::vector<size_t>> last_routes;
    // Routed-expert counts [layer * n_expert + expert], from the prompt, used to preload the decode hot set.
    std::vector<uint32_t> route_counts;
    // The STRATA_GLM_* switches (read once at load).
    size_t prefetch = 2;
    size_t chunk = 512;
    bool batch = true;
    bool route_print = false;
    bool prof = false;
    std::string dump_dir;
    bool preload_off = false;
    size_t preload_host = 1500;

    const GlmConfig& cfg() const { return self->cfg; }

    // port of glm/forward.rs: GlmModel::tr
    void tr(const char* name, int il, const float* v, size_t n) const {
        if (!dump_dir.empty()) {
            const std::string fname = il < 0 ? dump_dir + "/" + name + ".f32"
                                             : dump_dir + "/" + name + "-" + std::to_string(il) + ".f32";
            if (FILE* f = std::fopen(fname.c_str(), "ab")) {
                std::fwrite(v, 4, n, f);
                std::fclose(f);
            }
        }
        if (self->trace && n >= 3) {
            float sum = 0.0f;
            for (size_t i = 0; i < n; i++) sum += v[i];
            std::fprintf(stderr, "TRACE %s-%d [%12.4f %12.4f %12.4f ... %12.4f %12.4f %12.4f] sum=%.6f\n", name, il,
                         (double)v[0], (double)v[1], (double)v[2], (double)v[n - 3], (double)v[n - 2], (double)v[n - 1],
                         (double)sum);
        }
    }
    void tr(const char* name, int il, const Vec& v) const { tr(name, il, v.data(), v.size()); }

    // port of glm/forward.rs: GlmModel::preload_hot. Put the prompt's hottest experts in VRAM before decoding.
    void preload_hot() {
        if (preload_off) return;
        const Ffn* first = nullptr;
        for (const Layer& l : layers)
            if (l.ffn.moe) {
                first = &l.ffn;
                break;
            }
        if (!first) return;
        const size_t n = cfg().n_expert;
        const size_t slot = (first->g.len + first->u.len + first->d.len) / n;
        const size_t resident = env.dev ? env.dev->resident() / 3 : 0;
        const size_t budget_left = env.dev ? env.dev->budget_left() : 0;
        const size_t cap = std::min<size_t>(std::max(budget_left / std::max<size_t>(slot, 1), resident), 4096);
        if (cap == 0) return;
        const size_t host_cap = preload_host;
        if (prof) std::fprintf(stderr, "PRELOAD cap %zu host %zu\n", cap, host_cap);
        std::vector<std::pair<uint32_t, size_t>> hot;
        for (size_t i = 0; i < route_counts.size(); i++)
            if (route_counts[i] > 0) hot.emplace_back(route_counts[i], i);
        std::sort(hot.begin(), hot.end());
        const size_t vram_from = hot.size() > cap ? hot.size() - cap : 0;
        const size_t host_from = vram_from > host_cap ? vram_from - host_cap : 0;
        for (size_t idx = 0; idx < hot.size(); idx++) {
            if (idx < host_from) continue;
            const size_t i = hot[idx].second, il = i / n, e = i % n;
            if (il >= layers.size() || !layers[il].ffn.moe) continue;
            const Ffn& f = layers[il].ffn;
            if (idx >= vram_from) {
                (void)f.g.expert(e, n);
                (void)f.u.expert(e, n);
                (void)f.d.expert(e, n);
            } else {
                for (const Mat* m : {&f.g, &f.u, &f.d}) m->prefetch_expert(e, n);
            }
        }
    }

    // port of glm/forward.rs: GlmModel::prefetch_experts
    void prefetch_experts(size_t il, const std::vector<size_t>& es) const {
        const Ffn& f = layers[il].ffn;
        if (!f.moe) return;
        for (size_t e : es)
            for (const Mat* m : {&f.g, &f.u, &f.d}) m->prefetch_expert(e, cfg().n_expert);
    }

    // port of glm/forward.rs: GlmModel::hc_mix
    Hc hc_mix(const Hcw& h, const Vec& x) const {
        Vec xn(x.size());
        ds_rms_norm(x.data(), x.size(), cfg().eps, xn.data());
        Vec mixes(24, 0.0f);
        h.f.matvec(xn.data(), mixes.data());
        return hc_coeffs(mixes.data(), h.scale.data(), h.base.data(), cfg().hc_eps, cfg().hc_iters);
    }

    // port of glm/forward.rs: GlmModel::kda (Kimi Delta Attention for one token)
    Vec kda(size_t il, const Vec& cur) {
        const GlmConfig& c = cfg();
        const Kda& w = layers[il].kda;
        const size_t nh = c.n_head, hd = c.kda_head_dim, ck = c.conv_k;
        const size_t di = nh * hd;
        State& st = state[il];
        if (st.conv.empty()) {
            st.conv.assign(3 * (ck - 1) * di, 0.0f);
            st.s.assign(nh * hd * hd, 0.0f);
        }
        // All six projections read `cur`: one GPU round trip instead of six.
        const Vec proj = matvec_many({w.wq, w.wk, w.wv, w.f_a, w.g_a, w.beta}, cur.data(), cur.size(), 0);
        const size_t o_fa = 3 * di, o_ga = 3 * di + hd, o_b = 3 * di + 2 * hd;
        Vec qkv[3];
        for (size_t i = 0; i < 3; i++) {
            const float* raw = proj.data() + i * di;
            Vec out(di);
            float* hist = st.conv.data() + i * (ck - 1) * di;
            for (size_t ch = 0; ch < di; ch++) {
                float acc = 0.0f;
                for (size_t t = 0; t + 1 < ck; t++) acc += w.conv[i][ch * ck + t] * hist[t * di + ch];
                acc += w.conv[i][ch * ck + ck - 1] * raw[ch];
                out[ch] = silu(acc);
            }
            std::memmove(hist, hist + di, (ck - 2) * di * sizeof(float));
            std::memcpy(hist + (ck - 2) * di, raw, di * sizeof(float));
            qkv[i] = std::move(out);
        }
        Vec& q = qkv[0];
        Vec& k = qkv[1];
        const Vec& v = qkv[2];
        for (size_t h = 0; h < nh; h++) {
            glm_l2_norm(q.data() + h * hd, hd, 1e-6f);
            glm_l2_norm(k.data() + h * hd, hd, 1e-6f);
        }
        tr("kda_q_conv", (int)il, q);
        // decay gate g = lower_bound * sigmoid(-(g1 * A)), per key channel
        Vec fab(2 * hd);
        std::memcpy(fab.data(), proj.data() + o_fa, hd * sizeof(float));
        std::memcpy(fab.data() + hd, proj.data() + o_ga, hd * sizeof(float));
        Vec fg = matvec_many({w.f_b, w.g_b}, fab.data(), fab.size(), hd);
        float* g1 = fg.data();
        const float* g2 = fg.data() + di;
        const float lb = c.gate_lower_bound;
        for (size_t ch = 0; ch < di; ch++) g1[ch] = lb * sigmoid(-((g1[ch] + w.dt_b[ch]) * w.a[ch / hd]));
        Vec beta(proj.begin() + (ptrdiff_t)o_b, proj.begin() + (ptrdiff_t)(o_b + nh));
        for (float& b : beta) b = sigmoid(b);
        // recurrence per head: S = diag(exp(g)) S; S += k (beta (v - S^T k))^T; o = S^T q / sqrt(hd)
        const float scale = 1.0f / std::sqrt((float)hd);
        Vec o(di, 0.0f);
        float* s = st.s.data();
        par_for(pool.get(), nh, [&](size_t h) {
            float* sh = s + h * hd * hd;
            float* oh = o.data() + h * hd;
            const float* qh = q.data() + h * hd;
            const float* kh = k.data() + h * hd;
            const float* vh = v.data() + h * hd;
            const float* gh = g1 + h * hd;
            Vec kv(hd, 0.0f);
            for (size_t i = 0; i < hd; i++) {
                const float e = std::exp(gh[i]);
                for (size_t j = 0; j < hd; j++) {
                    sh[i * hd + j] *= e;
                    kv[j] += sh[i * hd + j] * kh[i];
                }
            }
            Vec d(hd);
            for (size_t j = 0; j < hd; j++) d[j] = (vh[j] - kv[j]) * beta[h];
            for (size_t i = 0; i < hd; i++) {
                for (size_t j = 0; j < hd; j++) {
                    sh[i * hd + j] += kh[i] * d[j];
                    oh[j] += sh[i * hd + j] * qh[i];
                }
            }
            for (size_t j = 0; j < hd; j++) oh[j] *= scale;
        });
        tr("kda_scan_out", (int)il, o);
        // output gate: rms_norm(o) * sigmoid(g_b(g_a x))
        Vec gated(di, 0.0f);
        for (size_t h = 0; h < nh; h++) ds_rms_norm_w(o.data() + h * hd, w.norm.data(), hd, c.eps, gated.data() + h * hd);
        for (size_t ch = 0; ch < di; ch++) gated[ch] *= sigmoid(g2[ch]);
        Vec res(c.n_embd, 0.0f);
        w.wo.matvec(gated.data(), res.data());
        return res;
    }

    // port of glm/forward.rs: GlmModel::dsa (nope MLA with absorbed K/V over the latent cache; the k-pool indexer
    // selects rows once the context passes idx_top_k)
    Vec dsa(size_t il, const Vec& cur) {
        const GlmConfig& c = cfg();
        const Dsa& w = layers[il].dsa;
        const size_t nh = c.n_head, kd = c.k_mla, kl = c.kv_lora;
        // q_a, kv_a and the two indexer projections all read `cur`: one GPU round trip instead of four.
        const size_t hd = c.idx_head_dim, nih = c.idx_n_head, kp = c.kpool;
        const Vec proj = matvec_many({w.q_a, w.kv_a, w.ix_k, w.ix_gate}, cur.data(), cur.size(), 0);
        const size_t o_kv = c.q_lora, o_ik = c.q_lora + kl, o_ig = c.q_lora + kl + hd;
        Vec qr(c.q_lora, 0.0f);
        ds_rms_norm_w(proj.data(), w.q_a_norm.data(), c.q_lora, c.eps, qr.data());
        Vec q(nh * kd, 0.0f);
        w.q_b.matvec(qr.data(), q.data());
        Vec lat(kl, 0.0f);
        ds_rms_norm_w(proj.data() + o_kv, w.kv_a_norm.data(), kl, c.eps, lat.data());
        tr("kv_cmpr", (int)il, lat);
        // indexer key (layer norm) and pool gate of this position
        Vec ik(proj.begin() + (ptrdiff_t)o_ik, proj.begin() + (ptrdiff_t)(o_ik + hd));
        float msum = 0.0f;
        for (float x : ik) msum += x;
        const float mean = msum / (float)hd;
        float vsum = 0.0f;
        for (float x : ik) vsum += (x - mean) * (x - mean);
        const float var = vsum / (float)hd;
        const float inv = 1.0f / std::sqrt(var + c.kda_eps);
        for (size_t i = 0; i < hd; i++) ik[i] = (ik[i] - mean) * inv * w.ix_kn_w[i] + w.ix_kn_b[i];
        const float* ig = proj.data() + o_ig;
        State& st = state[il];
        st.lat.push_back(std::move(lat));
        st.ik.insert(st.ik.end(), ik.begin(), ik.end());
        st.ig.insert(st.ig.end(), ig, ig + hd);
        const size_t pos = st.lat.size() - 1;
        if ((pos + 1) % kp == 0) {
            // pooled key = sum over the pool of softmax_k(gate + ape) * key, per dimension
            const size_t b0 = (pos + 1 - kp) * hd;
            Vec l(kp), e(kp);
            for (size_t d = 0; d < hd; d++) {
                for (size_t k = 0; k < kp; k++) l[k] = st.ig[b0 + k * hd + d] + w.ix_ape[k * hd + d];
                float mx = -std::numeric_limits<float>::infinity();
                for (float x : l) mx = std::max(mx, x);
                float z = 0.0f;
                for (size_t k = 0; k < kp; k++) {
                    e[k] = std::exp(l[k] - mx);
                    z += e[k];
                }
                float acc = 0.0f;
                for (size_t k = 0; k < kp; k++) acc += e[k] / z * st.ik[b0 + k * hd + d];
                st.pooled.push_back(acc);
            }
        }
        // sparse selection: only beyond idx_top_k rows; the top idx_top_k/kpool pools by indexer score plus the
        // incomplete tail
        std::vector<size_t> rows;
        if (pos < c.idx_top_k) {
            rows.resize(pos + 1);
            for (size_t i = 0; i <= pos; i++) rows[i] = i;
        } else {
            Vec iq(nih * hd, 0.0f);
            w.ix_qb.matvec(qr.data(), iq.data());
            Vec wt(nih, 0.0f);
            w.ix_proj.matvec(cur.data(), wt.data());
            const float sc = 1.0f / std::sqrt((float)(hd * nih));
            const size_t nv = (pos + 1) / kp;
            const Vec& pooled = st.pooled;
            Vec score(nv, 0.0f);
            par_for(pool.get(), nv, [&](size_t j) {
                float tot = 0.0f;
                for (size_t h = 0; h < nih; h++) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < hd; d++) dot += iq[h * hd + d] * pooled[j * hd + d];
                    tot += std::max(dot, 0.0f) * wt[h] * sc;
                }
                score[j] = tot;
            });
            std::vector<size_t> order(nv);
            for (size_t j = 0; j < nv; j++) order[j] = j;
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                if (score[a] != score[b]) return score[a] > score[b];
                return a < b;
            });
            order.resize(std::min(order.size(), c.idx_top_k / kp));
            std::sort(order.begin(), order.end());
            for (size_t j : order)
                for (size_t r = j * kp; r < (j + 1) * kp; r++) rows.push_back(r);
            for (size_t r = nv * kp; r <= pos; r++) rows.push_back(r);
        }
        const std::vector<Vec>& cache = st.lat;
        const float scale = 1.0f / std::sqrt((float)kd);
        std::vector<Mat> kb;
        for (size_t h = 0; h < nh; h++) kb.push_back(w.k_b.group(h, nh));
        const Vec qabs = matvec_many(kb, q.data(), q.size(), kd);
        Vec accs(nh * kl, 0.0f);
        par_for(pool.get(), nh, [&](size_t h) {
            const float* qa = qabs.data() + h * kl;
            Vec sc(rows.size());
            for (size_t r = 0; r < rows.size(); r++) {
                const Vec& row = cache[rows[r]];
                float dot = 0.0f;
                for (size_t j = 0; j < kl; j++) dot += row[j] * qa[j];
                sc[r] = dot * scale;
            }
            float mx = -std::numeric_limits<float>::infinity();
            for (float x : sc) mx = std::max(mx, x);
            float z = 0.0f;
            for (float& x : sc) {
                x = std::exp(x - mx);
                z += x;
            }
            float* acc = accs.data() + h * kl;
            for (size_t r = 0; r < rows.size(); r++) {
                const float p = sc[r] / z;
                const Vec& row = cache[rows[r]];
                for (size_t j = 0; j < kl; j++) acc[j] += p * row[j];
            }
        });
        std::vector<Mat> vb;
        for (size_t h = 0; h < nh; h++) vb.push_back(w.v_b.group(h, nh));
        const Vec out = matvec_many(vb, accs.data(), accs.size(), kl);
        Vec res(c.n_embd, 0.0f);
        w.wo.matvec(out.data(), res.data());
        return res;
    }

    // port of glm/forward.rs: GlmModel::ffn
    Vec ffn(size_t il, const Vec& cur) {
        const GlmConfig& c = cfg();
        const size_t n = c.n_embd;
        Vec out(n, 0.0f);
        const Ffn& f = layers[il].ffn;
        if (!f.moe) {
            Vec g(c.n_ff, 0.0f), u(c.n_ff, 0.0f);
            f.gate.matvec(cur.data(), g.data());
            f.up.matvec(cur.data(), u.data());
            for (size_t j = 0; j < c.n_ff; j++) g[j] = silu(g[j]) * u[j];
            f.down.matvec(g.data(), out.data());
            return out;
        }
        if (prefetch > 0) {
            for (size_t j = 1; j <= prefetch; j++) {
                if (il + j < last_routes.size() && !last_routes[il + j].empty()) prefetch_experts(il + j, last_routes[il + j]);
            }
        }
        Vec logits(c.n_expert, 0.0f);
        f.gate_inp.matvec(cur.data(), logits.data());
        tr("ffn_moe_logits", (int)il, logits);
        const auto sel = route_sigmoid(logits.data(), f.bias.data(), c.n_expert, c.n_used, c.weights_norm, c.weights_scale);
        Vec ids;
        for (const auto& s : sel) ids.push_back((float)s.first);
        tr("ffn_moe_topk", (int)il, ids);
        last_routes[il].clear();
        for (const auto& s : sel) last_routes[il].push_back(s.first);
        for (const auto& s : sel) route_counts[il * c.n_expert + s.first] += 1;
        if (route_print) {
            std::string line;
            for (size_t i = 0; i < sel.size(); i++) line += (i ? "," : "") + std::to_string(sel[i].first);
            std::fprintf(stderr, "ROUTE %zu %s\n", il, line.c_str());
        }
        for (const auto& s : sel)
            for (const Mat* m : {&f.g, &f.u, &f.d}) m->prefetch_expert(s.first, c.n_expert);
        const size_t nf = c.n_ff_exp;
        // One GPU round trip for all gate/up experts, one for all down experts, instead of 3 per expert.
        std::vector<Mat> gu, dm;
        for (const auto& s : sel) {
            gu.push_back(f.g.expert(s.first, c.n_expert));
            gu.push_back(f.u.expert(s.first, c.n_expert));
        }
        for (const auto& s : sel) dm.push_back(f.d.expert(s.first, c.n_expert));
        const Vec gu_y = matvec_many(gu, cur.data(), cur.size(), 0);
        const size_t k = sel.size();
        Vec acts(k * nf, 0.0f);
        for (size_t i = 0; i < k; i++)
            for (size_t j = 0; j < nf; j++)
                acts[i * nf + j] = swiglu_clamp(gu_y[2 * i * nf + j], gu_y[(2 * i + 1) * nf + j], c.clamp_exp[il]);
        const Vec dy = matvec_many(dm, acts.data(), acts.size(), nf);
        for (size_t i = 0; i < k; i++) {
            const float wgt = sel[i].second;
            for (size_t j = 0; j < n; j++) out[j] += wgt * dy[i * n + j];
        }
        tr("ffn_moe_out", (int)il, out);
        // shared expert: plain silu (llama.cpp builds it with build_ffn, no clamp)
        const size_t fs = c.n_ff_exp * c.n_shared;
        Vec g(fs, 0.0f), u(fs, 0.0f);
        f.gsh.matvec(cur.data(), g.data());
        f.ush.matvec(cur.data(), u.data());
        for (size_t j = 0; j < fs; j++) g[j] = silu(g[j]) * u[j];
        Vec d(n, 0.0f);
        f.dsh.matvec(g.data(), d.data());
        tr("ffn_shexp", (int)il, d);
        for (size_t j = 0; j < n; j++) out[j] += d[j];
        return out;
    }

    // port of glm/forward.rs: GlmModel::ffn_batch. FFN for several tokens of one layer: each routed expert is loaded
    // once and applied to every token that picked it. Per-token numerics equal `ffn` (expert outputs summed in the
    // token's own selection order).
    std::vector<Vec> ffn_batch(size_t il, const std::vector<Vec>& curs) {
        const GlmConfig& c = cfg();
        std::vector<const float*> xs;
        for (const Vec& v : curs) xs.push_back(v.data());
        const Ffn& f = layers[il].ffn;
        if (!f.moe) {
            if (!batch) {
                std::vector<Vec> r;
                for (const Vec& cu : curs) r.push_back(ffn(il, cu));
                return r;
            }
            const size_t n = c.n_embd, nf = c.n_ff;
            const Vec gy = matvec_rows(f.gate, xs);
            const Vec uy = matvec_rows(f.up, xs);
            Vec g(curs.size() * nf, 0.0f);
            for (size_t j = 0; j < g.size(); j++) g[j] = silu(gy[j]) * uy[j];
            std::vector<const float*> gs;
            for (size_t t = 0; t < curs.size(); t++) gs.push_back(g.data() + t * nf);
            const Vec dy = matvec_rows(f.down, gs);
            std::vector<Vec> r;
            for (size_t t = 0; t < curs.size(); t++) r.emplace_back(dy.begin() + (ptrdiff_t)(t * n), dy.begin() + (ptrdiff_t)((t + 1) * n));
            return r;
        }
        const size_t n = c.n_embd, nf = c.n_ff_exp;
        std::vector<std::vector<std::pair<size_t, float>>> sels;
        if (batch) {
            const Vec logits = matvec_rows(f.gate_inp, xs);
            for (size_t t = 0; t < curs.size(); t++)
                sels.push_back(route_sigmoid(logits.data() + t * c.n_expert, f.bias.data(), c.n_expert, c.n_used,
                                             c.weights_norm, c.weights_scale));
        } else {
            for (const Vec& cu : curs) {
                Vec logits(c.n_expert, 0.0f);
                f.gate_inp.matvec(cu.data(), logits.data());
                sels.push_back(route_sigmoid(logits.data(), f.bias.data(), c.n_expert, c.n_used, c.weights_norm,
                                             c.weights_scale));
            }
        }
        for (const auto& sel : sels)
            for (const auto& s : sel) route_counts[il * c.n_expert + s.first] += 1;
        std::map<size_t, std::vector<std::pair<size_t, size_t>>> by_e;
        for (size_t t = 0; t < sels.size(); t++)
            for (size_t k = 0; k < sels[t].size(); k++) by_e[sels[t][k].first].emplace_back(t, k);
        for (const auto& kv : by_e)
            for (const Mat* m : {&f.g, &f.u, &f.d}) m->prefetch_expert(kv.first, c.n_expert);
        std::vector<std::vector<Vec>> ds(sels.size());
        for (size_t t = 0; t < sels.size(); t++) ds[t].resize(sels[t].size());
        for (const auto& [e, uses] : by_e) {
            const Mat me_g = f.g.expert(e, c.n_expert);
            const Mat me_u = f.u.expert(e, c.n_expert);
            const Mat me_d = f.d.expert(e, c.n_expert);
            if (!batch) {
                Vec g(nf, 0.0f), u(nf, 0.0f);
                for (const auto& [t, k] : uses) {
                    me_g.matvec(curs[t].data(), g.data());
                    me_u.matvec(curs[t].data(), u.data());
                    for (size_t j = 0; j < nf; j++) g[j] = swiglu_clamp(g[j], u[j], c.clamp_exp[il]);
                    Vec d(n, 0.0f);
                    me_d.matvec(g.data(), d.data());
                    ds[t][k] = std::move(d);
                }
                continue;
            }
            std::vector<const float*> uxs;
            for (const auto& tk : uses) uxs.push_back(curs[tk.first].data());
            const Vec gy = matvec_rows(me_g, uxs);
            const Vec uy = matvec_rows(me_u, uxs);
            Vec g(uses.size() * nf, 0.0f);
            for (size_t j = 0; j < g.size(); j++) g[j] = swiglu_clamp(gy[j], uy[j], c.clamp_exp[il]);
            std::vector<const float*> gs;
            for (size_t i = 0; i < uses.size(); i++) gs.push_back(g.data() + i * nf);
            const Vec dy = matvec_rows(me_d, gs);
            for (size_t i = 0; i < uses.size(); i++)
                ds[uses[i].first][uses[i].second] = Vec(dy.begin() + (ptrdiff_t)(i * n), dy.begin() + (ptrdiff_t)((i + 1) * n));
        }
        const size_t fs = nf * c.n_shared;
        Vec shared;
        if (batch) {
            const Vec sg = matvec_rows(f.gsh, xs);
            const Vec su = matvec_rows(f.ush, xs);
            Vec g(curs.size() * fs, 0.0f);
            for (size_t j = 0; j < g.size(); j++) g[j] = silu(sg[j]) * su[j];
            std::vector<const float*> gs;
            for (size_t t = 0; t < curs.size(); t++) gs.push_back(g.data() + t * fs);
            shared = matvec_rows(f.dsh, gs);
        }
        std::vector<Vec> res;
        for (size_t t = 0; t < curs.size(); t++) {
            Vec out(n, 0.0f);
            for (size_t k = 0; k < sels[t].size(); k++) {
                const float wgt = sels[t][k].second;
                for (size_t j = 0; j < n; j++) out[j] += wgt * ds[t][k][j];
            }
            if (batch) {
                for (size_t j = 0; j < n; j++) out[j] += shared[t * n + j];
            } else {
                Vec g(fs, 0.0f), u(fs, 0.0f);
                f.gsh.matvec(curs[t].data(), g.data());
                f.ush.matvec(curs[t].data(), u.data());
                for (size_t j = 0; j < fs; j++) g[j] = silu(g[j]) * u[j];
                Vec d(n, 0.0f);
                f.dsh.matvec(g.data(), d.data());
                for (size_t j = 0; j < n; j++) out[j] += d[j];
            }
            res.push_back(std::move(out));
        }
        return res;
    }

    // The 4 hyper-connection streams of token `t`'s embedding.
    Vec embed4(uint32_t t) const {
        const size_t n = cfg().n_embd;
        Vec emb(n, 0.0f);
        tok_embd.row_into(t, emb.data());
        Vec x(4 * n);
        for (size_t s = 0; s < 4; s++) std::memcpy(x.data() + s * n, emb.data(), n * sizeof(float));
        return x;
    }

    // port of glm/forward.rs: GlmModel::prefill_chunk (layer-major pass over a chunk of prompt tokens, no logits)
    void prefill_chunk(const uint32_t* tk, size_t nt) {
        const GlmConfig& c = cfg();
        const size_t n = c.n_embd;
        self->toks.insert(self->toks.end(), tk, tk + nt);
        std::vector<Vec> xs;
        for (size_t i = 0; i < nt; i++) xs.push_back(embed4(tk[i]));
        Vec cur(n), nrm(n);
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t il = 0; il < c.n_layer; il++) {
            for (Vec& x : xs) {
                const Hc a = hc_mix(layers[il].hc_attn, x);
                hc_collapse(x.data(), a.pre, cur.data(), n);
                ds_rms_norm_w(cur.data(), layers[il].attn_norm.data(), n, c.eps, nrm.data());
                const Vec att = c.kind[il] == GlmKind::Kda ? kda(il, nrm) : dsa(il, nrm);
                Vec x2(4 * n);
                hc_post(att.data(), x.data(), a, x2.data(), n);
                x = std::move(x2);
            }
            std::vector<Hc> fs;
            std::vector<Vec> nrms;
            for (const Vec& x : xs) {
                const Hc f = hc_mix(layers[il].hc_ffn, x);
                hc_collapse(x.data(), f.pre, cur.data(), n);
                ds_rms_norm_w(cur.data(), layers[il].ffn_norm.data(), n, c.eps, nrm.data());
                nrms.push_back(nrm);
                fs.push_back(f);
            }
            const std::vector<Vec> ffs = ffn_batch(il, nrms);
            for (size_t i = 0; i < xs.size(); i++) {
                Vec x2(4 * n);
                hc_post(ffs[i].data(), xs[i].data(), fs[i], x2.data(), n);
                xs[i] = std::move(x2);
            }
        }
        if (prof) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "PROF chunk %zu toks, now %zu total %.0fms\n", nt, self->toks.size(), ms);
        }
    }

    // port of glm/forward.rs: GlmModel::step_opt
    Vec step_opt(uint32_t tok, bool want) {
        const GlmConfig& c = cfg();
        const size_t n = c.n_embd;
        self->toks.push_back(tok);
        Vec x = embed4(tok);
        tr("hc_init", -1, x);
        Vec cur(n), nrm(n);
        double prof_t[2] = {0, 0};
        const auto tstep = std::chrono::steady_clock::now();
        for (size_t il = 0; il < c.n_layer; il++) {
            const int li = (int)il;
            const Hc a = hc_mix(layers[il].hc_attn, x);
            hc_collapse(x.data(), a.pre, cur.data(), n);
            tr("hc_attn_pre", li, cur);
            ds_rms_norm_w(cur.data(), layers[il].attn_norm.data(), n, c.eps, nrm.data());
            tr("attn_norm", li, nrm);
            auto t0 = std::chrono::steady_clock::now();
            const Vec att = c.kind[il] == GlmKind::Kda ? kda(il, nrm) : dsa(il, nrm);
            prof_t[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            tr("attn_out", li, att);
            Vec x2(4 * n);
            hc_post(att.data(), x.data(), a, x2.data(), n);
            x = std::move(x2);
            tr("hc_attn_post", li, x);
            const Hc f = hc_mix(layers[il].hc_ffn, x);
            hc_collapse(x.data(), f.pre, cur.data(), n);
            tr("hc_ffn_pre", li, cur);
            ds_rms_norm_w(cur.data(), layers[il].ffn_norm.data(), n, c.eps, nrm.data());
            tr("ffn_norm", li, nrm);
            t0 = std::chrono::steady_clock::now();
            const Vec ff = ffn(il, nrm);
            prof_t[1] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            tr("ffn_out", li, ff);
            Vec x3(4 * n);
            hc_post(ff.data(), x.data(), f, x3.data(), n);
            x = std::move(x3);
            tr("l_out", li, x);
        }
        if (prof) {
            uint64_t h = 0, m = 0;
            if (env.dev) {
                h = env.dev->lookups_hit.exchange(0);
                m = env.dev->lookups_miss.exchange(0);
            }
            const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tstep).count();
            std::fprintf(stderr, "PROF tok%zu total %.0fms attn %.0fms ffn %.0fms expert-cache hit %llu miss %llu gpu %s\n",
                         self->toks.size(), total, prof_t[0] * 1e3, prof_t[1] * 1e3, (unsigned long long)h,
                         (unsigned long long)m,
                         env.dev ? ("Some((\"" + std::string(env.dev->backend_name()) + "\", " +
                                    std::to_string(env.dev->resident()) + ", " + std::to_string(env.dev->bytes()) + "))")
                                       .c_str()
                                 : "None");
        }
        if (!want) return {};
        hc_collapse(x.data(), {0.25f, 0.25f, 0.25f, 0.25f}, cur.data(), n);
        tr("hc_head", -1, cur);
        ds_rms_norm_w(cur.data(), out_norm.data(), n, c.eps, nrm.data());
        tr("result_norm", -1, nrm);
        Vec logits(c.n_vocab, 0.0f);
        out_w.matvec(nrm.data(), logits.data());
        tr("result_output", -1, logits);
        return logits;
    }
};

GlmModel::~GlmModel() = default;

namespace {

// StrataFreet addition: the dims of every tensor the forward reads, checked at load (the Rust matvec asserts them at
// run time; here a mismatch would read past a buffer). Tensors the file lacks are reported by Mat::load.
bool check_dims(const Gguf& g, const GlmConfig& c, std::string& err) {
    for (const auto& [name, dims] : c.expected_tensors()) {
        if (name.rfind("blk.", 0) == 0) {
            const size_t il = (size_t)std::strtoull(name.c_str() + 4, nullptr, 10);
            if (il >= c.n_layer) continue;
        }
        const auto it = g.tensors().find(name);
        if (it == g.tensors().end()) continue;
        std::vector<uint64_t> a = it->second.dims, b = dims;
        while (a.size() > 1 && a.back() == 1) a.pop_back();
        while (b.size() > 1 && b.back() == 1) b.pop_back();
        if (a != b) {
            err = name + ": dims do not match the glm5-next hyperparameters";
            return false;
        }
    }
    return true;
}

}  // namespace

// port of glm/forward.rs: GlmModel::new
bool GlmModel::create(const Gguf& g, const GlmLoad& opt, std::unique_ptr<GlmModel>& out, std::string& err) {
    std::unique_ptr<GlmModel> me(new GlmModel());
    if (!GlmConfig::from_gguf(g, me->cfg, err)) return false;
    if (!check_dims(g, me->cfg, err)) return false;
    const GlmConfig& cfg = me->cfg;
    auto im = std::make_unique<Impl>();
    im->self = me.get();
    im->pool = std::make_unique<ThreadPool>(opt.threads ? opt.threads : default_threads());
    im->env.pool = im->pool.get();
    im->env.read_switches();
    // dev::get runs on the first Mat::load: open the GPUs before any weight is read.
    im->env.dev = Dev::open(opt.gpu, opt.open_backend, me->gpu_note_);
    if (!im->env.dev && opt.gpu.on) std::fprintf(stderr, "[strata-cpp] glm5-next: %s\n", me->gpu_note_.c_str());
    DsEnv* env = &im->env;
    try {
        for (size_t il = 0; il < cfg.n_layer; il++) {
            const std::string pre = "blk." + std::to_string(il) + ".";
            auto m = [&](const char* n, Mat& o) { return Mat::load(g, env, pre + n, o, err); };
            auto v = [&](const char* n, Vec& o) { return vec_f32(g, env, pre + n, o, err); };
            Layer L;
            L.kind = cfg.kind[il];
            // Rust evaluates the mixer, then the FFN, then the norms and hyper-connections (the upload order).
            if (L.kind == GlmKind::Kda) {
                Kda& w = L.kda;
                if (!m("attn_q.weight", w.wq) || !m("attn_k.weight", w.wk) || !m("attn_v.weight", w.wv) ||
                    !m("attn_output.weight", w.wo) || !v("ssm_conv1d_q.weight", w.conv[0]) ||
                    !v("ssm_conv1d_k.weight", w.conv[1]) || !v("ssm_conv1d_v.weight", w.conv[2]) ||
                    !v("ssm_dt.bias", w.dt_b) || !v("ssm_a", w.a) || !m("ssm_beta.weight", w.beta) ||
                    !m("ssm_f_a.weight", w.f_a) || !m("ssm_f_b.weight", w.f_b) || !m("ssm_g_a.weight", w.g_a) ||
                    !m("ssm_g_b.weight", w.g_b) || !v("ssm_norm.weight", w.norm))
                    return false;
            } else {
                Dsa& w = L.dsa;
                if (!m("attn_q_a.weight", w.q_a) || !v("attn_q_a_norm.weight", w.q_a_norm) ||
                    !m("attn_q_b.weight", w.q_b) || !m("attn_kv_a_mqa.weight", w.kv_a) ||
                    !v("attn_kv_a_norm.weight", w.kv_a_norm) || !m("attn_k_b.weight", w.k_b) ||
                    !m("attn_v_b.weight", w.v_b) || !m("attn_output.weight", w.wo) ||
                    !m("indexer.attn_k.weight", w.ix_k) || !m("indexer.attn_q_b.weight", w.ix_qb) ||
                    !m("indexer.proj.weight", w.ix_proj) || !m("indexer_compressor_gate.weight", w.ix_gate) ||
                    !v("indexer.k_norm.weight", w.ix_kn_w) || !v("indexer.k_norm.bias", w.ix_kn_b) ||
                    !v("indexer_compressor_ape.weight", w.ix_ape))
                    return false;
            }
            Ffn& f = L.ffn;
            if (il < cfg.n_dense) {
                f.moe = false;
                if (!m("ffn_gate.weight", f.gate) || !m("ffn_up.weight", f.up) || !m("ffn_down.weight", f.down)) return false;
            } else {
                f.moe = true;
                if (!m("ffn_gate_inp.weight", f.gate_inp) || !v("exp_probs_b.bias", f.bias) ||
                    !m("ffn_gate_exps.weight", f.g) || !m("ffn_up_exps.weight", f.u) || !m("ffn_down_exps.weight", f.d) ||
                    !m("ffn_gate_shexp.weight", f.gsh) || !m("ffn_up_shexp.weight", f.ush) ||
                    !m("ffn_down_shexp.weight", f.dsh))
                    return false;
            }
            auto hcw = [&](const char* s, Hcw& h) {
                const std::string b = std::string("hc_") + s;
                return m((b + "_fn.weight").c_str(), h.f) && v((b + "_base.weight").c_str(), h.base) &&
                       v((b + "_scale.weight").c_str(), h.scale);
            };
            if (!v("attn_norm.weight", L.attn_norm) || !v("ffn_norm.weight", L.ffn_norm) || !hcw("attn", L.hc_attn) ||
                !hcw("ffn", L.hc_ffn))
                return false;
            if (L.hc_attn.base.size() < 24 || L.hc_attn.scale.size() < 3 || L.hc_ffn.base.size() < 24 ||
                L.hc_ffn.scale.size() < 3) {
                err = pre + "hc_*: unexpected hyper-connection sizes";
                return false;
            }
            im->layers.push_back(std::move(L));
        }
        if (!Mat::load(g, env, "token_embd.weight", im->tok_embd, err) ||
            !vec_f32(g, env, "output_norm.weight", im->out_norm, err) ||
            !Mat::load(g, env, "output.weight", im->out_w, err))
            return false;
    } catch (const std::exception& e) {
        err = std::string("glm5-next load: ") + e.what();
        return false;
    }
    const size_t n = cfg.n_layer;
    im->state.resize(n);
    im->last_routes.assign(n, {});
    im->route_counts.assign(n * cfg.n_expert, 0);
    me->trace = std::getenv("STRATA_GLM_TRACE") != nullptr;
    im->prefetch = glm_prefetch_default();
    im->chunk = env_usize("STRATA_GLM_CHUNK").value_or(512);
    im->batch = env_s("STRATA_GLM_BATCH") != "0";
    im->route_print = std::getenv("STRATA_GLM_ROUTE") != nullptr;
    im->prof = std::getenv("STRATA_GLM_PROF") != nullptr;
    im->dump_dir = env_s("STRATA_GLM_DUMP");
    im->preload_off = env_s("STRATA_GLM_PRELOAD") == "0";
    im->preload_host = env_usize("STRATA_GLM_PRELOAD_HOST").value_or(1500);
    me->ctx_ = opt.ctx ? opt.ctx : glm_max_ctx(cfg);
    me->m_ = std::move(im);
    out = std::move(me);
    return true;
}

// port of glm/forward.rs: GlmModel::reset
void GlmModel::reset() {
    for (State& s : m_->state) s = State();
    toks.clear();
    for (auto& r : m_->last_routes) r.clear();
    std::fill(m_->route_counts.begin(), m_->route_counts.end(), 0u);
    broken_ = false;
}

// port of glm/forward.rs: GlmModel::step
bool GlmModel::step(uint32_t tok, std::vector<float>& logits, std::string& err) {
    if (broken_) {
        err = "glm5-next: an earlier call failed mid-forward; sc_reset first";
        return false;
    }
    if (toks.size() >= ctx_) {
        err = "GLM context is limited to " + std::to_string(ctx_) + " tokens";
        return false;
    }
    try {
        logits = m_->step_opt(tok, true);
    } catch (const std::exception& e) {
        broken_ = true;
        err = std::string("glm5-next: ") + e.what();
        return false;
    }
    return true;
}

// port of glm/forward.rs: GlmModel::prefill
bool GlmModel::prefill(const std::vector<uint32_t>& ids, std::vector<float>& logits, std::string& err) {
    logits.clear();
    if (ids.empty()) return true;
    if (broken_) {
        err = "glm5-next: an earlier call failed mid-forward; sc_reset first";
        return false;
    }
    if (toks.size() + ids.size() > ctx_) {
        err = "GLM context is limited to " + std::to_string(ctx_) + " tokens";
        return false;
    }
    try {
        const size_t head = ids.size() - 1;
        const size_t chunk = m_->chunk;
        if (chunk > 1) {
            for (size_t i = 0; i < head; i += chunk) m_->prefill_chunk(ids.data() + i, std::min(chunk, head - i));
        } else {
            for (size_t i = 0; i < head; i++) m_->step_opt(ids[i], false);
        }
        logits = m_->step_opt(ids[head], true);
        m_->preload_hot();
    } catch (const std::exception& e) {
        broken_ = true;
        err = std::string("glm5-next: ") + e.what();
        return false;
    }
    return true;
}

std::string GlmModel::backend_name() const { return m_->env.dev ? m_->env.dev->backend_name() : "cpu"; }
std::string GlmModel::device_name() const { return m_->env.dev ? m_->env.dev->device_name() : "cpu"; }
size_t GlmModel::threads() const { return m_->pool ? m_->pool->size() : 0; }
size_t GlmModel::resident_experts() const { return m_->env.dev ? m_->env.dev->resident() : 0; }
size_t GlmModel::gpu_bytes() const { return m_->env.dev ? m_->env.dev->bytes() : 0; }
uint64_t GlmModel::hits() const { return m_->env.dev ? m_->env.dev->total_hit : 0; }
uint64_t GlmModel::misses() const { return m_->env.dev ? m_->env.dev->total_miss : 0; }

}  // namespace strata
