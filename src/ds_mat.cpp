// port of deepseek/mat.rs. See ds_mat.h for the deviations.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "ds_mat.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata {

namespace {

size_t sat_sub(size_t a, size_t b) { return a > b ? a - b : 0; }

// rayon's `par_*` over `n` items on the model's pool, or inline without one.
void par_for(ThreadPool* pool, size_t n, const std::function<void(size_t)>& f) {
    if (pool && n > 1) {
        pool->for_each(n, f);
    } else {
        for (size_t i = 0; i < n; i++) f(i);
    }
}

std::string env_str(const char* k) {
    const char* v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

}  // namespace

// half::f16::from_f32(v).to_f32()
float f16_round(float v) {
    if (!std::isfinite(v)) return v;
    const float a = std::fabs(v);
    if (a == 0.0f) return v;
    int e = 0;
    std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1): the f32 exponent is e - 1
    const int E = std::max(e - 1, -14);  // below 2^-14 the f16 quantum stays 2^-24 (subnormals)
    const double q = std::ldexp(1.0, E - 10);
    double r = std::nearbyint((double)v / q) * q;  // exact division by a power of two, ties to even
    if (std::fabs(r) > 65504.0) r = v < 0 ? -INFINITY : INFINITY;
    return (float)r;
}

// port of deepseek/mat.rs: q8_roundtrip
std::vector<float> q8_roundtrip(Ty ty, bool cpu, const float* x, size_t n) {
    std::vector<float> out(x, x + n);
    const bool k_types = ty == Ty::Q5K || ty == Ty::IQ1M || ty == Ty::IQ2XXS || ty == Ty::IQ2XS;
    if (k_types && cpu) {
        for (size_t b0 = 0; b0 + 256 <= n; b0 += 256) {
            float* b = out.data() + b0;
            float amax = 0.0f, mx = 0.0f;
            for (size_t i = 0; i < 256; i++) {
                if (std::fabs(b[i]) > amax) {
                    amax = std::fabs(b[i]);
                    mx = b[i];
                }
            }
            if (amax == 0.0f) {
                std::fill(b, b + 256, 0.0f);
                continue;
            }
            const float iscale = -127.0f / mx;
            for (size_t i = 0; i < 256; i++) b[i] = std::min(std::nearbyint(iscale * b[i]), 127.0f) / iscale;
        }
    } else if (k_types || ty == Ty::Q8_0 || ty == Ty::IQ4NL) {
        for (size_t b0 = 0; b0 + 32 <= n; b0 += 32) {
            float* b = out.data() + b0;
            float amax = 0.0f;
            for (size_t i = 0; i < 32; i++) amax = std::max(amax, std::fabs(b[i]));
            const float d = amax / 127.0f;
            if (d == 0.0f) continue;
            const float dh = f16_round(d);
            for (size_t i = 0; i < 32; i++) b[i] = std::round(b[i] / d) * dh;
        }
    }
    return out;
}

void DsEnv::read_switches() {
    batch = env_str("STRATA_DS_BATCH") != "0";
    cpu_ref = std::getenv("STRATA_CPU_REF") != nullptr;
}

// ---------------------------------------------------------------- dev

// port of deepseek/mat.rs: dev::get (the initialiser)
std::unique_ptr<Dev> Dev::open(const DsGpuConfig& c, const BackendOpen& open_backend, std::string& note) {
    if (!c.on) {
        note = "STRATA_DS_GPU is not set";
        return nullptr;
    }
    if (!c.backend.empty() && c.backend != "cuda") {
        note = "STRATA_DS_BACKEND=" + c.backend + ": only cuda is ported; running without it";
        return nullptr;
    }
    std::vector<std::unique_ptr<Backend>> bes;
    for (int d : c.devices) {
        std::string err;
        std::unique_ptr<Backend> be = open_backend ? open_backend(d, err) : nullptr;
        if (!be) {
            note = "GPU " + std::to_string(d) + ": " + (open_backend ? err : std::string("no backend in this build")) +
                   "; running without it";
            return nullptr;
        }
        bes.push_back(std::move(be));
    }
    std::vector<size_t> free;
    for (auto& be : bes) {
        size_t f = 0, t = 0;
        std::string err;
        free.push_back(be->mem_info(f, t, err) ? std::max<size_t>(f, 1) : 1);
    }
    size_t total = 0;
    for (size_t f : free) total += f;
    auto dev = std::make_unique<Dev>();
    for (size_t k = 0; k < bes.size(); k++) {
        Part p;
        const size_t f = free[k];
        std::string err;
        // Room for a prefill chunk of activations and outputs, so row batches reuse them instead of allocating per expert.
        if (!bes[k]->alloc(4 * 5000000, p.x, err) || !bes[k]->alloc(4 * 5000000, p.y, err) ||
            !bes[k]->alloc(STAGE, p.stage, err)) {
            note = "GPU " + std::to_string(c.devices[k]) + ": scratch allocation failed (" + err + "); running without it";
            return nullptr;
        }
        p.budget = c.gb ? (size_t)(*c.gb * 1e9 * (double)f / (double)total)
                        : (size_t)((double)sat_sub(f, (size_t)56 << 20) * c.frac);
        // Room for the plain weights before moving on to the next GPU: what is free, less the expert cache and a margin.
        p.room = sat_sub(f, p.budget + ((size_t)768 << 20));
        p.be = std::move(bes[k]);
        dev->parts_.push_back(std::move(p));
    }
    dev->noevict_ = std::getenv("STRATA_DS_NOEVICT") != nullptr;
    dev->lfu_ = std::getenv("STRATA_DS_LFU") != nullptr;
    return dev;
}

size_t Dev::bytes() const {
    size_t s = 0;
    for (const Part& p : parts_) s += p.bytes;
    return s;
}

const char* Dev::backend_name() const { return parts_[0].be->name(); }

std::string Dev::device_name() const { return parts_[0].be->device(); }

size_t Dev::budget_left() const {
    size_t s = 0;
    for (const Part& p : parts_) s += sat_sub(p.budget, p.bytes);
    return s;
}

size_t Dev::pick(size_t len) {
    while (cur_ + 1 < parts_.size() && parts_[cur_].bytes + len > parts_[cur_].room) cur_++;
    return cur_;
}

std::optional<uint32_t> Dev::upload_on(size_t d, const uint8_t* data, size_t len, Ty ty, size_t rows, size_t cols) {
    Part& p = parts_[d];
    Tensor t;
    t.ty = ty;
    t.cols = cols;
    t.rows = rows;
    t.data = data;
    t.len = len;
    GpuTensor g;
    std::string err;
    if (!p.be->upload(t, g, err)) return std::nullopt;
    p.bytes += len;
    p.ts.push_back(std::move(g));
    return (uint32_t)d << SHIFT | (uint32_t)(p.ts.size() - 1);
}

std::optional<uint32_t> Dev::upload(const uint8_t* data, size_t len, Ty ty, size_t rows, size_t cols) {
    const size_t d = pick(len);
    return upload_on(d, data, len, ty, rows, cols);
}

const GpuTensor& Dev::tensor(uint32_t i) const { return parts_[gpu_of(i)].ts[i & ((1u << SHIFT) - 1)]; }

// port of deepseek/mat.rs: Dev::expert. Uploaded on first use while under a GPU's share of the budget, then the least
// recently used slot of the same shape is overwritten. Slots touched in the last 24 lookups are kept (one layer's
// experts are in flight together). No slot to take: STAGED (copied up per use).
std::optional<uint32_t> Dev::expert(const uint8_t* data, size_t len, Ty ty, size_t rows, size_t cols) {
    const uintptr_t k = (uintptr_t)data;
    now_ += 1;
    lookups_ += 1;
    const uint64_t now = now_;
    if (lookups_ % 4096 == 0)
        for (Slot& e : exs_) e.freq *= 0.5f;
    auto hit = ex_.find(k);
    if (hit != ex_.end()) {
        Slot& s = exs_[hit->second];
        s.last = now;
        s.freq += 1.0f;
        lookups_hit++;
        total_hit++;
        return s.id;
    }
    lookups_miss++;
    total_miss++;
    // the GPU with the most room left (the last one on a tie, as Rust's max over (room, d) tuples)
    std::optional<size_t> best;
    size_t best_room = 0;
    for (size_t d = 0; d < parts_.size(); d++) {
        const Part& p = parts_[d];
        if (p.bytes + len <= p.budget) {
            const size_t r = p.budget - p.bytes;
            if (!best || r >= best_room) {
                best = d;
                best_room = r;
            }
        }
    }
    if (best) {
        const auto i = upload_on(*best, data, len, ty, rows, cols);
        if (!i) return std::nullopt;
        ex_[k] = exs_.size();
        exs_.push_back(Slot{*i, k, now, 1.0f});
        return i;
    }
    bool all_zero = true;
    for (const Part& p : parts_) all_zero = all_zero && p.budget == 0;
    if (all_zero || len > STAGE || noevict_) return std::nullopt;
    std::optional<size_t> victim;
    for (size_t p = 0; p < exs_.size(); p++) {
        const Slot& e = exs_[p];
        if (e.last + 24 > now) continue;
        const GpuTensor& t = tensor(e.id);
        if (!(t.ty == ty && t.rows == rows && t.cols == cols)) continue;
        if (!victim) {
            victim = p;
            continue;
        }
        const Slot& v = exs_[*victim];
        // min_by keeps the first minimum
        const bool less = lfu_ ? (e.freq < v.freq || (e.freq == v.freq && e.last < v.last)) : e.last < v.last;
        if (less) victim = p;
    }
    if (!victim) return STAGED;
    const size_t p = *victim;
    if (lfu_ && exs_[p].freq > 4.0f) return STAGED;
    const uint32_t i = exs_[p].id;
    const uintptr_t old = exs_[p].key;
    const GpuTensor& t = tensor(i);
    std::string err;
    if (!parts_[gpu_of(i)].be->htod(data, len, Buf(t.ptr, len, nullptr), 0, err)) return STAGED;
    ex_.erase(old);
    ex_[k] = p;
    exs_[p] = Slot{i, k, now, 1.0f};
    return i;
}

void Dev::gemv(uint32_t i, const float* x, size_t nx, float* y, size_t ny, const uint8_t* staged, size_t staged_len,
               Ty ty, size_t rows, size_t cols) {
    Part& p = parts_[i == STAGED ? 0 : gpu_of(i)];
    std::string err;
    if (!p.be->htod(x, nx * 4, p.x, 0, err)) throw DsError("gemv: " + err);
    if (i == STAGED) {
        if (!p.be->htod(staged, staged_len, p.stage, 0, err)) throw DsError("gemv: " + err);
        const GpuTensor w(ty, rows, cols, Buf(p.stage.addr(), staged_len, nullptr));
        if (!p.be->gemv(w, p.x.addr(), p.y.addr(), err)) throw DsError("gemv: " + err);
    } else {
        if (!p.be->gemv(tensor(i), p.x.addr(), p.y.addr(), err)) throw DsError("gemv: " + err);
    }
    if (!p.be->dtoh(p.y, 0, y, ny * 4, err)) throw DsError("gemv: " + err);
}

std::optional<size_t> Dev::part_of(const Mat& m) const {
    if (!m.gpu) return std::nullopt;
    return *m.gpu == STAGED ? 0 : gpu_of(*m.gpu);
}

// port of deepseek/mat.rs: Dev::batch_mats. One gemv per matrix, all queued before a single readback; all matrices sit
// on one GPU. Job i reads x[i*xs..] (xs = 0: every job reads x) and writes its rows after job i-1's. Staged experts
// are uploaded once each into a temporary buffer, so a layer with both resident and staged experts still batches.
bool Dev::batch_mats(const std::vector<Mat>& ms, const float* x, size_t x_len, size_t xs, std::vector<float>& y) {
    if (ms.empty()) return false;
    const auto first = part_of(ms[0]);
    if (!first) return false;
    for (const Mat& m : ms)
        if (part_of(m) != first) return false;
    Part& p = parts_[*first];
    const size_t nx = xs == 0 ? x_len : ms.size() * xs;
    if (nx > p.x.len() / 4) return false;
    std::string err;
    if (!p.be->htod(x, nx * 4, p.x, 0, err)) return false;
    std::vector<GpuTensor> owned;
    size_t off = 0;
    for (size_t i = 0; i < ms.size(); i++) {
        const Mat& m = ms[i];
        if (off + m.rows > p.y.len() / 4) return false;
        if (*m.gpu == STAGED) {
            Buf b;
            if (!p.be->alloc(m.len, b, err)) return false;
            if (!p.be->htod(m.data, m.len, b, 0, err)) return false;
            owned.emplace_back(m.ty, m.rows, m.cols, b);
            if (!p.be->gemv(owned.back(), p.x.addr() + (uint64_t)(4 * i * xs), p.y.addr() + (uint64_t)(4 * off), err))
                return false;
        } else {
            if (!p.be->gemv(tensor(*m.gpu), p.x.addr() + (uint64_t)(4 * i * xs), p.y.addr() + (uint64_t)(4 * off), err))
                return false;
        }
        off += m.rows;
    }
    y.resize(off);
    if (!p.be->dtoh(p.y, 0, y.data(), off * 4, err)) return false;
    owned.clear();
    return true;
}

// port of deepseek/mat.rs: Dev::rows. W x[t] for every row of `xs` (one matrix, several activations). Uses the
// tensor-core GEMM when there are enough rows and the GEMV otherwise; staged weights are uploaded once for the batch.
bool Dev::rows(const Mat& m, const std::vector<const float*>& xs, std::vector<float>& y) {
    if (!m.gpu) return false;
    const uint32_t i = *m.gpu;
    const size_t d = i == STAGED ? 0 : gpu_of(i);
    Part& p = parts_[d];
    if (xs.empty()) {
        y.clear();
        return true;
    }
    const size_t n = xs.size();
    std::vector<float> flat;
    flat.reserve(n * m.cols);
    for (const float* x : xs) flat.insert(flat.end(), x, x + m.cols);
    std::string err;
    GpuTensor owned;
    const GpuTensor* w = nullptr;
    if (i == STAGED) {
        Buf b;
        if (!p.be->alloc(m.len, b, err) || !p.be->htod(m.data, m.len, b, 0, err)) return false;
        owned = GpuTensor(m.ty, m.rows, m.cols, b);
        w = &owned;
    } else {
        w = &tensor(i);
    }
    y.resize(n * m.rows);
    // Reuse the scratch pair when the chunk fits; otherwise allocate a GEMM-sized pair for this call.
    if (n * m.cols <= p.x.len() / 4 && n * m.rows <= p.y.len() / 4) {
        if (!p.be->htod(flat.data(), flat.size() * 4, p.x, 0, err)) return false;
        if (n <= 4) {
            if (!p.be->gemvn(*w, p.x.addr(), m.cols, p.y.addr(), m.rows, n, err)) return false;
        } else {
            if (!p.be->gemm_dense(*w, n, p.x.addr(), m.cols, p.y.addr(), m.rows, 0, err)) return false;
        }
        if (!p.be->dtoh(p.y, 0, y.data(), n * m.rows * 4, err)) return false;
    } else {
        Buf xb, yb;
        if (!p.be->alloc(n * m.cols * 4, xb, err) || !p.be->alloc(n * m.rows * 4, yb, err)) return false;
        if (!p.be->htod(flat.data(), flat.size() * 4, xb, 0, err)) return false;
        if (!p.be->gemm_dense(*w, n, xb.addr(), m.cols, yb.addr(), m.rows, 0, err)) return false;
        if (!p.be->dtoh(yb, 0, y.data(), n * m.rows * 4, err)) return false;
    }
    return true;
}

// ---------------------------------------------------------------- free functions

// port of deepseek/mat.rs: matvec_many (with matvec_many_opt inlined: the batched path when the matrices share a GPU)
std::vector<float> matvec_many(const std::vector<Mat>& ms, const float* x, size_t x_len, size_t xs) {
    std::vector<float> y;
    if (ms.empty()) return y;
    DsEnv* env = ms[0].env;
    if (env && env->batch && env->dev && env->dev->batch_mats(ms, x, x_len, xs, y)) return y;
    size_t total = 0;
    for (const Mat& m : ms) total += m.rows;
    y.assign(total, 0.0f);
    size_t off = 0;
    for (size_t i = 0; i < ms.size(); i++) {
        ms[i].matvec(x + i * xs, y.data() + off);
        off += ms[i].rows;
    }
    return y;
}

// port of deepseek/mat.rs: matvec_rows
std::vector<float> matvec_rows(const Mat& m, const std::vector<const float*>& xs) {
    std::vector<float> y;
    if (m.env && m.env->dev && m.env->dev->rows(m, xs, y)) return y;
    y.assign(xs.size() * m.rows, 0.0f);
    for (size_t i = 0; i < xs.size(); i++) m.matvec(xs[i], y.data() + i * m.rows);
    return y;
}

// port of deepseek/mat.rs: vec_f32
bool vec_f32(const Gguf& g, DsEnv* env, const std::string& name, std::vector<float>& out, std::string& err) {
    Mat m;
    if (!Mat::load(g, env, name, m, err)) return false;
    out.assign(m.rows * m.cols, 0.0f);
    for (size_t r = 0; r < m.rows; r++) m.row_into(r, out.data() + r * m.cols);
    return true;
}

// ---------------------------------------------------------------- Mat

// port of deepseek/mat.rs: Mat::load
bool Mat::load(const Gguf& g, DsEnv* env, const std::string& name, Mat& out, std::string& err) {
    const auto it = g.tensors().find(name);
    if (it == g.tensors().end()) {
        err = "missing tensor " + name;
        return false;
    }
    const TensorInfo& t = it->second;
    const auto ty = ty_from_u32(t.ty);
    if (!ty) {
        err = name + ": unsupported GGML type " + std::to_string(t.ty);
        return false;
    }
    if (t.dims.empty()) {
        err = name + ": tensor has no dimensions";
        return false;
    }
    Mat m;
    m.ty = *ty;
    m.cols = (size_t)t.dims[0];
    m.rows = 1;
    for (size_t i = 1; i < t.dims.size(); i++) m.rows *= (size_t)t.dims[i];
    m.cpu = name.find("_exps") != std::string::npos;
    m.data = g.bytes(t, m.len, err);
    if (!m.data) return false;
    m.env = env;
    if (!m.cpu && t.dims.size() == 2 && env && env->dev) m.gpu = env->dev->upload(m.data, m.len, m.ty, m.rows, m.cols);
    out = m;
    return true;
}

// port of deepseek/mat.rs: Mat::group
Mat Mat::group(size_t i, size_t n) const {
    Mat m = *this;
    m.rows = rows / n;
    const size_t sz = m.rows * row_bytes(ty, cols);
    m.data = data + i * sz;
    m.len = sz;
    if (gpu) m.gpu = *gpu + (uint32_t)i;
    return m;
}

// port of deepseek/mat.rs: Mat::expert
Mat Mat::expert(size_t e, size_t n_expert) const {
    Mat m = *this;
    m.rows = rows / n_expert;
    const size_t sz = m.rows * row_bytes(ty, cols);
    m.data = data + e * sz;
    m.len = sz;
    m.gpu = (env && env->dev) ? env->dev->expert(m.data, sz, ty, m.rows, cols) : std::nullopt;
    return m;
}

// port of deepseek/mat.rs: Mat::row_into
void Mat::row_into(size_t r, float* out) const {
    const size_t rb = row_bytes(ty, cols);
    if (!dequant_row(ty, data + r * rb, rb, out, cols)) throw DsError("row_into: bad row size");
}

// port of deepseek/mat.rs: Mat::matvec
void Mat::matvec(const float* x, float* y) const {
    if (gpu && env && env->dev) {
        const bool st = *gpu == Dev::STAGED;
        env->dev->gemv(*gpu, x, cols, y, rows, st ? data : nullptr, st ? len : 0, ty, rows, cols);
        return;
    }
    ThreadPool* pool = env ? env->pool : nullptr;
    const size_t rb = row_bytes(ty, cols);
    // Integer Q8_K dots are several times faster than dequantising each row, and match llama.cpp's CPU path.
    // `STRATA_CPU_REF=1` keeps the f32 reference for debugging and exact old numerics.
    if ((cpu || rows >= 512) && !(env && env->cpu_ref) && cols % block_elems(ty) == 0) {
        if (const IntDotFn dot = int_dot_kernel(ty)) {
            const std::vector<Q8K> xq = quantize_q8k(x, cols);
            par_for(pool, (rows + 31) / 32, [&](size_t c) {
                const size_t hi = std::min(rows, c * 32 + 32);
                for (size_t r = c * 32; r < hi; r++) y[r] = dot(data + r * rb, rb, xq.data(), xq.size());
            });
            return;
        }
    }
    const std::vector<float> xq = q8_roundtrip(ty, cpu, x, cols);
    par_for(pool, (rows + 15) / 16, [&](size_t c) {
        std::vector<float> tmp(cols);
        const size_t hi = std::min(rows, c * 16 + 16);
        for (size_t r = c * 16; r < hi; r++) {
            if (!dequant_row(ty, data + r * rb, rb, tmp.data(), cols)) {
                y[r] = NAN;
                continue;
            }
            float s = 0.0f;
            for (size_t k = 0; k < cols; k++) s += tmp[k] * xq[k];
            y[r] = s;
        }
    });
}

}  // namespace strata
