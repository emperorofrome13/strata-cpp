// port of weights.rs. Function names match the Rust ones.
#include "weights.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <thread>

namespace strata {
namespace {

// port of rayon's par_chunks_mut(len, chunk).for_each(): the chunk boundaries are the same as rayon's
// (chunk c covers [c*chunk, min(len, (c+1)*chunk))), the chunks are handed to whole std::thread workers
// instead of a work-stealing pool, and every chunk runs its body exactly once.
template <typename F>
void par_chunks_mut(float* base, size_t len, size_t chunk, F body) {
    if (len == 0 || chunk == 0) return;
    const size_t n_chunks = (len + chunk - 1) / chunk;
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t n_threads = std::min<size_t>(n_chunks, hw ? hw : 1);
    auto run = [&](size_t t) {
        const size_t lo = n_chunks * t / n_threads;
        const size_t hi = n_chunks * (t + 1) / n_threads;
        for (size_t c = lo; c < hi; c++) {
            const size_t off = c * chunk;
            body(c, base + off, std::min(chunk, len - off));
        }
    };
    if (n_threads <= 1) {
        run(0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(n_threads - 1);
    for (size_t t = 1; t < n_threads; t++) pool.emplace_back(run, t);
    run(0);
    for (std::thread& th : pool) th.join();
}

std::string prefix_blk(size_t il) { return "blk." + std::to_string(il) + "."; }

}  // namespace

// ---------------------------------------------------------------- Tensor

// port of weights.rs: Tensor::row_bytes
size_t Tensor::row_bytes() const { return strata::row_bytes(ty, (uint64_t)cols); }

// port of weights.rs: Tensor::row
bool Tensor::row(size_t r, float* out) const {
    const size_t rb = row_bytes();
    if (!data || r >= rows || r * rb + rb > len) return false;
    return dequant_row(ty, data + r * rb, rb, out, cols);
}

// port of weights.rs: Tensor::matvec
void Tensor::matvec(const float* x, float* y) const {
    assert(x && y && data && rows * row_bytes() == len);
    par_chunks_mut(y, rows, 16, [&](size_t ci, float* ys, size_t n) {
        std::vector<float> buf(cols > 0 ? cols : 0);
        for (size_t k = 0; k < n; k++) {
            const bool ok = row(ci * 16 + k, buf.data());
            assert(ok);
            (void)ok;
            float s = 0.0f;  // Rust: buf.iter().zip(x).map(|(a, b)| a * b).sum()
            for (size_t j = 0; j < cols; j++) s += buf[j] * x[j];
            ys[k] = s;
        }
    });
}

// port of weights.rs: Tensor::matmul
void Tensor::matmul(const float* x, size_t n_tokens, float* out) const {
    assert(x && out && data && n_tokens > 0 && rows * row_bytes() == len);
    // Row-outer: each weight row is decoded once, whatever n_tokens is. Blocks of rows that fit the
    // L2 cache stay there while every token passes over them.
    const size_t block = (size_t)(1 << 16) / (cols ? cols : 1) + 1;
    std::vector<float> by_row(n_tokens * rows, 0.0f);
    par_chunks_mut(by_row.data(), by_row.size(), block * n_tokens, [&](size_t b, float* ys, size_t n) {
        const size_t nr = n / n_tokens;
        std::vector<float> buf(block * (cols ? cols : 0));
        for (size_t r = 0; r < nr; r++) {
            const bool ok = row(b * block + r, buf.data() + r * cols);
            assert(ok);
            (void)ok;
        }
        for (size_t t = 0; t < n_tokens; t++) {
            const float* xt = x + t * cols;
            for (size_t r = 0; r < nr; r++) ys[r * n_tokens + t] = dot(buf.data() + r * cols, xt, cols);
        }
    });
    par_chunks_mut(out, n_tokens * rows, rows, [&](size_t t, float* ys, size_t n) {
        for (size_t r = 0; r < n; r++) ys[r] = by_row[r * n_tokens + t];
    });
}

// port of weights.rs: Tensor::expert
Tensor Tensor::expert(size_t e, size_t n_expert) const {
    assert(n_expert > 0);
    const size_t rows_e = rows / n_expert;
    const size_t rb = row_bytes();
    Tensor t = *this;
    t.rows = rows_e;
    if (e >= n_expert || e * rows_e * rb + rows_e * rb > len) {
        t.data = nullptr;
        t.len = 0;
        return t;
    }
    t.data = data + e * rows_e * rb;
    t.len = rows_e * rb;
    return t;
}

// port of weights.rs: Tensor::to_f32
std::vector<float> Tensor::to_f32() const {
    std::vector<float> v(rows * cols, 0.0f);
    for (size_t r = 0; r < rows; r++) {
        const bool ok = row(r, v.data() + r * cols);
        assert(ok);
        (void)ok;
    }
    return v;
}

// ---------------------------------------------------------------- Weights

// port of weights.rs: Weights::new_with_aux
bool Weights::init(const Gguf* g_, const Gguf* aux_g_, std::string& err) {
    g = g_;
    aux_g = aux_g_;
    built.clear();
    owned_.clear();
    lora_baked_.clear();
    if (!cfg_.from_gguf(*g, err)) return false;

    // llama.cpp src/models/qwen3next.cpp: per key head group, r = nv / nk beta rows, then r alpha rows.
    for (size_t il = 0; il < cfg_.n_layer; il++) {
        Tensor ba;
        std::string ignored;
        if (!layer(il, "ssm_ba.weight", ba, ignored)) continue;  // Rust: `let Ok(ba) = .. else { continue }`
        const size_t nk = cfg_.ssm_n_group, nv = cfg_.ssm_dt_rank, rb = ba.row_bytes();
        if (!(nk > 0 && nv > 0 && nv % nk == 0 && ba.rows == 2 * nv && ba.len == ba.rows * rb)) {
            err = "blk." + std::to_string(il) + ".ssm_ba: " + std::to_string(ba.rows) + " rows, " +
                  std::to_string(ba.len) + " bytes for group_count " + std::to_string(nk) +
                  ", time_step_rank " + std::to_string(nv);
            return false;
        }
        const size_t r = nv / nk;
        std::vector<uint8_t> b, a;
        b.reserve(ba.len / 2);
        a.reserve(ba.len / 2);
        const size_t grp_bytes = 2 * r * rb;
        for (size_t grp = 0; grp + grp_bytes <= ba.len; grp += grp_bytes) {
            b.insert(b.end(), ba.data + grp, ba.data + grp + r * rb);
            a.insert(a.end(), ba.data + grp + r * rb, ba.data + grp + grp_bytes);
        }
        // The Rust version leaks these (`Box::leak`, a few KB per layer per load); here `owned_` owns them.
        auto own = [&](const std::vector<uint8_t>& v) -> const uint8_t* {
            std::unique_ptr<uint8_t[]> p(new uint8_t[v.size() ? v.size() : 1]);
            if (!v.empty()) std::memcpy(p.get(), v.data(), v.size());
            const uint8_t* d = p.get();
            owned_.push_back(std::move(p));
            return d;
        };
        Tensor beta = ba;
        beta.rows = ba.rows / 2;
        beta.data = own(b);
        beta.len = b.size();
        Tensor alpha = beta;
        alpha.data = own(a);
        alpha.len = a.size();
        built[prefix_blk(il) + "ssm_beta.weight"] = beta;
        built[prefix_blk(il) + "ssm_alpha.weight"] = alpha;
    }

    // Standalone EAGLE3 draft GGUF: remap layer-0 draft weights onto blk.{n_layer}.* so they never
    // shadow layer 0.
    if (aux_g && aux_g->tensors().count("fc.weight") != 0) {
        const size_t nl = cfg_.n_layer;
        const std::string nl_prefix = prefix_blk(nl);
        for (const auto& kv : aux_g->tensors()) {
            const std::string& name = kv.first;
            const TensorInfo& t = kv.second;
            if (name == "d2t" || name == "t2d" || name == "token_embd.weight") continue;
            auto ty = ty_from_u32(t.ty);
            if (!ty) continue;
            std::string dst;
            if (name == "fc.weight") {
                dst = nl_prefix + "eagle3.fc.weight";
            } else if (name == "output.weight") {
                dst = nl_prefix + "eagle3.output.weight";
            } else if (name == "output_norm.weight") {
                dst = nl_prefix + "nextn.shared_head_norm.weight";
            } else if (name == "blk.0.attn_norm.weight") {
                dst = nl_prefix + "nextn.enorm.weight";
            } else if (name == "blk.0.attn_norm_2.weight") {
                dst = nl_prefix + "nextn.hnorm.weight";
            } else if (name == "blk.0.ffn_norm.weight") {
                dst = nl_prefix + "post_attention_norm.weight";
            } else if (name.rfind("blk.0.", 0) == 0) {
                dst = nl_prefix + name.substr(6);
            } else {
                continue;
            }
            size_t blen = 0;
            std::string berr;
            const uint8_t* d = aux_g->bytes(t, blen, berr);
            if (!d) continue;  // Gguf::open already rejected truncated tensors; ty is known good
            Tensor nt;
            nt.ty = *ty;
            nt.cols = t.dims.empty() ? 0 : (size_t)t.dims[0];
            nt.rows = 1;
            for (size_t i = 1; i < t.dims.size(); i++) nt.rows *= (size_t)t.dims[i];
            nt.data = d;
            nt.len = blen;
            built[dst] = nt;
        }
    }
    return true;
}

// port of weights.rs: Weights::get - `name`, with the architecture's alias for a canonical suffix
// tried second.
bool Weights::get(const std::string& name, Tensor& out, std::string& err) const {
    auto bit = built.find(name);
    if (bit != built.end()) {
        out = bit->second;
        return true;
    }
    auto find_t = [&](const Gguf* gg, const TensorInfo*& ti, const Gguf*& owner) -> bool {
        auto it = gg->tensors().find(name);
        if (it != gg->tensors().end()) {
            ti = &it->second;
            owner = gg;
            return true;
        }
        for (const auto& al : cfg_.aliases) {
            const std::string& canon = al.first;
            if (name.size() < canon.size() ||
                name.compare(name.size() - canon.size(), canon.size(), canon) != 0)
                continue;
            auto jt = gg->tensors().find(name.substr(0, name.size() - canon.size()) + al.second);
            if (jt != gg->tensors().end()) {
                ti = &jt->second;
                owner = gg;
                return true;
            }
        }
        return false;
    };
    const TensorInfo* ti = nullptr;
    const Gguf* owner = nullptr;
    bool found = find_t(g, ti, owner);
    if (!found && aux_g) found = find_t(aux_g, ti, owner);
    if (!found) {
        err = "missing tensor " + name;
        return false;
    }
    auto ty = ty_from_u32(ti->ty);
    if (!ty) {
        err = "tensor " + name + ": unsupported quant type id " + std::to_string(ti->ty);
        return false;
    }
    size_t blen = 0;
    std::string berr;
    const uint8_t* d = owner->bytes(*ti, blen, berr);
    if (!d) {
        err = "tensor " + name + ": " + berr;
        return false;
    }
    out.ty = *ty;
    out.cols = ti->dims.empty() ? 0 : (size_t)ti->dims[0];
    out.rows = 1;
    for (size_t i = 1; i < ti->dims.size(); i++) out.rows *= (size_t)ti->dims[i];
    out.data = d;
    out.len = blen;
    return true;
}

// port of weights.rs: Weights::layer
bool Weights::layer(size_t il, const std::string& suffix, Tensor& out, std::string& err) const {
    // StrataFreet addition (INKLING2_NOTES.md): the looped architectures keep `n_layer` cache slots
    // but only `n_layer_phys` blocks of weights. A slot reads the weights of the block it loops
    // (loop_layer_of, PR #389's loop_layer). The remap is unconditional: every slot below loop_start
    // maps to itself, and the second/third visits of the core (slots 18..41 for this model, below
    // n_layer_phys = 24) must remap too — they are not past the physical blocks.
    if (cfg_.is_looped()) il = loop_layer_of(cfg_, il);
    return layer_phys(il, suffix, out, err);
}

// StrataFreet addition (INKLING2_NOTES.md): the physical-block lookup the remapped `layer` lands on.
// The expert cache is keyed by the physical blocks and must NOT go through the slot remap: for this
// model `layer(18, ..)` would read blk.6 (slot 18 is the core's second pass), caching block 6's
// experts as layer 18's and never placing blk.18..23's own experts.
bool Weights::layer_phys(size_t il, const std::string& suffix, Tensor& out, std::string& err) const {
    return get(prefix_blk(il) + suffix, out, err);
}

// port of weights.rs: Weights::clear_baked
void Weights::clear_baked() {
    for (const std::string& name : lora_baked_) built.erase(name);
    lora_baked_.clear();
}

// port of weights.rs: Weights::validate - refuse files containing quant types this engine does not
// implement, except in the MTP (nextn) layers, blocks `n_layer` on: `nextn` is false if one of those
// has such a type, so that the model loads without MTP.
bool Weights::validate(size_t n_layer, bool& nextn, std::string& err) const {
    nextn = true;
    auto check = [&](const std::map<std::string, TensorInfo>& ts) -> bool {
        for (const auto& kv : ts) {
            const TensorInfo& t = kv.second;
            if (t.name == "d2t" || t.name == "t2d") continue;
            if (ty_from_u32(t.ty)) continue;
            std::optional<size_t> il;
            if (t.name.rfind("blk.", 0) == 0) {
                size_t v = 0;
                bool digits = false;
                const char* r = t.name.c_str() + 4;
                while (*r >= '0' && *r <= '9') {
                    v = v * 10 + (size_t)(*r - '0');
                    r++;
                    digits = true;
                }
                if (digits && *r == '.') il = v;
            }
            if (il && *il >= n_layer) {
                nextn = false;
                continue;
            }
            err = "tensor " + t.name + ": unsupported quant type id " + std::to_string(t.ty);
            return false;
        }
        return true;
    };
    if (!check(g->tensors())) return false;
    if (aux_g && !check(aux_g->tensors())) return false;
    return true;
}

// ---------------------------------------------------------------- page cache advice

// port of weights.rs: will_need. Rust's `madvise` shim is empty off Linux, so Strata's Windows build reads nothing ahead
// and every first touch of a mapped expert page is a page fault on the copy's thread. Here the range is mapped in ahead
// with PrefetchVirtualMemory (the Windows counterpart of madvise WILLNEED): it is the readahead the Prefetch thread was
// written for (see MODIFICATIONS.md, W3-E3b). A range outside a view makes the call fail, and the copy then faults as before.
void will_need(size_t p, size_t n) {
    if (n == 0) return;
    WIN32_MEMORY_RANGE_ENTRY e{};
    const size_t a = p & ~size_t(4095);
    e.VirtualAddress = reinterpret_cast<void*>(a);
    e.NumberOfBytes = p + n - a;
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &e, 0);
}

// port of weights.rs: cold. Empty off Linux for the same reason as will_need.
void cold(const uint8_t* b, size_t len) {
    (void)b;
    (void)len;
}

// ---------------------------------------------------------------- Prefetch

// port of weights.rs: Prefetch::start
void Prefetch::start() {
    if (running_) return;
    running_ = true;
    stop_ = false;
    th_ = std::thread([this] {
        for (;;) {
            std::vector<std::pair<size_t, size_t>> rs;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (stop_) return;
                    continue;
                }
                rs = std::move(queue_.front());
                queue_.erase(queue_.begin());
            }
            for (const auto& pr : rs) will_need(pr.first, pr.second);
        }
    });
}

// port of weights.rs: Prefetch::send
void Prefetch::send(std::vector<std::pair<size_t, size_t>> rs) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!running_) return;  // Rust panics here (tx is None only after drop); a late send is a no-op
        queue_.push_back(std::move(rs));
    }
    cv_.notify_one();
}

// port of weights.rs: impl Drop for Prefetch - close the channel, then join. The worker drains the
// batches already queued before it exits, exactly as `for rs in rx` does.
void Prefetch::destroy() {
    if (!running_) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    running_ = false;
}

Prefetch::~Prefetch() { destroy(); }

// ---------------------------------------------------------------- dot

// port of weights.rs: dot
float dot(const float* a, const float* b, size_t n) {
    float acc[16] = {0.0f};
    const size_t blocks = n / 16;
    for (size_t blk = 0; blk < blocks; blk++) {
        const float* x = a + blk * 16;
        const float* y = b + blk * 16;
        for (size_t i = 0; i < 16; i++) acc[i] += x[i] * y[i];
    }
    float tail = 0.0f;
    for (size_t i = blocks * 16; i < n; i++) tail += a[i] * b[i];
    float s = 0.0f;
    for (size_t i = 0; i < 16; i++) s += acc[i];
    return s + tail;
}

}  // namespace strata
