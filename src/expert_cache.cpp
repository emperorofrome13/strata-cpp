// port of expert_cache.rs: ExpertCache::new (slots for a budget, placement from the routing profile), place,
// sync_table, the accessors the decode reads, record / record_prompt / settle / reset_counts / adapt / swap, and
// save_profile with profile_update.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "expert_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>

#include "weights.h"

namespace strata {

namespace {

// port of expert_cache.rs: KEEP and LONG_KEEP (share of each count kept per adapt: short-term and long-term)
constexpr float kKeep = 0.95f;
constexpr float kLongKeep = 0.999f;
// port of expert_cache.rs: MAX_POOL_FULL_BYTES (a pool holds at most 8 GiB of experts)
constexpr size_t kMaxPoolFullBytes = size_t(8192) << 20;

// port of expert_cache.rs: profile_path (profile-<arch>-<L>x<E>-<tensors>.f32 under cache_dir).
// StrataFreet addition: the looped architectures key the profile by their physical blocks.
std::string profile_file(const Model& m) {
    const ModelConfig& c = m.cfg;
    const size_t layers = c.is_looped() ? c.n_layer_phys : c.n_layer;
    const size_t n = m.w.g ? m.w.g->tensors().size() : 0;
    return (std::filesystem::path(cache_dir()) / ("profile-" + c.arch + "-" + std::to_string(layers) + "x" +
                                                  std::to_string(c.n_expert) + "-" + std::to_string(n) + ".f32"))
        .string();
}

// The f32 values of a file (little endian), empty when it cannot be read.
std::vector<float> read_floats(const std::string& path) {
    std::vector<float> out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    out.resize(b.size() / 4);
    if (!out.empty()) std::memcpy(out.data(), b.data(), out.size() * 4);
    return out;
}

// port of expert_cache.rs: profile_update. The counts to save: `existing` with every layer this cache holds replaced
// by its `long` counts; the other layers (another device's) keep the file's counts.
std::vector<float> profile_update(const std::vector<float>& existing, size_t n_expert, const std::vector<float>& long_,
                                  const std::vector<bool>& moe) {
    std::vector<float> all = existing;
    if (all.size() < long_.size()) all.resize(long_.size(), 0.0f);
    for (size_t il = 0; il < moe.size(); il++) {
        if (!moe[il]) continue;
        std::copy(long_.begin() + il * n_expert, long_.begin() + (il + 1) * n_expert, all.begin() + il * n_expert);
    }
    return all;
}

// port of expert_cache.rs: smallest. Keeps the k first elements of `v` in the order a stable sort by `cmp` gives
// (ties by value), without sorting all of `v`.
template <class Cmp>
void smallest(std::vector<size_t>& v, size_t k, Cmp cmp) {
    auto lt = [&](size_t a, size_t b) {
        if (cmp(a, b)) return true;
        if (cmp(b, a)) return false;
        return a < b;
    };
    if (k < v.size()) {
        std::nth_element(v.begin(), v.begin() + k, v.end(), lt);
        v.resize(k);
    }
    std::sort(v.begin(), v.end(), lt);
}

// StrataFreet addition: the bytes of expert `e` of mat `i` of layer `il` (data and length). The per-expert
// scales are one f32 each: Tensor::expert() would divide rows by n_expert and return an empty tensor, so
// the e-th float is sliced by hand.
bool expert_slice(const Model& m, size_t il, const LayerExperts& lx, size_t i, size_t e, size_t n, Tensor& x,
                  std::string& err) {
    const char* name = lx.fused_gu ? kExpertMatsInkling2[i] : kExpertMats[i];
    Tensor t;
    if (!m.w.layer_phys(il, name, t, err)) return false;
    if (lx.fused_gu && i >= 2) {
        x = Tensor{};
        x.ty = Ty::F32;
        x.cols = 1;
        x.rows = 1;
        x.data = t.data + 4 * e;
        x.len = 4;
    } else {
        x = t.expert(e, n);
    }
    return true;
}

}  // namespace

// port of expert_cache.rs: cache_dir
std::string cache_dir() {
    std::string base;
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) {
        base = x;
    } else if (const char* l = std::getenv("LOCALAPPDATA"); l && *l) {
        base = l;
    } else if (const char* h = std::getenv("HOME"); h && *h) {
        base = (std::filesystem::path(h) / ".cache").string();
    }
    return (std::filesystem::path(base) / "rusting_strata").string();
}

// port of expert_cache.rs: ExpertCache::new
bool ExpertCache::create(const Model& m, Backend& gpu, size_t budget, size_t layer_begin, size_t layer_end,
                         std::unique_ptr<ExpertCache>& out, std::string& err) {
    const ModelConfig& c = m.cfg;
    const size_t n = c.n_expert;
    const size_t n_layers = layer_end;
    struct Spec {
        size_t pid = 0;
        ExpertMat mats[4];
        size_t n_mats = 3;
        bool fused_gu = false;
        bool has_scales = false;
    };
    std::vector<std::optional<Spec>> specs;
    std::vector<std::pair<size_t, size_t>> pool_specs;  // (slot bytes, experts in pool)
    for (size_t il = 0; il < n_layers; il++) {
        if (!c.is_moe_layer(il) || il < layer_begin || il >= layer_end) {
            specs.push_back(std::nullopt);
            continue;
        }
        size_t off = 0;
        Spec sp;
        // StrataFreet addition (INKLING2_NOTES.md): inkling2 fuses the gate and up projections in one
        // tensor and (on the quantized blocks) carries a per-expert f32 scale per matrix. The fused
        // slot holds [0] gate_up, [1] down, [2] gate_up scale, [3] down scale (4 bytes each).
        {
            Tensor t;
            std::string probe;
            const bool fused = m.w.layer_phys(il, "ffn_gate_up_exps.weight", t, probe);
            err.clear();
            if (fused) {
                if (m.w.layer_phys(il, "ffn_gate_exps.weight", t, probe) || m.w.layer_phys(il, "ffn_up_exps.weight", t, probe)) {
                    err = "blk." + std::to_string(il) + ": both fused and separate gate/up expert tensors";
                    return false;
                }
                err.clear();
                sp.fused_gu = true;
                // [0] gate_up {n_embd, 2*n_ff, n_expert}
                Tensor t0;
                if (!m.w.layer_phys(il, kExpertMatsInkling2[0], t0, err)) return false;
                Tensor e0 = t0.expert(0, n);
                sp.mats[0] = ExpertMat{e0.ty, e0.rows, e0.cols, off, e0.len};
                off = (off + e0.len + 15) / 16 * 16;
                // [1] down {n_ff, n_embd, n_expert}
                Tensor t1;
                if (!m.w.layer_phys(il, kExpertMatsInkling2[1], t1, err)) return false;
                Tensor e1 = t1.expert(0, n);
                sp.mats[1] = ExpertMat{e1.ty, e1.rows, e1.cols, off, e1.len};
                off = (off + e1.len + 15) / 16 * 16;
                // [2]/[3] the per-expert scales (one f32 per expert per matrix), when the file has them.
                Tensor ts;
                if (m.w.layer_phys(il, kExpertMatsInkling2[2], ts, probe)) {
                    if (ts.len != 4 * n) {
                        err = "blk." + std::to_string(il) + ".ffn_gate_up_exps.scale: expected " + std::to_string(n) +
                              " f32 entries";
                        return false;
                    }
                    sp.mats[2] = ExpertMat{Ty::F32, 1, 1, off, 4};
                    off = (off + 4 + 15) / 16 * 16;
                    Tensor td;
                    if (!m.w.layer_phys(il, kExpertMatsInkling2[3], td, err)) return false;
                    if (td.len != 4 * n) {
                        err = "blk." + std::to_string(il) + ".ffn_down_exps.scale: expected " + std::to_string(n) +
                              " f32 entries";
                        return false;
                    }
                    sp.mats[3] = ExpertMat{Ty::F32, 1, 1, off, 4};
                    off = (off + 4 + 15) / 16 * 16;
                    sp.has_scales = true;
                    sp.n_mats = 4;
                } else {
                    err.clear();
                    sp.n_mats = 2;
                }
            } else {
                err.clear();
                for (size_t i = 0; i < 3; i++) {
                    Tensor t2;
                    if (!m.w.layer_phys(il, kExpertMats[i], t2, err)) return false;
                    Tensor e = t2.expert(0, n);
                    sp.mats[i] = ExpertMat{e.ty, e.rows, e.cols, off, e.len};
                    off = (off + e.len + 15) / 16 * 16;
                }
                sp.n_mats = 3;
            }
        }
        size_t pid = pool_specs.size();
        for (size_t p = 0; p < pool_specs.size(); p++) {
            if (pool_specs[p].first == off && (pool_specs[p].second + n) * off <= kMaxPoolFullBytes) {
                pid = p;
                break;
            }
        }
        if (pid == pool_specs.size()) pool_specs.push_back({off, 0});
        pool_specs[pid].second += n;
        sp.pid = pid;
        specs.push_back(sp);
    }
    size_t total = 0;
    for (const auto& ps : pool_specs) total += ps.first * ps.second;

    auto cache = std::unique_ptr<ExpertCache>(new ExpertCache());
    cache->n_expert_ = n;
    // The profile: read unless STRATA_FIXED_EXPERTS is set (placement ignores it and never adapts).
    if (!std::getenv("STRATA_FIXED_EXPERTS")) cache->profile_ = profile_file(m);
    std::vector<float> saved = cache->profile_.empty() ? std::vector<float>() : read_floats(cache->profile_);
    if (saved.size() < n_layers * n) saved.clear();
    cache->long_ = saved;
    cache->long_.resize(n_layers * n, 0.0f);
    // counts start from the profile at the size routing gives them in steady state (k experts per token, an adapt
    // every 8 tokens), so the first tokens of a run do not move the placement more than later ones.
    cache->steady_ = float(c.n_expert_used * 8) / (1.0f - kKeep);
    cache->counts_.assign(n_layers * n, 0.0f);
    for (size_t il = 0; il < n_layers; il++) {
        float s = 0.0f;
        for (size_t e = 0; e < n; e++) s += cache->long_[il * n + e];
        for (size_t e = 0; e < n; e++)
            cache->counts_[il * n + e] = s > 0.0f ? cache->long_[il * n + e] * cache->steady_ / s : 0.0f;
    }
    cache->prompt_.assign(n_layers * n, 0.0f);

    // The budget comes from the driver's free-memory estimate, which can be more than it will allocate: on a failed
    // allocation, free the pools and try a tenth less (expert_cache.rs: the loop in new).
    double frac = std::min(1.0, total ? double(budget) / double(total) : 1.0);
    for (;;) {
        bool ok = true;
        cache->pools_.clear();
        for (const auto& ps : pool_specs) {
            Pool p;
            p.slot_bytes = ps.first;
            const size_t slots = size_t(frac * double(ps.second));
            std::string e2;
            const bool made = slots == 0 || (gpu.alloc(ps.first * slots, p.buf, e2) && gpu.memset(p.buf, e2));
            if (!made) {
                if (frac > 0.01) {
                    ok = false;
                    break;
                }
                err = e2;
                return false;
            }
            cache->pools_.push_back(std::move(p));
        }
        if (ok) break;
        frac *= 0.9;
    }

    cache->layers_.clear();
    for (const auto& s : specs) {
        if (!s) {
            cache->layers_.push_back(std::nullopt);
            continue;
        }
        LayerExperts lx;
        lx.pool = s->pid;
        for (size_t i = 0; i < 4; i++) lx.mats[i] = s->mats[i];
        lx.n_mats = s->n_mats;
        lx.fused_gu = s->fused_gu;
        lx.has_scales = s->has_scales;
        lx.table.assign(n, -1);
        if (!gpu.alloc(4 * n, lx.table_dev, err)) return false;
        cache->layers_.push_back(std::move(lx));
    }

    // Initial fill: hottest first by profile; ties (and no profile) keep layer order. The experts that fit, read
    // ahead in upload order.
    std::vector<size_t> order;
    for (size_t i = 0; i < n_layers * n; i++) {
        if (cache->layers_[i / n]) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cache->counts_[a] > cache->counts_[b]; });
    {
        std::vector<size_t> room(cache->pools_.size());
        for (size_t p = 0; p < room.size(); p++) room[p] = cache->pools_[p].buf.len() / cache->pools_[p].slot_bytes;
        std::vector<size_t> fit;
        for (size_t i : order) {
            size_t& r = room[cache->layers_[i / n]->pool];
            if (r > 0) {
                r--;
                fit.push_back(i);
            }
        }
        order = std::move(fit);
    }

    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t i : order) {
        const LayerExperts& lx = *cache->layers_[i / n];
        for (size_t mi = 0; mi < lx.n_mats; mi++) {
            Tensor x;
            if (!expert_slice(m, i / n, lx, mi, i % n, n, x, err)) return false;
            ranges.push_back({reinterpret_cast<size_t>(x.data), x.len});
        }
    }
    {
        Prefetch pre;
        pre.start();
        pre.send(std::move(ranges));
        // `pre` joins its thread when it leaves scope, after the placement below has read the same pages.
        for (size_t i : order) {
            LayerExperts& lx = *cache->layers_[i / n];
            Pool& pool = cache->pools_[lx.pool];
            const size_t slot = pool.owner.size();
            pool.owner.push_back({i / n, i % n});
            if (!cache->place(m, gpu, i / n, i % n, slot, err)) return false;
        }
    }
    for (size_t il = 0; il < n_layers; il++) {
        if (cache->layers_[il] && !cache->sync_table(gpu, il, err)) return false;
    }
    // STRATA_LOAD_TRACE only: which pools were built and which physical blocks they hold (the looped
    // architectures key the cache by the physical blocks — a wrong keying shows up here as a pool
    // whose slot bytes do not match its blocks' tensors).
    if (std::getenv("STRATA_LOAD_TRACE")) {
        for (size_t p = 0; p < cache->pools_.size(); p++) {
            size_t lo = SIZE_MAX, hi = 0, owners = 0;
            for (const auto& [l, e] : cache->pools_[p].owner) {
                (void)e;
                lo = std::min(lo, l);
                hi = std::max(hi, l);
                owners++;
            }
            std::fprintf(stderr, "expert cache: pool %zu: slot %.2f MB, experts %zu (phys blocks %zu..%zu)\n",
                         p, double(cache->pools_[p].slot_bytes) / 1048576.0, owners, lo, hi);
        }
    }
    out = std::move(cache);
    return true;
}

// port of expert_cache.rs: ExpertCache::place
bool ExpertCache::place(const Model& m, Backend& gpu, size_t il, size_t e, size_t slot, std::string& err) {
    LayerExperts& lx = *layers_[il];
    const Pool& pool = pools_[lx.pool];
    for (size_t i = 0; i < lx.n_mats; i++) {
        Tensor x;
        if (!expert_slice(m, il, lx, i, e, n_expert_, x, err)) return false;
        const ExpertMat& mat = lx.mats[i];
        if (x.len != mat.bytes) {
            err = "expert cache: expert matrix of layer " + std::to_string(il) + " changes size";
            return false;
        }
        if (!gpu.htod(x.data, x.len, pool.buf, slot * pool.slot_bytes + mat.off, err)) return false;
        cold(x.data, x.len);
    }
    lx.table[e] = static_cast<int32_t>(slot);
    return true;
}

// port of expert_cache.rs: ExpertCache::sync_table
bool ExpertCache::sync_table(Backend& gpu, size_t il, std::string& err) {
    if (!layers_[il]) return true;
    const LayerExperts& lx = *layers_[il];
    return gpu.htod(lx.table.data(), 4 * lx.table.size(), lx.table_dev, 0, err);
}

// port of expert_cache.rs: ExpertCache::layer
const LayerExperts* ExpertCache::layer(size_t il) const {
    if (il >= layers_.size() || !layers_[il]) return nullptr;
    return &*layers_[il];
}

// port of expert_cache.rs: ExpertCache::base
std::pair<uint64_t, size_t> ExpertCache::base(const LayerExperts& lx) const {
    const Pool& p = pools_[lx.pool];
    return {p.buf.addr(), p.slot_bytes};
}

// port of expert_cache.rs: ExpertCache::slot
int32_t ExpertCache::slot(size_t il, size_t e) const {
    const LayerExperts* lx = layer(il);
    return lx && e < lx->table.size() ? lx->table[e] : -1;
}

// port of expert_cache.rs: ExpertCache::resident_count
size_t ExpertCache::resident_count() const {
    size_t n = 0;
    for (const auto& p : pools_) n += p.owner.size();
    return n;
}

// port of expert_cache.rs: ExpertCache::total_experts
size_t ExpertCache::total_experts() const {
    size_t n = 0;
    for (const auto& l : layers_) n += l ? 1 : 0;
    return n * n_expert_;
}

// port of expert_cache.rs: ExpertCache::slot_bytes_total
size_t ExpertCache::slot_bytes_total() const {
    size_t n = 0;
    for (const auto& p : pools_) n += p.buf.len();
    return n;
}

// port of expert_cache.rs: ExpertCache::active_set_bytes
size_t ExpertCache::active_set_bytes(size_t il, size_t k) const {
    const LayerExperts* lx = layer(il);
    return lx ? k * pools_[lx->pool].slot_bytes : 0;
}

// port of expert_cache.rs: ExpertCache::token_bytes
size_t ExpertCache::token_bytes(size_t k) const {
    size_t n = 0;
    for (size_t il = 0; il < layers_.size(); il++) n += active_set_bytes(il, k);
    return n;
}

// port of expert_cache.rs: ExpertCache::record
void ExpertCache::record(size_t il, const std::vector<int32_t>& ids) {
    recorded_ = true;
    for (int32_t e : ids) {
        counts_[il * n_expert_ + size_t(e)] += 1.0f;
        long_[il * n_expert_ + size_t(e)] += 1.0f;
    }
}

// port of expert_cache.rs: ExpertCache::record_prompt
void ExpertCache::record_prompt(size_t il, const std::vector<int32_t>& ids) {
    for (int32_t e : ids) prompt_[il * n_expert_ + size_t(e)] += 1.0f;
}

// port of expert_cache.rs: ExpertCache::settle
bool ExpertCache::settle(const Model& m, Backend& gpu, std::string& err) {
    const size_t n = n_expert_;
    if (!profile_.empty()) {
        for (size_t il = 0; il < layers_.size(); il++) {
            for (size_t i = il * n; i < (il + 1) * n; i++) counts_[i] += prompt_[i];
            float sum = 0.0f;
            for (size_t i = il * n; i < (il + 1) * n; i++) sum += counts_[i];
            if (sum > steady_) {
                for (size_t i = il * n; i < (il + 1) * n; i++) counts_[i] *= steady_ / sum;
            }
        }
        if (!swap(m, gpu, SIZE_MAX, err)) return false;
    }
    std::fill(prompt_.begin(), prompt_.end(), 0.0f);
    return true;
}

// port of expert_cache.rs: ExpertCache::reset_counts
void ExpertCache::reset_counts() { std::fill(counts_.begin(), counts_.end(), 0.0f); }

// port of expert_cache.rs: ExpertCache::adapt
bool ExpertCache::adapt(const Model& m, Backend& gpu, size_t max_swaps, std::string& err) {
    if (profile_.empty()) return true;
    for (float& c : counts_) c *= kKeep;
    for (float& c : long_) c *= kLongKeep;
    return swap(m, gpu, max_swaps, err);
}

// port of expert_cache.rs: ExpertCache::swap. Per pool: the hottest missed experts (up to max_swaps) replace the
// coldest residents while they are clearly hotter (1.5x + 2).
bool ExpertCache::swap(const Model& m, Backend& gpu, size_t max_swaps, std::string& err) {
    const size_t n = n_expert_;
    for (size_t p = 0; p < pools_.size(); p++) {
        std::vector<size_t> hot;
        for (size_t i = 0; i < counts_.size(); i++) {
            const auto& l = layers_[i / n];
            if (l && l->pool == p && l->table[i % n] < 0) hot.push_back(i);
        }
        smallest(hot, max_swaps, [&](size_t a, size_t b) { return counts_[a] > counts_[b]; });
        auto cnt = [&](size_t slot) {
            const auto [il, e] = pools_[p].owner[slot];
            return counts_[il * n + e];
        };
        std::vector<size_t> cold_slots(pools_[p].owner.size());
        std::iota(cold_slots.begin(), cold_slots.end(), size_t(0));
        smallest(cold_slots, max_swaps, [&](size_t a, size_t b) { return cnt(a) < cnt(b); });
        std::vector<size_t> touched;
        for (size_t j = 0; j < hot.size() && j < cold_slots.size() && j < max_swaps; j++) {
            const size_t h = hot[j], slot = cold_slots[j];
            if (counts_[h] < 1.5f * cnt(slot) + 2.0f) break;
            const auto [oil, oe] = pools_[p].owner[slot];
            layers_[oil]->table[oe] = -1;
            pools_[p].owner[slot] = {h / n, h % n};
            if (!place(m, gpu, h / n, h % n, slot, err)) return false;
            touched.push_back(oil);
            touched.push_back(h / n);
            swaps++;
        }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (size_t il : touched) {
            if (!sync_table(gpu, il, err)) return false;
        }
    }
    return true;
}

// port of expert_cache.rs: ExpertCache::save_profile. A run that recorded nothing (a fully resident cache skips
// record) leaves the saved profile alone.
bool ExpertCache::save_profile(const Model& m, std::string& err) const {
    (void)m;
    if (profile_.empty() || !recorded_) return true;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(profile_).parent_path(), ec);
    const std::vector<float> existing = read_floats(profile_);
    std::vector<bool> moe;
    for (const auto& l : layers_) moe.push_back(l.has_value());
    const std::vector<float> all = profile_update(existing, n_expert_, long_, moe);
    std::ofstream f(profile_, std::ios::binary | std::ios::trunc);
    if (!f) {
        err = "cannot write the routing profile " + profile_;
        return false;
    }
    f.write(reinterpret_cast<const char*>(all.data()), std::streamsize(all.size() * 4));
    return bool(f);
}

}  // namespace strata
