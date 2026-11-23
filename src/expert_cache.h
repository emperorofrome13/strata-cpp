// port of expert_cache.rs: the per-layer expert pools, the expert-to-slot tables the decode kernels read, the routing
// profile (initial placement, record / settle / adapt / swap, save_profile).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend.h"
#include "model.h"

namespace strata {

// port of expert_cache.rs: MATS. The three separate expert matrices of the qwen families.
inline constexpr const char* kExpertMats[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
// StrataFreet addition (INKLING2_NOTES.md): inkling2 stores the gate and up projections fused in one
// tensor (ffn_gate_up_exps.weight, {n_embd, 2*n_ff, n_expert}; rows 0..n_ff-1 are gate, the rest up),
// and the NVFP4 blocks carry a per-expert f32 scale per matrix (ffn_gate_up_exps.scale /
// ffn_down_exps.scale, {n_expert} floats each). The cache keeps at most 4 mats: gate_up, down, and the
// two one-float-per-expert scales when the file has them.
inline constexpr const char* kExpertMatsInkling2[4] = {
    "ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_up_exps.scale", "ffn_down_exps.scale"};

// port of expert_cache.rs: Mat
struct ExpertMat {
    Ty ty = Ty::F32;
    size_t rows = 0;
    size_t cols = 0;
    size_t off = 0;    // byte offset of this matrix inside one expert's slot
    size_t bytes = 0;  // bytes of one expert's matrix
};

// port of expert_cache.rs: LayerExperts
struct LayerExperts {
    size_t pool = 0;
    // StrataFreet addition: at most 4 mats (see kExpertMatsInkling2); the qwen families use 3.
    ExpertMat mats[4];
    size_t n_mats = 3;
    // The gate/up projections are fused (mats[0] = gate_up {n_embd, 2*n_ff}; mats[1] = down), and
    // mats[2]/mats[3] hold the per-expert f32 scales (4 bytes each, one per expert) when present.
    bool fused_gu = false;
    bool has_scales = false;
    std::vector<int32_t> table;  // expert id -> slot, -1 when the expert is on the CPU
    Buf table_dev;               // the table on the device (n_expert int32)
};

// port of expert_cache.rs: cache_dir. $XDG_CACHE_HOME, %LOCALAPPDATA% on Windows, else ~/.cache, plus rusting_strata.
std::string cache_dir();

// port of expert_cache.rs: ExpertCache
class ExpertCache {
public:
    // port of expert_cache.rs: ExpertCache::new. Slots for at most `budget` bytes of the MoE layers in
    // [layer_begin, layer_end), filled hottest first from the saved routing profile (ties and no profile: layer
    // order). Every expert that does not fit is a CPU expert (the mailbox path of the CPU worker).
    static bool create(const Model& m, Backend& gpu, size_t budget, size_t layer_begin, size_t layer_end,
                       std::unique_ptr<ExpertCache>& out, std::string& err);

    // port of expert_cache.rs: ExpertCache::layer (null when layer `il` has no experts in this cache)
    const LayerExperts* layer(size_t il) const;
    // port of expert_cache.rs: ExpertCache::base (device address and slot stride of a layer's pool)
    std::pair<uint64_t, size_t> base(const LayerExperts& lx) const;
    // port of expert_cache.rs: ExpertCache::slot (-1 when the expert is not resident)
    int32_t slot(size_t il, size_t e) const;
    // port of expert_cache.rs: ExpertCache::resident (the expert has a slot)
    bool resident(size_t il, size_t e) const { return slot(il, e) >= 0; }
    // port of expert_cache.rs: ExpertCache::resident_count / total_experts / fully_resident
    size_t resident_count() const;
    size_t total_experts() const;
    bool fully_resident() const { return total_experts() > 0 && resident_count() == total_experts(); }
    // port of expert_cache.rs: ExpertCache::slot_bytes_total
    size_t slot_bytes_total() const;
    // port of expert_cache.rs: ExpertCache::layers_len
    size_t layers_len() const { return layers_.size(); }
    // port of expert_cache.rs: ExpertCache::active_set_bytes (k slots of a layer's pool)
    size_t active_set_bytes(size_t il, size_t k) const;
    // port of expert_cache.rs: ExpertCache::token_bytes (routed-expert bytes one token of k experts reads)
    size_t token_bytes(size_t k) const;

    // port of expert_cache.rs: ExpertCache::record (routing of layer `il` for one token; counts and long)
    void record(size_t il, const std::vector<int32_t>& ids);
    // port of expert_cache.rs: ExpertCache::record_prompt (routing of a prefill chunk, for settle)
    void record_prompt(size_t il, const std::vector<int32_t>& ids);
    // port of expert_cache.rs: ExpertCache::settle (after a prefill: the prompt's routing joins counts, then swap all)
    bool settle(const Model& m, Backend& gpu, std::string& err);
    // port of expert_cache.rs: ExpertCache::reset_counts
    void reset_counts();
    // port of expert_cache.rs: ExpertCache::adapt (decay, then swap up to max_swaps per pool)
    bool adapt(const Model& m, Backend& gpu, size_t max_swaps, std::string& err);
    // port of expert_cache.rs: ExpertCache::save_profile (no-op when nothing was recorded or the profile is off)
    bool save_profile(const Model& m, std::string& err) const;
    // port of expert_cache.rs: ExpertCache::profile_path (empty under STRATA_FIXED_EXPERTS)
    const std::string& profile_path() const { return profile_; }

    // Swaps done since the load (port of expert_cache.rs: ExpertCache::swaps).
    size_t swaps = 0;

private:
    struct Pool {
        Buf buf;
        size_t slot_bytes = 0;
        std::vector<std::pair<size_t, size_t>> owner;  // slot -> (layer, expert)
    };
    // port of expert_cache.rs: ExpertCache::place (copy expert e of layer il into slot of its pool)
    bool place(const Model& m, Backend& gpu, size_t il, size_t e, size_t slot, std::string& err);
    // port of expert_cache.rs: ExpertCache::sync_table
    bool sync_table(Backend& gpu, size_t il, std::string& err);
    // port of expert_cache.rs: ExpertCache::swap
    bool swap(const Model& m, Backend& gpu, size_t max_swaps, std::string& err);

    std::vector<Pool> pools_;
    std::vector<std::optional<LayerExperts>> layers_;
    size_t n_expert_ = 0;
    std::vector<float> counts_;   // decayed routing counts [layer * n_expert + expert], for adapt
    std::vector<float> long_;     // the same counts with a memory of ~10000 tokens: profile and initial placement
    std::vector<float> prompt_;   // routing of the prompt being prefilled, until settle
    float steady_ = 0.0f;         // sum of one layer's counts in steady state
    std::string profile_;         // empty: STRATA_FIXED_EXPERTS (placement and profile kept, never adapts)
    bool recorded_ = false;
};

}  // namespace strata
