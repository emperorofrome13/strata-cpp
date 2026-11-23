// port of model.rs: Model::load, Model::embed, Model::ple_rows and Model::ple_embed (weight model, hyperparameters,
// token embedding row, per-layer embedding hashes).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "model.h"

#include <algorithm>

namespace strata {

namespace {

// port of model.rs: PLE_NGRAM, PLE_HEADS, PLE_KERNEL (the geometry checked against the GGUF at load)
constexpr size_t kPleNgram = 3;
constexpr size_t kPleHeads = 8;
constexpr size_t kPleKernel = 4;

}  // namespace

// port of model.rs: Model::load
bool Model::load(const Gguf& g, std::string& err) {
    if (!w.init(&g, nullptr, err)) return false;
    cfg = w.config();
    if (cfg.has_eagle3) {
        err = "standalone EAGLE3 draft heads are not ported yet (STRATA_CPP_PLAN.md W3-E2)";
        return false;
    }
    if (cfg.n_embd_ple > 0) {
        // port of model.rs: the qwen4exp PLE metadata (`{arch}.ple.*`), checked against the PLE geometry
        const std::string pre = cfg.arch + ".ple.";
        auto arr = [&](const char* k, std::vector<uint64_t>& out) -> bool {
            const Value* v = g.get(pre + k);
            const std::vector<Value>* a = v ? v->as_arr() : nullptr;
            if (!a) {
                err = "missing " + pre + k;
                return false;
            }
            out.clear();
            for (const Value& x : *a) {
                const auto u = x.as_u64();
                if (!u) {
                    err = "bad " + pre + k;
                    return false;
                }
                out.push_back(*u);
            }
            return true;
        };
        auto integer = [&](const char* k, uint64_t& out) -> bool {
            const Value* v = g.get(pre + k);
            const auto u = v ? v->as_u64() : std::nullopt;
            if (!u) {
                err = "missing " + pre + k;
                return false;
            }
            out = *u;
            return true;
        };
        uint64_t ngram = 0, heads_per = 0, kernel = 0, eos = 0;
        if (!integer("ngram_size", ngram) || !integer("heads_per_ngram", heads_per) || !integer("conv_kernel", kernel)) return false;
        if (ngram != kPleNgram || heads_per != kPleHeads || kernel != kPleKernel) {
            err = "unsupported PLE geometry";
            return false;
        }
        if (!arr("layer_multipliers", ple.mult) || !arr("head_offsets", ple.off) || !arr("head_vocab_sizes", ple.vocab)) return false;
        if (!integer("eos_token_id", eos)) return false;
        ple.eos = static_cast<int64_t>(eos);
        const size_t heads = (kPleNgram - 1) * kPleHeads;
        bool vocab_ok = true;
        for (uint64_t v : ple.vocab) vocab_ok = vocab_ok && v > 0;
        if (ple.mult.size() != kPleNgram || ple.off.size() != heads || ple.vocab.size() != heads || !vocab_ok ||
            heads * cfg.n_embd_ple != cfg.n_embd) {
            err = "bad PLE metadata";
            return false;
        }
        Tensor table;
        if (!w.get("per_layer_token_embd.weight", table, err)) return false;
    }
    bool nextn = true;
    if (!w.validate(cfg.n_layer, nextn, err)) return false;
    if (!nextn && cfg.n_nextn > 0) cfg.n_nextn = 0;  // model.rs: the MTP layer has a quant type this engine lacks
    Tensor emb;
    if (!w.get("token_embd.weight", emb, err)) return false;
    // model.rs pre-touches the token embedding pages (put_embed never stalls on a page fault); the mapping is read
    // on demand here, so no pre-touch is needed.
    return true;
}

// port of model.rs: Model::embed
bool Model::embed(uint32_t token, std::vector<float>& out, std::string& err) const {
    Tensor emb;
    if (!w.get("token_embd.weight", emb, err)) return false;
    const size_t idx = std::min<size_t>(token, cfg.n_vocab > 0 ? cfg.n_vocab - 1 : 0);
    out.assign(cfg.n_embd, 0.0f);
    if (!emb.row(idx, out.data())) {
        err = "embed: token " + std::to_string(token) + " is outside the embedding table";
        return false;
    }
    return true;
}

// port of model.rs: Model::ple_rows. The context is the token and the previous two (EOS or the sequence start cut it:
// that position and every older one read as EOS). Advances `prev` (newest first, -1 before the start).
void Model::ple_rows(std::array<int64_t, 2>& prev, uint32_t token, std::vector<uint64_t>& rows) const {
    rows.clear();
    if (cfg.n_embd_ple == 0) return;
    uint64_t ctx[kPleNgram] = {token, token, token};
    bool cut = false;
    for (size_t s = 1; s < kPleNgram; s++) {
        const int64_t t = prev[s - 1];
        cut = cut || t < 0 || t == ple.eos;
        ctx[s] = cut ? static_cast<uint64_t>(ple.eos) : static_cast<uint64_t>(t);
    }
    prev = {static_cast<int64_t>(token), prev[0]};
    for (size_t n = 2; n <= kPleNgram; n++) {
        uint64_t m = ctx[0] * ple.mult[0];
        for (size_t j = 1; j < n; j++) m ^= ctx[j] * ple.mult[j];
        for (size_t h = (n - 2) * kPleHeads; h < (n - 1) * kPleHeads; h++) rows.push_back(m % ple.vocab[h] + ple.off[h]);
    }
}

// port of model.rs: Model::ple_embed
bool Model::ple_embed(std::array<int64_t, 2>& prev, uint32_t token, std::vector<float>& out, std::string& err) const {
    out.clear();
    if (cfg.n_embd_ple == 0) return true;
    std::vector<uint64_t> rows;
    ple_rows(prev, token, rows);
    Tensor table;
    if (!w.get("per_layer_token_embd.weight", table, err)) return false;
    const size_t d = cfg.n_embd_ple;
    out.assign(cfg.n_embd, 0.0f);
    for (size_t i = 0; i < rows.size(); i++) {
        if (!table.row(static_cast<size_t>(rows[i]), out.data() + i * d)) {
            err = "ple_embed: row " + std::to_string(rows[i]) + " is outside the per-layer embedding table";
            return false;
        }
    }
    return true;
}

}  // namespace strata
