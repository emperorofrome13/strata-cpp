// port of model.rs: the weight model (architecture from the GGUF, hyperparameters, the weight map, the token
// embedding row, the per-layer embedding (PLE) n-gram geometry). The scalar CPU forward of model.rs (`step_h`) is not
// ported in W3-E1a: the GPU decode path (gpu_model.cpp) is what the port runs, and its parity is measured against
// `rustingstrata score`. See MODIFICATIONS.md, sections "W3-E1a" and "W3-E1b".
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "gguf.h"
#include "weights.h"

namespace strata {

// port of model.rs: Ple (the qwen4exp per-layer embedding's n-gram hash constants, read from the GGUF metadata).
struct PleGeom {
    std::vector<uint64_t> mult, off, vocab;
    int64_t eos = 0;
};

// port of model.rs: Model (the parts the GPU path reads: cfg, w and the PLE geometry; no LoRA overlays, no auxiliary GGUF).
struct Model {
    ModelConfig cfg;
    Weights w;
    // port of model.rs: Model::ple (set when cfg.n_embd_ple > 0)
    PleGeom ple;

    // port of model.rs: Model::load / Model::load_with_aux (no aux file, no LoRA). Refuses the standalone EAGLE3 head.
    // A MTP (nextn) block the file carries but the quant policy rejects is dropped, as model.rs does.
    bool load(const Gguf& g, std::string& err);
    // port of model.rs: Model::embed (the row of `token`, clamped to the vocabulary, `cfg.n_embd` floats).
    bool embed(uint32_t token, std::vector<float>& out, std::string& err) const;
    // port of model.rs: Model::ple_rows (the 16 table rows of `token`: its 2- and 3-grams hashed; `prev` advances)
    void ple_rows(std::array<int64_t, 2>& prev, uint32_t token, std::vector<uint64_t>& rows) const;
    // port of model.rs: Model::ple_embed (the PLE input of `token`, `cfg.n_embd` floats: its 16 table rows, head after
    // head). Leaves `out` empty for a model without PLE.
    bool ple_embed(std::array<int64_t, 2>& prev, uint32_t token, std::vector<float>& out, std::string& err) const;
};

}  // namespace strata
