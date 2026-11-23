// sc_score: the C++ twin of `rustingstrata score` (main.rs, Cmd::Score). Teacher-forces token ids and prints, for each
// continuation position, the argmax and the top-5 (token, log-probability) pairs of the next-token distribution.
//
//   sc_score --model <file.gguf> --input <json | ->  [--ctx N] [--vram-margin-mb N]
//
// Input JSON: {"prompt": [ids], "continuation": [ids]}. Output (one line, the same keys as the Rust command):
//   {"argmax":[...],"top5":[[[id,logprob],...],...]}
// The prompt goes through sc_prefill_logits (the batched prefill; a prompt shorter than the `short` threshold, 32 tokens,
// goes token by token inside it, as the Rust command's `prefill` does). The continuation goes through sc_decode one
// token at a time, as the Rust command's `step`.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../src/capi.h"

namespace {

// The integer array under `key` in a flat JSON object.
bool json_ids(const std::string& js, const char* key, std::vector<int32_t>& out) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return false;
    p = js.find('[', p + pat.size());
    if (p == std::string::npos) return false;
    const size_t q = js.find(']', p);
    if (q == std::string::npos) return false;
    out.clear();
    const char* s = js.c_str() + p + 1;
    const char* end = js.c_str() + q;
    while (s < end) {
        while (s < end && (*s == ' ' || *s == ',' || *s == '\n' || *s == '\r' || *s == '\t')) s++;
        if (s >= end) break;
        char* e = nullptr;
        const long v = std::strtol(s, &e, 10);
        if (e == s) return false;
        out.push_back(static_cast<int32_t>(v));
        s = e;
    }
    return true;
}

// Argmax and top-5 of one logits row, computed as ops.rs does (first index on ties; log-softmax in f32).
void record(const std::vector<float>& logits, std::vector<int32_t>& argmax, std::vector<std::vector<std::pair<int32_t, float>>>& top5) {
    size_t best = 0;
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > logits[best]) best = i;
    }
    float mx = -INFINITY;
    for (float x : logits) mx = std::max(mx, x);
    float sum = 0.0f;
    for (float x : logits) sum += std::exp(x - mx);
    const float lse = mx + std::log(sum);
    std::vector<size_t> idx(logits.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](size_t a, size_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
    std::vector<std::pair<int32_t, float>> row;
    for (size_t j = 0; j < 5; j++) row.push_back({static_cast<int32_t>(idx[j]), logits[idx[j]] - lse});
    argmax.push_back(static_cast<int32_t>(best));
    top5.push_back(std::move(row));
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, input;
    int ctx = 4096;
    int margin_mb = 512;
    int threads = 0;
    // StrataFreet addition: sc_load options "ram" and "pin" (the parity check of the RAM modes runs this twice).
    std::string ram = "off";
    bool pin = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--input" && i + 1 < argc) {
            input = argv[++i];
        } else if (a == "--ctx" && i + 1 < argc) {
            ctx = std::atoi(argv[++i]);
        } else if (a == "--vram-margin-mb" && i + 1 < argc) {
            margin_mb = std::atoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (a == "--ram" && i + 1 < argc) {
            ram = argv[++i];
        } else if (a == "--pin") {
            pin = true;
        } else {
            std::fprintf(stderr, "usage: sc_score --model <gguf> --input <json|-> [--ctx N] [--vram-margin-mb N] [--threads N]\n");
            return 2;
        }
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr, "usage: sc_score --model <gguf> --input <json|->\n");
        return 2;
    }
    std::string text;
    if (input == "-") {
        text.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    } else {
        std::ifstream f(input, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "sc_score: cannot read %s\n", input.c_str());
            return 1;
        }
        text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::vector<int32_t> prompt, cont;
    if (!json_ids(text, "prompt", prompt) || !json_ids(text, "continuation", cont) || prompt.empty() || cont.empty()) {
        std::fprintf(stderr, "sc_score: input needs non-empty \"prompt\" and \"continuation\" id arrays\n");
        return 1;
    }

    std::ostringstream opts;
    opts << "{\"ctx\":" << ctx << ",\"kv\":\"f16\",\"vram_margin_mb\":" << margin_mb << ",\"threads\":" << threads
         << ",\"ram\":\"" << ram << "\",\"pin\":" << (pin ? "true" : "false") << "}";
    void* h = sc_load(model.c_str(), opts.str().c_str());
    if (!h) {
        std::fprintf(stderr, "sc_score: load failed: %s\n", sc_last_error());
        return 1;
    }
    const int nv = sc_n_vocab(h);
    std::vector<float> logits(static_cast<size_t>(nv));
    for (int32_t t : prompt) {
        if (t < 0 || t >= nv) {
            std::fprintf(stderr, "sc_score: token id %d out of range (vocabulary has %d tokens)\n", t, nv);
            sc_free(h);
            return 1;
        }
    }
    for (int32_t t : cont) {
        if (t < 0 || t >= nv) {
            std::fprintf(stderr, "sc_score: token id %d out of range (vocabulary has %d tokens)\n", t, nv);
            sc_free(h);
            return 1;
        }
    }

    // The prompt, then the continuation: position i (from prompt.size() - 1) predicts continuation[i - P + 1].
    const size_t P = prompt.size();
    std::vector<int32_t> argmax;
    std::vector<std::vector<std::pair<int32_t, float>>> top5;
    // The prompt through the batched prefill (prompts shorter than the `short` threshold go token by token inside it),
    // as `rustingstrata score` runs `prefill` on it.
    if (sc_prefill_logits(h, prompt.data(), static_cast<int>(P), logits.data()) != 0) {
        std::fprintf(stderr, "sc_score: %s\n", sc_last_error());
        sc_free(h);
        return 1;
    }
    record(logits, argmax, top5);
    for (size_t j = 0; j + 1 < cont.size(); j++) {
        if (sc_decode(h, cont[j], logits.data()) != 0) {
            std::fprintf(stderr, "sc_score: %s\n", sc_last_error());
            sc_free(h);
            return 1;
        }
        record(logits, argmax, top5);
    }

    std::string out = "{\"argmax\":[";
    for (size_t i = 0; i < argmax.size(); i++) out += (i ? "," : "") + std::to_string(argmax[i]);
    out += "],\"top5\":[";
    for (size_t i = 0; i < top5.size(); i++) {
        out += i ? ",[" : "[";
        for (size_t j = 0; j < top5[i].size(); j++) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "[%d,%.9g]", top5[i][j].first, static_cast<double>(top5[i][j].second));
            out += (j ? "," : "") + std::string(buf);
        }
        out += "]";
    }
    out += "]}";
    std::printf("%s\n", out.c_str());
    sc_free(h);
    return 0;
}
