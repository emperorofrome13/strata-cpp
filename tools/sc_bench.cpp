// sc_bench: the C++ twin of `rustingstrata bench` (main.rs, Cmd::Bench; bench_record; GpuModel::finish's stat lines).
//
//   sc_bench --model <file.gguf> --prompts <ids.json> [--gpu] [--pp N[,N...]] [--tg N] [--reps N] [--ctx N]
//            [--vram-margin MB] [--threads N] [--json-out <file>]
//
// The prompts. `bench` tokenizes the same prose for every pp (main.rs prose: the paragraph repeated, the first pp
// tokens). This binary has no tokenizer, so the ids come in a JSON file written with the model's tokenizer:
//   {"eos": [ids], "prompts": {"512": [512 ids], ...}}
// tests/test_prefill_14b.py writes it from the same paragraph (StrataFreet's tokenizer reads the GGUF vocabulary).
//
// The output is the same as `rustingstrata bench`: one stdout line per pp
//   pp    512: prefill MEAN (MIN-MAX) tok/s | tg 128: decode MEAN (MIN-MAX) tok/s
// the `plan:` and `expert traffic:` lines of the load and per pp (stderr), then the `expert hit rate`, `worker busy`
// and `mailbox traffic` lines (stderr). `--json-out` writes one line of strata.bench.v1.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "../src/capi.h"

namespace {

// port of main.rs: round1
double round1(double x) { return std::round(x * 10.0) / 10.0; }

// port of main.rs: Stat (mean, min, max of the rep values, rounded to 0.1 as Stat::of does)
struct Stat {
    double mean = 0, min = 0, max = 0;
    static Stat of(const std::vector<double>& v) {
        Stat s;
        if (v.empty()) return s;
        double lo = INFINITY, hi = 0;
        double sum = 0;
        for (double x : v) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
            sum += x;
        }
        s.mean = round1(sum / static_cast<double>(v.size()));
        s.min = round1(lo);
        s.max = round1(hi);
        return s;
    }
    std::string text() const {
        char b[96];
        std::snprintf(b, sizeof b, "%.1f (%.1f-%.1f)", mean, min, max);
        return b;
    }
};

// A number under `key` in a flat JSON object (the stats string and the options this file writes).
double jnum(const std::string& js, const char* key, double dflt) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = js.find(pat);
    if (p == std::string::npos) return dflt;
    p = js.find(':', p + pat.size());
    if (p == std::string::npos) return dflt;
    char* end = nullptr;
    const double v = std::strtod(js.c_str() + p + 1, &end);
    return end == js.c_str() + p + 1 ? dflt : v;
}

// The integer array that starts at the first '[' at or after `pos`; `pos` moves past its ']'.
bool ids_at(const std::string& js, size_t& pos, std::vector<int32_t>& out) {
    const size_t p = js.find('[', pos);
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
    pos = q + 1;
    return true;
}

// The ids of prompt length `n` under "prompts", and the "eos" ids.
bool load_prompts(const std::string& js, size_t n, std::vector<int32_t>& ids, std::vector<int32_t>& eos) {
    size_t pos = js.find("\"eos\"");
    if (pos == std::string::npos || !ids_at(js, pos, eos)) return false;
    const size_t pp = js.find("\"prompts\"");
    if (pp == std::string::npos) return false;
    const std::string key = "\"" + std::to_string(n) + "\"";
    const size_t k = js.find(key, pp);
    if (k == std::string::npos) return false;
    size_t at = k + key.size();
    return ids_at(js, at, ids);
}

// port of main.rs: the greedy decode of `decode` (Sampler::greedy, no MTP): the argmax of the prefill's logits, then
// one step per accepted token, the next token picked by the device argmax of the step before. Stops at an EOS id.
bool greedy_decode(void* h, std::vector<float> logits, size_t tg, const std::vector<int32_t>& eos, size_t& count) {
    int32_t next = 0;
    {
        size_t best = 0;
        for (size_t i = 1; i < logits.size(); i++) {
            if (logits[i] > logits[best]) best = i;
        }
        next = static_cast<int32_t>(best);
    }
    count = 0;
    while (count < tg) {
        if (std::find(eos.begin(), eos.end(), next) != eos.end()) break;
        int32_t ahead = 0;
        if (sc_decode_sample(h, next, "{\"temp\":0}", &ahead) != 0) return false;
        count++;
        next = ahead;
    }
    return true;
}

struct Result {
    size_t pp = 0;
    Stat prefill, decode;
    size_t bytes_per_token = 0;
    double decode_median = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::string model, prompts_path, json_out;
    std::vector<size_t> pps;
    size_t tg = 128, reps = 3, ctx = 4096, threads = 0;
    size_t chunk = SIZE_MAX;  // StrataFreet addition: the prefill chunk (0 = token by token; default: the load's)
    double margin_mb = 512;
    // StrataFreet addition: sc_load options "ram" (off, copy, lock) and "pin" (true or false).
    std::string ram = "off";
    bool pin = false;
    // StrataFreet addition (eviction test): after the load, read these files in order until `evict_gb` GB have been
    // read, so the Windows file cache churns before the decode is timed.
    std::vector<std::string> evict_files;
    double evict_gb = 0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](void) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "sc_bench: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") {
            model = next();
        } else if (a == "--prompts") {
            prompts_path = next();
        } else if (a == "--pp") {
            std::stringstream ss(next());
            std::string item;
            while (std::getline(ss, item, ',')) pps.push_back(static_cast<size_t>(std::atoll(item.c_str())));
        } else if (a == "--tg") {
            tg = static_cast<size_t>(std::atoll(next().c_str()));
        } else if (a == "--reps") {
            reps = static_cast<size_t>(std::atoll(next().c_str()));
        } else if (a == "--ctx") {
            ctx = static_cast<size_t>(std::atoll(next().c_str()));
        } else if (a == "--vram-margin") {
            margin_mb = std::atof(next().c_str());
        } else if (a == "--threads") {
            threads = static_cast<size_t>(std::atoll(next().c_str()));
        } else if (a == "--chunk") {
            chunk = static_cast<size_t>(std::atoll(next().c_str()));
        } else if (a == "--json-out") {
            json_out = next();
        } else if (a == "--ram") {
            ram = next();
        } else if (a == "--pin") {
            pin = true;
        } else if (a == "--evict") {
            std::stringstream ss(next());
            std::string item;
            while (std::getline(ss, item, ',')) evict_files.push_back(item);
        } else if (a == "--evict-gb") {
            evict_gb = std::atof(next().c_str());
        } else if (a == "--gpu") {
            // the CUDA backend is the only one: accepted for the CLI of `rustingstrata bench`
        } else {
            std::fprintf(stderr,
                         "usage: sc_bench --model <gguf> --prompts <ids.json> [--pp N[,N...]] [--tg N] [--reps N] [--ctx N] "
                         "[--vram-margin MB] [--threads N] [--json-out <file>] [--ram off|copy|lock] [--pin] [--gpu]\n");
            return 2;
        }
    }
    if (model.empty() || prompts_path.empty()) {
        std::fprintf(stderr, "sc_bench: --model and --prompts are required\n");
        return 2;
    }
    if (pps.empty()) pps.push_back(512);
    if (reps == 0 || tg == 0) {
        std::fprintf(stderr, "sc_bench: --reps and --tg must be at least 1\n");
        return 2;
    }
    if (json_out == "-") {
        std::fprintf(stderr, "sc_bench: --json-out takes a file path, not -\n");
        return 2;
    }
    std::string pjs;
    {
        std::ifstream f(prompts_path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "sc_bench: cannot read %s\n", prompts_path.c_str());
            return 1;
        }
        pjs.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // `engine.ctx = engine.ctx.max(pp + tg)` of the bench command.
    size_t max_pp = 0;
    for (size_t n : pps) max_pp = std::max(max_pp, n);
    ctx = std::max(ctx, max_pp + tg);
    std::ostringstream opts;
    opts << "{\"ctx\":" << ctx << ",\"kv\":\"f16\",\"vram_margin_mb\":" << margin_mb << ",\"threads\":" << threads
         << ",\"ram\":\"" << ram << "\",\"pin\":" << (pin ? "true" : "false");
    if (chunk != SIZE_MAX) opts << ",\"chunk\":" << chunk;
    opts << "}";
    const auto t_load0 = std::chrono::steady_clock::now();
    void* h = sc_load(model.c_str(), opts.str().c_str());
    if (!h) {
        std::fprintf(stderr, "sc_bench: load failed: %s\n", sc_last_error());
        return 1;
    }
    std::fprintf(stderr, "load: %.1f s (ram %s%s)\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count(), ram.c_str(),
                 pin ? ", pin" : "");
    std::fprintf(stderr, "stats after load: %s\n", sc_stats(h));
    if (!evict_files.empty() && evict_gb > 0) {
        // Reads the files sequentially in 64 MB pieces until evict_gb GB are read (or the files end).
        const double want = evict_gb * 1073741824.0;
        double got = 0;
        std::vector<char> buf(size_t(64) << 20);
        const auto t_ev = std::chrono::steady_clock::now();
        for (const std::string& path : evict_files) {
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                std::fprintf(stderr, "sc_bench: cannot read %s for the eviction\n", path.c_str());
                sc_free(h);
                return 1;
            }
            while (got < want && f) {
                f.read(buf.data(), (std::streamsize)buf.size());
                const double n = (double)f.gcount();
                got += n;
                if (n <= 0) break;
            }
            if (got >= want) break;
        }
        const double ev_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ev).count();
        std::fprintf(stderr, "eviction: read %.2f GB of other files in %.1f s (%.0f MB/s)\n", got / 1073741824.0, ev_s,
                     ev_s > 0 ? got / 1e6 / ev_s : 0.0);
    }
    const size_t n_vocab = static_cast<size_t>(sc_n_vocab(h));
    std::vector<Result> results;
    std::vector<float> logits(n_vocab);
    for (size_t n : pps) {
        std::vector<int32_t> ids, eos;
        if (!load_prompts(pjs, n, ids, eos) || ids.size() != n) {
            std::fprintf(stderr, "sc_bench: %s has no prompt of %zu ids under \"prompts\"\n", prompts_path.c_str(), n);
            sc_free(h);
            return 1;
        }
        std::vector<double> p, d;
        for (size_t rep = 0; rep < reps; rep++) {
            if (sc_reset(h, 0) != 0) {
                std::fprintf(stderr, "sc_bench: %s\n", sc_last_error());
                sc_free(h);
                return 1;
            }
            auto t0 = std::chrono::steady_clock::now();
            if (sc_prefill_logits(h, ids.data(), static_cast<int>(n), logits.data()) != 0) {
                std::fprintf(stderr, "sc_bench: %s\n", sc_last_error());
                sc_free(h);
                return 1;
            }
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            p.push_back(static_cast<double>(n) / secs);
            t0 = std::chrono::steady_clock::now();
            size_t k = 0;
            if (!greedy_decode(h, logits, tg, eos, k)) {
                std::fprintf(stderr, "sc_bench: %s\n", sc_last_error());
                sc_free(h);
                return 1;
            }
            secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            d.push_back(static_cast<double>(k) / secs);
            // StrataFreet addition: one line per rep (the eviction test reads the decode rate of each rep).
            std::fprintf(stderr, "rep %zu/%zu: prefill %.1f tok/s, decode %.1f tok/s (%zu tokens in %.1f s)\n", rep + 1,
                         reps, p.back(), d.back(), k, secs);
        }
        Result r;
        r.pp = n;
        r.prefill = Stat::of(p);
        r.decode = Stat::of(d);
        std::printf("pp %6zu: prefill %s tok/s | tg %zu: decode %s tok/s\n", n, r.prefill.text().c_str(), tg,
                    r.decode.text().c_str());
        // The expert bytes one decode token reads, and the median decode rate the `expert traffic:` line prints.
        const std::string st = sc_stats(h);
        const double eb = jnum(st, "expert_bytes_per_token", 0);
        r.bytes_per_token = static_cast<size_t>(eb);
        if (r.bytes_per_token > 0) {
            std::vector<double> v = d;
            std::sort(v.begin(), v.end());
            r.decode_median = v[v.size() / 2];
            std::fprintf(stderr, "expert traffic: %.1f MB/token -> %.1f GB/s at %.1f tok/s\n", eb / 1048576.0,
                         eb * r.decode_median / 1e9, r.decode_median);
        }
        results.push_back(r);
    }

    // port of GpuModel::finish's stat lines (after the last pp, before the profile is saved)
    const std::string st = sc_stats(h);
    const double hits = jnum(st, "hits", 0), misses = jnum(st, "misses", 0), swaps = jnum(st, "swaps", 0);
    const double tokens = jnum(st, "tokens", 0), busy_ms_tok = jnum(st, "worker_busy_ms_per_token", 0);
    const double stalls = jnum(st, "resumed", 0), writes = jnum(st, "mailbox_writes", 0), reads = jnum(st, "mailbox_reads", 0);
    const double resident = jnum(st, "experts_resident", 0), total = jnum(st, "experts_total", 0);
    const bool fully = st.find("\"fully_resident\":true") != std::string::npos;
    std::fprintf(stderr, "expert hit rate %.1f%% (%.0f misses), cache swaps %.0f\n",
                 100.0 * hits / std::max(hits + misses, 1.0), misses, swaps);
    std::fprintf(stderr, "worker busy %.1f µs/token, worker stalls %.0f\n", tokens > 0 ? busy_ms_tok * 1000.0 : 0.0, stalls);
    const std::string mode = fully ? "fast path on, " : "fast path off, ";
    std::fprintf(stderr, "mailbox traffic: %.0f GPU writes, %.0f host routing reads (%s%.0f/%.0f experts resident)\n", writes,
                 reads, mode.c_str(), resident, total);
    if (sc_save_profile(h) != 0) {
        std::fprintf(stderr, "sc_bench: %s\n", sc_last_error());
    }

    if (!json_out.empty()) {
        // port of main.rs: bench_record (strata.bench.v1), one line
        const std::string name = [&] {
            const size_t s = model.find_last_of("/\\");
            return s == std::string::npos ? model : model.substr(s + 1);
        }();
        std::string esc;
        for (char c : model) {
            if (c == '\\' || c == '"') esc += '\\';
            esc += c;
        }
        std::ostringstream o;
        o << "{\"schema\":\"strata.bench.v1\",\"unix_time\":" << static_cast<long long>(std::time(nullptr))
          << ",\"backend\":\"cuda\",\"device\":[0],\"model\":{\"path\":\"" << esc << "\",\"name\":\"" << name << "\"}"
          << ",\"ctx\":" << ctx << ",\"chunk\":" << static_cast<long long>(jnum(st, "prefill_chunk", 0))
          << ",\"kv\":\"f16\",\"mtp\":0"
          << ",\"tg\":" << tg << ",\"reps\":" << reps << ",\"results\":[";
        for (size_t i = 0; i < results.size(); i++) {
            const Result& r = results[i];
            char b[512];
            std::snprintf(b, sizeof b,
                          "%s{\"pp\":%zu,\"prefill_tok_s\":{\"mean\":%.1f,\"min\":%.1f,\"max\":%.1f},"
                          "\"decode_tok_s\":{\"mean\":%.1f,\"min\":%.1f,\"max\":%.1f}",
                          i ? "," : "", r.pp, r.prefill.mean, r.prefill.min, r.prefill.max, r.decode.mean, r.decode.min,
                          r.decode.max);
            o << b;
            if (r.bytes_per_token > 0) {
                std::snprintf(b, sizeof b,
                              ",\"expert\":{\"bytes_per_token\":%zu,\"mb_per_token\":%.1f,\"gb_per_s\":%.1f,\"at_decode_tok_s\":%.1f}",
                              r.bytes_per_token, round1(static_cast<double>(r.bytes_per_token) / 1048576.0),
                              round1(static_cast<double>(r.bytes_per_token) * r.decode_median / 1e9), round1(r.decode_median));
                o << b;
            }
            o << "}";
        }
        o << "]";
        if (jnum(st, "expert_bytes_per_token", 0) > 0) {
            const double eb = jnum(st, "expert_bytes_per_token", 0);
            char b[512];
            std::snprintf(b, sizeof b,
                          ",\"expert\":{\"hit_rate_pct\":%.1f,\"misses\":%.0f,\"swaps\":%.0f,\"resident\":%.0f,\"total\":%.0f,"
                          "\"bytes_per_token\":%.0f,\"mb_per_token\":%.1f}",
                          round1(100.0 * hits / std::max(hits + misses, 1.0)), misses, swaps, resident, total, eb,
                          round1(eb / 1048576.0));
            o << b;
        }
        o << "}\n";
        std::ofstream f(json_out, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "sc_bench: writing the JSON record to %s failed\n", json_out.c_str());
            sc_free(h);
            return 1;
        }
        f << o.str();
    }
    sc_free(h);
    return 0;
}
