// Development surface for the tests and tools: the GGUF inventory and the quant reference, driven from
// Python through ctypes. It is a separate DLL (build/strata_cpp_dev.dll) so the product ABI in capi.cpp
// stays exactly what STRATA_CPP_PLAN.md section 5 specifies.
#include "backend.h"
#include "config.h"
#include "cuda_backend.h"
#include "gguf.h"
#include "memory.h"
#include "quant.h"
#include "sampling.h"
#include "weights.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#define SC_EXPORT extern "C" __declspec(dllexport)
#else
#define SC_EXPORT extern "C"
#endif

using namespace strata;

namespace {

thread_local std::string g_err;
thread_local std::string g_json;

void json_escape(std::string& o, const std::string& s) {
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char)c;
                }
        }
    }
    o += '"';
}

// FNV-1a over each element's canonical bytes: "+" + string, or "#" + decimal for numbers.
// The Python side (tests/test_gguf.py) computes the same hash for the same array.
struct Fnv {
    uint64_t h = 0xcbf29ce484222325ULL;
    void bytes(const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; i++) {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
    }
    void str(const std::string& s) {
        bytes((const uint8_t*)"+", 1);
        bytes((const uint8_t*)s.data(), s.size());
    }
    void num(int64_t v) {
        char buf[32];
        int n = std::snprintf(buf, sizeof buf, "#%lld", (long long)v);
        bytes((const uint8_t*)buf, (size_t)n);
    }
};

const char* tag_name(Value::Tag t) {
    switch (t) {
        case Value::Tag::U8: return "u8";
        case Value::Tag::I8: return "i8";
        case Value::Tag::U16: return "u16";
        case Value::Tag::I16: return "i16";
        case Value::Tag::U32: return "u32";
        case Value::Tag::I32: return "i32";
        case Value::Tag::U64: return "u64";
        case Value::Tag::I64: return "i64";
        case Value::Tag::F32: return "f32";
        case Value::Tag::F64: return "f64";
        case Value::Tag::Bool: return "bool";
        case Value::Tag::Str: return "str";
        case Value::Tag::Arr: return "arr";
    }
    return "?";
}

void emit_value(std::string& o, const Value& v) {
    o += "{\"t\":";
    json_escape(o, tag_name(v.tag));
    switch (v.tag) {
        case Value::Tag::Str: o += ",\"v\":"; json_escape(o, v.s); break;
        case Value::Tag::Bool: o += std::string(",\"v\":") + (v.b ? "true" : "false"); break;
        case Value::Tag::F32: {
            char buf[40];
            std::snprintf(buf, sizeof buf, ",\"v\":%.9g", (double)*v.as_f32());
            o += buf;
            break;
        }
        case Value::Tag::F64: {
            double d;
            uint64_t bits = v.u;
            std::memcpy(&d, &bits, sizeof d);
            char buf[40];
            std::snprintf(buf, sizeof buf, ",\"v\":%.17g", d);
            o += buf;
            break;
        }
        case Value::Tag::Arr: {
            Fnv fnv;
            const Value* first = v.arr.empty() ? nullptr : &v.arr.front();
            const char* et = first ? tag_name(first->tag) : "empty";
            for (const Value& e : v.arr) {
                if (e.tag == Value::Tag::Str) {
                    fnv.str(e.s);
                } else if (e.tag == Value::Tag::Bool) {
                    fnv.num(e.b ? 1 : 0);
                } else if (e.tag == Value::Tag::I8 || e.tag == Value::Tag::I16 || e.tag == Value::Tag::I32 ||
                           e.tag == Value::Tag::I64) {
                    fnv.num(e.i);
                } else {
                    fnv.num((int64_t)*e.as_u64());
                }
            }
            char buf[64];
            o += ",\"et\":";
            json_escape(o, et);
            std::snprintf(buf, sizeof buf, ",\"n\":%zu,\"hash\":\"%016llx\"", v.arr.size(),
                          (unsigned long long)fnv.h);
            o += buf;
            break;
        }
        default: {
            char buf[32];
            std::snprintf(buf, sizeof buf, ",\"v\":%lld", (long long)(v.i ? v.i : (int64_t)v.u));
            o += buf;
            break;
        }
    }
    o += "}";
}

}  // namespace

SC_EXPORT const char* sci_last_error(void) { return g_err.c_str(); }

// Full GGUF inventory as JSON: metadata values (arrays by type/count/hash) and every tensor with its
// dims, type id and byte size. Returns nullptr on failure (see sci_last_error).
SC_EXPORT const char* sci_gguf_info_json(const char* path) {
    g_err.clear();
    g_json.clear();
    Gguf g;
    if (!g.open(path, g_err)) return nullptr;

    std::string& o = g_json;
    o += "{\"path\":";
    json_escape(o, path);
    o += ",\"n_meta\":";
    o += std::to_string(g.meta().size());
    o += ",\"n_tensors\":";
    o += std::to_string(g.tensors().size());

    // each shard's data-section offset, so a caller can turn a tensor's relative offset into a file offset
    o += ",\"data_offsets\":[";
    {
        size_t shards = 0;
        for (const auto& kv : g.tensors()) shards = std::max(shards, kv.second.file + 1);
        for (size_t i = 0; i < shards; i++) {
            if (i) o += ",";
            o += std::to_string((unsigned long long)g.data_offset(i));
        }
    }
    o += "]";

    o += ",\"meta\":{";
    bool first = true;
    for (const auto& kv : g.meta()) {
        if (!first) o += ",";
        first = false;
        json_escape(o, kv.first);
        o += ":";
        emit_value(o, kv.second);
    }
    o += "},\"tensors\":{";
    first = true;
    for (const auto& kv : g.tensors()) {
        const TensorInfo& t = kv.second;
        if (!first) o += ",";
        first = false;
        json_escape(o, t.name);
        o += ":{\"dims\":[";
        for (size_t i = 0; i < t.dims.size(); i++) {
            if (i) o += ",";
            o += std::to_string(t.dims[i]);
        }
        o += "],\"type\":";
        o += std::to_string(t.ty);
        o += ",\"type_name\":";
        auto ty = ty_from_u32(t.ty);
        json_escape(o, ty ? ty_name(*ty) : "unsupported");
        o += ",\"offset\":";
        o += std::to_string(t.offset);
        o += ",\"file\":";
        o += std::to_string(t.file);
        o += ",\"bytes\":";
        size_t n = 0;
        if (ty) {
            uint64_t rb = row_bytes(*ty, t.dims.empty() ? 0 : t.dims[0]);
            for (size_t d = 1; d < t.dims.size(); d++) rb *= t.dims[d];
            n = (size_t)rb;
        }
        o += std::to_string(n);
        o += "}";
    }
    o += "}}";
    return o.c_str();
}

// port of quant/mod.rs: GgmlType::row_bytes through the C ABI (-1 when the type is unknown)
SC_EXPORT long long sci_row_bytes(unsigned int type, long long n_elems) {
    auto ty = ty_from_u32(type);
    if (!ty || n_elems < 0) return -1;
    return (long long)row_bytes(*ty, (uint64_t)n_elems);
}

// Dequantizes n_elems values of `type` from `src`. Returns 0 on success, 1 on a bad size/type.
SC_EXPORT int sci_dequant(unsigned int type, const unsigned char* src, long long src_len, float* dst,
                          long long n_elems) {
    g_err.clear();
    auto ty = ty_from_u32(type);
    if (!ty) {
        g_err = "unsupported quant type id " + std::to_string(type);
        return 1;
    }
    if (n_elems < 0 || src_len < 0) {
        g_err = "negative length";
        return 1;
    }
    if (!dequant_row(*ty, src, (size_t)src_len, dst, (size_t)n_elems)) {
        g_err = "dequant size mismatch for " + std::string(ty_name(*ty));
        return 1;
    }
    return 0;
}

// port of quant/mod.rs: to_q8_0
SC_EXPORT int sci_to_q8_0(unsigned int type, const unsigned char* src, long long src_len,
                          unsigned char* dst, long long dst_len) {
    g_err.clear();
    auto ty = ty_from_u32(type);
    if (!ty) {
        g_err = "unsupported quant type id " + std::to_string(type);
        return 1;
    }
    if (!to_q8_0(*ty, src, (size_t)src_len, dst, (size_t)dst_len)) {
        g_err = "to_q8_0: not Q4_0/Q5_0 or bad size";
        return 1;
    }
    return 0;
}

// ------------------------------------------------------------------ config / weights / memory (W2-A)

namespace {

// One GGUF + its Weights, alive for the duration of a call (the tensors point into the mapping).
struct Loaded {
    Gguf g;
    Weights w;
    bool open(const char* path) {
        if (!g.open(path, g_err)) return false;
        return w.init(&g, nullptr, g_err);
    }
};

void j_key(std::string& o, const char* k, bool first) {
    if (!first) o += ',';
    json_escape(o, k);
    o += ':';
}

void j_uint(std::string& o, const char* k, unsigned long long v, bool first) {
    j_key(o, k, first);
    o += std::to_string(v);
}

void j_int(std::string& o, const char* k, long long v, bool first) {
    j_key(o, k, first);
    o += std::to_string(v);
}

void j_float(std::string& o, const char* k, double v, bool first) {
    j_key(o, k, first);
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    o += buf;
}

void j_bool(std::string& o, const char* k, bool v, bool first) {
    j_key(o, k, first);
    o += v ? "true" : "false";
}

void j_bools(std::string& o, const char* k, const std::vector<bool>& v, bool first) {
    j_key(o, k, first);
    o += '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) o += ',';
        o += v[i] ? "true" : "false";
    }
    o += ']';
}

void j_str(std::string& o, const char* k, const std::string& v, bool first) {
    j_key(o, k, first);
    json_escape(o, v);
}

}  // namespace

// port of config.rs: ModelConfig::from_gguf, as JSON. Returns nullptr on failure (see sci_last_error).
SC_EXPORT const char* sci_model_config_json(const char* path) {
    g_err.clear();
    g_json.clear();
    Gguf g;
    if (!g.open(path, g_err)) return nullptr;
    ModelConfig c;
    if (!c.from_gguf(g, g_err)) return nullptr;

    std::string& o = g_json;
    o += '{';
    j_str(o, "arch", c.arch, true);
    j_uint(o, "n_layer", c.n_layer, false);
    j_uint(o, "n_nextn", c.n_nextn, false);
    j_uint(o, "n_embd", c.n_embd, false);
    j_uint(o, "n_head", c.n_head, false);
    j_uint(o, "n_head_kv", c.n_head_kv, false);
    j_uint(o, "head_dim", c.head_dim, false);
    j_uint(o, "n_rot", c.n_rot, false);
    if (c.mrope_sections) {
        j_key(o, "mrope_sections", false);
        o += '[';
        for (size_t i = 0; i < 4; i++) {
            if (i) o += ',';
            o += std::to_string((*c.mrope_sections)[i]);
        }
        o += ']';
    } else {
        j_key(o, "mrope_sections", false);
        o += "null";
    }
    j_float(o, "rope_base", c.rope_base, false);
    j_float(o, "eps", c.eps, false);
    j_uint(o, "n_vocab", c.n_vocab, false);
    j_uint(o, "n_ff", c.n_ff, false);
    j_uint(o, "n_expert", c.n_expert, false);
    j_uint(o, "n_expert_used", c.n_expert_used, false);
    j_uint(o, "n_ff_exp", c.n_ff_exp, false);
    j_uint(o, "n_ff_shexp", c.n_ff_shexp, false);
    j_uint(o, "n_dense", c.n_dense, false);
    j_float(o, "expert_weights_scale", c.expert_weights_scale, false);
    j_bool(o, "router_sigmoid", c.router_sigmoid, false);
    j_uint(o, "n_swa", c.n_swa, false);
    j_bools(o, "swa", c.swa, false);
    j_bools(o, "rope_layer", c.rope_layer, false);
    j_bool(o, "attn_sinks", c.attn_sinks, false);
    j_uint(o, "ssm_d_inner", c.ssm_d_inner, false);
    j_uint(o, "ssm_d_state", c.ssm_d_state, false);
    j_uint(o, "ssm_n_group", c.ssm_n_group, false);
    j_uint(o, "ssm_dt_rank", c.ssm_dt_rank, false);
    j_uint(o, "ssm_conv", c.ssm_conv, false);
    j_uint(o, "hc_n", c.hc_n, false);
    j_uint(o, "hc_rank", c.hc_rank, false);
    j_uint(o, "n_embd_ple", c.n_embd_ple, false);
    j_uint(o, "idx_n_head", c.idx_n_head, false);
    j_uint(o, "idx_head_dim", c.idx_head_dim, false);
    j_uint(o, "idx_top_k", c.idx_top_k, false);
    j_key(o, "layers", false);
    o += '[';
    for (size_t i = 0; i < c.layers.size(); i++) {
        if (i) o += ',';
        json_escape(o, c.layers[i] == LayerKind::Attn ? "attn" : "delta");
    }
    o += ']';
    j_bool(o, "q_gate", c.q_gate, false);
    j_bool(o, "k_interleave", c.k_interleave, false);
    j_key(o, "aliases", false);
    o += '[';
    for (size_t i = 0; i < c.aliases.size(); i++) {
        if (i) o += ',';
        o += '[';
        json_escape(o, c.aliases[i].first);
        o += ',';
        json_escape(o, c.aliases[i].second);
        o += ']';
    }
    o += ']';
    j_uint(o, "ssm_dv", c.ssm_dv(), false);
    j_bool(o, "is_moe", c.is_moe(), false);
    // StrataFreet addition (INKLING2_NOTES.md): the loop geometry and the inkling2 scalars. Emitted only
    // for looped models (config.h: 0/1 n_loop_passes = no loop, every field below unused), so the config
    // JSON of the plain architectures keeps its pre-inkling2 key set (test_weights.py compares the keys).
    if (c.is_looped()) {
        j_uint(o, "n_loop_passes", c.n_loop_passes, false);
        j_uint(o, "loop_start", c.loop_start, false);
        j_uint(o, "loop_end", c.loop_end, false);
        j_uint(o, "loop_gate_rank", c.loop_gate_rank, false);
        j_uint(o, "n_layer_phys", c.n_layer_phys, false);
        j_float(o, "situ_beta", c.situ_beta, false);
        j_float(o, "situ_linear_beta", c.situ_linear_beta, false);
        j_uint(o, "n_rel", c.n_rel, false);
        j_uint(o, "n_rel_extent", c.n_rel_extent, false);
        j_float(o, "f_rel_clip", c.f_rel_clip, false);
        j_float(o, "f_qk_clip", c.f_qk_clip, false);
        j_uint(o, "n_log_scale_floor", c.n_log_scale_floor, false);
        j_float(o, "f_log_scale_alpha", c.f_log_scale_alpha, false);
        j_uint(o, "n_shortconv", c.n_shortconv, false);
        j_float(o, "f_final_softcap", c.f_final_softcap, false);
        j_uint(o, "n_attn_res_block", c.n_attn_res_block, false);
        j_uint(o, "n_vocab_padded", c.n_vocab_padded, false);
        j_uint(o, "n_vocab_unpadded", c.n_vocab_unpadded, false);
    }
    o += '}';
    return o.c_str();
}

// port of weights.rs: Weights::init + validate, as JSON: the synthesized tensors and the MTP verdict.
SC_EXPORT const char* sci_weights_check_json(const char* path) {
    g_err.clear();
    g_json.clear();
    Loaded L;
    if (!L.open(path)) return nullptr;
    bool nextn = true;
    if (!L.w.validate(L.w.config().n_layer, nextn, g_err)) return nullptr;

    std::string& o = g_json;
    o += '{';
    j_bool(o, "nextn", nextn, true);
    j_uint(o, "n_built", L.w.built.size(), false);
    j_key(o, "built", false);
    o += '[';
    bool first = true;
    for (const auto& kv : L.w.built) {
        if (!first) o += ',';
        first = false;
        json_escape(o, kv.first);
    }
    o += ']';
    // Every tensor as Weights::get resolves it: [type, cols, rows, len].
    j_key(o, "tensors", false);
    o += '{';
    first = true;
    for (const auto& kv : L.g.tensors()) {
        if (!first) o += ',';
        first = false;
        json_escape(o, kv.first);
        o += ':';
        Tensor t;
        if (!L.w.get(kv.first, t, g_err)) return nullptr;
        o += '[';
        o += std::to_string((unsigned long long)t.ty);
        o += ',';
        o += std::to_string(t.cols);
        o += ',';
        o += std::to_string(t.rows);
        o += ',';
        o += std::to_string(t.len);
        o += ']';
    }
    o += "}}";
    return o.c_str();
}

// weights.rs: Weights::get, as JSON: type, shape and byte length of one tensor by name.
SC_EXPORT const char* sci_weights_tensor_json(const char* path, const char* name) {
    g_err.clear();
    g_json.clear();
    Loaded L;
    if (!L.open(path)) return nullptr;
    Tensor t;
    if (!L.w.get(name, t, g_err)) return nullptr;
    std::string& o = g_json;
    o += '{';
    j_str(o, "name", name, true);
    j_uint(o, "type", (unsigned long long)t.ty, false);
    j_str(o, "type_name", ty_name(t.ty), false);
    j_uint(o, "cols", t.cols, false);
    j_uint(o, "rows", t.rows, false);
    j_uint(o, "row_bytes", t.row_bytes(), false);
    j_uint(o, "len", t.len, false);
    o += '}';
    return o.c_str();
}

// weights.rs: Tensor::row - `out` receives `n` floats (must equal the tensor's cols).
SC_EXPORT int sci_weights_row(const char* path, const char* name, long long row, float* out, long long n) {
    g_err.clear();
    Loaded L;
    if (!L.open(path)) return 1;
    Tensor t;
    if (!L.w.get(name, t, g_err)) return 1;
    if (row < 0 || n < 0 || (size_t)n != t.cols || (size_t)row >= t.rows) {
        g_err = "sci_weights_row: row " + std::to_string(row) + " / n " + std::to_string(n) +
                " out of range for " + name;
        return 1;
    }
    if (!t.row((size_t)row, out)) {
        g_err = std::string("sci_weights_row: cannot read row of ") + name;
        return 1;
    }
    return 0;
}

// weights.rs: Tensor::expert + Tensor::row - one row of expert `e` of a 3-D expert tensor.
SC_EXPORT int sci_weights_expert_row(const char* path, const char* name, long long expert,
                                     long long n_expert, long long row, float* out, long long n) {
    g_err.clear();
    Loaded L;
    if (!L.open(path)) return 1;
    Tensor t;
    if (!L.w.get(name, t, g_err)) return 1;
    if (expert < 0 || n_expert <= 0 || row < 0 || n < 0) {
        g_err = "sci_weights_expert_row: bad arguments";
        return 1;
    }
    Tensor e = t.expert((size_t)expert, (size_t)n_expert);
    if (!e.data || (size_t)n != e.cols || (size_t)row >= e.rows) {
        g_err = "sci_weights_expert_row: out of range for " + std::string(name);
        return 1;
    }
    if (!e.row((size_t)row, out)) {
        g_err = std::string("sci_weights_expert_row: cannot read row of ") + name;
        return 1;
    }
    return 0;
}

// weights.rs: Tensor::matvec - y = A x (y holds `rows` floats, x holds `cols`).
SC_EXPORT int sci_weights_matvec(const char* path, const char* name, const float* x, long long x_len,
                                 float* y, long long y_len) {
    g_err.clear();
    Loaded L;
    if (!L.open(path)) return 1;
    Tensor t;
    if (!L.w.get(name, t, g_err)) return 1;
    if (x_len < 0 || y_len < 0 || (size_t)x_len != t.cols || (size_t)y_len != t.rows) {
        g_err = "sci_weights_matvec: x/y size mismatch for " + std::string(name);
        return 1;
    }
    t.matvec(x, y);
    return 0;
}

// weights.rs: Tensor::matmul - out[t*rows + r] = row_r . x[t]; x is n_tokens * cols, out n_tokens * rows.
SC_EXPORT int sci_weights_matmul(const char* path, const char* name, const float* x, long long n_tokens,
                                 float* out, long long out_len) {
    g_err.clear();
    Loaded L;
    if (!L.open(path)) return 1;
    Tensor t;
    if (!L.w.get(name, t, g_err)) return 1;
    if (n_tokens <= 0 || out_len < 0 || (size_t)out_len != (size_t)n_tokens * t.rows) {
        g_err = "sci_weights_matmul: out size mismatch for " + std::string(name);
        return 1;
    }
    t.matmul(x, (size_t)n_tokens, out);
    return 0;
}

// port of memory.rs: total_ram (-1 when undetectable).
SC_EXPORT long long sci_total_ram(void) {
    auto t = total_ram();
    return t ? (long long)*t : -1;
}

// port of memory.rs: ram_cap (-1 when unlimited).
SC_EXPORT long long sci_ram_cap(unsigned long long margin, unsigned long long max_ram) {
    auto c = ram_cap((uint64_t)margin, (uint64_t)max_ram);
    return c ? (long long)*c : -1;
}

// weights.rs: Prefetch::start / send / drop and the page-cache advice shims: sends `n` batches and
// joins the worker, so a hang or a lost wake-up shows up as a test timeout instead of in the model.
SC_EXPORT int sci_prefetch_roundtrip(long long n) {
    g_err.clear();
    if (n <= 0) {
        g_err = "sci_prefetch_roundtrip: bad count";
        return 1;
    }
    std::vector<uint8_t> buf(4096, 0);
    Prefetch p;
    p.start();
    for (long long i = 0; i < n; i++) {
        std::vector<std::pair<size_t, size_t>> rs;
        rs.emplace_back((size_t)(uintptr_t)buf.data(), buf.size());
        p.send(std::move(rs));
    }
    will_need((size_t)(uintptr_t)buf.data(), buf.size());
    cold(buf.data(), buf.size());
    return 0;  // ~Prefetch joins the worker
}

// ---- W2-C part 1: the CUDA backend's gemv / moe_gemv wrappers, driven from tests/test_backend_gemv.py ----
// Each call uploads its inputs, runs one launch through CudaBackend, syncs and copies the result back. The
// backend is created on first use for device 0 and lives for the process (the driver may already be torn
// down when the process exits, so it is not freed at unload).

namespace {

std::mutex g_be_mu;
CudaBackend* g_be = nullptr;

// Device 0's backend, or null with g_err set.
CudaBackend* dev_backend() {
    std::lock_guard<std::mutex> lk(g_be_mu);
    if (!g_be) {
        std::string err;
        std::unique_ptr<CudaBackend> be = CudaBackend::create(0, err);
        if (!be) {
            g_err = err;
            return nullptr;
        }
        g_be = be.release();
    }
    return g_be;
}

}  // namespace

// The device name the backend opened (empty with sci_last_error set when it cannot open).
SC_EXPORT const char* scd_device(void) {
    CudaBackend* be = dev_backend();
    if (!be) return "";
    g_json = be->device();
    return g_json.c_str();
}

// y = W x for one matrix of `rows` x `cols` of ggml type `qtype`; `w_host` holds the GGUF bytes of the matrix
// (w_bytes must equal row_bytes(qtype, cols) * rows); `x` holds `cols` floats; `y` receives `rows` floats.
SC_EXPORT int scd_gemv(int qtype, const uint8_t* w_host, long long w_bytes, const float* x, int rows, int cols, float* y) {
    g_err.clear();
    if (!w_host || !x || !y || rows <= 0 || cols <= 0 || w_bytes <= 0) {
        g_err = "scd_gemv: bad arguments";
        return 1;
    }
    auto ty = ty_from_u32((uint32_t)qtype);
    if (!ty) {
        g_err = "scd_gemv: unknown ggml type " + std::to_string(qtype);
        return 1;
    }
    if ((unsigned long long)w_bytes != (unsigned long long)row_bytes(*ty, (uint64_t)cols) * (uint64_t)rows) {
        g_err = "scd_gemv: " + std::to_string(w_bytes) + " bytes is not " + std::to_string(rows) + " rows of " +
                std::to_string(cols) + " " + ty_name(*ty);
        return 1;
    }
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    Buf wb, xb, yb;
    if (!be->alloc((size_t)w_bytes, wb, err) || !be->htod(w_host, (size_t)w_bytes, wb, 0, err) ||
        !be->alloc(4 * (size_t)cols, xb, err) || !be->htod(x, 4 * (size_t)cols, xb, 0, err) ||
        !be->alloc(4 * (size_t)rows, yb, err)) {
        g_err = err;
        return 1;
    }
    GpuTensor w(*ty, (size_t)rows, (size_t)cols, wb);
    if (!be->gemv(w, xb.addr(), yb.addr(), err) || !be->sync(err) || !be->dtoh(yb, 0, y, 4 * (size_t)rows, err)) {
        g_err = err;
        return 1;
    }
    return 0;
}

// y[t] = W x[t] for t < nt (1 to 4): x[t] is at x + x_stride * t floats, y[t] at y + y_stride * t floats.
SC_EXPORT int scd_gemvn(int qtype, const uint8_t* w_host, long long w_bytes, const float* x, int x_stride, int rows,
                        int cols, int nt, float* y, int y_stride) {
    g_err.clear();
    if (!w_host || !x || !y || rows <= 0 || cols <= 0 || w_bytes <= 0 || nt < 1 || nt > 4 || x_stride < cols ||
        y_stride < rows) {
        g_err = "scd_gemvn: bad arguments";
        return 1;
    }
    auto ty = ty_from_u32((uint32_t)qtype);
    if (!ty) {
        g_err = "scd_gemvn: unknown ggml type " + std::to_string(qtype);
        return 1;
    }
    if ((unsigned long long)w_bytes != (unsigned long long)row_bytes(*ty, (uint64_t)cols) * (uint64_t)rows) {
        g_err = "scd_gemvn: w_bytes does not match rows x row_bytes";
        return 1;
    }
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)(nt - 1) * (size_t)x_stride + (size_t)cols;
    const size_t yf = (size_t)(nt - 1) * (size_t)y_stride + (size_t)rows;
    Buf wb, xb, yb;
    if (!be->alloc((size_t)w_bytes, wb, err) || !be->htod(w_host, (size_t)w_bytes, wb, 0, err) ||
        !be->alloc(4 * xf, xb, err) || !be->htod(x, 4 * xf, xb, 0, err) || !be->alloc(4 * yf, yb, err)) {
        g_err = err;
        return 1;
    }
    GpuTensor w(*ty, (size_t)rows, (size_t)cols, wb);
    if (!be->gemvn(w, xb.addr(), (size_t)x_stride, yb.addr(), (size_t)y_stride, (size_t)nt, err) || !be->sync(err) ||
        !be->dtoh(yb, 0, y, 4 * yf, err)) {
        g_err = err;
        return 1;
    }
    return 0;
}

// moe_gemv: for j < k with ids[j] >= 0, y[j] = W_{ids[j]} x[j / xg], where expert e's matrix is `w_bytes` bytes
// starting at e * stride. x has (k - 1) / xg + 1 rows, x_stride floats apart; y receives k * rows floats, and
// the rows of experts with ids[j] < 0 are zero.
SC_EXPORT int scd_moe_gemv(int qtype, const uint8_t* w_host, long long w_bytes, long long stride, const int32_t* ids,
                           int k, const float* x, int x_stride, int xg, int rows, int cols, float* y) {
    g_err.clear();
    if (!w_host || !ids || !x || !y || rows <= 0 || cols <= 0 || k <= 0 || xg <= 0 || stride <= 0 || w_bytes <= 0 ||
        x_stride < cols) {
        g_err = "scd_moe_gemv: bad arguments";
        return 1;
    }
    auto ty = ty_from_u32((uint32_t)qtype);
    if (!ty) {
        g_err = "scd_moe_gemv: unknown ggml type " + std::to_string(qtype);
        return 1;
    }
    if ((unsigned long long)stride != (unsigned long long)row_bytes(*ty, (uint64_t)cols) * (uint64_t)rows) {
        g_err = "scd_moe_gemv: stride is not rows x row_bytes";
        return 1;
    }
    if (w_bytes < stride) {
        g_err = "scd_moe_gemv: w_bytes must hold the experts' matrices";
        return 1;
    }
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xrows = (size_t)(k - 1) / (size_t)xg + 1;
    const size_t xf = (xrows - 1) * (size_t)x_stride + (size_t)cols;
    Buf wb, idb, xb, yb;
    if (!be->alloc((size_t)w_bytes, wb, err) || !be->htod(w_host, (size_t)w_bytes, wb, 0, err) ||
        !be->alloc(4 * (size_t)k, idb, err) || !be->htod(ids, 4 * (size_t)k, idb, 0, err) ||
        !be->alloc(4 * xf, xb, err) || !be->htod(x, 4 * xf, xb, 0, err) ||
        !be->alloc(4 * (size_t)k * (size_t)rows, yb, err)) {
        g_err = err;
        return 1;
    }
    if (!be->moe_gemv(*ty, (size_t)rows, (size_t)cols, wb.addr(), (size_t)stride, idb.addr(), (size_t)k, xb.addr(),
                      (size_t)x_stride, (size_t)xg, yb.addr(), std::nullopt, 0, err) ||
        !be->sync(err) || !be->dtoh(yb, 0, y, 4 * (size_t)k * (size_t)rows, err)) {
        g_err = err;
        return 1;
    }
    return 0;
}

// ---- W2-C part 2a: the CUDA backend's gemm, norm, rope, attention and MoE routing launches, driven from
// tests/test_backend_ops.py. Each call uploads its host arrays, runs one op through CudaBackend, syncs and copies the
// outputs back. Strides are in elements (floats or u16 words), as CudaBackend takes them. Every host array is passed
// at its full size (rows * stride), so the caller owns the bounds.

namespace {

// Device memory of `bytes` bytes, filled from `host` when it is not null, else zeroed.
bool dev_up(CudaBackend& be, const void* host, size_t bytes, Buf& out, std::string& err) {
    if (!be.alloc(bytes, out, err)) return false;
    if (host && bytes) return be.htod(host, bytes, out, 0, err);
    return true;
}

// Copies `bytes` of `b` to `host` (a null host skips the copy).
bool dev_down(CudaBackend& be, const Buf& b, void* host, size_t bytes, std::string& err) {
    if (!host || !bytes) return true;
    return be.dtoh(b, 0, host, bytes, err);
}

// Sets g_err and returns 1 (the C ABI's failure code).
int dev_fail(const std::string& msg) {
    g_err = msg;
    return 1;
}

// The ggml type of `qtype`; false (with g_err set) when unknown.
bool dev_type(const char* fn, int qtype, Ty& ty) {
    auto t = ty_from_u32((uint32_t)qtype);
    if (!t) {
        g_err = std::string(fn) + ": unknown ggml type " + std::to_string(qtype);
        return false;
    }
    ty = *t;
    return true;
}

// Device copy of a GGUF weight matrix: w_bytes must be rows x row_bytes(ty, cols).
bool dev_weight(const char* fn, CudaBackend& be, Ty ty, const uint8_t* w_host, long long w_bytes, int rows, int cols,
                Buf& wb, std::string& err) {
    if ((unsigned long long)w_bytes != (unsigned long long)row_bytes(ty, (uint64_t)cols) * (uint64_t)rows) {
        err = std::string(fn) + ": w_bytes is not " + std::to_string(rows) + " rows of " + std::to_string(cols) + " " +
              ty_name(ty);
        return false;
    }
    return dev_up(be, w_host, (size_t)w_bytes, wb, err);
}

// A small int32 device array of `vals` (the step / position words the kernels read).
bool dev_ints(CudaBackend& be, std::initializer_list<int32_t> vals, Buf& out, std::string& err) {
    std::vector<int32_t> v(vals);
    return dev_up(be, v.data(), 4 * v.size(), out, err);
}

}  // namespace

// y[p] (op) W x[p] for the n positions of x (rows of x_stride, y rows of y_stride). act: 0 store, 1 y = silu(y) *
// result, 2 y += result; | 4 reuses the int8 copy of x (SAME_X). y is read for act 1 and 2 and written back.
SC_EXPORT int scd_gemm_dense(int qtype, const uint8_t* w_host, long long w_bytes, int rows, int cols, int n,
                             const float* x, int x_stride, float* y, int y_stride, int act) {
    g_err.clear();
    Ty ty = Ty::F32;
    if (!w_host || !x || !y || rows <= 0 || cols <= 0 || n <= 0 || x_stride < cols || y_stride < rows || w_bytes <= 0)
        return dev_fail("scd_gemm_dense: bad arguments");
    if (!dev_type("scd_gemm_dense", qtype, ty)) return 1;
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)n * (size_t)x_stride, yf = (size_t)n * (size_t)y_stride;
    Buf wb, xb, yb;
    if (!dev_weight("scd_gemm_dense", *be, ty, w_host, w_bytes, rows, cols, wb, err) ||
        !dev_up(*be, x, 4 * xf, xb, err) || !dev_up(*be, y, 4 * yf, yb, err))
        return dev_fail(err);
    GpuTensor w(ty, (size_t)rows, (size_t)cols, wb);
    if (!be->gemm_dense(w, (size_t)n, xb.addr(), (size_t)x_stride, yb.addr(), (size_t)y_stride, act, err) ||
        !dev_down(*be, yb, y, 4 * yf, err))
        return dev_fail(err);
    return 0;
}

// The tile-list gemm: tile i covers positions tile_m0[i] ... + tile_cnt[i] - 1 of the same matrix. xrow (n_y ints, or
// null) maps a position to its x row. x holds n_x rows, y n_y rows.
SC_EXPORT int scd_gemm_tiles(int qtype, const uint8_t* w_host, long long w_bytes, int rows, int cols, int n_tiles,
                             const int32_t* tile_m0, const int32_t* tile_cnt, const float* x, int x_stride, int n_x,
                             const int32_t* xrow, float* y, int y_stride, int n_y, int act) {
    g_err.clear();
    Ty ty = Ty::F32;
    if (!w_host || !tile_m0 || !tile_cnt || !x || !y || rows <= 0 || cols <= 0 || n_tiles <= 0 || n_x <= 0 || n_y <= 0 ||
        x_stride < cols || y_stride < rows || w_bytes <= 0)
        return dev_fail("scd_gemm_tiles: bad arguments");
    if (!dev_type("scd_gemm_tiles", qtype, ty)) return 1;
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)n_x * (size_t)x_stride, yf = (size_t)n_y * (size_t)y_stride;
    Buf wb, xb, yb, tb, rb;
    if (!dev_weight("scd_gemm_tiles", *be, ty, w_host, w_bytes, rows, cols, wb, err) ||
        !dev_up(*be, x, 4 * xf, xb, err) || !dev_up(*be, y, 4 * yf, yb, err))
        return dev_fail(err);
    // port of cuda/mod.rs: GemmTile { w, m0, cnt } (16 bytes: the pointer, then the two ints)
    std::vector<uint8_t> tiles(16 * (size_t)n_tiles);
    const uint64_t wa = wb.addr();
    for (int i = 0; i < n_tiles; i++) {
        const int32_t m0 = tile_m0[i], cnt = tile_cnt[i];
        std::memcpy(&tiles[16 * (size_t)i], &wa, 8);
        std::memcpy(&tiles[16 * (size_t)i + 8], &m0, 4);
        std::memcpy(&tiles[16 * (size_t)i + 12], &cnt, 4);
    }
    if (!dev_up(*be, tiles.data(), tiles.size(), tb, err)) return dev_fail(err);
    if (xrow && !dev_up(*be, xrow, 4 * (size_t)n_y, rb, err)) return dev_fail(err);
    if (!be->gemm(ty, (size_t)rows, (size_t)cols, tb.addr(), (size_t)n_tiles, xb.addr(), (size_t)x_stride, (size_t)n_x,
                  xrow ? rb.addr() : 0, yb.addr(), (size_t)y_stride, act, err) ||
        !dev_down(*be, yb, y, 4 * yf, err))
        return dev_fail(err);
    return 0;
}

// rms_norm_q over rows of n: out = rmsnorm(x) * w. res (rows of x_stride, or null) is added into x first; x_out
// (or null) receives x after that; xq_out (or null) receives the quant_xq layout of out (rows * (n + n / 8) bytes).
SC_EXPORT int scd_rms_norm(const float* x, int x_stride, const float* w, float* out, int out_stride, int n, int rows,
                           float eps, const float* res, float* x_out, signed char* xq_out) {
    g_err.clear();
    if (!x || !w || !out || n <= 0 || rows <= 0 || x_stride < n || out_stride < n)
        return dev_fail("scd_rms_norm: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)rows * (size_t)x_stride, of = (size_t)rows * (size_t)out_stride;
    const size_t qb = (size_t)rows * ((size_t)n + (size_t)n / 8);
    Buf xb, wb, ob, rb, qbuf;
    if (!dev_up(*be, x, 4 * xf, xb, err) || !dev_up(*be, w, 4 * (size_t)n, wb, err) ||
        !dev_up(*be, nullptr, 4 * of, ob, err) || (res && !dev_up(*be, res, 4 * xf, rb, err)) ||
        (xq_out && !dev_up(*be, nullptr, qb, qbuf, err)))
        return dev_fail(err);
    if (!be->rms_norm_q(xb.addr(), (size_t)x_stride, wb.addr(), ob.addr(), (size_t)out_stride, (size_t)n, (size_t)rows,
                        eps, xq_out ? qbuf.addr() : 0, res ? rb.addr() : 0, err) ||
        !dev_down(*be, ob, out, 4 * of, err) || !dev_down(*be, xb, x_out, 4 * xf, err) ||
        !dev_down(*be, qbuf, xq_out, qb, err))
        return dev_fail(err);
    return 0;
}

// out[r] = index of the largest of row r (rows of n floats), the first one on ties.
SC_EXPORT int scd_argmax_rows(const float* x, int n, int rows, int32_t* out) {
    g_err.clear();
    if (!x || !out || n <= 0 || rows <= 0) return dev_fail("scd_argmax_rows: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)rows * (size_t)n;
    Buf xb, ob;
    if (!dev_up(*be, x, 4 * xf, xb, err) || !dev_up(*be, nullptr, 4 * (size_t)rows, ob, err) ||
        !be->argmax_rows(xb.addr(), (size_t)n, (size_t)rows, ob.addr(), err) ||
        !dev_down(*be, ob, out, 4 * (size_t)rows, err))
        return dev_fail(err);
    return 0;
}

// l2_norm in place over x_len floats: vector r % per of token r / per (tokens `stride` apart), rows = tokens * per.
SC_EXPORT int scd_l2_norm(float* x, int x_len, int n, int per, int stride, int rows, float eps) {
    g_err.clear();
    if (!x || x_len <= 0 || n <= 0 || per <= 0 || stride < n * per || rows <= 0)
        return dev_fail("scd_l2_norm: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    Buf xb;
    if (!dev_up(*be, x, 4 * (size_t)x_len, xb, err) ||
        !be->l2_norm(xb.addr(), (size_t)n, (size_t)per, (size_t)stride, (size_t)rows, eps, err) ||
        !dev_down(*be, xb, x, 4 * (size_t)x_len, err))
        return dev_fail(err);
    return 0;
}

// rope_neox in place over x_len floats: heads rows of stride floats (per per token); token t is at position step0 + t.
SC_EXPORT int scd_rope_pairs(float* x, int x_len, int stride, int heads, int per, int n_rot, int step0, float base, int lo,
                             int hi) {
    g_err.clear();
    if (!x || x_len <= 0 || stride <= 0 || heads <= 0 || per <= 0 || heads % per != 0 || n_rot <= 0 || n_rot % 2 != 0 ||
        n_rot / 2 > 1024)
        return dev_fail("scd_rope_pairs: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    Buf xb, sb;
    if (!dev_up(*be, x, 4 * (size_t)x_len, xb, err) || !dev_ints(*be, {step0, 0}, sb, err) ||
        !be->rope_pairs(xb.addr(), (size_t)stride, (size_t)heads, (size_t)per, (size_t)n_rot, sb.addr(), base, (size_t)lo,
                        (size_t)hi, err) ||
        !dev_down(*be, xb, x, 4 * (size_t)x_len, err))
        return dev_fail(err);
    return 0;
}

// rms_norm_rope_kv: out = rope(rmsnorm(x)) per head (rows heads of n, per per token, token t at position step0 + t).
// With kc and vc (u16 f16 caches of cache_len words, in/out) the head and v are also stored; kc null: no store.
SC_EXPORT int scd_rms_norm_rope_kv(const float* x, int x_stride, const float* w, float* out, int n, int rows, float eps,
                                   int per, int n_rot, int step0, float base, const float* v, uint16_t* kc, uint16_t* vc,
                                   int cache_len) {
    g_err.clear();
    if (!x || !w || !out || n <= 0 || rows <= 0 || per <= 0 || rows % per != 0 || x_stride < n || n_rot <= 0 ||
        (kc && (!vc || !v || cache_len <= 0)))
        return dev_fail("scd_rms_norm_rope_kv: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)rows * (size_t)x_stride, of = (size_t)rows * (size_t)n;
    const size_t cb = 2 * (size_t)cache_len;
    Buf xb, wb, ob, vb, kb, vcb, sb;
    if (!dev_up(*be, x, 4 * xf, xb, err) || !dev_up(*be, w, 4 * (size_t)n, wb, err) ||
        !dev_up(*be, nullptr, 4 * of, ob, err) || !dev_ints(*be, {step0, 0}, sb, err) ||
        (v && !dev_up(*be, v, 4 * of, vb, err)) ||
        (kc && (!dev_up(*be, kc, cb, kb, err) || !dev_up(*be, vc, cb, vcb, err))))
        return dev_fail(err);
    if (!be->rms_norm_rope_kv(xb.addr(), (size_t)x_stride, wb.addr(), ob.addr(), (size_t)n, (size_t)rows, eps,
                              (size_t)per, (size_t)n_rot, sb.addr(), base, v ? vb.addr() : 0, kc ? kb.addr() : 0,
                              kc ? vcb.addr() : 0, err) ||
        !dev_down(*be, ob, out, 4 * of, err) || !dev_down(*be, kb, kc, cb, err) || !dev_down(*be, vcb, vc, cb, err))
        return dev_fail(err);
    return 0;
}

// x += y over n floats.
SC_EXPORT int scd_add(float* x, const float* y, int n) {
    g_err.clear();
    if (!x || !y || n <= 0) return dev_fail("scd_add: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t nb = 4 * (size_t)n;
    Buf xb, yb;
    if (!dev_up(*be, x, nb, xb, err) || !dev_up(*be, y, nb, yb, err) ||
        !be->add(xb.addr(), yb.addr(), (size_t)n, err) || !dev_down(*be, xb, x, nb, err))
        return dev_fail(err);
    return 0;
}

// g = silu(g) * u over rows of n. With xq_out (rows * (n + n / 8) bytes) the product is quantized there as quant_xq
// does, and g is left as is.
SC_EXPORT int scd_swiglu(float* g, const float* u, int n, int rows, signed char* xq_out) {
    g_err.clear();
    if (!g || !u || n <= 0 || rows <= 0) return dev_fail("scd_swiglu: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t gf = (size_t)rows * (size_t)n, qb = (size_t)rows * ((size_t)n + (size_t)n / 8);
    Buf gb, ub, qbuf;
    if (!dev_up(*be, g, 4 * gf, gb, err) || !dev_up(*be, u, 4 * gf, ub, err) ||
        (xq_out && !dev_up(*be, nullptr, qb, qbuf, err)))
        return dev_fail(err);
    if (xq_out) {
        uint64_t xo = 0;
        if (!be->swiglu_q(gb.addr(), ub.addr(), (size_t)n, (size_t)rows, qbuf.addr(), xo, err)) return dev_fail(err);
    } else if (!be->swiglu(gb.addr(), ub.addr(), gf, err)) {
        return dev_fail(err);
    }
    if (!dev_down(*be, gb, g, 4 * gf, err) || !dev_down(*be, qbuf, xq_out, qb, err)) return dev_fail(err);
    return 0;
}

// norm_gate in place over `heads` rows of n: o = rmsnorm(o) * w * silu(z) (sigmoid(z) when sig). xq_out (heads *
// (n + n / 8) bytes, or null) receives the quant_xq layout of o.
SC_EXPORT int scd_norm_gate(float* o, const float* w, const float* z, int n, int heads, float eps, int sig,
                            signed char* xq_out) {
    g_err.clear();
    if (!o || !w || !z || n <= 0 || heads <= 0) return dev_fail("scd_norm_gate: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t of = (size_t)heads * (size_t)n, qb = (size_t)heads * ((size_t)n + (size_t)n / 8);
    Buf ob, wb, zb, qbuf;
    if (!dev_up(*be, o, 4 * of, ob, err) || !dev_up(*be, w, 4 * (size_t)n, wb, err) || !dev_up(*be, z, 4 * of, zb, err) ||
        (xq_out && !dev_up(*be, nullptr, qb, qbuf, err)))
        return dev_fail(err);
    uint64_t xo = 0;
    const uint64_t xq = xq_out ? qbuf.addr() : 0;
    const bool ok = sig ? be->norm_gate_sig(ob.addr(), wb.addr(), zb.addr(), (size_t)n, (size_t)heads, eps, xq, xo, err)
                        : be->norm_gate_q(ob.addr(), wb.addr(), zb.addr(), (size_t)n, (size_t)heads, eps, xq, xo, err);
    if (!ok || !dev_down(*be, ob, o, 4 * of, err) || !dev_down(*be, qbuf, xq_out, qb, err)) return dev_fail(err);
    return 0;
}

// kv_store: t tokens of n values (k, v) stored at positions step0 ... in the f16 caches kc and vc (cache_len words).
SC_EXPORT int scd_kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int cache_len, int n, int t,
                           int step0, int hd) {
    g_err.clear();
    if (!k || !v || !kc || !vc || cache_len <= 0 || n <= 0 || t <= 0 || hd <= 0 || n % hd != 0)
        return dev_fail("scd_kv_store: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t kf = (size_t)n * (size_t)t, cb = 2 * (size_t)cache_len;
    Buf kb, vb, kcb, vcb, sb;
    if (!dev_up(*be, k, 4 * kf, kb, err) || !dev_up(*be, v, 4 * kf, vb, err) || !dev_up(*be, kc, cb, kcb, err) ||
        !dev_up(*be, vc, cb, vcb, err) || !dev_ints(*be, {step0, 0}, sb, err) ||
        !be->kv_store(kb.addr(), vb.addr(), kcb.addr(), vcb.addr(), (size_t)n, (size_t)t, sb.addr(), (size_t)hd, err) ||
        !dev_down(*be, kcb, kc, cb, err) || !dev_down(*be, vcb, vc, cb, err))
        return dev_fail(err);
    return 0;
}

// Decode attention, token t over positions 0 ..= steps[2 t] (steps: 2 * nt int32). q and out hold nt rows of nh * hd
// floats; the caches are f16 with nkv heads of hd values per position. gq_min 0: attn_decode_gqa only; i32 max:
// attn_decode only; else the regime of position gq_min. No gate and no sink. bias_arr (the query token's Inkling2
// relative-bias table {extent_in + 1, nh}, [d * nh + h]) with extent_in >= 0 adds the relative bias to the raw scores;
// null passes 0/0 (no bias).
SC_EXPORT int scd_attn_decode_gq(const float* q, const uint16_t* kc, const uint16_t* vc, int cache_len,
                                 const int32_t* steps, int nh, int nkv, int hd, float scale, int gq_min, int nt, float* out,
                                 const float* bias_arr, int extent_in) {
    g_err.clear();
    if (!q || !kc || !vc || !steps || !out || cache_len <= 0 || nh <= 0 || nkv <= 0 || nh % nkv != 0 || hd <= 0 || nt <= 0 ||
        (bias_arr && extent_in < 0))
        return dev_fail("scd_attn_decode_gq: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t qf = (size_t)nt * (size_t)nh * (size_t)hd;
    const size_t part = (size_t)nt * (size_t)nh * ATT_SPLITS * (size_t)(hd + 2);
    const size_t cb = 2 * (size_t)cache_len;
    const size_t bf = bias_arr ? ((size_t)extent_in + 1) * (size_t)nh : 0;
    Buf qb, kcb, vcb, sb, ob, pb, bb;
    if (!dev_up(*be, q, 4 * qf, qb, err) || !dev_up(*be, kc, cb, kcb, err) || !dev_up(*be, vc, cb, vcb, err) ||
        !dev_up(*be, steps, 8 * (size_t)nt, sb, err) || !dev_up(*be, nullptr, 4 * qf, ob, err) ||
        !dev_up(*be, nullptr, 4 * part, pb, err) || (bias_arr && !dev_up(*be, bias_arr, 4 * bf, bb, err)) ||
        !be->attn_decode_gq(qb.addr(), kcb.addr(), vcb.addr(), 0, ob.addr(), pb.addr(), sb.addr(), (size_t)nh, (size_t)nkv,
                            (size_t)hd, scale, gq_min, (size_t)nt, 0, 0, bias_arr ? bb.addr() : 0,
                            bias_arr ? (size_t)extent_in : 0, err) ||
        !dev_down(*be, ob, out, 4 * qf, err))
        return dev_fail(err);
    return 0;
}

// Prefill attention (flash, causal): query t of t rows sits at position step0 + t and sees keys 0 ..= step0 + t of the
// f16 caches. q and out: t rows of nh * hd floats. No mask, gate or sink.
SC_EXPORT int scd_attn_prefill_m(const float* q, const uint16_t* kc, const uint16_t* vc, int cache_len, int step0, int t,
                                 int nh, int nkv, int hd, float scale, float* out) {
    g_err.clear();
    if (!q || !kc || !vc || !out || cache_len <= 0 || t <= 0 || nh <= 0 || nkv <= 0 || nh % nkv != 0 || hd <= 0)
        return dev_fail("scd_attn_prefill_m: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t qf = (size_t)t * (size_t)nh * (size_t)hd, cb = 2 * (size_t)cache_len;
    Buf qb, kcb, vcb, sb, ob;
    if (!dev_up(*be, q, 4 * qf, qb, err) || !dev_up(*be, kc, cb, kcb, err) || !dev_up(*be, vc, cb, vcb, err) ||
        !dev_ints(*be, {step0, 0}, sb, err) || !dev_up(*be, nullptr, 4 * qf, ob, err) ||
        !be->attn_prefill_m(qb.addr(), kcb.addr(), vcb.addr(), 0, ob.addr(), sb.addr(), (size_t)t, (size_t)nh, (size_t)nkv,
                            (size_t)hd, scale, 0, 0, 0, 0, err) ||
        !dev_down(*be, ob, out, 4 * qf, err))
        return dev_fail(err);
    return 0;
}

// router_topk over t rows of n logits: ids and wts receive t rows of k.
SC_EXPORT int scd_router_topk(const float* p, int n, int k, float extra_scale, int sigmoid, int t, int32_t* ids,
                              float* wts) {
    g_err.clear();
    if (!p || !ids || !wts || n <= 0 || n > 512 || k <= 0 || k > 32 || t <= 0)
        return dev_fail("scd_router_topk: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t pf = (size_t)t * (size_t)n, of = (size_t)t * (size_t)k;
    Buf pb, ib, wb;
    if (!dev_up(*be, p, 4 * pf, pb, err) || !dev_up(*be, nullptr, 4 * of, ib, err) ||
        !dev_up(*be, nullptr, 4 * of, wb, err) ||
        !be->router_topk(pb.addr(), (size_t)n, (size_t)k, extra_scale, ib.addr(), wb.addr(), (size_t)t, sigmoid != 0, 0,
                         0.0f, err) ||
        !dev_down(*be, ib, ids, 4 * of, err) || !dev_down(*be, wb, wts, 4 * of, err))
        return dev_fail(err);
    return 0;
}

// moe_route for a block of t tokens. table: the slot of each of n_table experts (-1: on the CPU). Outputs: ids, wts,
// slots and wh (t rows of k); pid (k ints, may be null) receives the last token's ids. With mb (the mailbox words, in
// and out, mb_words long) the routing and h (nh rows of t) are written at the mailbox words as the kernel does.
// step0 and seq are the step words [position, seq].
SC_EXPORT int scd_moe_route(const float* p, int n, int k, float extra_scale, const int32_t* table, int n_table, int sigmoid,
                            const float* h, int nh, int h_off, int step0, unsigned seq, int t, uint32_t* mb, int mb_words,
                            int32_t* ids, float* wts, int32_t* slots, float* wh, int32_t* pid) {
    g_err.clear();
    if (!p || !table || !ids || !wts || !slots || !wh || n <= 0 || n > 512 || k <= 0 || k > 32 || n_table < n || t <= 0 ||
        (mb && (!h || mb_words <= 0)))
        return dev_fail("scd_moe_route: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t pf = (size_t)t * (size_t)n, of = (size_t)t * (size_t)k;
    Buf pb, tb, ib, wb, sl, whb, hb, pidb, sb, mbb;
    uint32_t* mbh = nullptr;
    if (!dev_up(*be, p, 4 * pf, pb, err) || !dev_up(*be, table, 4 * (size_t)n_table, tb, err) ||
        !dev_up(*be, nullptr, 4 * of, ib, err) || !dev_up(*be, nullptr, 4 * of, wb, err) ||
        !dev_up(*be, nullptr, 4 * of, sl, err) || !dev_up(*be, nullptr, 4 * of, whb, err) ||
        !dev_up(*be, nullptr, 4 * (size_t)k, pidb, err) || !dev_ints(*be, {step0, (int32_t)seq}, sb, err) ||
        (h && !dev_up(*be, h, 4 * (size_t)nh * (size_t)t, hb, err)))
        return dev_fail(err);
    if (mb) {
        if (!be->host_mailbox((size_t)mb_words, mbb, mbh, err)) return dev_fail(err);
        std::memcpy(mbh, mb, 4 * (size_t)mb_words);
    }
    if (!be->moe_route(pb.addr(), (size_t)n, (size_t)k, extra_scale, tb.addr(), ib.addr(), wb.addr(), sl.addr(), whb.addr(),
                       h ? hb.addr() : 0, mb ? mbb.addr() : 0, (size_t)nh, (size_t)h_off, sb.addr(), (size_t)t,
                       pid ? pidb.addr() : 0, sigmoid != 0, 0, 0.0f, err) ||
        !be->sync(err) || !dev_down(*be, ib, ids, 4 * of, err) || !dev_down(*be, wb, wts, 4 * of, err) ||
        !dev_down(*be, sl, slots, 4 * of, err) || !dev_down(*be, whb, wh, 4 * of, err) ||
        !dev_down(*be, pidb, pid, 4 * (size_t)k, err))
        return dev_fail(err);
    if (mb) std::memcpy(mb, mbh, 4 * (size_t)mb_words);
    return 0;
}

// moe_combine over t tokens: x (t rows of n, in/out) += sum_j w ye + sigmoid(sg) sh (sh, sg may be null) + the CPU
// result at mailbox word y_off when a slot of the token missed (mb null: no CPU part). ye: t * k rows of n; slots: t * k.
SC_EXPORT int scd_moe_combine(float* x, const float* ye, const float* w, int k, const float* sh, const float* sg,
                              const int32_t* slots, int n, int t, uint32_t* mb, int mb_words, int y_off, int step0,
                              unsigned seq) {
    g_err.clear();
    if (!x || !ye || !w || k <= 0 || n <= 0 || t <= 0 || (mb && (!slots || mb_words <= 0)))
        return dev_fail("scd_moe_combine: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)t * (size_t)n, yf = (size_t)t * (size_t)k * (size_t)n, of = (size_t)t * (size_t)k;
    Buf xb, yeb, wb, shb, sgb, slb, sb, mbb;
    uint32_t* mbh = nullptr;
    if (!dev_up(*be, x, 4 * xf, xb, err) || !dev_up(*be, ye, 4 * yf, yeb, err) || !dev_up(*be, w, 4 * of, wb, err) ||
        (sh && !dev_up(*be, sh, 4 * xf, shb, err)) || (sg && !dev_up(*be, sg, 4 * (size_t)t, sgb, err)) ||
        !dev_up(*be, slots, 4 * of, slb, err) || !dev_ints(*be, {step0, (int32_t)seq}, sb, err))
        return dev_fail(err);
    if (mb) {
        if (!be->host_mailbox((size_t)mb_words, mbb, mbh, err)) return dev_fail(err);
        std::memcpy(mbh, mb, 4 * (size_t)mb_words);
    }
    if (!be->moe_combine(xb.addr(), yeb.addr(), wb.addr(), (size_t)k, sh ? shb.addr() : 0, sg ? sgb.addr() : 0,
                         mb ? mbb.addr() : 0, (size_t)y_off, sb.addr(), slb.addr(), (size_t)n, (size_t)t, 0, 0, err) ||
        !be->sync(err) || !dev_down(*be, xb, x, 4 * xf, err))
        return dev_fail(err);
    return 0;
}

// moe_combine_seq over t tokens: x[t] += sum_j wts[t, j] ye[inv[t, j]] + sigmoid(sg[t]) sh[t]. ye holds n_ye rows of n.
SC_EXPORT int scd_moe_combine_seq(float* x, const float* ye, int n_ye, const float* wts, const int32_t* inv, int k,
                                  const float* sh, const float* sg, int n, int t) {
    g_err.clear();
    if (!x || !ye || !wts || !inv || k <= 0 || n <= 0 || t <= 0 || n_ye <= 0)
        return dev_fail("scd_moe_combine_seq: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t xf = (size_t)t * (size_t)n, of = (size_t)t * (size_t)k;
    Buf xb, yeb, wb, ib, shb, sgb;
    if (!dev_up(*be, x, 4 * xf, xb, err) || !dev_up(*be, ye, 4 * (size_t)n_ye * (size_t)n, yeb, err) ||
        !dev_up(*be, wts, 4 * of, wb, err) || !dev_up(*be, inv, 4 * of, ib, err) ||
        (sh && !dev_up(*be, sh, 4 * xf, shb, err)) || (sg && !dev_up(*be, sg, 4 * (size_t)t, sgb, err)))
        return dev_fail(err);
    if (!be->moe_combine_seq(xb.addr(), yeb.addr(), wb.addr(), ib.addr(), (size_t)k, sh ? shb.addr() : 0,
                             sg ? sgb.addr() : 0, (size_t)n, (size_t)t, err) ||
        !dev_down(*be, xb, x, 4 * xf, err))
        return dev_fail(err);
    return 0;
}

// ---- W2-C part 2b: the DeltaNet / conv ops, QSA, hyper-connections, PLE, the per-row KV kernels, moe_prefetch and the
// graph / beside machinery, driven from tests/test_backend_seq.py. Each call uploads its host arrays, runs the ops
// through CudaBackend, syncs and copies the outputs back (in/out arrays are updated in place).


// ssm_gates in place over n values (beta, g of n = tokens * nv) with the per-head dt_bias and a of nv values.
SC_EXPORT int scd_ssm_gates(float* beta, float* g, const float* dt_bias, const float* a, int n, int nv) {
    g_err.clear();
    if (!beta || !g || !dt_bias || !a || n <= 0 || nv <= 0) return dev_fail("scd_ssm_gates: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t nb = 4 * (size_t)n, vb = 4 * (size_t)nv;
    Buf bb, gb, db, ab;
    if (!dev_up(*be, beta, nb, bb, err) || !dev_up(*be, g, nb, gb, err) || !dev_up(*be, dt_bias, vb, db, err) ||
        !dev_up(*be, a, vb, ab, err) ||
        !be->ssm_gates(bb.addr(), gb.addr(), db.addr(), ab.addr(), (size_t)n, (size_t)nv, err) ||
        !dev_down(*be, bb, beta, nb, err) || !dev_down(*be, gb, g, nb, err))
        return dev_fail(err);
    return 0;
}

// conv_silu for one token: state ((kw - 1) x ch, oldest first, in/out), y = silu(causal conv), the first nn blocks of
// 128 channels l2-normalized.
SC_EXPORT int scd_conv_silu(const float* qkv, float* state, const float* w, float* y, int ch, int kw, int nn, float eps) {
    g_err.clear();
    if (!qkv || !state || !w || !y || ch <= 0 || kw < 2 || nn < 0 || nn * 128 > ch)
        return dev_fail("scd_conv_silu: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t cb = 4 * (size_t)ch, sb = 4 * (size_t)(kw - 1) * (size_t)ch, wb_ = 4 * (size_t)ch * (size_t)kw;
    Buf qb, sbuf, wb, yb;
    if (!dev_up(*be, qkv, cb, qb, err) || !dev_up(*be, state, sb, sbuf, err) || !dev_up(*be, w, wb_, wb, err) ||
        !dev_up(*be, nullptr, cb, yb, err) ||
        !be->conv_silu(qb.addr(), sbuf.addr(), wb.addr(), yb.addr(), (size_t)ch, (size_t)kw, (size_t)nn, eps, err) ||
        !dev_down(*be, sbuf, state, sb, err) || !dev_down(*be, yb, y, cb, err))
        return dev_fail(err);
    return 0;
}

// conv_seq over t tokens (y: t x ch, state in/out) and the snapshots of the first min(nsnap, t - 1) states (snap: nsnap x
// (kw - 1) x ch, may be null when nsnap is 0).
SC_EXPORT int scd_conv_seq(const float* qkv, float* state, const float* w, float* y, int ch, int kw, int t, float* snap,
                           int nsnap) {
    g_err.clear();
    if (!qkv || !state || !w || !y || ch <= 0 || kw < 2 || t <= 0 || nsnap < 0 || (nsnap > 0 && !snap))
        return dev_fail("scd_conv_seq: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t qb_ = 4 * (size_t)t * (size_t)ch, sb = 4 * (size_t)(kw - 1) * (size_t)ch;
    const size_t snb = 4 * (size_t)nsnap * (size_t)(kw - 1) * (size_t)ch;
    Buf qb, sbuf, wb, yb, snb_;
    if (!dev_up(*be, qkv, qb_, qb, err) || !dev_up(*be, state, sb, sbuf, err) ||
        !dev_up(*be, w, 4 * (size_t)ch * (size_t)kw, wb, err) || !dev_up(*be, nullptr, qb_, yb, err) ||
        !dev_up(*be, nullptr, snb > 0 ? snb : 4, snb_, err) ||
        !be->conv_seq(qb.addr(), sbuf.addr(), wb.addr(), yb.addr(), (size_t)ch, (size_t)kw, (size_t)t,
                      nsnap > 0 ? snb_.addr() : 0, (size_t)nsnap, err) ||
        !dev_down(*be, sbuf, state, sb, err) || !dev_down(*be, yb, y, qb_, err) ||
        (nsnap > 0 && !dev_down(*be, snb_, snap, snb, err)))
        return dev_fail(err);
    return 0;
}

// The Gated DeltaNet recurrence over t tokens (deltanet_seq). y: t rows of ych floats ([q | k | v] after conv and l2
// norm); beta, g: t x nv (gates already applied, or raw with dt_bias and a non-null); S: nv x dv x dk state (in/out);
// o: t x nv x dv out (before the output norm); snap: nsnap states (may be null when nsnap is 0).
SC_EXPORT int scd_deltanet(const float* y, int ych, float* S, const float* beta, const float* g, const float* dt_bias,
                           const float* a, float* o, int nk, int nv, int interleave, int dk, int dv, float scale, int t,
                           float* snap, int nsnap) {
    g_err.clear();
    if (!y || !S || !beta || !g || !o || ych <= 0 || nk <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || t <= 0 || nsnap < 0 ||
        (nsnap > 0 && !snap) || (dt_bias != nullptr) != (a != nullptr) ||
        (size_t)ych < (size_t)(2 * nk * dk + nv * dv))
        return dev_fail("scd_deltanet: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t yb_ = 4 * (size_t)t * (size_t)ych, sb = 4 * (size_t)nv * (size_t)dv * (size_t)dk;
    const size_t gb_ = 4 * (size_t)t * (size_t)nv, ob = 4 * (size_t)t * (size_t)nv * (size_t)dv;
    const size_t snb = 4 * (size_t)nsnap * (size_t)nv * (size_t)dv * (size_t)dk;
    Buf yb, sbuf, bb, gbuf, db, ab, ob_, snb_;
    if (!dev_up(*be, y, yb_, yb, err) || !dev_up(*be, S, sb, sbuf, err) || !dev_up(*be, beta, gb_, bb, err) ||
        !dev_up(*be, g, gb_, gbuf, err) || (dt_bias && !dev_up(*be, dt_bias, 4 * (size_t)nv, db, err)) ||
        (a && !dev_up(*be, a, 4 * (size_t)nv, ab, err)) || !dev_up(*be, nullptr, ob, ob_, err) ||
        !dev_up(*be, nullptr, snb > 0 ? snb : 4, snb_, err) ||
        !be->deltanet(yb.addr(), sbuf.addr(), bb.addr(), gbuf.addr(), dt_bias ? db.addr() : 0, a ? ab.addr() : 0, ob_.addr(),
                      (size_t)nk, (size_t)nv, interleave != 0, (size_t)dk, (size_t)dv, scale, (size_t)t, (size_t)ych,
                      nsnap > 0 ? snb_.addr() : 0, (size_t)nsnap, err) ||
        !dev_down(*be, sbuf, S, sb, err) || !dev_down(*be, ob_, o, ob, err) ||
        (nsnap > 0 && !dev_down(*be, snb_, snap, snb, err)))
        return dev_fail(err);
    return 0;
}

// hc_silu in place: x = silu(x * sc) over n values.
SC_EXPORT int scd_hc_silu(float* x, float sc, int n) {
    g_err.clear();
    if (!x || n <= 0) return dev_fail("scd_hc_silu: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t nb = 4 * (size_t)n;
    Buf xb;
    if (!dev_up(*be, x, nb, xb, err) || !be->hc_silu(xb.addr(), sc, (size_t)n, err) || !dev_down(*be, xb, x, nb, err))
        return dev_fail(err);
    return 0;
}

// hc_mix: out[t][i] = sum over the hc streams c of xn[t][c n + i] * sigmoid(g[t][c n + i]) / hc (xn, g: nt x hc x n).
SC_EXPORT int scd_hc_mix(const float* xn, const float* g, float* out, int n, int hc, int nt) {
    g_err.clear();
    if (!xn || !g || !out || n <= 0 || hc <= 0 || nt <= 0) return dev_fail("scd_hc_mix: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t inb = 4 * (size_t)nt * (size_t)hc * (size_t)n, ob = 4 * (size_t)nt * (size_t)n;
    Buf xb, gb, ob_;
    if (!dev_up(*be, xn, inb, xb, err) || !dev_up(*be, g, inb, gb, err) || !dev_up(*be, nullptr, ob, ob_, err) ||
        !be->hc_mix(xb.addr(), gb.addr(), ob_.addr(), (size_t)n, (size_t)hc, (size_t)nt, err) ||
        !dev_down(*be, ob_, out, ob, err))
        return dev_fail(err);
    return 0;
}

// hc_write: stream c of r[t] += out[t] * 2 sigmoid(inj[t][c] / hc); clear zeroes out afterwards. r (nt x hc x n, in/out),
// out (nt x n, in/out), inj (nt x hc).
SC_EXPORT int scd_hc_write(float* r, float* out, const float* inj, int n, int hc, int nt, int clear) {
    g_err.clear();
    if (!r || !out || !inj || n <= 0 || hc <= 0 || nt <= 0) return dev_fail("scd_hc_write: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t rb = 4 * (size_t)nt * (size_t)hc * (size_t)n, ob = 4 * (size_t)nt * (size_t)n, ib = 4 * (size_t)nt * (size_t)hc;
    Buf rbuf, obuf, ibuf;
    if (!dev_up(*be, r, rb, rbuf, err) || !dev_up(*be, out, ob, obuf, err) || !dev_up(*be, inj, ib, ibuf, err) ||
        !be->hc_write(rbuf.addr(), obuf.addr(), ibuf.addr(), (size_t)n, (size_t)hc, (size_t)nt, clear != 0, err) ||
        !dev_down(*be, rbuf, r, rb, err) || !dev_down(*be, obuf, out, ob, err))
        return dev_fail(err);
    return 0;
}

// ple_gate: per token and stream, s = key . query / sqrt(n); gated[t][c n + i] = value[t][i] sigmoid(sign(s) sqrt(|s|)).
// key, query, gated: nt x hc x n; value: nt x n.
SC_EXPORT int scd_ple_gate(const float* key, const float* query, const float* value, float* gated, int n, int hc, int nt) {
    g_err.clear();
    if (!key || !query || !value || !gated || n <= 0 || hc <= 0 || nt <= 0) return dev_fail("scd_ple_gate: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t db = 4 * (size_t)nt * (size_t)hc * (size_t)n, vb = 4 * (size_t)nt * (size_t)n;
    Buf kb, qb, vbuf, gb;
    if (!dev_up(*be, key, db, kb, err) || !dev_up(*be, query, db, qb, err) || !dev_up(*be, value, vb, vbuf, err) ||
        !dev_up(*be, nullptr, db, gb, err) ||
        !be->ple_gate(kb.addr(), qb.addr(), vbuf.addr(), gb.addr(), (size_t)n, (size_t)hc, (size_t)nt, err) ||
        !dev_down(*be, gb, gated, db, err))
        return dev_fail(err);
    return 0;
}

// ple_conv for nt tokens at positions step0 ...: r[t] += gated[t] + silu(causal 4-tap conv dilated by 3 over the normalized
// rows, which live in ring: 16 rows of d, position p in row p % 16). r (nt x d, in/out), gated and norm (nt x d), kw (d x 4),
// ring (16 x d, in/out).
SC_EXPORT int scd_ple_conv(float* r, const float* gated, const float* norm, const float* kw, float* ring, int step0, int d,
                           int nt) {
    g_err.clear();
    if (!r || !gated || !norm || !kw || !ring || d <= 0 || nt <= 0) return dev_fail("scd_ple_conv: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t td = 4 * (size_t)nt * (size_t)d, rb = 4 * 16 * (size_t)d, kb = 4 * 4 * (size_t)d;
    Buf rbuf, gbuf, nbuf, kbuf, ringb, sb;
    if (!dev_up(*be, r, td, rbuf, err) || !dev_up(*be, gated, td, gbuf, err) || !dev_up(*be, norm, td, nbuf, err) ||
        !dev_up(*be, kw, kb, kbuf, err) || !dev_up(*be, ring, rb, ringb, err) || !dev_ints(*be, {step0, 0}, sb, err) ||
        !be->ple_conv(rbuf.addr(), gbuf.addr(), nbuf.addr(), kbuf.addr(), ringb.addr(), sb.addr(), (size_t)d, (size_t)nt, err) ||
        !dev_down(*be, rbuf, r, td, err) || !dev_down(*be, ringb, ring, rb, err))
        return dev_fail(err);
    return 0;
}

// qsa_pool: pools the blocks of 4 raw indexer keys (t x 128, at positions step0 ...) into pool (nbs x 128, in/out), with the
// raw-key ring (32 x 128, in/out) for the members before step0. RMS-norm with w, then rope on the first n_rot dims at 4 b.
SC_EXPORT int scd_qsa_pool(const float* ik, float* ring, float* pool, const float* w, int step0, int t, int n_rot, float base,
                           float eps, int nbs) {
    g_err.clear();
    if (!ik || !ring || !pool || !w || t <= 0 || nbs <= 0 || n_rot <= 0 || n_rot > 128 || n_rot % 2 != 0)
        return dev_fail("scd_qsa_pool: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t ikb = 4 * (size_t)t * 128, rb = 4 * 32 * 128, pb = 4 * (size_t)nbs * 128;
    Buf ikbuf, ringb, poolb, wb, sb;
    if (!dev_up(*be, ik, ikb, ikbuf, err) || !dev_up(*be, ring, rb, ringb, err) || !dev_up(*be, pool, pb, poolb, err) ||
        !dev_up(*be, w, 4 * 128, wb, err) || !dev_ints(*be, {step0, 0}, sb, err) ||
        !be->qsa_pool(ikbuf.addr(), ringb.addr(), poolb.addr(), wb.addr(), sb.addr(), (size_t)t, (size_t)n_rot, base, eps, err) ||
        !dev_down(*be, ringb, ring, rb, err) || !dev_down(*be, poolb, pool, pb, err))
        return dev_fail(err);
    return 0;
}

// qsa_select for `rows` query rows at r0 ... (q: rows x 4 x 128, normed and roped; pool: nbs x 128). sel: rows x sst ints
// (count, then cells), or with mw > 0 mask: rows x mw u32 (bit b set for each selected block and the tail block).
SC_EXPORT int scd_qsa_select(const float* q, const float* pool, int step0, int r0, int rows, int nbs, int top, int sst, int mw,
                             int32_t* sel, uint32_t* mask) {
    g_err.clear();
    if (!q || !pool || !sel || rows <= 0 || nbs <= 0 || top <= 0 || sst <= 0 || mw < 0 || (mw > 0 && !mask))
        return dev_fail("scd_qsa_select: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t qb_ = 4 * (size_t)rows * 4 * 128, pb = 4 * (size_t)nbs * 128;
    const size_t selb = 4 * (size_t)rows * (size_t)sst, maskb = 4 * (size_t)rows * (size_t)mw;
    Buf qbuf, poolb, scoreb, selbuf, maskbuf, sb;
    if (!dev_up(*be, q, qb_, qbuf, err) || !dev_up(*be, pool, pb, poolb, err) ||
        !dev_up(*be, nullptr, 4 * (size_t)rows * (size_t)nbs, scoreb, err) || !dev_up(*be, nullptr, selb, selbuf, err) ||
        !dev_up(*be, nullptr, maskb > 0 ? maskb : 4, maskbuf, err) || !dev_ints(*be, {step0, 0}, sb, err) ||
        !be->qsa_select(qbuf.addr(), poolb.addr(), scoreb.addr(), selbuf.addr(), mw > 0 ? maskbuf.addr() : 0, sb.addr(),
                        (size_t)r0, (size_t)rows, (size_t)nbs, (size_t)top, (size_t)sst, (size_t)mw, err) ||
        !dev_down(*be, selbuf, sel, selb, err) || (mw > 0 && !dev_down(*be, maskbuf, mask, maskb, err)))
        return dev_fail(err);
    return 0;
}

// rms_norm_rope_qk_rows: nt tokens, token t at position steps[2t] with its own caches kc[t] / vc[t] (cache_len u16 each,
// in/out). q: nt x nh rows (stride q_stride), k and v: nt x nkv rows of hd; qo and ko receive the normed, roped q and k rows.
SC_EXPORT int scd_rms_norm_rope_qk_rows(const float* q, int q_stride, const float* qw, float* qo, int nh, const float* k,
                                        const float* kw, float* ko, int nkv, const float* v, int hd, float eps, int n_rot,
                                        const int32_t* steps, float base, int nt, uint16_t* kc, uint16_t* vc, int cache_len) {
    g_err.clear();
    if (!q || !qw || !qo || !k || !kw || !ko || !v || !steps || !kc || !vc || nh <= 0 || nkv <= 0 || nh % nkv != 0 ||
        hd <= 0 || q_stride < hd || nt <= 0 || cache_len <= 0 || n_rot <= 0)
        return dev_fail("scd_rms_norm_rope_qk_rows: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t qf = (size_t)nt * (size_t)nh * (size_t)q_stride, kf = (size_t)nt * (size_t)nkv * (size_t)hd;
    const size_t qo_b = 4 * (size_t)nt * (size_t)nh * (size_t)hd, kv_b = 4 * kf, cb = 2 * (size_t)cache_len;
    Buf qbuf, qwb, qob, kbuf, kwb, kob, vbuf, sb, kcb, vcb, kvt_b;
    if (!dev_up(*be, q, 4 * qf, qbuf, err) || !dev_up(*be, qw, 4 * (size_t)hd, qwb, err) ||
        !dev_up(*be, nullptr, qo_b, qob, err) || !dev_up(*be, k, kv_b, kbuf, err) ||
        !dev_up(*be, kw, 4 * (size_t)hd, kwb, err) || !dev_up(*be, nullptr, kv_b, kob, err) ||
        !dev_up(*be, v, kv_b, vbuf, err) || !dev_up(*be, steps, 8 * (size_t)nt, sb, err) ||
        !dev_up(*be, kc, cb * (size_t)nt, kcb, err) || !dev_up(*be, vc, cb * (size_t)nt, vcb, err))
        return dev_fail(err);
    const std::vector<uint64_t> kvt = [&] {
        std::vector<uint64_t> v2(2 * (size_t)nt);
        for (size_t t = 0; t < (size_t)nt; t++) {
            v2[2 * t] = kcb.addr() + t * cb;
            v2[2 * t + 1] = vcb.addr() + t * cb;
        }
        return v2;
    }();
    if (!dev_up(*be, kvt.data(), 8 * kvt.size(), kvt_b, err)) return dev_fail(err);
    if (!be->rms_norm_rope_qk_rows(qbuf.addr(), (size_t)q_stride, qwb.addr(), qob.addr(), (size_t)nh, kbuf.addr(), kwb.addr(),
                                   kob.addr(), (size_t)nkv, (size_t)hd, eps, (size_t)n_rot, sb.addr(), base, vbuf.addr(),
                                   kvt_b.addr(), (size_t)nt, err) ||
        !dev_down(*be, qob, qo, qo_b, err) || !dev_down(*be, kob, ko, kv_b, err) ||
        !dev_down(*be, kcb, kc, cb * (size_t)nt, err) || !dev_down(*be, vcb, vc, cb * (size_t)nt, err))
        return dev_fail(err);
    return 0;
}

// attn_rows: decode attention for nt tokens, token t over positions 0 ..= steps[2t] of its own caches kc[t] / vc[t]
// (cache_len u16 each). q and out: nt x nh x hd floats. splits: the per-row split count (ATT_SPLITS_1 in the model).
SC_EXPORT int scd_attn_rows(const float* q, int nh, int nkv, int hd, float scale, const int32_t* steps, int nt,
                            const uint16_t* kc, const uint16_t* vc, int cache_len, int splits, float* out) {
    g_err.clear();
    if (!q || !steps || !kc || !vc || !out || nh <= 0 || nkv <= 0 || nh % nkv != 0 || hd <= 0 || nt <= 0 || cache_len <= 0 ||
        splits <= 0)
        return dev_fail("scd_attn_rows: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t qf = (size_t)nt * (size_t)nh * (size_t)hd;
    const size_t part = (size_t)nt * (size_t)nh * (size_t)splits * (size_t)(hd + 2);
    const size_t cb = 2 * (size_t)cache_len;
    Buf qb, sb, kcb, vcb, ob, pb, kvt_b;
    if (!dev_up(*be, q, 4 * qf, qb, err) || !dev_up(*be, steps, 8 * (size_t)nt, sb, err) ||
        !dev_up(*be, kc, cb * (size_t)nt, kcb, err) || !dev_up(*be, vc, cb * (size_t)nt, vcb, err) ||
        !dev_up(*be, nullptr, 4 * qf, ob, err) || !dev_up(*be, nullptr, 4 * part, pb, err))
        return dev_fail(err);
    const std::vector<uint64_t> kvt = [&] {
        std::vector<uint64_t> v2(2 * (size_t)nt);
        for (size_t t = 0; t < (size_t)nt; t++) {
            v2[2 * t] = kcb.addr() + t * cb;
            v2[2 * t + 1] = vcb.addr() + t * cb;
        }
        return v2;
    }();
    if (!dev_up(*be, kvt.data(), 8 * kvt.size(), kvt_b, err)) return dev_fail(err);
    if (!be->attn_rows(qb.addr(), 0, ob.addr(), pb.addr(), sb.addr(), (size_t)nh, (size_t)nkv, (size_t)hd, scale, (size_t)nt, 0,
                       0, kvt_b.addr(), (size_t)splits, err) ||
        !dev_down(*be, ob, out, 4 * qf, err))
        return dev_fail(err);
    return 0;
}

// moe_prefetch: hints the slots of k experts (ids; table of n_table slots, -1 on the CPU) in the expert bytes base
// (n_table x stride bytes, in/out: a hint leaves the bytes as they are). bytes: the bytes per expert slot to hint.
SC_EXPORT int scd_moe_prefetch(const int32_t* ids, int k, const int32_t* table, int n_table, uint8_t* base, long long base_bytes,
                               int stride, int bytes) {
    g_err.clear();
    if (!ids || !table || !base || k <= 0 || n_table <= 0 || base_bytes <= 0 || stride <= 0 || bytes <= 0 ||
        (long long)n_table * stride > base_bytes || bytes > stride)
        return dev_fail("scd_moe_prefetch: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    Buf idb, tb, bb;
    if (!dev_up(*be, ids, 4 * (size_t)k, idb, err) || !dev_up(*be, table, 4 * (size_t)n_table, tb, err) ||
        !dev_up(*be, base, (size_t)base_bytes, bb, err) ||
        !be->moe_prefetch(idb.addr(), (size_t)k, tb.addr(), bb.addr(), (size_t)stride, (size_t)bytes, err) ||
        !dev_down(*be, bb, base, (size_t)base_bytes, err))
        return dev_fail(err);
    return 0;
}

// Number of times the op function of the last scd_graph_run ran: one capture plus one eager run per input, when the
// graph is captured (a replay does not run it).
static int g_graph_f_calls = 0;
SC_EXPORT int scd_graph_calls(void) { return g_graph_f_calls; }

// Graph replay: one `graph` key queues rms_norm(x) * w, then add y (inside batch), then swiglu with u. For each of nrep
// inputs x (rows of n) the input is uploaded and the graph is run (the first run captures and instantiates it, later runs
// replay it); the output is read back into out_graph. The same inputs run through the eager path into out_eager.
SC_EXPORT int scd_graph_run(const float* x, const float* w, const float* y, const float* u, int n, float eps, int nrep,
                            unsigned long long key, float* out_graph, float* out_eager) {
    g_err.clear();
    g_graph_f_calls = 0;
    if (!x || !w || !y || !u || !out_graph || !out_eager || n <= 0 || nrep <= 0)
        return dev_fail("scd_graph_run: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t nb = 4 * (size_t)n;
    Buf xb, wb, yb, ub, hb;
    if (!dev_up(*be, nullptr, nb, xb, err) || !dev_up(*be, w, nb, wb, err) || !dev_up(*be, y, nb, yb, err) ||
        !dev_up(*be, u, nb, ub, err) || !dev_up(*be, nullptr, nb, hb, err))
        return dev_fail(err);
    const OpFn f = [&](std::string& e) {
        g_graph_f_calls++;
        return be->rms_norm(xb.addr(), (size_t)n, wb.addr(), hb.addr(), (size_t)n, (size_t)n, 1, eps, e) &&
               be->batch([&](std::string& e2) { return be->add(hb.addr(), yb.addr(), (size_t)n, e2); }, e) &&
               be->swiglu(hb.addr(), ub.addr(), (size_t)n, e);
    };
    for (int r = 0; r < nrep; r++) {
        if (!be->htod(x + (size_t)r * n, nb, xb, 0, err) || !be->graph((uint64_t)key, f, err) ||
            !dev_down(*be, hb, out_graph + (size_t)r * n, nb, err))
            return dev_fail(err);
        if (!be->htod(x + (size_t)r * n, nb, xb, 0, err) || !f(err) ||
            !dev_down(*be, hb, out_eager + (size_t)r * n, nb, err))
            return dev_fail(err);
    }
    return 0;
}

// beside + beside_join: the side stream copies z into zb while the main stream waits for it at the join, so the add reads
// the copied values: out = x + z.
SC_EXPORT int scd_beside_copy(const float* x, const float* z, int n, float* out) {
    g_err.clear();
    if (!x || !z || !out || n <= 0) return dev_fail("scd_beside_copy: bad arguments");
    CudaBackend* be = dev_backend();
    if (!be) return 1;
    std::string err;
    const size_t nb = 4 * (size_t)n;
    Buf hb, zb;
    if (!dev_up(*be, x, nb, hb, err) || !dev_up(*be, nullptr, nb, zb, err)) return dev_fail(err);
    const bool ok = be->beside([&](std::string& e) { return be->htod(z, nb, zb, 0, e); }, err) && be->beside_join(err) &&
                    be->add(hb.addr(), zb.addr(), (size_t)n, err);
    if (!ok || !dev_down(*be, hb, out, nb, err)) return dev_fail(err);
    return 0;
}

// ---------------------------------------------------------------------------
// The sampler of the C ABI without a model (host only; no GPU): the exact pick of src/sampling.cpp for the tests.

// Applies the sampler JSON of capi.h (temp, top_k, top_p, min_p, presence_penalty, repeat_penalty, recent, seed) to
// `logits` (n floats) and writes the picked index to `out`. `rng` is the xorshift64 state in/out (null or 0 keeps the
// default 1; the seed in the JSON re-seeds it). Returns 0, or 1 with sci_last_error() set on bad input or non-finite
// logits.
SC_EXPORT int scd_sampler_pick(const float* logits, int n, const char* sampler_json, uint64_t* rng, uint32_t* out) {
    g_err.clear();
    if (!logits || n <= 0 || !out) return dev_fail("scd_sampler_pick: bad arguments");
    strata::Sampler smp;
    std::vector<int32_t> recent;
    double seed = 0;
    bool has_seed = false;
    std::string err;
    if (!strata::sampler_from_json(sampler_json ? sampler_json : "{}", smp, recent, seed, has_seed, err))
        return dev_fail(err);
    smp.rng = (rng && *rng) ? *rng : 1;
    if (has_seed) smp.rng = seed >= 1 ? static_cast<uint64_t>(seed) : 1;
    smp.recent = recent.data();
    smp.recent_n = recent.size();
    size_t pick = 0;
    if (!smp.pick(logits, static_cast<size_t>(n), pick, err)) return dev_fail(err);
    if (rng) *rng = smp.rng;
    *out = static_cast<uint32_t>(pick);
    return 0;
}
