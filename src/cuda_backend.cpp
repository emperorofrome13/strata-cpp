// port of cuda/mod.rs: CUDA context and streams, device buffers, host-mapped mailbox memory, the NVRTC module
// (source_with, Tune, dyn_shmem_need, cache_key / ptx / prune_ptx, kernel_names) and the gemv / moe_gemv launch
// wrappers (gemv_cols, rpw, gemv_block, gemv_cfg, gemvq_pair, moe_gemv, dequant, quant_xq).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Deviations (also in MODIFICATIONS.md): the cubin cache key is FNV-1a over (source, arch, NVRTC tag) rather
// than Rust's DefaultHasher, so cubins are not shared with Strata's cache; the carveout attributes of
// cuda/mod.rs are not applied (they are only set for sm_below_80, and this port is built for the sm_120 card
// it runs on); the CUDA graph capture of `graph` and the part 2b ops are in cuda_backend_seq.cpp.
#include "cuda_backend.h"

#include <immintrin.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "nvrtc.h"
#include "quant.h"

namespace strata {

// The kernel source the build embeds (build/kernel_text.cpp, written by tools/embed_kernels.py from
// kernels/tables.cuh and kernels/kernels.cu, byte for byte).
namespace embed {
extern const unsigned char kTablesCuh[];
extern const size_t kTablesCuhLen;
extern const unsigned char kKernelsCu[];
extern const size_t kKernelsCuLen;
}  // namespace embed

LaunchCfg cfg(unsigned gx, unsigned gy, unsigned block, unsigned smem) {
    LaunchCfg c;
    c.gx = gx;
    c.gy = gy;
    c.gz = 1;
    c.bx = block;
    c.by = 1;
    c.bz = 1;
    c.smem = smem;
    return c;
}

namespace {

// port of cuda/mod.rs: constants
constexpr size_t DEFAULT_SMEM = 48 << 10;
constexpr size_t RING = 64 << 20;
constexpr size_t RING_MIN = 256 << 10;
constexpr size_t BESIDE = SIZE_MAX - 1;
constexpr size_t FA_BQ = 64;
constexpr size_t FA_BK = 32;
constexpr size_t ATTN_SMEM_PAD = 8;
constexpr size_t GEMM_BK_DEFAULT = 64;

// STRATA_TRACE_PREFILL (opt-in): host time of the pinned-ring path of htod, summed until read. Read once from the env;
// unset, the ring path does no timing.
struct HtodTrace {
    bool on = false;
    double memcpy_ms = 0, wait_ms = 0, dma_ms = 0;
    uint64_t calls = 0, chunks = 0, bytes = 0;
};
HtodTrace& htod_trace() {
    static HtodTrace t;
    static bool init = false;
    if (!init) {
        init = true;
        const char* v = std::getenv("STRATA_TRACE_PREFILL");
        t.on = v && *v && std::strcmp(v, "0") != 0;
    }
    return t;
}
double htod_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// port of cuda/mod.rs: KERNELS (the non-family kernels every build loads)
const char* const KERNELS[] = {
    "rope_freq", "qsa_pool", "qsa_score", "qsa_select", "rms_norm", "quant_xq", "argmax_part", "argmax_rows", "l2_norm",
    "rope_neox", "rms_norm_rope", "add", "swiglu", "swiglu_q", "ssm_gates", "conv_silu", "deltanet_seq", "norm_gate",
    "kv_store", "attn_merge", "router_topk", "moe_combine", "moe_route", "conv_seq", "conv_state", "moe_combine_seq",
    "moe_prefetch", "attn_prefill_64", "attn_prefill_128", "attn_prefill_256", "quant_q8", "hc_silu", "hc_mix",
    "hc_write", "ple_gate", "ple_conv",
    // StrataFreet addition (INKLING2_NOTES.md): the inkling2 decode ops.
    "mix_dec", "conv_ink", "clip_rows", "gate_pack", "control_mix", "sig_mul", "situ_q", "rel_table", "softcap",
};

// The kernel types. MXFP4 (the last entry) is a StrataFreet addition (kernels.cu, MODIFICATIONS.md).
const Ty QTYPES[26] = {Ty::F32,   Ty::BF16,  Ty::Q8_0,  Ty::Q2K,   Ty::Q3K,   Ty::Q4K,  Ty::Q5K,  Ty::Q6K,
                       Ty::IQ1S,  Ty::IQ1M,  Ty::IQ2XXS, Ty::IQ2XS, Ty::IQ2S, Ty::IQ3XXS, Ty::IQ3S, Ty::IQ4XS,
                       Ty::IQ4NL, Ty::Q2_0,  Ty::TQ1_0, Ty::TQ2_0, Ty::Q4_1, Ty::Q5_1, Ty::Q4_0, Ty::Q5_0,
                       Ty::MXFP4, Ty::NVFP4};

std::optional<std::string> env_str(const char* k) {
    const char* v = std::getenv(k);
    if (!v) return std::nullopt;
    return std::string(v);
}

bool env_set(const char* k) { return std::getenv(k) != nullptr; }

// Rust `s.trim().parse::<usize>().ok()` for a decimal number.
std::optional<unsigned long long> parse_uint(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return std::nullopt;
    const size_t e = s.find_last_not_of(" \t\r\n");
    const std::string t = s.substr(b, e - b + 1);
    if (t.empty() || t.size() > 18) return std::nullopt;
    for (char c : t)
        if (!std::isdigit((unsigned char)c)) return std::nullopt;
    return std::strtoull(t.c_str(), nullptr, 10);
}

// port of cuda/mod.rs: tuned (`default` unless the variable parses to a value `valid` accepts)
template <class F>
size_t tuned(const char* var, size_t dflt, F valid) {
    if (auto v = env_str(var))
        if (auto p = parse_uint(*v))
            if (valid(*p)) return (size_t)*p;
    return dflt;
}

// port of cuda/mod.rs: rpw_from (family override first, then STRATA_RPW, then the shape default; 1..=64)
int32_t rows_per_warp_from_env(const char* family, size_t dflt) {
    auto ok = [](unsigned long long v) { return v >= 1 && v <= 64; };
    if (auto f = env_str(family))
        if (auto p = parse_uint(*f); p && ok(*p)) return (int32_t)*p;
    if (auto g = env_str("STRATA_RPW"))
        if (auto p = parse_uint(*g); p && ok(*p)) return (int32_t)*p;
    return (int32_t)dflt;
}

// port of cuda/mod.rs: gemv_block_from (threads per block: family, then STRATA_GEMV_BLOCK, then 256; 128/256/512)
uint32_t gemv_block_from_env(const char* family) {
    auto ok = [](unsigned long long v) { return v == 128 || v == 256 || v == 512; };
    if (auto f = env_str(family))
        if (auto p = parse_uint(*f); p && ok(*p)) return (uint32_t)*p;
    if (auto g = env_str("STRATA_GEMV_BLOCK"))
        if (auto p = parse_uint(*g); p && ok(*p)) return (uint32_t)*p;
    return 256;
}

size_t div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }

// port of cuda/mod.rs: kname (prefix + "_" + the type name lower-cased)
std::string kname(const std::string& prefix, Ty ty) {
    std::string s = ty_name(ty);
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return prefix + "_" + s;
}

// port of cuda/mod.rs: float_ty (F32 and BF16 read x as float)
bool float_ty(Ty ty) { return ty == Ty::F32 || ty == Ty::BF16; }

// port of cuda/mod.rs: xq_smem (x as int8 plus its per-32 scales, quantized weight types only)
uint32_t xq_smem(Ty ty, size_t cols) { return float_ty(ty) ? 0 : (uint32_t)(cols + cols / 8); }

// port of cuda/mod.rs: Gpu::rpw (rows per warp of the integer GEMV)
int32_t rows_per_warp(size_t rows, size_t cols, size_t k) {
    size_t d;
    if (k > 1) d = cols <= 512 ? 4 : std::max<size_t>(2048 / cols, 1);
    else if (rows >= 65536) d = 8;
    else if (rows >= 8192 || rows == 2048) d = 2;
    else d = 1;
    return rows_per_warp_from_env(k > 1 ? "STRATA_RPW_MOE" : "STRATA_RPW_DENSE", d);
}

// port of cuda/mod.rs: Gpu::gemv_block (32 * warps threads per block)
uint32_t gemv_block(size_t k) { return gemv_block_from_env(k > 1 ? "STRATA_GEMV_BLOCK_MOE" : "STRATA_GEMV_BLOCK_DENSE"); }

// port of cuda/mod.rs: Gpu::gemv_cols (column count and kernel name; `_p` when cols % 256 != 0)
bool gemv_cols(const char* prefix, Ty ty, size_t cols, int32_t& ncols, std::string& kn, std::string& err) {
    const size_t b = std::max<size_t>(block_elems(ty), 32);
    if (cols % b != 0) {
        err = std::string(ty_name(ty)) + " tensor with " + std::to_string(cols) + " columns: GEMV needs a multiple of " +
              std::to_string(b);
        return false;
    }
    ncols = (int32_t)cols;
    kn = kname(prefix, ty) + (cols % 256 != 0 ? "_p" : "");
    return true;
}

struct GemvCfg {
    LaunchCfg c;
    int32_t rpw;
};

// port of cuda/mod.rs: Gpu::gemv_cfg (launch config and rows-per-warp argument for `k` matrices of rows x cols)
GemvCfg gemv_cfg(Ty ty, size_t rows, size_t cols, size_t k) {
    if (float_ty(ty)) {
        // One warp per row; 8 warps a block from 4096 rows.
        const size_t wpb = rows >= 4096 ? 8 : 1;
        return {cfg((unsigned)div_ceil(rows, wpb), (unsigned)k, (unsigned)(32 * wpb), 0), 1};
    }
    const size_t rpw = (size_t)rows_per_warp(rows, cols, k);
    const size_t wpb = gemv_block(k) / 32;
    return {cfg((unsigned)div_ceil(rows, wpb * rpw), (unsigned)k, (unsigned)(32 * wpb), xq_smem(ty, cols)), (int32_t)rpw};
}

// port of cuda/mod.rs: attn_prefill_smem_bk (shared memory of one attn_prefill launch at a K/V depth of bk)
size_t attn_prefill_smem_bk(size_t hd, size_t bk) {
    const size_t pq = hd + ATTN_SMEM_PAD;
    return 2 * (FA_BQ + 2 * bk) * pq + 2 * FA_BQ * (bk + ATTN_SMEM_PAD);
}

struct DynNeed {
    std::string family;
    size_t need;
};

// port of cuda/mod.rs: dyn_shmem_need (the dynamic shared memory a launch of kernel `name` can request, or none
// for kernels that never ask for more than the 48 KiB default)
std::optional<DynNeed> dyn_shmem_need(const std::string& name) {
    auto starts = [&](const char* p) { return name.rfind(p, 0) == 0; };
    if (starts("attn_prefill")) {
        auto hd = parse_uint(name.substr(std::strlen("attn_prefill_")));
        if (!hd) return std::nullopt;
        return DynNeed{"attn_prefill_*", attn_prefill_smem_bk((size_t)*hd, FA_BK)};
    }
    if (starts("attn_decode_gqa")) {
        std::string rest = name.substr(std::strlen("attn_decode_gqa"));
        const size_t p = rest.find_first_not_of('4');
        rest = p == std::string::npos ? std::string() : rest.substr(p);
        if (rest.empty() || rest[0] != '_') return std::nullopt;
        auto hd = parse_uint(rest.substr(1));
        if (!hd) return std::nullopt;
        // attn_decode_gqa4_128 copies one K/V chunk, the generic kernel two: the larger value covers both.
        const size_t cp = (*hd == 128 || *hd == 256) ? 2 * 2 * 32 * (size_t)*hd * 2 : 0;
        return DynNeed{"attn_decode_gqa*", 8 * (size_t)*hd * 4 + cp};
    }
    if (starts("gemvn")) return DynNeed{"gemvn_*", SIZE_MAX};  // depends on the tensor shape: the device ceiling
    return std::nullopt;
}

// port of cuda/mod.rs: dyn_shmem_optin (the opt-in for `need` clamped to the device opt-in minus the static part;
// none when the 48 KiB default already covers the launch)
std::optional<size_t> dyn_shmem_optin(size_t need, size_t stat, size_t optin) {
    if (need <= DEFAULT_SMEM) return std::nullopt;
    const size_t room = optin > stat ? optin - stat : 0;
    return std::min(need, room);
}

// port of cuda/mod.rs: kernel_names (every kernel the module must provide; the int8 prefill GEMM only on sm_80+)
std::vector<std::string> kernel_names(bool sm80) {
    std::vector<std::string> v;
    for (const char* k : KERNELS) v.push_back(k);
    for (Ty t : QTYPES)
        for (const char* p : {"dequant", "gemv", "gemvn", "moe_gemv", "gemm", "gemms"}) v.push_back(kname(p, t));
    for (Ty t : QTYPES)
        for (const char* p : {"gemv", "gemvn", "moe_gemv"}) v.push_back(kname(p, t) + "_p");
    for (Ty t : QTYPES)
        if (sm80 && !float_ty(t)) {
            v.push_back(kname("gemmi", t));
            v.push_back(kname("gemmis", t));
        }
    for (size_t hd : {64, 128, 256, 512}) {
        v.push_back("attn_decode_" + std::to_string(hd));
        v.push_back("attn_decode_gqa_" + std::to_string(hd));
    }
    v.push_back("attn_decode_gqa4_128");
    return v;
}

// port of cuda/mod.rs: source_with (Tune, macros, then the kernel body). The tables are the first block of
// kernels/tables.cuh, which is the text source_with writes before the macros.
struct Tune {
    size_t gemm_bk;
    size_t gemm_min_blocks;
    size_t gemv_loadw;
};

// port of cuda/mod.rs: Tune::from_env
Tune tune_from_env() {
    Tune t;
    t.gemm_bk = tuned("STRATA_GEMM_BK", GEMM_BK_DEFAULT, [](size_t v) { return v > 0 && v % 8 == 0; });
    t.gemm_min_blocks = tuned("STRATA_GEMM_MIN_BLOCKS", 1, [](size_t v) { return v >= 1 && v <= 8; });
    t.gemv_loadw = tuned("STRATA_GEMV_LOADW", 0, [](size_t v) { return v <= 1; });
    return t;
}

std::string kernel_source(bool kv_q8, bool gemm_tc, const Tune& t) {
    std::string s((const char*)embed::kTablesCuh, embed::kTablesCuhLen);
    char macros[512];
    std::snprintf(macros, sizeof macros,
                  "#define LLAMA_Q8 %d\n#define KV_Q8 %d\n#define GEMM_TC %d\n#define GEMM_BM %zu\n#define GEMM_BK %zu\n"
                  "#define GEMM_MIN_BLOCKS %zu\n#define GEMV_LOADW %zu\n#define FA_BQ %zu\n#define FA_BK %zu\n",
                  env_set("STRATA_LLAMA_NUMERICS") ? 1 : 0, kv_q8 ? 1 : 0, gemm_tc ? 1 : 0, GEMM_BM, t.gemm_bk,
                  t.gemm_min_blocks, t.gemv_loadw, FA_BQ, FA_BK);
    s += macros;
    s.append((const char*)embed::kKernelsCu, embed::kKernelsCuLen);
    return s;
}

// port of cuda/mod.rs: cache_key (FNV-1a; Rust's DefaultHasher is not reproducible, see the file comment)
std::string cache_key(const std::string& src, const std::string& arch, const std::string& tag) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](const std::string& s) {
        for (unsigned char c : s) {
            h ^= c;
            h *= 0x100000001b3ULL;
        }
        h ^= 0xff;
        h *= 0x100000001b3ULL;
    };
    mix(src);
    mix(arch);
    mix(tag);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
    return buf;
}

// port of cuda/mod.rs: the cubin cache directory (XDG_CACHE_HOME, else HOME/.cache, then rusting_strata)
std::filesystem::path cache_dir() {
    std::filesystem::path base;
    if (auto x = env_str("XDG_CACHE_HOME")) base = *x;
    else base = std::filesystem::path(env_str("HOME").value_or("")) / ".cache";
    return base / "rusting_strata";
}

// port of cuda/mod.rs: prune_ptx (cached cubins not used for `age` are deleted)
void prune_cubins(const std::filesystem::path& dir, std::chrono::seconds age) {
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) return;
        std::error_code ec2;
        const auto mt = std::filesystem::last_write_time(e.path(), ec2);
        if (ec2) continue;
        const auto now = std::filesystem::file_time_type::clock::now();
        const bool old = now - mt > age;
        if (old && e.path().extension() == ".cubin") {
            std::error_code ec3;
            std::filesystem::remove(e.path(), ec3);
        }
    }
}

std::string cu_error(const char* what, CUresult r) {
    const char* nm = nullptr;
    const char* text = nullptr;
    cuGetErrorName(r, &nm);
    cuGetErrorString(r, &text);
    return std::string(what) + " failed: " + (nm ? nm : "?") + " (" + (text ? text : "?") + ")";
}

// port of cuda/mod.rs: min_arch_error (sm_70 or newer)
std::optional<std::string> min_arch_error(const std::string& name, int major, int minor) {
    if (std::make_pair(major, minor) < std::make_pair(7, 0)) {
        return name + " is sm_" + std::to_string(major) + std::to_string(minor) +
               ": the CUDA backend requires sm_70 or newer (Volta, Turing, Ampere and later); Pascal and older cards "
               "have no tensor cores and are not supported";
    }
    return std::nullopt;
}

// port of cuda/mod.rs: arch_support_error (NVRTC 13.x cannot target sm_70 / sm_72)
std::optional<std::string> arch_support_error(int major, int minor, const std::pair<int, int>& nv) {
    if (std::make_pair(major, minor) < std::make_pair(7, 5) && nv.first >= 13) {
        return "NVRTC " + std::to_string(nv.first) + "." + std::to_string(nv.second) + " cannot compile for sm_" +
               std::to_string(major) + std::to_string(minor) +
               ": 13.x removed Volta (sm_70) support; install toolkit/NVRTC 12.x";
    }
    return std::nullopt;
}

// port of cuda/mod.rs: device_log (the startup line for the device)
std::string device_log(int device, const std::string& name, const std::string& arch, int major, int minor, bool gemm_fma) {
    const std::pair<int, int> cc{major, minor};
    const char* path = "";
    if (cc < std::make_pair(7, 0)) path = " (pre-Volta, not supported)";
    else if (cc < std::make_pair(7, 5)) path = gemm_fma ? " (Volta fallback)" : " (Volta, m8n8k4)";
    else if (cc < std::make_pair(8, 0)) path = " (Turing, m16n8k8)";
    return "cuda device " + std::to_string(device) + ": " + name + ", " + arch + path;
}

// port of cuda/mod.rs: ptx (the cubin for `arch`, reused from the disk cache when present)
bool load_cubin(const std::string& src, const std::string& arch, int major, int minor, std::vector<uint8_t>& out,
                std::string& err) {
    const auto nv = strata::nvrtc::version();
    if (nv) {
        if (auto msg = arch_support_error(major, minor, *nv)) {
            err = *msg;
            return false;
        }
    }
    std::string tag = nv ? std::to_string(nv->first) + "." + std::to_string(nv->second) : "unverified";
    tag += " " + strata::nvrtc::extra_opts();
    while (!tag.empty() && (tag.back() == ' ' || tag.back() == '\t')) tag.pop_back();
    const auto dir = cache_dir();
    const auto path = dir / (cache_key(src, arch, tag) + ".cubin");

    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        std::ifstream f(path, std::ios::binary);
        if (f) {
            out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            f.close();
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
            return !out.empty();
        }
    }
    // The NVRTC library is needed past this point; a missing one is the error to report.
    if (!strata::nvrtc::present()) {
        err = strata::nvrtc::missing_message();
        return false;
    }
    std::filesystem::create_directories(dir, ec);
    prune_cubins(dir, std::chrono::seconds(7 * 86400));
    std::fprintf(stderr, "compiling CUDA kernels for %s with NVRTC %s (first run only, ~15 s)\n", arch.c_str(), tag.c_str());
    if (!strata::nvrtc::cubin(src, arch, out, err)) return false;
    std::ofstream w(path, std::ios::binary);
    if (w) w.write((const char*)out.data(), (std::streamsize)out.size());
    return true;
}

// Device memory and pinned host memory, freed with their last owner (in the context they came from).
struct DevMem {
    CUcontext ctx;
    CUdeviceptr p;
    ~DevMem() {
        if (p) {
            cuCtxSetCurrent(ctx);
            cuMemFree(p);
        }
    }
};

struct HostMem {
    CUcontext ctx;
    void* p;
    ~HostMem() {
        if (p) {
            cuCtxSetCurrent(ctx);
            cuMemFreeHost(p);
        }
    }
};

}  // namespace

namespace cuda_detail {

std::string kname(const std::string& prefix, Ty ty) { return strata::kname(prefix, ty); }
size_t div_ceil(size_t a, size_t b) { return strata::div_ceil(a, b); }
bool float_ty(Ty ty) { return strata::float_ty(ty); }
size_t attn_prefill_smem_bk(size_t hd, size_t bk) { return strata::attn_prefill_smem_bk(hd, bk); }
std::string cu_error(const char* what, CUresult r) { return strata::cu_error(what, r); }

}  // namespace cuda_detail

// port of cuda/mod.rs: Gpu::new
std::unique_ptr<CudaBackend> CudaBackend::create(int device, std::string& err, bool q8_requested) {
    const bool kv_q8 = q8_requested || env_set("STRATA_KV_Q8");
    if (kv_q8 && env_set("STRATA_LLAMA_NUMERICS")) {
        err = "--kv q8 does not combine with --llama-numerics (its attention reads f16 KV)";
        return nullptr;
    }
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        err = "CUDA init: " + cu_error("cuInit", r);
        return nullptr;
    }
    auto be = std::unique_ptr<CudaBackend>(new CudaBackend());
    r = cuDeviceGet(&be->dev_, device);
    if (r != CUDA_SUCCESS) {
        err = "CUDA init: " + cu_error("cuDeviceGet", r);
        return nullptr;
    }
    int major = 0, minor = 0;
    if ((r = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, be->dev_)) != CUDA_SUCCESS ||
        (r = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, be->dev_)) != CUDA_SUCCESS) {
        err = "CUDA init: " + cu_error("cuDeviceGetAttribute(compute capability)", r);
        return nullptr;
    }
    char namebuf[256] = {0};
    cuDeviceGetName(namebuf, sizeof namebuf, be->dev_);
    be->name_ = namebuf;
    if (auto msg = min_arch_error(be->name_, major, minor)) {
        err = "CUDA init: " + *msg;
        return nullptr;
    }
    be->cc_major_ = major;
    be->cc_minor_ = minor;
    be->int8_mma_ = major >= 8;
    be->i8_gemm_ = major >= 8;
    be->turing_ = !be->int8_mma_ && std::make_pair(major, minor) >= std::make_pair(7, 5);
    int optin_i = 0;
    cuDeviceGetAttribute(&optin_i, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, be->dev_);
    const size_t optin = std::max<size_t>((size_t)optin_i, DEFAULT_SMEM);
    be->max_smem_ = optin;
    const std::string arch = "sm_" + std::to_string(major) + std::to_string(minor);
    const bool gemm_fma = env_set("STRATA_GEMM_FMA");
    std::fprintf(stderr, "%s\n", device_log(device, be->name_, arch, major, minor, gemm_fma).c_str());

    // The primary context is the one the CUDA runtime uses, so this does not fork a second context next to
    // StrataFreet's own work in the same process.
    r = cuDevicePrimaryCtxRetain(&be->ctx_, be->dev_);
    if (r != CUDA_SUCCESS) {
        be->ctx_ = nullptr;
        err = "CUDA init: " + cu_error("cuDevicePrimaryCtxRetain", r);
        return nullptr;
    }
    if ((r = cuCtxSetCurrent(be->ctx_)) != CUDA_SUCCESS) {
        err = "CUDA init: " + cu_error("cuCtxSetCurrent", r);
        return nullptr;
    }
    if ((r = cuStreamCreate(&be->stream_, CU_STREAM_NON_BLOCKING)) != CUDA_SUCCESS ||
        (r = cuStreamCreate(&be->side_, CU_STREAM_NON_BLOCKING)) != CUDA_SUCCESS) {
        err = cu_error("cuStreamCreate", r);
        return nullptr;
    }

    const bool gemm_tc = !gemm_fma && std::make_pair(major, minor) < std::make_pair(7, 5);
    const std::string src = kernel_source(kv_q8, gemm_tc, tune_from_env());
    std::vector<uint8_t> cubin;
    if (!load_cubin(src, arch, major, minor, cubin, err)) return nullptr;
    if ((r = cuModuleLoadData(&be->module_, cubin.data())) != CUDA_SUCCESS) {
        err = "load cubin: " + cu_error("cuModuleLoadData", r);
        return nullptr;
    }
    for (const std::string& n : kernel_names(be->i8_gemm_)) {
        CUfunction f = nullptr;
        if ((r = cuModuleGetFunction(&f, be->module_, n.c_str())) != CUDA_SUCCESS) {
            err = "kernel " + n + ": " + cu_error("cuModuleGetFunction", r);
            return nullptr;
        }
        int stat = 0;
        if ((r = cuFuncGetAttribute(&stat, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f)) != CUDA_SUCCESS) {
            err = "static shared memory of " + n + ": " + cu_error("cuFuncGetAttribute", r);
            return nullptr;
        }
        // Turing runs the head_dim 256 prefill attention at 16 keys a tile (FA_BK_FOR in kernels.cu).
        auto need = dyn_shmem_need(n);
        if (need && n == "attn_prefill_256" && be->turing_) need = DynNeed{need->family, attn_prefill_smem_bk(256, 16)};
        if (need) {
            if (auto set = dyn_shmem_optin(need->need, (size_t)stat, optin)) {
                if ((r = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)*set)) != CUDA_SUCCESS) {
                    err = "set dynamic shared memory on " + n + ": " + cu_error("cuFuncSetAttribute", r);
                    return nullptr;
                }
            }
        }
        be->fns_[n] = f;
    }
    if ((r = cuMemHostAlloc(&be->ring_, RING, 0)) != CUDA_SUCCESS) {
        err = "pinned staging ring: " + cu_error("cuMemHostAlloc", r);
        return nullptr;
    }
    if ((r = cuEventCreate(&be->done_, CU_EVENT_DISABLE_TIMING)) != CUDA_SUCCESS) {
        err = cu_error("cuEventCreate", r);
        return nullptr;
    }
    // port of cuda/mod.rs: Gpu::new, the argmax partials (64 rows of kAmaxParts int2)
    if (!be->alloc(8 * cuda_detail::kAmaxParts * cuda_detail::kAmaxParts, be->am_, err)) return nullptr;
    be->kv_q8_ = kv_q8;
    return be;
}

CudaBackend::~CudaBackend() {
    if (!ctx_) return;
    cuCtxSetCurrent(ctx_);
    // Device buffers owned by the backend go before the context is released.
    q8_ = Buf();
    am_ = Buf();
    fr_.clear();
    for (auto& kv : graphs_) cuGraphExecDestroy(kv.second);
    graphs_.clear();
    if (lag_) cuEventDestroy(lag_);
    for (const RingBusy& b : busy_) {
        cuEventSynchronize(b.ev);
        cuEventDestroy(b.ev);
    }
    if (done_) cuEventDestroy(done_);
    if (arena_mem_) cuMemFree(arena_mem_);
    unregister_host();
    if (ring_) cuMemFreeHost(ring_);
    if (module_) cuModuleUnload(module_);
    if (side_) cuStreamDestroy(side_);
    if (stream_) cuStreamDestroy(stream_);
    cuDevicePrimaryCtxRelease(dev_);
}

bool CudaBackend::bind(std::string& err) {
    // Every backend op binds the context. The set costs ~5 us a call (the prefill issues thousands of ops), so it is
    // made only when this thread's current context is another one.
    CUcontext cur = nullptr;
    if (cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == ctx_) return true;
    const CUresult r = cuCtxSetCurrent(ctx_);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuCtxSetCurrent", r);
        return false;
    }
    return true;
}

// port of cuda/mod.rs: Gpu::wait (records an event and polls it with pauses between queries)
bool CudaBackend::wait(std::string& err) {
    if (!bind(err)) return false;
    CUresult r = cuEventRecord(done_, stream_);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuEventRecord", r);
        return false;
    }
    for (;;) {
        r = cuEventQuery(done_);
        if (r == CUDA_SUCCESS) return true;
        if (r != CUDA_ERROR_NOT_READY) {
            err = cu_error("cuEventQuery", r);
            return false;
        }
        for (int i = 0; i < 256; i++) _mm_pause();
    }
}

// port of cuda/mod.rs: Gpu::st (the stream for the next launch: inside batch, odd launches go to the side stream)
CUstream CudaBackend::st() {
    const size_t n = nb_.load();
    if (n >= BESIDE) return stream_;
    nb_.store(n + 1);
    return (n % 2 == 1) ? side_ : stream_;
}

// Launch `kname` with `c`. Each argument pointer is in `args`, in the kernel's parameter order.
bool CudaBackend::launch_raw(const char* kn, const LaunchCfg& c, void** args, std::string& err) {
    auto it = fns_.find(kn);
    if (it == fns_.end()) {
        err = std::string("kernel ") + kn + " is not loaded";
        return false;
    }
    if (!bind(err)) return false;
    const CUresult r = cuLaunchKernel(it->second, c.gx, c.gy, c.gz, c.bx, c.by, c.bz, c.smem, st(), args, nullptr);
    if (r != CUDA_SUCCESS) {
        err = std::string("launch ") + kn + ": " + cu_error("cuLaunchKernel", r);
        return false;
    }
    // STRATA_CUDA_SYNC: wait for every kernel right after its launch, so a device error names the kernel.
    if (env_set("STRATA_CUDA_SYNC")) {
        if (cuStreamSynchronize(stream_) != CUDA_SUCCESS || cuStreamSynchronize(side_) != CUDA_SUCCESS) {
            err = std::string("kernel ") + kn + " failed";
            return false;
        }
    }
    return true;
}

// Make `waiter` wait for the work queued on `waited` so far.
bool CudaBackend::join(CUstream waiter, CUstream waited, std::string& err) {
    CUevent e = nullptr;
    CUresult r = cuEventCreate(&e, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) r = cuEventRecord(e, waited);
    if (r == CUDA_SUCCESS) r = cuStreamWaitEvent(waiter, e, 0);
    if (e) cuEventDestroy(e);
    if (r != CUDA_SUCCESS) {
        err = cu_error("join", r);
        return false;
    }
    return true;
}

// port of cuda/mod.rs: Gpu::alloc (zeroed, at least 1 byte)
// port of cuda/mod.rs: Gpu::alloc's memory accounting (exact sizes, see reserve_arena in the header)
bool CudaBackend::reserve_arena(size_t bytes, std::string& err) {
    if (!bind(err)) return false;
    if (arena_mem_) {
        err = "reserve_arena: a block is already reserved";
        return false;
    }
    const size_t unit = size_t(2) << 20;
    const size_t n = std::max(unit, (bytes + unit - 1) / unit * unit);
    CUdeviceptr p = 0;
    const CUresult r = cuMemAlloc(&p, n);
    if (r != CUDA_SUCCESS) {
        err = "reserve_arena of " + std::to_string(n) + " bytes: " + cu_error("cuMemAlloc", r);
        return false;
    }
    arena_mem_ = p;
    arena_size_ = n;
    arena_used_ = 0;
    return true;
}

bool CudaBackend::alloc(size_t bytes, Buf& out, std::string& err) {
    if (!bind(err)) return false;
    const size_t n = bytes ? bytes : 1;
    if (arena_mem_) {
        const size_t need = (n + 255) & ~size_t(255);
        if (arena_used_ + need <= arena_size_) {
            const CUdeviceptr p = arena_mem_ + arena_used_;
            arena_used_ += need;
            CUresult r = cuMemsetD8Async(p, 0, n, stream_);
            if (r != CUDA_SUCCESS) {
                err = cu_error("cuMemsetD8Async", r);
                return false;
            }
            out = Buf((uint64_t)p, n, std::shared_ptr<void>());
            return true;
        }
    }
    CUdeviceptr p = 0;
    CUresult r = cuMemAlloc(&p, n);
    if (r != CUDA_SUCCESS) {
        err = "alloc of " + std::to_string(n) + " bytes: " + cu_error("cuMemAlloc", r);
        return false;
    }
    std::shared_ptr<void> owner(new DevMem{ctx_, p});
    // On the main stream, so the zeroing is ordered before the launches that follow. The legacy-stream memset
    // (cuMemsetD8) is not ordered against the non-blocking streams and could zero the buffer after a kernel wrote it.
    if ((r = cuMemsetD8Async(p, 0, n, stream_)) != CUDA_SUCCESS) {
        err = cu_error("cuMemsetD8Async", r);
        return false;
    }
    out = Buf((uint64_t)p, n, std::move(owner));
    return true;
}

// port of cuda/mod.rs: Gpu::htod (returns once `src` is copied; large copies go through the pinned ring)
bool CudaBackend::htod(const void* src, size_t n, const Buf& dst, size_t off, std::string& err) {
    if (off + n > dst.len()) {
        err = "htod: " + std::to_string(n) + " bytes at " + std::to_string(off) + " into " + std::to_string(dst.len());
        return false;
    }
    if (!bind(err)) return false;
    // Inside `batch` or `beside` a copy goes to the side stream, beside the next launch on the main stream.
    const CUstream s = nb_.load() == SIZE_MAX ? stream_ : side_;
    // StrataFreet addition: a source inside a registered host range (sc_load pin) is DMA'd directly, so the copy
    // through the pinned ring (a memcpy of every byte) is skipped. The range outlives every copy (it is the model's
    // RAM copy, freed after the backend).
    const bool reg = host_registered(src, n);
    if (n < RING_MIN || reg) {
        const CUresult r = cuMemcpyHtoDAsync((CUdeviceptr)(dst.addr() + off), src, n, s);
        if (r != CUDA_SUCCESS) {
            err = cu_error("cuMemcpyHtoDAsync", r);
            return false;
        }
        if (reg) host_direct_bytes_.fetch_add(n);
        return true;
    }
    std::lock_guard<std::mutex> g(ring_mu_);
    const char* data = (const char*)src;
    for (size_t i = 0; i * RING < n; i++) {
        const size_t len = std::min(RING, n - i * RING);
        if (ring_head_ + len > RING) ring_head_ = 0;
        const size_t a = ring_head_, b = ring_head_ + len;
        const bool tr = htod_trace().on;
        const double t0 = tr ? htod_now_ms() : 0.0;
        while (std::any_of(busy_.begin(), busy_.end(), [&](const RingBusy& x) { return x.a < b && a < x.b; })) {
            RingBusy f = busy_.front();
            busy_.pop_front();
            cuEventSynchronize(f.ev);
            cuEventDestroy(f.ev);
        }
        const double t1 = tr ? htod_now_ms() : 0.0;
        char* p = (char*)ring_ + a;
        std::memcpy(p, data + i * RING, len);
        const double t2 = tr ? htod_now_ms() : 0.0;
        CUresult r = cuMemcpyHtoDAsync((CUdeviceptr)(dst.addr() + off + i * RING), p, len, s);
        if (tr) {
            const double t3 = htod_now_ms();
            HtodTrace& ht = htod_trace();
            ht.wait_ms += t1 - t0;
            ht.memcpy_ms += t2 - t1;
            ht.dma_ms += t3 - t2;
            ht.chunks++;
            ht.bytes += len;
        }
        if (r != CUDA_SUCCESS) {
            err = cu_error("cuMemcpyHtoDAsync", r);
            return false;
        }
        CUevent ev = nullptr;
        if ((r = cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING)) != CUDA_SUCCESS || (r = cuEventRecord(ev, s)) != CUDA_SUCCESS) {
            err = cu_error("cuEventRecord", r);
            return false;
        }
        busy_.push_back({a, b, ev});
        ring_head_ = b;
    }
    return true;
}

// StrataFreet addition: cuMemHostRegister of one host range. The read-only flag is asked first (the device only reads
// weights); a driver that refuses it gets a plain portable registration. Its error is reported when both fail.
bool CudaBackend::register_host(const void* p, size_t n, std::string& err) {
    if (!bind(err)) return false;
    if (n == 0) {
        err = "register_host: empty range";
        return false;
    }
    CUresult r = cuMemHostRegister(const_cast<void*>(p), n, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_READ_ONLY);
    if (r != CUDA_SUCCESS) {
        const CUresult r2 = cuMemHostRegister(const_cast<void*>(p), n, CU_MEMHOSTREGISTER_PORTABLE);
        if (r2 != CUDA_SUCCESS) {
            err = cu_error("cuMemHostRegister", r2) + " (read-only attempt: " + cu_error("cuMemHostRegister", r) + ")";
            return false;
        }
    }
    host_regs_.push_back({static_cast<const char*>(p), n});
    return true;
}

// Unregisters every range this backend registered. Synchronizes first: a registered source must not be read by a
// copy still queued when its memory is freed.
void CudaBackend::unregister_host() {
    if (host_regs_.empty()) return;
    std::string ignored;
    if (bind(ignored)) cuCtxSynchronize();
    for (const auto& r : host_regs_) cuMemHostUnregister(const_cast<char*>(r.first));
    host_regs_.clear();
}

bool CudaBackend::host_registered(const void* p, size_t n) const {
    const char* a = static_cast<const char*>(p);
    for (const auto& r : host_regs_) {
        if (a >= r.first && n <= r.second && static_cast<size_t>(a - r.first) <= r.second - n) return true;
    }
    return false;
}

// port of cuda/mod.rs: Gpu::dtoh (waits for the queued work, copies, then syncs the stream)
bool CudaBackend::dtoh(const Buf& src, size_t off, void* dst, size_t n, std::string& err) {
    if (off + n > src.len()) {
        err = "dtoh: " + std::to_string(n) + " bytes at " + std::to_string(off) + " from " + std::to_string(src.len());
        return false;
    }
    if (!bind(err) || !wait(err)) return false;
    CUresult r = cuMemcpyDtoHAsync(dst, (CUdeviceptr)(src.addr() + off), n, stream_);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuMemcpyDtoHAsync", r);
        return false;
    }
    if ((r = cuStreamSynchronize(stream_)) != CUDA_SUCCESS) {
        err = cu_error("cuStreamSynchronize", r);
        return false;
    }
    return true;
}

bool CudaBackend::dtod(const Buf& src, size_t src_off, const Buf& dst, size_t dst_off, size_t bytes, std::string& err) {
    if (src_off + bytes > src.len() || dst_off + bytes > dst.len()) {
        err = "dtod: " + std::to_string(bytes) + " bytes out of range (src " + std::to_string(src.len()) + " @ " +
              std::to_string(src_off) + ", dst " + std::to_string(dst.len()) + " @ " + std::to_string(dst_off) + ")";
        return false;
    }
    if (!bind(err)) return false;
    const CUresult r = cuMemcpyDtoDAsync((CUdeviceptr)(dst.addr() + dst_off), (CUdeviceptr)(src.addr() + src_off), bytes, stream_);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuMemcpyDtoDAsync", r);
        return false;
    }
    return true;
}

bool CudaBackend::memset(const Buf& dst, std::string& err) {
    if (!bind(err)) return false;
    const CUresult r = cuMemsetD8Async((CUdeviceptr)dst.addr(), 0, dst.len(), stream_);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuMemsetD8Async", r);
        return false;
    }
    return true;
}

bool CudaBackend::mem_info(size_t& free_bytes, size_t& total_bytes, std::string& err) {
    if (!bind(err)) return false;
    const CUresult r = cuMemGetInfo(&free_bytes, &total_bytes);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuMemGetInfo", r);
        return false;
    }
    return true;
}

bool CudaBackend::sync(std::string& err) { return wait(err); }

// port of cuda/mod.rs: Gpu::batch (odd launches of `f` go to the side stream; the streams join after it)
bool CudaBackend::batch(const OpFn& f, std::string& err) {
    if (!bind(err) || !join(side_, stream_, err)) return false;
    nb_.store(0);
    const bool ok = f(err);
    nb_.store(SIZE_MAX);
    if (!join(stream_, side_, err)) return false;
    return ok;
}

// port of cuda/mod.rs: Gpu::host_mailbox (pinned host memory, mapped into the device address space)
bool CudaBackend::host_mailbox(size_t words, Buf& buf, uint32_t*& host, std::string& err) {
    if (!bind(err)) return false;
    const size_t n = words ? words : 1;
    void* h = nullptr;
    CUresult r = cuMemHostAlloc(&h, 4 * n, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuMemHostAlloc", r);
        return false;
    }
    std::shared_ptr<void> owner(new HostMem{ctx_, h});
    std::memset(h, 0, 4 * n);
    CUdeviceptr d = 0;
    r = cuMemHostGetDevicePointer(&d, h, 0);
    if (r != CUDA_SUCCESS) {
        err = cu_error("cuMemHostGetDevicePointer", r);
        return false;
    }
    buf = Buf((uint64_t)d, 4 * n, std::move(owner));
    host = (uint32_t*)h;
    return true;
}

// port of cuda/mod.rs: Gpu::gemvq_pair (`gemvq`; with z = 2 also the matrix of `pair` into the output at `pair`)
bool CudaBackend::gemvq_pair(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t xq, uint64_t y, size_t y_stride,
                             size_t nt, Pair pair, unsigned z, std::string& err) {
    if (nt < 1 || nt > 4) {
        err = "gemvn: " + std::to_string(nt) + " inputs (1 to 4)";
        return false;
    }
    int32_t ncols = 0;
    std::string k1;
    if (!gemv_cols("gemv", w.ty, w.cols, ncols, k1, err)) return false;
    GemvCfg g = gemv_cfg(w.ty, w.rows, w.cols, 1);
    LaunchCfg c = g.c;
    const unsigned per = c.smem;
    c.smem = xq == 0 ? per * (unsigned)nt : 0;
    c.gz = z;
    const int32_t rows = (int32_t)w.rows;
    // Turing-sized shared memory cannot hold nt cached x rows of a wide matrix: one token at a time there.
    if (xq == 0 && c.smem > max_smem_) {
        c.smem = per;
        for (size_t t = 0; t < nt; t++) {
            const uint64_t xt = x + 4 * t * x_stride, yt = y + 4 * t * y_stride;
            if (!launch(k1.c_str(), c, err, w.ptr, xt, yt, rows, ncols, g.rpw, xq, pair.w2, pair.y2, pair.rows2)) return false;
        }
        return true;
    }
    // One token: gemv, whose kernel has no nt branches.
    if (nt == 1) return launch(k1.c_str(), c, err, w.ptr, x, y, rows, ncols, g.rpw, xq, pair.w2, pair.y2, pair.rows2);
    const std::string kn = "gemvn" + k1.substr(4);  // k1 starts with "gemv": replacen("gemv", "gemvn", 1)
    return launch(kn.c_str(), c, err, w.ptr, x, (int32_t)x_stride, y, (int32_t)y_stride, rows, ncols, g.rpw, (int32_t)nt, xq,
                  pair.w2, pair.y2, pair.rows2);
}

bool CudaBackend::gemv(const GpuTensor& w, uint64_t x, uint64_t y, std::string& err) {
    int32_t ncols = 0;
    std::string kn;
    if (!gemv_cols("gemv", w.ty, w.cols, ncols, kn, err)) return false;
    const GemvCfg g = gemv_cfg(w.ty, w.rows, w.cols, 1);
    return launch(kn.c_str(), g.c, err, w.ptr, x, y, (int32_t)w.rows, ncols, g.rpw, (uint64_t)0, (int64_t)0, (int64_t)0,
                  (int32_t)0);
}

bool CudaBackend::gemvn(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t y, size_t y_stride, size_t nt, std::string& err) {
    return gemvq(w, x, x_stride, 0, y, y_stride, nt, err);
}

bool CudaBackend::gemvq(const GpuTensor& w, uint64_t x, size_t x_stride, uint64_t xq, uint64_t y, size_t y_stride, size_t nt,
                        std::string& err) {
    return gemvq_pair(w, x, x_stride, xq, y, y_stride, nt, Pair{}, 1, err);
}

// port of cuda/mod.rs: Gpu::gemvq2 (one launch for both matrices, sized for the larger one)
bool CudaBackend::gemvq2(const GpuTensor& a, const GpuTensor& b, uint64_t x, size_t x_stride, uint64_t xq, uint64_t ya,
                         uint64_t yb, size_t nt, std::string& err) {
    if (a.ty != b.ty || a.cols != b.cols) {
        return gemvq(a, x, x_stride, xq, ya, a.rows, nt, err) && gemvq(b, x, x_stride, xq, yb, b.rows, nt, err);
    }
    const GpuTensor* A = &a;
    const GpuTensor* B = &b;
    uint64_t YA = ya, YB = yb;
    if (a.rows < b.rows) {
        std::swap(A, B);
        std::swap(YA, YB);
    }
    Pair p;
    p.w2 = (int64_t)B->ptr - (int64_t)A->ptr;
    p.y2 = (int64_t)YB - (int64_t)YA;
    p.rows2 = (int32_t)B->rows;
    return gemvq_pair(*A, x, x_stride, xq, YA, A->rows, nt, p, 2, err);
}

// port of cuda/mod.rs: Gpu::moe_gemv (grid.y = k experts; `pair` runs a second matrix on grid.z)
bool CudaBackend::moe_gemv(Ty ty, size_t rows, size_t cols, uint64_t w, size_t stride, uint64_t ids, size_t k, uint64_t x,
                           size_t x_stride, size_t xg, uint64_t y, std::optional<std::pair<int64_t, int64_t>> pair, uint64_t xq,
                           std::string& err) {
    int32_t ncols = 0;
    std::string kn;
    if (!gemv_cols("moe_gemv", ty, cols, ncols, kn, err)) return false;
    GemvCfg g = gemv_cfg(ty, rows, cols, k);
    LaunchCfg c = g.c;
    int32_t rpw = g.rpw;
    // At least 2 rows per warp: an expert's x quantization (once per block) then covers 2 * warps rows.
    if (rpw == 1 && !float_ty(ty)) {
        rpw = 2;
        c.gx = (unsigned)div_ceil(rows, 2 * (gemv_block(k) / 32));
    }
    int64_t w2 = 0, y2 = 0;
    if (pair) {
        w2 = pair->first;
        y2 = pair->second;
    }
    c.gz = pair ? 2 : 1;
    if (xq != 0) c.smem = 0;
    return launch(kn.c_str(), c, err, w, ids, (int64_t)stride, x, (int32_t)x_stride, (int32_t)xg, y, (int32_t)rows, ncols, rpw,
                  w2, y2, xq);
}

// port of cuda/mod.rs: Gpu::dequant (test helper: one block per row)
bool CudaBackend::dequant(const GpuTensor& w, size_t rows, uint64_t out, std::string& err) {
    int32_t ncols = 0;
    std::string unused;
    if (!gemv_cols("dequant", w.ty, w.cols, ncols, unused, err)) return false;
    return launch(kname("dequant", w.ty).c_str(), cfg((unsigned)rows, 1, 256, 0), err, w.ptr, out, ncols);
}

// port of cuda/mod.rs: Gpu::quant_xq (one block per row)
bool CudaBackend::quant_xq(uint64_t x, size_t x_stride, size_t n, size_t nt, uint64_t out, std::string& err) {
    return launch("quant_xq", cfg((unsigned)nt, 1, 256, 0), err, x, (int32_t)x_stride, (int32_t)n, out);
}

// The part 2b ops (DeltaNet, conv, QSA, hyper-connections, PLE, graph capture) are in cuda_backend_seq.cpp.

// STRATA_TRACE_PREFILL: reads and resets the ring-path counters of htod (zeros when tracing is off).
void htod_trace_take(double& memcpy_ms, double& wait_ms, double& dma_ms, uint64_t& chunks, uint64_t& bytes) {
    HtodTrace& ht = htod_trace();
    memcpy_ms = ht.memcpy_ms;
    wait_ms = ht.wait_ms;
    dma_ms = ht.dma_ms;
    chunks = ht.chunks;
    bytes = ht.bytes;
    ht.memcpy_ms = ht.wait_ms = ht.dma_ms = 0;
    ht.chunks = ht.bytes = 0;
}
}  // namespace strata
