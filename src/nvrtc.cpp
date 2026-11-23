// port of cuda/nvrtc.rs.
#include "nvrtc.h"

#include <windows.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace strata::nvrtc {
namespace {

using Prog = void*;
using FnVersion = int (*)(int*, int*);
using FnCreate = int (*)(Prog*, const char*, const char*, int, const char* const*, const char* const*);
using FnCompile = int (*)(Prog, int, const char* const*);
using FnSize = int (*)(Prog, size_t*);
using FnGet = int (*)(Prog, char*);
using FnDestroy = int (*)(Prog*);

struct Lib {
    FnVersion version = nullptr;
    FnCreate create = nullptr;
    FnCompile compile = nullptr;
    FnSize log_size = nullptr;
    FnGet log = nullptr;
    FnSize cubin_size = nullptr;
    FnGet cubin = nullptr;
    FnDestroy destroy = nullptr;
    HMODULE handle = nullptr;
    bool ok = false;
};

std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }

// port of cuda/nvrtc.rs: open()
Lib open_one(const std::string& path) {
    Lib lib;
    std::filesystem::path p(path);
    const bool full = p.has_parent_path();
    HMODULE h = full ? LoadLibraryExW(p.wstring().c_str(), nullptr,
                                      LOAD_WITH_ALTERED_SEARCH_PATH | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
                     : LoadLibraryW(widen(path).c_str());
    if (!h) return lib;

    // NVRTC loads `nvrtc-builtins64_*.dll` by bare name at compile time. Windows does not search the
    // directory of a DLL loaded by full path, so load the builtins first: a bare-name load then resolves
    // to the module already in memory. The handles are deliberately leaked for the process lifetime.
    if (full && p.parent_path().is_absolute()) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(p.parent_path(), ec)) {
            const std::string name = e.path().filename().string();
            if (name.rfind("nvrtc-builtins", 0) == 0) LoadLibraryW(e.path().wstring().c_str());
        }
    }

    auto sym = [h](const char* n) -> void* { return (void*)GetProcAddress(h, n); };
    lib.version = (FnVersion)sym("nvrtcVersion");
    lib.create = (FnCreate)sym("nvrtcCreateProgram");
    lib.compile = (FnCompile)sym("nvrtcCompileProgram");
    lib.log_size = (FnSize)sym("nvrtcGetProgramLogSize");
    lib.log = (FnGet)sym("nvrtcGetProgramLog");
    lib.cubin_size = (FnSize)sym("nvrtcGetCUBINSize");
    lib.cubin = (FnGet)sym("nvrtcGetCUBIN");
    lib.destroy = (FnDestroy)sym("nvrtcDestroyProgram");
    lib.handle = h;
    lib.ok = lib.version && lib.create && lib.compile && lib.log_size && lib.log && lib.cubin_size &&
             lib.cubin && lib.destroy;
    if (!lib.ok) {
        FreeLibrary(h);
        lib.handle = nullptr;
    }
    return lib;
}

// port of cuda/nvrtc.rs: lib() (OnceLock<Option<Nvrtc>>)
const Lib& lib() {
    static Lib cached = [] {
        for (const std::string& c : candidates()) {
            Lib l = open_one(c);
            if (l.ok) return l;
        }
        return Lib{};
    }();
    return cached;
}

std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
        size_t j = i;
        while (j < s.size() && !(s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r')) j++;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

}  // namespace

// port of cuda/nvrtc.rs: names(cfg!(windows), STRATA_NVRTC, CUDA_PATH)
std::vector<std::string> candidates() {
    if (const char* forced = std::getenv("STRATA_NVRTC")) {
        if (*forced) return {forced};
    }
    // Every CUDA 12.x names its NVRTC `nvrtc64_120_0.dll`; 13.x `nvrtc64_130_0.dll`.
    static const char* DLLS[2] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll"};
    std::vector<std::string> v;
    for (const char* d : DLLS) v.push_back(d);
    // The toolkit installer sets CUDA_PATH; its DLL directory (`bin`, `bin\x64` since 13) is not always on PATH.
    if (const char* root = std::getenv("CUDA_PATH")) {
        for (const char* dir : {"\\bin", "\\bin\\x64"})
            for (const char* dll : DLLS) v.push_back(std::string(root) + dir + "\\" + dll);
    }
    return v;
}

bool present() { return lib().ok; }

std::optional<std::pair<int, int>> version() {
    const Lib& l = lib();
    if (!l.ok) return std::nullopt;
    int major = 0, minor = 0;
    if (l.version(&major, &minor) != 0) return std::nullopt;
    return std::make_pair(major, minor);
}

std::string missing_message() {
    std::string tried;
    for (const std::string& p : candidates()) {
        if (!tried.empty()) tried += ", ";
        tried += p;
    }
    return "the CUDA NVRTC library is not installed or not on the library path (tried: " + tried +
           "); install the CUDA toolkit or its NVRTC package (12.x for sm_70: 13.x dropped Volta support),"
           " or set STRATA_NVRTC to the library file";
}

std::string extra_opts() {
    const char* v = std::getenv("STRATA_NVRTC_OPTS");
    return v ? v : "";
}

// port of cuda/nvrtc.rs: cubin()
bool cubin(const std::string& src, const std::string& arch, std::vector<uint8_t>& out, std::string& err) {
    const Lib& l = lib();
    if (!l.ok) {
        err = missing_message();
        return false;
    }
    std::vector<std::string> opts_src = {"--gpu-architecture=" + arch, "-std=c++17"};
    for (std::string& o : split_ws(extra_opts())) opts_src.push_back(o);
    std::vector<const char*> opts;
    opts.reserve(opts_src.size());
    for (const std::string& o : opts_src) opts.push_back(o.c_str());

    Prog prog = nullptr;
    auto ok = [](int r, const char* what, std::string& e) {
        if (r == 0) return true;
        e = std::string(what) + " failed: NVRTC error " + std::to_string(r);
        return false;
    };
    if (!ok(l.create(&prog, src.c_str(), nullptr, 0, nullptr, nullptr), "nvrtcCreateProgram", err))
        return false;

    bool good = true;
    const int rc = l.compile(prog, (int)opts.size(), opts.data());
    if (rc != 0) {
        size_t len = 0;
        std::string log(1, '\0');
        if (l.log_size(prog, &len) == 0 && len > 0) {
            log.assign(len, '\0');
            l.log(prog, log.data());
            if (const char* nul = (const char*)std::memchr(log.data(), 0, log.size()))
                log.resize((size_t)(nul - log.data()));
        } else {
            log.clear();
        }
        err = "NVRTC compile of kernels.cu failed: NVRTC error " + std::to_string(rc) + "\n" + log;
        good = false;
    }
    if (good) {
        size_t len = 0;
        if (!ok(l.cubin_size(prog, &len), "nvrtcGetCUBINSize", err)) {
            good = false;
        } else {
            out.assign(len, 0);
            if (!ok(l.cubin(prog, (char*)out.data()), "nvrtcGetCUBIN", err)) {
                good = false;
            } else {
                out.resize(len);
            }
        }
    }
    l.destroy(&prog);
    return good;
}

std::string sm_arch(int major, int minor) {
    return "sm_" + std::to_string(major) + std::to_string(minor);
}

}  // namespace strata::nvrtc
