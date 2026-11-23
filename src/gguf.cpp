// port of gguf.rs. Function and struct names match the Rust ones.
#include "gguf.h"

#include "quant.h"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace strata {

// ---------------------------------------------------------------- Value

std::optional<uint64_t> Value::as_u64() const {
    switch (tag) {
        case Tag::U8:
        case Tag::U16:
        case Tag::U32:
        case Tag::U64: return u;
        case Tag::I8:
        case Tag::I16:
        case Tag::I32:
        case Tag::I64: return i >= 0 ? std::optional<uint64_t>((uint64_t)i) : std::nullopt;
        case Tag::Bool: return (uint64_t)(b ? 1 : 0);
        default: return std::nullopt;
    }
}

std::optional<float> Value::as_f32() const {
    if (tag == Tag::F32) {
        const uint32_t bits = (uint32_t)u;
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }
    if (tag == Tag::F64) {
        const uint64_t bits = u;
        double d;
        std::memcpy(&d, &bits, sizeof d);
        return (float)d;
    }
    auto v = as_u64();
    if (!v) return std::nullopt;
    return (float)*v;
}

const std::string* Value::as_str() const { return tag == Tag::Str ? &s : nullptr; }
const std::vector<Value>* Value::as_arr() const { return tag == Tag::Arr ? &arr : nullptr; }

// ---------------------------------------------------------------- Mapping

Mapping::~Mapping() { close(); }

void Mapping::close() {
    if (base_) UnmapViewOfFile(base_);
    if (mapping_) CloseHandle((HANDLE)mapping_);
    if (file_) CloseHandle((HANDLE)file_);
    base_ = nullptr;
    mapping_ = nullptr;
    file_ = nullptr;
    size_ = 0;
}

// Same handle and view teardown as close(), without forgetting the size: the RAM copy replaces the view.
void Mapping::release() {
    if (base_) UnmapViewOfFile(base_);
    if (mapping_) CloseHandle((HANDLE)mapping_);
    if (file_) CloseHandle((HANDLE)file_);
    base_ = nullptr;
    mapping_ = nullptr;
    file_ = nullptr;
}

bool Mapping::open(const std::string& path, std::string& err) {
    close();
    path_ = path;
    std::wstring wide(path.begin(), path.end());  // ASCII paths only; the project's model paths are
    HANDLE f = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = path + ": cannot open (" + std::to_string(GetLastError()) + ")";
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) {
        err = path + ": cannot read size";
        CloseHandle(f);
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) {
        err = path + ": cannot map";
        CloseHandle(f);
        return false;
    }
    void* base = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        err = path + ": cannot map view";
        CloseHandle(m);
        CloseHandle(f);
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*)base;
    size_ = (size_t)sz.QuadPart;
    return true;
}

// ---------------------------------------------------------------- cursor

namespace {

// port of gguf.rs: struct Cur (bounds-checked cursor over a shard)
struct Cur {
    const uint8_t* b;
    size_t len;
    size_t p = 0;
    std::string err;

    bool take(size_t n, const uint8_t*& out) {
        if (p > len || n > len - p) {
            err = "GGUF truncated";
            return false;
        }
        out = b + p;
        p += n;
        return true;
    }
    bool u32v(uint32_t& v) {
        const uint8_t* s;
        if (!take(4, s)) return false;
        v = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24);
        return true;
    }
    bool u64v(uint64_t& v) {
        const uint8_t* s;
        if (!take(8, s)) return false;
        v = 0;
        for (int i = 7; i >= 0; i--) v = (v << 8) | s[i];
        return true;
    }
    // port of gguf.rs: Cur::str
    bool strv(std::string& s) {
        uint64_t n;
        if (!u64v(n)) return false;
        if (n > len - p) {
            err = "GGUF string length out of range";
            return false;
        }
        const uint8_t* d;
        if (!take((size_t)n, d)) return false;
        s.assign((const char*)d, (size_t)n);
        return true;
    }
    // port of gguf.rs: Cur::val
    bool val(uint32_t t, Value& out);
};

bool Cur::val(uint32_t t, Value& out) {
    switch (t) {
        case 0: {
            const uint8_t* s;
            if (!take(1, s)) return false;
            out.tag = Value::Tag::U8;
            out.u = s[0];
            return true;
        }
        case 1: {
            const uint8_t* s;
            if (!take(1, s)) return false;
            out.tag = Value::Tag::I8;
            out.i = (int8_t)s[0];
            return true;
        }
        case 2: {
            const uint8_t* s;
            if (!take(2, s)) return false;
            out.tag = Value::Tag::U16;
            out.u = (uint16_t)(s[0] | (s[1] << 8));
            return true;
        }
        case 3: {
            const uint8_t* s;
            if (!take(2, s)) return false;
            out.tag = Value::Tag::I16;
            out.i = (int16_t)(s[0] | (s[1] << 8));
            return true;
        }
        case 4: {
            uint32_t v;
            if (!u32v(v)) return false;
            out.tag = Value::Tag::U32;
            out.u = v;
            return true;
        }
        case 5: {
            uint32_t v;
            if (!u32v(v)) return false;
            out.tag = Value::Tag::I32;
            out.i = (int32_t)v;
            return true;
        }
        case 6: {  // f32: keep the bits (as f64 when read through as_f32)
            uint32_t bits;
            if (!u32v(bits)) return false;
            out.tag = Value::Tag::F32;
            out.u = (uint64_t)bits;
            return true;
        }
        case 7: {
            const uint8_t* s;
            if (!take(1, s)) return false;
            out.tag = Value::Tag::Bool;
            out.b = s[0] != 0;
            return true;
        }
        case 8: {
            out.tag = Value::Tag::Str;
            return strv(out.s);
        }
        case 9: {
            uint32_t et;
            uint64_t n;
            if (!u32v(et) || !u64v(n)) return false;
            if (n > len - p) {
                err = "GGUF array length exceeds remaining data";
                return false;
            }
            out.tag = Value::Tag::Arr;
            out.arr.clear();
            out.arr.reserve((size_t)(n < 1024 ? n : 1024));
            for (uint64_t i = 0; i < n; i++) {
                Value v;
                if (!val(et, v)) return false;
                out.arr.push_back(std::move(v));
            }
            return true;
        }
        case 10: {
            uint64_t v;
            if (!u64v(v)) return false;
            out.tag = Value::Tag::U64;
            out.u = v;
            return true;
        }
        case 11: {
            uint64_t v;
            if (!u64v(v)) return false;
            out.tag = Value::Tag::I64;
            out.i = (int64_t)v;
            return true;
        }
        case 12: {
            uint64_t bits;
            if (!u64v(bits)) return false;
            out.tag = Value::Tag::F64;
            out.u = bits;
            return true;
        }
        default: err = "unknown GGUF value type " + std::to_string(t); return false;
    }
}

std::string dir_of(const std::string& p) {
    std::filesystem::path pp(p);
    return pp.parent_path().string();
}

std::string base_of(const std::string& p) { return std::filesystem::path(p).filename().string(); }

std::string join_path(const std::string& dir, const std::string& name) {
    std::filesystem::path d(dir);
    d /= name;
    return d.string();
}

std::string pad5(size_t v) {
    std::string s = std::to_string(v);
    while (s.size() < 5) s.insert(s.begin(), '0');
    return s;
}

// ---------------------------------------------------------------- RAM copy (StrataFreet addition)

constexpr size_t kRamChunk = size_t(64) << 20;    // one ReadFile of the copy
constexpr size_t kRamHeadroom = size_t(6) << 30;  // free memory the copy must leave for the rest of the process
constexpr size_t kLockChunk = size_t(1) << 30;    // one VirtualLock call
constexpr size_t kLockMargin = size_t(1) << 30;   // working set above the locked bytes
constexpr size_t kReportEvery = size_t(1) << 30;  // progress line every GB read

std::string gb_text(double bytes) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f", bytes / 1073741824.0);
    return b;
}

std::string win_code(DWORD e) { return std::to_string(e); }

std::wstring to_wide(const std::string& s) { return std::wstring(s.begin(), s.end()); }

// Enables one privilege on this process token. False when the account does not hold it (AdjustTokenPrivileges
// reports success either way, so ERROR_NOT_ALL_ASSIGNED is checked).
bool enable_privilege(const wchar_t* name, std::string& err) {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        err = "OpenProcessToken failed (" + win_code(GetLastError()) + ")";
        return false;
    }
    LUID luid{};
    if (!LookupPrivilegeValueW(nullptr, name, &luid)) {
        err = "LookupPrivilegeValue failed (" + win_code(GetLastError()) + ")";
        CloseHandle(tok);
        return false;
    }
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    const BOOL ok = AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
    const DWORD e = GetLastError();
    CloseHandle(tok);
    if (!ok || e == ERROR_NOT_ALL_ASSIGNED) {
        err = "the account does not hold " + std::string("SeIncreaseWorkingSetPrivilege");
        return false;
    }
    return true;
}

void unlock_shards(std::vector<RamShard>& rs) {
    for (RamShard& r : rs) {
        if (r.locked) VirtualUnlock(r.base, r.locked);
        r.locked = 0;
    }
}

// Makes the copies resident: raises the minimum working set above the data (QUOTA_LIMITS_HARDWS_MIN_ENABLE needs
// SeIncreaseWorkingSetPrivilege), then VirtualLock in 1 GB calls. All or nothing: a failed call unlocks what it got.
bool lock_shards(std::vector<RamShard>& rs, size_t total, std::string& err) {
    if (!enable_privilege(L"SeIncreaseWorkingSetPrivilege", err)) return false;
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &min_ws, &max_ws, &flags)) {
        err = "GetProcessWorkingSetSizeEx failed (" + win_code(GetLastError()) + ")";
        return false;
    }
    const SIZE_T want = (SIZE_T)total + kLockMargin;
    const SIZE_T new_min = std::max(min_ws, want);
    const SIZE_T new_max = std::max(max_ws, new_min + kLockMargin);
    if (!SetProcessWorkingSetSizeEx(GetCurrentProcess(), new_min, new_max, QUOTA_LIMITS_HARDWS_MIN_ENABLE)) {
        err = "SetProcessWorkingSetSizeEx(min " + gb_text((double)new_min) + " GB) failed (" +
              win_code(GetLastError()) + ")";
        return false;
    }
    size_t done = 0;
    for (RamShard& r : rs) {
        for (size_t off = 0; off < r.len; off += kLockChunk) {
            const size_t n = std::min(kLockChunk, r.len - off);
            if (!VirtualLock(r.base + off, n)) {
                err = "VirtualLock failed at " + gb_text((double)(done + off)) + " GB of " + gb_text((double)total) +
                      " GB (" + win_code(GetLastError()) + ")";
                unlock_shards(rs);
                return false;
            }
            r.locked = off + n;
        }
        done += r.len;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------- Gguf

Gguf::~Gguf() { free_ram(); }

// Frees the private copies (unlocking first). Called by the destructor, and by load_to_ram on failure.
void Gguf::free_ram() {
    for (RamShard& r : ram_) {
        if (r.locked) VirtualUnlock(r.base, r.locked);
        if (r.base) VirtualFree(r.base, 0, MEM_RELEASE);
    }
    ram_.clear();
}

// Copies the tensor data of every shard (from its data_offset to the end of the file) into VirtualAlloc'd memory,
// reading with ReadFile in 64 MB chunks from a sequential-scan handle, then releases the file views. Refuses first
// when the free physical memory does not cover the data plus kRamHeadroom (no thrashing). Progress goes to stderr.
bool Gguf::load_to_ram(RamMode mode, RamLoad& out, std::string& err) {
    out = RamLoad();
    if (mode == RamMode::Off) return true;
    if (!ram_.empty()) {
        err = "tensor data is already in RAM";
        return false;
    }
    if (files_.empty()) {
        err = "load_to_ram: no GGUF shard is open";
        return false;
    }
    std::vector<size_t> lens(files_.size());
    size_t total = 0;
    for (size_t i = 0; i < files_.size(); i++) {
        const uint64_t off = files_[i].second;
        const uint64_t sz = files_[i].first->size();
        // A metadata-only shard (no tensors) can have its aligned data offset land past the file end:
        // the data section is empty, not truncated. A shard with tensors cannot get here — open_one
        // refused files whose tensors lie past the end at open time.
        lens[i] = off >= sz ? 0 : (size_t)(sz - off);
        total += lens[i];
    }

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) {
        err = "RAM guard: GlobalMemoryStatusEx failed (" + win_code(GetLastError()) + ")";
        return false;
    }
    const uint64_t need = (uint64_t)total + kRamHeadroom;
    if (ms.ullAvailPhys < need) {
        err = "RAM guard: the model's tensor data is " + gb_text((double)total) + " GB and needs " +
              gb_text((double)need) + " GB free (6 GB headroom), but " + gb_text((double)ms.ullAvailPhys) +
              " GB is free; not loading into RAM (ram off, or free memory first)";
        return false;
    }
    std::fprintf(stderr, "[strata-cpp] ram %s: copying %s GB of tensor data into private memory (%s GB free)\n",
                 mode == RamMode::Lock ? "lock" : "copy", gb_text((double)total).c_str(),
                 gb_text((double)ms.ullAvailPhys).c_str());

    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    ram_.assign(files_.size(), RamShard{});
    size_t read_total = 0, next_report = kReportEvery;
    for (size_t i = 0; i < files_.size(); i++) {
        void* p = VirtualAlloc(nullptr, lens[i] ? lens[i] : 1, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!p) {
            err = "RAM copy: VirtualAlloc of " + gb_text((double)lens[i]) + " GB failed (" +
                  win_code(GetLastError()) + "); the system commit limit may be too low";
            free_ram();
            return false;
        }
        ram_[i].base = (uint8_t*)p;
        ram_[i].len = lens[i];
        const std::string& path = files_[i].first->path();
        HANDLE f = CreateFileW(to_wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                               OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (f == INVALID_HANDLE_VALUE) {
            err = path + ": cannot reopen for the RAM copy (" + win_code(GetLastError()) + ")";
            free_ram();
            return false;
        }
        LARGE_INTEGER pos;
        pos.QuadPart = (LONGLONG)files_[i].second;
        if (!SetFilePointerEx(f, pos, nullptr, FILE_BEGIN)) {
            err = path + ": seek to the tensor data failed (" + win_code(GetLastError()) + ")";
            CloseHandle(f);
            free_ram();
            return false;
        }
        size_t done = 0;
        while (done < lens[i]) {
            const DWORD want = (DWORD)std::min(kRamChunk, lens[i] - done);
            DWORD got = 0;
            if (!ReadFile(f, ram_[i].base + done, want, &got, nullptr) || got == 0) {
                err = path + ": read failed at " + gb_text((double)(done)) + " GB of shard " +
                      std::to_string(i + 1) + " (" + win_code(GetLastError()) + ")";
                CloseHandle(f);
                free_ram();
                return false;
            }
            done += got;
            read_total += got;
            if (read_total >= next_report) {
                const double s = elapsed();
                std::fprintf(stderr, "[strata-cpp] ram copy: %s / %s GB read, %.0f MB/s\n",
                             gb_text((double)read_total).c_str(), gb_text((double)total).c_str(),
                             s > 0 ? (double)read_total / 1e6 / s : 0.0);
                next_report += kReportEvery;
            }
        }
        CloseHandle(f);
    }
    const double copy_s = elapsed();
    std::fprintf(stderr, "[strata-cpp] ram copy: %s / %s GB read in %.1f s (%.0f MB/s)\n",
                 gb_text((double)read_total).c_str(), gb_text((double)total).c_str(), copy_s,
                 copy_s > 0 ? (double)read_total / 1e6 / copy_s : 0.0);
    // The views are no longer read (raw_bytes answers from ram_), so they go now.
    for (auto& m : files_) m.first->release();
    out.bytes = total;

    if (mode == RamMode::Lock) {
        std::string lerr;
        if (lock_shards(ram_, total, lerr)) {
            out.locked_bytes = total;
            std::fprintf(stderr, "[strata-cpp] ram lock: %s GB locked resident\n", gb_text((double)total).c_str());
        } else {
            const std::string w = "ram lock failed: " + lerr + "; continuing as ram copy (its pages can be paged out)";
            out.warnings.push_back(w);
            std::fprintf(stderr, "[strata-cpp] WARNING %s\n", w.c_str());
        }
    }
    out.seconds = elapsed();
    return true;
}

// port of gguf.rs: Gguf::open
bool Gguf::open(const std::string& path, std::string& err) {
    if (!open_one(path, 0, err)) return false;
    size_t n = 1;
    if (const Value* v = get("split.count")) {
        if (auto c = v->as_u64()) n = (size_t)*c;
    }
    if (n <= 1) return true;

    // port of gguf.rs: strip "-00001-of-0000N.gguf"; every shard opens from the same directory
    const std::string name = base_of(path);
    const std::string tag = "-00001-of-" + pad5(n) + ".gguf";
    if (name.size() < tag.size() || name.compare(name.size() - tag.size(), tag.size(), tag) != 0) {
        err = path + ": split model; open its first shard (*" + tag + ")";
        return false;
    }
    const std::string stem = name.substr(0, name.size() - tag.size());
    const std::string dir = dir_of(path);
    for (size_t i = 1; i < n; i++) {
        const std::string shard = join_path(dir, stem + "-" + pad5(i + 1) + "-of-" + pad5(n) + ".gguf");
        Gguf part;
        if (!part.open_one(shard, i, err)) return false;
        for (auto& kv : part.tensors_) {
            if (!tensors_.emplace(kv.first, kv.second).second) {
                err = "tensor " + kv.first + " in two shards";
                return false;
            }
        }
        for (auto& m : part.files_) files_.push_back(std::move(m));
    }
    return true;
}

// port of gguf.rs: Gguf::open_one
bool Gguf::open_one(const std::string& path, size_t file, std::string& err) {
    auto map = std::make_unique<Mapping>();
    if (!map->open(path, err)) return false;

    Cur c{map->data(), map->size()};
    const uint8_t* magic;
    if (!c.take(4, magic)) {
        err = "GGUF truncated";
        return false;
    }
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        err = path + ": not a GGUF file";
        return false;
    }
    uint32_t ver;
    if (!c.u32v(ver)) {
        err = "GGUF truncated";
        return false;
    }
    if (ver < 2 || ver > 3) {
        err = "unsupported GGUF version " + std::to_string(ver);
        return false;
    }
    uint64_t nt, nkv;
    if (!c.u64v(nt) || !c.u64v(nkv)) {
        err = "GGUF truncated";
        return false;
    }
    std::map<std::string, Value> meta;
    for (uint64_t i = 0; i < nkv; i++) {
        std::string k;
        uint32_t t;
        Value v;
        if (!c.strv(k) || !c.u32v(t) || !c.val(t, v)) {
            err = c.err.empty() ? std::string("GGUF truncated") : c.err;
            return false;
        }
        meta.emplace(std::move(k), std::move(v));
    }
    std::map<std::string, TensorInfo> tensors;
    for (uint64_t i = 0; i < nt; i++) {
        TensorInfo ti;
        if (!c.strv(ti.name)) {
            err = "GGUF truncated";
            return false;
        }
        uint32_t nd;
        if (!c.u32v(nd)) {
            err = "GGUF truncated";
            return false;
        }
        ti.dims.resize(nd);
        for (uint32_t d = 0; d < nd; d++)
            if (!c.u64v(ti.dims[d])) {
                err = "GGUF truncated";
                return false;
            }
        if (!c.u32v(ti.ty) || !c.u64v(ti.offset)) {
            err = "GGUF truncated";
            return false;
        }
        ti.file = file;
        tensors.emplace(ti.name, std::move(ti));
    }
    uint64_t align = 32;
    if (auto it = meta.find("general.alignment"); it != meta.end()) {
        if (auto a = it->second.as_u64()) align = *a;
    }
    if (align == 0 || (align & (align - 1)) != 0) {
        err = "invalid GGUF alignment";
        return false;
    }
    const uint64_t data_offset = ((uint64_t)c.p + align - 1) / align * align;

    // port of gguf.rs: a cut-off download keeps a valid header; catch it here, not as a slice overrun
    for (const auto& kv : tensors) {
        const TensorInfo& t = kv.second;
        auto ty = ty_from_u32(t.ty);
        if (!ty) continue;  // Weights::validate reports it
        uint64_t n = row_bytes(*ty, t.dims.empty() ? 0 : t.dims[0]);
        for (size_t d = 1; d < t.dims.size(); d++) {
            if (n > UINT64_MAX / (t.dims[d] ? t.dims[d] : 1)) {
                n = UINT64_MAX;
                break;
            }
            n *= t.dims[d];
        }
        const uint64_t room = (uint64_t)map->size() > data_offset ? (uint64_t)map->size() - data_offset : 0;
        if (n > room || t.offset > room - n) {
            err = path + ": GGUF truncated (tensor " + t.name +
                  " lies past the end of the file; re-download it)";
            return false;
        }
    }

    for (auto& kv : tensors) tensors_.emplace(kv.first, kv.second);
    for (auto& kv : meta) meta_.emplace(kv.first, std::move(kv.second));
    files_.emplace_back(std::move(map), data_offset);
    return true;
}

const Value* Gguf::get(const std::string& key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? nullptr : &it->second;
}

// port of gguf.rs: Gguf::bytes
const uint8_t* Gguf::bytes(const TensorInfo& t, size_t& len, std::string& err) const {
    auto ty = ty_from_u32(t.ty);
    if (!ty) {
        err = "unsupported quant type id " + std::to_string(t.ty);
        len = 0;
        return nullptr;
    }
    uint64_t n = row_bytes(*ty, t.dims.empty() ? 0 : t.dims[0]);
    for (size_t d = 1; d < t.dims.size(); d++) n *= t.dims[d];
    // Rust's return value is a &[u8]; the C++ port hands the length back separately.
    len = (size_t)n;
    return raw_bytes(t, (size_t)n, err);
}

const uint8_t* Gguf::raw_bytes(const TensorInfo& t, size_t byte_len, std::string& err) const {
    if (t.file >= files_.size()) {
        err = "tensor file index out of range";
        return nullptr;
    }
    if (!ram_.empty()) {  // tensor data copied into RAM (load_to_ram): the copy replaces the file view
        const RamShard& r = ram_[t.file];
        if (t.offset > r.len || byte_len > r.len - t.offset) {
            err = "GGUF tensor bytes out of range";
            return nullptr;
        }
        return r.base + t.offset;
    }
    const Mapping& m = *files_[t.file].first;
    const uint64_t off = files_[t.file].second + t.offset;
    if (off > (uint64_t)m.size() || byte_len > (uint64_t)m.size() - off) {
        err = "GGUF tensor bytes out of range";
        return nullptr;
    }
    return m.data() + off;
}

}  // namespace strata
