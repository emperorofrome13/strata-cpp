// port of gguf.rs (GGUF v2/v3 parser over a memory map).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strata {

// sc_load option "ram" (StrataFreet's strata_cpp_ram). Off keeps the file mapping (the original behaviour); Copy
// and Lock read every shard's tensor data into private memory at load and drop the file view. Lock also makes the
// copy resident (VirtualLock). Not in RustingStrata: the Rust port reads through its memory map only.
enum class RamMode : uint8_t { Off, Copy, Lock };

// What Gguf::load_to_ram did (sc_stats reports it).
struct RamLoad {
    size_t bytes = 0;         // tensor data copied into private memory
    size_t locked_bytes = 0;  // of which VirtualLock holds resident (0 unless the lock succeeded)
    double seconds = 0;       // the copy and the lock attempt
    std::vector<std::string> warnings;
};

// One shard's private copy of its tensor data. `base` holds the bytes from the shard's data_offset to its end, so
// a TensorInfo::offset indexes it directly.
struct RamShard {
    uint8_t* base = nullptr;
    size_t len = 0;
    size_t locked = 0;  // a prefix of the region that VirtualLock holds (all or nothing: see load_to_ram)
};

// port of gguf.rs: Value
struct Value {
    // port of gguf.rs: the Value enum's tag set (u8..Arr order preserved)
    enum class Tag : uint8_t { U8, I8, U16, I16, U32, I32, U64, I64, F32, F64, Bool, Str, Arr };

    Tag tag = Tag::U32;
    uint64_t u = 0;   // unsigned payloads, and the raw bits of F32/F64
    int64_t i = 0;    // signed payloads
    bool b = false;
    std::string s;            // Tag::Str
    std::vector<Value> arr;   // Tag::Arr

    // port of gguf.rs: Value::as_u64
    std::optional<uint64_t> as_u64() const;
    // port of gguf.rs: Value::as_f32
    std::optional<float> as_f32() const;
    // port of gguf.rs: Value::as_str
    const std::string* as_str() const;
    // port of gguf.rs: Value::as_arr
    const std::vector<Value>* as_arr() const;
};

// port of gguf.rs: TensorInfo
struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t ty = 0;
    uint64_t offset = 0;
    size_t file = 0;  // shard index
};

// One mapped shard. Windows equivalent of the (Mmap, data_offset) pair gguf.rs keeps in `files`.
class Mapping {
public:
    Mapping() = default;
    ~Mapping();
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping(Mapping&&) = delete;
    Mapping& operator=(Mapping&&) = delete;

    bool open(const std::string& path, std::string& err);
    const uint8_t* data() const { return base_; }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }
    // Unmaps the view and closes the handles, keeping size() and path(). Used once the tensor data is in RAM.
    void release();

private:
    void close();
    std::string path_;
    void* file_ = nullptr;    // HANDLE
    void* mapping_ = nullptr; // HANDLE
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
};

// port of gguf.rs: Gguf
class Gguf {
public:
    // port of gguf.rs: Gguf::open - for a split model pass the -00001-of-0000N shard.
    Gguf() = default;
    ~Gguf();
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;

    bool open(const std::string& path, std::string& err);

    // Copies every shard's tensor data into private committed memory (RamMode::Copy), or also makes it resident
    // (RamMode::Lock). Refuses when the free physical memory is below the data plus 6 GB. A failed lock is a
    // warning in `out` and the copy stays. Must follow open(); the file mapping is released afterwards. Not
    // ported from RustingStrata (see RamMode).
    bool load_to_ram(RamMode mode, RamLoad& out, std::string& err);
    // The private copies, one per shard in file order (empty until load_to_ram copies).
    const std::vector<RamShard>& ram() const { return ram_; }

    // port of gguf.rs: Gguf::get
    const Value* get(const std::string& key) const;

    // port of gguf.rs: Gguf::bytes (unsupported types are rejected by the caller's validate step)
    const uint8_t* bytes(const TensorInfo& t, size_t& len, std::string& err) const;

    // port of gguf.rs: Gguf::raw_bytes
    const uint8_t* raw_bytes(const TensorInfo& t, size_t byte_len, std::string& err) const;

    const std::map<std::string, Value>& meta() const { return meta_; }
    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    uint64_t data_offset(size_t file) const { return files_[file].second; }

private:
    bool open_one(const std::string& path, size_t file, std::string& err);
    void free_ram();

    std::map<std::string, Value> meta_;
    std::map<std::string, TensorInfo> tensors_;
    std::vector<std::unique_ptr<Mapping>> maps_;
    std::vector<std::pair<std::unique_ptr<Mapping>, uint64_t>> files_;
    std::vector<RamShard> ram_;
};

}  // namespace strata
