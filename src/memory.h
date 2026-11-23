// port of memory.rs (host RAM cap: total physical RAM - margin, or an explicit ceiling).
//
// The cap bounds memory the process owns outright (baked tensors, LoRA factors, parked conversation
// states). The mmapped GGUF is page cache the OS can evict, so a model larger than RAM still loads.
#pragma once

#include <cstdint>
#include <optional>

namespace strata {

// port of memory.rs: total_ram - total physical RAM in bytes, nullopt when undetectable.
std::optional<uint64_t> total_ram();

// port of memory.rs: ram_cap - `max_ram` if nonzero, else `total - margin`. nullopt means unlimited.
std::optional<uint64_t> ram_cap(uint64_t margin, uint64_t max_ram);

}  // namespace strata
