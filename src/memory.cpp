// port of memory.rs. Windows only, which is the branch the reference binary runs in.
#include "memory.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace strata {

// port of memory.rs: total_ram - the Windows branch reads GlobalMemoryStatusEx's ullTotalPhys.
std::optional<uint64_t> total_ram() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    if (GlobalMemoryStatusEx(&m) && m.ullTotalPhys > 0) return (uint64_t)m.ullTotalPhys;
    return std::nullopt;
}

// port of memory.rs: ram_cap
std::optional<uint64_t> ram_cap(uint64_t margin, uint64_t max_ram) {
    if (max_ram > 0) return max_ram;
    auto total = total_ram();
    if (!total) return std::nullopt;
    return *total > margin ? *total - margin : 0;
}

}  // namespace strata
