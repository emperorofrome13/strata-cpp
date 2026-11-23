// port of cuda/nvrtc.rs (NVRTC, loaded by path and not linked).
//
// The Windows DLL name carries the CUDA version (`nvrtc64_120_0.dll`), and Windows does not search the
// directory of a DLL opened by full path for that DLL's own dependencies, so nvrtc-builtins is loaded
// first. Both details are Strata's; see the Rust file's comments.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::nvrtc {

// port of cuda/nvrtc.rs: candidates()
std::vector<std::string> candidates();

// port of cuda/nvrtc.rs: present()
bool present();

// port of cuda/nvrtc.rs: version() - device free
std::optional<std::pair<int, int>> version();

// port of cuda/nvrtc.rs: missing()
std::string missing_message();

// port of cuda/nvrtc.rs: extra_opts() (environment STRATA_NVRTC_OPTS)
std::string extra_opts();

// port of cuda/nvrtc.rs: cubin(). Compiles `src` for `arch` (e.g. "sm_120").
bool cubin(const std::string& src, const std::string& arch, std::vector<uint8_t>& out, std::string& err);

// The `sm_XY` name of a device's compute capability, as cuda/mod.rs derives it.
std::string sm_arch(int major, int minor);

}  // namespace strata::nvrtc
