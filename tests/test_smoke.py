"""W0-B acceptance: the strata-cpp DLL loads, reports its version, and runs a CUDA kernel end to end.

    python tests\\test_smoke.py

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).

Needs an NVIDIA GPU and CUDA_PATH (the CUDA toolkit root, which its installer sets). The DLL compiles a
3-line kernel with NVRTC for this device's compute capability, loads the CUBIN and launches it through
the CUDA driver API on the primary context. The test checks all n results. Exit code 0 on success, 1
on any failure.
"""

from __future__ import annotations

import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DLL = os.path.join(ROOT, os.environ.get("SC_BUILD_DIR", "build"), "strata_cpp.dll")
CUDA_PATH = os.environ.get("CUDA_PATH", "")
N = 1000


def main() -> int:
    if CUDA_PATH:
        cuda_bin = os.path.join(CUDA_PATH, "bin")
        if os.path.isdir(cuda_bin):
            # NVRTC (nvrtc64_*.dll) is loaded by path from the CUDA bin directory; the directory must be
            # on the DLL search path for its own dependencies.
            os.add_dll_directory(cuda_bin)
    if not os.path.isfile(DLL):
        print(f"FAIL: missing {DLL}; run build.bat first")
        return 1

    lib = ctypes.CDLL(DLL)
    lib.sc_version.restype = ctypes.c_char_p
    lib.sc_last_error.restype = ctypes.c_char_p
    lib.sc_selftest_kernel.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.sc_selftest_kernel.restype = ctypes.c_int

    version = lib.sc_version().decode()
    print(f"sc_version: {version}")

    out = (ctypes.c_float * N)()
    rc = lib.sc_selftest_kernel(out, N)
    if rc != 0:
        print(f"FAIL: sc_selftest_kernel returned {rc}")
        print(f"      {lib.sc_last_error().decode()}")
        return 1

    bad = [(i, out[i], 2.0 * i) for i in range(N) if out[i] != 2.0 * i]
    if bad:
        print(f"FAIL: {len(bad)} of {N} values differ; first: {bad[:5]}")
        return 1

    print(f"sc_selftest_kernel: n={N} out[0]={out[0]:g} out[{N - 1}]={out[N - 1]:g} all out[i]==2*i")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
