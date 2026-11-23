"""Wave 2-B acceptance: the C++ dequantizers must be exact against the reference.

    python tests/test_quant.py

For every GGML type Strata supports, schematic blocks are fed to the C++ `dequant_row`
(build/strata_cpp_dev.dll) and to `stratafreet.quant.dequant`, and the two are compared element by
element. The bytes are random on purpose: any disagreement in bit unpacking shows up even when the
scale is nonsense, and NaN/Inf compare equal when both sides produce the same class.

Needs STRATAFREET_ROOT (the Python reference oracle); the real MXFP4 rows also need SC_GPT_OSS_GGUF.
CPU only: no CUDA context is created. Without the oracle the test prints SKIP and exits 0.
"""

from __future__ import annotations

import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

DLL = os.path.join(ROOT, os.environ.get("SC_BUILD_DIR", "build"), "strata_cpp_dev.dll")

# Imported in main() after the SKIP gate.
np = None
torch = None
dequant = None

# Every id GgmlType::from_u32 accepts (quant/mod.rs)
# Every id GgmlType::from_u32 accepts, plus 39 (MXFP4), a StrataFreet addition (quant.h, MODIFICATIONS.md)
TYPES = [0, 1, 30, 42, 34, 35, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39]
NAMES = {0: "F32", 1: "F16", 30: "BF16", 42: "Q2_0", 34: "TQ1_0", 35: "TQ2_0", 2: "Q4_0", 3: "Q4_1",
         6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
         16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
         22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 39: "MXFP4"}

# Real MXFP4 expert tensors of gpt-oss-20b (read through GGUFModelFile; the file is mapped, not loaded)
GPT_OSS = sc_env.get("SC_GPT_OSS_GGUF")

N = 512


def load():
    if not os.path.isfile(DLL):
        raise SystemExit(f"missing {DLL}; run build.bat first")
    lib = ctypes.CDLL(DLL)
    lib.sci_last_error.restype = ctypes.c_char_p
    lib.sci_row_bytes.argtypes = [ctypes.c_uint, ctypes.c_longlong]
    lib.sci_row_bytes.restype = ctypes.c_longlong
    lib.sci_dequant.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_longlong,
                                ctypes.POINTER(ctypes.c_float), ctypes.c_longlong]
    lib.sci_dequant.restype = ctypes.c_int
    return lib


def main() -> int:
    global np, torch, dequant
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    try:
        import numpy
        import torch as torch_mod
        from stratafreet.quant import dequant as dequant_mod
    except Exception as e:  # missing torch, a broken checkout, ...
        return sc_env.skip(f"cannot import the StrataFreet oracle: {e}")
    np, torch, dequant = numpy, torch_mod, dequant_mod

    lib = load()
    rng = np.random.default_rng(20261120)
    failures = []
    skipped = []
    exact = 0

    for ty in TYPES:
        nbytes = lib.sci_row_bytes(ty, N)
        if nbytes <= 0:
            failures.append(f"{NAMES[ty]}: sci_row_bytes returned {nbytes}")
            continue
        raw = rng.integers(0, 256, size=nbytes, dtype=np.uint8)

        out = (ctypes.c_float * N)()
        rc = lib.sci_dequant(ty, raw.tobytes(), nbytes, out, N)
        if rc != 0:
            failures.append(f"{NAMES[ty]}: sci_dequant failed: {lib.sci_last_error().decode()}")
            continue
        got = np.frombuffer(out, dtype=np.float32).copy()

        try:
            ref = dequant(torch.from_numpy(raw.copy()), ty, (N,), torch.float32).numpy()
        except NotImplementedError as e:
            skipped.append(f"{NAMES[ty]}: no Python reference ({e})")
            continue

        if not np.array_equal(got, ref, equal_nan=True):
            diff = np.flatnonzero(~((got == ref) | (np.isnan(got) & np.isnan(ref))))
            worst = [(int(i), float(got[i]), float(ref[i])) for i in diff[:5]]
            failures.append(f"{NAMES[ty]}: {len(diff)} of {N} values differ, first: {worst}")
            continue
        print(f"{NAMES[ty]:8s} exact ({nbytes} bytes -> {N} floats)")
        exact += 1

    # Real rows: whole experts of three gpt-oss MXFP4 tensors, one dequant call per expert (row-major blocks are
    # contiguous, so the C++ side sees the same bytes as torch does).
    if os.path.isfile(GPT_OSS):
        from stratafreet.gguf_io import GGUFModelFile
        m = GGUFModelFile(GPT_OSS)
        for name, e in [("blk.0.ffn_gate_exps.weight", 0), ("blk.0.ffn_down_exps.weight", 31),
                        ("blk.23.ffn_up_exps.weight", 7)]:
            t = m[name]
            if t.qtype != 39:
                failures.append(f"{name}: qtype {t.qtype}, expected 39")
                continue
            n_exp, rows, cols = (int(v) for v in t.shape)
            rb = cols // 32 * 17
            span = t.data[e * rows * rb:(e + 1) * rows * rb].copy()
            out = (ctypes.c_float * (rows * cols))()
            rc = lib.sci_dequant(39, span.tobytes(), span.nbytes, out, rows * cols)
            if rc != 0:
                failures.append(f"{name} expert {e}: sci_dequant failed: {lib.sci_last_error().decode()}")
                continue
            got = np.frombuffer(out, dtype=np.float32).copy()
            ref = dequant(torch.from_numpy(span), 39, (rows, cols), torch.float32).reshape(-1).numpy()
            if not np.array_equal(got, ref):
                diff = np.flatnonzero(got != ref)
                failures.append(f"{name} expert {e}: {len(diff)} of {got.size} values differ, first {diff[:3].tolist()}")
                continue
            print(f"MXFP4 {name} expert {e} exact ({rows}x{cols}, {span.nbytes} bytes)")
    elif not GPT_OSS:
        skipped.append("set SC_GPT_OSS_GGUF=<path to gpt-oss-20b MXFP4 gguf> for the real MXFP4 rows")
    else:
        skipped.append(f"MXFP4 real rows: {GPT_OSS} not found")

    for s in skipped:
        print(f"SKIP {s}")
    if failures:
        print()
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print(f"\nPASS: {exact} types bit-exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
