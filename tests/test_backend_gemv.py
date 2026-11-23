"""W2-C part 1 acceptance: the C++ CUDA gemv / gemvn / moe_gemv launches against StrataFreet's kernels.

    python tests/test_backend_gemv.py

The C++ side is build/strata_cpp_dev.dll (scd_gemv, scd_gemvn, scd_moe_gemv): it uploads the same GGUF bytes
and x, launches through CudaBackend (NVRTC module built from kernels/kernels.cu with Strata's macros and
tables), and copies the result back. The reference is stratafreet.strata_kernels (sk.gemv, sk.moe_gemv), which
compiles StrataFreet's own, older copy of kernels.cu (53 hunks differ, see kernels/DIFF-vs-stratafreet.md) with
its own rows-per-warp choice and -use_fast_math. So the pass rule is max relative error <= 1e-3 (the error
relative to the peak of the reference output), and bit-identity is reported, not required.

Inputs:
  - ggml type 14 (Q6_K): real rows from the 14B MoE in the LM Studio library: a dense weight
    (blk.0.attn_gate, 4096 x 2048) and the layer-0 expert tensors at their model shapes (gate 512 x 2048, down
    2048 x 512) with routed ids that include a skipped (-1) entry.
  - the types gguf can encode (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and the others the installed gguf package
    quantizes): random float weights quantized by gguf.quants.quantize. A column count that is not a multiple
    of 256 (544) exercises the `_p` kernels.
  - F32 and BF16 (no StrataFreet kernel): compared with a numpy float64 oracle.
  - every other type is listed as skipped: no encoder here and not in the 14B file.

Needs STRATAFREET_ROOT, SC_MODEL_14B, and an NVIDIA GPU. SC_GPT_OSS_GGUF adds the real MXFP4 rows.
Without the oracle or the model the test prints SKIP and exits 0 (before any CUDA context exists).
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
GPT_OSS = sc_env.get("SC_GPT_OSS_GGUF")
GGUF_14B = sc_env.get("SC_MODEL_14B")
TOL = 1e-3

# Imported in main() after the SKIP gate.
np = None
torch = None
sk = None
GGUFModelFile = None
GGMLQuantizationType = None
gquants = None

# ggml id -> (name, StrataFreet kernel suffix in sk.TYPES, gguf enum name)
GGML = {
    0: "F32", 30: "BF16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 42: "Q2_0", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}
# the quant types StrataFreet can compare against (sk.TYPES) and that the C++ families cover
COMPARE_TYPES = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42]


def load_dll():
    if not os.path.isfile(DLL):
        raise SystemExit(f"missing {DLL}; run build.bat first")
    lib = ctypes.CDLL(DLL)
    vp = ctypes.c_void_p
    lib.sci_last_error.restype = ctypes.c_char_p
    lib.scd_device.restype = ctypes.c_char_p
    lib.scd_gemv.argtypes = [ctypes.c_int, vp, ctypes.c_longlong, vp, ctypes.c_int, ctypes.c_int, vp]
    lib.scd_gemv.restype = ctypes.c_int
    lib.scd_gemvn.argtypes = [ctypes.c_int, vp, ctypes.c_longlong, vp, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, vp, ctypes.c_int]
    lib.scd_gemvn.restype = ctypes.c_int
    lib.scd_moe_gemv.argtypes = [ctypes.c_int, vp, ctypes.c_longlong, ctypes.c_longlong, vp, ctypes.c_int, vp,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, vp]
    lib.scd_moe_gemv.restype = ctypes.c_int
    return lib


def cpp_error(lib) -> str:
    return (lib.sci_last_error() or b"").decode(errors="replace")


def cpp_gemv(lib, qtype, w, x, rows, cols):
    w = np.ascontiguousarray(w, dtype=np.uint8).reshape(-1)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    y = np.zeros(rows, dtype=np.float32)
    rc = lib.scd_gemv(qtype, w.ctypes.data, w.nbytes, x.ctypes.data, rows, cols, y.ctypes.data)
    if rc != 0:
        raise RuntimeError(cpp_error(lib))
    return y


def cpp_gemvn(lib, qtype, w, x, x_stride, rows, cols, nt, y_stride):
    w = np.ascontiguousarray(w, dtype=np.uint8).reshape(-1)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    y = np.zeros((nt - 1) * y_stride + rows, dtype=np.float32)
    rc = lib.scd_gemvn(qtype, w.ctypes.data, w.nbytes, x.ctypes.data, x_stride, rows, cols, nt, y.ctypes.data, y_stride)
    if rc != 0:
        raise RuntimeError(cpp_error(lib))
    return np.stack([y[t * y_stride: t * y_stride + rows] for t in range(nt)])


def cpp_moe(lib, qtype, w, ids, x, rows, cols, stride, k, xg):
    w = np.ascontiguousarray(w, dtype=np.uint8).reshape(-1)
    ids = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    y = np.zeros(k * rows, dtype=np.float32)
    rc = lib.scd_moe_gemv(qtype, w.ctypes.data, w.nbytes, stride, ids.ctypes.data, k, x.ctypes.data, cols, xg,
                          rows, cols, y.ctypes.data)
    if rc != 0:
        raise RuntimeError(cpp_error(lib))
    return y.reshape(k, rows)


def ref_gemv(qtype, w, x, rows, cols):
    wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.uint8).reshape(-1).copy()).cuda()
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32).reshape(-1).copy()).cuda()
    return sk.gemv(qtype, wt, xt, rows, cols).cpu().numpy()


def ref_moe(qtype, w, ids, x, rows, cols, stride, k, xg):
    wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.uint8).reshape(-1).copy()).cuda()
    it = torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int32).reshape(-1).copy()).cuda()
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32).reshape(-1).copy()).cuda()
    y = torch.zeros((k, rows), dtype=torch.float32, device="cuda")
    sk.moe_gemv(qtype, wt, it, xt, y, rows, cols, stride, k, xg)
    torch.cuda.synchronize()
    return y.cpu().numpy()


def rel_err(a, b) -> float:
    """max |a - b| relative to the peak of the reference b (the output scale), inf when not finite."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    peak = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / peak)


def bit_equal(a, b) -> bool:
    return np.array_equal(np.asarray(a, np.float32).view(np.uint32), np.asarray(b, np.float32).view(np.uint32))


class Report:
    def __init__(self):
        self.rows = []
        self.failures = []

    def add(self, ty: str, case: str, err: float, bit: bool | None, ok: bool):
        self.rows.append((ty, case, err, bit))
        if not ok:
            self.failures.append(f"{ty} {case}: max rel err {err:.3e}")

    def print(self):
        print(f"{'type':<8} {'case':<42} {'max_rel_err':>12} {'bit-identical':>14}")
        for ty, case, err, bit in self.rows:
            b = "n/a" if bit is None else ("yes" if bit else "no")
            print(f"{ty:<8} {case:<42} {err:>12.3e} {b:>14}")


def encode(qtype: int, w: np.ndarray):
    """Bytes of `w` (rows x cols float32) in the GGUF layout of `qtype`, or None when gguf cannot encode it."""
    if gquants is None:
        return None
    name = GGML[qtype]
    try:
        enum = getattr(GGMLQuantizationType, name)
    except AttributeError:
        return None
    try:
        q = gquants.quantize(w.astype(np.float32), enum)
    except (NotImplementedError, Exception):
        return None
    return np.ascontiguousarray(q, dtype=np.uint8)


def main() -> int:
    global np, torch, sk, GGUFModelFile, GGMLQuantizationType, gquants
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    if not sc_env.require("SC_MODEL_14B"):
        return 0
    try:
        import numpy as numpy_mod
        import torch as torch_mod
        from stratafreet import strata_kernels as sk_mod
        from stratafreet.gguf_io import GGUFModelFile as gguf_model_file
    except Exception as e:
        return sc_env.skip(f"cannot import the StrataFreet oracle: {e}")
    np, torch, sk, GGUFModelFile = numpy_mod, torch_mod, sk_mod, gguf_model_file
    try:
        from gguf import GGMLQuantizationType as qtype_enum
        from gguf import quants as gguf_quants
    except Exception:  # pragma: no cover - reported as "no encoder" below
        qtype_enum, gguf_quants = None, None
    GGMLQuantizationType, gquants = qtype_enum, gguf_quants

    lib = load_dll()
    print("device:", lib.scd_device().decode(errors="replace"))
    print("reference: stratafreet.strata_kernels (sk.gemv / sk.moe_gemv / PTX-or-cubin of StrataFreet's kernels.cu)")
    rng = np.random.default_rng(20261120)
    rep = Report()
    skipped = []
    checked_types = set()

    def check(ty_id: int, case: str, cpp: np.ndarray, ref: np.ndarray, bit_ok=True):
        err = rel_err(cpp, ref)
        bit = bit_equal(cpp, ref) if bit_ok else None
        rep.add(GGML[ty_id], case, err, bit, err <= TOL)
        checked_types.add(ty_id)

    # ---- real rows of the 14B model (Q6_K) ----
    m = GGUFModelFile(GGUF_14B)
    t = m["blk.0.attn_gate.weight"]
    rows, cols = int(t.shape[0]), int(t.shape[1])
    rb = t.row_bytes
    w = np.array(t.data[: rows * rb], dtype=np.uint8)
    x = rng.standard_normal(cols).astype(np.float32) * 0.5
    check(14, f"gemv real attn_gate {rows}x{cols}", cpp_gemv(lib, 14, w, x, rows, cols), ref_gemv(14, w, x, rows, cols))

    for name, xg_mode in [("blk.0.ffn_gate_exps.weight", "broadcast"), ("blk.0.ffn_down_exps.weight", "per-expert")]:
        t = m[name]
        n_exp, rows, cols = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
        rb = cols // 256 * 210           # one Q6_K row: 256 values in 210 bytes
        stride = rows * rb               # one expert
        assert n_exp * stride == t.nbytes, (name, n_exp * stride, t.nbytes)
        ids = np.array([0, 7, 13, 40, 89, -1, 2, 5], dtype=np.int32)
        k = len(ids)
        w = np.array(t.data[: n_exp * stride], dtype=np.uint8)
        xg = k if xg_mode == "broadcast" else 1
        xrows = (k - 1) // xg + 1
        x = rng.standard_normal((xrows, cols)).astype(np.float32) * 0.5
        case = f"moe real {name.split('.')[1]} k={k} xg={xg}"
        check(14, case, cpp_moe(lib, 14, w, ids, x, rows, cols, stride, k, xg),
              ref_moe(14, w, ids, x, rows, cols, stride, k, xg))

    # ---- MXFP4 (type 39, a StrataFreet addition): real gpt-oss-20b expert matrices, 2880 x 2880 (the _p kernels) ----
    if os.path.isfile(GPT_OSS):
        g = GGUFModelFile(GPT_OSS)
        for name, xg_mode in [("blk.0.ffn_gate_exps.weight", "broadcast"), ("blk.0.ffn_down_exps.weight", "per-expert")]:
            t = g[name]
            n_exp, rows, cols = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
            rb = cols // 32 * 17             # one MXFP4 row: 32 values in 17 bytes per block
            stride = rows * rb               # one expert
            assert n_exp * stride == t.nbytes, (name, n_exp * stride, t.nbytes)
            w = np.array(t.data[: n_exp * stride], dtype=np.uint8)
            x = rng.standard_normal(cols).astype(np.float32) * 0.5
            check(39, f"gemv real {name.split('.')[1]} expert 0 {rows}x{cols}", cpp_gemv(lib, 39, w[:stride], x, rows, cols),
                  ref_gemv(39, w[:stride], x, rows, cols))
            ids = np.array([0, 7, 13, 31, -1, 2, 5, 19], dtype=np.int32)
            k = len(ids)
            xg = k if xg_mode == "broadcast" else 1
            xrows = (k - 1) // xg + 1
            x = rng.standard_normal((xrows, cols)).astype(np.float32) * 0.5
            case = f"moe real {name.split('.')[1]} k={k} xg={xg}"
            check(39, case, cpp_moe(lib, 39, w, ids, x, rows, cols, stride, k, xg),
                  ref_moe(39, w, ids, x, rows, cols, stride, k, xg))
    else:
        skipped.append(f"MXFP4 real rows: {GPT_OSS} not found")

    # ---- encoded types (gguf) and the float types ----
    for qt in COMPARE_TYPES:
        if qt == 14:
            continue
        cases = [(256, 512), (96, 544)]  # 512: the full-row kernels; 544: the `_p` kernels
        encoded_any = False
        for rows, cols in cases:
            wf = (rng.standard_normal((rows, cols)).astype(np.float32) * 0.1)
            enc = encode(qt, wf)
            if enc is None:
                break
            encoded_any = True
            wb = enc.reshape(-1)
            x = rng.standard_normal(cols).astype(np.float32) * 0.5
            check(qt, f"gemv {rows}x{cols}", cpp_gemv(lib, qt, wb, x, rows, cols), ref_gemv(qt, wb, x, rows, cols))
            # gemvn: 3 tokens, x and y rows padded apart
            nt = 3
            xs = cols + 32
            xn = rng.standard_normal((nt, xs)).astype(np.float32) * 0.5
            cppn = cpp_gemvn(lib, qt, wb, xn, xs, rows, cols, nt, rows + 8)
            refn = np.stack([ref_gemv(qt, wb, xn[i, :cols], rows, cols) for i in range(nt)])
            check(qt, f"gemvn nt=3 {rows}x{cols} x_stride={xs}", cppn, refn)
        if encoded_any and cols == 544:
            # moe at the `_p` shape: 6 experts, expert e routes to its own x row (xg = 1) and to a shared one (xg = k)
            E, rows, cols = 6, 64, 544
            wf = rng.standard_normal((E, rows, cols)).astype(np.float32) * 0.1
            enc = encode(qt, wf.reshape(E * rows, cols))
            if enc is not None:
                wb = enc.reshape(-1)
                stride = rows * (wb.size // (E * rows))
                ids = np.array([3, 0, -1, 5], dtype=np.int32)
                k = len(ids)
                for xg in (k, 1):
                    xrows = (k - 1) // xg + 1
                    x = rng.standard_normal((xrows, cols)).astype(np.float32) * 0.5
                    check(qt, f"moe _p 64x544 k={k} xg={xg}", cpp_moe(lib, qt, wb, ids, x, rows, cols, stride, k, xg),
                          ref_moe(qt, wb, ids, x, rows, cols, stride, k, xg))
        if not encoded_any:
            skipped.append(f"{GGML[qt]} (id {qt}): the installed gguf package has no encoder for it"
                           f", and it is not in the 14B file")

    # F32 and BF16: no StrataFreet kernel, so a float64 oracle (x is read as float on the device)
    for qt, rows, cols in [(0, 256, 512), (30, 96, 544)]:
        wf = rng.standard_normal((rows, cols)).astype(np.float32) * 0.1
        if qt == 0:
            wb = wf.view(np.uint8).reshape(-1)
            wdec = wf
        else:
            bits = (wf.view(np.uint32) >> 16).astype(np.uint16)
            wb = bits.view(np.uint8).reshape(-1)
            wdec = (bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
        x = rng.standard_normal(cols).astype(np.float32) * 0.5
        oracle = wdec.astype(np.float64) @ x.astype(np.float64)
        cpp = cpp_gemv(lib, qt, wb, x, rows, cols)
        err = rel_err(cpp, oracle)
        rep.add(GGML[qt], f"gemv {rows}x{cols} vs float64 oracle", err, None, err <= TOL)
        checked_types.add(qt)

    rep.print()
    print()
    print("types compared:", ", ".join(GGML[q] for q in sorted(checked_types)))
    print("skipped (not encoded here, not in the 14B file):", "; ".join(skipped) if skipped else "none")
    print("not compared: F16 (no kernel, upload converts it to F32), TQ1_0/TQ2_0 (not in sk.TYPES), FP8/NVFP4 (not ported)")
    if rep.failures:
        print()
        print("FAILED:")
        for f in rep.failures:
            print("  " + f)
        print("FAIL")
        return 1
    print()
    print(f"PASS: {len(rep.rows)} comparisons, max relative error <= {TOL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
