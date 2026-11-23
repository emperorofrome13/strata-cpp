"""Test of the CPU expert path (src\\cpu_moe.cpp) through build\\strata_cpu_test.dll.

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).

Checks, on the Qwen3.6-14B-A3B Q6_K model (the small model of PORT_MAP section 8):
  1. int_dots_selftest: every integer kernel, scalar and AVX2, against the dequantized dot product.
  2. 20 random cases (layer, 4 expert ids, routing weights, x ~ N(0, 1) * 0.5): sct_experts against
     StrataFreet's ggml CPU path (stratafreet.cpu_ggml.CpuExperts, act "silu"). Relative L2 error must be
     <= 2e-2 per case (the activation is quantized differently, which is expected).
  3. experts_multi against experts, token by token: must be bit-identical.
  4. Timing of experts() per expert at 4, 6, 7 and 8 threads: 200 calls of 2 experts, three blocks each,
     page cache warmed first.

Run:  python tests/test_cpu_moe.py   (after build_cpu.bat)

Needs SC_MODEL_14B and STRATAFREET_ROOT (the Python + ggml CPU oracle); the MXFP4 checks also use
SC_GPT_OSS_GGUF when set. CPU only. Without the model or the oracle the test prints SKIP and exits 0.
"""
import ctypes
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

DLL = os.path.join(ROOT, os.environ.get("SC_BUILD_DIR", "build"), "strata_cpu_test.dll")
MODEL = sc_env.get("SC_MODEL_14B")
TOL = 2e-2
N_CASES = 20
THREADS = (4, 6, 7, 8)
TIMING_CALLS = 200
TIMING_BLOCKS = 3

# Imported in main() after the SKIP gate.
np = None
GGUFModelFile = None
CpuExperts = None

P_F = ctypes.POINTER(ctypes.c_float)
P_I32 = ctypes.POINTER(ctypes.c_int32)


def load_lib():
    lib = ctypes.CDLL(DLL)
    lib.sct_open.argtypes, lib.sct_open.restype = [ctypes.c_char_p], ctypes.c_void_p
    lib.sct_close.argtypes, lib.sct_close.restype = [ctypes.c_void_p], None
    lib.sct_last_error.argtypes, lib.sct_last_error.restype = [], ctypes.c_char_p
    lib.sct_info.argtypes, lib.sct_info.restype = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)], ctypes.c_int
    lib.sct_experts.argtypes = [ctypes.c_void_p, ctypes.c_int, P_I32, P_F, ctypes.c_int, P_F, P_F, ctypes.c_int]
    lib.sct_experts.restype = ctypes.c_int
    lib.sct_experts_multi.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, P_I32, P_F, ctypes.c_int, P_F, P_F,
                                      ctypes.c_int]
    lib.sct_experts_multi.restype = ctypes.c_int
    lib.sct_dot_selftest.argtypes, lib.sct_dot_selftest.restype = [], ctypes.c_int
    lib.sct_default_threads.argtypes, lib.sct_default_threads.restype = [], ctypes.c_int
    lib.sct_dot_rows.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
                                 ctypes.c_longlong, P_F, ctypes.c_longlong, P_F]
    lib.sct_dot_rows.restype = ctypes.c_int
    return lib


def last_error(lib):
    e = lib.sct_last_error()
    return e.decode("utf-8", "replace") if e else ""


def fptr(a):
    return a.ctypes.data_as(P_F)


def ours(lib, h, layer, ids, weights, x, threads):
    n = len(ids)
    id_arr = (ctypes.c_int32 * n)(*[int(i) for i in ids])
    w_arr = (ctypes.c_float * n)(*[float(v) for v in weights])
    xv = np.ascontiguousarray(x, dtype=np.float32)
    y = np.zeros(xv.shape[0], dtype=np.float32)
    rc = lib.sct_experts(h, layer, id_arr, w_arr, n, fptr(xv), fptr(y), threads)
    if rc != 0:
        raise RuntimeError(f"sct_experts failed: {last_error(lib)}")
    return y


def ggml_parts(g, layer, ids):
    """Per expert {role: (qtype, data pointer, rows, cols)} as stratafreet/experts.py _cpu_compute builds them: the
    TensorRef row_bytes is one expert's byte size, and shape is (n_expert, rows, cols)."""
    parts = []
    for e in ids:
        part = {}
        for role, name in (("gate", "ffn_gate_exps.weight"), ("up", "ffn_up_exps.weight"),
                           ("down", "ffn_down_exps.weight")):
            t = g[f"blk.{layer}.{name}"]
            part[role] = (int(t.qtype), t.data.ctypes.data + int(e) * t.row_bytes, int(t.shape[1]), int(t.shape[2]))
        parts.append(part)
    return parts


def touch(g, layer, ids):
    """Read the bytes of the given experts once so the timing runs see a warm page cache."""
    total = 0
    for e in ids:
        for name in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            t = g[f"blk.{layer}.{name}"]
            seg = np.asarray(t.data[int(e) * t.row_bytes:(int(e) + 1) * t.row_bytes])
            total += int(seg[::4096].sum())
    return total


# ---- MXFP4 (type 39, StrataFreet addition): the integer kernel against the float64 dot of the exact dequant.
GPT_OSS = sc_env.get("SC_GPT_OSS_GGUF")
MXFP4 = 39
MX_CASES = (("blk.0.ffn_gate_exps.weight", 0), ("blk.0.ffn_down_exps.weight", 31))


def q8k_values(x):
    """The activation as quantize_q8k (cpu_moe.cpp) leaves it: per 256 values, d * q with q = nearbyint(is * v),
    is = -127 / (the first value of largest magnitude), d = 1 / is; the last block is zero-padded. Returns the
    dequantized values and the step d of each value's block (the rounding error of a value is at most d / 2)."""
    x = np.asarray(x, dtype=np.float32)
    pad = (-len(x)) % 256
    xp = np.concatenate([x, np.zeros(pad, dtype=np.float32)]).reshape(-1, 256)
    out = np.zeros_like(xp)
    step = np.zeros(len(xp), dtype=np.float64)
    for bi, b in enumerate(xp):
        amax = np.max(np.abs(b))
        if amax == 0.0:
            continue
        mx = b[np.flatnonzero(np.abs(b) == amax)[0]]
        is_ = np.float32(-127.0) / np.float32(mx)
        q = np.clip(np.rint(is_ * b), -128, 127).astype(np.float32)
        d = np.float32(1.0) / is_
        out[bi] = d * q
        step[bi] = abs(float(d))   # d carries the sign of the block's largest value
    return out.reshape(-1)[:len(x)], np.repeat(step, 256)[:len(x)]


def mxfp4_checks(lib):
    """Real gpt-oss-20b MXFP4 expert matrices (2880 x 2880, read through GGUFModelFile; the file is mapped only).
    Results per matrix: scalar against AVX2 (relative to max |y|, must be <= 1e-6); the kernel against the float64
    dot of the exact dequant with the Q8_K activation (<= 1e-5); and the kernel against the float64 dot with the
    unquantized activation, which must lie within the activation rounding bound of each row. Needs no model besides
    these rows."""
    if not os.path.isfile(GPT_OSS):
        print(f"[1b] MXFP4 matvec: {sc_env.skip_line('SC_GPT_OSS_GGUF')}")
        return None
    from stratafreet.quant import dequant
    import torch
    m = GGUFModelFile(GPT_OSS)
    rng = np.random.default_rng(20261120)
    ok = True
    for name, e in MX_CASES:
        t = m[name]
        n_exp, rows, cols = (int(v) for v in t.shape)
        rb = cols // 32 * 17
        span = np.ascontiguousarray(t.data[e * rows * rb:(e + 1) * rows * rb])
        x = (rng.standard_normal(cols) * 0.5).astype(np.float32)
        xq, step = q8k_values(x)
        w64 = dequant(torch.from_numpy(span.copy()), MXFP4, (rows, cols), torch.float32).reshape(rows, cols).double().numpy()
        ref_q = w64 @ xq.astype(np.float64)
        ref_x = w64 @ x.astype(np.float64)
        bytes_p = span.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ys, ya = np.zeros(rows, np.float32), np.zeros(rows, np.float32)
        for scalar, y in ((1, ys), (0, ya)):
            rc = lib.sct_dot_rows(MXFP4, scalar, bytes_p, rows, rb, fptr(x), cols, fptr(y))
            if rc != 0:
                print(f"[1b] MXFP4 matvec {name}: sct_dot_rows failed: {last_error(lib)}")
                return False
        scale = max(np.max(np.abs(ref_q)), 1e-30)
        r_sa = np.max(np.abs(ys.astype(np.float64) - ya)) / scale
        r_q = np.max(np.abs(ya.astype(np.float64) - ref_q)) / scale
        r_s = np.max(np.abs(ys.astype(np.float64) - ref_q)) / scale
        # The activation rounding bound per row: |y - y_exact| <= sum_j |w_j| * d_j / 2 (each |x_j - xq_j| <= d_j / 2).
        bound = np.abs(w64) @ (step.astype(np.float64) / 2.0)
        r_x = np.max(np.abs(ya.astype(np.float64) - ref_x) / np.maximum(bound, 1e-30))
        good = r_sa <= 1e-6 and r_q <= 1e-5 and r_s <= 1e-5 and r_x <= 1.0 + 1e-4
        ok &= good
        print(f"[1b] MXFP4 matvec {name} expert {e} ({rows}x{cols}): scalar-vs-avx2 {r_sa:.2e} (<=1e-6), "
              f"avx2-vs-f64(q8) {r_q:.2e} (<=1e-5), scalar-vs-f64(q8) {r_s:.2e} (<=1e-5), "
              f"|y - f64(exact x)| / q8 rounding bound max {r_x:.3f} (<=1) {'ok' if good else 'FAILED'}")
    return ok


def main():
    global np, GGUFModelFile, CpuExperts
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    if not sc_env.get("SC_MODEL_14B"):
        print(sc_env.skip_line("SC_MODEL_14B"))
        return 0
    try:
        import numpy as numpy_mod
        from stratafreet.gguf_io import GGUFModelFile as gguf_model_file
        from stratafreet.cpu_ggml import CpuExperts as cpu_experts
    except Exception as e:
        return sc_env.skip(f"cannot import the StrataFreet oracle: {e}")
    np, GGUFModelFile, CpuExperts = numpy_mod, gguf_model_file, cpu_experts

    ok = True
    if not os.path.exists(DLL):
        print(f"FAIL: {DLL} is missing; run build_cpu.bat first")
        return 1
    lib = load_lib()

    # 1. kernel self-test (Rust int_dots_match_dequant, plus the scalar/AVX2 pair per type)
    rc = lib.sct_dot_selftest()
    print(f"[1] int_dots_selftest: {'ok' if rc == 0 else 'FAILED: ' + last_error(lib)}")
    ok &= rc == 0
    mxfp4 = mxfp4_checks(lib)
    if mxfp4 is not None:   # None: SC_GPT_OSS_GGUF is unset, the case was skipped
        ok &= mxfp4

    if not os.path.exists(MODEL):
        print(f"FAIL: model not found: {MODEL}")
        return 1
    g = GGUFModelFile(MODEL)
    h = lib.sct_open(MODEL.encode("utf-8"))
    if not h:
        print(f"FAIL: sct_open: {last_error(lib)}")
        return 1
    info = (ctypes.c_int64 * 5)()
    lib.sct_info(h, info)
    n_layer, n_embd, n_expert, n_used, n_ff = (int(v) for v in info)
    print(f"model: {os.path.basename(MODEL)} n_layer={n_layer} n_embd={n_embd} n_expert={n_expert} "
          f"n_expert_used={n_used} n_ff_exp={n_ff}")
    moe_layers = sorted(int(k.split(".")[1]) for k in g.tensors if k.endswith(".ffn_gate_exps.weight"))
    if not moe_layers or n_expert != int(g[f"blk.{moe_layers[0]}.ffn_gate_exps.weight"].shape[0]):
        print("FAIL: no MoE layers or n_expert mismatch with the GGUF")
        return 1

    from stratafreet.cpu_ggml import load as ggml_load
    if ggml_load() is None:
        return sc_env.skip("the ggml CPU oracle is not available (StrataFreet needs an LM Studio "
                           "llama.cpp runtime with ggml-cpu.dll)")
    ref = CpuExperts("silu", threads=7)

    # 2. 20 random cases against ggml
    rng = np.random.default_rng(20261120)
    errs = []
    print("[2] random cases vs ggml (rel L2, tol %.0e):" % TOL)
    for c in range(N_CASES):
        layer = int(rng.choice(moe_layers))
        ids = [int(v) for v in rng.choice(n_expert, size=4, replace=False)]
        weights = [float(v) for v in rng.uniform(0.1, 0.6, size=4)]
        x = (rng.standard_normal(n_embd) * 0.5).astype(np.float32)
        y_ours = ours(lib, h, layer, ids, weights, x, 7)
        y_ref = np.asarray(ref(ggml_parts(g, layer, ids), x, weights), dtype=np.float32)
        denom = float(np.linalg.norm(y_ref))
        rel = float(np.linalg.norm(y_ours.astype(np.float64) - y_ref.astype(np.float64)) / max(denom, 1e-30))
        errs.append(rel)
        flag = "ok" if rel <= TOL else "FAIL"
        print(f"  case {c:2d} layer {layer:2d} ids {ids} |y_ref|={denom:.4g} rel_l2={rel:.3e} {flag}")
        ok &= rel <= TOL
    print(f"  max rel_l2 = {max(errs):.3e}, mean = {statistics.mean(errs):.3e}")

    # 3. experts_multi == experts per token
    nt = 3
    layer = moe_layers[0]
    ids_t = [[int(v) for v in rng.choice(n_expert, size=4, replace=False)] for _ in range(nt)]
    w_t = [[float(v) for v in rng.uniform(0.1, 0.6, size=4)] for _ in range(nt)]
    xs = (rng.standard_normal((nt, n_embd)) * 0.5).astype(np.float32)
    flat_ids = (ctypes.c_int32 * (nt * 4))(*[i for row in ids_t for i in row])
    flat_w = (ctypes.c_float * (nt * 4))(*[v for row in w_t for v in row])
    ys_m = np.zeros((nt, n_embd), dtype=np.float32)
    rc = lib.sct_experts_multi(h, layer, nt, flat_ids, flat_w, 4, fptr(xs), fptr(ys_m), 7)
    if rc != 0:
        print(f"FAIL: sct_experts_multi: {last_error(lib)}")
        return 1
    max_diff = 0.0
    for t in range(nt):
        y1 = ours(lib, h, layer, ids_t[t], w_t[t], xs[t], 7)
        max_diff = max(max_diff, float(np.max(np.abs(y1 - ys_m[t]))))
    print(f"[3] experts_multi vs experts, {nt} tokens: max abs diff = {max_diff:.3e} "
          f"{'ok' if max_diff == 0.0 else 'FAIL'}")
    ok &= max_diff == 0.0

    # 4. timing: experts() per expert, 2 experts per call, warm cache
    layer = moe_layers[0]
    ids2 = [int(v) for v in rng.choice(n_expert, size=2, replace=False)]
    w2 = [0.5, 0.5]
    x = (rng.standard_normal(n_embd) * 0.5).astype(np.float32)
    touch(g, layer, ids2)
    print(f"[4] experts() timing, layer {layer}, ids {ids2}, {TIMING_CALLS} calls x 2 experts, "
          f"{TIMING_BLOCKS} blocks (median), warm cache")
    print(f"    default_threads() = {lib.sct_default_threads()} (physical cores)")
    timing = {}
    for th in THREADS:
        for _ in range(5):          # warm-up: builds the pool for this thread count, touches the code
            ours(lib, h, layer, ids2, w2, x, th)
        blocks = []
        for _ in range(TIMING_BLOCKS):
            t0 = time.perf_counter()
            for _ in range(TIMING_CALLS):
                ours(lib, h, layer, ids2, w2, x, th)
            dt = time.perf_counter() - t0
            blocks.append(dt / (TIMING_CALLS * len(ids2)) * 1e6)
        timing[th] = statistics.median(blocks)
        print(f"    threads {th}: " + "  ".join(f"{b:.1f}" for b in blocks) +
              f" us/expert (median {timing[th]:.1f})")

    lib.sct_close(h)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
