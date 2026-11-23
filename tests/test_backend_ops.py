"""W2-C part 2a acceptance: the C++ CUDA launches of CudaBackend against NumPy references written from the Rust.
Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see LICENSE-RustingStrata.md).

    python tests/test_backend_ops.py

The C++ side is build/strata_cpp_dev.dll (scd_*): each call uploads host arrays, runs one op through CudaBackend
(the NVRTC module built from kernels/kernels.cu with Strata's macros), syncs and copies the outputs back. The
references are float64 NumPy, written from the Rust source (ops.rs, model.rs, cuda/mod.rs, kernels.cu):

  gemm / gemm_dense      y = x W^T with act 0 store, 1 silu(y) * result, 2 y += result; W from the 14B GGUF
                         (Q6_K, the int8 path on sm_80+) through stratafreet.quant.dequant, and F32 weights.
  rms_norm(_q)           ops::rms_norm: x * rsqrt(mean(x^2) + eps) * w; res added first; quant_xq layout.
  argmax_rows            ops::argmax: first index of the largest value (exact); ties checked explicitly.
  l2_norm                ops::l2_norm: x / max(||x||, eps) per vector (exact layout of token / vector).
  rope_pairs             ops::rope_neox on pairs lo..hi, theta = pos * base^(-2i / n_rot) in float64.
  rms_norm_rope_kv       rms_norm then rope on the first n_rot dims; the head and v go to the f16 caches.
  add, swiglu(_q)        x + y; silu(g) * u (quant_xq layout for _q).
  norm_gate(_q/_sig)     rmsnorm(o) * w * silu(z) (sigmoid(z) for _sig); quant_xq layout for _q.
  kv_store               f16 of k and v at positions step0 ... (rows of n values per position).
  attn_decode_gq         softmax(q K^T * scale) V over positions 0 ..= step[2 t], GQA: head h reads KV head
                         h // (nh / nkv); K/V are the f16 cache values.
  attn_prefill_m         the same per query row, query t at position step0 + t (causal).
  router_topk            ops::softmax (or sigmoid), top k with ties to the lower index (exact ids), weights
                         renormalized over the k picks with the 6.1e-5 floor, times the extra scale when set.
  moe_route              router_topk, slots = table[id], wh = w on hits; mailbox words as moe_route writes them.
  moe_combine(_seq)      x += sum_j w ye + sigmoid(sg) sh (+ the CPU result of a token with a missed slot).

Tolerance: relative error (max abs error / peak of the reference) <= 1e-3 for fp32 paths, <= 1e-2 where the
kernel reads f16 caches or int8 activations or uses the int8 tensor cores. Ids are compared exactly.

Shapes: the 14B test model (qwen35moe: n_embd 2048, 16 heads over 2 KV heads, head_dim 256, n_rot 64, 90 experts,
8 used) and Flash-Next (qwen4exp: n_embd 2560, 24 heads over 2, head_dim from its metadata, n_rot 64, 512 experts,
10 used). Only the GGUF headers of Flash-Next are read; no model is loaded.

Needs STRATAFREET_ROOT, SC_MODEL_14B, SC_MODEL_FLASHNEXT, and an NVIDIA GPU. Without the oracle or a
model the test prints SKIP and exits 0 (before any CUDA context exists).
"""

from __future__ import annotations

import ctypes
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

DLL = os.path.join(ROOT, os.environ.get("SC_BUILD_DIR", "build"), "strata_cpp_dev.dll")
GGUF_14B = sc_env.get("SC_MODEL_14B")
GGUF_FLASH = sc_env.get("SC_MODEL_FLASHNEXT")
TOL_FP32 = 1e-3
TOL_LOWP = 1e-2
INT_MAX = 2**31 - 1
Q6_K = 14
F32 = 0

# Imported in main() after the SKIP gate.
np = None
torch = None
GGUFModelFile = None
dequant = None

vp = ctypes.c_void_p
ci = ctypes.c_int
cf = ctypes.c_float
cll = ctypes.c_longlong
cu = ctypes.c_uint

FAILS: list[str] = []
LINES: list[str] = []


def load_dll():
    if not os.path.isfile(DLL):
        raise SystemExit(f"missing {DLL}; run build.bat first")
    lib = ctypes.CDLL(DLL)
    lib.sci_last_error.restype = ctypes.c_char_p
    lib.scd_device.restype = ctypes.c_char_p
    sigs = {
        "scd_gemm_dense": [ci, vp, cll, ci, ci, ci, vp, ci, vp, ci, ci],
        "scd_gemm_tiles": [ci, vp, cll, ci, ci, ci, vp, vp, vp, ci, ci, vp, vp, ci, ci, ci],
        "scd_rms_norm": [vp, ci, vp, vp, ci, ci, ci, cf, vp, vp, vp],
        "scd_argmax_rows": [vp, ci, ci, vp],
        "scd_l2_norm": [vp, ci, ci, ci, ci, ci, cf],
        "scd_rope_pairs": [vp, ci, ci, ci, ci, ci, ci, cf, ci, ci],
        "scd_rms_norm_rope_kv": [vp, ci, vp, vp, ci, ci, cf, ci, ci, ci, cf, vp, vp, vp, ci],
        "scd_add": [vp, vp, ci],
        "scd_swiglu": [vp, vp, ci, ci, vp],
        "scd_norm_gate": [vp, vp, vp, ci, ci, cf, ci, vp],
        "scd_kv_store": [vp, vp, vp, vp, ci, ci, ci, ci, ci],
        "scd_attn_decode_gq": [vp, vp, vp, ci, vp, ci, ci, ci, cf, ci, ci, vp, vp, ci],
        "scd_attn_prefill_m": [vp, vp, vp, ci, ci, ci, ci, ci, ci, cf, vp],
        "scd_router_topk": [vp, ci, ci, cf, ci, ci, vp, vp],
        "scd_moe_route": [vp, ci, ci, cf, vp, ci, ci, vp, ci, ci, ci, cu, ci, vp, ci, vp, vp, vp, vp, vp],
        "scd_moe_combine": [vp, vp, vp, ci, vp, vp, vp, ci, ci, vp, ci, ci, ci, cu],
        "scd_moe_combine_seq": [vp, vp, ci, vp, vp, ci, vp, vp, ci, ci],
    }
    for name, args in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ci
    return lib


LIB = None


def cerr() -> str:
    return (LIB.sci_last_error() or b"").decode(errors="replace")


def call(name, *args):
    rc = getattr(LIB, name)(*args)
    if rc != 0:
        raise RuntimeError(f"{name}: {cerr()}")


def P(a):
    """The address of a contiguous NumPy array, or NULL for None."""
    return None if a is None else a.ctypes.data


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def i32(a):
    return np.ascontiguousarray(a, dtype=np.int32)


def u16_of_f16(a):
    """f16 bit patterns (uint16) of a float array, rounded to nearest even as the kernels do."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).astype(np.float16)).view(np.uint16)


def f16_back(bits):
    """float64 values of uint16 f16 bit patterns."""
    return bits.view(np.float16).astype(np.float64)


def f32_bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def check(label: str, got, want, tol: float):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    peak = max(float(np.max(np.abs(want))), 1e-30)
    rel = float(np.max(np.abs(got - want))) / peak if got.size else 0.0
    ok = rel <= tol and np.all(np.isfinite(got))
    line = f"  {label:<58s} rel {rel:9.2e}  (tol {tol:.0e})  {'ok' if ok else 'FAIL'}"
    LINES.append(line)
    print(line)
    if not ok:
        FAILS.append(label)


def check_exact(label: str, got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    ok = got.shape == want.shape and np.array_equal(got, want)
    mism = int(np.sum(got != want)) if got.shape == want.shape else -1
    line = f"  {label:<58s} exact, {mism} mismatches of {want.size}  {'ok' if ok else 'FAIL'}"
    LINES.append(line)
    print(line)
    if not ok:
        FAILS.append(label)


def check_bool(label: str, ok: bool, detail: str = ""):
    line = f"  {label:<58s} {detail}  {'ok' if ok else 'FAIL'}"
    LINES.append(line)
    print(line)
    if not ok:
        FAILS.append(label)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax_rows(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def xq_deq(xq_bytes: np.ndarray, rows: int, n: int) -> np.ndarray:
    """Dequantize the quant_xq layout: rows * n int8 values, then rows * n/32 f32 scales."""
    raw = np.ascontiguousarray(xq_bytes[: rows * n + 4 * rows * (n // 32)])
    q = raw[: rows * n].view(np.int8).astype(np.float64).reshape(rows, n)
    d = raw[rows * n:].view(np.float32).astype(np.float64).reshape(rows, n // 32)
    return (q.reshape(rows, n // 32, 32) * d[:, :, None]).reshape(rows, n)


def xq_buf(rows: int, n: int) -> np.ndarray:
    return np.zeros(rows * (n + n // 8), dtype=np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# model shapes from the GGUF headers

def model_shapes():
    g14 = GGUFModelFile(GGUF_14B)
    fl = GGUFModelFile(GGUF_FLASH)
    shapes = {}
    for name, g, pre in (("14B", g14, "qwen35moe"), ("FlashNext", fl, "qwen4exp")):
        hd = g.get(f"{pre}.attention.key_length")
        if hd is None:
            raise SystemExit(f"{name}: no {pre}.attention.key_length in the GGUF header")
        shapes[name] = dict(
            n_embd=int(g.get(f"{pre}.embedding_length")),
            nh=int(g.get(f"{pre}.attention.head_count")),
            nkv=int(g.get(f"{pre}.attention.head_count_kv")),
            hd=int(hd),
            n_rot=int(g.get(f"{pre}.rope.dimension_count")),
            n_expert=int(g.get(f"{pre}.expert_count")),
            k=int(g.get(f"{pre}.expert_used_count")),
            base=float(g.get(f"{pre}.rope.freq_base")),
        )
    return g14, shapes


# ---------------------------------------------------------------------------------------------------------------
# GEMM

def test_gemm(g14, shapes):
    print("gemm / gemm_dense")
    rng = np.random.default_rng(11)
    # 14B: the dense attn_gate weight (Q6_K, int8 path on sm_80+), dequantized by the Python reference.
    t = g14["blk.0.attn_gate.weight"]
    raw = np.array(t.data, copy=True)
    rows, cols = int(t.shape[0]), int(t.shape[1])
    W = dequant(torch.from_numpy(raw), Q6_K, (rows, cols), torch.float32).numpy().astype(np.float64).reshape(rows, cols)
    n = 37
    X = rng.standard_normal((n, cols)).astype(np.float32)
    Y0 = rng.standard_normal((n, rows)).astype(np.float32)
    R = X.astype(np.float64) @ W.T
    for act, want in ((0, R), (2, Y0 + R), (1, silu(Y0.astype(np.float64)) * R)):
        y = np.ascontiguousarray(Y0.copy())
        call("scd_gemm_dense", Q6_K, P(raw), raw.nbytes, rows, cols, n, P(X), cols, P(y), rows, act)
        check(f"14B Q6_K gemm_dense act {act} (n {n}, {rows}x{cols})", y, want, TOL_LOWP)
    # the same with x read through the reused int8 copy (SAME_X): identical math
    y = np.ascontiguousarray(Y0.copy())
    call("scd_gemm_dense", Q6_K, P(raw), raw.nbytes, rows, cols, n, P(X), cols, P(y), rows, 0 | 4)
    check("14B Q6_K gemm_dense act 0 | SAME_X (fresh copy)", y, R, TOL_LOWP)

    # tile lists over the same matrix: positions 0..19, 32..131 and 200..231, x rows through xrow
    n_x = 240
    Xl = rng.standard_normal((n_x, cols)).astype(np.float32)
    perm = rng.permutation(n_x).astype(np.int32)
    tiles = [(0, 20), (32, 100), (200, 32)]
    tm0 = i32([a for a, _ in tiles])
    tcn = i32([b for _, b in tiles])
    y = np.zeros((n_x, rows), dtype=np.float32)
    call("scd_gemm_tiles", Q6_K, P(raw), raw.nbytes, rows, cols, len(tiles), P(tm0), P(tcn), P(Xl), cols, n_x,
         P(perm), P(y), rows, n_x, 0)
    want = np.zeros((n_x, rows), dtype=np.float64)
    rowsel = []
    for m0, cnt in tiles:
        for r in range(cnt):
            rowsel.append(m0 + r)
    rowsel = np.array(rowsel)
    want[rowsel] = Xl[perm[rowsel]].astype(np.float64) @ W.T
    check("14B Q6_K gemm tile lists with xrow (3 tiles)", y[rowsel], want[rowsel], TOL_LOWP)

    # F32 weights (the float path, one warp per row): 2560 columns as Flash-Next's hidden size
    fr, fc = 384, shapes["FlashNext"]["n_embd"]
    Wf = (rng.standard_normal((fr, fc)) * 0.05).astype(np.float32)
    nf = 9
    Xf = rng.standard_normal((nf, fc)).astype(np.float32)
    yf = np.zeros((nf, fr), dtype=np.float32)
    wf_raw = np.ascontiguousarray(Wf).view(np.uint8).reshape(-1)
    call("scd_gemm_dense", F32, P(wf_raw), wf_raw.nbytes, fr, fc, nf, P(Xf), fc, P(yf), fr, 0)
    check(f"F32 gemm_dense (FlashNext width {fc}, n {nf})", yf, Xf.astype(np.float64) @ Wf.astype(np.float64).T, TOL_LOWP)


# ---------------------------------------------------------------------------------------------------------------
# norms, argmax, l2, rope

def test_norms(sh):
    print("rms_norm, rms_norm + res, rms_norm_q, argmax_rows, l2_norm, rope_pairs")
    rng = np.random.default_rng(12)
    n = sh["n_embd"]
    rows = 5
    eps = 1e-6
    x = rng.standard_normal((rows, n)).astype(np.float32) * 3.0
    w = (1.0 + 0.5 * rng.standard_normal(n)).astype(np.float32)
    out = np.zeros((rows, n), dtype=np.float32)
    call("scd_rms_norm", P(x), n, P(w), P(out), n, n, rows, eps, None, None, None)
    ms = (x.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
    check(f"rms_norm ({rows} x {n})", out, x * (1.0 / np.sqrt(ms + eps)) * w, TOL_FP32)

    # with res: x += res first, the output and the updated x
    res = rng.standard_normal((rows, n)).astype(np.float32)
    x2 = np.ascontiguousarray(x.copy())
    xo = np.zeros((rows, n), dtype=np.float32)
    out2 = np.zeros((rows, n), dtype=np.float32)
    call("scd_rms_norm", P(x2), n, P(w), P(out2), n, n, rows, eps, P(res), P(xo), None)
    s = x.astype(np.float64) + res
    check("rms_norm with res (x += res, then the norm): x", xo, s, TOL_FP32)
    check("rms_norm with res: out", out2, s * (1.0 / np.sqrt((s ** 2).mean(axis=1, keepdims=True) + eps)) * w, TOL_FP32)

    # quant_xq output: the dequantized values must match out to the int8 step (peak / 127 per 32 values)
    xq = xq_buf(rows, n)
    out3 = np.zeros((rows, n), dtype=np.float32)
    call("scd_rms_norm", P(x), n, P(w), P(out3), n, n, rows, eps, None, None, P(xq))
    check("rms_norm_q: out equals the rms_norm output", out3, out, TOL_FP32)
    check("rms_norm_q: dequantized xq vs out (int8 step)", xq_deq(xq, rows, n), out3, TOL_LOWP)

    # argmax: random rows, plus a tie (first index wins) in row 1 and a 248320-wide row
    vocab = 248320
    lg = rng.standard_normal((3, vocab)).astype(np.float32)
    lg[1, 5] = 9.0
    lg[1, 900] = 9.0
    lg[2, 0] = 100.0
    got = np.zeros(3, dtype=np.int32)
    call("scd_argmax_rows", P(lg), vocab, 3, P(got))
    want = np.array([int(np.argmax(lg[r])) for r in range(3)], dtype=np.int32)
    check_exact(f"argmax_rows (3 rows over {vocab}; row 1 ties at 5 and 900)", got, want)
    check_bool("argmax_rows: the tie resolves to the first index", int(got[1]) == 5, f"got {int(got[1])}")

    # l2_norm: per = 4 vectors of 128 per token, tokens 3 apart by stride
    tok, per, vn = 3, 4, 128
    stride = per * vn + 64  # padded token stride
    lx = np.zeros(tok * stride, dtype=np.float32)
    lx_view = lx.reshape(tok, stride)
    lx_view[:, : per * vn] = rng.standard_normal((tok, per * vn)).astype(np.float32)
    l_in = lx.copy()
    l_eps = 1e-12
    call("scd_l2_norm", P(lx), lx.size, vn, per, stride, tok * per, l_eps)
    ref = l_in.astype(np.float64).reshape(tok, stride).copy()
    for t in range(tok):
        for v in range(per):
            seg = ref[t, v * vn: (v + 1) * vn]
            seg[:] = seg / max(math.sqrt(float((seg ** 2).sum())), l_eps)
    check("l2_norm (3 tokens x 4 vectors of 128, padded stride)", lx.reshape(tok, stride), ref, TOL_FP32)

    # rope_pairs: heads = tokens * per, NEOX pairs lo..hi (full, and a middle M-RoPE section)
    base = sh["base"]
    n_rot = sh["n_rot"]
    half = n_rot // 2
    heads_per, toks, hstride = 4, 3, 256
    heads = heads_per * toks
    step0 = 77
    xr = (rng.standard_normal((heads, hstride)) * 0.7).astype(np.float32)
    for lo, hi in ((0, half), (5, 11)):
        xc = np.ascontiguousarray(xr.reshape(-1).copy())
        call("scd_rope_pairs", P(xc), xc.size, hstride, heads, heads_per, n_rot, step0, base, lo, hi)
        ref = xr.astype(np.float64).copy()
        for h in range(heads):
            pos = step0 + h // heads_per
            for i in range(lo, hi):
                th = pos * base ** (-2.0 * i / n_rot)
                c, s = math.cos(th), math.sin(th)
                a, b = ref[h, i], ref[h, i + half]
                ref[h, i] = a * c - b * s
                ref[h, i + half] = a * s + b * c
        check(f"rope_pairs NEOX lo {lo} hi {hi} (base {base:g}, n_rot {n_rot})", xc.reshape(heads, hstride), ref, TOL_FP32)


# ---------------------------------------------------------------------------------------------------------------
# rms_norm_rope_kv (+ the f16 caches), add, swiglu(_q), norm_gate(_q/_sig), kv_store

def test_rope_kv_and_elementwise(sh):
    print("rms_norm_rope_kv, kv_store, add, swiglu(_q), norm_gate(_q/_sig)")
    rng = np.random.default_rng(13)
    hd = sh["hd"]
    n_rot = sh["n_rot"]
    per = sh["nkv"]
    toks = 3
    rows = toks * per
    eps = 1e-6
    base = sh["base"]
    step0 = 40
    w = (1.0 + 0.3 * rng.standard_normal(hd)).astype(np.float32)
    x = rng.standard_normal((rows, hd)).astype(np.float32)
    v = rng.standard_normal((rows, hd)).astype(np.float32)
    cache_pos = step0 + toks
    cache_len = cache_pos * per * hd + 8
    kc = np.zeros(cache_len, dtype=np.uint16)
    vc = np.zeros(cache_len, dtype=np.uint16)
    out = np.zeros((rows, hd), dtype=np.float32)
    call("scd_rms_norm_rope_kv", P(x), hd, P(w), P(out), hd, rows, eps, per, n_rot, step0, base, P(v), P(kc), P(vc),
         cache_len)
    ms = (x.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
    o = x * (1.0 / np.sqrt(ms + eps)) * w
    half = n_rot // 2
    for r in range(rows):
        pos = step0 + r // per
        for i in range(half):
            th = pos * base ** (-2.0 * i / n_rot)
            c, s = math.cos(th), math.sin(th)
            a, b = o[r, i], o[r, i + half]
            o[r, i] = a * c - b * s
            o[r, i + half] = a * s + b * c
    check(f"rms_norm_rope_kv out (rows {rows}, head_dim {hd}, n_rot {n_rot})", out, o, TOL_FP32)
    kref = np.zeros(cache_len, dtype=np.float64)
    vref = np.zeros(cache_len, dtype=np.float64)
    for r in range(rows):
        pos = step0 + r // per
        e = (pos * per + r % per) * hd
        kref[e: e + hd] = f16_back(u16_of_f16(o[r]))
        vref[e: e + hd] = f16_back(u16_of_f16(v[r]))
    used = slice(step0 * per * hd, cache_pos * per * hd)
    check("rms_norm_rope_kv: K cache (f16) at the new positions", f16_back(kc)[used], kref[used], TOL_LOWP)
    check("rms_norm_rope_kv: V cache (f16) at the new positions", f16_back(vc)[used], vref[used], TOL_LOWP)

    # kv_store: t tokens of n = nkv * hd values at positions step0 ...
    t = 5
    nval = per * hd
    kk = rng.standard_normal((t, nval)).astype(np.float32)
    vv = rng.standard_normal((t, nval)).astype(np.float32)
    kcs = np.zeros(cache_len, dtype=np.uint16)
    vcs = np.zeros(cache_len, dtype=np.uint16)
    sp0 = 9
    call("scd_kv_store", P(kk), P(vv), P(kcs), P(vcs), cache_len, nval, t, sp0, hd)
    ksel = slice(sp0 * nval, (sp0 + t) * nval)
    check("kv_store: K f16 rows at positions step0 ...", f16_back(kcs)[ksel], u16_of_f16(kk.reshape(-1)).view(np.float16).astype(np.float64), 1e-6)
    check("kv_store: V f16 rows at positions step0 ...", f16_back(vcs)[ksel], u16_of_f16(vv.reshape(-1)).view(np.float16).astype(np.float64), 1e-6)

    # add and swiglu over rows
    nn = 3001
    a0 = rng.standard_normal(nn).astype(np.float32)
    b0 = rng.standard_normal(nn).astype(np.float32)
    a = np.ascontiguousarray(a0.copy())
    call("scd_add", P(a), P(b0), nn)
    check("add (x += y, 3001 values)", a, a0.astype(np.float64) + b0, TOL_FP32)

    sr, sn = 3, 512
    g0 = rng.standard_normal((sr, sn)).astype(np.float32) * 2
    u0 = rng.standard_normal((sr, sn)).astype(np.float32)
    g = np.ascontiguousarray(g0.copy())
    call("scd_swiglu", P(g), P(u0), sn, sr, None)
    check("swiglu (g = silu(g) * u)", g, silu(g0.astype(np.float64)) * u0, TOL_FP32)
    xqb = xq_buf(sr, sn)
    g2 = np.ascontiguousarray(g0.copy())
    call("scd_swiglu", P(g2), P(u0), sn, sr, P(xqb))
    prod = silu(g0.astype(np.float64)) * u0
    check("swiglu_q: g left as is", g2, g0, 0.0)
    check("swiglu_q: dequantized xq vs silu(g) * u (int8 step)", xq_deq(xqb, sr, sn), prod, TOL_LOWP)

    # norm_gate: heads of n, the gate silu(z) or (sig) sigmoid(z)
    heads, gn = 6, 128
    o0 = rng.standard_normal((heads, gn)).astype(np.float32)
    z0 = rng.standard_normal((heads, gn)).astype(np.float32)
    wg = (1.0 + 0.2 * rng.standard_normal(gn)).astype(np.float32)
    ms = (o0.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
    base_o = o0 * (1.0 / np.sqrt(ms + eps))
    for sig, gate in ((0, silu), (1, sigmoid)):
        oo = np.ascontiguousarray(o0.copy())
        call("scd_norm_gate", P(oo), P(wg), P(z0), gn, heads, eps, sig, None)
        check(f"norm_gate sig {sig} (heads {heads}, n {gn})", oo, base_o * wg * gate(z0.astype(np.float64)), TOL_FP32)
    xqg = xq_buf(heads, gn)
    oq = np.ascontiguousarray(o0.copy())
    call("scd_norm_gate", P(oq), P(wg), P(z0), gn, heads, eps, 0, P(xqg))
    gq = base_o * wg * silu(z0.astype(np.float64))
    check("norm_gate_q: out equals the plain output", oq, gq, TOL_FP32)
    check("norm_gate_q: dequantized xq vs out (int8 step)", xq_deq(xqg, heads, gn), gq, TOL_LOWP)


# ---------------------------------------------------------------------------------------------------------------
# attention

def attn_ref(q, K, V, positions, nh, nkv, scale):
    """q: (nq, nh, hd); K, V: (npos, nkv, hd) float64 f16 values; positions[i] is the last visible position of row i."""
    nq, _, hd = q.shape
    grp = nh // nkv
    out = np.zeros((nq, nh, hd), dtype=np.float64)
    for i in range(nq):
        p = positions[i]
        Kp = K[: p + 1]
        Vp = V[: p + 1]
        for h in range(nh):
            g = h // grp
            s = (Kp[:, g, :] @ q[i, h].astype(np.float64)) * scale
            s = s - s.max()
            w = np.exp(s)
            w /= w.sum()
            out[i, h] = w @ Vp[:, g, :]
    return out


def test_attention(name, sh):
    print(f"attention decode and prefill ({name}: {sh['nh']} heads over {sh['nkv']}, head_dim {sh['hd']})")
    rng = np.random.default_rng(14 + sh["hd"])
    nh, nkv, hd = sh["nh"], sh["nkv"], sh["hd"]
    scale = 1.0 / math.sqrt(hd)
    npos = 2101
    cache_len = npos * nkv * hd
    kf = (rng.standard_normal((npos, nkv, hd)) * 0.5).astype(np.float32)
    vf = (rng.standard_normal((npos, nkv, hd)) * 0.5).astype(np.float32)
    kc = u16_of_f16(kf.reshape(-1))
    vc = u16_of_f16(vf.reshape(-1))
    K = f16_back(kc).reshape(npos, nkv, hd)
    V = f16_back(vc).reshape(npos, nkv, hd)
    # decode: three tokens at positions below ATT_GQA_MIN, between the regimes, and past it
    positions = [5, 300, 2100]
    nt = len(positions)
    q = (rng.standard_normal((nt, nh, hd)) * 0.5).astype(np.float32)
    steps = np.zeros(2 * nt, dtype=np.int32)
    for t, p in enumerate(positions):
        steps[2 * t] = p
    want = attn_ref(q, K, V, positions, nh, nkv, scale)
    for gq_min, label in ((0, "gqa only (gq_min 0)"), (INT_MAX, "one head per block (gq_min i32::MAX)"),
                          (2048, "regime by position (gq_min 2048)")):
        out = np.zeros((nt, nh, hd), dtype=np.float32)
        call("scd_attn_decode_gq", P(q), P(kc), P(vc), cache_len, P(steps), nh, nkv, hd, scale, gq_min, nt, P(out), None, 0)
        check(f"attn_decode_gq {label}, positions {positions}", out, want, TOL_LOWP)
    # prefill: 20 query rows from position 7 (causal over the cache), and a 0-based block of 12
    for step0, T in ((0, 12), (7, 20)):
        qp = (rng.standard_normal((T, nh, hd)) * 0.5).astype(np.float32)
        outp = np.zeros((T, nh, hd), dtype=np.float32)
        call("scd_attn_prefill_m", P(qp), P(kc), P(vc), cache_len, step0, T, nh, nkv, hd, scale, P(outp))
        wantp = attn_ref(qp, K, V, [step0 + t for t in range(T)], nh, nkv, scale)
        check(f"attn_prefill_m causal (step0 {step0}, {T} rows)", outp, wantp, TOL_LOWP)


# ---------------------------------------------------------------------------------------------------------------
# routing and the MoE combine

def router_ref(P_, k, sigmoid, extra):
    p = sigmoid_fn(P_) if sigmoid else softmax_rows(P_)
    ids = np.zeros((P_.shape[0], k), dtype=np.int64)
    wts = np.zeros((P_.shape[0], k), dtype=np.float64)
    for t in range(P_.shape[0]):
        order = np.argsort(-p[t], kind="stable")[:k]
        s = max(p[t, order].sum(), 6.103515625e-5)
        scale = 1.0 / s
        if extra != 0.0:
            scale *= extra
        ids[t] = order
        wts[t] = p[t, order] * scale
    return ids, wts


def sigmoid_fn(x):
    return sigmoid(x.astype(np.float64))


def test_routing(name, sh):
    print(f"router_topk, moe_route, moe_combine, moe_combine_seq ({name}: {sh['n_expert']} experts, top {sh['k']})")
    rng = np.random.default_rng(21 + sh["n_expert"])
    n, k = sh["n_expert"], sh["k"]
    assert n <= 512 and k <= 32
    t = 4
    Pl = (rng.standard_normal((t, n)) * 2.0).astype(np.float32)
    for sigmoid_flag in (0, 1):
        for extra in (0.0, 2.5):
            ids = np.zeros((t, k), dtype=np.int32)
            wts = np.zeros((t, k), dtype=np.float32)
            call("scd_router_topk", P(Pl), n, k, extra, sigmoid_flag, t, P(ids), P(wts))
            wid, wwt = router_ref(Pl.astype(np.float64), k, sigmoid_flag, extra)
            tag = "sigmoid" if sigmoid_flag else "softmax"
            check_exact(f"router_topk {tag} extra {extra:g}: ids ({t} tokens, top {k})", ids, wid)
            check(f"router_topk {tag} extra {extra:g}: weights", wts, wwt, TOL_FP32)

    # moe_route, decode block of 1 token and a block of 2: table with holes (-1), mailbox words checked
    for tt in (1, 2):
        table = np.arange(n, dtype=np.int32)
        table[rng.random(n) < 0.4] = -1
        nh = sh["nkv"] * 4  # h rows of the layer: any width works for the copy
        h = rng.standard_normal((nh, tt)).astype(np.float32)
        h_flat = np.ascontiguousarray(h.T.reshape(-1))  # token-major rows of nh
        kt = k * tt
        h_off = 32 + 3 * kt
        mb_words = h_off + nh * tt + 16
        seq = 7
        logits = (rng.standard_normal((tt, n)) * 2.0).astype(np.float32)
        ids = np.zeros((tt, k), dtype=np.int32)
        wts = np.zeros((tt, k), dtype=np.float32)
        slots = np.zeros((tt, k), dtype=np.int32)
        wh = np.zeros((tt, k), dtype=np.float32)
        pid = np.zeros(k, dtype=np.int32)
        mb = np.zeros(mb_words, dtype=np.uint32)
        call("scd_moe_route", P(logits), n, k, 1.0, P(table), n, 0, P(h_flat), nh, h_off, 30, seq, tt, P(mb), mb_words,
             P(ids), P(wts), P(slots), P(wh), P(pid))
        wid, wwt = router_ref(logits.astype(np.float64), k, 0, 1.0)
        check_exact(f"moe_route t {tt}: ids", ids, wid)
        check(f"moe_route t {tt}: weights", wts, wwt, TOL_FP32)
        want_slots = table[wid.astype(np.int64)]
        check_exact(f"moe_route t {tt}: slots = table[ids]", slots, want_slots)
        want_wh = np.where(want_slots >= 0, wwt, 0.0)
        check(f"moe_route t {tt}: wh (weights of hits)", wh, want_wh, TOL_FP32)
        check_exact(f"moe_route t {tt}: pid = last token's ids", pid, wid[-1].astype(np.int32))
        miss = bool(np.any(want_slots < 0))
        check_bool(f"moe_route t {tt}: a miss in this block (mailbox path taken)", miss, f"misses {int(np.sum(want_slots < 0))}")
        check_exact(f"moe_route t {tt}: mailbox token count word", mb[1:2], np.array([tt], dtype=np.uint32))
        o = np.arange(kt)
        check_exact(f"moe_route t {tt}: mailbox ids", mb[32 + o], ids.reshape(-1).astype(np.uint32))
        check_exact(f"moe_route t {tt}: mailbox weight bits", mb[32 + kt + o], f32_bits(wts.reshape(-1)))
        check_exact(f"moe_route t {tt}: mailbox slot words", mb[32 + 2 * kt + o],
                    want_slots.reshape(-1).astype(np.int32).view(np.uint32))
        if miss:
            check_exact(f"moe_route t {tt}: request flag (word 0 = seq)", mb[0:1], np.array([seq], dtype=np.uint32))
            check_exact(f"moe_route t {tt}: h copied to the mailbox", mb[h_off: h_off + nh * tt],
                        f32_bits(h_flat))
        else:
            check_exact(f"moe_route t {tt}: skip flag (word 2 = seq)", mb[2:3], np.array([seq], dtype=np.uint32))

    # moe_combine: t tokens, k slots each, the CPU result of a missed token from the mailbox, shared expert gated
    width = sh["n_embd"]
    tc = 2
    kc_ = k
    ye = (rng.standard_normal((tc, kc_, width)) * 0.3).astype(np.float32)
    x0 = rng.standard_normal((tc, width)).astype(np.float32)
    wgt = np.abs(rng.standard_normal((tc, kc_))).astype(np.float32)
    wgt /= wgt.sum(axis=1, keepdims=True)
    sh_v = (rng.standard_normal((tc, width)) * 0.5).astype(np.float32)
    sg_v = rng.standard_normal(tc).astype(np.float32)
    slots_c = np.zeros((tc, kc_), dtype=np.int32)
    slots_c[0, :] = 0
    slots_c[1, :] = 0
    slots_c[1, 2] = -1  # token 1 misses one slot
    y_off = 1024
    cpu = (rng.standard_normal((tc, width)) * 0.2).astype(np.float32)
    mb_c_words = y_off + tc * width + 64
    seq_c = 9
    for use_mb in (True, False):
        mbw = np.zeros(mb_c_words, dtype=np.uint32)
        mbw[16] = seq_c
        mbw[y_off: y_off + tc * width] = f32_bits(cpu.reshape(-1))
        x = np.ascontiguousarray(x0.copy())
        call("scd_moe_combine", P(x), P(ye), P(wgt), kc_, P(sh_v), P(sg_v), P(slots_c), width, tc,
             P(mbw) if use_mb else None, mb_c_words if use_mb else 0, y_off, 0, seq_c)
        ref = x0.astype(np.float64).copy()
        gate = sigmoid(sg_v.astype(np.float64))[:, None]
        for tk in range(tc):
            for j in range(kc_):
                ref[tk] += wgt[tk, j].astype(np.float64) * ye[tk, j].astype(np.float64)
            ref[tk] += gate[tk, 0] * sh_v[tk].astype(np.float64)
            if use_mb and np.any(slots_c[tk] < 0):
                ref[tk] += cpu[tk].astype(np.float64)
        label = "with the CPU part" if use_mb else "no mailbox (mb 0)"
        check(f"moe_combine ({tc} tokens, token 1 missed a slot), {label}", x, ref, TOL_FP32)

    # moe_combine_seq: prefill rows, inv picks rows of ye, no shared expert on one call and shared on the other
    tq = 4
    n_ye = tq * kc_
    yes = (rng.standard_normal((n_ye, width)) * 0.3).astype(np.float32)
    xs0 = rng.standard_normal((tq, width)).astype(np.float32)
    wsq = np.abs(rng.standard_normal((tq, kc_))).astype(np.float32)
    inv = rng.integers(0, n_ye, size=(tq, kc_)).astype(np.int32)
    shq = (rng.standard_normal((tq, width)) * 0.5).astype(np.float32)
    sgq = rng.standard_normal(tq).astype(np.float32)
    for with_sh in (True, False):
        xs = np.ascontiguousarray(xs0.copy())
        call("scd_moe_combine_seq", P(xs), P(yes), n_ye, P(wsq), P(inv), kc_,
             P(shq) if with_sh else None, P(sgq) if with_sh else None, width, tq)
        ref = xs0.astype(np.float64).copy()
        for tk in range(tq):
            for j in range(kc_):
                ref[tk] += wsq[tk, j].astype(np.float64) * yes[inv[tk, j]].astype(np.float64)
            if with_sh:
                ref[tk] += sigmoid(float(sgq[tk])) * shq[tk].astype(np.float64)
        label = "with the shared expert" if with_sh else "no shared expert"
        check(f"moe_combine_seq ({tq} rows, {kc_} slots), {label}", xs, ref, TOL_FP32)


def main() -> int:
    global LIB, np, torch, GGUFModelFile, dequant
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    if not sc_env.require("SC_MODEL_14B", "SC_MODEL_FLASHNEXT"):
        return 0
    try:
        import numpy as numpy_mod
        import torch as torch_mod
        from stratafreet.gguf_io import GGUFModelFile as gguf_model_file
        from stratafreet.quant import dequant as dequant_mod
    except Exception as e:
        return sc_env.skip(f"cannot import the StrataFreet oracle: {e}")
    np, torch, GGUFModelFile, dequant = numpy_mod, torch_mod, gguf_model_file, dequant_mod

    LIB = load_dll()
    print(f"device: {LIB.scd_device().decode(errors='replace')}")
    g14, shapes = model_shapes()
    for nm, s in shapes.items():
        print(f"{nm}: {s}")
    test_gemm(g14, shapes)
    test_norms(shapes["14B"])
    test_rope_kv_and_elementwise(shapes["14B"])
    test_attention("14B", shapes["14B"])
    test_attention("FlashNext", shapes["FlashNext"])
    test_routing("14B", shapes["14B"])
    test_routing("FlashNext", shapes["FlashNext"])
    print()
    print(f"{len(LINES)} checks, {len(FAILS)} failed")
    if FAILS:
        for f in FAILS:
            print(f"  FAILED: {f}")
        return 1
    print("test_backend_ops: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
