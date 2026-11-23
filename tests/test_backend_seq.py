"""W2-C part 2b acceptance: the C++ DeltaNet / conv, QSA, hyper-connection, PLE, per-row KV, moe_prefetch and graph
launches of CudaBackend against NumPy references written from the Rust CPU forward (model.rs), run through
build/strata_cpp_dev.dll (scd_*). Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata
License 1.0, see LICENSE-RustingStrata.md).

    python tests/test_backend_seq.py

References (float64 NumPy, from model.rs, cuda/mod.rs and kernels.cu):

  ssm_gates           beta = sigmoid(beta); g = softplus(g + dt_bias[h]) * a[h], h = i % nv (ops::softplus).
  conv_silu           model.rs deltanet: y = silu(sum_k state[k] w[k] + x w[kw-1]); the state rolls; the first nn
                      blocks of 128 channels are l2-normalized (ops::l2_norm: x / max(|x|, eps)).
  conv_seq            the same conv over t tokens in one launch, checked against t launches of conv_silu, and its
                      snapshots against the state after 1 .. min(nsnap, t-1) tokens.
  deltanet            model.rs deltanet recurrence per value head: S *= exp(g); sk = S k; d = beta (v - sk);
                      S += d k^T; o = (S q) / sqrt(dk). Heads: kh = vh % nk (or vh / (nv / nk) interleaved). Checked
                      over 8 tokens in two calls, so the recurrent state carries across launches; snapshots too.
  hc_silu/mix/write   model.rs hc_read / hc_write: silu(x sc); mixed = mean over streams of xn sigmoid(g);
                      stream c += out * 2 sigmoid(inj[c] / hc).
  ple_gate / ple_conv model.rs ple(): s = key . query / sqrt(n) per stream, gate = sigmoid(sign(s) sqrt(max(|s|,1e-6)));
                      r += gated + silu(kw[3] norm + sum_k<3 kw[k] row(p - 9 + 3k)), rows before 0 are zero.
  qsa_pool            model.rs qsa: mean of 4 raw indexer keys, RMS-normed with w, NEOX-roped at 4 b (complete blocks
                      only; the ring holds the raw keys of earlier positions).
  qsa_select          qsa: score_b = sum over the 4 heads of relu(q_h . k_b) / sqrt(128) for complete blocks; the top
                      top blocks (ties to the lower block) and the tail cells, in block order; dense while the row has
                      no more complete blocks than top. mw = 1: the same as bits of a mask row.
  rms_norm_rope_qk_rows   rms_norm then NEOX rope on the first n_rot dims at steps[2t]; K and V stored as f16 at
                      position steps[2t] of token t's own caches.
  attn_rows           softmax(q K^T / sqrt(hd)) V over positions 0 .. steps[2t] of token t's caches, GQA: head h reads
                      KV head h // (nh / nkv).
  moe_prefetch        a hint: the expert bytes are unchanged, and the call succeeds for resident and CPU slots.
  graph / beside      the replayed graph (3 replays with new inputs) equals the eager run bit for bit and the NumPy
                      reference; beside + beside_join orders a side-stream copy before the main stream's add.

Tolerance: relative error (max abs error / peak of the reference) <= 1e-3 for fp32 paths, <= 1e-2 where the kernel
reads or writes f16 caches. Selections and masks are compared exactly.

Shapes come from the GGUF headers (only the headers are read): the 14B test model (qwen35moe: DeltaNet 16 key heads,
32 value heads, state 128; 2048 wide) and Flash-Next (qwen4exp: 48 value heads, 4 hyper-connection streams, PLE width
160, QSA head dim 128 with 4 heads, 24 query heads over 2 KV heads, head_dim 256, n_rot 64).

Needs STRATAFREET_ROOT, SC_MODEL_14B, SC_MODEL_FLASHNEXT, and an NVIDIA GPU. Without the oracle or a
model the test prints SKIP and exits 0 (before any CUDA context exists).
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
GGUF_14B = sc_env.get("SC_MODEL_14B")
GGUF_FLASH = sc_env.get("SC_MODEL_FLASHNEXT")
TOL_FP32 = 1e-3
TOL_LOWP = 1e-2

# Imported in main() after the SKIP gate.
np = None
GGUFModelFile = None

vp = ctypes.c_void_p
ci = ctypes.c_int
cf = ctypes.c_float
cll = ctypes.c_longlong
cull = ctypes.c_ulonglong

FAILS: list[str] = []
LINES: list[str] = []


def load_dll():
    if not os.path.isfile(DLL):
        raise SystemExit(f"missing {DLL}; run build.bat first")
    lib = ctypes.CDLL(DLL)
    lib.sci_last_error.restype = ctypes.c_char_p
    lib.scd_device.restype = ctypes.c_char_p
    sigs = {
        "scd_ssm_gates": [vp, vp, vp, vp, ci, ci],
        "scd_conv_silu": [vp, vp, vp, vp, ci, ci, ci, cf],
        "scd_conv_seq": [vp, vp, vp, vp, ci, ci, ci, vp, ci],
        "scd_deltanet": [vp, ci, vp, vp, vp, vp, vp, vp, ci, ci, ci, ci, ci, cf, ci, vp, ci],
        "scd_hc_silu": [vp, cf, ci],
        "scd_hc_mix": [vp, vp, vp, ci, ci, ci],
        "scd_hc_write": [vp, vp, vp, ci, ci, ci, ci],
        "scd_ple_gate": [vp, vp, vp, vp, ci, ci, ci],
        "scd_ple_conv": [vp, vp, vp, vp, vp, ci, ci, ci],
        "scd_qsa_pool": [vp, vp, vp, vp, ci, ci, ci, cf, cf, ci],
        "scd_qsa_select": [vp, vp, ci, ci, ci, ci, ci, ci, ci, vp, vp],
        "scd_rms_norm_rope_qk_rows": [vp, ci, vp, vp, ci, vp, vp, vp, ci, vp, ci, cf, ci, vp, cf, ci, vp, vp, ci],
        "scd_attn_rows": [vp, ci, ci, ci, cf, vp, ci, vp, vp, ci, ci, vp],
        "scd_moe_prefetch": [vp, ci, vp, ci, vp, cll, ci, ci],
        "scd_graph_run": [vp, vp, vp, vp, ci, cf, ci, cull, vp, vp],
        "scd_graph_calls": [],
        "scd_beside_copy": [vp, vp, ci, vp],
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


def check(label: str, got, want, tol: float):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    peak = max(float(np.max(np.abs(want))), 1e-30)
    rel = float(np.max(np.abs(got - want))) / peak if got.size else 0.0
    ok = rel <= tol and bool(np.all(np.isfinite(got)))
    line = f"  {label:<60s} rel {rel:9.2e}  (tol {tol:.0e})  {'ok' if ok else 'FAIL'}"
    LINES.append(line)
    print(line)
    if not ok:
        FAILS.append(label)


def check_exact(label: str, got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    ok = got.shape == want.shape and bool(np.array_equal(got, want))
    mism = int(np.sum(got != want)) if got.shape == want.shape else -1
    line = f"  {label:<60s} exact, {mism} mismatches of {want.size}  {'ok' if ok else 'FAIL'}"
    LINES.append(line)
    print(line)
    if not ok:
        FAILS.append(label)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0.0, x)


def rms(x, w, eps):
    """ops::rms_norm over the last axis."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope_vec(x, n_rot, pos, base):
    """ops::rope_neox on the first n_rot dims of x (last axis), position pos (scalar or array broadcast to x's rows)."""
    y = np.array(x, dtype=np.float64, copy=True)
    half = n_rot // 2
    i = np.arange(half, dtype=np.float64)
    theta = np.asarray(pos, dtype=np.float64)[..., None] * base ** (-2.0 * i / n_rot)
    c, s = np.cos(theta), np.sin(theta)
    a, b = x[..., :half], x[..., half:n_rot]
    y[..., :half] = a * c - b * s
    y[..., half:n_rot] = a * s + b * c
    return y


# ---------------------------------------------------------------------------------------------------------------
# shapes from the GGUF headers

def model_shapes():
    g14 = GGUFModelFile(GGUF_14B)
    fl = GGUFModelFile(GGUF_FLASH)
    out = {}
    for name, g, pre in (("14B", g14, "qwen35moe"), ("FlashNext", fl, "qwen4exp")):
        d = dict(
            n_embd=int(g.get(f"{pre}.embedding_length")),
            nh=int(g.get(f"{pre}.attention.head_count")),
            nkv=int(g.get(f"{pre}.attention.head_count_kv")),
            hd=int(g.get(f"{pre}.attention.key_length")),
            n_rot=int(g.get(f"{pre}.rope.dimension_count")),
            base=float(g.get(f"{pre}.rope.freq_base")),
            eps=float(g.get(f"{pre}.attention.layer_norm_rms_epsilon")),
            dk=int(g.get(f"{pre}.ssm.state_size")),
            nk=int(g.get(f"{pre}.ssm.group_count")),
            nv=int(g.get(f"{pre}.ssm.time_step_rank")),
            d_inner=int(g.get(f"{pre}.ssm.inner_size")),
            kw=int(g.get(f"{pre}.ssm.conv_kernel")),
        )
        d["dv"] = d["d_inner"] // d["nv"]
        out[name] = d
    fx = out["FlashNext"]
    fx["hc"] = int(fl.get("qwen4exp.hyper_connection.count"))
    fx["n_ple"] = int(fl.get("qwen4exp.embedding_length_per_layer_input"))
    fx["qsa_top"] = int(fl.get("qwen4exp.attention.indexer.top_k"))
    fx["qsa_h"] = int(fl.get("qwen4exp.attention.indexer.head_count"))
    fx["qsa_d"] = int(fl.get("qwen4exp.attention.indexer.key_length"))
    return out


# ---------------------------------------------------------------------------------------------------------------
# DeltaNet: ssm_gates, conv_silu, conv_seq, deltanet

def test_ssm_gates(sh):
    print("ssm_gates (beta = sigmoid, g = softplus(g + dt) a)")
    rng = np.random.default_rng(21)
    for name in ("14B", "FlashNext"):
        nv, t = sh[name]["nv"], 5
        n = t * nv
        beta = f32(rng.normal(0, 1.5, n))
        g = f32(rng.normal(0, 1.0, n))
        dt = f32(rng.normal(0, 0.5, nv))
        a = f32(-np.abs(rng.normal(0.5, 0.2, nv)))
        b_out, g_out = beta.copy(), g.copy()
        call("scd_ssm_gates", P(b_out), P(g_out), P(dt), P(a), n, nv)
        h = np.arange(n) % nv
        check(f"{name} (nv {nv}) beta = sigmoid(beta)", b_out, sigmoid(beta.astype(np.float64)), TOL_FP32)
        check(f"{name} (nv {nv}) g = softplus(g + dt) * a", g_out, softplus(g.astype(np.float64) + dt[h]) * a[h], TOL_FP32)


def conv_step_ref(state, x, w, kw, nn, eps):
    """One token of model.rs deltanet's conv (state: (kw - 1) x ch, oldest first)."""
    acc = np.zeros(x.shape[0])
    for k in range(kw - 1):
        acc += state[k] * w[:, k]
    acc += x * w[:, kw - 1]
    y = silu(acc)
    if nn:
        blk = y[: nn * 128].reshape(nn, 128)
        blk *= 1.0 / np.maximum(np.sqrt((blk * blk).sum(axis=1, keepdims=True)), eps)
        y[: nn * 128] = blk.reshape(-1)
    new_state = np.vstack([state[1:], x[None, :]])
    return y, new_state


def test_conv(sh):
    print("conv_silu (one token, 3 steps) and conv_seq (6 tokens, snapshots) against the step-by-step reference")
    rng = np.random.default_rng(31)
    for name in ("14B", "FlashNext"):
        s = sh[name]
        nk, dk = s["nk"], s["dk"]
        ch = s["d_inner"] + 2 * nk * dk
        kw, eps = s["kw"], s["eps"]
        nn = 2 * nk  # the q and k heads of the token, one block of 128 channels each
        w = f32(rng.normal(0, 0.3, (ch, kw)))
        st0 = f32(rng.normal(0, 0.5, (kw - 1, ch)))
        xs = f32(rng.normal(0, 0.5, (3, ch)))
        # conv_silu, three steps with the state carried between launches
        st = st0.copy()
        ref_st = st0.astype(np.float64).copy()
        for step in range(3):
            y = np.zeros(ch, dtype=np.float32)
            call("scd_conv_silu", P(xs[step]), P(st), P(w), P(y), ch, kw, nn, eps)
            y_ref, ref_st = conv_step_ref(ref_st, xs[step].astype(np.float64), w.astype(np.float64), kw, nn, eps)
            check(f"{name} conv_silu step {step} (ch {ch}, l2 on {nn} blocks): y", y, y_ref, TOL_FP32)
        check(f"{name} conv_silu after 3 steps: state", st, ref_st, TOL_FP32)

        # conv_seq over 6 tokens in one launch vs 6 conv_silu launches (nn 0), and the snapshots
        t, nsnap = 6, 3
        xseq = f32(rng.normal(0, 0.5, (t, ch)))
        st_seq = st0.copy()
        y_seq = np.zeros((t, ch), dtype=np.float32)
        snap = np.zeros((nsnap, kw - 1, ch), dtype=np.float32)
        call("scd_conv_seq", P(xseq), P(st_seq), P(w), P(y_seq), ch, kw, t, P(snap), nsnap)
        st_step = st0.copy()
        y_steps = np.zeros((t, ch), dtype=np.float32)
        snap_ref = []
        for tok in range(t):
            y1 = np.zeros(ch, dtype=np.float32)
            call("scd_conv_silu", P(xseq[tok]), P(st_step), P(w), P(y1), ch, kw, 0, eps)
            y_steps[tok] = y1
            if tok < nsnap:
                snap_ref.append(st_step.copy())
        check(f"{name} conv_seq (t 6): y vs 6 conv_silu launches", y_seq, y_steps, TOL_FP32)
        check(f"{name} conv_seq: final state vs 6 conv_silu launches", st_seq, st_step, TOL_FP32)
        check(f"{name} conv_seq: snapshots after tokens 1 .. 3", snap, np.stack(snap_ref), TOL_FP32)


def deltanet_ref(y, S, beta, g, dt, a, nk, nv, ilv, dk, dv, scale):
    """model.rs deltanet over the rows of y (t x ych), with the state S (nv x dv x dk) carried. Returns o (t x nv x dv),
    the final state and the state after each token."""
    S = S.astype(np.float64).copy()
    t = y.shape[0]
    o = np.zeros((t, nv, dv))
    snaps = []
    for tok in range(t):
        yt = y[tok].astype(np.float64)
        q = yt[: nk * dk].reshape(nk, dk)
        k = yt[nk * dk: 2 * nk * dk].reshape(nk, dk)
        v = yt[2 * nk * dk: 2 * nk * dk + nv * dv].reshape(nv, dv)
        for vh in range(nv):
            kh = vh // (nv // nk) if ilv else vh % nk
            bt, gt = float(beta[tok, vh]), float(g[tok, vh])
            if dt is not None:
                bt = float(sigmoid(bt))
                gt = float(softplus(gt + dt[vh]) * a[vh])
            Sh = S[vh] * np.exp(gt)
            sk = Sh @ k[kh]
            d = bt * (v[vh] - sk)
            Sh = Sh + np.outer(d, k[kh])
            S[vh] = Sh
            o[tok, vh] = (Sh @ q[kh]) * scale
        snaps.append(S.copy())
    return o, S, snaps


def test_deltanet(sh):
    print("deltanet (state carried over two launches of 4 tokens; raw gates with dt / a; snapshots)")
    rng = np.random.default_rng(41)
    for name in ("14B", "FlashNext"):
        s = sh[name]
        nk, nv, dk, dv = s["nk"], s["nv"], s["dk"], s["dv"]
        ych = 2 * nk * dk + nv * dv
        scale = 1.0 / np.sqrt(dk)
        for ilv in (0, 1):
            raw = ilv == 0  # the raw-gate variant runs once per model; the other interleave with gates applied
            t_total, t1 = 8, 4
            y = f32(rng.normal(0, 0.2, (t_total, ych)))
            S0 = f32(rng.normal(0, 0.05, (nv, dv, dk)))
            if raw:
                beta = f32(rng.normal(0, 1.0, (t_total, nv)))
                g = f32(rng.normal(0, 0.5, (t_total, nv)))
                dt = f32(rng.normal(0, 0.5, nv))
                a = f32(-np.abs(rng.normal(0.5, 0.2, nv)))
            else:
                beta = f32(sigmoid(rng.normal(0, 1.0, (t_total, nv))))
                g = f32(-np.abs(rng.normal(0.3, 0.2, (t_total, nv))))
                dt, a = None, None
            o_ref, S_ref, snaps_ref = deltanet_ref(y, S0, beta, g, dt, a, nk, nv, ilv, dk, dv, scale)
            S = S0.copy()
            o = np.zeros((t_total, nv, dv), dtype=np.float32)
            nsnap = 2
            snap = np.zeros((nsnap, nv, dv, dk), dtype=np.float32)
            beta_c, g_c = f32(beta.copy()), f32(g.copy())
            # the first launch: 4 tokens, snapshots of its first two states
            call("scd_deltanet", P(y[:t1]), ych, P(S), P(beta_c[:t1]), P(g_c[:t1]), P(dt), P(a), P(o[:t1]), nk, nv, ilv,
                 dk, dv, scale, t1, P(snap), nsnap)
            # the second launch: 4 tokens, the state of the first launch carried in S
            call("scd_deltanet", P(y[t1:]), ych, P(S), P(beta_c[t1:]), P(g_c[t1:]), P(dt), P(a), P(o[t1:]), nk, nv, ilv,
                 dk, dv, scale, t_total - t1, None, 0)
            tag = f"{name} interleave {ilv} ({'raw gates' if raw else 'gates applied'})"
            check(f"{tag}: o over 8 tokens (2 launches)", o, o_ref, TOL_FP32)
            check(f"{tag}: final state", S, S_ref, TOL_FP32)
            check(f"{tag}: snapshots after tokens 1 and 2", snap, np.stack(snaps_ref[:nsnap]), TOL_FP32)


# ---------------------------------------------------------------------------------------------------------------
# hyper-connections and PLE (Flash-Next)

def test_hc(sh):
    print("hc_silu, hc_mix, hc_write (qwen4exp hyper-connections)")
    rng = np.random.default_rng(51)
    s = sh["FlashNext"]
    n, hc, nt = s["n_embd"], s["hc"], 3
    x = f32(rng.normal(0, 1.0, n))
    sc = 1.7
    out = x.copy()
    call("scd_hc_silu", P(out), sc, n)
    check("hc_silu: x = silu(x sc)", out, silu(x.astype(np.float64) * sc), TOL_FP32)

    xn = f32(rng.normal(0, 1.0, (nt, hc, n)))
    g = f32(rng.normal(0, 1.0, (nt, hc, n)))
    mixed = np.zeros((nt, n), dtype=np.float32)
    call("scd_hc_mix", P(xn), P(g), P(mixed), n, hc, nt)
    ref = (xn.astype(np.float64) * sigmoid(g.astype(np.float64))).sum(axis=1) / hc
    check(f"hc_mix ({nt} tokens, {hc} streams of {n})", mixed, ref, TOL_FP32)

    r = f32(rng.normal(0, 1.0, (nt, hc, n)))
    o_in = f32(rng.normal(0, 1.0, (nt, n)))
    inj = f32(rng.normal(0, 1.0, (nt, hc)))
    r0 = r.copy()
    o_w = o_in.copy()
    call("scd_hc_write", P(r), P(o_w), P(inj), n, hc, nt, 0)
    ref_r = r0.astype(np.float64) + o_in[:, None, :].astype(np.float64) * (2.0 * sigmoid(inj.astype(np.float64) / hc))[:, :, None]
    check("hc_write: r += out * 2 sigmoid(inj / hc)", r, ref_r, TOL_FP32)
    check("hc_write without clear: out unchanged", o_w, o_in, 0.0)
    r2 = r0.copy()
    o_c = o_in.copy()
    call("scd_hc_write", P(r2), P(o_c), P(inj), n, hc, nt, 1)
    check("hc_write with clear: r as without clear", r2, ref_r, TOL_FP32)
    check_exact("hc_write with clear: out zeroed", o_c, np.zeros_like(o_c))


def test_ple(sh):
    print("ple_gate, ple_conv (Flash-Next per-layer embedding, 2 launches: state in the ring carried)")
    rng = np.random.default_rng(61)
    s = sh["FlashNext"]
    n, hc = s["n_embd"], s["hc"]
    d = hc * n
    kw_ = s["kw"]
    nt1, nt2 = 3, 2
    nt = nt1 + nt2
    key = f32(rng.normal(0, 1.0, (nt, d)))
    query = f32(rng.normal(0, 1.0, (nt, d)))
    value = f32(rng.normal(0, 1.0, (nt, n)))
    gated = np.zeros((nt, d), dtype=np.float32)
    # ple_gate on the rows of each stream (n-wide blocks), one token per row of hc streams
    call("scd_ple_gate", P(key), P(query), P(value), P(gated), n, hc, nt)
    kk = key.astype(np.float64).reshape(nt, hc, n)
    qq = query.astype(np.float64).reshape(nt, hc, n)
    sc = (kk * qq).sum(axis=2) / np.sqrt(n)
    gate = sigmoid(np.sign(sc) * np.sqrt(np.maximum(np.abs(sc), 1e-6)))
    ref_gated = (value.astype(np.float64)[:, None, :] * gate[:, :, None]).reshape(nt, d)
    check(f"ple_gate ({nt} tokens, {hc} streams of {n})", gated, ref_gated, TOL_FP32)

    kwt = f32(rng.normal(0, 0.4, (d, kw_)))
    r = f32(rng.normal(0, 0.5, (nt, d)))
    norm = f32(rng.normal(0, 1.0, (nt, d)))
    ring = np.zeros((16, d), dtype=np.float32)
    r_ref = r.astype(np.float64).copy()
    hist: dict[int, np.ndarray] = {}
    kwd = kwt.astype(np.float64)
    # reference: rows of earlier positions (none before 0), updated token by token
    for p in range(nt):
        acc = kwd[:, 3] * norm[p].astype(np.float64)
        for k in range(3):
            q_pos = p - 9 + 3 * k
            if q_pos >= 0:
                acc = acc + kwd[:, k] * hist[q_pos]
        r_ref[p] = r_ref[p] + gated[p].astype(np.float64) + silu(acc)
        hist[p] = norm[p].astype(np.float64)
    r_gpu = r.copy()
    r_gpu_split = r_gpu.copy()
    ring_gpu = ring.copy()
    call("scd_ple_conv", P(r_gpu_split[:nt1]), P(gated[:nt1]), P(norm[:nt1]), P(kwt), P(ring_gpu), 0, d, nt1)
    call("scd_ple_conv", P(r_gpu_split[nt1:]), P(gated[nt1:]), P(norm[nt1:]), P(kwt), P(ring_gpu), nt1, d, nt2)
    check("ple_conv: r += gated + silu(conv), 2 launches (positions 0-2, then 3-4)", r_gpu_split, r_ref, TOL_FP32)
    # the ring holds the normalized rows of the last 16 positions, slot = position % 16
    ring_ref = np.zeros((16, d))
    for p in range(nt):
        ring_ref[p & 15] = norm[p].astype(np.float64)
    check("ple_conv: ring holds the normalized rows (slot = position % 16)", ring_gpu[: nt], ring_ref[: nt], TOL_FP32)


# ---------------------------------------------------------------------------------------------------------------
# QSA (Flash-Next sparse attention)

def qsa_pool_ref(ik_all, w, n_rot, base, eps, nbs, computed):
    """model.rs qsa pooling: block b = mean of raw keys 4b .. 4b+3, rms-normed with w, roped at 4b."""
    pool = np.zeros((nbs, ik_all.shape[1]))
    for b in computed:
        mean = ik_all[4 * b: 4 * b + 4].mean(axis=0)
        normed = rms(mean, w, eps)
        pool[b] = rope_vec(normed, n_rot, 4 * b, base)
    return pool


def test_qsa(sh):
    print("qsa_pool (2 launches, ring carried) and qsa_select (rows at positions 9 .. 15, top 2; sel and mask forms)")
    rng = np.random.default_rng(71)
    s = sh["FlashNext"]
    D, H = s["qsa_d"], s["qsa_h"]
    n_rot, base, eps = s["n_rot"], s["base"], s["eps"]
    assert D == 128 and H == 4, (D, H)
    nbs = 4
    w = f32(rng.normal(1.0, 0.1, D))
    ik = f32(rng.normal(0, 1.0, (16, D)))
    ring = np.zeros((32, D), dtype=np.float32)
    pool = np.zeros((nbs, D), dtype=np.float32)
    call("scd_qsa_pool", P(ik[:10]), P(ring), P(pool), P(w), 0, 10, n_rot, base, eps, nbs)
    call("scd_qsa_pool", P(ik[10:16]), P(ring), P(pool), P(w), 10, 6, n_rot, base, eps, nbs)
    pool_ref = qsa_pool_ref(ik.astype(np.float64), w.astype(np.float64), n_rot, base, eps, nbs, range(4))
    check("qsa_pool: blocks 0 .. 3 (complete blocks over two launches)", pool, pool_ref, TOL_FP32)
    ring_ref = np.zeros((32, D))
    for p in range(16):
        ring_ref[p & 31] = ik[p].astype(np.float64)
    check("qsa_pool: ring holds the raw keys of the last 32 positions", ring[:16], ring_ref[:16], TOL_FP32)

    # queries: 7 rows at positions 9 .. 15, normed and roped per head (the values only matter for the scores)
    rows, step0, top = 7, 9, 2
    qv = f32(rng.normal(0, 1.0, (rows, H, D)))
    sst = 1 + 4 * top + 4
    for mw in (0, 1):
        sel = np.zeros((rows, sst), dtype=np.int32)
        mask = np.zeros((rows, max(mw, 1)), dtype=np.uint32)
        call("scd_qsa_select", P(qv), P(pool), step0, 0, rows, nbs, top, sst, mw, P(sel), P(mask))
        if mw == 0:
            ok_all = True
            for r in range(rows):
                pos = step0 + r
                nb = (pos + 1) // 4
                if nb <= top:
                    want_count = -1
                    if sel[r, 0] != -1:
                        ok_all = False
                    continue
                sc = np.array([sum(max(float(qv[r, h].astype(np.float64) @ pool[b].astype(np.float64)), 0.0) for h in range(H))
                               for b in range(nb)]) / np.sqrt(D)
                order = sorted(range(nb), key=lambda b: (-sc[b], b))[:top]
                blocks = sorted(order)
                cells = [4 * b + m for b in blocks for m in range(4)] + list(range(4 * nb, pos + 1))
                want_count = len(cells)
                got = list(sel[r, 1: 1 + want_count])
                if sel[r, 0] != want_count or got != cells:
                    ok_all = False
                    print(f"    row {r} (pos {pos}): got {list(sel[r])} want count {want_count} cells {cells}")
            check_bool_line("qsa_select sel: dense rows (-1), selected blocks + tail cells (ids exact)", ok_all)
        else:
            ok_all = True
            for r in range(rows):
                pos = step0 + r
                nb = (pos + 1) // 4
                if nb <= top:
                    if int(mask[r, 0]) != 0xFFFFFFFF:
                        ok_all = False
                    continue
                sc = np.array([sum(max(float(qv[r, h].astype(np.float64) @ pool[b].astype(np.float64)), 0.0) for h in range(H))
                               for b in range(nb)]) / np.sqrt(D)
                order = sorted(range(nb), key=lambda b: (-sc[b], b))[:top]
                bits = 0
                for b in order:
                    bits |= 1 << b
                bits |= 1 << nb
                if int(mask[r, 0]) != bits:
                    ok_all = False
                    print(f"    row {r} (pos {pos}): mask {int(mask[r, 0]):#x} want {bits:#x}")
            check_bool_line("qsa_select mask: dense rows (all bits), selected blocks + tail bit (exact)", ok_all)


def check_bool_line(label: str, ok: bool):
    line = f"  {label:<60s} {'ok' if ok else 'FAIL'}"
    LINES.append(line)
    print(line)
    if not ok:
        FAILS.append(label)


# ---------------------------------------------------------------------------------------------------------------
# per-row KV kernels and attention (Flash-Next attention shape: 24 heads over 2, head_dim 256)

def test_rows(sh):
    print("rms_norm_rope_qk_rows and attn_rows (3 tokens, each with its own cache and position)")
    rng = np.random.default_rng(81)
    s = sh["FlashNext"]
    nh, nkv, hd, n_rot, base, eps = s["nh"], s["nkv"], s["hd"], s["n_rot"], s["base"], s["eps"]
    nt = 3
    steps = np.array([5, 17, 2], dtype=np.int32)
    steps2 = np.zeros(2 * nt, dtype=np.int32)
    steps2[0::2] = steps
    cache_len = 32 * nkv * hd
    q = f32(rng.normal(0, 1.0, (nt * nh, hd)))
    qw = f32(rng.normal(1.0, 0.1, hd))
    k = f32(rng.normal(0, 1.0, (nt * nkv, hd)))
    kw = f32(rng.normal(1.0, 0.1, hd))
    v = f32(rng.normal(0, 1.0, (nt * nkv, hd)))
    # the caches hold history at every position (random f16); the kernel writes position steps[t] of token t
    kc = np.ascontiguousarray(u16_of_f16(rng.normal(0, 0.5, (nt, cache_len))))
    vc = np.ascontiguousarray(u16_of_f16(rng.normal(0, 0.5, (nt, cache_len))))
    qo = np.zeros((nt * nh, hd), dtype=np.float32)
    ko = np.zeros((nt * nkv, hd), dtype=np.float32)
    call("scd_rms_norm_rope_qk_rows", P(q), hd, P(qw), P(qo), nh, P(k), P(kw), P(ko), nkv, P(v), hd, eps, n_rot, P(steps2),
         base, nt, P(kc), P(vc), cache_len)

    # references
    qn = np.zeros((nt * nh, hd))
    kn = np.zeros((nt * nkv, hd))
    for t in range(nt):
        pos = int(steps[t])
        qn[t * nh:(t + 1) * nh] = rope_vec(rms(q[t * nh:(t + 1) * nh].astype(np.float64), qw.astype(np.float64), eps), n_rot, pos, base)
        kn[t * nkv:(t + 1) * nkv] = rope_vec(rms(k[t * nkv:(t + 1) * nkv].astype(np.float64), kw.astype(np.float64), eps), n_rot, pos, base)
    check("rms_norm_rope_qk_rows: q rows (norm, rope)", qo, qn, TOL_FP32)
    check("rms_norm_rope_qk_rows: k rows (norm, rope)", ko, kn, TOL_FP32)
    kc_ref = f16_back(kc.copy())
    vc_ref = f16_back(vc.copy())
    for t in range(nt):
        pos = int(steps[t])
        base_e = pos * nkv * hd
        kc_ref[t, base_e: base_e + nkv * hd] = f16_back(u16_of_f16(kn[t * nkv:(t + 1) * nkv].reshape(-1)))
        vc_ref[t, base_e: base_e + nkv * hd] = f16_back(u16_of_f16(v[t * nkv:(t + 1) * nkv].reshape(-1)))
    kc_gpu = f16_back(kc)
    vc_gpu = f16_back(vc)
    used = np.zeros((nt, cache_len), dtype=bool)
    for t in range(nt):
        used[t, int(steps[t]) * nkv * hd: (int(steps[t]) + 1) * nkv * hd] = True
    check("rms_norm_rope_qk_rows: K caches at the new positions (f16)", kc_gpu[used], kc_ref[used], TOL_LOWP)
    check("rms_norm_rope_qk_rows: V caches at the new positions (f16)", vc_gpu[used], vc_ref[used], TOL_LOWP)
    check_exact("rms_norm_rope_qk_rows: history positions untouched", kc_gpu[~used], f16_back(kc)[~used])

    # attn_rows over the caches the kernel just wrote, with the q rows of the reference (fp32 inputs)
    qa = f32(qn)
    out = np.zeros((nt, nh, hd), dtype=np.float32)
    splits = 16
    scale = 1.0 / np.sqrt(hd)
    call("scd_attn_rows", P(qa), nh, nkv, hd, scale, P(steps2), nt, P(kc), P(vc), cache_len, splits, P(out))
    ref = np.zeros((nt, nh, hd))
    grp = nh // nkv
    for t in range(nt):
        pos = int(steps[t])
        K = kc_gpu[t].reshape(cache_len // (nkv * hd), nkv, hd)[: pos + 1]
        V = vc_gpu[t].reshape(cache_len // (nkv * hd), nkv, hd)[: pos + 1]
        for h in range(nh):
            kvh = h // grp
            sc = K[:, kvh, :] @ qn[t * nh + h] * scale
            p = np.exp(sc - sc.max())
            p /= p.sum()
            ref[t, h] = p @ V[:, kvh, :]
    check(f"attn_rows (nt {nt}, positions {steps.tolist()}, {nh}/{nkv} heads, hd {hd})", out, ref, TOL_LOWP)


def test_prefetch():
    print("moe_prefetch (a hint: the expert bytes stay as they are)")
    rng = np.random.default_rng(91)
    n_table, stride, bytes_ = 8, 8192, 4096
    base = rng.integers(0, 255, n_table * stride, dtype=np.uint8)
    before = base.copy()
    ids = np.array([3, 0, 7], dtype=np.int32)
    table = np.array([0, 1, -1, 2, 3, 4, 5, 6], dtype=np.int32)  # expert 2 is on the CPU (-1)
    call("scd_moe_prefetch", P(ids), 3, P(table), n_table, P(base), base.nbytes, stride, bytes_)
    check_exact("moe_prefetch: expert bytes unchanged after the hint (3 experts, one on the CPU slot)", base, before)


def test_graph_and_beside(sh):
    print("graph (capture, instantiate, 3 replays with new inputs) and beside + beside_join")
    rng = np.random.default_rng(101)
    n = sh["FlashNext"]["n_embd"]
    eps = sh["FlashNext"]["eps"]
    nrep = 4
    x = f32(rng.normal(0, 1.0, (nrep, n)))
    w = f32(rng.normal(1.0, 0.1, n))
    y = f32(rng.normal(0, 0.5, n))
    u = f32(rng.normal(0, 1.0, n))
    out_g = np.zeros((nrep, n), dtype=np.float32)
    out_e = np.zeros((nrep, n), dtype=np.float32)
    key = 0x5EED0001
    call("scd_graph_run", P(x), P(w), P(y), P(u), n, eps, nrep, key, P(out_g), P(out_e))
    xr = x.astype(np.float64)
    h = xr / np.sqrt(np.mean(xr * xr, axis=1, keepdims=True) + eps) * w.astype(np.float64)
    ref = silu(h + y.astype(np.float64)) * u.astype(np.float64)
    check(f"graph: {nrep} runs (capture, then 3 replays with new x) vs NumPy", out_g, ref, TOL_FP32)
    check(f"eager: the same {nrep} inputs vs NumPy", out_e, ref, TOL_FP32)
    check_exact("graph replays equal the eager run bit for bit", out_g, out_e)
    # the op function runs once to capture and once per eager input; the 3 replays do not run it
    calls = int(LIB.scd_graph_calls())
    check_bool_line(f"graph captured once: op function ran {calls} times (1 capture + {nrep} eager = {nrep + 1})", calls == nrep + 1)

    z = f32(rng.normal(0, 1.0, n))
    xb = f32(rng.normal(0, 1.0, n))
    ob = np.zeros(n, dtype=np.float32)
    call("scd_beside_copy", P(xb), P(z), n, P(ob))
    check_exact("beside copy, beside_join, then add: out = x + z", ob, (xb + z).astype(np.float32))


def main() -> int:
    global LIB, np, GGUFModelFile
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    if not sc_env.require("SC_MODEL_14B", "SC_MODEL_FLASHNEXT"):
        return 0
    try:
        import numpy as numpy_mod
        from stratafreet.gguf_io import GGUFModelFile as gguf_model_file
    except Exception as e:
        return sc_env.skip(f"cannot import the StrataFreet oracle: {e}")
    np, GGUFModelFile = numpy_mod, gguf_model_file

    LIB = load_dll()
    print(f"device: {LIB.scd_device().decode(errors='replace')}")
    sh = model_shapes()
    for nm, s in sh.items():
        print(f"{nm}: {s}")
    test_ssm_gates(sh)
    test_conv(sh)
    test_deltanet(sh)
    test_hc(sh)
    test_ple(sh)
    test_qsa(sh)
    test_rows(sh)
    test_prefetch()
    test_graph_and_beside(sh)
    print()
    print(f"{len(LINES)} checks, {len(FAILS)} failed")
    if FAILS:
        for f in FAILS:
            print(f"  FAILED: {f}")
        return 1
    print("test_backend_seq: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
