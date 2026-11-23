"""test_glm_cpu.py - the GLM-5.3-Flash (glm5-next) port on the CPU, against an independent numpy forward.

No model file, no GPU, no CUDA context: the test writes a tiny synthetic glm5-next GGUF (n_embd 64, 4 decoding layers
- KDA + dense FFN, DSA + MoE, KDA + MoE, DSA + MoE - plus a NextN block, F32 weights, about 45 MB) into
build\\glm_test\\ and drives build\\strata_cpu_test.dll (build_cpu.bat), which opens the model with the GPU path off.

1. Unit checks of the ported helpers against numpy: hc_coeffs (Sinkhorn), hc_post, route_sigmoid (including the
   Rust unit test's case), l2_norm (including the Rust unit test), swiglu_clamp, the f16 rounding and q8_roundtrip of
   deepseek/mat.rs, and Mat::matvec on Q8_0 weights (the q8_roundtrip path below 512 rows, the integer Q8_K path at
   512 rows and more).
2. GlmConfig: from_gguf on both arch spellings (glm5-next, glm5next), validate() passes, and refuses a file with an
   extra tensor.
3. Forward: 20 tokens one at a time (the k-pool indexer takes over after idx_top_k = 8 rows), every position's
   logits against the float64 numpy forward written from llama.cpp src/models/glm5-next.cpp (not from the Rust
   code); the chunked layer-major prefill (default chunk and STRATA_GLM_CHUNK=3) against the token-by-token path
   (bit-identical on the CPU path); reset; the context limit.

Usage: python tests/test_glm_cpu.py

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).
"""
import ctypes
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

BUILD = sc_env.build_dir(PORT)
DLL = os.path.join(BUILD, "strata_cpu_test.dll")
WORK = os.path.join(BUILD, "glm_test")

FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


# ---------------------------------------------------------------- GGUF writer (v3, F32 tensors)

def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key, val):
    out = _str(key)
    if isinstance(val, bool):
        return out + struct.pack("<I?", 7, val)
    if isinstance(val, int):
        return out + struct.pack("<II", 4, val)
    if isinstance(val, float):
        return out + struct.pack("<If", 6, val)
    if isinstance(val, str):
        return out + struct.pack("<I", 8) + _str(val)
    if isinstance(val, list):
        if all(isinstance(v, int) and not isinstance(v, bool) for v in val):
            return out + struct.pack("<IIQ", 9, 4, len(val)) + b"".join(struct.pack("<I", v) for v in val)
        if all(isinstance(v, float) for v in val):
            return out + struct.pack("<IIQ", 9, 6, len(val)) + b"".join(struct.pack("<f", v) for v in val)
    raise TypeError(key)


def write_gguf(path, meta, tensors):
    """tensors: list of (name, ggml dims fastest first, float32 array in row-major numpy order)."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    head += b"".join(_kv(k, v) for k, v in meta.items())
    off = 0
    infos, blobs = [], []
    for name, dims, arr in tensors:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        assert a.size == int(np.prod(dims)), name
        infos.append(_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
                     + struct.pack("<IQ", 0, off))
        blobs.append(a.tobytes())
        off += (a.nbytes + 31) // 32 * 32
    head += b"".join(infos)
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\0" * pad)
        for b in blobs:
            f.write(b + b"\0" * ((-len(b)) % 32))


# ---------------------------------------------------------------- synthetic model

C = dict(n_embd=64, n_head=64, hd=128, conv_k=4, lb=-5.0, q_lora=32, kv_lora=32, k_mla=16, v_mla=16,
         ix_heads=4, ix_dim=16, top_k=8, kpool=4, n_expert=8, n_used=3, ffe=32, n_ff=48, n_dense=1,
         kinds=[0, 1, 0, 1, 1], n_nextn=1, vocab=96, eps=1e-5, kda_eps=1e-6, hc_eps=1e-6, hc_iters=20,
         scale=2.5, clamp=1.5)


def make_model(seed=7):
    rng = np.random.default_rng(seed)
    c = C
    e, di, hd = c["n_embd"], c["n_head"] * c["hd"], c["hd"]

    def w(rows, cols, s=1.0):
        return (rng.standard_normal((rows, cols)) * s / np.sqrt(cols)).astype(np.float32)

    def vec(n, base=1.0, s=0.1):
        return (base + rng.standard_normal(n) * s).astype(np.float32)

    T = {}
    T["token_embd.weight"] = ([e, c["vocab"]], (rng.standard_normal((c["vocab"], e))).astype(np.float32))
    T["output_norm.weight"] = ([e], vec(e))
    T["output.weight"] = ([e, c["vocab"]], w(c["vocab"], e, 2.0))
    n_file = len(c["kinds"])
    n_layer = n_file - c["n_nextn"]
    for il, kind in enumerate(c["kinds"]):
        b = f"blk.{il}."

        def put(n, dims, arr):
            T[b + n] = (dims, arr)

        put("attn_norm.weight", [e], vec(e))
        put("ffn_norm.weight", [e], vec(e))
        if il < n_layer:
            for s in ("attn", "ffn"):
                put(f"hc_{s}_fn.weight", [4 * e, 24], w(24, 4 * e, 1.5))
                put(f"hc_{s}_base.weight", [24], (rng.standard_normal(24) * 0.5).astype(np.float32))
                put(f"hc_{s}_scale.weight", [3], (0.5 + rng.random(3)).astype(np.float32))
        if kind == 0:
            for n in ("attn_q", "attn_k", "attn_v"):
                put(n + ".weight", [e, di], w(di, e, 2.0))
            put("attn_output.weight", [di, e], w(e, di, 1.0))
            put("ssm_a", [c["n_head"]], (-np.exp(rng.standard_normal(c["n_head"]) * 0.5)).astype(np.float32))
            put("ssm_beta.weight", [e, c["n_head"]], w(c["n_head"], e, 1.0))
            for s in "qkv":
                put(f"ssm_conv1d_{s}.weight", [c["conv_k"], 1, di],
                    (rng.standard_normal((di, 1, c["conv_k"])) * 0.6).astype(np.float32))
            put("ssm_dt.bias", [di], (rng.standard_normal(di) * 0.5).astype(np.float32))
            put("ssm_f_a.weight", [e, hd], w(hd, e))
            put("ssm_f_b.weight", [hd, di], w(di, hd, 2.0))
            put("ssm_g_a.weight", [e, hd], w(hd, e))
            put("ssm_g_b.weight", [hd, di], w(di, hd, 2.0))
            put("ssm_norm.weight", [hd], vec(hd))
        else:
            nh, kd, kl, vd = c["n_head"], c["k_mla"], c["kv_lora"], c["v_mla"]
            ni, nih = c["ix_dim"], c["ix_heads"]
            put("attn_q_a.weight", [e, c["q_lora"]], w(c["q_lora"], e))
            put("attn_q_a_norm.weight", [c["q_lora"]], vec(c["q_lora"]))
            put("attn_q_b.weight", [c["q_lora"], nh * kd], w(nh * kd, c["q_lora"], 2.0))
            put("attn_kv_a_mqa.weight", [e, kl], w(kl, e))
            put("attn_kv_a_norm.weight", [kl], vec(kl))
            put("attn_k_b.weight", [kd, kl, nh], (rng.standard_normal((nh, kl, kd)) * 2 / np.sqrt(kd)).astype(np.float32))
            put("attn_v_b.weight", [kl, vd, nh], (rng.standard_normal((nh, vd, kl)) / np.sqrt(kl)).astype(np.float32))
            put("attn_output.weight", [nh * vd, e], w(e, nh * vd))
            put("indexer.attn_k.weight", [e, ni], w(ni, e))
            put("indexer.attn_q_b.weight", [c["q_lora"], nih * ni], w(nih * ni, c["q_lora"], 2.0))
            put("indexer.k_norm.weight", [ni], vec(ni))
            put("indexer.k_norm.bias", [ni], (rng.standard_normal(ni) * 0.1).astype(np.float32))
            put("indexer.proj.weight", [e, nih], w(nih, e))
            put("indexer_compressor_ape.weight", [ni, c["kpool"]], (rng.standard_normal((c["kpool"], ni)) * 0.5).astype(np.float32))
            put("indexer_compressor_gate.weight", [e, ni], w(ni, e))
        if il < c["n_dense"]:
            put("ffn_gate.weight", [e, c["n_ff"]], w(c["n_ff"], e, 2.0))
            put("ffn_up.weight", [e, c["n_ff"]], w(c["n_ff"], e, 2.0))
            put("ffn_down.weight", [c["n_ff"], e], w(e, c["n_ff"]))
        else:
            ne, ffe = c["n_expert"], c["ffe"]
            put("ffn_gate_inp.weight", [e, ne], w(ne, e, 2.0))
            put("exp_probs_b.bias", [ne], (rng.standard_normal(ne) * 0.05).astype(np.float32))
            put("ffn_gate_exps.weight", [e, ffe, ne], (rng.standard_normal((ne, ffe, e)) * 3 / np.sqrt(e)).astype(np.float32))
            put("ffn_up_exps.weight", [e, ffe, ne], (rng.standard_normal((ne, ffe, e)) * 3 / np.sqrt(e)).astype(np.float32))
            put("ffn_down_exps.weight", [ffe, e, ne], (rng.standard_normal((ne, e, ffe)) / np.sqrt(ffe)).astype(np.float32))
            put("ffn_gate_shexp.weight", [e, ffe], w(ffe, e, 2.0))
            put("ffn_up_shexp.weight", [e, ffe], w(ffe, e, 2.0))
            put("ffn_down_shexp.weight", [ffe, e], w(e, ffe))
        if il >= n_layer:
            put("nextn.eh_proj.weight", [2 * e, e], w(e, 2 * e))
            put("nextn.enorm.weight", [e], vec(e))
            put("nextn.hnorm.weight", [e], vec(e))
            put("nextn.shared_head_norm.weight", [e], vec(e))
    return T


def meta_for(arch):
    c = C
    n_file = len(c["kinds"])
    p = arch + "."
    return {
        "general.architecture": arch,
        "general.alignment": 32,
        p + "block_count": n_file,
        p + "context_length": 4096,
        p + "embedding_length": c["n_embd"],
        p + "feed_forward_length": c["n_ff"],
        p + "attention.head_count": c["n_head"],
        p + "attention.head_count_kv": list(c["kinds"]),
        p + "attention.layer_norm_rms_epsilon": c["eps"],
        p + "attention.layer_norm_epsilon": c["kda_eps"],
        p + "attention.q_lora_rank": c["q_lora"],
        p + "attention.kv_lora_rank": c["kv_lora"],
        p + "attention.key_length_mla": c["k_mla"],
        p + "attention.value_length_mla": c["v_mla"],
        p + "attention.indexer.head_count": c["ix_heads"],
        p + "attention.indexer.key_length": c["ix_dim"],
        p + "attention.indexer.top_k": c["top_k"],
        p + "attention.indexer.kpool": c["kpool"],
        p + "ssm.conv_kernel": c["conv_k"],
        p + "kda.head_dim": c["hd"],
        p + "kda.gate_lower_bound": c["lb"],
        p + "hyper_connection.count": 4,
        p + "hyper_connection.sinkhorn_iterations": c["hc_iters"],
        p + "hyper_connection.epsilon": c["hc_eps"],
        p + "expert_count": c["n_expert"],
        p + "expert_used_count": c["n_used"],
        p + "expert_gating_func": 2,
        p + "expert_feed_forward_length": c["ffe"],
        p + "expert_shared_feed_forward_length": c["ffe"],
        p + "expert_shared_count": 1,
        p + "leading_dense_block_count": c["n_dense"],
        p + "expert_weights_scale": c["scale"],
        p + "expert_weights_norm": True,
        p + "swiglu_clamp_exp": [c["clamp"]] * n_file,
        p + "swiglu_clamp_shexp": [c["clamp"]] * n_file,
        p + "nextn_predict_layers": c["n_nextn"],
    }


# ---------------------------------------------------------------- numpy reference (from llama.cpp glm5-next.cpp)

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sig(x)


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def hc_pre(f, base, sc, x, c):
    """build_hc_pre: rms over the flat streams, the 24 mixes, then hc_from_mixes."""
    flat = x.reshape(-1)
    mixes = f @ (flat / np.sqrt(np.mean(flat * flat) + c["eps"]))
    return hc_from_mixes(mixes, base, sc, c["hc_eps"], c["hc_iters"])


def hc_from_mixes(mixes, base, sc, eps, iters):
    """pre, post and the Sinkhorn-normalised comb[dst, src] of llama.cpp build_hc_pre / build_hc_sinkhorn."""
    pre = sig(mixes[0:4] * sc[0] + base[0:4]) + eps
    post = 2.0 * sig(mixes[4:8] * sc[1] + base[4:8])
    comb = (mixes[8:24] * sc[2] + base[8:24]).reshape(4, 4).T  # ggml [dst(ne0), src(ne1)] -> comb[dst, src]
    comb = np.exp(comb - comb.max(axis=0, keepdims=True))
    comb = comb / comb.sum(axis=0, keepdims=True) + eps  # soft_max over dst for each src
    comb = comb / (comb.sum(axis=1, keepdims=True) + eps)  # norm_cols: sum over src
    for _ in range(1, iters):
        comb = comb / (comb.sum(axis=0, keepdims=True) + eps)  # norm_rows: sum over dst
        comb = comb / (comb.sum(axis=1, keepdims=True) + eps)
    return pre, post, comb


def hc_post_ref(f, res, post, comb):
    return post[:, None] * f[None, :] + comb @ res


class Ref:
    def __init__(self, T):
        self.T = {k: v[1].astype(np.float64) for k, v in T.items()}
        self.c = C
        self.n_layer = len(C["kinds"]) - C["n_nextn"]
        self.state = [dict() for _ in range(self.n_layer)]

    def t(self, n):
        return self.T[n]

    def kda(self, il, x):
        c, b = self.c, f"blk.{il}."
        nh, hd, ck = c["n_head"], c["hd"], c["conv_k"]
        st = self.state[il]
        if not st:
            st["hist"] = [np.zeros((ck - 1, nh * hd)) for _ in range(3)]
            st["S"] = np.zeros((nh, hd, hd))
        outs = []
        for i, n in enumerate(("q", "k", "v")):
            raw = self.t(b + f"attn_{n}.weight") @ x
            w = self.t(b + f"ssm_conv1d_{n}.weight")[:, 0, :]  # [channel, tap]
            seq = np.vstack([st["hist"][i], raw[None, :]])  # taps oldest .. current
            outs.append(silu((seq.T * w).sum(axis=1)))
            st["hist"][i] = seq[1:]
        q, k, v = (o.reshape(nh, hd) for o in outs)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
        k = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-6)
        g1 = self.t(b + "ssm_f_b.weight") @ (self.t(b + "ssm_f_a.weight") @ x) + self.t(b + "ssm_dt.bias")
        g1 = g1.reshape(nh, hd) * self.t(b + "ssm_a")[:, None]
        g = c["lb"] * sig(-g1)
        beta = sig(self.t(b + "ssm_beta.weight") @ x)
        S = st["S"] * np.exp(g)[:, :, None]
        kv = np.einsum("hij,hi->hj", S, k)
        S = S + np.einsum("hi,hj->hij", k, beta[:, None] * (v - kv))
        st["S"] = S
        o = np.einsum("hij,hi->hj", S, q) / np.sqrt(hd)
        g2 = (self.t(b + "ssm_g_b.weight") @ (self.t(b + "ssm_g_a.weight") @ x)).reshape(nh, hd)
        nw = self.t(b + "ssm_norm.weight")
        o = o / np.sqrt(np.mean(o * o, axis=1, keepdims=True) + c["eps"]) * nw[None, :]
        return self.t(b + "attn_output.weight") @ (o * sig(g2)).reshape(-1)

    def dsa(self, il, x):
        c, b = self.c, f"blk.{il}."
        nh, kd, kl, kp, ni, nih = c["n_head"], c["k_mla"], c["kv_lora"], c["kpool"], c["ix_dim"], c["ix_heads"]
        st = self.state[il]
        if not st:
            st.update(lat=[], ik=[], ig=[], pooled=[])
        qr = rms(self.t(b + "attn_q_a.weight") @ x, self.t(b + "attn_q_a_norm.weight"), c["eps"])
        q = (self.t(b + "attn_q_b.weight") @ qr).reshape(nh, kd)
        lat = rms(self.t(b + "attn_kv_a_mqa.weight") @ x, self.t(b + "attn_kv_a_norm.weight"), c["eps"])
        ik = self.t(b + "indexer.attn_k.weight") @ x
        ik = (ik - ik.mean()) / np.sqrt(ik.var() + c["kda_eps"]) * self.t(b + "indexer.k_norm.weight") \
            + self.t(b + "indexer.k_norm.bias")
        ig = self.t(b + "indexer_compressor_gate.weight") @ x
        st["lat"].append(lat)
        st["ik"].append(ik)
        st["ig"].append(ig)
        pos = len(st["lat"]) - 1
        if (pos + 1) % kp == 0:
            ks = np.array(st["ik"][-kp:])
            lg = np.array(st["ig"][-kp:]) + self.t(b + "indexer_compressor_ape.weight")
            p = np.exp(lg - lg.max(axis=0))
            p /= p.sum(axis=0)
            st["pooled"].append((p * ks).sum(axis=0))
        if pos < c["top_k"]:
            rows = list(range(pos + 1))
        else:
            iq = (self.t(b + "indexer.attn_q_b.weight") @ qr).reshape(nih, ni)
            wt = (self.t(b + "indexer.proj.weight") @ x) / np.sqrt(ni * nih)
            pooled = np.array(st["pooled"])
            score = (np.maximum(pooled @ iq.T, 0.0) * wt[None, :]).sum(axis=1)
            nv = (pos + 1) // kp
            top = sorted(sorted(range(nv), key=lambda j: (-score[j], j))[:c["top_k"] // kp])
            rows = [r for j in top for r in range(j * kp, (j + 1) * kp)] + list(range(nv * kp, pos + 1))
        cache = np.array(st["lat"])[rows]
        kb = self.t(b + "attn_k_b.weight")  # [nh, kl, kd]
        vb = self.t(b + "attn_v_b.weight")  # [nh, vd, kl]
        out = []
        for h in range(nh):
            qa = kb[h] @ q[h]
            s = cache @ qa / np.sqrt(kd)
            p = np.exp(s - s.max())
            p /= p.sum()
            out.append(vb[h] @ (p @ cache))
        return self.t(b + "attn_output.weight") @ np.concatenate(out)

    def ffn(self, il, x):
        c, b = self.c, f"blk.{il}."
        if il < c["n_dense"]:
            return self.t(b + "ffn_down.weight") @ (silu(self.t(b + "ffn_gate.weight") @ x) * (self.t(b + "ffn_up.weight") @ x))
        probs = sig(self.t(b + "ffn_gate_inp.weight") @ x)
        key = probs + self.t(b + "exp_probs_b.bias")
        sel = sorted(range(len(key)), key=lambda i: (-key[i], i))[:c["n_used"]]
        wts = probs[sel] / max(probs[sel].sum(), 6.103515625e-5) * c["scale"]
        out = np.zeros(c["n_embd"])
        lim = c["clamp"]
        for e_, wgt in zip(sel, wts):
            g = np.minimum(self.t(b + "ffn_gate_exps.weight")[e_] @ x, lim)
            u = np.clip(self.t(b + "ffn_up_exps.weight")[e_] @ x, -lim, lim)
            out += wgt * (self.t(b + "ffn_down_exps.weight")[e_] @ (silu(g) * u))
        sh = silu(self.t(b + "ffn_gate_shexp.weight") @ x) * (self.t(b + "ffn_up_shexp.weight") @ x)
        return out + self.t(b + "ffn_down_shexp.weight") @ sh

    def step(self, tok):
        c = self.c
        x = np.tile(self.t("token_embd.weight")[tok], (4, 1))
        for il in range(self.n_layer):
            b = f"blk.{il}."
            hw = [self.t(b + "hc_attn" + s) for s in ("_fn.weight", "_base.weight", "_scale.weight")]
            pre, post, comb = hc_pre(*hw, x, c)
            cur = rms(pre @ x, self.t(b + "attn_norm.weight"), c["eps"])
            att = self.kda(il, cur) if c["kinds"][il] == 0 else self.dsa(il, cur)
            x = hc_post_ref(att, x, post, comb)
            hw = [self.t(b + "hc_ffn" + s) for s in ("_fn.weight", "_base.weight", "_scale.weight")]
            pre, post, comb = hc_pre(*hw, x, c)
            cur = rms(pre @ x, self.t(b + "ffn_norm.weight"), c["eps"])
            x = hc_post_ref(self.ffn(il, cur), x, post, comb)
        h = rms(x.mean(axis=0), self.t("output_norm.weight"), c["eps"])
        return self.t("output.weight") @ h


# ---------------------------------------------------------------- DLL

def load_dll():
    d = ctypes.CDLL(DLL)
    P = ctypes.POINTER
    f32p, i32p, i64p = P(ctypes.c_float), P(ctypes.c_int32), P(ctypes.c_int64)
    d.sct_last_error.restype = ctypes.c_char_p
    d.sctg_open.restype = ctypes.c_void_p
    d.sctg_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_longlong]
    d.sctg_close.argtypes = [ctypes.c_void_p]
    d.sctg_info.argtypes = [ctypes.c_void_p, i64p]
    d.sctg_validate.argtypes = [ctypes.c_void_p]
    d.sctg_check.argtypes = [ctypes.c_char_p, i64p]
    d.sctg_step.argtypes = [ctypes.c_void_p, ctypes.c_int32, f32p]
    d.sctg_prefill.argtypes = [ctypes.c_void_p, i32p, ctypes.c_int, f32p]
    d.sctg_reset.argtypes = [ctypes.c_void_p]
    d.sctg_hc_coeffs.argtypes = [f32p, f32p, f32p, ctypes.c_float, ctypes.c_int, f32p]
    d.sctg_hc_post.argtypes = [f32p, f32p, f32p, ctypes.c_int, f32p]
    d.sctg_route.argtypes = [f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, i32p, f32p]
    d.sctg_l2_norm.argtypes = [f32p, ctypes.c_int, ctypes.c_float]
    d.sctg_swiglu_clamp.argtypes = [f32p, f32p, ctypes.c_int, ctypes.c_float, f32p]
    d.sctg_q8_roundtrip.argtypes = [ctypes.c_uint, ctypes.c_int, f32p, ctypes.c_int, f32p]
    d.sctg_f16_round.argtypes = [ctypes.c_float]
    d.sctg_f16_round.restype = ctypes.c_float
    d.sctg_matvec.argtypes = [ctypes.c_uint, P(ctypes.c_uint8), ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p, f32p]
    return d


def fp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def err(d):
    return (d.sct_last_error() or b"").decode(errors="replace")


# ---------------------------------------------------------------- 1. unit checks

def unit_checks(d):
    print("[1] helpers against numpy")
    rng = np.random.default_rng(3)
    # hc_coeffs
    mixes, base, sc = f32(rng.standard_normal(24)), f32(rng.standard_normal(24) * 0.5), f32(0.5 + rng.random(3))
    out = np.zeros(24, np.float32)
    d.sctg_hc_coeffs(fp(mixes), fp(sc), fp(base), 1e-6, 20, fp(out))
    pre, post, comb = hc_from_mixes(mixes.astype(np.float64), base.astype(np.float64), sc.astype(np.float64), 1e-6, 20)
    ref = np.concatenate([pre, post, comb.T.reshape(-1)])  # comb[d + 4s] = comb[dst=d, src=s]
    check(np.allclose(out, ref, rtol=2e-5, atol=2e-6), f"hc_coeffs vs numpy Sinkhorn (max diff {np.abs(out - ref).max():.2e})")
    cm = out[8:].reshape(4, 4)  # [s][d]
    check(np.allclose(cm.sum(axis=0), 1, atol=1e-3) and np.allclose(cm.sum(axis=1), 1, atol=1e-3),
          "hc_coeffs comb is doubly stochastic (Rust test sinkhorn_doubly_stochastic)")
    # hc_post
    n = 37
    fvec, res = f32(rng.standard_normal(n)), f32(rng.standard_normal((4, n)))
    o = np.zeros(4 * n, np.float32)
    d.sctg_hc_post(fp(fvec), fp(res), fp(out), n, fp(o))
    ref = hc_post_ref(fvec.astype(np.float64), res.astype(np.float64), out[4:8].astype(np.float64),
                      out[8:].reshape(4, 4).T.astype(np.float64))
    check(np.allclose(o.reshape(4, n), ref, rtol=1e-5, atol=1e-5), "hc_post vs numpy")
    # route_sigmoid: the Rust unit test router_picks_by_bias_weights_by_prob
    lg = f32(np.full(8, -5.0))
    lg[2], lg[6] = 2.0, 1.0
    bias = f32(np.zeros(8))
    bias[4] = 5.0
    ids, wts = np.zeros(3, np.int32), np.zeros(3, np.float32)
    d.sctg_route(fp(lg), fp(bias), 8, 3, 1, 2.5, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), fp(wts))
    check(list(ids) == [4, 2, 6] and abs(wts.sum() - 2.5) < 1e-5 and wts[0] < wts[1],
          f"route_sigmoid Rust unit case: ids {[int(i) for i in ids]}, weights sum {wts.sum():.6f}")
    for trial in range(20):
        ne, nu = 288, 8
        lg, bias = f32(rng.standard_normal(ne) * 3), f32(rng.standard_normal(ne) * 0.1)
        ids, wts = np.zeros(nu, np.int32), np.zeros(nu, np.float32)
        d.sctg_route(fp(lg), fp(bias), ne, nu, 1, 2.5, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), fp(wts))
        p = sig(lg.astype(np.float64))
        key = p + bias
        sel = sorted(range(ne), key=lambda i: (-key[i], i))[:nu]
        rw = p[sel] / p[sel].sum() * 2.5
        if list(ids) != sel or not np.allclose(wts, rw, rtol=1e-5):
            check(False, f"route_sigmoid random trial {trial}")
            break
    else:
        check(True, "route_sigmoid on 20 random 288-expert routings (top-8, norm, scale 2.5)")
    # l2_norm (Rust unit test l2_norm_unit_and_floor)
    v = f32([3.0, 4.0])
    d.sctg_l2_norm(fp(v), 2, 1e-6)
    z = f32(np.zeros(4))
    d.sctg_l2_norm(fp(z), 4, 1e-6)
    check(abs(v[0] - 0.6) < 1e-6 and abs(v[1] - 0.8) < 1e-6 and not z.any(), "l2_norm unit and floor (Rust unit test)")
    # swiglu_clamp
    g, u = f32(rng.standard_normal(1000) * 20), f32(rng.standard_normal(1000) * 20)
    o = np.zeros(1000, np.float32)
    d.sctg_swiglu_clamp(fp(g), fp(u), 1000, 10.0, fp(o))
    gg = np.minimum(g.astype(np.float64), 10)
    check(np.allclose(o, silu(gg) * np.clip(u, -10, 10), rtol=1e-5, atol=1e-5), "swiglu_clamp vs ggml formula")
    # f16 rounding (half::f16::from_f32 is round-to-nearest-even, as numpy's float16 cast)
    vals = np.concatenate([rng.standard_normal(20000) * s for s in (1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4)]
                          + [np.array([65504.0, 65519.0, 65520.0, 70000.0, -65520.0, 6.1e-5, 5.96e-8, 2.98e-8, 0.0])])
    vals = f32(vals)
    got = np.array([d.sctg_f16_round(float(x)) for x in vals], np.float32)
    with np.errstate(over="ignore"):
        ref = vals.astype(np.float16).astype(np.float32)
    check(np.array_equal(got, ref), f"f16 rounding equals numpy float16 on {len(vals)} values "
                                     f"({int((got != ref).sum())} differ)")
    # q8_roundtrip
    x = f32(rng.standard_normal(1024) * 3)
    for ty, cpu, name in ((8, 0, "Q8_0"), (20, 0, "IQ4_NL"), (13, 1, "Q5_K cpu (Q8_K)"), (13, 0, "Q5_K gpu (Q8_0)"),
                          (12, 1, "Q4_K (unchanged)")):
        o = np.zeros_like(x)
        d.sctg_q8_roundtrip(ty, cpu, fp(x), len(x), fp(o))
        if ty == 12:
            ref = x.copy()
        elif cpu:
            ref = x.copy()
            for b0 in range(0, len(x), 256):
                b = ref[b0:b0 + 256]
                i = int(np.argmax(np.abs(b)))
                isc = np.float32(-127.0) / b[i]
                ref[b0:b0 + 256] = np.minimum(np.round(isc * b), 127).astype(np.float32) / isc
        else:
            ref = x.copy()
            for b0 in range(0, len(x), 32):
                b = ref[b0:b0 + 32]
                dd = np.float32(np.abs(b).max()) / np.float32(127)
                dh = np.float32(np.float16(dd))
                y_ = b / dd
                q = np.trunc(y_)
                q = q + np.sign(y_) * (np.abs(y_ - q) >= 0.5)  # Rust f32::round: half away from zero
                ref[b0:b0 + 32] = q * dh
        check(np.allclose(o, ref, rtol=0, atol=1e-6), f"q8_roundtrip {name} (max diff {np.abs(o - ref).max():.2e})")
    # Mat::matvec on Q8_0 weights
    for rows, cols, path in ((96, 256, "q8_roundtrip path"), (512, 256, "integer Q8_K path")):
        wq = rng.integers(-127, 128, size=(rows, cols // 32, 32)).astype(np.int8)
        ws = (rng.random((rows, cols // 32)) * 0.01 + 0.001).astype(np.float16)
        blob = np.zeros((rows, cols // 32, 34), np.uint8)
        blob[:, :, :2] = ws.view(np.uint8).reshape(rows, cols // 32, 2)
        blob[:, :, 2:] = wq.view(np.uint8)
        blob = np.ascontiguousarray(blob.reshape(-1))
        W = (wq.astype(np.float64) * ws.astype(np.float64)[:, :, None]).reshape(rows, cols)
        xv = f32(rng.standard_normal(cols))
        y = np.zeros(rows, np.float32)
        rc = d.sctg_matvec(8, blob.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), blob.size, rows, cols, 0, 0, 4,
                           fp(xv), fp(y))
        if rows < 512:
            xr = np.zeros_like(xv)
            d.sctg_q8_roundtrip(8, 0, fp(xv), cols, fp(xr))
            ref = W @ xr.astype(np.float64)
            tol = 1e-4
        else:
            ref = W @ xv.astype(np.float64)
            tol = 2e-2
        rel = np.abs(y - ref).max() / np.abs(ref).max()
        check(rc == 0 and rel < tol, f"Mat::matvec Q8_0 {rows}x{cols}, {path}: max rel diff {rel:.2e} (tol {tol})")


# ---------------------------------------------------------------- 2 + 3. model checks

def model_checks(d):
    os.makedirs(WORK, exist_ok=True)
    T = make_model()
    tens = [(k, v[0], v[1]) for k, v in T.items()]
    paths = {}
    for arch in ("glm5next", "glm5-next"):
        paths[arch] = os.path.join(WORK, f"tiny-{arch}.gguf")
        write_gguf(paths[arch], meta_for(arch), tens)
    bad = os.path.join(WORK, "tiny-extra-tensor.gguf")
    write_gguf(bad, meta_for("glm5next"), tens + [("blk.0.surprise.weight", [4], np.zeros(4, np.float32))])
    print(f"[2] config and tensor table ({os.path.getsize(paths['glm5next']) / 1e6:.1f} MB synthetic GGUF)")
    info = (ctypes.c_int64 * 9)()
    for arch, p in paths.items():
        rc = d.sctg_check(p.encode(), info)
        check(rc == 0 and list(info)[:4] == [4, 2, 2, 1], f"from_gguf + validate, arch {arch}: layers/kda/dsa/nextn "
                                                          f"{list(info)[:4]}" + ("" if rc == 0 else f" ({err(d)})"))
    rc = d.sctg_check(bad.encode(), info)
    check(rc == 2 and "unexpected tensor blk.0.surprise.weight" in err(d), "validate refuses an extra tensor")

    print("[3] forward: C++ CPU path vs float64 numpy (llama.cpp glm5-next.cpp formulas)")
    rng = np.random.default_rng(11)
    toks = [int(t) for t in rng.integers(0, C["vocab"], 20)]
    h = d.sctg_open(paths["glm5next"].encode(), 4, 0)
    check(bool(h), "open glm5next on the CPU path" + ("" if h else f": {err(d)}"))
    if not h:
        return
    d.sctg_info(h, info)
    check(list(info)[8] == 4096, f"context limit defaults to the trained context ({list(info)[8]})")
    ref = Ref(T)
    nv = C["vocab"]
    step_logits = []
    worst = 0.0
    argmax_same = 0
    for i, t in enumerate(toks):
        lo = np.zeros(nv, np.float32)
        if d.sctg_step(h, t, fp(lo)) != 0:
            check(False, f"step {i}: {err(d)}")
            return
        r = ref.step(t)
        step_logits.append(lo)
        worst = max(worst, np.abs(lo - r).max() / np.abs(r).max())
        argmax_same += int(np.argmax(lo) == np.argmax(r))
    check(worst < 2e-3, f"20 positions (indexer selection from position 8 on): max |logit diff| / max |logit| = {worst:.2e}")
    check(argmax_same == len(toks), f"argmax agrees at {argmax_same}/{len(toks)} positions")
    # The sparse selection matters: the same reference with every row visible differs from the C++ logits.
    dense = Ref(T)
    dense.c = dict(C, top_k=10 ** 9)
    last_dense = [dense.step(t) for t in toks][-1]
    gap = np.abs(step_logits[-1] - last_dense).max() / np.abs(last_dense).max()
    check(gap > 100 * worst, f"the k-pool selection is exercised (all-rows reference differs by {gap:.2e})")

    lo = np.zeros(nv, np.float32)
    d.sctg_reset(h)
    ids = (ctypes.c_int32 * len(toks))(*toks)
    rc = d.sctg_prefill(h, ids, len(toks), fp(lo))
    check(rc == 0 and np.array_equal(lo, step_logits[-1]),
          "prefill (chunk 512, layer-major, batched FFN) equals the token-by-token logits bit for bit"
          + ("" if rc == 0 else f": {err(d)}"))
    d.sctg_reset(h)
    rc = d.sctg_step(h, toks[0], fp(lo))
    check(rc == 0 and np.array_equal(lo, step_logits[0]), "reset forgets the sequence")
    # prefill continues an existing sequence
    d.sctg_reset(h)
    lo2 = np.zeros(nv, np.float32)
    rc = d.sctg_prefill(h, ids, 9, fp(lo2)) or d.sctg_prefill(h, (ctypes.c_int32 * 11)(*toks[9:]), 11, fp(lo2))
    check(rc == 0 and np.array_equal(lo2, step_logits[-1]), "two prefills (9 + 11 tokens) equal one sequence")
    d.sctg_close(h)

    os.environ["STRATA_GLM_CHUNK"] = "3"
    h = d.sctg_open(paths["glm5-next"].encode(), 3, 0)
    del os.environ["STRATA_GLM_CHUNK"]
    rc = d.sctg_prefill(h, ids, len(toks), fp(lo))
    check(rc == 0 and np.array_equal(lo, step_logits[-1]),
          "arch spelling glm5-next, STRATA_GLM_CHUNK=3, 3 threads: same logits bit for bit")
    d.sctg_close(h)

    h = d.sctg_open(paths["glm5next"].encode(), 2, 5)
    rc = d.sctg_prefill(h, ids, 6, fp(lo))
    check(rc != 0 and "limited to 5 tokens" in err(d), f"ctx 5 refuses a 6-token prompt ({err(d)!r})")
    rc = d.sctg_prefill(h, ids, 5, fp(lo))
    check(rc == 0, "ctx 5 takes a 5-token prompt")
    rc = d.sctg_step(h, toks[5], fp(lo))
    check(rc != 0 and "limited to 5 tokens" in err(d), "ctx 5 refuses a sixth token")
    d.sctg_close(h)


def main():
    if not os.path.isfile(DLL):
        print(f"FAIL: {DLL} missing; run build_cpu.bat first")
        return 1
    d = load_dll()
    unit_checks(d)
    model_checks(d)
    print("RESULT: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)} checks)"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
