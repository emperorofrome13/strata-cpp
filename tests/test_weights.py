"""Wave 2-A acceptance (weights half): config.rs, weights.rs and memory.rs against Python oracles.

    python tests/test_weights.py [model.gguf ...]

With no arguments it uses SC_MODEL_14B and SC_MODEL_FLASHNEXT (whichever are set). For every model it checks:

  * `ModelConfig::from_gguf` field by field against the GGUF metadata, recomputed here independently
    (tests/test_weights.py does not import the C++ values);
  * every tensor as `Weights::get` resolves it - quant type, cols, rows and byte length - against
    `stratafreet/gguf_io.py`;
  * one row per GGML type present, dequantized on both sides and compared bit for bit;
  * `Tensor::matvec` and `Tensor::matmul` against a float32 sequential-sum oracle (the same summation
    order the Rust code uses: one running sum per element, and 16 partial sums for `dot`);
  * `Tensor::expert` on a 3-D expert tensor;
  * the synthesized `ssm_beta`/`ssm_alpha` split of `ssm_ba.weight`  (weights.rs: Weights::new_with_aux);
  * `Weights::validate` on a real file;
  * `memory.rs`: total_ram and ram_cap, and the Prefetch worker's start/send/join.

Needs STRATAFREET_ROOT (the Python reference oracle) and at least one model. Without them the test
prints SKIP and exits 0.
"""

from __future__ import annotations

import ctypes
import json
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
GGUFModelFile = None
dequant = None

# port of config.rs: arch(name) - (hybrid, q_gate, k_interleave, aliases)
FFN_NORM = [["post_attention_norm.weight", "ffn_norm.weight"]]
ARCHS = {
    "qwen35": (True, True, False, []),
    "qwen35moe": (True, True, False, []),
    "qwen4exp": (True, True, False, []),
    "qwen3next": (True, True, True, []),
    "flint": (False, False, False, FFN_NORM),
    "qwen3": (False, False, False, FFN_NORM),
    "qwen3moe": (False, False, False, FFN_NORM),
}

def env_models() -> list:
    return [p for p in (sc_env.get("SC_MODEL_14B"), sc_env.get("SC_MODEL_FLASHNEXT")) if p]


def load():
    if not os.path.isfile(DLL):
        raise SystemExit(f"missing {DLL}; run build.bat first")
    lib = ctypes.CDLL(DLL)
    lib.sci_last_error.restype = ctypes.c_char_p
    for fn in (lib.sci_model_config_json, lib.sci_weights_check_json):
        fn.argtypes = [ctypes.c_char_p]
        fn.restype = ctypes.c_char_p
    lib.sci_weights_tensor_json.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.sci_weights_tensor_json.restype = ctypes.c_char_p
    fp = ctypes.POINTER(ctypes.c_float)
    lib.sci_weights_row.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_longlong, fp, ctypes.c_longlong]
    lib.sci_weights_row.restype = ctypes.c_int
    lib.sci_weights_expert_row.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_longlong,
                                           ctypes.c_longlong, ctypes.c_longlong, fp, ctypes.c_longlong]
    lib.sci_weights_expert_row.restype = ctypes.c_int
    lib.sci_weights_matvec.argtypes = [ctypes.c_char_p, ctypes.c_char_p, fp, ctypes.c_longlong, fp,
                                       ctypes.c_longlong]
    lib.sci_weights_matvec.restype = ctypes.c_int
    lib.sci_weights_matmul.argtypes = [ctypes.c_char_p, ctypes.c_char_p, fp, ctypes.c_longlong, fp,
                                       ctypes.c_longlong]
    lib.sci_weights_matmul.restype = ctypes.c_int
    lib.sci_total_ram.argtypes = []
    lib.sci_total_ram.restype = ctypes.c_longlong
    lib.sci_ram_cap.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong]
    lib.sci_ram_cap.restype = ctypes.c_longlong
    lib.sci_prefetch_roundtrip.argtypes = [ctypes.c_longlong]
    lib.sci_prefetch_roundtrip.restype = ctypes.c_int
    return lib


def call_json(lib, fn, *args):
    raw = fn(*args)
    if raw is None:
        raise RuntimeError(lib.sci_last_error().decode())
    return json.loads(raw.decode())


def f32(a) -> ctypes.Array:
    return np.ascontiguousarray(a, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ------------------------------------------------------------------ oracles

def expected_config(ref) -> dict:
    """config.rs: ModelConfig::from_gguf, recomputed from the GGUF metadata."""
    meta = ref.meta
    arch = str(meta["general.architecture"])
    hybrid, q_gate, k_interleave, aliases = ARCHS[arch]

    def opt(k):
        v = meta.get(f"{arch}.{k}")
        return 0 if v is None else int(v)

    def req(k):
        return int(meta[f"{arch}.{k}"])

    n_nextn = opt("nextn_predict_layers")
    n_layer = req("block_count") - n_nextn
    if hybrid:
        every = opt("full_attention_interval") or 4
        layers = ["attn" if (il + 1) % every == 0 else "delta" for il in range(n_layer)]
    else:
        layers = ["attn"] * n_layer
    has_delta = "delta" in layers
    n_embd, n_head = req("embedding_length"), req("attention.head_count")
    head_dim = opt("attention.key_length") or n_embd // n_head

    def flags(k):
        a = meta.get(f"{arch}.{k}")
        return None if a is None else [bool(int(x)) for x in a[:n_layer]]

    swa = flags("attention.sliding_window_pattern")
    if swa is None:
        swa = [False] * n_layer
    rope_layer = flags("attention.rope_pattern")
    if rope_layer is None:
        rope_layer = [True] * n_layer
    ssm = req if has_delta else opt
    ms = meta.get(f"{arch}.rope.dimension_sections")
    mrope = [int(x) for x in ms[:4]] if ms is not None and len(ms) >= 4 else None
    ews = meta.get(f"{arch}.expert_weights_scale")
    ssm_inner, ssm_dt = ssm("ssm.inner_size"), ssm("ssm.time_step_rank")
    return {
        "arch": arch,
        "n_layer": n_layer,
        "n_nextn": n_nextn,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": req("attention.head_count_kv"),
        "head_dim": head_dim,
        "n_rot": opt("rope.dimension_count") or head_dim,
        "mrope_sections": mrope,
        "rope_base": float(meta.get(f"{arch}.rope.freq_base", 10000.0)),
        "eps": float(meta[f"{arch}.attention.layer_norm_rms_epsilon"]),
        "n_vocab": int(ref.tensors["token_embd.weight"].shape[0]),
        "n_ff": opt("feed_forward_length"),
        "n_expert": opt("expert_count"),
        "n_expert_used": opt("expert_used_count"),
        "n_ff_exp": opt("expert_feed_forward_length"),
        "n_ff_shexp": opt("expert_shared_feed_forward_length"),
        "n_dense": opt("leading_dense_block_count"),
        "expert_weights_scale": 0.0 if ews is None else float(ews),
        "router_sigmoid": arch == "flint" and opt("expert_gating_func") == 2,
        "n_swa": opt("attention.sliding_window"),
        "swa": swa,
        "rope_layer": rope_layer,
        "attn_sinks": "blk.0.attn_sinks.weight" in ref.tensors,
        "ssm_d_inner": ssm_inner,
        "ssm_d_state": ssm("ssm.state_size"),
        "ssm_n_group": ssm("ssm.group_count"),
        "ssm_dt_rank": ssm_dt,
        "ssm_conv": ssm("ssm.conv_kernel"),
        "hc_n": opt("hyper_connection.count"),
        "hc_rank": opt("hyper_connection.low_rank"),
        "n_embd_ple": opt("embedding_length_per_layer_input"),
        "idx_n_head": opt("attention.indexer.head_count"),
        "idx_head_dim": opt("attention.indexer.key_length"),
        "idx_top_k": opt("attention.indexer.top_k"),
        "layers": layers,
        "q_gate": q_gate,
        "k_interleave": k_interleave,
        "aliases": aliases,
        "ssm_dv": ssm_inner // max(1, ssm_dt),
        "is_moe": opt("expert_count") > 0,
    }


def same(got, want) -> bool:
    if isinstance(want, list) or want is None or isinstance(want, str):
        return got == want
    if isinstance(want, bool):
        return bool(got) == want
    g, w = float(got), float(want)
    return g == w or abs(g - w) <= 1e-6 * max(1.0, abs(w))


def dequant_row_of(ref, name: str, cols: int, r: int) -> np.ndarray:
    """Row `r` of the tensor flattened over `cols`-wide rows - the shape weights.rs: Tensor sees
    (gguf_io's TensorRef.row_bytes divides by the leading dim, which is wrong for 1-D and 3-D)."""
    t = ref.tensors[name]
    rows_total = t.n_elements // cols
    rb = t.nbytes // rows_total
    raw = np.frombuffer(bytes(t.data[r * rb:(r + 1) * rb]), dtype=np.uint8).copy()
    return dequant(torch.from_numpy(raw), int(t.qtype), (cols,), torch.float32).numpy()


def dequant_rows(ref, name: str, cols: int, rows) -> list:
    return [dequant_row_of(ref, name, cols, r) for r in rows]


def seq_dot(a: np.ndarray, b: np.ndarray) -> np.float32:
    """weights.rs: Tensor::matvec - one running f32 sum, in row order."""
    return np.cumsum((a * b).astype(np.float32), dtype=np.float32)[-1]


def dot16(a: np.ndarray, b: np.ndarray) -> np.float32:
    """weights.rs: dot - 16 partial sums over 16-element blocks, then their sum, then the tail."""
    n = a.size
    blocks = n // 16
    acc = np.zeros(16, dtype=np.float32)
    if blocks:
        prod = (a[:blocks * 16] * b[:blocks * 16]).astype(np.float32).reshape(blocks, 16)
        acc = np.cumsum(prod, axis=0, dtype=np.float32)[-1]
    tail = np.float32(0.0)
    for i in range(blocks * 16, n):
        tail = np.float32(tail + np.float32(a[i] * b[i]))
    s = np.float32(0.0)
    for j in range(16):
        s = np.float32(s + acc[j])
    return np.float32(s + tail)


# ------------------------------------------------------------------ per model

def check_model(lib, path: str) -> list:
    bad: list = []
    p = path.encode()
    ref = GGUFModelFile(path)

    # 1. config
    got = call_json(lib, lib.sci_model_config_json, p)
    want = expected_config(ref)
    if set(got) != set(want):
        bad.append(f"config keys differ: {sorted(set(got) ^ set(want))}")
    for k in sorted(set(got) & set(want)):
        if not same(got[k], want[k]):
            bad.append(f"config.{k}: {got[k]!r} != {want[k]!r}")

    # 2. every tensor as Weights::get resolves it
    wc = call_json(lib, lib.sci_weights_check_json, p)
    if not wc["nextn"]:
        bad.append("validate: nextn reported false for a model with no unsupported nextn tensors")
    for name, t in ref.tensors.items():
        info = wc["tensors"].get(name)
        if info is None:
            bad.append(f"{name}: missing from the C++ tensor map")
            continue
        ty, cols, rows, blen = info
        want_cols = int(t.shape[-1])
        want_rows = t.n_elements // want_cols
        if [ty, cols, rows, blen] != [int(t.qtype), want_cols, want_rows, t.nbytes]:
            bad.append(f"{name}: [{ty}, {cols}, {rows}, {blen}] != "
                       f"[{int(t.qtype)}, {want_cols}, {want_rows}, {t.nbytes}]")

    # 3. one row per GGML type, exact
    by_type: dict = {}
    for name, t in ref.tensors.items():
        by_type.setdefault(int(t.qtype), []).append(name)
    for ty in sorted(by_type):
        name = min(by_type[ty], key=lambda n: ref.tensors[n].n_elements)
        t = ref.tensors[name]
        rows = int(t.n_elements // int(t.shape[-1]))
        idx = sorted({0, rows - 1})
        cols = int(t.shape[-1])
        want_rows = dequant_rows(ref, name, cols, idx)
        for r, expect in zip(idx, want_rows):
            out = (ctypes.c_float * cols)()
            if lib.sci_weights_row(p, name.encode(), r, out, cols) != 0:
                bad.append(f"{name} row {r}: {lib.sci_last_error().decode()}")
                continue
            got_row = np.frombuffer(out, dtype=np.float32).copy()
            if not np.array_equal(got_row, expect, equal_nan=True):
                bad.append(f"{name} row {r}: {int((got_row != expect).sum())} of {cols} values differ")

    # 4. matvec / matmul against the float32 summation oracle
    cands = [n for n, t in ref.tensors.items()
             if len(t.shape) == 2 and int(t.shape[-1]) >= 64 and 1 << 21 >= t.n_elements]
    if cands:
        name = min(cands, key=lambda n: ref.tensors[n].n_elements)
        t = ref.tensors[name]
        cols, rows = int(t.shape[-1]), t.n_elements // int(t.shape[-1])
        w = dequant(torch.from_numpy(np.frombuffer(bytes(t.data), dtype=np.uint8).copy()),
                    int(t.qtype), (rows * cols,), torch.float32).numpy().reshape(rows, cols)
        rng = np.random.default_rng(20261120)
        x = rng.standard_normal(cols).astype(np.float32)

        y = np.zeros(rows, dtype=np.float32)
        if lib.sci_weights_matvec(p, name.encode(), f32(x), cols, f32(y), rows) != 0:
            bad.append(f"matvec {name}: {lib.sci_last_error().decode()}")
        else:
            want_y = np.cumsum((w * x).astype(np.float32), axis=1, dtype=np.float32)[:, -1]
            diff = int((y != want_y).sum())
            if diff:
                worst = float(np.abs(y - want_y).max())
                bad.append(f"matvec {name} ({rows}x{cols}): {diff} of {rows} rows differ, max |d| {worst:g}")
            else:
                print(f"    matvec  {name} {rows}x{cols}: exact")

        n_tokens = 3
        xs = rng.standard_normal((n_tokens, cols)).astype(np.float32)
        out = np.zeros(n_tokens * rows, dtype=np.float32)
        if lib.sci_weights_matmul(p, name.encode(), f32(xs), n_tokens, f32(out), n_tokens * rows) != 0:
            bad.append(f"matmul {name}: {lib.sci_last_error().decode()}")
        else:
            want_out = np.zeros((n_tokens, rows), dtype=np.float32)
            for ti in range(n_tokens):
                for r in range(rows):
                    want_out[ti, r] = dot16(w[r], xs[ti])
            got_out = out.reshape(n_tokens, rows)
            diff = int((got_out != want_out).sum())
            if diff:
                worst = float(np.abs(got_out.astype(np.float64) - want_out.astype(np.float64)).max())
                bad.append(f"matmul {name}: {diff} of {n_tokens * rows} values differ, max |d| {worst:g}")
            else:
                print(f"    matmul  {name} {n_tokens}x{rows}x{cols}: exact")

    # 5. expert slicing
    exps = [n for n, t in ref.tensors.items() if len(t.shape) == 3]
    if exps:
        name = min(exps, key=lambda n: ref.tensors[n].n_elements)
        t = ref.tensors[name]
        n_expert = int(t.shape[0])
        cols = int(t.shape[-1])
        rows_e = t.n_elements // (n_expert * cols)
        for e in (0, n_expert - 1):
            for r in (0, rows_e - 1):
                out = (ctypes.c_float * cols)()
                if lib.sci_weights_expert_row(p, name.encode(), e, n_expert, r, out, cols) != 0:
                    bad.append(f"{name} expert {e} row {r}: {lib.sci_last_error().decode()}")
                    continue
                got_row = np.frombuffer(out, dtype=np.float32).copy()
                # row r of expert e is row e * rows_e + r of the flat tensor
                expect = dequant_row_of(ref, name, cols, e * rows_e + r)
                if not np.array_equal(got_row, expect, equal_nan=True):
                    bad.append(f"{name} expert {e} row {r}: values differ")
        print(f"    expert  {name}: {n_expert} experts x {rows_e}x{cols}, first and last checked")

    # 6. ssm_beta / ssm_alpha synthesis (weights.rs: Weights::new_with_aux)
    c = want
    nk, nv = c["ssm_n_group"], c["ssm_dt_rank"]
    split_layers = [n for n in wc["built"] if n.endswith(".ssm_beta.weight")]
    # A file that already stores ssm_beta/ssm_alpha has no ssm_ba to split, and Weights::new_with_aux
    # leaves `built` empty for it - exactly as the Rust `let Ok(ba) = .. else { continue }` does.
    ba_layers = [il for il in range(c["n_layer"]) if f"blk.{il}.ssm_ba.weight" in ref.tensors]
    if not ba_layers:
        print(f"    ssm     file stores split ssm_beta/ssm_alpha already, nothing to synthesize")
    if ba_layers and c["ssm_d_inner"] and nk and nv and nv % nk == 0:
        if len(split_layers) != len(ba_layers):
            bad.append(f"ssm split: {len(split_layers)} synthesized beta tensors for "
                       f"{len(ba_layers)} ssm_ba layers")
        r = nv // nk
        for il in ba_layers:
            ba_name = f"blk.{il}.ssm_ba.weight"
            if f"blk.{il}.ssm_beta.weight" not in wc["built"]:
                bad.append(f"{ba_name} exists but was not split")
                continue
            ba = ref.tensors[ba_name]
            rb = ba.row_bytes
            raw = bytes(ba.data)
            grp = 2 * r * rb
            beta = b"".join(raw[o:o + r * rb] for o in range(0, len(raw), grp))
            alpha = b"".join(raw[o + r * rb:o + grp] for o in range(0, len(raw), grp))
            cols = int(ba.shape[-1])
            for suffix, expect_bytes in (("ssm_beta", beta), ("ssm_alpha", alpha)):
                name = f"blk.{il}.{suffix}.weight"
                for row in (0, nv - 1):
                    out = (ctypes.c_float * cols)()
                    if lib.sci_weights_row(p, name.encode(), row, out, cols) != 0:
                        bad.append(f"{name} row {row}: {lib.sci_last_error().decode()}")
                        continue
                    got_row = np.frombuffer(out, dtype=np.float32).copy()
                    raw_row = np.frombuffer(expect_bytes[row * rb:(row + 1) * rb], dtype=np.uint8).copy()
                    expect = dequant(torch.from_numpy(raw_row), int(ba.qtype), (cols,), torch.float32).numpy()
                    if not np.array_equal(got_row, expect, equal_nan=True):
                        bad.append(f"{name} row {row}: split values differ")
        if split_layers:
            print(f"    ssm     {len(split_layers)} layers split into beta/alpha, rows 0 and {nv - 1} exact")

    print(f"    {len(ref.tensors)} tensors, {len(by_type)} quant types, config {len(want)} fields")
    return bad


def check_memory(lib) -> list:
    bad = []
    total = lib.sci_total_ram()
    if total <= 0:
        bad.append(f"total_ram: {total}")
    cap = lib.sci_ram_cap(1 << 30, 0)
    if cap != total - (1 << 30):
        bad.append(f"ram_cap(1 GiB margin): {cap} != {total - (1 << 30)}")
    if lib.sci_ram_cap(512 << 20, 8 << 30) != 8 << 30:
        bad.append("ram_cap: an explicit max_ram did not win")
    if lib.sci_prefetch_roundtrip(64) != 0:
        bad.append(f"prefetch roundtrip: {lib.sci_last_error().decode()}")
    print(f"    memory  total {total / 2**30:.1f} GiB, ram_cap ok, prefetch 64 batches ok")
    return bad


def main(argv) -> int:
    global np, torch, GGUFModelFile, dequant
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    requested = list(argv) or env_models()
    if not requested:
        for name in ("SC_MODEL_14B", "SC_MODEL_FLASHNEXT"):
            print(sc_env.skip_line(name))
        return 0
    missing = [m for m in requested if not os.path.isfile(m)]
    if missing:
        for m in missing:
            print(f"FAIL: model not found: {m}")
        return 1
    try:
        import numpy as numpy_mod
        import torch as torch_mod
        from stratafreet.gguf_io import GGUFModelFile as gguf_model_file
        from stratafreet.quant import dequant as dequant_mod
    except Exception as e:
        return sc_env.skip(f"cannot import the StrataFreet oracle: {e}")
    np, torch, GGUFModelFile, dequant = numpy_mod, torch_mod, gguf_model_file, dequant_mod

    lib = load()
    models = requested
    failures = []
    for path in models:
        print(f"--- {os.path.basename(path)}")
        try:
            bad = check_model(lib, path)
        except RuntimeError as e:
            failures.append(f"{path}: {e}")
            print(f"    ERROR {e}")
            continue
        if bad:
            failures += [f"{os.path.basename(path)}: {b}" for b in bad]
            print(f"    {len(bad)} mismatch(es)")
        else:
            print("    ok")
    print()
    bad = check_memory(lib)
    if bad:
        failures += bad
    if failures:
        for f in failures[:40]:
            print(f"FAIL {f}")
        return 1
    print(f"PASS: config, weights and memory agree for {len(models)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
