"""test_score_glm5.py - strata-cpp against the official RustingStrata 1.2.5 on GLM-5.3-Flash (glm5-next).

Both sides run RustingStrata's GLM path: the CPU reference forward of glm/forward.rs with the dense matvecs and the
expert cache on the GPU (strata-cpp: src/glm_forward.cpp, src/ds_mat.cpp through build\\strata_cpp.dll).

Parity. For each prompt the official binary decodes greedily:
    STRATA_GLM_TOP5=1 STRATA_GLM_TRACE=1 rustingstrata generate --model <gguf> --prompt <text> -n N
and prints its prompt ids ("prompt ids [...]") and, per step, the top-5 token log-probabilities ("TOP5 i id:lp ...").
Its greedy tokens become the continuation that `build\\sc_score.exe` teacher-forces through the DLL (sc_prefill_logits,
then sc_decode), so every position compares the two engines on the same history. Gate: the same argmax on >= 97% of
the positions. The max |top-1 logprob difference| is reported (the oracle prints 4 decimals).

The arch key. RustingStrata 1.2.5 recognizes `general.architecture = glm5-next` only; llama.cpp and the unsloth GGUFs
spell it `glm5next` (strata-cpp accepts both). For such a file the test writes an alias set for the oracle: shard 1
re-serialized with the arch value and every `glm5next.*` key renamed to `glm5-next.*` (tensor infos and data copied
unchanged; refused when shard 1 is over 2 GB), and the other shards hard-linked next to it (no copy; the alias
directory must be on the model's volume: SC_GLM5_ALIAS_DIR, default build\\glm5_alias). strata-cpp reads the original
file.

Options: --n N continuation tokens per prompt (default 64); --long adds one prompt of about 2300 tokens (8 positions)
so the k-pool indexer's selection (past attention.indexer.top_k = 2048 rows) is compared too, which takes tens of
minutes per side; --speed prints greedy decode tok/s of the DLL next to `rustingstrata bench --pp 16 --tg 32`.

Needs: the GPU (nothing else on it), SC_MODEL_GLM5 (first shard of GLM-5.3-Flash, e.g. UD-IQ1_S
GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf), SC_RUSTINGSTRATA (official 1.2.5 binary), build.bat. Every process that
loads the model starts only with >= 24 GB of available RAM and holds .gpu.lock while it runs. SKIP (exit 0) when a
variable is unset.
Usage: python tests/test_score_glm5.py [--n N] [--long] [--speed]

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).
"""
import argparse
import ctypes
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

BUILD = sc_env.build_dir(PORT)
SCORE = os.path.join(BUILD, "sc_score.exe")
DLL = os.path.join(BUILD, "strata_cpp.dll")
WORK = os.path.join(BUILD, "glm5_test")
LOCK = os.path.join(PORT, ".gpu.lock")

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    ", "Once upon a time, a small fox"]
LONG_TEXT = ("In the middle of the old harbour, the fishermen mended their nets while the gulls circled overhead. "
             "Every morning the boats left before dawn, and every evening they returned with whatever the sea had "
             "given them. ") * 60 + "Summarize the paragraph above in one sentence:"
GATE_ARGMAX = 0.97
RAM_MIN_MB = 24000
SHARD1_COPY_MAX = 2 << 30


# ---------------------------------------------------------------- environment

def avail_mb():
    """Available physical RAM in MB (PowerShell Get-Counter, as the other model tests read it)."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        r"(Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue"],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def run_locked(cmd, label, env=None):
    """One model process: available RAM checked first, .gpu.lock held while it runs."""
    have = avail_mb()
    print(f"  [{label}] available RAM {have:.0f} MB (needs >= {RAM_MIN_MB} MB)")
    if have < RAM_MIN_MB:
        raise RuntimeError(f"only {have:.0f} MB of RAM available for {label}")
    if os.path.exists(LOCK):
        raise RuntimeError(f"{LOCK} already exists: another GPU process is running")
    with open(LOCK, "w", encoding="utf-8") as f:
        f.write(f"test_score_glm5 {label}\n")
    try:
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        return r, time.time() - t
    finally:
        os.remove(LOCK)


# ---------------------------------------------------------------- GGUF header (arch alias for the oracle)

def _rd(f, fmt):
    n = struct.calcsize("<" + fmt)
    b = f.read(n)
    if len(b) != n:
        raise ValueError("truncated GGUF header")
    return struct.unpack("<" + fmt, b)


def _rstr(f):
    (n,) = _rd(f, "Q")
    return f.read(n)


_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_val(f, t):
    if t in _SCALAR:
        f.seek(_SCALAR[t], 1)
    elif t == 8:
        (n,) = _rd(f, "Q")
        f.seek(n, 1)
    elif t == 9:
        et, n = _rd(f, "IQ")
        if et in _SCALAR:
            f.seek(_SCALAR[et] * n, 1)
        else:
            for _ in range(n):
                _skip_val(f, et)
    else:
        raise ValueError(f"GGUF value type {t}")


def read_arch(path):
    """general.architecture of a GGUF (header only)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF")
        _, _, nkv = _rd(f, "IQQ")
        for _ in range(nkv):
            k = _rstr(f).decode()
            (t,) = _rd(f, "I")
            if k == "general.architecture" and t == 8:
                return _rstr(f).decode()
            _skip_val(f, t)
    return ""


def write_alias_shard(src, dst, old, new):
    """`src` re-serialized with the arch value `old` -> `new` and every `old.*` key renamed to `new.*`. The tensor
    infos are copied unchanged; the data section is re-aligned (its offsets are relative to the data start)."""
    with open(src, "rb") as f:
        magic = f.read(4)
        ver, nt, nkv = _rd(f, "IQQ")
        kvs = []
        align = 32
        for _ in range(nkv):
            k = _rstr(f).decode()
            (t,) = _rd(f, "I")
            start = f.tell()
            if k == "general.alignment" and t == 4:
                (align,) = _rd(f, "I")
                f.seek(start)
            if k == "general.architecture" and t == 8:
                val = _rstr(f).decode()
                raw = struct.pack("<Q", len(new.encode())) + new.encode() if val == old else None
                f.seek(start)
                _skip_val(f, t)
                if raw is None:
                    end = f.tell()
                    f.seek(start)
                    raw = f.read(end - start)
            else:
                _skip_val(f, t)
                end = f.tell()
                f.seek(start)
                raw = f.read(end - start)
            if k.startswith(old + "."):
                k = new + k[len(old):]
            kvs.append(struct.pack("<Q", len(k.encode())) + k.encode() + struct.pack("<I", t) + raw)
        info_start = f.tell()
        for _ in range(nt):
            _rstr(f)
            (nd,) = _rd(f, "I")
            f.seek(8 * nd + 4 + 8, 1)
        info_end = f.tell()
        f.seek(info_start)
        infos = f.read(info_end - info_start)
        data_start = (info_end + align - 1) // align * align
        head = magic + struct.pack("<IQQ", ver, nt, nkv) + b"".join(kvs) + infos
        head += b"\0" * ((-len(head)) % align)
        f.seek(data_start)
        with open(dst + ".tmp", "wb") as o:
            o.write(head)
            shutil.copyfileobj(f, o, 16 << 20)
    os.replace(dst + ".tmp", dst)


def oracle_model(model):
    """The file the official binary reads: the model itself (arch glm5-next), or the alias set (arch glm5next)."""
    arch = read_arch(model)
    if arch == "glm5-next":
        return model
    if arch != "glm5next":
        raise RuntimeError(f"{model}: general.architecture is {arch!r}, not glm5-next / glm5next")
    name = os.path.basename(model)
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", name)
    n = int(m.group(2)) if m else 1
    if os.path.getsize(model) > SHARD1_COPY_MAX:
        raise RuntimeError(f"{model}: the first shard is {os.path.getsize(model) >> 20} MB; the alias copies it whole, "
                           f"so it is limited to {SHARD1_COPY_MAX >> 20} MB. Rename the arch key of a copy by hand.")
    alias_dir = sc_env.get("SC_GLM5_ALIAS_DIR") or os.path.join(BUILD, "glm5_alias")
    os.makedirs(alias_dir, exist_ok=True)
    first = os.path.join(alias_dir, name)
    if not os.path.exists(first) or os.path.getmtime(first) < os.path.getmtime(model):
        write_alias_shard(model, first, "glm5next", "glm5-next")
    for i in range(2, n + 1):
        src = os.path.join(os.path.dirname(model), f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf")
        dst = os.path.join(alias_dir, os.path.basename(src))
        if os.path.exists(dst):
            if os.path.samefile(src, dst):
                continue
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError as e:
            raise RuntimeError(f"cannot hard-link {src} into {alias_dir} ({e}); set SC_GLM5_ALIAS_DIR to a folder on "
                               f"the model's volume") from e
    print(f"  oracle alias: {first} (arch glm5-next, {n - 1} shard(s) hard-linked)")
    return first


# ---------------------------------------------------------------- the two engines

def oracle(rust_model, prompt, n, label):
    """Greedy decode of the official binary: (prompt ids, [top-5 [(id, logprob)] per step], seconds)."""
    env = dict(os.environ, STRATA_GLM_TOP5="1", STRATA_GLM_TRACE="1")
    r, dt = run_locked([sc_env.get("SC_RUSTINGSTRATA"), "generate", "--model", rust_model, "--prompt", prompt,
                        "-n", str(n)], label, env)
    if r.returncode != 0:
        raise RuntimeError(f"rustingstrata generate exited {r.returncode}: {r.stderr[-1500:]}")
    ids, steps = None, {}
    for line in r.stderr.splitlines():
        if line.startswith("prompt ids "):
            ids = json.loads(line[len("prompt ids "):])
        elif line.startswith("TOP5 "):
            parts = line.split()
            steps[int(parts[1])] = [(int(a), float(b)) for a, b in (p.split(":") for p in parts[2:])]
    if ids is None or not steps:
        raise RuntimeError("rustingstrata printed no prompt ids / TOP5 lines (STRATA_GLM_TRACE / STRATA_GLM_TOP5)")
    top5 = [steps[i] for i in range(len(steps))]
    return ids, top5, dt


def ours(model, prompt_ids, cont, label, ctx):
    path = os.path.join(WORK, f"{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt_ids, "continuation": cont}, f)
    r, dt = run_locked([SCORE, "--model", model, "--input", path, "--ctx", str(ctx)], label)
    if r.returncode != 0:
        raise RuntimeError(f"sc_score exited {r.returncode}: {r.stderr[-1500:]}")
    out = json.loads([x for x in r.stdout.splitlines() if x.startswith("{")][-1])
    return out, dt


def parity(model, rust_model, cases):
    os.makedirs(WORK, exist_ok=True)
    total_same = total_n = 0
    worst = 0.0
    for i, (text, n, ctx) in enumerate(cases):
        label = f"prompt{i + 1}"
        pid, top5, t_rust = oracle(rust_model, text, n, label + "-rustingstrata")
        cont = [s[0][0] for s in top5]
        mine, t_mine = ours(model, pid, cont, label + "-strata-cpp", ctx)
        same = sum(1 for k in range(len(cont)) if mine["argmax"][k] == top5[k][0][0])
        diffs = [abs(mine["top5"][k][0][1] - top5[k][0][1]) for k in range(len(cont)) if mine["argmax"][k] == top5[k][0][0]]
        w = max(diffs) if diffs else float("nan")
        worst = max(worst, w) if diffs else worst
        total_same += same
        total_n += len(cont)
        print(f"  {label} ({len(pid)} prompt tok, {text[:40]!r}): argmax {same}/{len(cont)}, max |top-1 logprob diff| "
              f"{w:.4f}   [rustingstrata {t_rust:.0f} s, strata-cpp {t_mine:.0f} s]")
        bad = [k for k in range(len(cont)) if mine["argmax"][k] != top5[k][0][0]]
        for k in bad[:8]:
            print(f"    position {k}: strata-cpp {mine['top5'][k][:2]}, rustingstrata {top5[k][:2]}")
    frac = total_same / max(total_n, 1)
    print(f"parity total: argmax {total_same}/{total_n} = {100 * frac:.2f}% (gate >= {100 * GATE_ARGMAX:.0f}%), "
          f"max |top-1 logprob diff| {worst:.4f} (reported, not gated)")
    return frac >= GATE_ARGMAX


def speed(model, prompt_ids, rust_model):
    """Greedy decode tok/s from the DLL (32 tokens after the prompt) and `rustingstrata bench --pp 16 --tg 32`."""
    have = avail_mb()
    if have < RAM_MIN_MB:
        print(f"speed: skipped, {have:.0f} MB RAM available")
        return
    with open(LOCK, "w", encoding="utf-8") as f:
        f.write("test_score_glm5 speed\n")
    try:
        d = ctypes.CDLL(DLL)
        d.sc_load.restype = ctypes.c_void_p
        d.sc_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        d.sc_prefill_sample.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_char_p,
                                        ctypes.POINTER(ctypes.c_int32)]
        d.sc_decode_sample.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
        d.sc_last_error.restype = ctypes.c_char_p
        d.sc_stats.restype = ctypes.c_char_p
        d.sc_stats.argtypes = [ctypes.c_void_p]
        d.sc_free.argtypes = [ctypes.c_void_p]
        h = d.sc_load(model.encode(), b'{"ctx":4096}')
        if not h:
            raise RuntimeError(d.sc_last_error().decode())
        out = ctypes.c_int32(0)
        ids = (ctypes.c_int32 * len(prompt_ids))(*prompt_ids)
        if d.sc_prefill_sample(h, ids, len(prompt_ids), b'{"temp":0}', ctypes.byref(out)) != 0:
            raise RuntimeError(d.sc_last_error().decode())
        t0 = time.perf_counter()
        for _ in range(32):
            if d.sc_decode_sample(h, out.value, b'{"temp":0}', ctypes.byref(out)) != 0:
                raise RuntimeError(d.sc_last_error().decode())
        dt = time.perf_counter() - t0
        st = json.loads(d.sc_stats(h).decode())
        d.sc_free(h)
    finally:
        os.remove(LOCK)
    print(f"speed: strata-cpp 32 greedy tokens in {dt:.1f} s = {32 / dt:.2f} tok/s (backend {st['backend']}, "
          f"expert-cache hits {st['hits']} misses {st['misses']}, {st['experts_resident']} slots, {st['gpu_mb']} MB VRAM)")
    r, _ = run_locked([sc_env.get("SC_RUSTINGSTRATA"), "bench", "--model", rust_model, "--pp", "16", "--tg", "32",
                       "--reps", "1"], "rustingstrata-bench")
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("pp "):
            print("  rustingstrata " + line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--speed", action="store_true")
    args = ap.parse_args()
    if not sc_env.require("SC_MODEL_GLM5", "SC_RUSTINGSTRATA"):
        return 0
    model, rust = sc_env.get("SC_MODEL_GLM5"), sc_env.get("SC_RUSTINGSTRATA")
    for p, what in ((model, "SC_MODEL_GLM5"), (rust, "SC_RUSTINGSTRATA"), (SCORE, "build.bat"), (DLL, "build.bat")):
        if not os.path.isfile(p):
            print(f"FAIL: {p} not found ({what})")
            return 1
    print(f"GLM-5.3-Flash parity: {os.path.basename(model)}, {len(PROMPTS)} prompts x {args.n} greedy positions"
          + (", plus the long prompt" if args.long else ""))
    rust_model = oracle_model(model)
    cases = [(p, args.n, 4096) for p in PROMPTS]
    if args.long:
        cases.append((LONG_TEXT, 8, 8192))
    ok = parity(model, rust_model, cases)
    if args.speed:
        pid, _, _ = oracle(rust_model, PROMPTS[0], 1, "speed-prompt-ids")
        speed(model, pid, rust_model)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
