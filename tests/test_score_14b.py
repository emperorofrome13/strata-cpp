"""test_score_14b.py - strata-cpp against RustingStrata 1.2.5 on the Qwen3.6-14B-A3B MoE (qwen35moe, all experts
resident in VRAM).

1. Parity: three prompts (under the 32-token `short` threshold, so both sides run the prompt token by token) and a 64-token
   continuation each. `build\\sc_score.exe` and the official `rustingstrata score --gpu` read the same JSON. Gate: the same
   argmax on >= 99% of the 192 positions, and |top-1 logprob difference| <= 0.05 at every position.
2. Speed: 128 greedy tokens after a 16-token prompt, decoded by the DLL (`sc_decode_sample`, timed after a warm-up that
   records the CUDA graph), and `rustingstrata bench --pp 16 --tg 128 --reps 3 --ctx 4096` on the same model.

Each process loads the model (about 10 GB), so the runs go one after another. Needs the GPU, SC_MODEL_14B,
SC_RUSTINGSTRATA (the official 1.2.5 binary) and STRATAFREET_ROOT (its tokenizer).
Usage: python tests/test_score_14b.py [--skip-speed] [--skip-bench] [--model PATH]

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).
"""
import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

BUILD = sc_env.build_dir(PORT)
DEFAULT_MODEL = sc_env.get("SC_MODEL_14B")
RUST = sc_env.get("SC_RUSTINGSTRATA")
SCORE = os.path.join(BUILD, "sc_score.exe")
DLL = os.path.join(BUILD, "strata_cpp.dll")
WORK = os.path.join(BUILD, "score_test")

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    ", "Once upon a time, a small fox"]
LONG = ("In the middle of the old harbour, the fishermen mended their nets while the gulls circled overhead. "
        "Every morning the boats left before dawn, and every evening they returned with whatever the sea had given them. ") * 3
SPEED_PROMPT = "The history of the port city begins with a small fishing village on the northern shore"
N_POS = 64
GATE_ARGMAX = 0.99
GATE_LOGPROB = 0.05


def make_inputs(model):
    """Token ids of the prompts and continuations, with StrataFreet's tokenizer (the GGUF's own vocabulary)."""
    from stratafreet.tokenizer import make_tokenizer
    tok = make_tokenizer(model)
    cont_src = tok.encode(LONG)
    cases = []
    for i, p in enumerate(PROMPTS):
        pid = tok.encode(p)
        cont = cont_src[i * 7:i * 7 + N_POS]
        assert len(pid) < 32, f"prompt {i} has {len(pid)} tokens: must be under the short threshold (32)"
        assert len(cont) == N_POS, f"continuation source too short ({len(cont)})"
        cases.append({"prompt": pid, "continuation": cont, "text": p})
    speed = tok.encode(SPEED_PROMPT)[:16]
    assert len(speed) == 16
    return cases, speed


def run_json(cmd):
    """Runs a command that prints one JSON object as its last stdout line."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} exited {r.returncode}: {r.stderr[-800:]}")
    line = [x for x in r.stdout.splitlines() if x.startswith("{")][-1]
    return json.loads(line)


def compare(a, b):
    """Positions where argmax agrees, and the largest top-1 logprob difference, of two score outputs."""
    n = len(a["argmax"])
    assert n == len(b["argmax"]) == N_POS, (n, len(b["argmax"]))
    same = sum(1 for x, y in zip(a["argmax"], b["argmax"]) if x == y)
    worst = max(abs(x[0][1] - y[0][1]) for x, y in zip(a["top5"], b["top5"]))
    return same, n, worst


def parity(model, cases):
    os.makedirs(WORK, exist_ok=True)
    total_same = total_n = 0
    worst_all = 0.0
    print(f"parity: {len(cases)} prompts x {N_POS} continuation positions, {os.path.basename(model)}")
    for i, c in enumerate(cases):
        path = os.path.join(WORK, f"input{i + 1}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"prompt": c["prompt"], "continuation": c["continuation"]}, f)
        t = time.time()
        mine = run_json([SCORE, "--model", model, "--input", path])
        t_mine = time.time() - t
        t = time.time()
        theirs = run_json([RUST, "score", "--model", model, "--input", path, "--gpu"])
        t_theirs = time.time() - t
        same, n, worst = compare(mine, theirs)
        total_same += same
        total_n += n
        worst_all = max(worst_all, worst)
        print(f"  prompt {i + 1} ({len(c['prompt'])} tok, {c['text']!r}): argmax {same}/{n}, "
              f"max |top-1 logprob diff| {worst:.6f}   [sc_score {t_mine:.0f} s, rustingstrata {t_theirs:.0f} s]")
        if same < n:
            bad = [k for k in range(n) if mine["argmax"][k] != theirs["argmax"][k]]
            print(f"    argmax differs at positions {bad[:10]}")
    frac = total_same / total_n
    ok = frac >= GATE_ARGMAX and worst_all <= GATE_LOGPROB
    print(f"parity total: argmax {total_same}/{total_n} = {100 * frac:.2f}% (gate >= {100 * GATE_ARGMAX:.0f}%), "
          f"max |top-1 logprob diff| {worst_all:.6f} (gate <= {GATE_LOGPROB})")
    return ok


def speed(model, prompt):
    """Greedy decode tokens/s from the DLL: 128 tokens after `prompt`, the first 8 of them as warm-up."""
    d = ctypes.CDLL(DLL)
    d.sc_load.restype = ctypes.c_void_p
    d.sc_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    d.sc_decode_sample.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
    d.sc_decode_sample.restype = ctypes.c_int
    d.sc_last_error.restype = ctypes.c_char_p
    d.sc_free.argtypes = [ctypes.c_void_p]
    d.sc_stats.restype = ctypes.c_char_p
    d.sc_stats.argtypes = [ctypes.c_void_p]
    h = d.sc_load(model.encode(), b'{"ctx":4096,"kv":"f16","vram_margin_mb":512}')
    if not h:
        raise RuntimeError(d.sc_last_error().decode())
    out = ctypes.c_int32(0)

    def step(tok):
        if d.sc_decode_sample(h, tok, b'{"temp":0}', ctypes.byref(out)) != 0:
            raise RuntimeError(d.sc_last_error().decode())
        return out.value

    tok = 0
    for t in prompt:
        tok = step(t)
    gen = []
    for _ in range(8):
        tok = step(tok)
    t0 = time.perf_counter()
    for _ in range(128):
        tok = step(tok)
        gen.append(tok)
    dt = time.perf_counter() - t0
    stats = json.loads(d.sc_stats(h).decode())
    d.sc_free(h)
    tps = 128 / dt
    print(f"decode: 128 greedy tokens in {dt:.2f} s = {tps:.1f} tok/s (sc_decode_sample, CUDA graph replayed; "
          f"experts resident {stats['experts_resident']}/{stats['experts_total']}, hits {stats['hits']}, misses {stats['misses']})")
    return tps


def bench(model):
    r = subprocess.run([RUST, "bench", "--model", model, "--gpu", "--pp", "16", "--tg", "128", "--reps", "3", "--ctx", "4096"],
                       capture_output=True, text=True)
    text = r.stdout + r.stderr
    print("rustingstrata bench --pp 16 --tg 128 --reps 3 --ctx 4096 (exit %d):" % r.returncode)
    for line in text.splitlines():
        if re.search(r"tg|pp ", line) and not line.startswith("cuda"):
            print("  " + line)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    model = args.model or ""
    if not model:
        print(sc_env.skip_line("SC_MODEL_14B"))
        return 0
    if not sc_env.get("SC_RUSTINGSTRATA"):
        print(sc_env.skip_line("SC_RUSTINGSTRATA"))
        return 0
    if not os.path.isfile(model):
        print(f"FAIL: model not found: {model}")
        return 1
    if not os.path.isfile(RUST):
        print(f"FAIL: oracle not found: {RUST} (SC_RUSTINGSTRATA)")
        return 1
    if not os.path.isfile(SCORE) or not os.path.isfile(DLL):
        print(f"FAIL: {SCORE} / {DLL} missing; run build.bat first")
        return 1
    cases, speed_prompt = make_inputs(model)
    ok = True
    if not args.skip_parity:
        ok = parity(model, cases) and ok
    if not args.skip_speed:
        speed(model, speed_prompt)
    if not args.skip_bench:
        ok = bench(model) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
