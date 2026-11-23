"""test_flashnext.py - strata-cpp on Qwen3.8-Flash-Next (qwen4exp: hyper-connection streams, per-layer embeddings (PLE),
QSA sparse attention, DeltaNet and full-attention layers, 48 MoE layers of 512 experts, top-8 routing; W3-E1b).

(a) Parity: two prompts (8 tokens, and 300 tokens of the prose) with a 32-position continuation each. `build\\sc_score.exe`
    and the official `rustingstrata score --gpu --threads 7` read the same JSON, one process at a time. Gate: the same argmax
    on >= 97% of the 64 positions; the largest |top-1 logprob difference| is reported (CPU vs GPU expert rounding with a
    partial residency is expected to differ slightly).

(b) Bench: only when >= 90 GB of RAM is available. `build\\sc_bench.exe` against the official
    `rustingstrata bench --gpu --pp 512 --tg 128 --reps 3 --ctx 8192 --threads 7` (both outputs are printed side by side).
    Gates: decode >= 40 tok/s and >= 95% of the official decode mean; pp512 >= 95% of the official prefill mean.

Every Flash-Next process loads the 83 GB model. Before each one the available RAM is read (PowerShell Get-Counter), the
parity run needs >= 85 GB and the speed run >= 90 GB; below that the run is skipped and the RAM is reported. While a
Flash-Next process runs, native\\strata_cpp\\.gpu.lock exists. Nothing is left running after the test.

Usage: python tests/test_flashnext.py [--skip-parity] [--skip-bench] [--model PATH]

Needs the GPU, SC_MODEL_FLASHNEXT (the first shard), SC_RUSTINGSTRATA (the official 1.2.5 binary) and
STRATAFREET_ROOT (its tokenizer). Without any of them the test prints SKIP and exits 0.

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

BUILD = sc_env.build_dir(PORT)
DEFAULT_MODEL = sc_env.get("SC_MODEL_FLASHNEXT")
RUST = sc_env.get("SC_RUSTINGSTRATA")
SCORE = os.path.join(BUILD, "sc_score.exe")
BENCH = os.path.join(BUILD, "sc_bench.exe")
WORK = os.path.join(BUILD, "flashnext_test")
LOCK = os.path.join(PORT, ".gpu.lock")

# port of main.rs: prose (the text the bench prompt repeats; Rust's string literal, its line continuations joined)
PROSE = ("The river town woke before dawn. Fishermen checked their nets, bakers lit ovens, and a clockmaker "
         "argued with his apprentice about gears, tides and the price of brass. In 1874 the bridge washed away; "
         "the council voted 7 to 5 to rebuild it in stone. ")
SCORE_LENGTHS = [8, 300]
N_CONT = 32
THREADS = 7
GATE_ARGMAX = 0.97
GATE_SPEED = 0.95
GATE_DECODE_TOKS = 40.0
RAM_PARITY_MB = 85000
RAM_BENCH_MB = 90000
PP, TG, REPS, CTX_BENCH = 512, 128, 3, 8192


def avail_mb():
    """Available physical RAM in MB (PowerShell Get-Counter, as the project's RAM rule requires)."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        r"(Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue"],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def run_locked(cmd, need_mb, label):
    """Runs one Flash-Next process: RAM checked first, the GPU lock held while it runs."""
    have = avail_mb()
    print(f"  [{label}] available RAM {have:.0f} MB (needs >= {need_mb} MB)")
    if have < need_mb:
        return None, have
    if os.path.exists(LOCK):
        raise RuntimeError(f"{LOCK} already exists: another GPU process is running")
    with open(LOCK, "w", encoding="utf-8") as f:
        f.write(f"test_flashnext {label}\n")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        os.remove(LOCK)
    return r, have


def prose_ids(tok, n):
    """main.rs prose(): `n` tokens of the paragraph repeated (the first n of its encoding)."""
    ids = tok.encode(PROSE * (n // 20 + 1))[:n]
    assert len(ids) == n, f"prose prompt too short ({len(ids)} of {n})"
    return ids


def json_line(text):
    return json.loads([x for x in text.splitlines() if x.startswith("{")][-1])


def compare(a, b):
    n = len(a["argmax"])
    assert n == len(b["argmax"]), (n, len(b["argmax"]))
    same = sum(1 for x, y in zip(a["argmax"], b["argmax"]) if x == y)
    worst = max(abs(x[0][1] - y[0][1]) for x, y in zip(a["top5"], b["top5"]))
    return same, n, worst


def parity(model, tok):
    os.makedirs(WORK, exist_ok=True)
    cont_src = tok.encode(PROSE * 40)
    total_same = total_n = 0
    worst_all = 0.0
    print(f"(a) parity: {len(SCORE_LENGTHS)} prompts x {N_CONT} continuation positions, sc_score vs rustingstrata score --gpu")
    for i, n in enumerate(SCORE_LENGTHS):
        prompt = prose_ids(tok, n)
        cont = cont_src[37 * i:37 * i + N_CONT]
        assert len(cont) == N_CONT
        path = os.path.join(WORK, f"score{n}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "continuation": cont}, f)
        mine, _ = run_locked([SCORE, "--model", model, "--input", path, "--ctx", "4096", "--threads", str(THREADS)],
                             RAM_PARITY_MB, f"sc_score {n} tok")
        if mine is None:
            print(f"  prompt {n} tok: SKIPPED, not enough RAM for a Flash-Next load")
            return None
        if mine.returncode != 0:
            print(mine.stderr[-2000:])
            return False
        theirs, _ = run_locked([RUST, "score", "--model", model, "--input", path, "--gpu", "--threads", str(THREADS)],
                               RAM_PARITY_MB, f"rustingstrata {n} tok")
        if theirs is None:
            print(f"  prompt {n} tok: SKIPPED, not enough RAM for the official load")
            return None
        if theirs.returncode != 0:
            print(theirs.stderr[-2000:])
            return False
        a, b = json_line(mine.stdout), json_line(theirs.stdout)
        same, cnt, worst = compare(a, b)
        total_same += same
        total_n += cnt
        worst_all = max(worst_all, worst)
        print(f"  prompt {n:4d} tok: argmax {same}/{cnt}, max |top-1 logprob diff| {worst:.6f}")
        if same < cnt:
            bad = [k for k in range(cnt) if a["argmax"][k] != b["argmax"][k]]
            print(f"    argmax differs at positions {bad[:16]}")
    frac = total_same / total_n
    ok = frac >= GATE_ARGMAX
    print(f"parity total: argmax {total_same}/{total_n} = {100 * frac:.2f}% (gate >= {100 * GATE_ARGMAX:.0f}%), "
          f"max |top-1 logprob diff| {worst_all:.6f}  {'ok' if ok else 'FAIL'}")
    return ok


def pp_tg(text):
    """(pp mean, tg mean) of the `pp N: prefill X (...) tok/s | tg N: decode Y (...) tok/s` line."""
    m = re.search(r"pp\s+\d+: prefill ([\d.]+) \([\d.-]+\) tok/s \| tg \d+: decode ([\d.]+) \(", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def bench(model, tok):
    have = avail_mb()
    print(f"(b) bench: available RAM {have:.0f} MB (needs >= {RAM_BENCH_MB} MB)")
    if have < RAM_BENCH_MB:
        print("  SKIPPED: not enough RAM for the speed run; the numbers are not measured")
        return None
    os.makedirs(WORK, exist_ok=True)
    ids = prose_ids(tok, PP)
    eog = getattr(tok, "_eog", None) or set()
    eos = sorted(int(x) for x in eog) or ([tok.eos_id] if getattr(tok, "eos_id", None) is not None else [])
    path = os.path.join(WORK, "bench_prompts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"eos": eos, "prompts": {str(PP): ids}}, f)
    mine, _ = run_locked([BENCH, "--model", model, "--prompts", path, "--pp", str(PP), "--tg", str(TG), "--reps", str(REPS),
                          "--ctx", str(CTX_BENCH), "--threads", str(THREADS)], RAM_BENCH_MB, "sc_bench")
    if mine is None:
        print("  SKIPPED: RAM fell below the bench floor before the run")
        return None
    theirs, _ = run_locked([RUST, "bench", "--model", model, "--gpu", "--pp", str(PP), "--tg", str(TG), "--reps", str(REPS),
                            "--ctx", str(CTX_BENCH), "--threads", str(THREADS)], RAM_BENCH_MB, "rustingstrata bench")
    if theirs is None:
        print("  SKIPPED: RAM fell below the bench floor before the official run")
        return None
    print("--- ours (sc_bench):")
    for line in (mine.stdout + mine.stderr).splitlines():
        if line.strip():
            print("  " + line)
    print("--- official (rustingstrata bench):")
    keep = re.compile(r"^pp\s|plan:|expert hit|expert traffic|worker busy|mailbox traffic")
    for line in (theirs.stdout + theirs.stderr).splitlines():
        if keep.search(line):
            print("  " + line)
    if mine.returncode != 0 or theirs.returncode != 0:
        print(f"  exit codes: ours {mine.returncode}, official {theirs.returncode}")
        print(mine.stderr[-1500:])
        return False
    ours, theirs_v = pp_tg(mine.stdout), pp_tg(theirs.stdout)
    if not (ours and theirs_v):
        print("  could not read the pp/tg numbers")
        return False
    pp_ratio = ours[0] / theirs_v[0]
    tg_ratio = ours[1] / theirs_v[1]
    ok_pp = pp_ratio >= GATE_SPEED
    ok_tg = ours[1] >= GATE_DECODE_TOKS and tg_ratio >= GATE_SPEED
    print(f"  pp{PP}: ours {ours[0]:.1f} vs official {theirs_v[0]:.1f} tok/s = {100 * pp_ratio:.1f}% "
          f"(gate >= {100 * GATE_SPEED:.0f}%)  {'ok' if ok_pp else 'FAIL'}")
    print(f"  tg{TG}: ours {ours[1]:.1f} vs official {theirs_v[1]:.1f} tok/s = {100 * tg_ratio:.1f}% "
          f"(gate >= {100 * GATE_SPEED:.0f}% and >= {GATE_DECODE_TOKS:.0f} tok/s)  {'ok' if ok_tg else 'FAIL'}")
    return ok_pp and ok_tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    model = args.model or ""
    if not model:
        print(sc_env.skip_line("SC_MODEL_FLASHNEXT"))
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
    if not os.path.isfile(SCORE) or not os.path.isfile(BENCH):
        print(f"FAIL: {SCORE} / {BENCH} missing; run build.bat first")
        return 1
    from stratafreet.tokenizer import make_tokenizer
    tok = make_tokenizer(model)
    results = {}
    if not args.skip_parity:
        results["parity"] = parity(model, tok)
    if not args.skip_bench:
        results["bench"] = bench(model, tok)
    print("results: " + ", ".join(f"{k} {'SKIPPED' if v is None else ('PASS' if v else 'FAIL')}" for k, v in results.items()))
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
