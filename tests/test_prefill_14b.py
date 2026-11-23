"""test_prefill_14b.py - strata-cpp's batched prefill (W3-E3) on the Qwen3.6-14B-A3B MoE (qwen35moe).

(a) Parity, fully resident: three prompts of 64, 300 and 1000 tokens (the batched prefill; the short threshold is 32)
    with a 64-token continuation each. `build\\sc_score.exe` (sc_prefill_logits + sc_decode) and the official
    `rustingstrata score --gpu` read the same JSON. Gate: the same argmax on >= 99% of the 192 positions and
    |top-1 logprob difference| <= 0.05 at every position.

(b) Bench: `build\\sc_bench.exe` against the official `rustingstrata bench --gpu --pp 512 --tg 128 --reps 3 --ctx 4096`,
    fully resident and with `--vram-margin 8500 --threads 7`; both outputs are printed side by side. Gates: our pp512
    >= 95% of the official prefill mean, our tg128 >= 95% of the official decode mean, and the resident expert count of
    the `plan:` line equal to the official one.

Each process loads the model (about 10 GB), so the runs go one after another. Needs the GPU, SC_MODEL_14B,
SC_RUSTINGSTRATA (the official 1.2.5 binary) and STRATAFREET_ROOT (its tokenizer).
Usage: python tests/test_prefill_14b.py [--skip-parity] [--skip-bench] [--model PATH]

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).
"""
import argparse
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
BENCH = os.path.join(BUILD, "sc_bench.exe")
WORK = os.path.join(BUILD, "prefill_test")

# port of main.rs: prose (the text the bench prompt repeats; Rust's string literal, its line continuations joined)
PROSE = ("The river town woke before dawn. Fishermen checked their nets, bakers lit ovens, and a clockmaker "
         "argued with his apprentice about gears, tides and the price of brass. In 1874 the bridge washed away; "
         "the council voted 7 to 5 to rebuild it in stone. ")
SCORE_LENGTHS = [64, 300, 1000]
N_CONT = 64
GATE_ARGMAX = 0.99
GATE_LOGPROB = 0.05
GATE_SPEED = 0.95
PP, TG, REPS = 512, 128, 3


def prose_ids(tok, n):
    """main.rs prose(): `n` tokens of the paragraph repeated (the first n of its encoding)."""
    ids = tok.encode(PROSE * (n // 20 + 1))[:n]
    assert len(ids) == n, f"prose prompt too short ({len(ids)} of {n})"
    return ids


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return r


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
    ok = True
    print(f"(a) parity, fully resident: {len(SCORE_LENGTHS)} prompts x {N_CONT} continuation positions")
    for i, n in enumerate(SCORE_LENGTHS):
        prompt = prose_ids(tok, n)
        cont = cont_src[37 * i:37 * i + N_CONT]
        assert len(cont) == N_CONT
        path = os.path.join(WORK, f"score{n}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "continuation": cont}, f)
        t = time.time()
        mine = run([SCORE, "--model", model, "--input", path, "--ctx", "4096"])
        t_mine = time.time() - t
        if mine.returncode != 0:
            print(mine.stderr[-1500:])
            return False
        t = time.time()
        theirs = run([RUST, "score", "--model", model, "--input", path, "--gpu"])
        t_theirs = time.time() - t
        if theirs.returncode != 0:
            print(theirs.stderr[-1500:])
            return False
        a, b = json_line(mine.stdout), json_line(theirs.stdout)
        same, cnt, worst = compare(a, b)
        total_same += same
        total_n += cnt
        worst_all = max(worst_all, worst)
        print(f"  prompt {n:5d} tok: argmax {same}/{cnt}, max |top-1 logprob diff| {worst:.6f}"
              f"   [sc_score {t_mine:.0f} s, rustingstrata {t_theirs:.0f} s]")
        if same < cnt:
            bad = [k for k in range(cnt) if a["argmax"][k] != b["argmax"][k]]
            print(f"    argmax differs at positions {bad[:12]}")
    frac = total_same / total_n
    ok = frac >= GATE_ARGMAX and worst_all <= GATE_LOGPROB
    print(f"parity total: argmax {total_same}/{total_n} = {100 * frac:.2f}% (gate >= {100 * GATE_ARGMAX:.0f}%), "
          f"max |top-1 logprob diff| {worst_all:.6f} (gate <= {GATE_LOGPROB})  {'ok' if ok else 'FAIL'}")
    return ok


def resident_of(text):
    m = re.findall(r"resident experts (\d+)/(\d+)", text)
    return m[-1] if m else None


def pp_tg(text):
    """(pp mean, tg mean) of the `pp N: prefill X (...) tok/s | tg N: decode Y (...) tok/s` line."""
    m = re.search(r"pp\s+\d+: prefill ([\d.]+) \([\d.-]+\) tok/s \| tg \d+: decode ([\d.]+) \(", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def bench_pair(label, model, ids_path, margin_args):
    mine = run([BENCH, "--model", model, "--prompts", ids_path, "--pp", str(PP), "--tg", str(TG), "--reps", str(REPS),
                "--ctx", "4096", *margin_args])
    theirs = run([RUST, "bench", "--model", model, "--gpu", "--pp", str(PP), "--tg", str(TG), "--reps", str(REPS),
                  "--ctx", "4096", *margin_args])
    print(f"--- {label}")
    print("ours (sc_bench):")
    for line in (mine.stdout + mine.stderr).splitlines():
        if line.strip():
            print("  " + line)
    print("official (rustingstrata bench):")
    keep = re.compile(r"^pp\s|plan:|expert hit|expert traffic|worker busy|mailbox traffic")
    for line in (theirs.stdout + theirs.stderr).splitlines():
        if keep.search(line):
            print("  " + line)
    if mine.returncode != 0 or theirs.returncode != 0:
        print(f"  exit codes: ours {mine.returncode}, official {theirs.returncode}")
        print(mine.stderr[-1200:])
        return False
    ours_pp = pp_tg(mine.stdout)
    theirs_pp = pp_tg(theirs.stdout)
    res_ours = resident_of(mine.stderr)
    res_theirs = resident_of(theirs.stderr)
    if not (ours_pp and theirs_pp):
        print("  could not read the pp/tg numbers")
        return False
    pp_ratio = ours_pp[0] / theirs_pp[0]
    tg_ratio = ours_pp[1] / theirs_pp[1]
    ok_pp = pp_ratio >= GATE_SPEED
    ok_tg = tg_ratio >= GATE_SPEED
    ok_res = res_ours is not None and res_ours == res_theirs
    print(f"  pp{PP}: ours {ours_pp[0]:.1f} vs official {theirs_pp[0]:.1f} tok/s = {100 * pp_ratio:.1f}% "
          f"(gate >= {100 * GATE_SPEED:.0f}%)  {'ok' if ok_pp else 'FAIL'}")
    print(f"  tg{TG}: ours {ours_pp[1]:.1f} vs official {theirs_pp[1]:.1f} tok/s = {100 * tg_ratio:.1f}% "
          f"(gate >= {100 * GATE_SPEED:.0f}%)  {'ok' if ok_tg else 'FAIL'}")
    print(f"  resident experts: ours {res_ours} vs official {res_theirs}  {'ok' if ok_res else 'FAIL'}")
    return ok_pp and ok_tg and ok_res


def bench(model, tok):
    os.makedirs(WORK, exist_ok=True)
    ids = prose_ids(tok, PP)
    eog = getattr(tok, "_eog", None) or set()  # the end-of-generation ids (Rust: tok.eos_ids())
    eos = sorted(int(x) for x in eog) or ([tok.eos_id] if getattr(tok, "eos_id", None) is not None else [])
    path = os.path.join(WORK, "bench_prompts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"eos": eos, "prompts": {str(PP): ids}}, f)
    ok = True
    ok = bench_pair("fully resident (default margin)", model, path, []) and ok
    ok = bench_pair("--vram-margin 8500 --threads 7", model, path, ["--vram-margin", "8500", "--threads", "7"]) and ok
    return ok


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
    if not os.path.isfile(SCORE) or not os.path.isfile(BENCH):
        print(f"FAIL: {SCORE} / {BENCH} missing; run build.bat first")
        return 1
    from stratafreet.tokenizer import make_tokenizer
    tok = make_tokenizer(model)
    ok = True
    if not args.skip_parity:
        ok = parity(model, tok) and ok
    if not args.skip_bench:
        ok = bench(model, tok) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
