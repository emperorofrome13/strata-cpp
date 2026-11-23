"""verify_ram_lock.py - the strata-cpp RAM lock feature (sc_load "ram"/"pin") end to end on the 14B
partial-residency load, with the file-cache eviction test that motivated the feature.

The failure mode: with "ram" off the weights are read through the memory map, so the model's pages live in
the Windows file cache (standby list). When another big file read (or another model load) cycles the list,
every expert read faults from the drive and decode collapses. "copy"/"lock" hold the tensor data in private
committed RAM instead, so the eviction cannot touch it.

Runs (one at a time, .gpu.lock held, each needs >= 85 GB available RAM):
  A. ram off, warm                 - baseline decode speed
  B. ram off + evict --evict-gb    - the model's pages are evicted; rep 1 should collapse, reps 2-3 recover
  C. ram copy + evict              - weights in private RAM; decode should stay at baseline
  D. ram lock + evict              - same as C but locked; stats ram_locked_bytes > 0
  E. ram lock + pin + evict        - cuMemHostRegister; stats ram_pinned > 0, ram_direct_bytes > 0
  F. greedy parity ram off vs lock - same prompt + continuation, the argmax sequence must be identical

Usage: python tests/verify_ram_lock.py [--evict-gb 130] [--skip-parity] [--model PATH]

Needs the GPU, SC_MODEL_14B, SC_EVICT_GGUFS (semicolon-separated GGUF files used to evict the Windows
file cache) and STRATAFREET_ROOT (its tokenizer). Without any of them the test prints SKIP and exits 0.

Part of strata-cpp, a C++ port of RustingStrata 1.2.5 (RustingStrata License 1.0).
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
BENCH = os.path.join(BUILD, "sc_bench.exe")
SCORE = os.path.join(BUILD, "sc_score.exe")
WORK = os.path.join(BUILD, "ram_test")
LOCK = os.path.join(PORT, ".gpu.lock")
MARGIN = 8500
THREADS = 7
PP, TG, REPS, CTX = 512, 128, 3, 4096
RAM_FLOOR_MB = 85000
EVICT_FILES = [p.strip().strip('"') for p in sc_env.get("SC_EVICT_GGUFS").split(";") if p.strip()]
PROSE = ("The river town woke before dawn. Fishermen checked their nets, bakers lit ovens, and a clockmaker "
         "argued with his apprentice about gears, tides, and the price of brass. In 1874 the bridge washed "
         "away; the council voted to rebuild it in stone. ")


def avail_mb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        r"(Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue"],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def run_locked(cmd, label):
    """One model process at a time: RAM checked, .gpu.lock held while the process runs."""
    have = avail_mb()
    print(f"  [{label}] available RAM {have:.0f} MB (needs >= {RAM_FLOOR_MB} MB)")
    if have < RAM_FLOOR_MB:
        return None
    if os.path.exists(LOCK):
        raise RuntimeError(f"{LOCK} already exists: another GPU process is running")
    with open(LOCK, "w", encoding="utf-8") as f:
        f.write(f"verify_ram_lock {label}\n")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    took = time.time() - t0
    os.remove(LOCK)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise RuntimeError(f"{label} exited {r.returncode}")
    print(f"  [{label}] finished in {took:.0f} s")
    return r


def prose_ids(tok, n):
    ids = tok.encode(PROSE * (n // 20 + 1))[:n]
    assert len(ids) == n, f"prose prompt too short ({len(ids)} of {n})"
    return ids


def make_assets(model):
    os.makedirs(WORK, exist_ok=True)
    from stratafreet.tokenizer import make_tokenizer
    tok = make_tokenizer(model)
    ids = prose_ids(tok, PP)
    eog = getattr(tok, "_eog", None) or set()
    eos = sorted(int(x) for x in eog) or ([tok.eos_id] if getattr(tok, "eos_id", None) is not None else [])
    prompts = os.path.join(WORK, "bench_prompts_14b.json")
    with open(prompts, "w", encoding="utf-8") as f:
        json.dump({"eos": eos, "prompts": {str(PP): ids}}, f)
    # parity input: a 300-token prose prompt and a 32-position continuation from a different offset
    cont_src = tok.encode(PROSE * 40)
    p = prose_ids(tok, 300)
    cont = cont_src[200:200 + 32]
    inp = os.path.join(WORK, "parity_input.json")
    with open(inp, "w", encoding="utf-8") as f:
        json.dump({"prompt": p, "continuation": cont}, f)
    return prompts, inp


def parse_bench(text):
    """Per-rep decode rates and the summary line, from sc_bench output."""
    reps = [float(m.group(1)) for m in re.finditer(r"rep \d+/\d+: prefill [\d.]+ tok/s, decode ([\d.]+) tok/s", text)]
    m = re.search(r"pp\s+\d+: prefill ([\d.]+) \([^)]*\) tok/s \| tg\s+\d+: decode ([\d.]+) \([^)]*\) tok/s", text)
    summary = (float(m.group(1)), float(m.group(2))) if m else None
    stats = {}
    m = re.search(r"stats after load: (\{.*\})", text)
    if m:
        stats = json.loads(m.group(1))
    return summary, reps, stats


def jnum(stats, key):
    return stats.get(key, 0)


def bench_run(label, ram, evict_gb, prompts, model, pin=False):
    cmd = [BENCH, "--model", model, "--prompts", prompts, "--pp", str(PP), "--tg", str(TG),
           "--reps", str(REPS), "--ctx", str(CTX), "--vram-margin", str(MARGIN), "--threads", str(THREADS),
           "--ram", ram, "--json-out", os.path.join(WORK, f"bench_{label}.json")]
    if pin:
        cmd += ["--pin"]
    if evict_gb:
        cmd += ["--evict", ",".join(EVICT_FILES), "--evict-gb", str(evict_gb)]
    r = run_locked(cmd, label)
    if r is None:
        return None
    summary, reps, stats = parse_bench(r.stdout + r.stderr)
    return {"label": label, "ram": ram, "summary": summary, "reps": reps, "stats": stats}


def parity_run(inp, ram, model):
    """sc_score with the given ram mode; returns the argmax + top5 list."""
    cmd = [SCORE, "--model", model, "--input", inp, "--ctx", str(CTX), "--threads", str(THREADS),
           "--vram-margin-mb", str(MARGIN), "--ram", ram]
    r = run_locked(cmd, f"parity {ram}")
    if r is None:
        return None
    line = [x for x in r.stdout.splitlines() if x.startswith("{")][-1]
    return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--evict-gb", type=float, default=130.0)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    model = args.model or ""
    if not model:
        print(sc_env.skip_line("SC_MODEL_14B"))
        return 0
    if not sc_env.get("SC_EVICT_GGUFS"):
        print(sc_env.skip_line("SC_EVICT_GGUFS"))
        return 0
    if not os.path.isfile(model):
        print(f"FAIL: model not found: {model}")
        return 1
    missing = [p for p in EVICT_FILES if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"FAIL: eviction file not found: {p}")
        return 1
    if not os.path.isfile(BENCH) or not os.path.isfile(SCORE):
        print(f"FAIL: {BENCH} / {SCORE} missing; run build.bat first")
        return 1
    print(f"verify_ram_lock: {os.path.basename(model)}, margin {MARGIN} MB, threads {THREADS}, "
          f"evict {args.evict_gb} GB over {len(EVICT_FILES)} files")
    prompts, inp = make_assets(model)

    rows = []
    # A. baseline, warm
    r = bench_run("A-off-warm", "off", 0, prompts, model)
    if r:
        rows.append(r)
    # B. ram off + evict: the model's file-cache pages are evicted before the bench
    r = bench_run("B-off-evict", "off", args.evict_gb, prompts, model)
    if r:
        rows.append(r)
    # C. ram copy + evict
    r = bench_run("C-copy-evict", "copy", args.evict_gb, prompts, model)
    if r:
        rows.append(r)
    # D. ram lock + evict
    r = bench_run("D-lock-evict", "lock", args.evict_gb, prompts, model)
    if r:
        rows.append(r)
    # E. ram lock + pin + evict
    r = bench_run("E-lock-pin-evict", "lock", args.evict_gb, prompts, model, pin=True)
    if r:
        rows.append(r)

    base = next((x for x in rows if x["label"] == "A-off-warm"), None)
    print()
    print(f"{'run':16} {'ram':5} {'prefill':>9} {'decode':>9}   per-rep decode")
    for x in rows:
        pre, dec = x["summary"] if x["summary"] else ("-", "-")
        pre_s = f"{pre:.1f}" if isinstance(pre, float) else str(pre)
        dec_s = f"{dec:.1f}" if isinstance(dec, float) else str(dec)
        print(f"{x['label']:16} {x['ram']:5} {pre_s:>9} {dec_s:>9}   " +
              ", ".join(f"{v:.1f}" for v in x["reps"]))
    if base and base["summary"]:
        bd = base["summary"][1]
        print(f"baseline decode {bd:.1f} tok/s")
        for x in rows:
            if x["label"] == "A-off-warm" or not x["summary"]:
                continue
            d = x["summary"][1]
            print(f"  {x['label']}: {d:.1f} tok/s ({d / bd * 100:.0f}% of baseline)")

    ok = True
    print()
    print("ram stats:")
    for x in rows:
        st = x["stats"]
        print(f"  {x['label']:16} mode={st.get('ram_mode')} bytes={jnum(st, 'ram_bytes') / 1e9:.2f} GB "
              f"locked={jnum(st, 'ram_locked_bytes') / 1e9:.2f} GB pinned={jnum(st, 'ram_pinned') / 1e9:.2f} GB "
              f"direct={jnum(st, 'ram_direct_bytes') / 1e9:.2f} GB load={st.get('load_seconds', 0):.0f}s "
              f"ram_load={st.get('ram_load_seconds', 0):.0f}s warnings={st.get('ram_warnings')}")
        if x["label"] == "D-lock-evict" and jnum(st, "ram_locked_bytes") <= 0:
            print("  FAIL: ram lock did not lock anything")
            ok = False
        if x["label"] == "E-lock-pin-evict" and jnum(st, "ram_pinned") <= 0:
            print("  FAIL: pin did not register anything")
            ok = False

    if not args.skip_parity:
        print()
        a = parity_run(inp, "off", model)
        b = parity_run(inp, "lock", model)
        if a and b:
            same = sum(1 for x, y in zip(a["argmax"], b["argmax"]) if x == y)
            worst = max(abs(x[0][1] - y[0][1]) for x, y in zip(a["top5"], b["top5"]))
            n = len(a["argmax"])
            print(f"parity ram off vs lock: argmax {same}/{n}, max |top1 logprob diff| {worst:.6f}")
            # Two separate loads: the expert-cache routing profile (profile-<arch>-<L>x<E>.f32,
            # expert_cache.cpp) is read at load and updated by save_profile after every run, so
            # placement differs slightly between loads and top-1 logprobs drift even with
            # identical weights. Measured on this 14B Q6_K file: cross-load same-binary noise
            # off-vs-off 0.0454 and off-vs-lock 0.006 (argmax 32/32 both, control script
            # build/ram_test/parity_control.py); cross-implementation diff strata-cpp vs official
            # RustingStrata is 0.195-0.317 (MODIFICATIONS.md, W3-E1b). The 1e-4 gate was written
            # for same-process repeats (test_score_14b: 0.000000) and can never pass across two
            # loads. Gate: argmax exact + diff within 0.10 (2x the largest observed cross-load
            # noise, 2x below the smallest implementation-level diff).
            if same != n:
                print("  FAIL: the two ram modes disagree (argmax)")
                ok = False
            elif worst > 0.10:
                print(f"  FAIL: top-1 logprob diff {worst:.6f} exceeds the cross-load bound 0.10")
                ok = False
        else:
            ok = False

    print()
    print("verify_ram_lock:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
