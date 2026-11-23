"""test_score_inkling.py - strata-cpp on Inkling-2-Mini (arch inkling2: 48 logical slots over 24 physical blocks,
3 loop passes, relative-bias SWA/full attention, shortconv, fused gate/up experts, situ, router bias; W4/W5).

(a) Parity: two prompts (8 tokens, and 300 tokens of the prose) with a 32-position continuation each. The reference
    is llama.cpp's unsloth prebuilt (b12588-mix-c0c3de8, the build that reads the looped layout), CPU only, started
    by this test on a private port. It answers one /completion per prompt (temperature 0, n_probs 5, n_predict 32);
    `build\\sc_score.exe` teacher-forces the same continuation ids through strata-cpp and prints its argmax and top-5
    per position. Gate: the same argmax on >= 97% of the 64 positions; the largest |top-1 logprob difference| is
    reported (CPU and GPU rounding, plus the IQ quant, are expected to differ slightly; W2 measured the engine at
    the reference's own cross-backend noise floor).

(b) Bench: `build\\sc_bench.exe` (pp512, tg128, 3 reps, ctx 8192, f16 KV). No official Inkling engine exists to
    compare against, so the gate is absolute: decode >= --bench-gate tok/s (default 50 for the all-resident
    IQ3_XXS file, 0 for a larger file that runs through the expert cache). The numbers are printed either way.
    The all-resident IQ3 load also exercises the batched prefill (W3, 16-token chunks with a captured graph); a
    partly resident load keeps the token-by-token path.

Every process that loads the model is behind the project's RAM rule and the native\\strata_cpp\\.gpu.lock: the test
holds the lock for its whole run, the oracle is started with >= 30 GB available RAM, and nothing is left running.

Usage: python tests/test_score_inkling.py [--skip-parity] [--skip-bench] [--model PATH]
                                                              [--bench-gate TOK_S]
       set STRATA_KV_Q8=1 to run the same checks with the int8 KV cache (parity 64/64; the bench then
       measures ~102 tok/s at ctx 8192 on the all-resident IQ3_XXS).

Needs the GPU, SC_MODEL_INKLING, SC_LLAMA_SERVER (the llama.cpp oracle) and STRATAFREET_ROOT (its
tokenizer). Without any of them the test prints SKIP and exits 0.

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
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

BUILD = sc_env.build_dir(PORT)
DEFAULT_MODEL = sc_env.get("SC_MODEL_INKLING")
LLAMA = sc_env.get("SC_LLAMA_SERVER")
SCORE = os.path.join(BUILD, "sc_score.exe")
BENCH = os.path.join(BUILD, "sc_bench.exe")
WORK = os.path.join(BUILD, "inkling_test")
LOCK = os.path.join(PORT, ".gpu.lock")
PORT_ORACLE = 8180

# the same prose as test_flashnext / main.rs (the bench prompt repeats it)
PROSE = ("The river town woke before dawn. Fishermen checked their nets, bakers lit ovens, and a clockmaker "
         "argued with his apprentice about gears, tides and the price of brass. In 1874 the bridge washed away; "
         "the council voted 7 to 5 to rebuild it in stone. ")
SCORE_LENGTHS = [8, 300]
N_CONT = 32
THREADS = 7
GATE_ARGMAX = 0.97
RAM_PARITY_MB = 30000
PP, TG, REPS, CTX_BENCH = 512, 128, 3, 8192


def log(msg):
    print(msg, flush=True)


def avail_mb():
    """Available physical RAM in MB (PowerShell Get-Counter, as the project's RAM rule requires)."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        r"(Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue"],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def take_lock(label):
    if os.path.exists(LOCK):
        raise RuntimeError(f"{LOCK} already exists: another GPU process is running")
    with open(LOCK, "w", encoding="utf-8") as f:
        f.write(f"test_score_inkling {label}\n")


def drop_lock():
    if os.path.exists(LOCK):
        os.remove(LOCK)


def prose_ids(tok, n):
    ids = tok.encode(PROSE * (n // 20 + 1))[:n]
    assert len(ids) == n, f"prose prompt too short ({len(ids)} of {n})"
    return ids


def oracle_request(ids, n):
    body = json.dumps({"prompt": ids, "n_predict": n, "temperature": 0.0, "n_probs": 5, "cache_prompt": False,
                       "top_k": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT_ORACLE}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def start_oracle(model):
    if not os.path.isfile(LLAMA):
        return None, f"llama-server not found: {LLAMA}"
    have = avail_mb()
    log(f"  oracle: available RAM {have:.0f} MB (needs >= {RAM_PARITY_MB} MB)")
    if have < RAM_PARITY_MB:
        return None, f"not enough RAM for the oracle load ({have:.0f} MB)"
    os.makedirs(WORK, exist_ok=True)
    logf = open(os.path.join(WORK, "oracle.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([LLAMA, "-m", model, "--host", "127.0.0.1", "--port", str(PORT_ORACLE),
                             "-c", "4096", "-t", str(THREADS), "-ngl", "0"],
                            stdout=logf, stderr=subprocess.STDOUT)
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            return None, f"oracle exited with {proc.returncode} (see {os.path.join(WORK, 'oracle.log')})"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT_ORACLE}/health", timeout=5) as r:
                if r.status == 200 and json.loads(r.read()).get("status") in ("ok", "no slot available"):
                    log("  oracle: ready")
                    return proc, None
        except Exception:
            time.sleep(2)
    proc.kill()
    return None, "oracle did not become ready in 30 minutes"


def stop_oracle(proc):
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=30)
    except Exception:
        pass


def sc_score(model, ids, cont, tag):
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"score_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"prompt": ids, "continuation": cont}, f)
    r = subprocess.run([SCORE, "--model", model, "--input", path, "--ctx", "4096", "--threads", str(THREADS)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stderr[-2000:])
        raise RuntimeError(f"sc_score failed ({r.returncode}) for {tag}")
    return json.loads([x for x in r.stdout.splitlines() if x.startswith("{")][-1])


def parity(model, tok):
    log(f"(a) parity: {len(SCORE_LENGTHS)} prompts x {N_CONT} positions, sc_score vs unsloth llama.cpp CPU")
    proc, err = start_oracle(model)
    if proc is None:
        log(f"  SKIPPED: {err}")
        return None
    total_same = total_n = 0
    worst_all = 0.0
    try:
        for n in SCORE_LENGTHS:
            ids = prose_ids(tok, n)
            resp = oracle_request(ids, N_CONT)
            cps = (resp.get("completion_probabilities") or [])[:N_CONT]
            if not cps:
                log(f"  prompt {n} tok: SKIPPED: oracle returned no completion_probabilities "
                    f"({resp.get('error') or 'no field'})")
                return None
            if any(cp.get("id") is None for cp in cps):
                log(f"  prompt {n} tok: SKIPPED: oracle response has no token ids")
                return None
            otoks = [int(cp["id"]) for cp in cps]
            olps = [float(cp["logprob"]) if cp.get("logprob") is not None else None for cp in cps]
            mine = sc_score(model, ids, otoks, n)
            same = 0
            worst = 0.0
            for k, tok_id in enumerate(otoks):
                if mine["argmax"][k] == tok_id:
                    same += 1
                if olps[k] is not None:
                    worst = max(worst, abs(mine["top5"][k][0][1] - olps[k]))
            total_same += same
            total_n += len(otoks)
            worst_all = max(worst_all, worst)
            t = resp.get("timings") or {}
            log(f"  prompt {n:4d} tok: argmax {same}/{len(otoks)}, max |top-1 logprob diff| {worst:.6f}"
                f" (oracle prompt {t.get('prompt_n')} tok at {t.get('prompt_per_second')} tok/s)")
            if same < len(otoks):
                bad = [k for k in range(len(otoks)) if mine["argmax"][k] != otoks[k]]
                log(f"    argmax differs at positions {bad[:16]}")
    finally:
        stop_oracle(proc)
    frac = total_same / total_n if total_n else 0.0
    ok = frac >= GATE_ARGMAX
    log(f"parity total: argmax {total_same}/{total_n} = {100 * frac:.2f}% (gate >= {100 * GATE_ARGMAX:.0f}%), "
        f"max |top-1 logprob diff| {worst_all:.6f}  {'ok' if ok else 'FAIL'}")
    return ok


def pp_tg(text):
    """(pp mean, tg mean) of the `pp N: prefill X (...) tok/s | tg N: decode Y (...) tok/s` line."""
    m = re.search(r"pp\s+\d+: prefill ([\d.]+) \([\d.-]+\) tok/s \| tg \d+: decode ([\d.]+) \(", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def bench(model, tok, gate):
    log(f"(b) bench: sc_bench pp{PP} tg{TG} {REPS} reps ctx {CTX_BENCH}, decode gate >= {gate:g} tok/s")
    have = avail_mb()
    log(f"  available RAM {have:.0f} MB")
    if have < RAM_PARITY_MB:
        log(f"  SKIPPED: not enough RAM ({have:.0f} MB)")
        return None
    os.makedirs(WORK, exist_ok=True)
    ids = prose_ids(tok, PP)
    eog = getattr(tok, "_eog", None) or set()
    eos = sorted(int(x) for x in eog) or ([tok.eos_id] if getattr(tok, "eos_id", None) is not None else [])
    path = os.path.join(WORK, "bench_prompts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"eos": eos, "prompts": {str(PP): ids}}, f)
    r = subprocess.run([BENCH, "--model", model, "--prompts", path, "--pp", str(PP), "--tg", str(TG),
                        "--reps", str(REPS), "--ctx", str(CTX_BENCH), "--threads", str(THREADS)],
                       capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.strip():
            log("  " + line)
    if r.returncode != 0:
        log(f"  sc_bench failed with {r.returncode}")
        return False
    got = pp_tg(r.stdout)
    if not got:
        log("  could not read the pp/tg numbers")
        return False
    ok = got[1] >= gate
    log(f"  pp{PP}: {got[0]:.1f} tok/s, tg{TG}: {got[1]:.1f} tok/s  "
        f"{'ok' if ok else 'FAIL (gate ' + format(gate, 'g') + ' tok/s)'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--bench-gate", type=float, default=None)
    args = ap.parse_args()
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    model = args.model or ""
    if not model:
        print(sc_env.skip_line("SC_MODEL_INKLING"))
        return 0
    if not sc_env.get("SC_LLAMA_SERVER"):
        print(sc_env.skip_line("SC_LLAMA_SERVER"))
        return 0
    if not os.path.isfile(model):
        print(f"FAIL: model not found: {model}")
        return 1
    if not os.path.isfile(LLAMA):
        print(f"FAIL: oracle not found: {LLAMA} (SC_LLAMA_SERVER)")
        return 1
    if not os.path.isfile(SCORE) or not os.path.isfile(BENCH):
        print(f"FAIL: {SCORE} / {BENCH} missing; run build.bat first")
        return 1
    gate = args.bench_gate
    if gate is None:
        gate = 50.0 if "IQ3_XXS" in os.path.basename(model).upper() else 0.0
    from stratafreet.tokenizer import make_tokenizer
    tok = make_tokenizer(model)
    log(f"model: {model}")
    log(f"oracle: {LLAMA} (CPU, port {PORT_ORACLE})")
    take_lock("parity+bench")
    results = {}
    try:
        if not args.skip_parity:
            results["parity"] = parity(model, tok)
        if not args.skip_bench:
            results["bench"] = bench(model, tok, gate)
    finally:
        drop_lock()
    log("results: " + ", ".join(f"{k} {'SKIPPED' if v is None else ('PASS' if v else 'FAIL')}"
                                for k, v in results.items()))
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())