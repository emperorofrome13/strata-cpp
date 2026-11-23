"""test_partial_14b.py - strata-cpp against RustingStrata 1.2.5 on the Qwen3.6-14B-A3B MoE with PARTIAL expert residency:
the routed experts that do not fit in VRAM run on the CPU expert worker (cpu_worker / cpu_moe), on both sides, with the
same VRAM margin and the same CPU thread count.

1. Parity: three prompts x 64 continuation positions. `build\\sc_score.exe --vram-margin-mb M --threads T` and the
   official `rustingstrata score --gpu --vram-margin M --threads T` read the same JSON. Gate: the same argmax on >= 97% of
   the 192 positions (CPU and GPU experts round differently, so placement can flip near-ties; the exact numbers are
   printed). The top-1 logprob difference is reported.
2. Speed: 128 greedy tokens after a 16-token prompt decoded by the DLL (`sc_decode_sample`, `{"temp":0}`, timed after an
   8-token warm-up) against `rustingstrata bench --gpu --pp 16 --tg 128 --reps 3 --ctx 4096 --threads T --vram-margin M`.
   Gate: our decode >= 95% of the official mean. Prints hit rate, swaps and worker busy time per token from sc_stats, and
   the official bench's own hit / worker lines.
3. Profile round trip: the routing profile both sides write is the same file (profile-<arch>-<L>x<E>-<n>.f32 under
   %LOCALAPPDATA%\\rusting_strata). Each load starts from the same saved profile and must resident the same number of
   experts as the official binary's "resident experts N/M" line. The profile file size is checked (4 bytes per count).

Each process loads the 14B model (about 10 GB), so the runs go one after another. Needs the GPU, SC_MODEL_14B,
SC_RUSTINGSTRATA (the official 1.2.5 binary) and STRATAFREET_ROOT (its tokenizer).
Usage: python tests/test_partial_14b.py [--margin 8500] [--threads 7] [--skip-parity]
       [--skip-speed] [--skip-profile] [--model PATH]

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).
"""
import argparse
import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_env  # noqa: E402
from test_score_14b import (DEFAULT_MODEL, DLL, PORT, PROMPTS, RUST, SCORE, N_POS, make_inputs,  # noqa: E402
                            compare)

WORK = os.path.join(sc_env.build_dir(PORT), "partial_test")
GATE_ARGMAX = 0.97
GATE_DECODE = 0.95
SPEED_TOKENS = 128


def run_cmd(cmd):
    """Runs a command; returns (stdout, stderr). Raises on a nonzero exit with the stderr tail."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} exited {r.returncode}: {r.stderr[-800:]}")
    return r.stdout, r.stderr


def run_json(cmd):
    out, err = run_cmd(cmd)
    line = [x for x in out.splitlines() if x.startswith("{")][-1]
    return json.loads(line), err


def load_dll():
    d = ctypes.CDLL(DLL)
    d.sc_load.restype = ctypes.c_void_p
    d.sc_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    d.sc_decode_sample.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
    d.sc_decode_sample.restype = ctypes.c_int
    d.sc_last_error.restype = ctypes.c_char_p
    d.sc_free.argtypes = [ctypes.c_void_p]
    d.sc_stats.restype = ctypes.c_char_p
    d.sc_stats.argtypes = [ctypes.c_void_p]
    d.sc_save_profile.argtypes = [ctypes.c_void_p]
    d.sc_save_profile.restype = ctypes.c_int
    return d


def load_opts(margin, threads):
    return json.dumps({"ctx": 4096, "kv": "f16", "vram_margin_mb": margin, "threads": threads}).encode()


def profile_path_of(d, model, margin, threads):
    """Loads once without decoding (records nothing, so it writes nothing) and returns the stats at load."""
    h = d.sc_load(model.encode(), load_opts(margin, threads))
    if not h:
        raise RuntimeError(d.sc_last_error().decode())
    stats = json.loads(d.sc_stats(h).decode())
    d.sc_free(h)
    return stats


def official_resident(stderr):
    m = re.findall(r"resident experts (\d+)/(\d+)", stderr)
    if not m:
        raise RuntimeError("the official binary printed no 'resident experts N/M' line")
    return int(m[-1][0]), int(m[-1][1])


def snapshot(path, dest):
    if os.path.exists(path):
        shutil.copyfile(path, dest)
        return True
    return False


def restore(path, src, existed):
    if existed:
        shutil.copyfile(src, path)
    elif os.path.exists(path):
        os.remove(path)


def n_expert_of(path):
    """Experts per layer from the profile name profile-<arch>-<L>x<E>-<n>.f32 (the stats do not carry it)."""
    m = re.search(r"-(\d+)x(\d+)-\d+\.f32$", path)
    return int(m.group(2))


def check_profile_format(path, n_layer, n_expert):
    size = os.path.getsize(path)
    ok = size >= 4 * n_layer * n_expert and size % 4 == 0
    return ok, size


def parity(model, margin, threads, cases, prof, snap0, existed0):
    os.makedirs(WORK, exist_ok=True)
    total_same = total_n = 0
    worst_all = 0.0
    print(f"parity: {len(cases)} prompts x {N_POS} continuation positions, margin {margin} MB, threads {threads}")
    for i, c in enumerate(cases):
        path = os.path.join(WORK, f"input{i + 1}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"prompt": c["prompt"], "continuation": c["continuation"]}, f)
        restore(prof, snap0, existed0)
        t = time.time()
        mine, _ = run_json([SCORE, "--model", model, "--input", path, "--vram-margin-mb", str(margin),
                            "--threads", str(threads)])
        t_mine = time.time() - t
        restore(prof, snap0, existed0)
        t = time.time()
        theirs, _ = run_json([RUST, "score", "--model", model, "--input", path, "--gpu", "--vram-margin", str(margin),
                              "--threads", str(threads)])
        t_theirs = time.time() - t
        same, n, worst = compare(mine, theirs)
        total_same += same
        total_n += n
        worst_all = max(worst_all, worst)
        print(f"  prompt {i + 1} ({len(c['prompt'])} tok, {c['text']!r}): argmax {same}/{n}, "
              f"max |top-1 logprob diff| {worst:.6f}   [sc_score {t_mine:.0f} s, rustingstrata {t_theirs:.0f} s]")
        if same < n:
            bad = [k for k in range(n) if mine["argmax"][k] != theirs["argmax"][k]]
            print(f"    argmax differs at positions {bad[:12]}")
    frac = total_same / total_n
    print(f"parity total: argmax {total_same}/{total_n} = {100 * frac:.2f}% (gate >= {100 * GATE_ARGMAX:.0f}%), "
          f"max |top-1 logprob diff| {worst_all:.6f} (reported, no gate)")
    return frac >= GATE_ARGMAX


def our_speed(d, model, margin, threads, prompt):
    """One load: the stats right after it (the initial placement), then 16 prompt tokens, 8 warm-up tokens and
    SPEED_TOKENS timed greedy tokens. Writes the profile this run recorded (sc_free would do the same)."""
    h = d.sc_load(model.encode(), load_opts(margin, threads))
    if not h:
        raise RuntimeError(d.sc_last_error().decode())
    stats_load = json.loads(d.sc_stats(h).decode())
    out = ctypes.c_int32(0)
    greedy = b'{"temp":0}'

    def step(tok):
        if d.sc_decode_sample(h, tok, greedy, ctypes.byref(out)) != 0:
            raise RuntimeError(d.sc_last_error().decode())
        return out.value

    tok = 0
    for t in prompt:
        tok = step(t)
    for _ in range(8):
        tok = step(tok)
    t0 = time.perf_counter()
    for _ in range(SPEED_TOKENS):
        tok = step(tok)
    dt = time.perf_counter() - t0
    stats = json.loads(d.sc_stats(h).decode())
    if d.sc_save_profile(h) != 0:
        raise RuntimeError(d.sc_last_error().decode())
    d.sc_free(h)
    return SPEED_TOKENS / dt, stats, stats_load


def official_bench(model, margin, threads):
    r = subprocess.run([RUST, "bench", "--model", model, "--gpu", "--pp", "16", "--tg", str(SPEED_TOKENS), "--reps", "3",
                        "--ctx", "4096", "--threads", str(threads), "--vram-margin", str(margin)],
                       capture_output=True, text=True)
    text = r.stdout + r.stderr
    if r.returncode != 0:
        raise RuntimeError(f"rustingstrata bench exited {r.returncode}: {text[-800:]}")
    m = re.search(r"tg\s+%d:\s+decode\s+([\d.]+)" % SPEED_TOKENS, text)
    if not m:
        raise RuntimeError("no 'tg 128: decode' line in the official bench output")
    print(f"rustingstrata bench --gpu --pp 16 --tg {SPEED_TOKENS} --reps 3 --ctx 4096 --threads {threads} "
          f"--vram-margin {margin}:")
    for line in text.splitlines():
        if re.search(r"tg\s+\d+:|pp\s+\d+:|hit|worker|busy|resident|swap|expert traffic", line, re.I) and not line.startswith("cuda"):
            print("  " + line.strip())
    return float(m.group(1)), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--margin", type=int, default=8500)
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
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
    margin, threads, model = args.margin, args.threads, model
    cases, speed_prompt = make_inputs(model)
    os.makedirs(WORK, exist_ok=True)
    ok = True

    d = load_dll()
    base = profile_path_of(d, model, margin, threads)
    prof = base["profile"]
    print(f"strata-cpp partial residency: margin {margin} MB, threads {threads}; at load: resident "
          f"{base['experts_resident']}/{base['experts_total']} experts ({base['expert_mb']} MB of slots); profile {prof}")
    # The profile is shared with the official binary (same file, same format): every run starts from the snapshot taken
    # here, so both sides load the same placement, and the user's file is put back at the end.
    snap0 = os.path.join(WORK, "profile_before.f32")
    existed0 = snapshot(prof, snap0)
    try:
        if not args.skip_parity:
            ok = parity(model, margin, threads, cases, prof, snap0, existed0) and ok

        if not args.skip_speed:
            restore(prof, snap0, existed0)
            bench_tps, bench_text = official_bench(model, margin, threads)
            off_n = official_resident(bench_text)
            print(f"official bench: decode mean {bench_tps:.1f} tok/s; resident experts {off_n[0]}/{off_n[1]} "
                  f"(ours at load {base['experts_resident']}/{base['experts_total']})")
            restore(prof, snap0, existed0)
            tps, stats, stats_load = our_speed(d, model, margin, threads, speed_prompt)
            frac = tps / bench_tps
            print(f"decode: ours {tps:.1f} tok/s (sc_decode_sample greedy, {SPEED_TOKENS} tokens) vs official "
                  f"{bench_tps:.1f} tok/s = {100 * frac:.1f}% (gate >= {100 * GATE_DECODE:.0f}%)")
            print(f"  sc_stats: hit rate {stats['hit_rate']} (hits {stats['hits']}, misses {stats['misses']}), swaps "
                  f"{stats['swaps']}, worker busy {stats['worker_busy_ms_per_token']} ms/token (total "
                  f"{stats['worker_busy_ms']:.0f} ms), mailbox writes {stats['mailbox_writes']}, reads "
                  f"{stats['mailbox_reads']}, resident {stats['experts_resident']}/{stats['experts_total']}")
            if frac < GATE_DECODE:
                ok = False
                print("  decode gate FAILED")
            ours_snap = os.path.join(WORK, "profile_ours.f32")
            ours_written = snapshot(prof, ours_snap)
        else:
            ours_written = False

        if not args.skip_profile:
            # Round trip: the profile our run wrote (Strata's format) must place the same experts for the official binary
            # and for us, both loaded from that file.
            if not ours_written:
                print("profile round trip needs the speed run (drop --skip-speed)")
                ok = False
            else:
                size_ok, size = check_profile_format(ours_snap, base["n_layer"], n_expert_of(prof))
                print(f"profile written by our run: {size} bytes (4 per count), format ok: {size_ok}")
                ok = ok and size_ok
                restore(prof, ours_snap, True)
                fresh = profile_path_of(d, model, margin, threads)
                restore(prof, ours_snap, True)
                _, off_err = run_cmd([RUST, "score", "--model", model, "--input", os.path.join(WORK, "input1.json"),
                                      "--gpu", "--vram-margin", str(margin), "--threads", str(threads)])
                off_n = official_resident(off_err)
                same = fresh["experts_resident"] == off_n[0]
                print(f"initial placement from that profile: ours {fresh['experts_resident']}/{fresh['experts_total']}, "
                      f"official {off_n[0]}/{off_n[1]} -> {'match' if same else 'MISMATCH'}")
                ok = ok and same
    finally:
        restore(prof, snap0, existed0)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
