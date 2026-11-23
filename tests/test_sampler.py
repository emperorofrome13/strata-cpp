"""Acceptance: the native sampler (src/sampling.cpp) must match engine.py _sample exactly.

    python tests/test_sampler.py

Every check runs `scd_sampler_pick` (build/strata_cpp_dev.dll) — the model-free entry into the same
`strata::Sampler` the product DLL uses. The greedy cases are exact IEEE-f32 arithmetic (penalties + argmax);
the sampling cases use searched seeds whose first rand() lands far from every draw boundary, so the expected
pick is unambiguous. The xorshift64 state is checked exactly (pure integer math) after every pick.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DLL = os.path.join(ROOT, os.environ.get("SC_BUILD_DIR", "build"), "strata_cpp_dev.dll")

U64 = (1 << 64) - 1


def xorshift_once(rng: int) -> int:
    """One Sampler::rand step: rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17 (uint64 wrap)."""
    rng ^= (rng << 13) & U64
    rng ^= rng >> 7
    rng ^= (rng << 17) & U64
    return rng & U64


def find_seed(k_lo: int, k_hi: int, n: int = 1 << 24):
    """The first seed of the form 2^40 + i*2^20 whose first rand() value k = (rng' >> 40) lands in [k_lo, k_hi].

    Consecutive seeds give an identical first rand: the low bits of a small seed never reach the top 24 bits
    through three xorshift64 steps (they need i >= 2^23 via the <<17), so the scan strides 2^20, which perturbs
    the top bits immediately. Scanned with numpy uint64 (wrap-around like C), deterministic.
    """
    s = np.uint64(1 << 40) + np.arange(n, dtype=np.uint64) * np.uint64(1 << 20)
    s = s ^ ((s << np.uint64(13)) & np.uint64(U64))
    s = s ^ (s >> np.uint64(7))
    s = s ^ ((s << np.uint64(17)) & np.uint64(U64))
    r = (s >> np.uint64(40)).astype(np.int64)
    hits = np.flatnonzero((r >= k_lo) & (r <= k_hi))
    return int(np.uint64(1 << 40) + np.uint64(hits[0]) * np.uint64(1 << 20)) if hits.size else None


def load():
    if not os.path.isfile(DLL):
        raise SystemExit(f"missing {DLL}; run build.bat first")
    lib = ctypes.CDLL(DLL)
    lib.sci_last_error.restype = ctypes.c_char_p
    lib.scd_sampler_pick.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p,
                                     ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
    lib.scd_sampler_pick.restype = ctypes.c_int
    return lib


def pick(lib, logits, sampler_json, rng=None):
    """scd_sampler_pick: returns (rc, picked index, rng state after, error)."""
    arr = np.asarray(logits, dtype=np.float32)
    rng_in = ctypes.c_uint64(rng if rng is not None else 0)
    rng_ptr = ctypes.byref(rng_in) if rng is not None else None
    out = ctypes.c_uint32(0)
    rc = lib.scd_sampler_pick(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size,
                              sampler_json.encode(), rng_ptr, ctypes.byref(out))
    err = lib.sci_last_error().decode() if rc else ""
    return rc, out.value, rng_in.value, err


def main() -> int:
    lib = load()
    failures = []
    skipped = []
    ran = 0

    def check(name, cond, detail=""):
        nonlocal ran
        ran += 1
        if cond:
            print(f"ok   {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"FAIL {name}: {detail}")

    # ---- temp=0: plain argmax; ties keep the first index; a NaN is selected (total_cmp order) then rejected
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0], "{}")
    check("greedy argmax", rc == 0 and out == 1, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [6.0, 6.0, 4.0], "{}")
    check("greedy tie keeps the first index", rc == 0 and out == 0, f"rc={rc} out={out}")
    rc, out, _, err = pick(lib, [1.0, float("nan")], "{}")
    check("greedy NaN sorts above every number (then errors)", rc == 1 and "nan" in err.lower(),
          f"rc={rc} err={err!r}")
    rc, out, _, err = pick(lib, [float("nan"), 5.0, 4.0], "{}")
    check("greedy NaN before the max also errors", rc == 1 and "nan" in err.lower(), f"rc={rc} err={err!r}")

    # ---- temp=0 with penalties: engine.py _sample order — repeat on the unique recent ids, then presence
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"repeat_penalty":2.0,"recent":[1]}')
    check("repeat divides positive logits (10/2=5 < 6)", rc == 0 and out == 2, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, -10.0, 6.0], '{"temp":0,"repeat_penalty":2.0,"recent":[1]}')
    check("repeat multiplies non-positive logits (-10*2)", rc == 0 and out == 2, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, 5.0, 6.0], '{"temp":0,"presence_penalty":2.0,"recent":[0]}')
    check("presence subtracts from the recent logits (4-2=2)", rc == 0 and out == 2, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, 10.0, 5.5],
                         '{"temp":0,"repeat_penalty":2.0,"presence_penalty":1.0,"recent":[1]}')
    check("repeat then presence on one id (10/2-1=4 < 5.5)", rc == 0 and out == 2, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [-1.25, -0.25, -2.0],
                         '{"temp":0,"repeat_penalty":2.0,"presence_penalty":0.5,"recent":[1]}')
    check("repeat runs before presence (-0.25*2-0.5=-1.0 > -1.25; presence first would give -1.5)",
          rc == 0 and out == 1, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"repeat_penalty":0.0,"recent":[1]}')
    check("repeat_penalty 0 is off (engine.py truthiness)", rc == 0 and out == 1, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"repeat_penalty":2.0,"recent":[1,1,1]}')
    check("duplicate recent ids penalize once", rc == 0 and out == 2, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"repeat_penalty":2.0,"recent":[99]}')
    check("out-of-range recent ids are skipped", rc == 0 and out == 1, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"repeat_penalty":2.0}')
    check("penalties without recent are ignored (engine.py parity)", rc == 0 and out == 1, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"repeat_penalty":2.0,"recent":[]}')
    check("empty recent ignores penalties", rc == 0 and out == 1, f"rc={rc} out={out}")

    # ---- validation: the sampler JSON is rejected outside its ranges
    for name, js in [
        ("negative temp", '{"temp":-1}'),
        ("negative top_k", '{"top_k":-1}'),
        ("zero top_p", '{"top_p":0}'),
        ("top_p above one", '{"top_p":1.5}'),
        ("negative min_p", '{"min_p":-0.1}'),
        ("min_p above one", '{"min_p":1.1}'),
        ("negative repeat_penalty", '{"repeat_penalty":-1}'),
        ("non-finite presence_penalty", '{"presence_penalty":inf}'),
        ("recent not an array", '{"recent":5}'),
        ("malformed recent", '{"recent":[1,2}'),
        ("recent longer than 4096", '{"recent":[' + ",".join(["0"] * 4097) + "]}"),
    ]:
        rc, out, _, err = pick(lib, [0.0, 1.0], js)
        check(f"rejects {name}", rc == 1 and err != "", f"rc={rc} err={err!r}")
    rc, out, _, _ = pick(lib, [4.0, 10.0, 6.0],
                         '{"temp":0,"repeat_penalty":2.0,"recent":[' + ",".join(["1"] * 4096) + "]}")
    check("recent of exactly 4096 ids is accepted", rc == 0 and out == 2, f"rc={rc} out={out}")

    # ---- the xorshift64 state: exact after every pick (integer math, no float rounding involved)
    rc, out, rng_out, _ = pick(lib, [4.0, 10.0, 6.0], "{}", rng=5)
    check("temp=0 consumes no rand", rc == 0 and rng_out == 5, f"rc={rc} rng_out={rng_out}")
    rc, out, rng_out, _ = pick(lib, [4.0, 10.0, 6.0], "{}", rng=0)
    check("rng=0 defaults to 1", rc == 0 and rng_out == 1, f"rc={rc} rng_out={rng_out}")
    rc, out, rng_out, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"seed":42}', rng=5)
    check("seed re-seeds over the rng pointer", rc == 0 and rng_out == 42, f"rc={rc} rng_out={rng_out}")
    rc, out, rng_out, _ = pick(lib, [4.0, 10.0, 6.0], '{"temp":0,"seed":0.5}', rng=5)
    check("seed below 1 re-seeds to 1", rc == 0 and rng_out == 1, f"rc={rc} rng_out={rng_out}")
    rc, out, rng_out, _ = pick(lib, [0.0, math.log(0.5)], '{"temp":1,"top_k":1,"seed":42}', rng=5)
    check("seed overrides the rng pointer (sampling)", rc == 0 and out == 0 and rng_out == xorshift_once(42),
          f"rc={rc} out={out} rng_out={rng_out}")
    rc, out, rng_out, _ = pick(lib, [0.0, math.log(0.5)], '{"temp":1,"top_k":1}', rng=5)
    check("one rand per sampling pick", rc == 0 and out == 0 and rng_out == xorshift_once(5),
          f"rc={rc} out={out} rng_out={rng_out}")

    # ---- sampling: the candidate cuts, observed through draws with searched seeds
    s9 = find_seed(int(0.89 * (1 << 24)), int(0.91 * (1 << 24)))
    if s9 is None:
        skipped.append("no seed with rand in [0.89, 0.91]")
    else:
        # probabilities exp(logit - max): [1, 0.6, 0.3]
        logits = [0.0, math.log(0.6), math.log(0.3)]
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"min_p":0.25,"seed":{s9}}}')
        check("min_p 0.25 keeps all three candidates", rc == 0 and out == 2, f"rc={rc} out={out}")
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"min_p":0.5,"seed":{s9}}}')
        check("min_p 0.5 drops the 0.3 candidate", rc == 0 and out == 1, f"rc={rc} out={out}")
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"min_p":0.7,"seed":{s9}}}')
        check("min_p 0.7 keeps only the max", rc == 0 and out == 0, f"rc={rc} out={out}")
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"top_k":2,"seed":{s9}}}')
        check("top_k trims to the two largest", rc == 0 and out == 1, f"rc={rc} out={out}")
        # probabilities [1, 0.5, 0.25]
        logits = [0.0, math.log(0.5), math.log(0.25)]
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"top_p":0.5,"seed":{s9}}}')
        check("top_p 0.5 keeps only the max", rc == 0 and out == 0, f"rc={rc} out={out}")
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"top_p":0.8,"seed":{s9}}}')
        check("top_p 0.8 keeps the top two", rc == 0 and out == 1, f"rc={rc} out={out}")
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"top_p":0.88,"seed":{s9}}}')
        check("top_p 0.88 keeps all three", rc == 0 and out == 2, f"rc={rc} out={out}")
        # tie order: equal probabilities keep index order, and the draw can tell
        logits = [0.0, 0.0, math.log(0.5)]
        rc, out, _, _ = pick(lib, logits, f'{{"temp":1,"top_k":2,"seed":{s9}}}')
        check("equal probabilities sort by index (tie)", rc == 0 and out == 1, f"rc={rc} out={out}")

    s1 = find_seed(16776713, 16777048)  # rand in [1-3e-5, 1-1e-5]: past the max, but far from the boundary
    if s1 is None:
        skipped.append("no seed with rand above 1 - 1.3e-5")
    else:
        # probabilities [1, exp(-10) ≈ 4.54e-5]: the walk passes the max and lands on the last candidate
        rc, out, _, _ = pick(lib, [0.0, -10.0], f'{{"temp":1,"seed":{s1}}}')
        check("the draw falls through to the last candidate when rand is near one", rc == 0 and out == 1,
              f"rc={rc} out={out}")
        # probabilities [1, exp(-19.5) ≈ 3.4e-9]: too small for any rand to reach — the max always wins
        rc, out, _, _ = pick(lib, [0.0, -19.5], f'{{"temp":1,"seed":{s1}}}')
        check("a candidate with probability below one rand step cannot win", rc == 0 and out == 0,
              f"rc={rc} out={out}")

    s97 = find_seed(int(0.965 * (1 << 24)), int(0.975 * (1 << 24)))
    if s97 is None:
        skipped.append("no seed with rand in [0.965, 0.975]")
    else:
        # repeat_penalty=2 on the recent id 1 turns 5 into 2.5: the walk then falls through the max to the
        # last candidate; without the penalty the 5.0 max wins the walk outright. The penalty changed the pick.
        rc, out, _, _ = pick(lib, [0.0, 5.0], f'{{"temp":1,"repeat_penalty":2.0,"recent":[1],"seed":{s97}}}')
        check("penalties land before the candidate cut (5/2=2.5 loses the walk to the last candidate)",
              rc == 0 and out == 0, f"rc={rc} out={out}")
        rc, out, _, _ = pick(lib, [0.0, 5.0], f'{{"temp":1,"seed":{s97}}}')
        check("without the penalty the max wins outright", rc == 0 and out == 1, f"rc={rc} out={out}")

    # top_k=1 is the argmax regardless of rand; NaN logits are skipped by the stability window (>= fails)
    rc, out, _, _ = pick(lib, [0.0, math.log(0.5), math.log(0.25)], '{"temp":1,"top_k":1,"seed":42}')
    check("top_k=1 is the argmax", rc == 0 and out == 0, f"rc={rc} out={out}")
    rc, out, _, _ = pick(lib, [0.0, float("nan")], '{"temp":1,"seed":42}')
    check("NaN logits are skipped when sampling", rc == 0 and out == 0, f"rc={rc} out={out}")

    for s in skipped:
        print(f"SKIP {s}")
    if failures:
        print()
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print(f"\nPASS: {ran} sampler checks exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
