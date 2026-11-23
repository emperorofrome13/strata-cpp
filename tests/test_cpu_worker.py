"""Test of the CPU mailbox worker (src\\cpu_worker.cpp) through build\\strata_cpu_test.dll.

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).

Python plays the GPU's role. For each token it calls sctw_begin for the sequence number, then for each MoE layer
it writes the routing (ids, weights, slots; slot -1 = missed) and the activation into the mailbox buffer that the
worker was created over, raises REQ (or SKIP when nothing missed), waits for DONE, and reads y. Every wait has a
timeout, so a bug fails the test instead of hanging it.

Checks on the Qwen3.6-14B-A3B Q6_K model (the small model of PORT_MAP section 8):
  1. Layout: the mailbox offsets and stride equal the Rust Mailbox::new formula.
  2. Single-token requests (3 tokens, 4 layers, one layer per token with no miss raising SKIP): y of each served
     layer is bit-identical to sct_experts over the missed experts only; the skipped layer is not computed (its
     DONE stays unset and its y keeps the sentinel); routing(il) returns what was written.
  3. Stale request: a seq that the GPU never answered is passed over once a newer seq is written for every layer;
     the newer seq is answered and bit-identical. A wait on the stale seq times out (no hang).
  4. stalled_layer / the verdict reset in wait_done.
  5. NTOK = 3 (verify batch): y rows are bit-identical to sct_experts_multi over the missed experts; a token with
     no miss leaves its row untouched.
  6. busy() grows with the work.
  7. STRATA_TEST_WORKER_DELAY: a worker created with the switch answers the (seq, layer) pairs it selects late,
     and the answers are still bit-identical.
  8. stop() and the free join the worker thread (each call must return within a timeout).

Run:  python tests/test_cpu_worker.py   (after build_cpu.bat)

Needs SC_MODEL_14B. CPU only. Without the model the test prints SKIP and exits 0.
"""
import ctypes
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sc_env  # noqa: E402

DLL = os.path.join(ROOT, os.environ.get("SC_BUILD_DIR", "build"), "strata_cpu_test.dll")
MODEL = sc_env.get("SC_MODEL_14B")

TIMEOUT_MS = 10000          # every wait on the worker
JOIN_SECS = 10.0            # every stop/free must return within this
LAYERS = [0, 1, 2, 3]       # the MoE layers of the worker
THREADS = 4                 # the worker's expert pool
SENT = None                 # sentinel in y (float32 7.25): a layer the worker did not compute keeps it
DELAY_MS = 300

# Imported in main() after the SKIP gate.
np = None

# word offsets of cpu_worker.h
REQ, NTOK, SKIP, VERDICT, DONE, IDS = 0, 1, 2, 8, 16, 32
TMAX = 17

P_F = ctypes.POINTER(ctypes.c_float)
P_I32 = ctypes.POINTER(ctypes.c_int32)
P_I64 = ctypes.POINTER(ctypes.c_int64)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def load_lib():
    lib = ctypes.CDLL(DLL)
    lib.sct_open.argtypes, lib.sct_open.restype = [ctypes.c_char_p], ctypes.c_void_p
    lib.sct_close.argtypes, lib.sct_close.restype = [ctypes.c_void_p], None
    lib.sct_last_error.argtypes, lib.sct_last_error.restype = [], ctypes.c_char_p
    lib.sct_info.argtypes, lib.sct_info.restype = [ctypes.c_void_p, P_I64], ctypes.c_int
    lib.sct_experts.argtypes = [ctypes.c_void_p, ctypes.c_int, P_I32, P_F, ctypes.c_int, P_F, P_F, ctypes.c_int]
    lib.sct_experts.restype = ctypes.c_int
    lib.sct_experts_multi.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, P_I32, P_F, ctypes.c_int, P_F, P_F,
                                      ctypes.c_int]
    lib.sct_experts_multi.restype = ctypes.c_int
    lib.sctw_create.argtypes, lib.sctw_create.restype = [ctypes.c_void_p, P_I32, ctypes.c_int, ctypes.c_int], ctypes.c_void_p
    lib.sctw_free.argtypes, lib.sctw_free.restype = [ctypes.c_void_p], None
    lib.sctw_stop.argtypes, lib.sctw_stop.restype = [ctypes.c_void_p], None
    lib.sctw_layout.argtypes, lib.sctw_layout.restype = [ctypes.c_void_p, P_I64], ctypes.c_int
    lib.sctw_begin.argtypes, lib.sctw_begin.restype = [ctypes.c_void_p], ctypes.c_uint32
    lib.sctw_routing.argtypes = [ctypes.c_void_p, ctypes.c_int, P_I32, P_F, P_I32, ctypes.c_int]
    lib.sctw_routing.restype = ctypes.c_int
    lib.sctw_stalled_layer.argtypes, lib.sctw_stalled_layer.restype = [ctypes.c_void_p], ctypes.c_int
    lib.sctw_wait_done.argtypes, lib.sctw_wait_done.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
    lib.sctw_busy_ns.argtypes, lib.sctw_busy_ns.restype = [ctypes.c_void_p], ctypes.c_uint64
    return lib


def last_error(lib):
    e = lib.sct_last_error()
    return e.decode("utf-8", "replace") if e else ""


def fptr(a):
    return a.ctypes.data_as(P_F)


def iptr(a):
    return a.ctypes.data_as(P_I32)


def round16(x):
    return (x + 15) // 16 * 16


class Box:
    """The mailbox buffer, viewed through numpy (the GPU's side of the memory)."""

    def __init__(self, lib, wh):
        out = (ctypes.c_int64 * 7)()
        check(lib.sctw_layout(wh, out) == 0, "sctw_layout")
        self.k, self.n, self.h, self.y, self.stride, addr, nbytes = [int(v) for v in out]
        check(addr % 64 == 0, f"mailbox buffer is 64-byte aligned (address {addr:#x})")
        self.nwords = nbytes // 4
        self.U = np.ctypeslib.as_array((ctypes.c_uint32 * self.nwords).from_address(addr))
        self.I = self.U.view(np.int32)
        self.F = self.U.view(np.float32)

    def base(self, il):
        return il * self.stride


def write_request(box, il, nt, ids, ws, slots, hs):
    """The GPU's request words of layer il (no flag yet): ids, weights, slots, activations, NTOK."""
    b = box.base(il)
    kt = box.k * nt
    box.U[b + IDS:b + IDS + kt] = ids.astype(np.uint32)
    box.F[b + IDS + kt:b + IDS + 2 * kt] = ws.astype(np.float32)
    box.I[b + IDS + 2 * kt:b + IDS + 3 * kt] = slots.astype(np.int32)
    box.F[b + box.h:b + box.h + nt * box.n] = hs.reshape(-1).astype(np.float32)
    box.U[b + NTOK] = nt


def raise_flag(box, il, seq, skip=False):
    box.U[box.base(il) + (SKIP if skip else REQ)] = seq


def make_request(box, nt, miss, rng, n_expert):
    """Random routing for nt tokens. miss[t] lists the positions (0..k-1) of token t whose expert is missed."""
    k = box.k
    ids = np.empty(nt * k, np.int32)
    ws = np.empty(nt * k, np.float32)
    slots = rng.integers(0, 64, nt * k).astype(np.int32)
    for t in range(nt):
        ids[t * k:(t + 1) * k] = rng.choice(n_expert, k, replace=False)
        ws[t * k:(t + 1) * k] = rng.uniform(0.05, 0.5, k)
        for p in miss[t]:
            slots[t * k + p] = -1
    hs = (rng.standard_normal((nt, box.n)) * 0.5).astype(np.float32)
    return ids, ws, slots, hs


def missed_of(box, ids, ws, slots, t):
    sl = slice(t * box.k, (t + 1) * box.k)
    m = slots[sl] < 0
    return ids[sl][m].astype(np.int32), ws[sl][m].astype(np.float32)


def ref_one(lib, model, il, ids, ws, x, n):
    """sct_experts over the given (missed) experts only."""
    ids = np.ascontiguousarray(ids, np.int32)
    ws = np.ascontiguousarray(ws, np.float32)
    x = np.ascontiguousarray(x, np.float32)
    y = np.zeros(n, np.float32)
    rc = lib.sct_experts(model, il, iptr(ids), fptr(ws), len(ids), fptr(x), fptr(y), 0)
    if rc != 0:
        raise AssertionError(f"sct_experts failed: {last_error(lib)}")
    return y


def ref_multi(lib, model, il, ids2, ws2, xs, n):
    """sct_experts_multi over nt tokens, each with the same number of missed experts."""
    nt, m = ids2.shape
    ids = np.ascontiguousarray(ids2.reshape(-1), np.int32)
    ws = np.ascontiguousarray(ws2.reshape(-1), np.float32)
    xs = np.ascontiguousarray(xs, np.float32)
    y = np.zeros(nt * n, np.float32)
    rc = lib.sct_experts_multi(model, il, nt, iptr(ids), fptr(ws), m, fptr(xs), fptr(y), 0)
    if rc != 0:
        raise AssertionError(f"sct_experts_multi failed: {last_error(lib)}")
    return y.reshape(nt, n)


def same_bits(a, b):
    a = np.ascontiguousarray(a, np.float32)
    b = np.ascontiguousarray(b, np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def wait_layer(lib, wh, il, what):
    rc = lib.sctw_wait_done(wh, il, TIMEOUT_MS)
    if rc != 0:
        raise AssertionError(f"{what}: layer {il} was not answered: {last_error(lib)}")


def call_with_timeout(fn, args, what):
    th = threading.Thread(target=fn, args=args, daemon=True)
    th.start()
    th.join(JOIN_SECS)
    check(not th.is_alive(), f"{what} returned within {JOIN_SECS:.0f} s (the worker thread joined)")


def check_routing(lib, wh, box, il, ids, ws, slots):
    cap = box.k
    bi = (ctypes.c_int32 * cap)()
    bw = (ctypes.c_float * cap)()
    bs = (ctypes.c_int32 * cap)()
    cnt = lib.sctw_routing(wh, il, bi, bw, bs, cap)
    check(cnt == box.k, f"routing({il}) returns k = {box.k} entries (got {cnt})")
    ri = np.ctypeslib.as_array(bi)
    rw = np.ctypeslib.as_array(bw)
    rs = np.ctypeslib.as_array(bs)
    check(np.array_equal(ri, ids) and same_bits(rw, ws) and np.array_equal(rs, slots),
          f"routing({il}) equals the ids, weights and slots written")


def test_layout(box):
    k, n = box.k, box.n
    h = round16(32 + 3 * TMAX * k)
    y = h + TMAX * n
    stride = round16(h + 2 * TMAX * n)
    check((box.h, box.y, box.stride) == (h, y, stride),
          f"layout h={box.h} y={box.y} stride={box.stride} equals Mailbox::new (k={k}, n={n})")


def test_single_tokens(lib, model, wh, box, rng, n_expert):
    n, k = box.n, box.k
    served_total = 0
    for tok in range(3):
        seq = lib.sctw_begin(wh)
        skip_il = LAYERS[tok % len(LAYERS)]
        skipped = []
        for il in LAYERS:
            b = box.base(il)
            box.F[b + box.y:b + box.y + n] = SENT
            if il == skip_il:
                raise_flag(box, il, seq, skip=True)
                skipped.append(il)
                continue
            miss = [sorted(rng.choice(k, 2, replace=False).tolist())]
            ids, ws, slots, hs = make_request(box, 1, miss, rng, n_expert)
            write_request(box, il, 1, ids, ws, slots, hs)
            raise_flag(box, il, seq)
            wait_layer(lib, wh, il, f"token {tok}")
            y = box.F[b + box.y:b + box.y + n].copy()
            mids, mws = missed_of(box, ids, ws, slots, 0)
            ref = ref_one(lib, model, il, mids, mws, hs[0], n)
            check(same_bits(y, ref), f"token {tok} layer {il}: y is bit-identical to sct_experts over {len(mids)} missed experts")
            check_routing(lib, wh, box, il, ids, ws, slots)
            served_total += 1
        for il in skipped:
            b = box.base(il)
            check(int(box.U[b + DONE]) != seq, f"token {tok} layer {il} (no miss, SKIP): DONE not written")
            check(bool(np.all(box.F[b + box.y:b + box.y + n] == SENT)),
                  f"token {tok} layer {il} (no miss, SKIP): y untouched (not computed)")
    return served_total


def test_stale(lib, model, wh, box, rng, n_expert):
    n = box.n
    seq_a = lib.sctw_begin(wh)   # the GPU never answers this one
    rc = lib.sctw_wait_done(wh, LAYERS[0], 200)
    check(rc != 0 and "timed out" in last_error(lib), f"wait on unanswered seq {seq_a} times out (no hang)")
    seq_b = lib.sctw_begin(wh)
    reqs = {}
    for il in LAYERS:   # the GPU wrote every layer of seq_b before waiting; the stale seq_a is passed over
        miss = [[1, 4]]
        ids, ws, slots, hs = make_request(box, 1, miss, rng, n_expert)
        write_request(box, il, 1, ids, ws, slots, hs)
        raise_flag(box, il, seq_b)
        reqs[il] = (ids, ws, slots, hs)
    for il in LAYERS:
        wait_layer(lib, wh, il, f"newer seq {seq_b}")
        ids, ws, slots, hs = reqs[il]
        y = box.F[box.base(il) + box.y:box.base(il) + box.y + n].copy()
        mids, mws = missed_of(box, ids, ws, slots, 0)
        check(same_bits(y, ref_one(lib, model, il, mids, mws, hs[0], n)),
              f"stale seq {seq_a} skipped, seq {seq_b} layer {il} bit-identical")


def test_stalled(lib, box, wh):
    il = LAYERS[2]
    box.U[box.base(il) + VERDICT] = 2
    check(lib.sctw_stalled_layer(wh) == il, f"stalled_layer() reports layer {il} with verdict 2")
    wait_layer(lib, wh, il, "verdict reset")   # DONE of the current seq is already set
    check(lib.sctw_stalled_layer(wh) == -1 and int(box.U[box.base(il) + VERDICT]) == 0,
          "wait_done resets the verdict; stalled_layer() is none")


def play_token(lib, wh, box, seq, reqs):
    """One token as the GPU plays it: every layer of LAYERS in order. reqs maps a layer to (nt, ids, ws, slots, hs)
    for a layer that missed; the other layers raise SKIP. Each served layer is awaited before the next one, and its
    y rows (nt x n) are returned, with the sentinel written first so an unwritten row shows."""
    n = box.n
    out = {}
    for il in LAYERS:
        if il not in reqs:
            raise_flag(box, il, seq, skip=True)
            continue
        nt, ids, ws, slots, hs = reqs[il]
        b = box.base(il)
        box.F[b + box.y:b + box.y + nt * n] = SENT
        write_request(box, il, nt, ids, ws, slots, hs)
        raise_flag(box, il, seq)
        wait_layer(lib, wh, il, f"seq {seq}")
        out[il] = box.F[b + box.y:b + box.y + nt * n].reshape(nt, n).copy()
    return out


def test_multi(lib, model, wh, box, rng, n_expert):
    n, k = box.n, box.k
    if k < 6:
        raise AssertionError(f"the test needs k >= 6 (n_expert_used = {k})")
    # (a) three tokens, two misses each
    il = LAYERS[1]
    seq = lib.sctw_begin(wh)
    miss = [[0, 1], [2, 3], [4, 5]]
    ids, ws, slots, hs = make_request(box, 3, miss, rng, n_expert)
    y = play_token(lib, wh, box, seq, {il: (3, ids, ws, slots, hs)})[il]
    mi = np.stack([missed_of(box, ids, ws, slots, t)[0] for t in range(3)])
    mw = np.stack([missed_of(box, ids, ws, slots, t)[1] for t in range(3)])
    ref = ref_multi(lib, model, il, mi, mw, hs, n)
    check(same_bits(y, ref), "NTOK=3: the 3 rows are bit-identical to sct_experts_multi")
    # (b) token 1 has no miss: its row is not written
    il = LAYERS[2]
    seq = lib.sctw_begin(wh)
    miss = [[0, 1], [], [2, 3]]
    ids, ws, slots, hs = make_request(box, 3, miss, rng, n_expert)
    y = play_token(lib, wh, box, seq, {il: (3, ids, ws, slots, hs)})[il]
    mi2 = np.stack([missed_of(box, ids, ws, slots, 0)[0], missed_of(box, ids, ws, slots, 2)[0]])
    mw2 = np.stack([missed_of(box, ids, ws, slots, 0)[1], missed_of(box, ids, ws, slots, 2)[1]])
    ref2 = ref_multi(lib, model, il, mi2, mw2, hs[[0, 2]], n)
    check(same_bits(y[0], ref2[0]) and same_bits(y[2], ref2[1]),
          "NTOK=3 with token 1 missless: rows 0 and 2 bit-identical to sct_experts_multi over those tokens")
    check(bool(np.all(y[1] == SENT)), "NTOK=3: the row of the token without a miss is not written")


def test_delay(lib, model, rng, n_expert):
    os.environ["STRATA_TEST_WORKER_DELAY"] = str(DELAY_MS)
    try:
        arr = (ctypes.c_int32 * len(LAYERS))(*LAYERS)
        wh = lib.sctw_create(model, arr, len(LAYERS), THREADS)
    finally:
        del os.environ["STRATA_TEST_WORKER_DELAY"]
    if not wh:
        raise AssertionError(f"sctw_create (delay): {last_error(lib)}")
    box = Box(lib, wh)
    n, k = box.n, box.k
    hit_done = False
    for _ in range(60):
        seq = lib.sctw_begin(wh)
        hits = [il for il in LAYERS if (seq * 31 + il) % 17 == 0]
        if not hits:   # the GPU passes this token: every layer is skipped
            for il in LAYERS:
                raise_flag(box, il, seq, skip=True)
            continue
        for il in LAYERS:
            b = box.base(il)
            miss = [sorted(rng.choice(k, 2, replace=False).tolist())]
            ids, ws, slots, hs = make_request(box, 1, miss, rng, n_expert)
            write_request(box, il, 1, ids, ws, slots, hs)
            t0 = time.perf_counter()
            raise_flag(box, il, seq)
            wait_layer(lib, wh, il, f"delay seq {seq}")
            dt = time.perf_counter() - t0
            mids, mws = missed_of(box, ids, ws, slots, 0)
            ref = ref_one(lib, model, il, mids, mws, hs[0], n)
            check(same_bits(box.F[b + box.y:b + box.y + n].copy(), ref),
                  f"delay: seq {seq} layer {il} bit-identical")
            if il in hits:
                check(dt >= DELAY_MS / 1000.0 * 0.9, f"delay: seq {seq} layer {il} answered {dt * 1000:.0f} ms late (>= {DELAY_MS} ms)")
                hit_done = True
        break
    check(hit_done, "delay: a (seq, layer) pair was selected by the switch and measured")
    call_with_timeout(lib.sctw_free, (wh,), "sctw_free (delay worker)")


def main():
    global np, SENT
    if not sc_env.get("SC_MODEL_14B"):
        print(sc_env.skip_line("SC_MODEL_14B"))
        return 0
    try:
        import numpy as numpy_mod
    except Exception as e:
        return sc_env.skip(f"cannot import numpy: {e}")
    np = numpy_mod
    SENT = np.float32(7.25)
    if not os.path.exists(DLL):
        raise SystemExit(f"missing {DLL}: run build_cpu.bat first")
    lib = load_lib()
    model = lib.sct_open(MODEL.encode())
    if not model:
        raise SystemExit(f"sct_open: {last_error(lib)}")
    info = (ctypes.c_int64 * 6)()
    lib.sct_info(model, info)
    n_layer, n_embd, n_expert, k = int(info[0]), int(info[1]), int(info[2]), int(info[3])
    print(f"model: n_layer={n_layer} n_embd={n_embd} n_expert={n_expert} n_expert_used={k}")
    check(max(LAYERS) < n_layer and n_embd % 4 == 0, "test layers exist and n_embd is a multiple of 4")
    rng = np.random.default_rng(20261120)

    arr = (ctypes.c_int32 * len(LAYERS))(*LAYERS)
    wh = lib.sctw_create(model, arr, len(LAYERS), THREADS)
    if not wh:
        raise SystemExit(f"sctw_create: {last_error(lib)}")
    box = Box(lib, wh)

    print("1. layout")
    test_layout(box)
    print("2. single-token requests, SKIP, routing")
    served = test_single_tokens(lib, model, wh, box, rng, n_expert)
    check(served == 9, f"9 served layers over 3 tokens, one skipped per token (got {served})")
    busy0 = lib.sctw_busy_ns(wh)
    check(busy0 > 0, f"busy() > 0 after the requests ({busy0 / 1e6:.2f} ms)")
    print("3. stale seq passed over; wait timeout")
    test_stale(lib, model, wh, box, rng, n_expert)
    print("4. stalled_layer and the verdict reset")
    test_stalled(lib, box, wh)
    print("5. NTOK = 3 (experts_multi)")
    test_multi(lib, model, wh, box, rng, n_expert)
    busy1 = lib.sctw_busy_ns(wh)
    check(busy1 > busy0, f"busy() grows with the work ({busy0 / 1e6:.2f} ms -> {busy1 / 1e6:.2f} ms)")
    print("6. STRATA_TEST_WORKER_DELAY")
    test_delay(lib, model, rng, n_expert)
    print("7. stop joins; free joins")
    call_with_timeout(lib.sctw_stop, (wh,), "sctw_stop")
    call_with_timeout(lib.sctw_free, (wh,), "sctw_free")
    for rep in range(3):   # create and drop repeatedly: no leaked or stuck threads
        wh = lib.sctw_create(model, arr, len(LAYERS), THREADS)
        if not wh:
            raise AssertionError(f"sctw_create (repeat {rep}): {last_error(lib)}")
        call_with_timeout(lib.sctw_free, (wh,), f"sctw_free (repeat {rep})")
    lib.sct_close(model)
    print("ALL PASSED: test_cpu_worker")


if __name__ == "__main__":
    main()
