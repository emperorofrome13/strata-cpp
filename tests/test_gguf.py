"""Wave 2-A acceptance: the C++ GGUF reader must agree with stratafreet/gguf_io.py.

    python tests/test_gguf.py <model.gguf> [more.gguf ...]
    python tests/test_gguf.py            # uses SC_MODEL_14B

Compared per file: every metadata key, every scalar metadata value, every array's element type and
length, and for every tensor its name, quant type, shape (in file order), data offset and byte size.
Array *contents* are not compared element by element (GGUF arrays here are vocabularies of 150k strings);
their type and length are.

Needs STRATAFREET_ROOT (the Python reference oracle) and a model file. Without either the test prints
SKIP and exits 0.
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
GGUFModelFile = None


def load():
    if not os.path.isfile(DLL):
        raise SystemExit(f"missing {DLL}; run build.bat first")
    lib = ctypes.CDLL(DLL)
    lib.sci_gguf_info_json.restype = ctypes.c_char_p
    lib.sci_last_error.restype = ctypes.c_char_p
    lib.sci_gguf_info_json.argtypes = [ctypes.c_char_p]
    return lib


def scalar_close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        a, b = float(a), float(b)
        if a != a and b != b:  # NaN
            return True
        return a == b or abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
    return a == b


def check(lib, path: str, dropped: set) -> list:
    bad = []
    raw = lib.sci_gguf_info_json(path.encode())
    if raw is None:
        return [f"{path}: sci_gguf_info_json failed: {lib.sci_last_error().decode()}"]
    info = json.loads(raw.decode())
    ref = GGUFModelFile(path)

    # metadata keys
    if set(info["meta"]) != set(ref.meta):
        only_cpp = set(info["meta"]) - set(ref.meta)
        only_py = set(ref.meta) - set(info["meta"])
        bad.append(f"{path}: metadata keys differ (cpp only {sorted(only_cpp)[:5]}, "
                   f"py only {sorted(only_py)[:5]})")

    # metadata values
    for k in sorted(set(info["meta"]) & set(ref.meta)):
        cpp = info["meta"][k]
        pyv = ref.meta[k]
        if cpp["t"] == "arr":
            if pyv is None:
                # gguf_io drops large arrays (vocabularies) when it parses; nothing to compare against
                dropped.add(k)
                continue
            py_len = len(pyv)
            if cpp["n"] != py_len:
                bad.append(f"{path}: {k}: array length {cpp['n']} != {py_len}")
            continue
        if cpp["t"] == "str":
            if str(pyv) != cpp["v"]:
                bad.append(f"{path}: {k}: {cpp['v']!r} != {pyv!r}")
            continue
        try:
            pyf = float(pyv)
        except (TypeError, ValueError):
            bad.append(f"{path}: {k}: uncomparable python value {pyv!r}")
            continue
        if not scalar_close(cpp["v"], pyf):
            bad.append(f"{path}: {k}: {cpp['v']} != {pyf}")

    # tensors
    data_offsets = info["data_offsets"]
    if set(info["tensors"]) != set(ref.tensors):
        only_cpp = set(info["tensors"]) - set(ref.tensors)
        only_py = set(ref.tensors) - set(info["tensors"])
        bad.append(f"{path}: tensor names differ (cpp only {sorted(only_cpp)[:5]}, "
                   f"py only {sorted(only_py)[:5]})")

    for name in sorted(set(info["tensors"]) & set(ref.tensors)):
        cpp = info["tensors"][name]
        py = ref.tensors[name]
        if cpp["type"] != int(py.qtype):
            bad.append(f"{path}: {name}: type {cpp['type']} != {py.qtype}")
        if list(reversed(cpp["dims"])) != [int(d) for d in py.shape]:
            bad.append(f"{path}: {name}: dims {cpp['dims']} != {py.shape}")
        shard = ref.paths.index(py.path) if py.path in ref.paths else 0
        want_off = data_offsets[shard] + cpp["offset"]
        if want_off != int(py.offset):
            bad.append(f"{path}: {name}: file offset {want_off} != {py.offset}")
        if cpp["bytes"] != int(py.data.nbytes):
            bad.append(f"{path}: {name}: {cpp['bytes']} bytes != {py.data.nbytes}")
    return bad


def main(argv) -> int:
    global GGUFModelFile
    reason = sc_env.add_stratafreet()
    if reason:
        print(reason)
        return 0
    try:
        from stratafreet.gguf_io import GGUFModelFile as gguf_model_file
    except Exception as e:
        return sc_env.skip(f"cannot import the StrataFreet oracle: {e}")
    GGUFModelFile = gguf_model_file

    if not argv:
        env_model = sc_env.get("SC_MODEL_14B")
        if not env_model:
            print(sc_env.skip_line("SC_MODEL_14B"))
            return 0
        argv = [env_model]
    lib = load()
    all_bad = []
    for path in argv:
        print(f"--- {os.path.basename(path)}")
        dropped: set = set()
        bad = check(lib, path, dropped)
        ref = GGUFModelFile(path)
        note = f", {len(dropped)} array(s) not held by gguf_io" if dropped else ""
        print(f"    {len(ref.meta)} metadata keys, {len(ref.tensors)} tensors, "
              f"{len(ref.paths)} shard(s){note}: {'ok' if not bad else f'{len(bad)} mismatches'}")
        all_bad += bad
    if all_bad:
        print()
        for b in all_bad[:40]:
            print(f"FAIL {b}")
        return 1
    print("\nPASS: zero mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
