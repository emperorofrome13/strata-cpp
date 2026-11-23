# HANDOFF — strata.cpp 1.00

Public-ready standalone repository of **strata.cpp**, the C++/CUDA port of RustingStrata 1.2.5's
decode and prefill path. Built from the read-only internal StrataFreet checkout (the `native/strata_cpp`
directory of that project); nothing in that project was modified.

## What is in the repository

| Path | Contents |
|---|---|
| `src/` | 36 files — GGUF, quant (+ `tables_host.h`), config, weights, memory, NVRTC, backend, CUDA backend (3 parts), model, expert cache, GPU model, prefill, sampling, CPU MoE, CPU worker, CPU test API, product C ABI, dev/test ABI |
| `kernels/` | `kernels.cu`, `tables.cuh` (verbatim from 1.2.5), `DIFF-vs-stratafreet.md` |
| `tools/` | `embed_kernels.py`, `gen_tables.py`, `gen_inventory.py`, `sc_score.cpp`, `sc_bench.cpp` |
| `tests/` | 16 tests (`test_*.py`, `verify_ram_lock.py`) plus the shared `sc_env.py` |
| `docs/` | `DEVLOG.md` (the old README status table, verbatim), `INKLING2_NOTES.md` |
| root | `README.md`, `VERSION` (`1.00`), `LICENSE`, `LICENSE-RustingStrata.md`, `MODIFICATIONS.md`, `PORT_MAP.md`, `.gitignore`, `build.bat`, `build.ninja`, `build_cpu.bat`, `build_cpu.ninja`, `build_mx.bat`, `build_mx.ninja`, `build_cpu_mx.ninja` |

Not copied (as required): `build/`, `build_mx/`, `backup/`, `.cache/`, `.ec14/`, `.pytest_cache/`,
`.ninja_log`, `ink_dbg*.bin`, all logs/binaries, the vendored llama.cpp sources in `reference/`, and the
upstream Rust source tree `strata_src_v1.2.5/`. llama.cpp is referenced in the README and credits as the
Inkling-2 implementation reference; RustingStrata is named as RustingStrata 1.2.5 by Vasyl Trefilov.

## What changed relative to the StrataFreet copy

- **Build scripts** (`build.bat`, `build_cpu.bat`, `build_mx.bat`): no hardcoded Visual Studio or CUDA
  paths. `VCVARS` is used when set; otherwise `vswhere.exe` (tried under `%ProgramFiles(x86)%` and
  `%ProgramFiles%`) finds the newest install with the C++ tools. `CUDA_PATH` must be set (the CUDA
  installer sets it); a clear error names the missing variable. `SC_NO_PAUSE` is unchanged.
  `build_mx.ninja` and `build_cpu_mx.ninja` now link `advapi32.lib` like their `build/` counterparts:
  the copies were missing it, so `build_mx.bat` failed with three unresolved token-privilege symbols
  before this fix.
- **Tests**: every model, oracle and external executable comes from an environment variable. A missing
  variable prints `SKIP: set NAME=<hint>` and exits 0 before any DLL load or CUDA context. Variables:
  `SC_MODEL_14B`, `SC_MODEL_FLASHNEXT`, `SC_MODEL_INKLING`, `SC_GPT_OSS_GGUF` (existing name kept),
  `STRATAFREET_ROOT`, `SC_RUSTINGSTRATA`, `SC_LLAMA_SERVER`, `SC_EVICT_GGUFS`, `SC_BUILD_DIR`.
  `tests/sc_env.py` holds the shared names and messages.
- **Tools**: `gen_tables.py` and `gen_inventory.py` read `STRATA_SRC_V125` (and optionally
  `STRATA_SRC_ZIP`) instead of assuming a sibling Rust source tree.
- **Docs**: `README.md` rewritten for the public repository; `docs/DEVLOG.md` holds the old status table
  verbatim; `docs/INKLING2_NOTES.md` moved from `reference/`; `VERSION` added; `LICENSE` is a byte-for-byte
  copy of `LICENSE-RustingStrata.md` (both kept).
- **Engine code**: unchanged in behaviour. Only comments changed: the generated-header line in
  `src/tables_host.h`, and ninja comments that referred to the internal plan document or "this machine".
  No `src/` or `kernels/` logic, constants or signatures were modified.
- `build/kernel_text.cpp` is not committed; `build.ninja` regenerates it with `tools/embed_kernels.py`
  (verified on a clean tree: `[1/5] embed kernel source build/kernel_text.cpp`).

## Build (verified from a clean tree)

Requirements: VS 2022 Build Tools (C++ workload), CUDA 12.x (`CUDA_PATH`), `ninja` and Python 3 on PATH.

```bat
set SC_NO_PAUSE=1
build.bat
build_cpu.bat
```

Windows 10, PowerShell:

```powershell
$env:SC_NO_PAUSE = "1"
.\build.bat
.\build_cpu.bat
```

Output tail (clean tree, `build.bat`):

```text
[1/5] embed kernel source build/kernel_text.cpp
[2/5] DLL build/strata_cpp_dev.dll
...
[3/5] DLL build/strata_cpp.dll
...
[4/5] EXE build/sc_score.exe
sc_score.cpp
[5/5] EXE build/sc_bench.exe
sc_bench.cpp
[build] ok: build\strata_cpp.dll, build\strata_cpp_dev.dll, build\sc_score.exe, build\sc_bench.exe
```

Output tail (`build_cpu.bat`):

```text
[9/9] DLL build/strata_cpu_test.dll
   Creating library build/strata_cpu_test.lib and object build/strata_cpu_test.exp
[build_cpu] ok: build\strata_cpu_test.dll
```

Output tail (`build_mx.bat`, after the two link-line fixes):

```text
[9/9] DLL build_mx/strata_cpu_test.dll
   Creating library build_mx/strata_cpu_test.lib and object build_mx/strata_cpu_test.exp
[build_mx] ok: build_mx\strata_cpp.dll, build_mx\strata_cpp_dev.dll, build_mx\sc_score.exe, build_mx\sc_bench.exe, build_mx\strata_cpu_test.dll
```

All three builds exit 0. `build.bat` found MSVC through `vswhere`
(`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`) and
CUDA through the installed `CUDA_PATH` (v12.9). The build emits pre-existing `C4189` warnings for unused
debug-trace locals in `src/gpu_model.cpp`; there are no errors.

## Verification (CPU only)

`python -m py_compile tests/*.py tools/*.py` — **OK** (every file compiles).

No environment variables set — every CPU test prints SKIP and exits 0:

```text
test_quant           SKIP: set STRATAFREET_ROOT=<StrataFreet checkout> (reference oracle)   -> exit=0
test_gguf            SKIP: set STRATAFREET_ROOT=<StrataFreet checkout> (reference oracle)   -> exit=0
test_weights         SKIP: set STRATAFREET_ROOT=<StrataFreet checkout> (reference oracle)   -> exit=0
test_cpu_moe         SKIP: set STRATAFREET_ROOT=<StrataFreet checkout> (reference oracle)   -> exit=0
test_cpu_worker      SKIP: set SC_MODEL_14B=<path to Qwen3.6-14B-A3B Q6_K gguf>             -> exit=0
verify_ram_lock      SKIP: set STRATAFREET_ROOT=<StrataFreet checkout> (reference oracle)   -> exit=0
```

Tests that need a model/oracle, run CPU-only with real files:

- `STRATAFREET_ROOT=<StrataFreet checkout> python tests\test_quant.py`
  → `PASS: 26 types bit-exact` (F32…MXFP4; the optional real MXFP4 rows skipped because
  `SC_GPT_OSS_GGUF` was unset). No CUDA context: the test only calls `sci_row_bytes`/`sci_dequant`.
- `SC_MODEL_14B=... python tests\test_gguf.py`
  → `47 metadata keys, 733 tensors, 1 shard(s), 3 array(s) not held by gguf_io: ok` / `PASS: zero mismatches`.
- `SC_MODEL_14B=... python tests\test_weights.py`
  → `733 tensors, 2 quant types, config 41 fields` / `ok`; `memory total 111.9 GiB, ram_cap ok,
  prefetch 64 batches ok`; `PASS: config, weights and memory agree for 1 model(s)`.
- `python tests\test_sampler.py` → `PASS: 46 sampler checks exact`.
- `SC_MODEL_14B=... STRATAFREET_ROOT=... python tests\test_cpu_moe.py`
  → `[1] int_dots_selftest: ok`; 20/20 random cases against ggml, `max rel_l2 = 6.070e-07`
  (tolerance 2e-2); `experts_multi vs experts: max abs diff = 0.000e+00`; timing 75.2 / 62.6 / 67.3 / 65.4
  µs per expert at 4 / 6 / 7 / 8 threads; `RESULT: PASS`.
- `SC_MODEL_14B=... python tests\test_cpu_worker.py` → `ALL PASSED: test_cpu_worker`
  (layout, bit-identical served layers, SKIP, routing, stale seq, NTOK=3, busy growth, delay switch,
  stop/free joins).

Path scan (PowerShell, `rg` from the Codex install), with the brief's five terms and the repository
root as the search target, excluding `.git/` and `.ec14/`:

```text
(no output)
```

One note for the literal form of the pattern: the brief's user-name term is a bare substring, so it also
matches the middle of an ordinary English word used as a llama.cpp `/completion` JSON field name in
`tests/test_score_inkling.py` (the sampler's heat setting, which the API requires verbatim). No path,
user-profile prefix, user name or model-library path matches. With whole-word matching — the intended
meaning of that term — the scan is empty, as pasted above.

## Not verified — GPU tests deferred

The GPU was in use, so **no CUDA context was created and no model was loaded onto the GPU**. These were
not run: `test_smoke.py`, `test_backend_gemv.py`, `test_backend_ops.py`, `test_backend_seq.py`,
`test_score_14b.py`, `test_partial_14b.py`, `test_prefill_14b.py`, `test_flashnext.py`,
`test_score_inkling.py`, `verify_ram_lock.py`, `sc_bench.exe`, `sc_score.exe`.

Their SKIP gates were verified by reading the code: in every one, `sc_env.add_stratafreet()` and the
`SC_MODEL_*` / oracle checks run first, and only afterwards is the dev DLL loaded (`load_dll()`), the
tokenizer constructed or a model process started. Executing them with the variables unset would therefore
exit 0 before any CUDA call; they were left unexecuted because HARD RULE 2 forbids running them at all.

Consequently the speed, parity and residency numbers in the README are the ones measured earlier during
the port (recorded in `docs/DEVLOG.md`), not re-measured tonight.

## Known items

- `sc_version()` still returns `"strata-cpp 0.0.1 (port of RustingStrata 1.2.5 by Vasyl Trefilov)"`.
  The package version is 1.00 (`VERSION`, README); the brief forbids engine-code changes beyond
  comments/paths, so the string was left alone. Bump it in `src/capi.cpp` when that restriction is lifted.
- Non-ASCII model paths are unsupported (documented in `MODIFICATIONS.md`).
- GGML types 250 (FP8) and 251 (NVFP4) are rejected on purpose; MXFP4 (39) is the one type added beyond
  RustingStrata 1.2.5's 25.
- `PORT_MAP.md` and `docs/DEVLOG.md` are historical engineering documents. They still mention the
  relative path `native/strata_src_v1.2.5/` (the internal checkout used during the port); that directory
  is not part of this repository and is not needed to build.

## Next steps

1. When the GPU is free, run the GPU suite one test at a time (each holds `.gpu.lock`):

   ```powershell
   $env:SC_MODEL_14B = "<Qwen3.6-14B-A3B Q6_K gguf>"
   $env:SC_MODEL_FLASHNEXT = "<Qwen3.8-Flash-Next UD-Q3_K_XL first shard>"
   $env:SC_MODEL_INKLING = "<Inkling-2-Mini IQ3_XXS gguf>"
   $env:SC_GPT_OSS_GGUF = "<gpt-oss-20b MXFP4 gguf>"
   $env:STRATAFREET_ROOT = "<StrataFreet checkout>"
   $env:SC_RUSTINGSTRATA = "<rustingstrata.exe v1.2.5>"
   $env:SC_LLAMA_SERVER = "<llama-server.exe>"
   $env:SC_EVICT_GGUFS = "<gguf>;<gguf>"
   $env:CUDA_PATH = "<CUDA 12.x toolkit root>"
   python tests\test_smoke.py
   python tests\test_backend_gemv.py
   python tests\test_backend_ops.py
   python tests\test_backend_seq.py
   python tests\test_score_14b.py
   python tests\test_prefill_14b.py
   python tests\test_partial_14b.py
   python tests\test_flashnext.py
   python tests\test_score_inkling.py
   python tests\verify_ram_lock.py
   ```

2. Create the GitHub repository and push. This repository was initialised and committed locally only:
   no remote was added and nothing was pushed.
3. If the GPU numbers differ from `docs/DEVLOG.md`, update the README status table.
4. Optional cleanup: bump `sc_version()` to 1.00 and remove the pre-existing `C4189` warnings.
