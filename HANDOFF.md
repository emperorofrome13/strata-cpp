# HANDOFF — strata.cpp 1.00

Public-ready standalone repository of **strata.cpp**, the C++/CUDA port of RustingStrata 1.2.5's
decode and prefill path. Built from the read-only internal StrataFreet checkout (the `native/strata_cpp`
directory of that project); nothing in that project was modified.

## GLM-5.3-Flash port (branch `glm5next`, 2026-11-23) — ported, not yet verified on GPU

Port of RustingStrata 1.2.5's `src/glm/` (+ the `deepseek/mat.rs` and `deepseek/ops.rs` parts it runs on) into
`src/glm.h`, `src/glm.cpp`, `src/glm_forward.cpp`, `src/ds_mat.{h,cpp}`, `src/ds_ops.{h,cpp}`, wired into the product
ABI (`sc_load` dispatches on `general.architecture` `glm5-next` / `glm5next`). Deviations: `MODIFICATIONS.md`, last
section. Model facts, the formula cross-check against llama.cpp `glm5-next.cpp`, what is and is not ported:
`docs/GLM5_NOTES.md`. Committed on branch `glm5next` only; **not pushed**.

The machine's GPU and most of its RAM were in use by another job, so **no CUDA context was created, no GPU test, no
`sc_score`/`sc_bench`/`rustingstrata` run, and no GGUF over 1 GB was opened**.

Done and verified (CPU only):

- `build.bat` and `build_cpu.bat` build; the new files add no warnings (the existing `C4189`/`C4458` warnings in
  `gpu_model.cpp`/`prefill.cpp` are unchanged).
- `tests/test_glm_cpu.py` (new): **PASS**. Synthetic glm5-next GGUF (n_embd 64, KDA+dense, DSA+MoE, KDA+MoE, DSA+MoE
  and a NextN block, F32), the C++ CPU forward against an independent float64 numpy forward written from llama.cpp's
  formulas: 20 positions, max |logit diff| / max |logit| = 6.1e-6, argmax 20/20, k-pool selection active from
  position 8 (an all-rows reference differs by 4.9e-1); chunked prefill (512 and 3), two consecutive prefills, both
  arch spellings and 2-4 threads bit-identical to token-by-token; `validate` passes and refuses an extra tensor; the
  context limit refuses before any state change; unit checks of hc_coeffs, hc_post, route_sigmoid, l2_norm,
  swiglu_clamp, the f16 rounding (140,009 values), q8_roundtrip and Mat::matvec on Q8_0 weights.
- Existing CPU tests on small inputs, GPU hidden (`CUDA_VISIBLE_DEVICES=`): `test_quant` PASS (26 types),
  `test_sampler` PASS (46 checks), `test_gguf` PASS on the two synthetic GLM files. **Not run** (they need the
  10.5 GB Qwen3.6-14B model, which was not allowed to be opened): `test_weights`, `test_cpu_moe`, `test_cpu_worker`.
  Without variables all three print SKIP and exit 0. `cpu_moe.cpp` only gained the public `int_dot_kernel` wrapper.
- The real UD-IQ1_S tensor table (read from the shard headers only) equals `GlmConfig::expected_tensors()`: 1412
  tensors, no problems; all its quant types are supported.
- The oracle alias writer of `tests/test_score_glm5.py` was run on the real shard 1 (9.4 MB, metadata only, output in
  a scratch folder): all 330 metadata entries identical apart from the 40 renamed keys and the arch value.

**Finding:** the unsloth GGUF says `general.architecture = glm5next` (llama.cpp's spelling); RustingStrata 1.2.5 (source
and the official exe) only takes `glm5-next`. strata-cpp accepts both; `tests/test_score_glm5.py` gives the official
binary an alias set (shard 1 rewritten, shards 2-3 hard-linked into `build\glm5_alias\`, or `SC_GLM5_ALIAS_DIR` on
the model's volume).

### To run when the GPU is free (in this order)

```powershell
Set-Location E:\aiprojects\strata-cpp
git switch glm5next
$env:SC_NO_PAUSE = "1"
& .\build.bat
& .\build_cpu.bat
python tests\test_glm_cpu.py
$env:SC_MODEL_GLM5 = "E:\aiprojects\LM Studio Models\hub\models--unsloth--GLM-5.3-Flash-GGUF\snapshots\2975ab414d30340466d8c51533c6e91f0cca64c1\UD-IQ1_S\GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf"
$env:SC_RUSTINGSTRATA = "E:\aiprojects\stratafreet\tools\strata125\rustingstrata-windows-x86_64.exe"
python -c "import sys; sys.path.insert(0, 'tests'); import test_score_glm5 as t; print(t.oracle_model(sys.argv[1]))" $env:SC_MODEL_GLM5
& $env:SC_RUSTINGSTRATA check --model build\glm5_alias\GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf
python tests\test_score_glm5.py
python tests\test_score_glm5.py --n 8 --long --speed
```

1. The `python -c` line builds the alias for the official binary and prints its path; `check` should print
   `check: glm5-next ok: 45 layers (34 KDA, 11 DSA) + 1 NextN, 288 experts (8 used), 1412 tensors, tensor table valid`.
2. `test_score_glm5.py`: 3 prompts x 64 greedy positions; the official `generate` (with `STRATA_GLM_TOP5` /
   `STRATA_GLM_TRACE`) against `build\sc_score.exe` teacher-forced on its tokens. Gate: argmax >= 97%. Each process
   needs >= 24 GB of available RAM and holds `.gpu.lock`.
3. `--long` adds a ~2300-token prompt (past the 2048-row indexer threshold, tens of minutes per side); `--speed`
   prints the DLL's greedy decode tok/s next to `rustingstrata bench --pp 16 --tg 32` (reported, not gated).
4. Then run the whole existing suite (section "Next steps" below), which the shared files (`capi.cpp`, `cpu_moe.cpp`,
   the ninja files) touch: `test_smoke`, `test_backend_*`, `test_score_14b`, `test_cpu_moe`, `test_cpu_worker`,
   `test_weights` at least.

If parity fails: `STRATA_GLM_DUMP=<dir>` makes both engines append the same per-tensor dumps (`<name>-<layer>.f32`:
`hc_attn_pre`, `attn_norm`, `kda_q_conv`, `kda_scan_out`, `kv_cmpr`, `attn_out`, `ffn_moe_logits`, `ffn_moe_topk`,
`ffn_moe_out`, `ffn_shexp`, `l_out`, `result_output`, ...) for a side-by-side diff (use `-n 1` / a one-token
continuation); `STRATA_GLM_TRACE=1` prints the same values' heads and sums; `STRATA_GLM_BATCH=0`, `STRATA_DS_BATCH=0`
and `STRATA_GLM_CHUNK=1` switch off the batched paths on both sides; `STRATA_GLM_CPU=1` keeps both on the CPU.

Open items: the GPU path (`ds_mat.cpp` `Dev`: upload, expert cache, staging, batched gemv/gemm) has never run; speed is
unmeasured; the NextN/MTP block is not used (as in Rust); README status row says "ported, not yet verified on GPU"
until `test_score_glm5.py` passes.

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

## Review pass (2026-11-22, after the EC14 build)

- Every engine file in `src/`, `kernels/` and `tools/*.cpp` was compared byte for byte with
  `stratafreet/native/strata_cpp`: identical, except one comment line in `src/tables_host.h`. The StrataFreet
  folder was not modified.
- A fresh `git clone` of the first commit built with `build.bat` and `build_cpu.bat` (`strata_cpp.dll`,
  `strata_cpp_dev.dll`, `sc_score.exe`, `sc_bench.exe`, `strata_cpu_test.dll`).
- CPU-only tests on that clone, GPU hidden with `CUDA_VISIBLE_DEVICES=`, `SC_MODEL_14B` = Qwen3.6-14B-A3B Q6_K:
  `test_quant` PASS (26 types bit-exact), `test_gguf` PASS (zero mismatches), `test_weights` PASS,
  `test_cpu_moe` PASS (58.7 us/expert median at 8 threads), `test_cpu_worker` ALL PASSED. With no variables
  set, each of the five prints a SKIP line and exits 0.
- Added `.gitattributes` (`*.bat text eol=crlf`). The batch files were committed with LF endings, which cmd
  can mis-parse (labels, `goto`) on a checkout without CRLF conversion.
- README: removed the unverified claims that the DLL runs without AVX2 and on Volta GPUs.
- Not run: `test_sampler.py` and every GPU test (GPU busy).

## GPU test run (2026-11-23)

GPU free (1.2 GB in use, no other CUDA job), 73-75 GB RAM free. Clean `build.bat` + `build_cpu.bat`, then one test
at a time. Results are in the README "Latest test run" table. Summary: every kernel, quant, sampler and CPU test
passes; `test_score_14b` and `test_score_inkling` pass; `test_prefill_14b` and `test_partial_14b` fail on the
same partial-residency gates already open in `docs/DEVLOG.md` (pp512 and decode speed at an 8500 MB margin, and
1725 vs 1715 resident experts); the engine code is byte-identical to StrataFreet's, so these are not caused by
the repository copy. `test_flashnext` skipped itself (needs 85 GB free RAM). `verify_ram_lock` not run.

Next: run `test_flashnext.py` with 90 GB of RAM free; then create the GitHub repository and push.
