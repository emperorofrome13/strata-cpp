# strata.cpp

**Version 1.00**

strata.cpp is a C++/CUDA port of the decode and prefill path of **RustingStrata 1.2.5**, built as one
DLL with a C ABI. It keeps MoE experts in page-locked host RAM behind a VRAM expert cache, and runs the
experts that do not fit on the GPU on a CPU expert worker. The host application passes token ids in and
gets logits or a sampled token out; tokenization and chat templates stay in the host.

**Uses RustingStrata by Vasyl Trefilov.** This is a modified version, distributed under the
RustingStrata License 1.0 — see [License](#license) and `MODIFICATIONS.md`.

## Supported architectures

| Arch key | Models | Status on this port (measured, not projected) |
|---|---|---|
| `qwen35moe` | Qwen3.5 / Qwen3.6 MoE | **Main target.** Fully resident decode 199.5 vs 207.7 tok/s for the official RustingStrata (96%, gate met); parity 192/192 argmax positions and max top-1 logprob difference 0.000000. Batched prefill pp512 102.9% and tg128 97.0% of official, resident expert count matched (3600/3600). With a forced 8500 MB VRAM margin (partial residency + CPU worker): parity 97.4%, decode 98.4% of official, pp512 92.7% (**gate 95%, not met**), and the profile round-trip places 1726 experts vs the official 1715 (**not matched**). |
| `qwen4exp` | Qwen3.8-Flash-Next | Ported (hyper-connection streams, per-layer embeddings, QSA sparse attention, DeltaNet and full attention layers, sigmoid-gated norms). **Gates not met:** parity 61/64 = 95.3% (gate 97%), max top-1 logprob difference 0.317; decode 34.4 vs 31.0 tok/s (111%); pp512 98.9 vs 124.8 tok/s (**79%, gate 95%, not met**); CPU worker 25.8 / 19.0 ms per token vs official 31.6 / 22.0. |
| `glm5-next` / `glm5next` | GLM-5.3-Flash | **Ported, not yet verified on GPU.** Port of RustingStrata 1.2.5's `glm/` (2026-11-23): Kimi delta attention, NoPE MLA with the k-pool sparse indexer, four-stream hyper-connections, sigmoid-routed MoE with a shared expert; the CPU reference forward with the dense matvecs and a VRAM expert cache on the GPU, as RustingStrata runs it. Accepts both arch spellings (RustingStrata's `glm5-next`, llama.cpp's and unsloth's `glm5next`). CPU path checked against an independent numpy forward on a synthetic model (max relative logit difference 6.1e-6, prefill bit-identical to token-by-token); the real UD-IQ1_S tensor table matches (1412/1412). GPU parity against the official binary (`tests/test_score_glm5.py`, gate argmax >= 97%) and speed are **not measured yet**. See `docs/GLM5_NOTES.md`. |
| `inkling2` | Inkling-2-Mini (a StrataFreet addition, not in RustingStrata 1.2.5) | Implemented: 48 logical slots over 24 physical blocks, 3 loop passes, relative-bias sliding-window and full attention, shortconv, fused gate/up experts, Situ, router and down biases. Parity vs llama.cpp CPU: 255/256 argmax (99.6%), max logprob difference 0.0612; vs llama.cpp CUDA: 252/256 (98.4%), 0.152. All-resident IQ3_XXS at ctx 8192: pp512 145.3 tok/s, tg128 98.8 tok/s. The 21.3 GB Q8 file runs through the expert cache: pp512 14.4, tg128 15.8 tok/s. |

### Latest test run (2026-11-23, this repository's build)

| Test | Result |
|---|---|
| `test_smoke`, `test_sampler` (46 checks), `test_quant` (26 types, real MXFP4 rows) | PASS |
| `test_backend_gemv` (39 comparisons), `test_backend_ops` (111 checks), `test_backend_seq` (55 checks) | PASS |
| `test_gguf`, `test_weights`, `test_cpu_moe`, `test_cpu_worker` (14B model) | PASS |
| `test_score_14b` | PASS: argmax 192/192, max top-1 logprob diff 0.000000; decode 191.1 vs 196.0 tok/s official |
| `test_prefill_14b` | FAIL (open gates): margin 8500 MB pp512 569.0 vs 661.3 tok/s official (86.0%, gate 95%); tg128 98.1% (ok); resident 1725 vs 1715 |
| `test_partial_14b` | FAIL (open gates): parity 188/192 = 97.9% (ok, gate 97%); decode 69.1 vs 83.8 tok/s (82.4%, gate 95%); placement 1725 vs 1715 |
| `test_score_inkling` | PASS: argmax 63/64 = 98.4% vs llama.cpp CPU; pp512 143.9, tg128 97.9 tok/s (all 3072 experts resident) |
| `test_flashnext` | not measured: needs 85 GB free RAM, 75 GB was free |
| `verify_ram_lock` | not run |

Partial-residency speed varies between runs on this machine (the official binary's own decode ranged 55.9 to
108.4 tok/s in the same run); the open gates above match the ones recorded in `docs/DEVLOG.md`.

Everything here is measured on the machine listed under [Measured hardware](#measured-hardware).
Numbers for other model files, margins and contexts will differ. The full wave-by-wave history, including
every gate that was missed, is in `docs/DEVLOG.md`.

## Weight types

26 GGML types are accepted: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
Q2_0, TQ1_0, TQ2_0, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, and MXFP4
(type 39, a StrataFreet addition). The 25 types inherited from RustingStrata 1.2.5 are bit-exact against
the Rust reference dequantizers; MXFP4 is bit-exact against StrataFreet's `quant.dequant`. GGML types 250
(FP8) and 251 (NVFP4) are **not** ported and are rejected at load.

## Requirements

- Windows x64 (the code uses the Win32 file mapping, `CreateFileW`, and `VirtualLock` paths).
- Visual Studio 2022 Build Tools with the **Desktop development with C++** workload. `build.bat` finds
  it with `vswhere.exe`, or uses the `VCVARS` environment variable when set.
- CUDA 12.x toolkit (`CUDA_PATH` must point at its root; the CUDA installer sets it). NVRTC is loaded at
  run time, not linked, so the toolkit is needed at run time as well as at build time. CUDA 13.x drops
  NVRTC support for Volta (sm_70); use 12.x if that is your GPU.
- `ninja` on `PATH`, and Python 3 on `PATH` for the kernel-embedding build step.
- An NVIDIA GPU. Kernels are compiled by NVRTC at run time for the compute capability of the device in
  use; the int8 tensor-core prefill path needs sm_80+. It has only been tested on an RTX 5070 Ti (sm_120)
  with CUDA 12.9. No Vulkan or CPU-only backend exists (GLM-5.3-Flash alone can keep its whole forward on the CPU
  with `STRATA_GLM_CPU`, as RustingStrata can).
- An x86-64 CPU with AVX2. The build uses `/arch:AVX2`, so the compiler may emit AVX2 instructions
  anywhere in the DLL; it is not expected to run on a CPU without AVX2.

## Build

```
build.bat
```

Outputs:

- `build\strata_cpp.dll` — the product ABI (`src\capi.h`)
- `build\strata_cpp_dev.dll` — the same GGUF/quant/config/weights code behind an inspect/dequant ABI for
  the tests, plus the `scd_*` kernel wrappers
- `build\sc_score.exe` — the C++ twin of `rustingstrata score`
- `build\sc_bench.exe` — the C++ twin of `rustingstrata bench`

Other build entry points:

- `build_cpu.bat` — only the CPU expert path and its test ABI, `build\strata_cpu_test.dll`.
- `build_mx.bat` — all of the above into `build_mx\`, so a DLL held open by a running benchmark in
  `build\` is not overwritten. The tests read the directory from `SC_BUILD_DIR` (default `build`).

`build\kernel_text.cpp` is generated during the build by `tools\embed_kernels.py`; it is not committed.

## Tests

All tests are Python 3 scripts under `tests\`. They load the DLLs with `ctypes`, and each one that needs a
model, an oracle or a specific executable reads it from an environment variable. When a variable is unset
the test prints a `SKIP:` line and exits 0; there are no built-in machine paths.

| Variable | Meaning |
|---|---|
| `SC_MODEL_14B` | Qwen3.6-14B-A3B Q6_K GGUF (the `qwen35moe` test model) |
| `SC_MODEL_FLASHNEXT` | Qwen3.8-Flash-Next UD-Q3_K_XL first shard (`qwen4exp`) |
| `SC_MODEL_INKLING` | Inkling-2-Mini IQ3_XXS GGUF (`inkling2`) |
| `SC_MODEL_GLM5` | GLM-5.3-Flash first shard, e.g. UD-IQ1_S `GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf` (`glm5next`) |
| `SC_GLM5_ALIAS_DIR` | Folder on the GLM model's volume for the official binary's arch alias (default `build\glm5_alias`) |
| `SC_GPT_OSS_GGUF` | gpt-oss-20b MXFP4 GGUF (real MXFP4 expert rows) |
| `STRATAFREET_ROOT` | StrataFreet checkout — the Python reference oracle (`quant`, `gguf_io`, `tokenizer`) |
| `SC_RUSTINGSTRATA` | `rustingstrata.exe` 1.2.5 — the official oracle |
| `SC_LLAMA_SERVER` | `llama-server.exe` from a llama.cpp build that reads the Inkling-2 looped layout |
| `SC_EVICT_GGUFS` | Semicolon-separated GGUF files used to evict the Windows file cache (`verify_ram_lock.py`) |
| `SC_BUILD_DIR` | Build directory to load the DLLs from (default `build`) |

| Test | Needs | Runs without a GPU? |
|---|---|---|
| `test_sampler.py` | built `build\strata_cpp_dev.dll` | **yes** — sampler arithmetic and RNG only |
| `test_quant.py` | `STRATAFREET_ROOT`; `SC_GPT_OSS_GGUF` for the real MXFP4 rows | **yes** — dequantizes host buffers, no CUDA context |
| `test_gguf.py` | `STRATAFREET_ROOT`, a GGUF (`SC_MODEL_14B` by default) | **yes** |
| `test_weights.py` | `STRATAFREET_ROOT`, `SC_MODEL_14B` and/or `SC_MODEL_FLASHNEXT` | **yes** |
| `test_cpu_moe.py` | `build_cpu.bat`, `STRATAFREET_ROOT` + its ggml runtime, `SC_MODEL_14B`, optionally `SC_GPT_OSS_GGUF` | **yes** |
| `test_cpu_worker.py` | `build_cpu.bat`, `SC_MODEL_14B` | **yes** |
| `test_glm_cpu.py` | `build_cpu.bat`, numpy (writes a 38 MB synthetic GLM-5.3 GGUF) | **yes** — the GLM port on the CPU path against a numpy forward |
| `test_smoke.py` | built DLL, NVIDIA GPU, `CUDA_PATH` | no |
| `test_backend_gemv.py` | GPU, `STRATAFREET_ROOT`, `SC_MODEL_14B`, optionally `SC_GPT_OSS_GGUF` | no |
| `test_backend_ops.py` | GPU, `STRATAFREET_ROOT`, `SC_MODEL_14B`, `SC_MODEL_FLASHNEXT` | no |
| `test_backend_seq.py` | GPU, `STRATAFREET_ROOT`, `SC_MODEL_14B`, `SC_MODEL_FLASHNEXT` | no |
| `test_score_14b.py` | GPU, `SC_MODEL_14B`, `SC_RUSTINGSTRATA`, `STRATAFREET_ROOT` | no |
| `test_partial_14b.py` | same, plus the routing profile under `%LOCALAPPDATA%\rusting_strata` | no |
| `test_prefill_14b.py` | GPU, `SC_MODEL_14B`, `SC_RUSTINGSTRATA`, `STRATAFREET_ROOT` | no |
| `test_flashnext.py` | GPU, ≥ 85 GB free RAM, `SC_MODEL_FLASHNEXT`, `SC_RUSTINGSTRATA`, `STRATAFREET_ROOT` | no |
| `test_score_inkling.py` | GPU, `SC_MODEL_INKLING`, `SC_LLAMA_SERVER`, `STRATAFREET_ROOT` | no |
| `test_score_glm5.py` | GPU, >= 24 GB free RAM, `SC_MODEL_GLM5`, `SC_RUSTINGSTRATA` | no |
| `verify_ram_lock.py` | GPU, `SC_MODEL_14B`, `SC_EVICT_GGUFS`, `STRATAFREET_ROOT` | no |

Example (PowerShell):

```powershell
$env:SC_MODEL_14B = "D:\models\Qwen3.6-14B-A3B-Q6_K.gguf"
$env:STRATAFREET_ROOT = "D:\src\stratafreet"
python tests\test_weights.py
python tests\test_cpu_moe.py
```

The GPU tests load models of 10 GB to 83 GB and hold `.gpu.lock` in the repository root while they run;
run them one at a time. `test_flashnext.py` and `verify_ram_lock.py` check available RAM first and skip
themselves when there is not enough.

## C ABI

Declarations are in `src\capi.h`. Every call that can fail returns 0 on success and nonzero on failure;
`sc_last_error()` returns the text of the last failure on the calling thread. A model handle is one
loaded model with its KV cache and recurrent state and must be used from one thread at a time.

| Function | Purpose |
|---|---|
| `sc_version()` | Version string of the DLL build (the string still says 0.0.1; the release version is `VERSION` = 1.00) |
| `sc_last_error()` | Text of the last error on this thread |
| `sc_nvrtc_version()` | NVRTC version found by the loader (needs no device) |
| `sc_selftest_kernel(out, n)` | NVRTC → CUBIN → driver-API self-test (`out[i] == 2*i`); allocates 4·n bytes of VRAM |
| `sc_load(gguf_path, options_json)` | Load a model onto the GPU. Options: `ctx`, `kv`, `threads`, `vram_margin_mb`, `seed`, `device`, `chunk`, `short` |
| `sc_n_vocab(h)` | Vocabulary size (logits row length) |
| `sc_decode(h, tok, logits_out)` | Feed one token, return the next-token logits |
| `sc_decode_sample(h, tok, sampler_json, out)` | Feed one token, sample from a JSON sampler (`temp`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `repeat_penalty`, `recent`, `seed`) |
| `sc_prefill(h, ids, n)` | Run a prompt through the batched prefill |
| `sc_prefill_logits(h, ids, n, logits_out)` | Prefill, then the logits of the last position |
| `sc_prefill_sample(h, ids, n, sampler_json, out)` | Prefill, then sample the next token |
| `sc_save_profile(h)` | Write the MoE routing profile (`profile-<arch>-<L>x<E>-<n>.f32`) |
| `sc_reset(h, keep_tokens)` | Clear the KV cache and recurrent state (`keep_tokens` must be 0) |
| `sc_stats(h)` | JSON of layers, positions, expert residency, hits and misses |
| `sc_free(h)` | Free the handle and its device memory |

For GLM-5.3-Flash (`glm5-next` / `glm5next`) `sc_load` builds the port of RustingStrata's GLM forward instead; its
options and environment variables are listed in `src\capi.h` and `docs/GLM5_NOTES.md`.

The load reads the calibration file `%LOCALAPPDATA%\rusting_strata\strata-<model stem>.json` for `chunk`,
`mtp` and `short` (explicit load options win). MTP is reported but not implemented. A routing profile is
read from the same directory at load and written after a run that recorded routing.

## Measured hardware

All speed, parity and residency numbers in this README and in `docs/DEVLOG.md` were measured on:

- NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM
- AMD Ryzen 7 5800X (8 cores / 16 threads, AVX2)
- 112 GB DDR4 system RAM
- Windows 10 x64

## License

This repository is a modified version of RustingStrata. RustingStrata License 1.0 section 3.4 requires
that modified versions are distributed under the same license, so:

- `LICENSE` is a copy of `LICENSE-RustingStrata.md`, the RustingStrata License 1.0.
- `LICENSE-RustingStrata.md` is kept verbatim, including the copyright notice of Vasyl Trefilov.
- `MODIFICATIONS.md` lists the significant modifications, as section 3.3 of the license requires.

The license permits use, modification and distribution, including commercial applications; it forbids
selling a product that is substantially RustingStrata or marketed as a substitute for it, and forbids
implying that Vasyl Trefilov endorses this work.

## Credits

- **RustingStrata 1.2.5 by Vasyl Trefilov** — the engine this port follows. Its decode, prefill, CUDA
  backend, expert cache, CPU expert code and kernel source are the reference for every function here.
- **llama.cpp** (MIT License, ggml authors) — its `src/models/inkling2.cpp`, `inkling2-loop.cpp` and the
  related model/attention/KV sources were the reference for the Inkling-2-Mini implementation, which is
  a StrataFreet addition and not part of RustingStrata 1.2.5. Its `src/models/glm5-next.cpp` was the cross-check for
  every formula of the GLM-5.3-Flash port (`docs/GLM5_NOTES.md`). The GGML quantization tables and scalar
  kernels also originate in llama.cpp/ggml via RustingStrata.
- **StrataFreet** — the host application this DLL was developed for. It is not included in this
  repository; its Python package is used by the tests as an optional reference oracle through
  `STRATAFREET_ROOT`.

## Repository layout

```
src/                 the port (GGUF, quant, config, weights, CUDA backend, model, prefill, expert cache, GLM, C ABI)
kernels/             kernels.cu and tables.cuh from RustingStrata 1.2.5; DIFF-vs-stratafreet.md
tools/               embed_kernels.py, gen_tables.py, gen_inventory.py, sc_score.cpp, sc_bench.cpp
tests/               ctypes-driven tests and the shared environment helper sc_env.py
docs/                DEVLOG.md (port history), INKLING2_NOTES.md (Inkling-2), GLM5_NOTES.md (GLM-5.3-Flash)
build.bat            build the product DLL, dev DLL and tools
build_cpu.bat        build the CPU expert test DLL
build_mx.bat         build everything into build_mx\
```

`MODIFICATIONS.md` lists every significant deviation from RustingStrata 1.2.5. `PORT_MAP.md` is the
Rust→C++ blueprint: file map, kernel launch table, mailbox protocol, and the profile/calibration file
formats.
