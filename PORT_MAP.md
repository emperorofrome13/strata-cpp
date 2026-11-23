# PORT_MAP — RustingStrata 1.2.5 → strata-cpp

Written 2026-11-20 against `native/strata_src_v1.2.5/` (tag v1.2.5, sha256 of the archive in
`native/strata_src_v1.2.5/INVENTORY.md`). Every claim below was read out of the Rust source, with the
file and line given so a reviewer can check it. Where something could not be read off the source without
a device (a launch shape that depends on a measured tune value), it is marked **confirm in W2-C**.

Scope: the decode and prefill path for the Qwen3.5/3.6 family and Flash-Next (`qwen4exp`), which is all
strata-cpp must reach. DeepSeek / Gemma4 / GLM / HY4 forwards, Vulkan, vision, LoRA, `serve`/`mcp`/`chat`
and the tokenizer stay out (see section 9).

---

## 1. The path, function by function

Load → placement → prefill → decode → sample. Line numbers are 1.2.5.

### 1.1 Open and parse

| Rust | Where | C++ | Notes |
|---|---|---|---|
| `Gguf::open` | `gguf.rs:112` | `src/gguf.cpp` | split-model discovery from `split.count`; requires the `-00001-of-0000N` shard |
| `Gguf::open_one` | `gguf.rs:141` | `src/gguf.cpp` | header walk, `general.alignment`, truncation check over every tensor |
| `Gguf::bytes` / `raw_bytes` | `gguf.rs:196`, `:205` | `src/gguf.cpp` | slice of the map at `data_offset + t.offset` |
| `GgmlType::from_u32` … `row_bytes` | `quant/mod.rs:15-46` | `src/quant.cpp` | 25 accepted type ids |
| `dequant_row` | `quant/mod.rs:60` | `src/quant.cpp` | scalar reference for every type |

`Gguf` keeps `BTreeMap` for metadata and tensors (ordered); the port uses `std::map`. Tensor iteration
order is therefore identical, which the `sci_gguf_info_json` comparison relies on.

### 1.2 Hyperparameters

| Rust | Where | C++ |
|---|---|---|
| `ModelConfig::from_gguf` | `config.rs:139` | `src/config.cpp` (ported, verified by `tests/test_weights.py`) |
| `arch()` table | `config.rs:40` | same — `qwen35`/`qwen35moe`/`qwen4exp` share `hybrid+q_gate`, `flint`/`qwen3*` differ by `aliases` |
| `LayerKind` / hybrid layer split | `config.rs:7`, `:152` | `LayerKind::Attn` every `full_attention_interval` (default 4), else `Delta` |
| `is_moe_layer` | `config.rs:200` | `n_expert > 0 && il >= n_dense` |

Facts that decide the model code, all from `config.rs`:
- `n_layer = block_count - nextn_predict_layers`.
- `head_dim` from `attention.key_length`, else `n_embd / n_head`.
- `n_rot` from `rope.dimension_count`, else `head_dim`.
- Flash-Next extras: `hyper_connection.count`/`.low_rank` (`hc_n`, `hc_rank`),
  `embedding_length_per_layer_input` (`n_embd_ple`), `attention.indexer.head_count`/`.key_length`/`.top_k`.
- `attn_sinks` is inferred from the *existence* of `blk.0.attn_sinks.weight`.
- DeltaNet dims must satisfy `nv % nk == 0` and `ssm_d_inner % nv == 0` (`config.rs:190`).

### 1.3 Placement and weights

| Rust | Where | C++ |
|---|---|---|
| `Tensor` (view over the map, `.expert(e, n)` slicing) | `weights.rs:11` | `src/weights.cpp` (ported and verified) |
| `Weights` (per-layer name lookup, `validate`, placement) | `weights.rs:72` | same — `ssm_ba` split verified against the file's own bytes |
| `Prefetch` reach-ahead thread | `weights.rs:251` | same — one thread, `Vec<(addr, len)>` batches over a mutex/condvar queue |
| `will_need` / `cold` (page-cache readahead / drop) | `weights.rs:224`, `:244` | same file — **no-ops on Windows**, exactly as the Rust `#[cfg(not(target_os = "linux"))]` shim |
| `ram_cap(margin, max_ram)` / `total_ram` | `memory.rs:39`, `:9` | `src/memory.cpp` (ported, `GlobalMemoryStatusEx`) |
| `ExpertCache::new` (slot pools + initial fill) | `expert_cache.rs:74` | `src/expert_cache.cpp` (W3-E2) |

`ExpertCache` details that the port must keep (section 3.2 has the file format):
- experts of equal byte size share a pool; a slot holds gate, up, down back to back, each matrix aligned
  to 16 bytes (`expert_cache.rs:88`, comment: "GEMV kernels read 8-byte words").
- a pool is never larger than `MAX_POOL_FULL_BYTES` = 8 GiB (`expert_cache.rs:79`).
- slot allocation is `frac * experts` with `frac = budget / total`, retried at `0.9 * frac` on an
  allocation failure (`expert_cache.rs:120-134`).
- initial fill order: hottest by saved profile first, ties in layer order (`:146`).

### 1.4 Prefill

| Rust | Where | C++ |
|---|---|---|
| `GpuModel::load*`, `load_part` | `gpu_model.rs:293-346` | `src/gpu_model.cpp` (W3-E1) |
| `prefill` (chunked batched) | `gpu_model/prefill.rs:79`+ | `src/prefill.cpp` (W3-E3) |
| `Scratch` (per-stage activation buffers) | `gpu_model/prefill.rs:31` | same |
| chunk size (`STAGE` / calibration) | `gpu_model/prefill.rs:12`, `main.rs:577` | from `strata-<stem>.json` (section 3.3) |

`prefill.rs` runs `STAGE = 16` rows at a time through the same layer function with `nt > 1`, using
`attn_prefill_<hd>` instead of `attn_decode_*`, `gemm` instead of `gemv`, and `moe_gemv` with `t` tokens.

### 1.5 Decode

| Rust | Where | C++ |
|---|---|---|
| `GpuModel::step` | `gpu_model.rs:827` | `src/gpu_model.cpp` |
| `GpuModel::step_greedy` | `gpu_model.rs:852` | same |
| `GpuModel::verify` / `verify_greedy` / `verify_run` (MTP + spec) | `:858`, `:869`, `:902` | same (MTP optional, see section 9) |
| `GpuModel::forward` | `gpu_model.rs:1310` | the per-token graph |
| `GpuModel::layer` | `gpu_model.rs:1753` | **the function to port first** — it is the whole model |
| `put_embed`, `step_embed`, `gnorm`, `hc_at`, `hmv`, `hc_read`, `ple`, `mv` | `:814`, `:661`, `:1345`, `:1350`, `:1356`, `:1362`, `:1382`, `:1395` | same file |
| `finish` (logits + argmax) | `:1647` | same |

`layer()` order (from `:1753` onward): norm → hyper-connection read (`hc_read`) → mixer
(`Mixer::Attn` / `Mixer::Delta`) → `hc_write` → post-norm → per-layer embedding injection (`ple`) →
FFN (`Ffn::Dense` / `Ffn::Shared` / `Ffn::MoE`) → residual streams.

`cpu_only(name)` (`gpu_model.rs:39`) returns true only for `per_layer_token_embd.weight`: the PLE table
stays on the CPU (section 4).

### 1.6 Sample

| Rust | Where | C++ |
|---|---|---|
| `Sampler` (greedy / temp / top-k / top-p, seed) | `ops.rs:119` | `src/sampling.cpp` (W3-E1) |
| `log_probs` helper (score output) | `main.rs:531` | `tools/sc_score.cpp` (W4-A) |
| `decode()` bench loop | `main.rs:1689` | `tools/sc_bench.cpp` (W4-A) |

The sampler also carries StrataFreet additions with no Rust side: `min_p`, `presence_penalty`, `repeat_penalty`
and the recent-token window, matched to `stratafreet/engine.py _sample` (see MODIFICATIONS.md, "Sampler" section).
`sampler_from_json` (`src/sampling.cpp`) is the shared parser for both DLLs' sampler JSON.

---

## 2. Kernel launch table

Source: `src/cuda/kernels.cu` (kernel bodies, copied verbatim to `kernels/kernels.cu`) and the launch
wrappers in `src/cuda/mod.rs` (`Backend` impl for `Gpu`). `cfg(grid, block, smem)` builds the launch
config (`cuda/mod.rs:410`). `k` = active experts, `nt` = tokens, `hd` = head_dim.

| Wrapper | Kernel | grid | block | dynamic smem | Line |
|---|---|---|---|---|---|
| `dequant` | `dequant_<ty>` | `(rows, 1)` | 256 | 0 | 905 |
| `quant_xq` | `quant_xq` | `(nt, 1)` | 256 | 0 | 910 |
| `quant_q8` | `quant_q8` | `(n_x, 1)` | 256 | 0 | 601 |
| `gemv` / `gemvn` / `gemvq` / `gemvq2` | `gemv_<ty>` / `gemvn_<ty>` | `(rows / (wpb*rpw), k)` | `32*wpb` | `xq_smem(ty, cols)` | 645-648 |
| `gemm` / `gemm_dense` | `gemm_<ty>` | `(rows / 64, n_tiles)` → `(rows / 64, nd / 32)` when `sm` | 128 (`GEMM_BM`) | 0 | 556-561 |
| `moe_gemv` | `moe_gemv_<ty>` | same shape as `gemv`, `k` experts on the y axis | `32*wpb` | `xq_smem` | 889 |
| `rms_norm` | `rms_norm` | `(rows, 1)` | 512 | 0 | 919 |
| `rms_norm_streams` | `rms_norm` | `(nt*hc, 1)` | 512 | 0 | 924 |
| `rms_norm_rope` / `_kv` / `_qk` / `_qk_rows` | `rms_norm_rope` | `(rows, 1)` | 256 | 0 | 954-975 |
| `rope_pairs` | `rope_neox` | `(heads / per, 1)` | `n_rot / 2` | 0 | 943 |
| `argmax_rows` | `argmax_part` then `argmax_rows` | `(AMAX_PARTS, rows)` then `(rows, 1)` | 256 then `AMAX_PARTS` | 0 | 932-933 |
| `l2_norm` | `l2_norm` | `(rows, 1)` | 128 | 0 | 938 |
| `swiglu` / `swiglu_q` | `swiglu` / `swiglu_q` | `(n / 256, rows)` | 32 | 0 | 994 |
| `add` | `add` | `(n / 256, 1)` | 256 | 0 | 989 |
| `conv_silu` | `conv_silu` | `(ch / 128, 1)` | 128 | 0 | 1004 |
| `deltanet` | `deltanet_seq` | `(nv, dv / (8*rows))` | 256 | 0 | 1011 |
| `attn_decode_<hd>` | `attn_decode_<hd>` | `(nh*nt, ns1)` | 256 | `8*hd*4` | 691 |
| `attn_decode_gqa*` | `attn_decode_gqa4_<hd>` / `attn_decode_gqa_<hd>` | `(nh/g*nt, ns)` | 256 | `8*hd*4 + cp` | 698 |
| `attn_prefill_m` | `attn_prefill_<hd>` | `(t/64, nh)` | 128 | `attn_prefill_smem_bk(hd, bk)` | 1023-1028 |
| `qsa_pool` | `qsa_pool` | `(t/4+1, 1)` | 128 | 0 | 1034 |
| `qsa_select` | `qsa_score` then `qsa_select` | `(64, rows/8)` then `(rows, 1)` | 256 then 1024 | 0 | 1039-1040 |
| `router_topk` | `router_topk` | `(t, 1)` | 32 | 0 | 1115 |
| `moe_route` | `moe_route` | `(1, 1)` | 256 | 0 | 1121 |
| `moe_prefetch` | `moe_prefetch` | `(lines/256, k)` | 256 | 0 | 1131 |
| `moe_combine` / `_seq` | `moe_combine` / `moe_combine_seq` | see `:1135`, `:1044` | — | — | 1044, 1135 |
| `norm_gate` / `_q` / `_sig` | `norm_gate` | `(heads, 1)` | 128 | 0 | 1056-1062 |
| `hc_mix` / `hc_write` | `hc_mix` / `hc_write` | `(n/256, nt)` | 256 | 0 | 1072-1077 |
| `ple_gate` | `ple_gate` | `(hc, nt)` | 256 | 0 | 1082 |
| `ple_conv` | `ple_conv` | see `:1086` | — | — | 1086 |
| `kv_store` | `kv_store` | see `:1093` | — | — | 1093 |

Constants (`backend.rs`): `ATT_SPLITS = 32`, `ATT_SPLITS_1 = 16` (one token), `ATT_GQA_MIN = 2048`,
`GEMM_BM = 128`. `ATT_SPLITS <= 256` is a compile-time assert (`attn_merge` keeps one weight per split
in a 256-thread block).

Every `<ty>` suffix comes from `kname(prefix, ty)` = `prefix + "_" + ty.name().to_lowercase()`
(`cuda/mod.rs:431`).

### 2.1 What is prepended to `kernels.cu`

`source_with()` (`cuda/mod.rs:158`) builds the compilation unit as:

1. six grid tables as `__device__ const` arrays: `iq2xs_grid[512]`, `iq2xxs_grid[256]`, `iq2s_grid[1024]`,
   `iq1s_grid[2048]`, `iq3xxs_grid[256]`, `iq3s_grid[512]`, plus `ksigns_iq2xs[128]`,
   `kvalues_iq4nl[16]`;
2. the macros `LLAMA_Q8`, `KV_Q8`, `GEMM_TC`, `GEMM_BM`, `GEMM_BK`, `GEMM_MIN_BLOCKS`, `GEMV_LOADW`,
   `FA_BQ`, `FA_BK`;
3. `include_str!("kernels.cu")`.

`LLAMA_Q8` is `llama_numerics()` (`STRATA_LLAMA_NUMERICS`), `KV_Q8` is `kv_q8()`, `GEMM_TC` is 1 unless
`STRATA_GEMM_FMA`. The tune values default to `GEMM_BK = 32`, `GEMM_MIN_BLOCKS = 1`, `GEMV_LOADW = 1`,
`FA_BQ = 64`, `FA_BK = 32`, and can be overridden per family by environment
(`Tune::from_env`, `cuda/mod.rs:125`). **A port must expose the same overrides or the tuned numbers
will not reproduce.**

Compilation is to a **cubin** for the device's `sm_XY`, cached on disk under a key of
`(source, arch, nvrtc version)` (`cache_key`, `cuda/mod.rs:362`). StrataFreet's Python path compiles the
same source to **PTX** instead (`stratafreet/strata_kernels.py`); both work, cubin is what Strata
benchmarks.

`dyn_shmem_need()` (`cuda/mod.rs:247`) opts in to more than 48 KiB for `attn_prefill_*`,
`attn_decode_gqa*` and `gemvn_*`; every other kernel relies on the default 48 KiB. This is where a
port silently loses a launch, so port it before the attention kernels, not after.

---

## 3. Protocols and file formats that must stay compatible

### 3.1 The CPU mailbox (`cpu_worker.rs`)

One host-mapped pinned buffer per MoE layer. Word offsets (`cpu_worker.rs:15-32`), all `u32`, flags on
separate cache lines:

| word | name | meaning |
|---:|---|---|
| 0 | `REQ` | the GPU asks the worker to compute this layer's missed experts |
| 1 | `NTOK` | tokens in the request (1 for decode, up to `TMAX` for a verify) |
| 2 | `SKIP` | raised *instead of* `REQ`, with no data, when nothing in the layer missed (CUDA only) |
| 8 | `VERDICT` | Vulkan only: the combine's verdict on its wait (0 open, 1 answered, 2 gave up) |
| 16 | `DONE` | the worker answered; the combine kernel waits on it only if the layer missed |
| 32 | `IDS` | start of the data area: ids, weights and slots for `NTOK` tokens |

`TMAX = 17` (1 token + up to 16 draft tokens). Layout from `Mailbox::new` (`cpu_worker.rs:33`):

```
h      = round_up16(IDS + 3 * TMAX * k)      // ids, weights, slots: 3 arrays of NTOK*k
y      = h + TMAX * n                        // results (TMAX rows of n floats)
h_     = y + TMAX * n                        // activations (TMAX rows of n floats)
stride = round_up16(h_ )                     // one layer's mailbox size
```

`k` = active experts, `n` = model width. Token `t`'s ids are at `t*k`, and its activation row at
`h + t*n`; the result row at `y + t*n`. The GPU writes the request, then `fence(Release)` before raising
`REQ` (or `SKIP`); the worker answers with `DONE` and the combine kernel only waits when the layer
missed. That asymmetry is the point of the design — a layer with no miss costs the GPU nothing.

### 3.2 Expert routing profile

- Directory: `cache_dir()` = `$XDG_CACHE_HOME` || `%LOCALAPPDATA%` || `~/.cache`, then `rusting_strata`
  (`expert_cache.rs:53`). On this machine: `%LOCALAPPDATA%\rusting_strata`.
- Name: `profile-{arch}-{n_layer}x{n_expert}-{n_tensors}.f32` (`expert_cache.rs:59`) — the tensor count
  is part of the key, so a differently sharded file gets its own profile.
- Format: **raw little-endian `f32`, `n_layer * n_expert` values**, index `layer * n_expert + expert`
  (`:104`, `:296`). No header. A file shorter than `4 * n_layer * n` is ignored.
- Semantics: long-term decayed routing counts (`LONG_KEEP = 0.999` per adapt; `KEEP = 0.95` for the
  short-term `counts`). `save_profile` merges: layers this cache holds are replaced, other layers keep
  the file's values (`profile_update`, `:300`), and a run that never recorded does not write at all.
- `STRATA_FIXED_EXPERTS` disables placement-by-profile and `adapt` entirely.
- Swap rule (`swap`, `:255`): per pool, take the `max_swaps` hottest missed experts and the `max_swaps`
  coldest residents, and swap only while `counts[hot] >= 1.5 * counts[cold] + 2.0`.

### 3.3 Calibration file

`strata-{model_stem}.json` (CUDA; `-{backend}` when not CUDA) in the same `cache_dir()`
(`main.rs:577`). Written by `calibrate`, read for the prefill chunk size and MTP settings. strata-cpp
must **read** it (W3-E3 chunk size) and ideally write it, so the two engines share one calibration.

### 3.4 KV cache

`kv_words(n)` = `n * 17 / 32` words when `KV_Q8`, else `n` (`cuda/mod.rs:1091`): 32 values become 34
bytes (32 int8 codes + 2 bytes of f16 scale) per row. `kv_store` writes both layouts.

---

## 4. Flash-Next specifics

- **Per-layer n-gram embedding** (`per_layer_token_embd.weight`, 28.8 GB IQ4_NL): `cpu_only()` returns
  true for this one name (`gpu_model.rs:39`), so `Weights` keeps it in the map and it is read on the CPU.
  The n-gram geometry is in `model.rs:46-53`: `PLE_NGRAM = 3`, `PLE_HEADS = 8`, `PLE_KERNEL = 4`,
  `PLE_HIST = (PLE_KERNEL-1) * PLE_NGRAM`; `Ple { mult, off, vocab, eos }` is built from GGUF metadata.
  The GPU side is `ple_gate` + `ple_conv` (`gpu_model.rs:1382`). This is why Flash-Next needs 83 GB of
  RAM: 52 GB experts + 27 GB table.
- **Hyper-connections**: `hc_n` residual streams with `hc_rank`, mixed by `hc_mix` / `hc_write` through
  `HcRead` (norm/down/up/inject). `row_width()` is `hc_n * n_embd` (`gpu_model.rs:1596`) — every row
  buffer in the decode graph is that wide.
- **QSA sparse attention**: `QSA_POOL = 4`, `QSA_D = 128`, `QSA_H = 4`, `QSA_RING = 32 * QSA_D`,
  `QSA_ROWS = 64` (`gpu_model.rs:86-91`). Dense while the visible pool count is under `idx_top_k`
  (2048 tokens), indexer-selected beyond.
- **DeltaNet layers**: hybrid split from `full_attention_interval` (4), sigmoid-gated, `deltanet_seq`.

---

## 4b. Inkling-2-Mini specifics (StrataFreet addition)

Inkling-2-Mini is not in RustingStrata 1.2.5; the reference is ggml-org's `inkling2.cpp` /
`inkling2-loop.cpp` (PR #389) with unsloth's looped GGUF layout. The port is line-by-line against that
reference (`build/ref_pos0.py` is the independent numpy ground truth). RustingStrata's machinery is reused
where the shape matches.

- **Architecture** (`config.cpp`, `Arch::Kind::Inkling2`): 24 physical blocks (`blk.0`-`blk.23`), 128
  experts, top-8, hidden 2304, expert width 1152. The engine runs **48 logical slots** (`n_layer`): the
  prelude (slots 0-5), the loop core blocks 6-17 replayed for 3 passes (slots 6-41), and the tail blocks
  18-23 (slots 42-47). `loop_layer_of(il)` maps a slot to its physical block and `Weights::layer` serves
  the same tensors to every pass; the expert cache is keyed by the **physical** block (`n_layer_phys`),
  while the CPU mailbox and the worker's expert lookup remap slot -> physical (each pass routes
  independently). `scratch_sizes` adds the decode buffers: `bks`/`bnk`/`bkl`/`pre`, the message pairs
  `mat`/`mkv`, the controller gates `gat`/`g64`/`cg`/`cf`, the fused `gu` rows, `tbl`/`rel`/`sink`,
  `kpj`/`vpj`, `moe`, `ao`, `inpb`, each allocated `kTmax` rows.
- **Experts** (`expert_cache.{h,cpp}`): `kExpertMatsInkling2` is four matrices per expert - the fused
  gate/up (`ffn_gate_up_exps`, 2 x 1152 rows), the down matrix, and the per-expert `gu`/`dn` scales read
  at compute time. The probe/adapt logic is the RustingStrata one over this layout; the profile name is
  `profile-inkling2-24x128-N.f32`.
- **Decode** (`gpu_model.cpp`, `forward_inkling2`): the port of the PR #389 graph. The block bank `bks`
  holds the residual streams at the `attn_res` boundaries; later loop passes reset the residual to the
  prelude output (`pre`) and read the previous pass's bank rows and last output (`bkl`) through two depth
  mixes (`loop_res_query`, `loop_kv_res_query`) whose columns feed
  `gate_pack`/`loop_down`/`loop_gain`/`loop_forget`/`control_mix` for Q, K, V and the FFN. The layer body
  is `rel_table` (per-query relative-bias table with the sink logit) + `attn_decode_gq` (the
  `bias`/`extent` form) + `sig_mul` + `conv_ink` shortconvs (K/V dynamic taps on full-attention blocks,
  static on SWA) + `clip_rows` + the fused gate/up `moe_gemv` + `situ_q` + `moe_combine` with the
  per-expert down scale. `STRATA_INK_XQ=1` selects the int8-quantized activation path; the default is
  exact f32. `STRATA_INK_DEBUG` / `STRATA_INK_DEBUG_FILE` drive the row-grid capture
  (`build/grid_check.py`); `STRATA_INK_DUMP` writes the slot-1 MoE record (`build/moe_check.py`).
- **Prefill** (`gpu_model.cpp`, `prefill.cpp`): `forward_inkling2_chunk` runs chunks of up to 16 tokens through
  the slot graph; the per-token work uses the decode launches, while the routed-expert pass is batched over the
  chunk (one `moe_route` with t rows, one fused gate/up `moe_gemv` with `xg = ke`, one `situ_q`, one down
  `moe_gemv`, one `moe_combine`). Per-token state lives in per-token regions of the decode buffers; the chunk
  body is captured by the backend's graph cache (one graph per chunk size and gq regime). The load enables it
  only when the expert cache is fully resident (all-resident IQ3_XXS at ctx 8192: pp512 145.3 / tg128 98.8
  tok/s); a partly resident cache keeps the token-by-token steps, measured faster on the paged Q8 file
  (14.4 / 15.8). The A/B against the token path is bit-identical on the all-resident quant.
- **ABI and app**: `capi.cpp` `sc_load` accepts the usual options (`ctx`, `ram`, `pin`, `threads`,
  `vram_margin`) and `"kv": "f16" | "q8"` (q8 selects the backend's int8 KV kernels; the app forwards it
  for Inkling when `kv_dtype` is q8, which at ctx 31232 is what lets all 3072 IQ3_XXS expert slots stay
  resident); `stratafreet/strata_cpp.py` `ARCHS` includes `"inkling2"` so the engine selects this backend
  automatically; `tests/test_score_inkling.py` is the acceptance test.

---

## 5. Threading

- `--threads N` sets the CPU expert pool **and** rayon's global pool. This machine's fast setting is 7
  (leaving a core for the GPU-feeding thread); 16 logical threads halves decode
  (`HANDOFF.md`, 2026-11-19 table).
- `physical_cores()` (`main.rs:466`) reads the topology to pick the default.
- `cpu_worker.rs` runs one thread that spins on the mailbox flags (`read_volatile` in Rust terms; the
  C++ port needs `std::atomic` with relaxed/acquire pairing that matches the fences, or `_ReadWriteBarrier`
  plus volatile — decide once in W2-D and record it in `MODIFICATIONS.md`).
- One thread for file `Prefetch`, one for the worker, one for the GPU feed. Nothing else.

---

## 6. Where the 42.3 tok/s comes from (why the port must be line-by-line)

Strata 1.2.5 on this machine: 42.3 tok/s decode at `--threads 7`, 12.5 ms of worker CPU time per token,
80% expert hit. StrataFreet's Python/Triton port of the *same design* gets 21 tok/s at 63% hit with the
same kernels. The gap is therefore in placement, adapt cadence and host overhead — not in the kernels.
That is the reason the plan forbids rewriting kernels and asks for a line-by-line port of
`expert_cache.rs`, `gpu_model.rs::layer` and `cpu_worker.rs` rather than a re-derivation.

---

## 7. Corrected package split

The plan's split holds, with these corrections read off the source:

1. **`cuda/nvrtc.rs` must come before the backend, not with it** — done (Wave 0-B). The Windows DLL name
   carries the CUDA version and `nvrtc-builtins64_*.dll` must be pre-loaded by path, or every compile
   fails on a machine with more than one toolkit. Linking `nvrtc.lib` pins one toolkit and breaks the
   DLL load: the port loads it dynamically exactly as the Rust does.
2. **`dyn_shmem_need` belongs to the first backend drop** (section 2.1), not to a later tuning pass.
3. **`expert_cache.rs` and `cpu_worker.rs` are one package**, not two: `gpu_model::layer` interleaves
   them (request/skip/done, combine wait), and neither can be verified alone.
4. **`main.rs` is not "the bench/score parts"** — it is 2752 lines and holds `Runner`, the engine, the
   calibration writer and the JSON schema (`bench_record`, `:663`). W4-A should port `bench_record` and
   the `score` JSON by reading those functions, not the whole file.
5. **`model.rs`'s CPU forward is needed by `cpu_moe.rs`** (`experts()` calls into `Model`), so W2-D
   depends on `model.rs`'s weight map at minimum. `Model` here is the *weight* model; the CPU forward
   (`model.rs` `mv`, `:99`) is only the oracle.
6. `ngram.rs` is **not** needed for Flash-Next decode: `ngram_spec_step` (`gpu_model.rs:1545`) is a
   speculative path, opt-in via a flag, and the bench that produced 42.3 tok/s uses `--tg` greedy decode
   without it. Keep it out.

---

## 8. Small fast-iteration model

Found during the port (checked 2026-11-20, no load):

| File | Arch | Why |
|---|---|---|
| `<SC_MODEL_14B>` (Qwen3.6-14B-A3B, Q6_K) | `qwen35moe` | **pick this.** Same family and same MoE path as Flash-Next, 10.5 GB, 733 tensors, 1 shard, loads in ~30 s instead of ~6 min. StrataFreet already benchmarks it at `--vram-gb 6`. |

Verified with `tests/test_gguf.py`: 47 metadata keys, 733 tensors, 0 mismatches against
`stratafreet/gguf_io.py`. Its quant types are exercised by the 25-type exact dequant test.

Secondary (dense, for the non-MoE path): a dense Qwen3.5-4B Q8_0 GGUF.

---

## 9. Out of scope, and why

`serve.rs`/`mcp.rs`/`chat.rs`/`tokenizer.rs` (StrataFreet passes token ids), `vision/*`, `hf.rs`,
`vulkan/*`, `deepseek/*`, `gemma4/*`, `glm/*`, `ngram.rs` (section 7.6), LoRA, multi-GPU split
(`load_split*`, out until single-device parity is reached), MTP drafting (`mtp_*`: optional, and the
reference bench does not use it).

---

## 10. Open questions to settle in Wave 2/3

1. `gemv_cfg`'s rows-per-warp rule (`rpw()`, `:617`) reads a measured tune table; confirm the C++ port
   reproduces it from `tools/sk_tune.py`'s numbers rather than by copying the constants alone.
2. `attn_prefill_smem_bk` has a Turing head_dim-256 special case (`bk = 16`); confirm on this card
   (`sm_120`) that the `FA_BK` path is the one taken.
3. `moe_combine`'s wait: whether the stall-recovery path in `layer()` (`gpu_model.rs:~1900`) is reached
   on this machine at 80% hit rate — instrument before porting the recovery, not after.
4. `LLAMA_Q8` (llama.cpp numerics mode) changes rounding inside the kernels. The 42.3 tok/s bench does
   **not** set `STRATA_LLAMA_NUMERICS`, so the port must default it off and keep the switch.
