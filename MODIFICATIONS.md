# Modifications

strata-cpp is a modified version of RustingStrata 1.2.5 by **Vasyl Trefilov**, under the RustingStrata
License 1.0 (see `LICENSE-RustingStrata.md`, kept verbatim). This file lists the significant
modifications, as that license requires.

## Standing deviations (every file)

These are language consequences, not algorithm changes. Each ported function names the Rust function it
ports in a comment (`// port of gguf.rs: Gguf::open`), and the algorithm, constants and loop order are
kept.

| Rust | C++ | Reason |
|---|---|---|
| `anyhow::Result` + `Context` | `bool` return + a `std::string& err` out-parameter (or `std::optional`) | the C ABI carries errors as text (`sc_last_error`) |
| `BTreeMap<String, _>` | `std::map<std::string, _>` | same ordered container, same iteration order |
| `memmap2::Mmap` | `CreateFileW` + `CreateFileMappingW` + `MapViewOfFile` | no crate dependencies; read-only view, same lazy paging |
| `libloading::Library` | `LoadLibraryExW` + `GetProcAddress` | same dynamic loading, including the `nvrtc-builtins` pre-load |
| `half::{f16, bf16}` | hand-written `h2f`/`bf` bit conversions in `quant.cpp` | exact widening only (no rounding needed in this direction) |
| `rayon` `par_chunks_mut` | `par_chunks_mut` in `weights.cpp`: the same chunk boundaries, handed to whole `std::thread` workers | rayon is not a dependency; chunk boundaries and per-element summation order are unchanged, so results are bit-identical |
| Rust slice indexing with bounds checks | explicit `src_len` checks in `dequant_row`, `raw_bytes` | the Rust code panics; the C++ refuses and reports |

## Per-file notes

### `src/quant.cpp` (port of `quant/mod.rs`)

- `dequant_row` takes `src_len` and returns `false` instead of asserting: a truncated buffer is the one
  case where Rust would panic and a DLL must not.
- The grid/sign/value tables are `constexpr` arrays generated from `quant/tables.rs` by
  `tools/gen_tables.py` (`src/tables_host.h`) rather than prepended as text; the values are identical.
- **MXFP4 (type 39) is ported as a StrataFreet addition** (see the MXFP4 section at the end). Types 250 (FP8) and
  251 (NVFP4) are still not ported on purpose: Strata's own `GgmlType` does not have them (`quant/mod.rs:15`); they are
  StrataFreet's additions to its own copy of `kernels.cu`. A model whose experts use 250 or 251 cannot run on
  strata-cpp yet, and `ty_from_u32` rejects them with the same message Strata prints.

### `src/gguf.cpp` (port of `gguf.rs`)

- `Gguf::bytes` returns the byte length through its `size_t& len` out-parameter; Rust's `&[u8]` carries
  that length implicitly. (`tests/test_weights.py` caught the first version, which left `len` at 0.)

- The truncation check multiplies `dims` with overflow guards and compares against the mapped size, as
  the Rust does with `checked_mul`. The `O(n·dims)` semantic is unchanged.
- `Value` keeps `F32`/`F64` payloads as raw bits and converts on `as_f32()`, so no precision is lost
  between parse and read. `BTreeMap`'s `Value` enum order is preserved as `Tag`.
- Paths are converted from `std::string` to `std::wstring` by widening, i.e. **non-ASCII model paths are
  not supported yet** (the Rust uses `Path`, which is UTF-16 on Windows). Model paths on this machine are
  ASCII; this is the one known input limitation and it is recorded here rather than hidden.

### `src/config.cpp` (port of `config.rs`)

- `Arch::aliases` (`&'static [(&str, &str)]`) and `ModelConfig::aliases` are
  `std::vector<std::pair<std::string, std::string>>` copies of the same two static tables; `arch(name)`
  still hands out pointers to static `Arch` objects.
- `Option<[usize; 4]>` (`mrope_sections`) is `std::optional<std::array<size_t, 4>>`; the "every entry must
  parse" rule of the Rust `and_then(|a| ...)` closure is kept as an explicit loop.

### `src/weights.cpp` (port of `weights.rs`)

- `Tensor` carries `len`, since a C++ pointer has no length; `row()` refuses an out-of-range row instead of
  panicking, and the internal invariants (`rows * row_bytes() == len`) are `assert`s, as in Rust.
- The synthesized `ssm_beta`/`ssm_alpha` bytes are owned by `Weights` (`std::vector<std::unique_ptr<uint8_t[]>>`)
  instead of `Box::leak`, so a reload frees them; the tensors in `built` point into that storage.
- `Weights` keeps its `ModelConfig` (the Rust version borrows a `&'static` alias table from a local);
  `clear_baked` is ported, `bake_lora` is not (LoRA is out of scope in `STRATA_CPP_PLAN.md` section 4).
- `will_need` and `cold` are **no-ops**, which is what the reference does on Windows: `weights.rs` calls an
  empty `madvise` shim under `#[cfg(not(target_os = "linux"))]`. Porting them as working readahead would
  change behaviour relative to Strata 1.2.5.
- `Prefetch` is one `std::thread` behind a mutex/condvar queue instead of an `mpsc` channel; `destroy()`
  (from the destructor) drains the queued batches and joins, as dropping the Rust sender does.

### `src/memory.cpp` (port of `memory.rs`)

- Windows only: `GlobalMemoryStatusEx`'s `ullTotalPhys`, the same call the Rust Windows branch makes.

### `src/nvrtc.cpp` (port of `cuda/nvrtc.rs`)

- Ported early (Wave 0-B) rather than with the backend, because linking `nvrtc.lib` pins one CUDA
  toolkit and makes the DLL fail to load on a machine with several installed. The candidate list,
  `STRATA_NVRTC`, `STRATA_NVRTC_OPTS` and the `nvrtc-builtins` pre-load are the Rust behaviour.
- Compiles to **cubin** for the device's `sm_XY` like Strata; StrataFreet's Python path compiles to PTX.

### `src/capi.cpp`, `src/dev_api.cpp`

- `capi.cpp` implements the ABI from `STRATA_CPP_PLAN.md` section 5. `sc_load`/`sc_prefill`/`sc_decode`/
  `sc_reset`/`sc_stats`/`sc_free` are declared in the plan but **not implemented yet** — they arrive with
  Wave 3. Only `sc_version`, `sc_last_error`, `sc_nvrtc_version` and `sc_selftest_kernel` exist today (the header is `src/capi.h`).
- `dev_api.cpp` is an addition, not in the plan: a second DLL exposing the GGUF inventory and the quant
  reference so the tests can drive them from Python. It exists because the plan's test strategy is
  "ctypes-driven tests vs Python oracles", which needs a stable surface that is not the product ABI.

### `src/cpu_moe.cpp`, `src/cpu_moe.h` (port of `cpu_moe.rs`, and rayon's pool)

Ported: `quantize_q8k`, every integer kernel of `int_dot` (scalar and AVX2: IQ2_XS, IQ2_XXS, IQ2_S, IQ3_XXS,
IQ3_S, IQ1_S, IQ1_M, IQ4_XS, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, plus
TQ1_0 and TQ2_0 which have only the scalar kernel, as in Rust), `batch_rows`, `rows_multi`, `matvec`, `experts`,
`experts_multi`, the expert part of `model.rs`'s weight map (`load_expert_layer`), and the `int_dots_match_dequant`
unit test (`int_dots_selftest`). F32, F16 and BF16 experts take the f32 fallback path, as in Rust.

- **rayon -> `ThreadPool`.** N persistent workers that sleep between jobs; `for_each(n, f)` hands out indices
  under a mutex and the caller waits, as rayon does for a non-pool thread. Tasks are the same 16-row chunks as
  Rust's `par_chunks_mut(16)`; each output element is computed by one task in a fixed order, so the result does
  not depend on scheduling. The pool is passed explicitly (rayon's global pool is implicit). One caller at a time.
- **Default thread count.** `default_threads()` takes the physical core count (plan section 5). Strata's
  `physical_cores()` returns `None` on Windows (only the Linux branch reads the topology), so Strata falls back
  to rayon's default of every logical core. On this machine (Ryzen 7 5800X, 8 cores / 16 threads) that is 16
  in Strata and 8 here. Not changed to match Strata: the plan asks for physical cores, and `--threads` overrides
  either way.
- **`q2_0_crumbs`.** Only the AVX2 form is ported (the one the Rust code compiles unless `avx512vbmi` and
  `avx512vl` are enabled). Both forms produce the same crumb bytes; this CPU has no AVX-512 so the Rust native
  build takes the same path.
- **LoRA is absent.** The `m.lora_delta(..)` calls and the `if !m.loras.is_empty()` branch of `experts_multi`
  are not ported; LoRA is out of scope (plan section 4). A model loaded with LoRA cannot use this path.
- **Errors instead of panics.** Rust indexing panics on a missing tensor, a bad expert id or a shape mismatch;
  `load_expert_layer` and `experts` check these and return `false` with a message. `load_expert_layer` checks the
  three expert tensor shapes against `n_embd`, `n_ff_exp` and `n_expert`, which the Rust code leaves to indexing.
- **Float sums start at +0.0.** Rust's `Iterator::sum` over `f32` may start at -0.0; this changes only the sign
  of an exact zero.
- `STRATA_SCALAR` is read once per batch call (Rust reads it at each `int_dot`); same switch.
- **`cpu_worker.rs`** is ported in `src/cpu_worker.{h,cpp}` (see the section below).

Test ABI: `src/cpu_test_api.cpp` (built into `build\strata_cpu_test.dll` by `build_cpu.bat`, not the product
DLL) exposes `sct_open`, `sct_close`, `sct_info`, `sct_experts`, `sct_experts_multi`, `sct_dot_selftest`,
`sct_default_threads` and `sct_last_error` for `tests\test_cpu_moe.py`.

### `kernels/`

- `kernels.cu` and `tables.cuh` are unmodified. See `kernels/DIFF-vs-stratafreet.md` for how 1.2.5's
  kernel file differs from the copy StrataFreet has been shipping, and why nothing was merged back.

### `src/backend.h`, `src/backend.cpp` (port of `backend.rs`, W2-C part 1)

- The whole `Backend` trait is declared, with the same method names and argument order. Rust trait default
  bodies are kept as C++ virtual defaults (`batch`, `beside`, `rms_norm_streams`, `rope_neox`, `rope_mrope`,
  `rms_norm_rope`, `rms_norm_rope_qk`, `gemvq2`, `swiglu_q`, `norm_gate_q`, `attn_decode`, `attn_prefill`,
  `attn_decode_q`, `moe_down_combine`, ...); the required methods are pure virtual. Rust's default bodies that
  return "not implemented" (QSA, hyper-connections, PLE, `norm_gate_sig`, per-row KV) keep that error text.
- A Rust return value of `anyhow::Result<u64>` (`swiglu_q`, `norm_gate_q`, `norm_gate_sig`, `attn_decode_q`)
  is returned through an out-parameter `xq_out`, because the error channel is the `bool` + `err` pair above.
- `Buf`'s `Box<dyn Any + Send + Sync>` owner is a `std::shared_ptr<void>`; the device memory is freed by the
  deleter of its last copy. `Buf` and `GpuTensor` copies share ownership.
- The typed `zeros`/`put`/`get` of `impl dyn Backend` are free templates over trivially copyable `T` (Rust's
  `Plain` marker).
- `open(kind, device)` is not ported: `vulkan` is out of scope (`STRATA_CPP_PLAN.md` section 4) and the CUDA
  constructor is `CudaBackend::create`.
- `cuda_mmq_policy` (`cuda/mod.rs` `Gpu::mmq`) lives in `backend.cpp`: the type policy has no device input, and
  the trait default needs it without a dependency on the CUDA backend.

### `src/cuda_backend.h`, `src/cuda_backend.cpp`, `tools/embed_kernels.py` (port of `cuda/mod.rs`, part 1)

- **Driver API, not cudarc.** The primary context is retained and made current (`cuDevicePrimaryCtxRetain`, as
  cudarc's `CudaContext::new` does), streams are non-blocking, and every op binds the context to its thread.
- **Kernel text.** `tools/embed_kernels.py` writes `build/kernel_text.cpp`, the bytes of `kernels/tables.cuh`
  and `kernels/kernels.cu` as C arrays (`include_str!` in Rust; a string literal would hit the MSVC literal
  length limit). `source` = tables + the nine macro lines (`LLAMA_Q8`, `KV_Q8`, `GEMM_TC`, `GEMM_BM = 128`,
  `GEMM_BK`, `GEMM_MIN_BLOCKS`, `GEMV_LOADW`, `FA_BQ`, `FA_BK`) + the kernel body, as `source_with` writes them.
  The environment overrides are the same: `STRATA_LLAMA_NUMERICS`, `STRATA_KV_Q8`, `STRATA_GEMM_FMA`,
  `STRATA_GEMM_BK`, `STRATA_GEMM_MIN_BLOCKS`, `STRATA_GEMV_LOADW`, `STRATA_RPW`, `STRATA_RPW_MOE`,
  `STRATA_RPW_DENSE`, `STRATA_GEMV_BLOCK`, `STRATA_GEMV_BLOCK_MOE`, `STRATA_GEMV_BLOCK_DENSE`, `STRATA_MMQ`,
  `STRATA_CUDA_SYNC`, `STRATA_NVRTC`, `STRATA_NVRTC_OPTS`.
- **Compiled with no `-use_fast_math`.** StrataFreet's Python path adds it; Strata 1.2.5 does not, so the port
  does not. This is one of the reasons the StrataFreet comparison is by tolerance.
- **Cubin cache key.** FNV-1a over (source, arch, NVRTC tag) instead of Rust's `DefaultHasher`, which is not
  reproducible across builds. A cubin written by Strata is therefore not reused by strata-cpp (and the reverse).
  The directory is the same as `ptx()`'s: `XDG_CACHE_HOME`, else `HOME/.cache`, then `rusting_strata`. On Windows
  `HOME` is normally unset, so the directory is relative to the working directory, as it is in Strata.
- **Loaded kernels.** `kernel_names(sm80)` loads what `cuda/mod.rs` `kernel_names` loads: every entry of `KERNELS`
  (the DeltaNet, QSA and PLE entries too, as Rust does), the gemv families, the `gemm_*`/`gemms_*` families for the 24
  types, `gemmi_*`/`gemmis_*` on sm_80+ only, and `attn_decode_*`/`attn_decode_gqa*` for 64/128/256/512. Part 2a
  added the `KERNELS` entries and the GEMM and attention families to the loaded list.
- **`dyn_shmem_need` and the opt-in.** The function is ported in full. `gemvn_*` requests the device ceiling
  (`usize::MAX`, clamped to the opt-in minus the static part), as in Rust. The per-kernel carveout
  (`carveout_for`, `STRATA_SMEM_CARVEOUT`) is **not ported**: Strata only sets it below sm_80, and this build runs
  on sm_120. Part 2 adds it if a Volta run needs it.
- **Zeroed allocations on the main stream.** `alloc` zeroes with `cuMemsetD8Async` on the backend's stream. The
  first version used the legacy-stream `cuMemsetD8`, which is not ordered against the non-blocking streams and
  let the zeroing land after a kernel had written its output (found by `test_backend_gemv.py`, which returned
  zeros for about half the cases). Rust's `alloc_zeros` is stream-ordered; so is this.
- **Graph capture not ported.** `graph(key, f)` runs `f` directly, the path Strata takes under
  `STRATA_NO_GRAPH`. The ops read their per-call values from device memory, so the results are the same; only the
  launch overhead differs. Capture arrives with the decode graph in part 2b.
- **`beside` runs in order** (the trait default). Strata's CUDA override overlaps the copies on the side
  stream; that is a performance change for part 2b. `batch` does use the side stream, with event joins.
- **`moe_prefetch` is a no-op** (`moe_prefetch_supported` is false): the L2 hint is a knob, not a result.
- **Pinned staging ring** of 64 MiB, allocated at `create` as in Rust; copies of 256 KiB and more go through
  it, smaller ones from pageable memory.
- **Part 2b (declared, returns "not ported yet")**: `ssm_gates`, `conv_silu`, `deltanet`, `conv_seq`. Part 2a's
  ops moved out of this file (see `src/cuda_backend_ops.cpp` below). The QSA, hyper-connection and PLE methods, and
  the per-row KV kernels, keep their trait defaults (error text).
- **Helpers shared with `cuda_backend_ops.cpp`** are the `cuda_detail` namespace in `cuda_backend.h`: `kname`,
  `div_ceil`, `float_ty`, `attn_prefill_smem_bk` and `cu_error`, defined here as forwards to the anonymous-namespace
  versions, so each exists once.
- **Dev DLL links `cuda.lib`** now (it did not before), because `dev_api.cpp` drives the CUDA backend. The
  DLL still loads only on a machine with an NVIDIA driver.
- **Test-only surface** (`dev_api.cpp`): `scd_device`, `scd_gemv`, `scd_gemvn`, `scd_moe_gemv`. The backend is
  created on first use for device 0 and not freed at unload (the driver may be gone at process exit).

### `src/cuda_backend_ops.cpp` (port of `cuda/mod.rs`, W2-C part 2a)

Ported: `gemm`, `gemm_dense` (through `gemm_raw`, with `q8_x` for the int8 path and the SAME_X reuse), `rms_norm`,
`rms_norm_q`, `argmax_rows` (with `am_`, 64 partials per row), `l2_norm`, `rope_pairs`, `rms_norm_rope_kv` (with
`rope_fr`), `add`, `swiglu`, `norm_gate`, `kv_store`, `attn_decode_gq` and `attn_prefill_m` (through `attn`),
`router_topk`, `moe_route`, `moe_combine`, `moe_combine_seq`. Each keeps the grid, block, dynamic shared memory and
argument order of its Rust wrapper; every integer argument is passed as the 32-bit `int` the kernel declares, and the
`launch` template takes it from there.

Also overridden, because the Rust `Gpu` overrides them with the same launches (the trait defaults would call a
different kernel or none): `rms_norm_streams`, `rms_norm_rope`, `rms_norm_rope_qk`, `prequant_gemm_x`, `swiglu_q`,
`norm_gate_q`, `norm_gate_sig`, `attn_decode_q`. The tests exercise `rms_norm_rope`, `swiglu_q`, `norm_gate_q` and
`norm_gate_sig`; `rms_norm_streams`, `rms_norm_rope_qk`, `prequant_gemm_x` and `attn_decode_q` compile and share the
launches above but have no dedicated check yet.

- **Errors instead of panics.** The Rust `ensure!`/`bail!` checks (head dims, `n <= 512`, `k <= 32`, the GEMM column
  multiple, the q8 layout) return `false` with the same text. `attn` also refuses a `nh` that is not a multiple of
  `nkv` (Rust divides by zero there).
- **Out-parameters for `Result<u64>`.** `swiglu_q`, `norm_gate_q`, `norm_gate_sig` and `attn_decode_q` return the xq
  they used through `xq_out`, as the header states.
- **`rope_fr` launches on the main stream and waits.** Rust launches `rope_freq` through `self.stream` directly and
  synchronizes, not through `st()`; the port does the same (the side stream would make the wait miss it).
- **The q8 copy and the argmax partials are backend members.** `q8_` (with `q8_key_`) and `am_` are allocated with
  `alloc` (zeroed, stream-ordered) and freed in the destructor before the context is released. `q8_mu_` and `fr_mu_`
  replace Rust's `Mutex` around `q8` and `fr`; `gemm_raw` holds `q8_mu_` for its whole call, as Rust does.
- **Turing shared memory.** `turing_` (sm_75 to sm_79) selects 16 keys a tile for `attn_prefill_256`, both at load
  (`dyn_shmem_need` opt-in) and in `attn_prefill_m`, as `Gpu::new` and `attn_prefill_m` do.
- **`BESIDE`** is repeated as `kBeside` in this file (the constant is private to `cuda_backend.cpp`); the value is the
  same.
- **Not ported:** the per-row KV kernels (`rms_norm_rope_qk_rows`, `attn_rows`, `kv_rows` stays false), so
  `GpuModel::step_batch` has no CUDA path yet; the QSA launches (`qsa_pool`, `qsa_select`) and `moe_prefetch`. The
  `sel` argument of `attn_decode_q` is passed through to the kernel, which reads QSA cell lists when it is set.
- **Dev ABI** (`dev_api.cpp`, `scd_*`): each call uploads its host arrays at their full size (rows x stride), runs one
  op through `CudaBackend`, syncs and copies the outputs back. Strides are in elements. Mailbox calls copy the
  caller's words into the pinned mailbox, run the op, and copy the words back, so the test can play the CPU side.

### `src/cpu_worker.cpp`, `src/cpu_worker.h` (port of `cpu_worker.rs`, W2-D worker)

Ported: `REQ`/`NTOK`/`SKIP`/`VERDICT`/`DONE`/`IDS`, `TMAX`, `Mailbox::new` (the same stride formula, with word
offsets), `CpuWorker::new`/`begin`/`routing`/`stalled_layer`/`wait_done`/`busy`, Drop, the thread loop with the
`ahead` skip rule, the nt > 1 path through `experts_multi`, the nt = 1 path through `experts`, the `Prefetch` hook
(`weights.cpp`'s `Prefetch`, which fits: `start()` then `send()` of (address, bytes) ranges for the missed experts of
`gate`, `up`, `down`, in the Rust order), `read_wc` (SSE4.1 `_mm_stream_load_si128`, or a copy without SSE4.1), and
`STRATA_TEST_WORKER_DELAY`.

- **Memory ordering (decided).** The flags and the mailbox words are read and written with `volatile` accesses, and
  every fence is `std::atomic_thread_fence(std::memory_order_seq_cst)`, as Rust does with `read_volatile`,
  `write_volatile` and `fence(SeqCst)`. No `std::atomic` is used for the mailbox words: the memory is host-mapped
  pinned memory (and on Vulkan write-combining BAR memory), where atomic operations on a `uint32_t` are not what the
  GPU side expects; the fence provides the ordering the GPU's `fence(Release)` pairs with. MSVC's default
  `/volatile:ms` on x64 also gives the volatile accesses acquire and release ordering, which the Rust code relies on
  as well. Shared atomics are used only for the stop flag, the busy counter and the thread-exit flag, which are not
  in the mailbox.
- **The mailbox is a raw host pointer.** `create` takes `uint32_t* host`, its byte length and its device address
  (`dev`, 0 when none) from the caller. In production the caller is `CudaBackend::host_mailbox` (W3-E2 wires it in
  `GpuModel::load`); the test ABI allocates a 64-byte aligned buffer. `create` checks the length and the 64-byte alignment.
- **The channel is a queue.** Rust's `mpsc::channel` is a `std::deque<uint32_t>` under a mutex with a condition
  variable; `closed_` plays the dropped sender (`rx.recv()` returns `Err` when the queue is empty and closed).
- **The loop thread takes part in the passes (W3-E2).** Rust runs the loop inside `pool.install`, so the loop thread
  works in the rayon passes with the pool. `ThreadPool::for_each` now runs items on its caller too (the loop thread
  in the worker, the test thread in `test_cpu_moe`). `CpuWorker::create` gives the pool `threads - 1` threads, so
  `threads` threads compute, as rayon's `threads` workers do under `pool.install`. The pool itself changed from a
  mutex/condvar hand-out to spinning workers (generation counter, 200000 polls, then a condition variable), with
  atomic claim and finish counters; the test of `experts` at 4/6/7/8 threads went from 92 us to 55-57 us per expert.
- **Expert pool size.** The pool has `threads` workers (`default_threads()` when 0). Rust uses
  `rayon::current_num_threads()`, the global pool, which `--threads` sets; the product will pass the same value.
- **Layers resolved at creation.** `load_expert_layer` runs for each worker layer in `create`, so a missing expert
  tensor fails there with a message. Rust looks the tensors up on each request (`m.w.layer(il, s).ok()`), which
  silently skips a missing one in the prefetch and panics in the expert pass.
- **Errors instead of panics.** An error in the expert pass (which panics in Rust, killing the thread) stops the
  loop, records the message, and makes `wait_done` return `false` with "the CPU expert worker stopped: <message>",
  so the GPU side sees an error instead of waiting forever.
- **`wait_done` takes a timeout** (`timeout_ms`, negative = wait as long as Rust does) and returns `bool` + an error
  string instead of `anyhow::Result`. The sleep between polls is 50 us, as in Rust.
- **`read_wc` needs `n_embd % 4 == 0`.** Rust copies whole 4-word chunks and would leave a tail of an activation
  unread; `create` refuses a model whose `n_embd` is not a multiple of 4 (every Qwen and Flash-Next width is).
- **`routing` and `wait_done` check the layer** (`has_layer`): Rust indexes the mailbox with any `il`. An unknown
  layer gives an empty routing or an error, not a read outside the buffer.
- **`busy()` is nanoseconds** (`busy_ns()`), summed over the layers the worker computed, as in Rust.
- **Test ABI** (`cpu_test_api.cpp`, `sctw_*`): `sctw_create` allocates the test buffer (64-byte aligned, zeroed) and
  the worker over it; `sctw_layout` returns the offsets and the buffer address so the Python test can play the GPU;
  `sctw_begin`, `sctw_routing`, `sctw_stalled_layer`, `sctw_wait_done`, `sctw_busy_ns`, `sctw_stop` and `sctw_free`
  are the Rust methods. `sctw_free` drops the worker before freeing the buffer.

### `src/cuda_backend_seq.cpp`, `tests/test_backend_seq.py` (port of `cuda/mod.rs`, W2-C part 2b)

Launch wrappers ported with the grid, block, shared memory and argument order of `cuda/mod.rs`:
`ssm_gates`, `conv_silu`, `conv_seq` (+ `conv_state`), `deltanet` (`deltanet_seq`), `qsa_pool`, `qsa_select` (+
`qsa_score`), `hc_silu`, `hc_mix`, `hc_write`, `ple_gate`, `ple_conv`, `rms_norm_rope_qk_rows`, `attn_rows`
(through the existing `attn`), `moe_prefetch`, `graph`, `beside`, `beside_join`, and `kv_rows` / `beside_overlaps` /
`moe_prefetch_supported` (true, as in Rust).

- **`graph` uses the driver API directly.** `cudarc`'s `begin_capture` / `end_capture` / `CudaGraph::launch` become
  `cuStreamBeginCapture` (THREAD_LOCAL mode, as Rust), `cuStreamEndCapture`, `cuGraphInstantiateWithFlags` with
  `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`, and `cuGraphLaunch`. Graphs are kept by key in a map (32 at most,
  `GRAPHS_MAX`), freed in the destructor. The `STRATA_CUDA_SYNC` / `STRATA_NO_GRAPH` switches and the cap run `f`
  directly, as Rust does.
- **A rope table the capture lacked** is made after the capture and the graph is not kept, as in Rust. The
  capture check is `cuStreamIsCapturing` in `rope_fr` (`cuda_backend_ops.cpp`), which now returns 0 and records the
  table in `fr_want_` during a capture (Rust `rope_fr`: `fr.1.push`). Before this, `rope_fr` always waited on an
  event, which is illegal inside a capture.
- **`beside` / `beside_join`** keep Rust's `lag` event (`lag_`): the side stream waits for the main stream, `f` runs
  with `nb` at `BESIDE` (launches stay on the main stream, copies go to the side stream, as `htod` already does), the
  main stream waits for the previous beside's event, and the new event is kept. `beside_join` drops the event and
  joins the main stream to the side stream.
- **`rms_norm_rope_qk_rows` / `attn_rows`** take `u64` device addresses of the per-token caches (`kvt`), as Rust does.
  The test ABI builds `kvt` on the host (two caches per token, one allocation each).
- **Test ABI** (`dev_api.cpp`, `scd_*`, 16 entry points plus `scd_graph_calls`): every entry uploads its host arrays,
  runs one op through the shared backend, and copies the in/out arrays back. `scd_graph_calls` counts the runs of the
  op function of the last `scd_graph_run`, so the test can show that a capture happened (1 capture + one eager run per
  input = `nrep + 1`) and that replays do not run it. `scd_graph_run` and `scd_beside_copy` are the replay and
  join checks; `scd_moe_prefetch` checks that the hint leaves the bytes unchanged. These are test helpers, not part of
  the product ABI in `capi.cpp`.
- **Not exercised:** DeltaNet head dims 32, 64 and 256 (the test models use 128); QSA tie-breaking (the test scores
  have no ties, so the lower-block rule is only read from the kernel); `moe_prefetch`'s effect (a hint, not checkable
  from the host); the captured decode forward (`gpu_model.rs` is part 3, so graph capture is checked on the test ops).

### W3-E1a: `model.cpp`, `gpu_model.cpp`, `expert_cache.cpp`, `sampling.cpp`, `capi.cpp`, `tools/sc_score.cpp` (port of `model.rs`, `gpu_model.rs` decode part, `expert_cache.rs` fully-resident subset, `ops.rs` sampler)

Author of the port: Vasyl Trefilov's RustingStrata 1.2.5, ported line by line. Scope: the qwen35moe decode path (DeltaNet and full-attention layers, MoE with every routed expert resident, shared expert), one part, one slot, one token per forward.

- **Batched prefill is not ported.** Prompts go token by token through `step`, which is what the Rust does for prompts under `short` (32 tokens, the default with no calibration file). `prefill.rs` is W3-E3.
- **Expert cache: fully resident only.** `expert_cache.cpp` allocates the pools and fills the tables in layer order (the profile read is skipped; the order only sets slot numbers, not results). Partial residency errors out with "needs the CPU expert worker (W3-E2)". `record`/`settle`/`adapt`/`swap`/`save_profile` are W3-E2.
- **No CPU worker is started.** With every expert resident the Rust fast path uses `mb = 0`, so the worker is never asked; the mailbox layout is taken from `Mailbox::make(k, n)` (the same numbers `CpuWorker::new` computes). `threads` is accepted and reported in `sc_stats`, not used. `stalled()` is checked after the graph and refuses.
- **Batched rows, MTP, hyper-connections, PLE, QSA, LoRA, image tokens, eagle3: refused with an error naming the package** (W3-E1b, W3-E2, or out of scope). `step_rows`/`verify` (slots, speculative verify) are not ported.
- **`rope_offset` and M-RoPE image positions are always 0 here** (text only). The non-fused RoPE branch of `layer()` is taken only when a layer has `rope_layer` false (none in the 14B).
- **`batch(...)`** is called where the Rust calls it (`CudaBackend::batch` joins its side stream around the closure).
- **`GpuTensor` gained a default constructor** (`backend.h`) so layer structs can hold tensors by value. No other change to the existing backend.
- **`model.cpp`** ports `Model::load` (config, validate, nextn drop) and `Model::embed`. The scalar CPU forward (`model.rs` `step_h`) is not ported: the GPU path is checked directly against `rustingstrata score --gpu`.
- **Sampler** (`sampling.cpp`): greedy/temp/top-k/top-p with the same xorshift64 and candidate cut as `ops.rs`. `sc_decode_sample` does the argmax on the host from the logits.
- **Upfront tensor-type gate of `load_part`** (every tensor must have a CUDA kernel) is not ported; an unsupported type fails at its first launch.
- **C ABI** (`capi.cpp`): `sc_load`, `sc_n_vocab`, `sc_decode`, `sc_decode_sample`, `sc_reset` (keep_tokens must be 0), `sc_stats`, `sc_free`; `kv` must be `f16`. `sc_score.exe` (`tools/sc_score.cpp`) is the `score` twin: same input and output keys, fed through `sc_decode`.

Measured on `Qwen3.6-14B-A3B-FableVibes-Q6_K.gguf` (see `tests/test_score_14b.py`): argmax identical on 192/192 positions over 3 prompts x 64 continuation positions; max |top-1 logprob diff| 0.000000. Decode, 128 greedy tokens after a 16-token prompt: 149.8 tok/s from `sc_decode_sample`, 209.2 tok/s from `rustingstrata bench --pp 16 --tg 128 --reps 3 --ctx 4096` (prefill 194.2 tok/s). The gap is the host argmax and logits copy per token and the non-concurrent launches; not optimized in this package.

### W3-E2: `expert_cache.cpp/.h`, `gpu_model.cpp/.h`, `cpu_moe.cpp/.h`, `cpu_worker.cpp/.h`, `capi.cpp/.h`, `tools/sc_score.cpp` (port of `expert_cache.rs`, the CPU-worker wiring of `gpu_model.rs` decode, `cpu_moe.rs` pool)

- **Expert cache: partial residency.** `ExpertCache::create` takes the budget (`free - vram_margin`, capped by
  `STRATA_EXPERT_MB`, as `load_part`), allocates pools for it (a failed allocation frees them and tries 0.9x, as in
  Rust), and fills the slots hottest first from the routing profile (ties in layer order). Experts without a slot have
  table entry -1 and run on the CPU worker. `record`, `record_prompt`, `settle`, `reset_counts`, `adapt` (KEEP 0.95,
  LONG_KEEP 0.999), `swap` (max_swaps per pool, the 1.5x + 2 rule, `smallest` as stable order) and `save_profile`
  with `profile_update` are ported as written. `STRATA_FIXED_EXPERTS` turns the profile off. `save_profile` writes
  only after a recorded run and keeps other layers of the file.
- **Pool cap.** Rust's `MAX_POOL_FULL_BYTES` is 8 GiB (`expert_cache.rs`), so that is the value here. The plan text
  said 2 GB; the source is the authority, and the difference only changes how experts are cut into pools.
- **Profile path.** `profile-<arch>-<L>x<E>-<tensors>.f32` under `%LOCALAPPDATA%
usting_strata` (`cache_dir`, the
  same order Rust uses: XDG_CACHE_HOME, LOCALAPPDATA, HOME/.cache). The tensor count is `Gguf::tensors().size()`.
  Format: little-endian f32, `n_layer * n_expert` counts or more (other devices' layers).
- **Calibration file.** `sc_load` reads `strata-<stem>.json` (`chunk`, `mtp`, `short`) and reports them in
  `sc_stats`. The prefill chunk and MTP that use them are W3-E3; nothing is applied yet.
- **Decode wiring (`gpu_model.cpp`).** The MoE layer takes `mb = 0` on the fully resident fast path and
  `mb = worker.dev + 4 * il * stride` otherwise (as `forward`/`layer` in Rust). `begin` gives the sequence the step
  publishes in `step_buf[1]`; `after_experts` reads `routing(il)` per layer (hits and misses by slot, `record`), and
  calls `adapt` when the position crosses an `adapt_every` multiple (default 8, `STRATA_ADAPT_EVERY`; swaps
  `STRATA_ADAPT_SWAPS`, default 64) and never on the fast path. `finish` waits on `stalled()` and resumes the forward
  from the combine (`forward(from)`, `layer(resume)`), as Rust does. CUDA never reports a stall (Rust's CUDA backend
  does not override `stalled` either), so this recovery is ported but never runs on CUDA; it has no test.
- **Host mailbox.** `GpuModel::load` allocates the mailbox with `CudaBackend::host_mailbox` for the MoE layers and
  builds the `CpuWorker` over it, with `threads` (0: `default_threads()`, the physical cores, as Strata's GPU default).
- **Device argmax.** `step_greedy` runs `argmax_rows` on the logits and copies one int; `sc_decode_sample` with
  `{"temp":0}` uses it. Sampling with `temp > 0` still copies the logits and samples on the host (`ops.rs` Sampler).
  The C ABI `sc_decode_sample(h, tok, sampler_json, int* out)` replaces the earlier argument list; `test_score_14b.py`
  is updated to it.
- **Pool dispatch.** `ThreadPool` spins before it sleeps and runs items on its caller too (see the `cpu_worker`
  deviation above). Standalone expert time at 7 threads went from 92 to 55 us. The decode gap below is not explained
  by the pool: a 200000-poll spin and a never-sleeping spin measured the same worker time.
- **Placement count versus Strata.** At `--vram-margin 8500` ours loads 1892/3600 experts (4656 MB of slots); Strata
  prints `resident experts 1715/3600` (4221 MB of slots). Strata's plan line reserves 568 MB of prefill scratch before
  it sizes the slots; the prefill is W3-E3, so that reservation is not made here. The residual (~435 MB) is not
  explained yet. The profile round trip therefore compares ours with Strata from the same file, not the first-load
  counts.
- **Results (2026-11-20, RTX 5070 Ti, 14B Q6_K, `--threads 7`).** Fully resident (`test_score_14b.py`): argmax 192/192,
  max top-1 logprob diff 0.000000, decode 199.5 tok/s against Strata's 207.7 (96%, gate 95% met). Partial at
  `--vram-margin 8500` (`test_partial_14b.py`): parity argmax 188/192 = 97.92% (gate 97% met; positions 25 and 1, 5, 19
  differ), max top-1 logprob diff 0.69 (reported, no gate); decode 56.8 tok/s against Strata's 101.3 (56%, gate 95%
  NOT met); worker busy 15.8 ms per token (Strata 6.3 ms, 1892/3600 experts resident here, 85.3% hit in Strata and
  92.7% here); profile written in Strata's format (14400 bytes = 3600 f32); initial placement from that profile
  1892 (ours) against 1715 (Strata): NOT matched. The worker time per miss in decode is about 10 times the standalone
  per-expert time of `test_cpu_moe` (55 us); the cause is not found (the spin pool and page pre-touch did not change it).
- **Decode gap: root cause and fix (W3-E2 speed gate, 2026-11-20).** The product DLL (`build.ninja`) was compiled without
  `/arch:AVX2`, so `cpu_moe.cpp` took its scalar integer kernels (`SC_AVX2` 0; `int_dot` returns `scalar_dot`). The
  standalone test DLL (`build_cpu.ninja`, `/arch:AVX2`) ran the AVX2 kernels, so it measured 55 us per expert while the
  decode worker took about 650 us. Rust is built with `target-cpu=native`, which enables AVX2 on this machine, so the
  product build now uses `/arch:AVX2` as well (`build.ninja`, `cflags`). This is a build-flag deviation from the Rust
  build settings, not an algorithm change. Measured per pool job: 22.2 us per 16-row item before, 8.5 us after; worker
  busy 15.8 -> 6.4 ms per token.
- **Ruled out by measurement (`STRATA_TRACE_WORKER`).** Page faults: first touches only (about 390 per miss, zero on
  re-touched experts), so not the steady state. Pool wake-up: pool threads start a job in 0.3 us (they spin). Host sync
  spin: one background spinning thread changed the standalone time by about 8%. Gaps between calls (up to 3 ms): no
  effect. Cold memory (distinct experts): 123 us per expert standalone, the same order as Strata's 120 us.
- **`STRATA_TRACE_WORKER` (opt-in, measurement only, not in Rust).** Unset or `0`: the cost is a few predictable
  branches per layer and per pool job. Set to `1` (writes `strata_worker_trace.csv` in the working directory) or to a
  path: per layer and sequence, steady-clock timestamps (QueryPerformanceCounter on MSVC) for the request seen, the
  mailbox read, the experts() passes (gate/up, silu, down, accumulate: `ExpertsTrace` in `cpu_moe.h`), the done flag,
  page faults and the thread's user/kernel time, plus the worker's wait before the request. `<path>.pool.txt` gets the
  pool's start latency, busy time per item and the caller's wait per `for_each`. The GPU-side combine wait is not
  measured (the kernel is verbatim from Strata and has no counter).
- **Results after the fix (2026-11-20, RTX 5070 Ti, 14B Q6_K, `--threads 7`, margin 8500).** Partial
  (`test_partial_14b.py`): parity argmax 189/192 = 98.44% (gate 97% met; prompt 1 64/64, prompt 2 62/64, prompt 3 63/64),
  max top-1 logprob diff 0.69 (reported). Decode 120.9 tok/s against the official 102.8 (117.6%, gate 95% met; the
  official bench mean in the same run). Worker busy 4.97 ms per token in the timed run (official 6.1 ms in its bench).
  The test still exits FAIL on the placement round trip: the initial placement from the shared profile is 1892 (ours)
  against 1715 (official) (see the placement bullet above, not changed here). Earlier numbers in the results bullet
  below are before the fix.
- **Not ported in W3-E2:** `prefetch_next` (`STRATA_MOE_PREFETCH`, off by default in Rust), the `beside` overlap of
  the step's host copies (the step has none beyond the embedding put), `reset_counts_after_prefill` and `settle`
  (called by the prefill, W3-E3), and the MTP worker.

### W3-E3: `src/prefill.cpp` (new), `src/gpu_model.h/.cpp`, `src/capi.h/.cpp`, `src/cuda_backend.h/.cpp`, `src/backend.h`, `tools/sc_bench.cpp` (new), `tools/sc_score.cpp` (port of `gpu_model/prefill.rs`, the plan of `gpu_model.rs` load_part, `main.rs` bench and score)

- **Ported.** `prefill_sizes` / `make_prefill` (`Scratch::new`: slab layout, ints, tiles, staging), `prefill` (`prefill_with`
  for one part: short prompts token by token through `step`, chunks of `chunk` tokens, `STRATA_PF_ADAPT`, `adapt_experts`
  (settle), `STRATA_RESET_COUNTS`, last hidden state in x row 0 and the logits), `prefill_chunk` (per layer: DeltaNet
  with `conv_seq`, the l2 norms, `ssm_gates`, `deltanet`, `norm_gate`; full attention with `rope_neox`, `kv_store`,
  `attn_prefill_m`; dense FFN; the MoE layers in `prefill_moe`: router, shared expert, `record_prompt`, the expert sort
  (xrow/inv), the GEMM tile list per step, the staged copies of missed experts beside the GEMMs of the previous step,
  `Prefetch` of the mapped weights, `moe_combine_seq`). The chunk defaults to 4096 (min ctx) and `short` to 32, as
  main.rs (`get("chunk").unwrap_or(4096)`, `get("short").unwrap_or(32)`); sc_load takes the options first, then the
  calibration file, then the defaults.
- **Not ported (refused or absent as before).** The MTP draft fill inside the prefill, the hyper-connection, PLE and
  QSA branches (W3-E1b), `prefill_cached` / `prefill_slots` (serve), `calibrate`, LoRA.
- **Load plan (deviation: exact device accounting).** The driver rounds each `cuMemAlloc` up to 2 MB (measured:
  1000-byte allocations take 4 KB, 5 MB ones 6 MB), and the load makes ~700 of them, so the first version used 2438 MB
  for "weights+state+scratch" against Strata's 2272 MB and resident 1660 against 1715. `load` now computes the exact
  device total of its allocations (`plan_bytes`, the same size tables as the allocations: `scratch_sizes`,
  `prefill_sizes`), reserves it as one block (`CudaBackend::reserve_arena`, the default of `Backend` does nothing) and
  `alloc` carves 256-byte-aligned buffers from it. The expert slots are allocated after the arena (direct, as before).
  Result: 2274 MB (2 MB rounding of the block), KV 80 MB and prefill 568 MB exact. `load` prints Strata's `plan:` and
  `expert traffic:` lines to stderr. `STRATA_LOAD_TRACE=1` prints the VRAM used after each load stage.
- **Speed (deviation: context bind).** `CudaBackend::bind` called `cuCtxSetCurrent` before every backend op (about 5 us
  each; the nsys profile showed ~19k calls). It now reads `cuCtxGetCurrent` and sets the context only when another
  one is current. Before this the prefill ran at 85% of Strata's; after it at 103%.
- **`sc_bench` input (deviation: no tokenizer in C++).** `bench` tokenizes the prose text for each pp. `sc_bench`
  reads the same ids from a JSON file (`--prompts`, `{"eos": [...], "prompts": {"512": [...]}}`) that
  `tests/test_prefill_14b.py` writes with StrataFreet's tokenizer (`_eog` gives the end-of-generation ids). The
  decode loop is the greedy one of `main.rs` `decode` (one step per token, EOS checked before the step). Strata
  detokenizes each token inside its decode timing; `sc_bench` does not, so its tg number is a little optimistic.
- **`sc_score`.** The prompt goes through `sc_prefill_logits` (the batched prefill; prompts under 32 tokens go token by
  token inside it), the continuation through `sc_decode` as before. C ABI added: `sc_prefill`, `sc_prefill_logits`,
  `sc_prefill_sample` (capi.h). `sc_stats` gains `expert_bytes_per_token`, `prefill_chunk`, `prefill_short`,
  `prefill_mb`, `n_expert`, `n_expert_used`.
- **Measured (2026-11-20, RTX 5070 Ti, 14B Q6_K, ctx 4096).**
  - `test_prefill_14b.py` parity (fully resident, prompts of 64, 300, 1000 tokens, 64 continuation positions each): argmax
    192/192 = 100%, max top-1 logprob diff 0.000000 (gate 99% / 0.05 met).
  - Fully resident bench, pp512 tg128 reps 3 (ours / official): pp512 5482.2 / 5325.5 tok/s = 102.9% (gate 95% met);
    tg128 199.2 / 205.3 tok/s = 97.0% (met); resident 3600/3600 both (met).
  - Margin 8500 and 7 threads: pp512 631.4 / 695.6 tok/s = 90.8% (gate not met; the run-to-run spread of both binaries is
    large, 330 to 880 tok/s, from the staged copies of the missed experts); tg128 149.7 / 152.2 = 98.4% (met);
    resident 1726 / 1715 (NOT met).
  - `test_score_14b.py`: argmax 192/192, max top-1 logprob diff 0.000000; decode 203.1 tok/s (sc_decode_sample).
  - `test_partial_14b.py`: parity 187/192 = 97.4% (gate 97% met; prompt 1 63/64, prompt 2 60/64, prompt 3 64/64).
    Decode gate NOT met: ours 83.7 vs official 95.6 tok/s (87.5%), with no prompt before the timed decode (this test
    decodes from the load placement; the bench after a 512-token prefill decodes at 98%). The placement round trip
    is NOT met (1726 vs 1715).
- **Open: the resident count is 11 experts (about 27 MB of free VRAM) above Strata's.** The load's own accounting now
  matches Strata's to 2 MB (2274 against 2272), and the expert slot size is identical (2.4614 MB). The difference is
  therefore in the free VRAM that Strata's process has before its load starts (its `free1` is about 27 MB lower).
  Eager module loading (`CUDA_MODULE_LOADING=EAGER`) did not change it. Not resolved; the source of Strata's extra
  baseline was not identified.

### W3-E3b: `src/prefill.cpp`, `src/cuda_backend.cpp/.h`, `src/backend.h`, `src/weights.cpp` (prefill speed: measurement, opt-in trace, Windows readahead)

- **Goal and result (not met).** The gate was pp512 >= 95% of the official binary on Flash-Next and on the 14B at margin 8500.
  Neither gate is met after this package. Decode is not worse: 14B tg128 152.5 vs 153.8 official (99%), Flash-Next tg128
  42.4 vs 37.1 (114%).
- **Opt-in trace (new, zero cost off).** `STRATA_TRACE_PREFILL=1` prints, per `prefill` call: the chunk wall time, host
  time per phase (embed, ple, hc, mixer, router, ids sync, build lists, beside enqueue, of which copy, join, dense), the
  staged bytes and missed experts, and the pinned-ring counters (memcpy, waits, DMA enqueue). `=2` also drains the device
  at the end of the fenced phases (serialized device time per phase; not run yet). Off: one cached flag test per phase.
  `htod_trace_take` (backend.h) reads the ring counters; they are kept in `cuda_backend.cpp` under the same flag.
- **Phase breakdown (per 512-token prefill, steady rep, STRATA_TRACE_PREFILL=1):**

  | phase | 14B, margin 8500 | Flash-Next |
  |---|---|---|
  | chunk wall (host, device drained at the end) | 633-665 ms | 4190 ms |
  | ids sync (the dtoh of the routing, per layer) | 120-140 ms (3.5 ms per layer) | 390 ms (8 ms per layer) |
  | beside enqueue (the copies and the GEMMs of the routed experts) | 500 ms | 3770 ms |
  | of which copy (memcpy into the pinned ring) | 480-490 ms | 3600 ms |
  | ring waits + DMA enqueue | 6 + 7 ms | 37 + 50 ms |
  | staged copies (missed experts) | 4.2 GB, 1628 experts | 26.1 GB, 11421 experts |
  | embed, ple, hc, mixer, router, dense, build, join | < 20 ms together | < 40 ms together |

  The first prefill of a run is far slower (14B 1.3-1.4 s of memcpy; Flash-Next 9.1 s): the mapped expert pages are read
  from disk or first touched. The official binary shows the same first rep (Flash-Next 55 tok/s in its 3 reps).
- **Causes.** (1) The copy of the missed experts dominates: Flash-Next keeps 16% of its experts resident, so one 512-token
  prefill copies 26 GB; the 14B copies 4.2 GB. Every copy is a host memcpy from the mapped GGUF into the 64 MB staging
  ring, and it runs at 7.2 GB/s on Flash-Next and 8.6 GB/s on the 14B. (2) The schedule matches Strata's (same ring, same
  steps, same tile lists, same order; the host blocks on the routing of each layer, as `prefill.rs` does). The fully
  resident 14B prefill is 103% of the official, so the GEMMs are not the gap. (3) The routing sync is a dependency chain
  (the last routed GEMMs, the combine and the next attention), not overlappable with the copies.
- **Probes (scratch programs outside the port, not kept).** A plain DRAM copy on this machine is 12.5 GB/s with one or
  four threads (so the memory bus, not the copy thread, is the limit). A copy from the mapped GGUF into a registered heap
  ring is 12.7 GB/s in isolation. CUDA cannot register the file mapping (`cuMemHostRegister` returns 801 for read-only,
  2 for read-write), so a DMA straight from the page cache is not available. The first touch of a mapped page runs at
  about 3 GB/s.
- **Tried and reverted (no gain in the engine).** (a) Copying each ring chunk with four threads (the ThreadPool of
  `cpu_moe`, the caller included): memcpy 480 to 495 ms against 470 to 490 ms with one thread. (b) The staging ring as
  registered heap memory (`malloc` + `cuMemHostRegister`) instead of `cuMemHostAlloc`: memcpy 474 to 492 ms, the same
  as the pinned ring. Both are removed; the ring is `cuMemHostAlloc` as before.
- **Kept (deviation: Windows readahead).** `will_need` (weights.cpp) calls `PrefetchVirtualMemory` on the range
  (page-aligned start), the Windows counterpart of `madvise(WILLNEED)`. Rust's `madvise` shim is empty off Linux, so
  Strata's Windows build reads nothing ahead and the Prefetch thread (started per prefill chunk, one request per layer)
  does nothing. The call succeeds on a 1 GB view (62 ms). Effect on the first 14B rep: memcpy 1329 ms against 1373 to
  1417 ms before (noise-level). Steady state unchanged.
- **Measured (2026-11-20, RTX 5070 Ti, same process and margin as before; RAM at each Flash-Next load 91326-91328 MB).**
  - 14B, margin 8500, 7 threads, pp512 tg128 reps 3, ctx 4096 (untraced, one pair): ours 643.5 (328.6-803.7) tok/s,
    official 694.5 (332.8-894.0); pp512 92.7% of the official mean (best rep 90%). Gate 95% NOT met. tg128 152.5 vs
    153.8 (99%). Resident 1726 (ours) vs 1715 (official).
  - Flash-Next, pp512 tg128 reps 3, ctx 8192, untraced (RAM 91330 MB at the load): ours 98.4 (53.4-121.1) tok/s, decode
    42.4 (39.6-44.3); official 124.7 (55.4-159.4), decode 37.1 (31.7-42.3). pp512 79% of official (gate 95% NOT met);
    decode 114% (met). Steady state ours about 121 against official about 159. Resident 4009 (ours) vs 3996 (official):
    13 more experts resident in ours. A traced run (STRATA_TRACE_PREFILL=1) gave 98.4 (52.3-121.6) and decode 31.6.
- **Open.** The steady-state gap on Flash-Next (about 25%) and on the 14B (about 8%) is not explained by the copy rate
  alone: official moves the same 26 GB in about 3.2 s, which needs about 8 GB/s with the GEMMs and routing hidden,
  while our memcpy alone takes 3.6 s. The next steps would be a copy-volume comparison per layer against the official
  (its own trace is not available) and a `STRATA_TRACE_PREFILL=2` run to see the device time of each phase.

### W3-E1b: `model.cpp/.h`, `gpu_model.cpp/.h`, `prefill.cpp`, `tests/test_flashnext.py` (new) (port of `model.rs` PLE geometry, `ple_rows`, `ple_embed`; `gpu_model.rs` `hc_read`, `ple`, the hyper-connection, PLE and QSA branches of `layer`, `forward`, `put_embed`, `reset` and the load plan; `gpu_model/prefill.rs` `prefill_chunk` with the same branches)

- **Ported.**
  - `Model::load`: the PLE metadata (`{arch}.ple.*`: ngram_size, heads_per_ngram, conv_kernel, layer_multipliers, head_offsets,
    head_vocab_sizes, eos_token_id) with the geometry checks of model.rs. `per_layer_token_embd.weight` stays a host mapping
    (cpu_only), read a token at a time.
  - `Model::ple_rows` and `Model::ple_embed`: the 2- and 3-gram hashes (XOR of token times multiplier, mod each head's vocab, plus
    the offset; EOS or the sequence start cut the context), 16 table rows per token, head after head.
  - `GpuModel::load` and `plan_bytes`: the hyper-connection reads (`blk.N.hc_attn_*`, `hc_ffn_*`, `output_hc_*`, inject when the file
    has one), no `attn_norm`/`post_attention_norm` with hyper-connections (as Rust), the PLE blocks (`ple_conv1d` as f32 on the
    device), the QSA indexers of the attention layers (BF16 projections, the two norms), the pooled keys (ctx / 4 blocks of 128
    floats), the raw indexer key ring (`n_layer * QSA_RING` floats), the PLE ring (16 rows of hc_n * n_embd), the QSA score, sel
    and mask buffers, and the `hc_n`-wide residual stack `x` and its scratch rows (`scratch_sizes`). Every allocation is counted.
  - decode: `run_token` (the embedding repeated per stream, the PLE input into `emb`, `ple_prev` advanced), `layer` (PLE block
    first; the mixer input from `hc_read` or the norm; `norm_gate_sig` for the DeltaNet output gate; the attention with the
    QSA indexer: `gemvq2` of the query and key projections, `qsa_pool`, the query norm and RoPE, `qsa_select`, and the selected
    cells to `attn_decode_q`; `hc_write` after the mixer and after the FFN (cleared); the FFN's combine into `mo`), `forward`
    (output `hc_read`), `reset` (`ple_prev` back to -1). `hc_read`, `ple_block`, `gnorm`, `hc_at` are the members of the same names.
  - prefill: `prefill_sizes` with the hyper-connection regions (xn, lo, hg, pk, pq, pg, pn, pv after the mixer and FFN regions,
    inj, mix, mo, emb after them) and the QSA region in `c`; `prefill_chunk` with the PLE input put into the `emb` region, the
    hyper-connection reads as batched GEMMs (`HcAt.gemm`), the mixer output into `mix` (act 0) and the streams through
    `hc_write`, the QSA GEMMs and pool, `qsa_select` by rows of `QSA_ROWS` into the mask when the visible blocks exceed `top`
    (no mask before), `norm_gate_sig`, and `attn_prefill_m` with the mask; `prefill_moe` takes the combine target `fx` (`mo`
    with hyper-connections); the last stack goes to row 0 of `x`, and the prefill's output head is `hc_read`.
  - The load refusals for hyper-connections, PLE and QSA are gone. The dense refusal stays (hyper-connections need MoE in Rust
    too).
- **Deviations.**
  - No `save_state` / `rollback` copies (`ple_undo`, the `sr` ring and QSA `saved`). They belong to the speculative path, which
    is not ported. One PLE ring and one QSA ring are kept, as in Rust's decode.
  - The MTP draft head and its fill inside the prefill, and the EAGLE3 head, are not ported. Image tokens (`prefill_image`) are
    not ported.
  - `hc_n` has no upper bound check beyond the kernels' own; Rust has none either.
- **Not covered by a test here.** QSA's sparse selection only changes the result past `idx_top_k` (2048) visible tokens. Both
  parity prompts (8 and 300 tokens) run the QSA kernels with every block selected, so a selection bug would not show. Decode
  runs `qsa_select` on every step, prefill only past the threshold.
- **Measured (2026-11-20, RTX 5070 Ti, 16 GB; available RAM 91.1 to 91.5 GB at every Flash-Next load).**
  - Regression, all pass: `test_smoke.py` (1000 of 1000), `test_quant.py` (25 types bit-exact), `test_backend_gemv.py` (35
    comparisons, max relative error 9.1e-8 on the 14B file), `test_backend_ops.py` (111 checks), `test_backend_seq.py` (55 checks),
    `test_score_14b.py` (argmax 192/192, max top-1 logprob diff 0.000000, decode 203.1 tok/s), `test_prefill_14b.py` part (a)
    (argmax 192/192, prompts of 64, 300, 1000 tokens, max diff 0.000000).
  - Flash-Next parity (`tests/test_flashnext.py` (a), `sc_score --threads 7` vs `rustingstrata score --gpu --threads 7`, 32
    continuation positions each): 8-token prompt argmax 30/32 (max |top-1 logprob diff| 0.317); 300-token prompt argmax 32/32
    (max 0.195). Total 62/64 = 96.88%, gate 97% **not met by one position**. Residency is the same in both binaries (4057 of 24576
    experts resident, plan line of each). Diagnostics on the 8-token prompt: all routed experts on the CPU worker (`STRATA_EXPERT_MB=0`)
    against the official: 29/32, mean |d| 0.072; the default placement against the official: 30/32, mean |d| 0.054; the two of
    ours against each other: 31/32, mean |d| 0.053. The differences are spread over every position (typical 0.00 to 0.1), with
    the two argmax flips at near-ties (-2.75 against -3.02, and -1.75 against -1.90). That points at expert arithmetic (the CPU
    worker's and the GPU's expert kernels differ from the official's rounding), not at the hyper-connection, PLE or attention
    ops, which give the same figures on both paths. The Flash-Next experts are IQ3_XXS (gate and up, 38 layers), IQ4_XS, IQ2_S,
    IQ4_NL, Q5_0 and Q2_0 (down). `test_backend_gemv.py` covers no IQ type, and `test_cpu_moe.py`'s checks are on the 14B Q6_K
    file, so the IQ expert arithmetic is verified only by `sct_dot_selftest` and the dequant tests. Not resolved.
  - Flash-Next speed (`tests/test_flashnext.py` (b), `sc_bench --threads 7 --ctx 8192` vs `rustingstrata bench --gpu --threads 7`,
    pp512 tg128 reps 3): ours pp512 83.7 tok/s (9.4 to 121.1) against 123.8 (54.9 to 158.5) official = 67.6%; tg128 9.2 tok/s
    (3.5 to 14.1) against 12.4 (8.3 to 16.3) = 74.2%. Both are below the 40 tok/s decode gate in absolute terms and the 95%
    ratio gate. Both runs had the same plan (4009 against 3996 experts resident, expert hit 81.3% against 81.3%), and both show
    large run-to-run spread (cold page cache of the 83 GB model with 91 GB free). The CPU worker is the clearest difference: 141.4
    ms of worker time per token against the official 76.9 ms, with the same thread count. Not resolved: the worker's IQ kernels
    (`cpu_moe.cpp`) are the first place to look.

### W3-E1c: `src/cpu_moe.cpp` (port of `cpu_moe.rs`: `signs`, `batch_rows`), `tools` probes not kept

- **Cause of the Flash-Next CPU worker gap (measured).** Per-type expert checks on real Flash-Next bytes: scalar and AVX2
  integer kernels for IQ3_XXS, IQ4_XS, IQ2_S, IQ4_NL, Q5_0 and Q2_0 agree with a float64 reference built from the exact
  dequant and the same Q8_K activation to 1e-7 relative (max |err| / max |ref|, 1e-7 to 2e-7). The IQ3_XXS gate and up
  matrices, used in 38 of 48 layers, were 1.5x slower than ggml. The sign table `signs()` was a function-local static
  with a thread-safe initialization guard, looked up four times per 32 values; the IQ3_XXS AVX2 kernel alone measured
  0.55 us/row against 0.21 us/row with a constant table (one thread, identical integers). Fix: `SignTable` is a
  constexpr constant (Rust builds `SIGNS` in a const block, so this is the same table, not an algorithm change).
- **Q2_0 task size.** `batch_rows` now uses 64-row tasks for Q2_0 on AVX2 when `out_len / 64 >= pool size`, as
  `cpu_moe.rs` does; the header comment claimed this before the code did it. Results are unchanged (one task computes a
  row). Q2_0 down: 62 to 56 us per two-expert call.
- **Measured (7 threads, warm, 2-expert calls, pages touched before timing; the same process for ours and ggml).**

  | layer (gate/up, down) | ours before us/expert | ours after | ggml us/expert | after / ggml |
  |---|---|---|---|---|
  | 1 (IQ3_XXS, Q5_0) | 197.8 | 117.1 | 132.1 | 0.89 |
  | 7 (IQ3_XXS, IQ4_NL) | 189.5 | 109.3 | 129.0 | 0.85 |
  | 2 (IQ3_XXS, Q5_0) | - | 115.7 | - | - |
  | 15 (IQ4_XS, IQ4_NL) | 116.3 | 117.5 | 123.6 | 0.95 |
  | 0 (IQ2_S, Q5_0) | 108.1 | 110.0 | 122.3 | 0.90 |
  | 9 (IQ2_S, IQ4_NL) | 99.6 | 102.9 | 116.3 | 0.89 |
  | 8 (IQ2_S, Q2_0) | 88.5 | 85.0 | 206.6 | 0.41 |

  The "before" column for layers 1, 7, 8, 15, 0 and 9 is from the warm run before the fix; the first run without the
  page touch read cold pages and is not used.
- **Why ours and ggml differ by about 1.5e-2 relative L2 per expert (not a port error).** ggml quantizes the expert
  activation with the weight type's `vec_dot_type`: Q8_K for IQ types, Q8_0 for Q5_0, IQ4_NL and Q2_0 (the down
  projection in every Flash-Next layer). Rust and this port quantize every activation to Q8_K (`batch_rows` and
  `rows_multi`). Reproduced in numpy from the exact dequant on real expert bytes: ggml's output matches
  x in Q8_K and the down input in Q8_0 to 2.4e-4 relative, and ours (Q8_K, Q8_K) differs from it by 1.1e-2 to 1.6e-2.
  So the difference is an arithmetic choice of ggml, the same one the official Rust build does not make.
- **Checked against Rust (`avx2` module and `int_dot`).** Every type Rust vectorizes is vectorized here and selected by
  `avx2_dot` (the same set; TQ1_0 and TQ2_0 stay scalar as in Rust). Gate and up share one Q8_K quantization of `h`
  when their types match, as in Rust. The per-call `std::vector` allocations match Rust's `Vec` use; they are not
  measurable next to the kernels (silu and the activation pass cost about 6 us per call). `IQ1M_W` and the TQ1_0 table
  are function-local statics as before; neither type is in Flash-Next.
- **Tests.** `test_cpu_moe.py` PASS (self-test; 20 cases vs ggml with max rel L2 6.1e-7 on the 14B model;
  `experts_multi` bit-identical). Regression PASS: `test_smoke` (1000 of 1000), `test_quant` (25 types), `test_backend_gemv`
  (35 comparisons), `test_backend_ops`, `test_backend_seq`, `test_cpu_worker`, `test_score_14b`.
- **Flash-Next end to end (2026-11-20; RAM before each load 91.2 to 91.3 GB, floor 90 GB; one process at a time, `.gpu.lock`
  held; one discarded official warm-up; then ours, official, ours, official).**

  | run | pp512 tok/s (range) | tg128 tok/s (range) | worker busy ms/token |
  |---|---|---|---|
  | ours round 1 | 98.7 (52.8-121.9) | 27.7 (24.7-30.1) | 25.8 |
  | official round 1 | 124.5 (55.4-159.2) | 24.3 (21.6-28.9) | 31.6 |
  | ours round 2 | 98.9 (52.2-122.3) | 34.4 (28.7-39.1) | 19.0 |
  | official round 2 | 124.8 (55.5-159.5) | 31.0 (29.5-32.0) | 22.0 |

  Expert hit 81.6% in both binaries. Resident experts: ours 4009/24576, official 3996/24576. Prefill is 79% of the
  official (gate 95%, not met); decode is 111% to 114% of the official across the two rounds, but the decode ranges
  overlap heavily.
- **Parity (`test_flashnext.py --skip-bench`, same run).** 8-token prompt argmax 29/32 (max |top-1 logprob diff| 0.260;
  argmax differs at positions 1, 9, 25); 300-token prompt argmax 32/32 (max 0.094). Total 61/64 = 95.3%, gate 97%:
  **not met**. The previous measurement (W3-E1b) was 62/64.
- **Not finished.** The parity gate fails by two positions. The prefill gate fails at 79% of the official. The
  remaining W3-E1b parity question (the argmax flips at near-ties) is open; the expert arithmetic is now verified against
  an exact reference and ggml, so it is not the explanation for the flips. The prefill path's expert handling (the
  batched `prefill_moe` path) was not profiled in this package.

## Not yet ported (tracked in `STRATA_CPP_PLAN.md`)

(GLM-5.3-Flash's `glm/` was ported on 2026-11-23, see the last section.) The MTP draft head (and its fill in the prefill), the EAGLE3 head, `prefill_cached` and `prefill_slots` (serve), `calibrate`,
the image path of the prefill, and the StrataFreet integration (W4-B). The hyper-connections, PLE and QSA are ported (W3-E1b),
with the parity and speed gates open (see W3-E1b). What exists: the build, the GGUF reader,
the quant reference, the NVRTC loader, `config.rs`, `weights.rs`, `memory.rs`, the CUDA backend (gemv family,
batched GEMM, sequence ops, graph), the CPU worker, the fully resident and partial decode, the batched prefill, and
the `sc_score` / `sc_bench` tools.

## MXFP4 (type 39): StrataFreet additions (build_mx, 2026-11-20)

Strata 1.2.5 has no MXFP4 (`quant/mod.rs` `GgmlType` and `cuda/mod.rs` `kernel_names`). The format and the kernels
are StrataFreet's (`stratafreet/quant.py` type 39, `strata_kernels.cu` `QI<39>` and `deq8<39>`, `cpu_ggml`). These
changes add it to strata-cpp. Each one is a deviation from Strata 1.2.5, and none changes the result of an existing type.

- **`src/quant.h`, `src/quant.cpp`**: `Ty::MXFP4 = 39`, `block_elems` 32, `block_bytes` 17 (one E8M0 byte and 16 bytes
  of E2M1 codes), `ty_from_u32(39)`, `ty_name` "MXFP4". `kKvaluesMxfp4` is ggml's `kvalues_mxfp4` (the E2M1 values
  doubled: 0 1 2 3 4 6 8 12 and their negatives). `e8m0_half` is `ggml_e8m0_to_fp32_half`, exact for every byte
  including the subnormal scales of e = 0 and 1. `mxfp4()` gives the values of the torch path in `stratafreet/quant.py`
  (`kv[code] * 2^(e - 128)`). `tests/test_quant.py` is bit-exact for 26 types (MXFP4 included), on random bytes and on
  whole experts of `gpt-oss-20b-MXFP4.gguf` (gate, down and up, 2880 x 2880 each).
- **`kernels/kernels.cu`** is no longer byte-identical to Strata 1.2.5. It gains a block "MXFP4 (StrataFreet addition)"
  after `QI<20>`: `e8m0_half`, `mxfp4_lut` (the `iq4nl_lut` mechanism with the MXFP4 table), `ldb4` (byte loads, since
  the blocks are 1-byte aligned), `QI<39>`, `deq8<39>`, and `X(39, mxfp4, 136)` at the end of `ITYPES`. 136 is 8 blocks
  of 17 bytes per 256 values, the super-block form of StrataFreet's copy. Every macro over `QTYPES` (dequant, gemv,
  gemvn, moe_gemv and their `_p` forms, gemm, gemms, gemmi, gemmis) then covers the type with the launch rules of the
  other 32-value types.
- **`src/cuda_backend.cpp`**: `QTYPES` has 25 entries, MXFP4 last. `gemv_cols` uses `block_elems` (32), so the column
  count must be a multiple of 32, and `cols % 256 != 0` takes the `_p` kernels (2880 columns in gpt-oss).
- **`src/cpu_moe.cpp`, `src/cpu_moe.h`**: `dot_mxfp4` (scalar) and `avx2::dot_mxfp4`, the structure of `dot_iq4nl` and
  `avx2::dot_iq4nl` with 17-byte blocks. The codes index `kKvaluesMxfp4` (the doubled table, so the integer sums are
  exact) and each block is scaled by `e8m0_half(e) * y.d`. Dispatched in `scalar_dot` and `avx2_dot`. `int_dots_selftest`
  covers MXFP4 (E8M0 bytes 120 to 135, so the scales differ). New test hook `int_dot_rows` (the scalar or AVX2 kernel
  over the rows of a raw buffer), exported as `sct_dot_rows` in `src/cpu_test_api.cpp`. Strata's `cpu_moe.rs` has no
  MXFP4 kernel, so this part is an addition, not a port.
- **Not done: `experts()` for gpt-oss.** gpt-oss uses the clamped `swiglu_oai` activation and expert biases, which
  `experts()` (silu, no biases) cannot express. The CPU check for MXFP4 is therefore the matvec only. Qwen3.6 (silu
  experts) needs nothing beyond the kernel.
- **Build directory `build_mx`.** `build_mx.ninja` and `build_cpu_mx.ninja` are copies of `build.ninja` and
  `build_cpu.ninja` with every output under `build_mx/`, so a benchmark that has `build/strata_cpp.dll` loaded is not
  overwritten. `build_mx.bat` runs both files with the vcvars64 and CUDA setup of `build.bat`, with `/arch:AVX2` kept.
  Those two ninja files are copies made by hand; `build.ninja` and `build_cpu.ninja` are unchanged.
- **Tests honor `SC_BUILD_DIR`** (default `build`): `test_smoke`, `test_quant`, `test_backend_gemv`, `test_backend_ops`,
  `test_backend_seq`, `test_cpu_moe` and `test_cpu_worker`. `SC_GPT_OSS_GGUF` overrides the gpt-oss path (default: the
  LM Studio copy). `test_cpu_moe.py` gains `mxfp4_checks`: the matvec of two real gpt-oss matrices, scalar against AVX2,
  against the float64 dot of the exact dequant with the Q8_K activation, and against the unquantized float64 dot
  within the activation rounding bound of each row. `test_backend_gemv.py` gains MXFP4 `gemv` (expert 0, the `_p`
  kernel) and `moe_gemv` (8 experts with one skipped id, shared x and per-expert x) checks against
  `stratafreet.strata_kernels` (`sk.gemv` and `sk.moe_gemv` with qtype 39).

## RAM lock: `ram` and `pin` options (StrataFreet addition, 2026-11-20)

Not in RustingStrata. The Rust engine reads weights through its memory map only, so when the Windows file cache drops
the model's pages (another model loaded), every expert read on the CPU or copied to the GPU faults from the slow drive
(about 18 MB/s random). Two opt-in sc_load options keep the tensor data in private RAM instead. Default is `off`, which
is the original mapping, so existing callers are unchanged.

- **`"ram": "off" | "copy" | "lock"`** (`src/gguf.h`, `src/gguf.cpp`, `src/capi.cpp`). `Gguf::load_to_ram` reads each
  shard's tensor data (from its data offset to the file end) with ReadFile in 64 MB chunks from a
  `FILE_FLAG_SEQUENTIAL_SCAN` handle, into `VirtualAlloc(MEM_RESERVE|MEM_COMMIT)` memory, then releases the file views
  (`Mapping::release`). `Gguf::raw_bytes` answers from that memory when it is loaded, so every weight, expert, PLE and
  CPU-worker pointer that comes from `Gguf::bytes` points into RAM with no other change. `lock` also makes the copy
  resident: `SeIncreaseWorkingSetPrivilege` is enabled on the process token, the minimum working set is raised to the
  data plus 1 GB (`SetProcessWorkingSetSizeEx`, `QUOTA_LIMITS_HARDWS_MIN_ENABLE`), then `VirtualLock` runs in 1 GB calls.
  All or nothing: a refused call unlocks what it got and the load goes on as `copy`, with the reason in
  `sc_stats` `ram_warnings`.
- **RAM guard.** Before the copy, `GlobalMemoryStatusEx` `ullAvailPhys` must cover the tensor data plus 6 GB. A smaller
  value is a refusal (`sc_last_error`), not a load that pages.
- **Progress.** stderr lines `[strata-cpp] ram copy: X / Y GB read, Z MB/s` every GB, and the total at the end.
- **`"pin": true|false`** (`src/cuda_backend.h`, `src/cuda_backend.cpp`, `src/capi.cpp`). Needs `ram` copy or lock
  (otherwise sc_load refuses). `CudaBackend::register_host` calls `cuMemHostRegister` on each RAM region
  (`CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_READ_ONLY`, retried without READ_ONLY if the driver refuses that
  flag). In `htod`, a source wholly inside a registered range is a `cuMemcpyHtoDAsync` straight from it; the copy
  through the 64 MB pinned ring (one memcpy per byte) is skipped. The ring path is unchanged for every other source. A
  registration failure is a warning (`ram_warnings`) and the load goes on unpinned. Destructor: `unregister_host` after a
  stream synchronize, before the RAM copy is freed.
- **Stats** (`sc_stats`): `ram_mode`, `ram_bytes`, `ram_locked_bytes`, `ram_pinned`, `ram_direct_bytes` (bytes copied
  from registered memory), `ram_load_seconds` (copy and lock), `load_seconds` (whole sc_load), `ram_warnings`.
- **Build.** `build.ninja` links `advapi32.lib` (the token and working-set calls). `build_mx.ninja` and
  `build_cpu.ninja` are unchanged.
- **Tools.** `sc_bench` and `sc_score` take `--ram` and `--pin`. `sc_bench` also takes `--evict <file[,file...]>
  --evict-gb N` (reads those files after the load, before the timed decode, to churn the file cache for the eviction
  test) and prints `load:` and `stats after load:` lines.
- **Deviation from the Rust ports:** none in the kernels or the algorithms. The copy is a C++ addition; its memory
  layout matches the mapping exactly (the same offsets), so the parity of `dot`, `matvec` and the kernel launches is
  not affected.

## Sampler: `min_p`, `presence_penalty`, `repeat_penalty` (StrataFreet addition, 2026-11-21)

Not in RustingStrata (its sampler is temp/top_k/top_p only, `ops.rs::Sampler`). StrataFreet's engine (`engine.py`
`_sample`) implements min_p and the two penalties; the native sampler now matches it exactly, so a temp=0 request
with penalties is penalized-argmax on the host, not the on-device argmax of raw logits.

- **`src/sampling.h`, `src/sampling.cpp`** (`Sampler::min_p`, `presence_penalty`, `repeat_penalty`, `recent`,
  `recent_n`, `has_penalties()`; `Sampler::penalize`; `Sampler::cands` min_p cut; `sampler_from_json`). Semantics are
  `engine.py _sample` (the torch oracle): the penalties land on the raw logits **before** the greedy/sampling choice —
  repeat first (`v > 0 ? v / repeat_penalty : v * repeat_penalty`), then presence (`v -= presence_penalty` — a
  positive penalty lowers the logit of tokens that already appeared, as OpenAI/llama.cpp), over the
  unique recent ids, most recent last; min_p keeps candidates with `p >= min_p * p_max` on the top_k-trimmed
  softmax probs, before the top_p cut. Out-of-range recent ids are skipped (engine.py would crash on them; the DLL
  is defensive). `sampler_from_json` parses the capi.h sampler JSON (all eight keys) for both DLLs and validates the
  ranges (temp/top_k not negative, top_p in (0, 1], min_p in [0, 1], penalties finite, repeat_penalty not negative,
  recent no longer than 4096 ids).
- **`src/capi.cpp`** (`sc_decode_sample`, `sc_prefill_sample`). Both call sites parse through `sampler_from_json`
  (one shared parser) and pass the recent window they get. The device-argmax fast path (`GpuModel::step_greedy`)
  runs only when `temp == 0` **and** no penalty is active (`has_penalties()`); otherwise the step runs on-device and
  the pick is host-side, penalized first. `{"temp":0}` with no penalties keeps the exact old behavior and speed.
- **`src/dev_api.cpp`** (`scd_sampler_pick`). Model-free host-only entry into the same `strata::Sampler` for the
  exact sampler tests (`tests/test_sampler.py`): 46 checks (argmax/tie/NaN, every penalty rule, JSON validation,
  xorshift64 state, min_p/top_k/top_p cuts, draw fall-through).
- **Deviation from the Rust ports:** the candidate cut and draw math of `ops.rs::Sampler` are untouched; the
  additions run strictly before them (penalties) or between top_k and top_p (min_p), so sampling parity with the
  Rust engine holds for every JSON without the new keys.

## Inkling-2-Mini: native port (StrataFreet addition, 2026-11-21/22)

Not in RustingStrata 1.2.5. The reference is ggml-org's `inkling2.cpp` / `inkling2-loop.cpp` (PR #389) and the
unsloth GGUF layout; `build/ref_pos0.py` is an independent numpy forward that matches the oracle's `result_norm`
at cos 1.00000. RustingStrata's decode/prefill/cache machinery is reused where the shape matches.

- **W1 loader/config** (`src/config.{h,cpp}`, `src/weights.{h,cpp}`, `src/expert_cache.{h,cpp}`):
  `Arch::Kind::Inkling2`; 24 physical blocks, 128 experts, top-8, hidden 2304, expert width 1152, 48 logical
  slots over 3 loop passes (`loop_layer_of`); the Inkling scalars (SWA window, rel-bias extent, shortconv
  width, loop gate rank, situ betas, scale floors, softcap, router bias/eps, unpadded vocab). `Weights::layer`
  serves a slot's physical block. `kExpertMatsInkling2` is the 4-matrix expert layout (fused gate/up, down,
  per-expert scales); the cache is keyed by the physical block and the worker remaps slot -> physical. The
  decode scratch rows (`bks`, `bnk`, `bkl`, `pre`, `inpb`, `mat`, `mkv`, `gat`, `g64`, `cg`, `cf`, `gu`, `tbl`,
  `rel`, `sink`, `kpj`, `vpj`, `moe`, `ao`) are allocated `kTmax` rows; the debug grid gets its own 200*1024
  float buffer (rows 0-87) and the SLOT-1 dump 16*ne+1024.
- **W2 decode** (`src/gpu_model.cpp`, `forward_inkling2`; kernels in `kernels/kernels.cu` and
  `src/cuda_backend_ops.cpp`): the full 48-slot graph - embedding norm, block bank and cross-pass message
  controllers (`mix_dec`, `gate_pack`, loop `gemv`s, `control_mix`), SWA/full relative-bias attention with a
  per-query table and zero sink (`rel_table`, `attn_decode_gq` bias/extent), shortconv states (`conv_ink`),
  Q/K/R clips (`clip_rows`), sigmoid output gate (`sig_mul`), fused gate/up MoE (`moe_gemv`) with `situ_q`,
  the router bias/eps (`moe_route`), the per-expert down scale (`moe_combine` ds/ids), and the softcapped head.
  Iterated against the oracle: 255/256 argmax (99.6%), max |top-1 logprob diff| 0.0612 vs llama.cpp CPU on
  Q8; 252/256 (98.4%), 0.152 vs CUDA (the reference's own CUDA-vs-CPU control scores 140/256, 3.02). The
  mixed GPU/CPU expert path is not bit-reproducible (CPU q8_K activations vs GPU f32); `STRATA_EXPERT_MB=1`
  pins the all-CPU path for reproducible parity, and `STRATA_INK_XQ=1` selects int8 activations (default off).
- **W3 batched prefill** (`src/gpu_model.cpp` `forward_inkling2_chunk` + `ink_chunk_body`, `src/prefill.cpp`
  `prefill_inkling2`): chunks of up to 16 tokens (the mailbox's per-token regions are TMAX = 17 rows) through the
  slot graph, with the per-token work running token by token through the decode launches and the routed-expert
  pass batched over the chunk: one `moe_route` with t rows (the CPU worker answers all t tokens' misses in one
  `experts_multi` pass), one fused gate/up `moe_gemv` with ke slots per token (`xg = ke`), one `situ_q`, one down
  `moe_gemv` and one `moe_combine` with t tokens. Per-token state that spans slots (x, bks, bnk, bkl, pre, h, moe,
  step) lives in per-token regions of the decode buffers (bks 4 rows a token, bnk 2); the slot-local scratch is
  reused token by token. The whole body is captured by the backend's CUDA-graph cache (one graph per chunk size and
  gq regime; a chunk straddling `ATT_GQA_MIN` runs eagerly once). `Buf::slice` gives the per-token views to
  `put`/`dtod`/`rms_norm`.
  - **Correctness:** A/B against the verified token-by-token path on IQ3_XXS is **bit-identical** (argmax,
    log-probability and max |logit difference| 0) for 1, 2, 8, 16, 17, 33 and 64 tokens. On the paged Q8 path the
    argmax matches and the top-1 logprob differs by 0.004-0.072 (the established mixed CPU/GPU expert rounding).
    The sanitizer run found the two bugs this exposed: the full-attention blocks with cross-pass messages were
    taking the SWA input-norm branch (null `attn_norm`), and `moe_combine` did not advance the Inkling down-scale
    `ids` pointer with `blockIdx.y` (the qwen path never passes `ds`; now `if (ds) ids += tk * k`).
  - **Speed (ctx 8192, pp512/tg128, 3 reps):** all-resident IQ3_XXS **145.3 (132.4-151.9) prefill / 98.8
    (97.8-99.9) decode tok/s**; the token path in the same session measured 73.7 prefill. The paged Q8 file keeps
    the token path (the chunk's per-layer worker granularity measured 9.2 vs 13.9 pp512), chosen automatically at
    load: the chunk is used only when the expert cache is fully resident.
  - **Parity:** the 512-token oracle run and `tests/test_score_inkling.py` exercise the chunk path (an all-resident
    IQ3 load) and pass. Debug dumps (`STRATA_INK_DEBUG` / `STRATA_INK_DUMP`) fall back to the token path.
- **W4 app + speed** (`stratafreet/strata_cpp.py`, `src/capi.cpp`, `src/cuda_backend.{h,cpp}`): `ARCHS`
  includes `"inkling2"`, so `Engine` selects the backend automatically. The q8 KV cache (the backend's
  `STRATA_KV_Q8` kernel variant) is now reachable from the ABI: `sc_load` accepts `"kv": "q8"` and passes it to
  `CudaBackend::create`, and the app forwards `EngineConfig.kv_dtype == "q8"` for Inkling only (the other
  architectures keep f16 until their q8 path is checked). Parity with q8 KV: 64/64 argmax vs the llama.cpp CPU
  oracle (f16: 63/64), max |dlogp| 0.2106. At ctx 31232 q8 KV halves the cache to 3111 MB, so all 3072 IQ3_XXS
  expert slots fit: `sc_bench` pp512 101.7 / tg128 101.9 tok/s (f16 KV leaves 2366/3072 resident and the ON
  bench ran at 50.3-61.5 tok/s), and the live app measured 103.4-103.7 tok/s decode at the same ctx.
  Live Engine runs: IQ3_XXS 11.66 GB all-resident, ctx 2048 -> 9821 MB slots + 1462 MB weights/state/scratch +
  384 MB KV, 3072/3072 experts, worker idle, decode 86.4-98.7 tok/s; Q8_K_XL 21.29 GB with `ram: lock` ->
  1761/3072 slots at ctx 8192 (95.2% hit, 63.7 ms/token worker), sc_bench pp512 13.9 / tg128 16.4 tok/s.
  The largest quant that fits entirely on this 16 GB card is IQ3_XXS (Q3_K_XL 13.63 GB next; IQ4_XS at/over
  the ceiling). `STRATA_INK_XQ=1` measured pp512 104.0 / tg128 104.2 tok/s vs 102.5/102.6 (ctx 8192, IQ3_XXS);
  the +1.6% is small, so the exact f32 activation path stays the default and the switch stays opt-in.
- **W5 tests/docs**: `tests/test_score_inkling.py` (parity vs the unsloth llama.cpp CPU `/completion`: 63/64
  = 98.44% argmax, max |dlogp| 0.1105 on IQ3_XXS; sc_bench gate: decode 102.6 tok/s >= 50). `README.md`,
  `PORT_MAP.md` (section 4b) and this section updated.
- **Debug knobs** (`gpu_model.cpp`, `cuda_backend_ops.cpp`): `STRATA_INK_DEBUG` now uses one rule everywhere -
  set and non-zero enables - so `STRATA_INK_DEBUG=0` no longer turns on the per-token embedding readback or the
  attention print. The `STRATA_INK_DUMP` record is `4*ne+88` floats: the graph writes the pre-MoE `h` and `mix`
  at bytes `8*ne+352` and `8*ne+352+4*ne`; the old `2*ne+88` read truncated the mix. Existing consumers
  (`build/moe_check.py`) read only the prefix and are unchanged.

## GLM-5.3-Flash (`glm5-next`): port of `glm/` (2026-11-23)

RustingStrata 1.2.5 implements GLM-5.3-Flash in `src/glm/` (`mod.rs`: hyperparameters, tensor table, `gpu_defaults`;
`forward.rs`: the CPU reference forward with GPU dense matvecs, the VRAM expert cache, batching, chunked prefill and
the k-pool DSA indexer) on top of `src/deepseek/mat.rs` and `src/deepseek/ops.rs`. The first port skipped it; this
section ports it line by line. Reference notes, the model facts and the formula cross-check against llama.cpp
`glm5-next.cpp`: `docs/GLM5_NOTES.md`.

Files:

- `src/glm.h`, `src/glm.cpp` (port of `glm/mod.rs`: `GlmConfig::from_gguf`, `n_kda`, `n_dsa`, `expected_tensors`,
  `validate`, `gpu_defaults`), `src/glm_forward.cpp` (port of `glm/forward.rs`: `GlmModel` and its `new`, `reset`,
  `step`, `prefill`, `step_opt`, `prefill_chunk`, `kda`, `dsa`, `ffn`, `ffn_batch`, `hc_mix`, `preload_hot`,
  `prefetch_experts`, `tr`; `max_ctx`, `route_sigmoid`, `l2_norm`).
- `src/ds_mat.h`, `src/ds_mat.cpp` (port of `deepseek/mat.rs`: `Mat`, `matvec_many`, `matvec_rows`, `vec_f32`,
  `q8_roundtrip`, and `mod dev` - the per-GPU parts, weight upload, the LRU expert cache with its 16 MB staging slot,
  `gemv`, `batch_mats`, `rows`).
- `src/ds_ops.h`, `src/ds_ops.cpp` (port of the `deepseek/ops.rs` functions GLM calls: `rms_norm`, `rms_norm_w`, `Hc`,
  `hc_coeffs`, `hc_collapse`, `hc_post`, `swiglu_clamp`). `Rope` and `route` (DeepSeek-V4.1 only) are not ported.
- `src/cpu_moe.h/.cpp`: `int_dot_kernel(Ty)`, the public form of `cpu_moe.rs` `int_dot` that `deepseek/mat.rs` calls
  (the existing kernel selection, unchanged).
- `src/capi.cpp`, `src/capi.h`: `sc_load` builds a `GlmModel` for `general.architecture` `glm5-next` or `glm5next`
  (port of the `main.rs` dispatch of `generate`/`bench`), and every `sc_*` entry dispatches to it.
- `src/cpu_test_api.cpp`: CPU-only `sctg_*` hooks (open with the GPU path off, step, prefill, reset, validate, and the
  helpers) for `tests/test_glm_cpu.py`.
- Build: the four GLM sources join `strata_cpp.dll` (`build.ninja`, `build_mx.ninja`); they and `src/backend.cpp` join
  `strata_cpu_test.dll` (`build_cpu.ninja`, `build_cpu_mx.ninja`). No new CUDA kernels: the GPU side uses the backend's
  existing `upload`, `gemv`, `gemvn`, `gemm_dense`, `alloc`, `htod`, `dtoh`, `mem_info`.
- Tests: `tests/test_glm_cpu.py` (new, CPU), `tests/test_score_glm5.py` (new, GPU parity against the official binary),
  `tests/sc_env.py` (`SC_MODEL_GLM5`, `SC_GLM5_ALIAS_DIR`).

Deviations from the Rust code:

- **Arch spelling (StrataFreet addition).** 1.2.5 accepts only `general.architecture = glm5-next` and reads
  `glm5-next.<key>`. llama.cpp and the unsloth GGUFs write `glm5next`; the port accepts both and reads the keys under
  the file's own spelling. The official binary still needs `glm5-next`, so `tests/test_score_glm5.py` hands it an alias
  (shard 1 re-serialized with the keys renamed, the other shards hard-linked).
- **Device state per model.** Rust keeps `deepseek/mat.rs`'s device in a process-wide `OnceLock` (`dev::get()`) and
  computes on rayon's global pool. The port keeps both in a `DsEnv` the model owns (`Mat` points at it), so `sc_free`
  releases the VRAM and two handles do not share one budget. The Rust `Mutex` around it is dropped (one thread drives a
  model). The backend is opened through a factory the caller passes (`capi.cpp`: `CudaBackend::create`), so the same
  sources build into the CPU-only test DLL. `STRATA_DS_BACKEND=vulkan` is refused (no Vulkan backend in this port).
- **Environment.** Rust's `gpu_defaults` sets `STRATA_DS_GPU=1`, `STRATA_DS_GPU_FRACTION=0.9` and
  `STRATA_GLM_PREFETCH=2` in the process environment unless `STRATA_GLM_CPU` is set. A DLL must not change its host's
  environment, so `glm_gpu_config` applies the same defaults to the values it reads. The switches the Rust code reads
  per call (`STRATA_DS_BATCH`, `STRATA_CPU_REF`, `STRATA_DS_NOEVICT`, `STRATA_DS_LFU`, `STRATA_GLM_CHUNK`,
  `STRATA_GLM_BATCH`, `STRATA_GLM_PREFETCH`, `STRATA_GLM_ROUTE`, `STRATA_GLM_PROF`, `STRATA_GLM_DUMP`,
  `STRATA_GLM_PRELOAD`, `STRATA_GLM_PRELOAD_HOST`, `STRATA_GLM_TRACE`) are read once at load. sc_load's `device` is
  the default of `STRATA_DS_DEVICES` (as `main.rs` hands `--device` to that variable).
- **Errors instead of panics.** The Rust `.unwrap()`s on device calls and the context `assert!`s panic. Here a failed
  device call inside a matvec throws `DsError`, which `GlmModel::step`/`prefill` catch and report through
  `sc_last_error`; the model then refuses further calls until `sc_reset`. The context limit is checked before any state
  changes (Rust asserts per chunk). The limit is sc_load's `ctx` when given, else `max_ctx` (`STRATA_GLM_CTX`, else the
  trained context).
- **Load-time dims check (StrataFreet addition).** The Rust matvec asserts input/output lengths at run time; with raw
  pointers that would be an out-of-bounds read, so `GlmModel::create` compares the dims of every tensor the forward
  reads with `expected_tensors()` (trailing 1s ignored) and refuses a mismatch. `from_gguf` also refuses
  `conv_kernel < 2`, `kpool 0`, `expert_used_count` of 0 or above `expert_count`, and zero head counts or width
  (values the Rust code divides by or indexes with).
- **Not ported**, as for every architecture of this DLL: `glm::generate`, `chat`, `serve`, `chat_prompt*`,
  `parse_calls`, `decode_greedy`, `stop_ids` (tokenization, templates, tool calls and HTTP stay in the host). The
  sampler of `sc_decode_sample`/`sc_prefill_sample` (temperature, top-k/p, min-p, penalties) works on the GLM logits;
  RustingStrata's own GLM path is greedy only. `prefetch_expert` is the Rust non-Linux body (a no-op).
  `sc_save_profile` writes nothing for GLM (glm/forward.rs keeps no profile file); `pin` is refused; the calibration
  file is not read.
- **Summation order** follows the Rust iterators everywhere (left folds in f32), rayon's parallel items become
  `ThreadPool::for_each` items computing the same values; the CPU forward is therefore deterministic, and the chunked
  prefill equals the token-by-token path bit for bit (checked by `tests/test_glm_cpu.py`).

Verification (2026-11-23, CPU only - the GPU was in use by another job): `build.bat` and `build_cpu.bat` build with no
warning from the new files; `tests/test_glm_cpu.py` PASS (numbers in `docs/GLM5_NOTES.md`); `test_quant` (26 types),
`test_sampler` (46 checks) and `test_gguf` (on the synthetic GLM files) PASS; the real UD-IQ1_S tensor table equals
`expected_tensors()` (1412 tensors, checked from the header). GPU, later the same day: `tests/test_score_glm5.py
--n 4` against the official binary argmax 12/12, max |top-1 logprob diff| 0.0010; the full 3 x 64 run and speed were not
measured (`HANDOFF.md`).
