# kernels/ — verbatim copies from RustingStrata 1.2.5

| file | source | sha256 |
|---|---|---|
| `kernels.cu` | `native/strata_src_v1.2.5/src/cuda/kernels.cu` | `c7baea90952a5917d678ecf2a23d02b9f4817a4acd6d8473ae01881f4b75eec7` |
| `tables.cuh` | `stratafreet/strata_tables.cuh` (the tables `cuda/mod.rs` prepends) | `c530dab89c18b6cb897ac96491c7851a2b16d24970131cb76ec7b7fa8ecf7603` |

Both are unmodified. They are compiled with NVRTC at run time, exactly as Strata does; nothing here is
hand-edited, and `tools/gen_tables.py` derives the host-side copy of the tables from `quant/tables.rs`
instead of retyping them.

## Diff against StrataFreet's copy of the same file

`diff kernels/kernels.cu ../../strata_kernels.cu` → **142 lines only in 1.2.5, 125 only in StrataFreet's
copy, 267 changed lines in total.** The two files are not the same revision, and the difference is
exactly the two projects' own additions:

**Only in 1.2.5** (upstream, new since the copy StrataFreet took):
- `TQ1_0` and `TQ2_0` (ggml types 34, 35): `tq1_trit`, `tq1_vals`, `tq2_vals`, `deq8<34>`, `deq8<35>`,
  `QI<34>`, `QI<35>`.
- Changes in `attn_prefill_t<HD>` / `attn_prefill_llama<HD>` and a `qsa_bit` helper.

**Only in StrataFreet's copy** (this project's own additions, not Strata's):
- `MXFP4` (type 39), `FP8` (type 250) and `NVFP4` (type 251): `deq8<39>`, `deq8<250>`, `deq8<251>`,
  `QI<39>`, `QI<251>`.

**Consequence for the port:** `native/strata_cpp/kernels/kernels.cu` stays *verbatim 1.2.5*, so it has
TQ1_0/TQ2_0 and does **not** have 39/250/251. Flash-Next UD-Q3_K_XL and Qwen3.6-14B-A3B use neither set.
If a model with MXFP4/FP8/NVFP4 experts is ever run through strata-cpp, those three block types must be
added as a documented modification (they exist in StrataFreet's copy and in `quant.cpp`'s reference
dequantizers are *not* ported for them — Strata's `GgmlType` rejects 39/250/251 outright, see
`quant/mod.rs:15`). That is recorded in `MODIFICATIONS.md`.
