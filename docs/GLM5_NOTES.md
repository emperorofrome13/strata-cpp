# GLM-5.3-Flash (`glm5-next`) port — reference notes

Written 2026-11-23 for the port of RustingStrata 1.2.5's `src/glm/` into strata-cpp (branch `glm5next`).

## Sources

| Source | License | Used for |
|---|---|---|
| RustingStrata 1.2.5 `src/glm/mod.rs` (536 lines), `src/glm/forward.rs` (772), `src/deepseek/mat.rs` (396), `src/deepseek/ops.rs` (236), the glue in `src/main.rs`, README section "GLM-5.3-Flash" | RustingStrata License 1.0 | the primary source, ported line by line |
| llama.cpp `src/models/glm5-next.cpp` (master, fetched 2026-11-23, 1069 lines; not committed) | MIT | cross-check of every formula (table below) |
| FreeToken `freetoken/models/glm5_next/` (`args.py`, `attention.py`, `kda.py`, ...) | Apache-2.0 | read for the model description (NoPE MLA, k-pool indexer, mHC); torch/GPU code, not used as an oracle |
| unsloth `GLM-5.3-Flash-GGUF` UD-IQ1_S header (shard 1 metadata, shards 2-3 tensor tables) | MIT (weights) | model facts below; read with a header-only parser, no tensor data touched |

RustingStrata's own README reports for its GLM path: greedy output equal to llama.cpp b12119 (`--cpu-moe`) for 32
tokens on 3 prompts on the CPU path, and 188/192 teacher-forced top-1 matches with the GPU matvecs (the four misses are
llama.cpp near-ties, margins 0.03-0.53 logprob); a later 3 x 32 check 96/96.

## Model facts (UD-IQ1_S, 87 GB in 3 shards)

- `general.architecture = glm5next` — **llama.cpp's spelling**. RustingStrata 1.2.5 (source and the official
  `rustingstrata-windows-x86_64.exe`, whose strings contain `glm5-next` and no `glm5next`) only recognizes
  `glm5-next` (`main.rs` compares `general.architecture == "glm5-next"`, and every key is read as `glm5-next.<key>`),
  so on this file it would not take its GLM path (not run: the GPU was busy). strata-cpp accepts both spellings and
  reads the metadata keys under the one the file uses.
  `tests/test_score_glm5.py` gives the official binary an alias set: shard 1 (metadata only, 9.4 MB, 0 tensors)
  re-serialized with the 40 `glm5next.*` keys and the arch value renamed, shards 2-3 hard-linked. The round trip was
  checked: all 330 metadata entries are identical apart from the rename.
- 46 blocks = 45 decoding layers + 1 NextN block (`nextn_predict_layers 1`). `head_count_kv` is 0 for KDA layers and
  1 for DSA layers: 34 KDA, 11 DSA (layers 3, 7, ..., 43) plus the NextN block 45 (DSA).
- n_embd 4096, 64 heads, KDA head_dim 128 (8192 = 64 x 128, the shape both ports require), conv kernel 4,
  gate_lower_bound -5.0.
- MLA: q_lora 1536, kv_lora 512, key/value_length_mla 256, `rope.dimension_count 0` (NoPE).
- Indexer: 32 heads x 128, top_k 2048, kpool 4 (select 512 pools of 4 rows plus the incomplete tail).
- mHC: 4 streams, 20 Sinkhorn iterations, eps 1e-6.
- MoE: 288 experts, 8 used, expert/shared width 2048, 1 shared expert, 3 leading dense layers (n_ff 12288), sigmoid
  gating (2), weights norm true, scale 2.5, swiglu clamp 10.0 (exp and shexp).
- eps: rms 1e-5, layer norm (indexer key) 1e-6. Context 1,048,576. Vocab 154880.
- Tensor table: **1412 tensors, equal to `GlmConfig::expected_tensors()`** (no missing, no unexpected, no dim
  mismatch; checked from the header dumps). Types: F32 638, Q8_0 346, Q5_K 248, IQ1_S 56, Q6_K 49, IQ3_XXS 39,
  IQ2_XXS 28, IQ4_XS 3, Q2_K 2, Q4_K 2, Q3_K 1 — all among the 26 accepted types.

## Formula cross-check: RustingStrata 1.2.5 vs llama.cpp glm5-next.cpp

| Part | glm/forward.rs, deepseek/ops.rs | llama.cpp glm5-next.cpp | Same? |
|---|---|---|---|
| Embedding | row of `token_embd`, copied to 4 streams | `ggml_repeat_4d(inp, n_embd, hc, ...)` | yes |
| hc mixes | `hc_fn @ rms_norm(flat 4n, eps)` | `ggml_rms_norm(flat, norm_rms_eps)`, `mul_mat(hc_fn, ...)` | yes |
| pre / post | `sigmoid(m*s0 + b) + hc_eps`, `2*sigmoid(m*s1 + b)` | `sigmoid(affine)`, `scale_bias(1, hc_eps)`; `sigmoid`, `scale 2` | yes |
| comb | `comb[d + 4s] = m[8..24]*s2 + b`; softmax over d, + eps; norm_cols; (iters-1) x (norm_rows, norm_cols) | `[dst, src]` view, `soft_max` over dst, `+eps`, `norm_cols`, loop `norm_rows`, `norm_cols` | yes |
| collapse / post | `sum_s pre[s] x[s]`; `f*post[d] + sum_s res[s]*comb[d+4s]` | `build_hc_pre` weighted sum; `build_hc_post` | yes |
| Head | mean of the 4 streams (`[0.25; 4]`), rms * output_norm, output | `build_hc_mean`, `build_norm`, `mul_mat(output)` | yes |
| KDA conv | depthwise conv over the last `conv_k - 1` raw projections + current, then silu | `ggml_ssm_conv` on `concat(state, x)`, `ggml_silu` | yes |
| KDA q/k norm | `l2_norm(eps 1e-6)` per head | `build_gdn_l2_norm(1e-6)` | yes |
| KDA decay | `lb * sigmoid(-((f_b f_a x + dt_b) * A))` per key channel | `g1 + dt_b`, `* A`, `sigmoid(scale -1)`, `scale lb` | yes |
| KDA beta | `sigmoid(W_beta x)` per head | `sigmoid(mul_mat(ssm_beta, cur))` | yes |
| KDA recurrence | `S = diag(e^g) S; S += k (beta (v - S^T k))^T; o = S^T q / sqrt(hd)` | `build_recurrent_attn` (gated delta rule, q scaled by 1/sqrt(d_k)) | yes |
| KDA output | `rms_norm(o, ssm_norm, rms eps) * sigmoid(g_b g_a x)`, `wo` | `build_norm(o, ssm_o_norm, RMS)`, `* sigmoid(g2)`, `wo` | yes |
| MLA q / latent | `q_b(rms_norm(q_a x))`, `rms_norm(kv_a x)` (NoPE, kv_a has no rope part) | same, `GGML_ASSERT(n_rot == 0)` | yes |
| MLA attention | `k_b` absorbed into q per head, softmax over the latent rows with scale `1/sqrt(k_mla)`, `v_b` per head, `wo` | `q_absorbed = wk_b q`, `build_attn_mha(..., wv_b, ..., kq_scale = 1/sqrt(n_embd_head_k_mla))` | yes |
| Indexer key | `layer_norm(ix_k x, eps = layer_norm_epsilon) * w + b` | `build_norm(ik, k_norm, k_norm_b, LLM_NORM)` | yes |
| Pool | per dim: `softmax_k(gate + ape) . key` over each complete pool of 4 | `soft_max(pg + ape)` over kpool, `sum_rows(probs * pk)` | yes |
| Score | `sum_h max(iq_h . pooled, 0) * w_h / sqrt(hd * nih)` | `weights * 1/sqrt(...)`, `ggml_lightning_indexer` | yes |
| Selection | all rows while `pos < top_k`; else the top `top_k / kpool` pools (ties to the lower pool) plus the tail | `top_k` of `min(n_pool, top_k / kpool)` pools plus the tail (`indexer_kpool_select_tail`) | yes, see differences |
| Router | `sigmoid` probs, top-8 by `probs + exp_probs_b`, weights `probs / max(sum, 6.1035e-5) * 2.5` | `build_moe_ffn` sigmoid gating, norm, scale | yes |
| Experts | `swiglu_clamp(gate, up, 10)`: `silu(min(g, 10)) * clamp(u, -10, 10)` | swiglu clamp of `swiglu_clamp_exp` | yes |
| Shared expert / dense FFN | plain `silu(gate) * up`, no clamp | `build_ffn(... LLM_FFN_SILU, LLM_FFN_PAR)` | yes |

### Differences between RustingStrata 1.2.5 and llama.cpp master (kept as 1.2.5 has them)

- llama.cpp master reads `attention.indexer.types` (per-layer full or shared indexer; a shared layer reuses the
  previous DSA layer's selection `prev_sel`) and `attention.indexer.kpool_select_tail`. 1.2.5 gives every DSA layer its
  own indexer and always selects the tail. The UD-IQ1_S file has neither key, and llama.cpp's defaults
  (`llama-hparams.h`: `indexer_kpool_select_tail = true`; `is_indexer_full_impl` filled with 1 in `load_arch_hparams`)
  are exactly 1.2.5's behaviour, so for this file both select the same rows.
- llama.cpp's fused ops (`ggml_dsv4_hc_pre`, `ggml_dsv4_hc_comb`, `ggml_top_k_mask`, flash attention) compute the same
  values in another order; both RustingStrata and this port run the unfused formulas in f32.
- The NextN/MTP block is loaded by neither 1.2.5 nor this port.
- RustingStrata's README checked the indexer path against llama.cpp up to 3.6k tokens (same top-5 sets, same first
  token). This port has not been run on the real model yet.

## What the port runs (and what it does not)

- `sc_load` on a `glm5-next`/`glm5next` file builds `GlmModel` (src/glm_forward.cpp) instead of `Model` + `GpuModel`.
  The weights are read from the file mapping (or the `ram` copy); every non-expert 2-D weight is uploaded to VRAM at
  load (including `token_embd`, whose rows are still dequantized on the CPU, as in Rust); routed experts go through
  the VRAM cache: uploaded on first use under the budget (`STRATA_DS_GPU_EXPERTS` GB, else 90% of free VRAM minus
  56 MB), then the least recently used slot of the same shape is overwritten, or the expert is staged per use.
- Prefill: 512-token chunks layer by layer (`STRATA_GLM_CHUNK`), each routed expert read once per chunk; the last
  prompt token goes through `step` for the logits, then the prompt's hottest experts are preloaded.
- Not ported: `generate`, `chat`, `serve`, the chat template, tool-call parsing (the host's job, as for every arch of
  this DLL); the Vulkan backend (`STRATA_DS_BACKEND=vulkan` is refused); `prefetch_expert` stays the no-op the Rust
  code is outside Linux.

## Verification so far (2026-11-23, GPU busy: CPU only)

- `tests/test_glm_cpu.py`: PASS. A synthetic glm5-next GGUF (n_embd 64, 4 layers + NextN, F32) through the CPU path
  against an independent float64 numpy forward written from the llama.cpp formulas above: 20 positions, max |logit
  diff| / max |logit| 6.1e-6, argmax 20/20, the k-pool selection active from position 8 (an all-rows reference
  differs by 4.9e-1). The chunked prefill (chunk 512 and 3), two consecutive prefills, both arch spellings and 2-4
  threads give bit-identical logits to the token-by-token path. Plus unit checks of hc_coeffs, hc_post,
  route_sigmoid (the Rust unit tests' cases and 20 random 288-expert routings), l2_norm, swiglu_clamp, the f16
  rounding (140,009 values equal to numpy's), q8_roundtrip (Q8_0, IQ4_NL, Q5_K on both paths) and Mat::matvec on
  Q8_0 weights (both CPU paths).
- The GPU path (`ds_mat.cpp` `Dev`) compiles but has not run. `tests/test_score_glm5.py` is the acceptance test;
  the commands are in `HANDOFF.md`.
