# Inkling-2-Mini native port — reference notes (W0)

Fetched 2026-11-21 from ggml-org/llama.cpp master (files in this folder):
- `inkling2.cpp` (458 lines) — plain arch: hparams, tensors, graph (build_inputs/build_layer/build_output, clip, mix, convolve, control)
- `inkling2-loop.cpp` (116 lines) — master's looped arch (⚠ different mechanism; see Loop structure below)
- `models.h` — class declarations (llama_model_inkling2 :2501, n_layer_phys :2508)
- `llama-model-master.cpp` — generic `.scale` tensor pass (:1920-1950)
- `llama-graph-master.cpp` — build_moe_ffn (situ :2603-2616, sigmoid gating :2378, weights norm :2476, w_s :1912), build_lora_mm w_s (:1912)
- `llama-model-loader-master.cpp` — loader

**PR #389 (unslothai) — the GGUF's actual implementation** (extracted to `commit8832860/pr389__*.diff`,
18 files, 2026-11-21): hparams loop fields + `loop_layer/loop_visit/loop_n_extra/n_layer_weights`,
`llama_model_loop_views` (per-visit host weight views), the model graph with global
`loop_res_query`/`loop_kv_res_query` {n_embd, 2*(core_end−core_start)} and per-core-block
`loop_{attn,k,v,mlp}_{down,gain,forget}` (BF16), the converter (`conversion/inkling2.py`: visit order =
prelude + core×passes + tail, `add_num_loops/loop_core_start/loop_core_end/loop_gate_rank`, native NVFP4
expert packing, checkpoint controller names `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight`),
gguf constants/tensor_mapping (KV names `loop.core_start/end/gate_rank`, tensor names `loop_res_query`,
`loop_kv_res_query`, `blk.%d.loop_{attn,k,v,mlp}_{down,gain,forget}`), arch.cpp/h, context.cpp,
model-saver, include/llama.h, tests. The prebuilt b12588-mix-c0c3de8 llama.dll pins this PR.

## Model facts (measured from UD-IQ3_XXS GGUF, 10.85 GiB)

- 26.07B total params; 128 experts × 24 blocks, 8 used/token; n_embd 2304, n_ff_exp 1152
- vocab 200058 (padded 201024); ctx 1048576 (max); `head_count` 24, `head_count_kv` 8, key/value_length 128 (GQA 3:1). **n_head·head_dim = 3072 ≠ n_embd 2304** — Q/K/V are wider than the model width; attn_output {3072, 2304} folds them back. Metadata arrays (`head_count`, `head_count_kv`, `sliding_window_pattern`) are visit-expanded: 48 entries, values per-visit equal to their phys layer's.
- eps 1e-6; softcap 30.0; **num_loops 3**, core [6, 18), gate_rank 64; attn_res block_size 6
- **48 logical slots = 24 phys + 2×12 loop slots** (block_count 48 in metadata; tensors blk.0..23)
- SWA pattern: 5×1 (True×5, False×1, repeating) → 40 SWA layers, 8 full; n_swa 1024; key/value_length_swa 128
- rel: length 16, extent 1024, clip 8.0; qk_clip 22.627416 (= 16·√2); log_scale floor 8192, alpha 0.1
- situ: beta 4.0, linear_beta 25.0; expert_weights_scale 8.0; weights_norm_eps 1e-20; router f32; gating SIGMOID
- shortconv l_cache 4; rope.dimension_count 0 (no RoPE)

## Tensor inventory (734 tensors, per-kind quant types in UD-IQ3_XXS)

| tensor | quant | notes |
|---|---|---|
| token_embd | Q4_K | padded vocab 201024 |
| output | Q5_K | padded |
| attn_q/k/v | Q8_0/Q6_K/Q5_K/Q4_K mix | |
| attn_output | Q8_0/Q6_K | |
| attn_gate | Q8_0/Q6_K/Q5_K | |
| attn_rel | Q8_0/Q6_K/Q5_K | {n_embd, 16×24} = {2304, 384} |
| attn_rel_bias | F32 | {16, 1024} |
| attn_q_clip / rel_clip | F32 | {128,24} / {16,24} |
| attn_sink_proj | BF16 | {n_embd, 24} |
| attn_k_conv / v_conv / post_conv / ffn_post_conv | F32 | {4, out} |
| attn_k_conv_proj / v_conv_proj | BF16 | {n_embd, 4×8} — full-attn layers only (4) |
| attn_q_inp_norm / k_inp_norm / v_inp_norm | F32 | full-attn layers only |
| attn_norm | F32 | SWA layers only (20) |
| attn_res_query | F32 | {n_embd, 3} full-attn (il>0), {n_embd, 1} SWA |
| ffn_res_query / ffn_norm / ffn_gate_inp | F32 | |
| ffn_gate_up_exps | IQ2_XXS×1, IQ2_S×9, IQ2_XS×1, IQ3_XXS×13 | {2304, 2304, 128} |
| ffn_down_exps | **NVFP4 (40) ×20, IQ4_NL ×4** | {1152, 2304, 128} |
| ffn_down_exps.scale / ffn_gate_up_exps.scale | F32 {128} | blocks 0..19 (20 blocks) |
| exp_probs_b | F32 {128} | 24 blocks |
| loop_res_query / loop_kv_res_query | F32 | **global** {n_embd, 24} each (2 rows per core block) |
| loop_{attn,k,v,mlp}_{down,gain,forget} | **BF16** | 12 blocks each; down {4608, 64}, gain/forget {64, 2304} |

**Loop tensors — GGUF ground truth (verified by parsing UD-Q8_K_XL directly, 2026-11-21).** The GGUF
follows **unslothai PR #389**, not master's inkling2-loop.cpp (master's names differ; its loader would
not find these tensors):
- **Global** `loop_res_query.weight` and `loop_kv_res_query.weight`, both **F32 {2304, 24}** — n_query =
  2*(core_end − core_start) = 2×12, NOT per-block `{n_embd, 4}` msg_query.
- Per core block (blk.6..17): `loop_{attn,k,v,mlp}_{down,gain,forget}.weight` — **BF16** (type 30), not
  F32. down {2*n_embd, 64} = {4608, 64}; gain/forget {64, 2304}.
- KV arrays are per-visit: `head_count`/`head_count_kv` 48×24, `sliding_window_pattern` 48 bools.

**Master's inkling2-loop.cpp is a different (newer) mechanism** — per-core-block `loop_msg_query`
{n_embd, 4}, tensor names `loop_{q,k,v,ffn}_ctrl_{down,gain,forget}`, KVs `loop.start/end/count/gate_rank/varying`,
slot layout prelude/core/tail/loop-slots(24-47). The GGUF matches none of that: the port must follow
**PR #389** (`pr389__src__models__inkling2.cpp.diff` etc., saved in commit8832860/).

## Exact forward math (from inkling2.cpp)

**clip(cur, scale, threshold):** per row: cur *= scale (optional); norm = sqrt(Σ cur²) + 1e-6; factor = min(threshold/norm, 1) clamped [0,1]; return cur * factor.
- Q: clip by attn_q_clip {128,24}, f_qk_clip; K: clip by nullptr (no per-channel scale), f_qk_clip; R: clip by attn_rel_clip, f_relative_clip.

**mix(blocks, cur, query):** states = concat([block outputs] + [cur], dim 1) → rms_norm(1e-6); queries = query repeated to n_tokens; scores = matmul(states, queries) → softmax(1/sqrt(n_embd)); out = matmul(states^T, scores). Used for attn_res_query (block residuals) and ffn_res_query (before MoE) and output_res_query.

**convolve(cur, weight, dynamic, state, history, offset):** causal shortconv over channels; x = concat(past (n_conv-1 = 3 channels from recurrent state at offset), cur); out = ssm_conv(x, weight); dynamic (full-attn K/V only): for each lag 0..3, window (n_seq_tokens channels) * tap {n_heads} per head → out += term; final: (cur + out) scaled by 1/sqrt((dynamic?2:1)*n_conv + 1) = 1/3 for static, 1/sqrt(9)=1/3 for dynamic too (2*4+1=9 → same).

**control(cur, message, loop, ctrl):** gate = concat(rms_norm(cur,1e-6), rms_norm(message,1e-6)) [2*n_embd]; gate = ctrl_down[ctrl]·gate [64]; gain = tanh(ctrl_gain[ctrl]·gate) [n_embd]; forget = tanh(ctrl_forget[ctrl]·gate); return (1 - forget)*cur + gain*message = cur - forget*cur + gain*message.

**build_inputs:** tau = 1 + alpha*log(max(1, (pos+1)/floor)) cast f32, shape {1,1,n_tokens} — multiplies Q and R on full-attention layers only. cur = tok_embd (padded) → rms_norm(tok_norm).

**build_layer(il, ip, cur, blocks, messages, loop):**
1. inpSA = cur.
2. attn inputs: SWA layers mix 1 (blocks + cur → attn_res_query {n_embd,1}); full-attn layers mix 3 (→ {n_embd,3}, one per Q/K/V input).
3. If messages (loop pass): inputs[j] = control(inputs[j], message[LOOP_CTRL_Q+j], loop, ...) for j=0,1,2 (Q/K/V); before norms.
4. Norms: SWA+messages → attn_norm each of 3; SWA → attn_norm on inputs[0], 1=2=0; full-attn → q_inp_norm/k_inp_norm/v_inp_norm.
5. Q = wq·inputs[0] → reshape {128,48,1} → clip(attn_q_clip, 22.627416)
   K = wk·inputs[1] → convolve(attn_k_conv, dynamic=attn_k_conv_proj·inputs[1] (full-attn), state, offset 0) → reshape {128,8,1} → clip(nullptr, 22.627416)
   V = wv·inputs[2] → convolve(attn_v_conv, dynamic=attn_v_conv_proj·inputs[2], offset n_embd_k_gqa=1024) → reshape {128,8,1} → rms_norm(eps)
   R = attn_rel·inputs[0] → {16,24} → clip(attn_rel_clip, 8.0)
6. Full-attn: Q *= tau, R *= tau.
7. sinks = attn_sink_proj·inputs[0] ({n_embd,24} → {1,24,n_tokens} broadcast).
8. rel bias: table = rel_bias (padded {16,1024+1}) · R → {n_rel, n_tokens, 48}? (permute 1,0,2) → table - sinks → get_rows by rel_pos (distance table, iswa: swa layers use self_rel_pos_swa, full-attn self_rel_pos) → kq_b {n_head, n_kv, n_tokens}.
9. Attention: build_attn(kq_b as bias, sinks as fused-zero sink: exp(s)/(exp(t)+Σexp(s)) with zero sink), scale 1/sqrt(128).
10. attn out: cur *= sigmoid(wqkv_gate·inputs[0]); → wo → convolve(attn_post_conv, static, offset 2*n_embd_k_gqa=2048).
11. Residuals: if ip % block_size == 0 → blocks.push_back(inpSA) (new residual stream); else cur += inpSA.
12. FFN: cur = mix(blocks, cur, ffn_res_query) → [if messages] control(cur, message(LOOP_CTRL_FFN), loop, ctrl) → ffn_norm → build_moe_ffn(situ, sigmoid gating, exp_probs_b bias on selection, weights_scale 8.0, norm eps 1e-20, fused gate_up) → convolve(ffn_post_conv, static, offset 2048).
13. Output: cur = mix(blocks, cur, output_res_query) → output_norm → output (padded vocab) → softcap 30.0 → strip padded vocab.

**MoE (build_moe_ffn):** logits = ffn_gate_inp·cur {128}; probs = sigmoid(logits); selection_probs = probs + exp_probs_b; top-k 8 by selection_probs; weights = probs[selected] / (Σ + 1e-20); weights *= 8.0 (expert_weights_scale); situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb); down = down_exps·situ; out = Σ weights*down (per-expert .scale multiplied after mul_mat_id per expert row).

**Shortconv state:** channels = 2*n_embd_k_gqa + 2*n_embd = 2*1024 + 2*2304 = 6656 max; layout: [0, n_embd_k_gqa) K conv state, [n_embd_k_gqa, 2*n_embd_k_gqa) V conv state, [2*n_embd_k_gqa, 2*n_embd_k_gqa + n_embd) attn post conv state, last n_embd ffn post conv state. l_cache 4 → 3 history channels per layer.

## Loop structure — PR #389 (GGUF ground truth)

**Slot layout (PR #389, sequential il 0..47):** prelude 0-5, core pass 0 6-17, **loop slots 18-41**
(pass 1 = 18-29, pass 2 = 30-41, both over phys 6-17), tail 42-47. Master's layout (loop slots after
the tail, 24-47) is different — but the computation order is the same (prelude → core×3 → tail).

- `loop_layer(il)`: il < core_start → il; il < core_start + n_pass*n_core → core_start + (il−core_start)%n_core;
  else il − loop_n_extra(). Here: slots 0-17 → phys il, 18-29 → phys il−12, 30-41 → phys il−24, 42-47 → phys il−24.
- `loop_n_extra()` = (n_pass−1)*n_core = 24; `n_layer_weights()` = n_layer_all − 24 = 24 (the unit of n_gpu_layers).
- pass of a core slot = (il − core_start)/n_core; reset at ip == core_start && pass > 0:
  `bank = blocks[prelude_blocks.size()+1:]`, `bank_last = cur`, then `blocks = prelude_blocks; cur = prelude_cur`.
- **Messages (pass > 0):** `q_attn` = 2-row view of global `loop_res_query` at row offset
  `2*(ip − core_start)`; `q_kv` same on `loop_kv_res_query` (both {n_embd, 2}). `m_attn = mix(bank,
  bank_last, q_attn)`, `m_kv = mix(bank, bank_last, q_kv)` (each {n_embd, 2}). Mapping: col 0 of m_attn
  → messages[0] (Q), col 1 → messages[3] (FFN); col 0 of m_kv → messages[1] (K), col 1 → messages[2] (V).
- Passes 1,2: SWA layers get messages too (inputs[j] = control(inputs[j], messages[j], …) for j=0,1,2
  before the norms; SWA layers then rms_norm each of the 3 separately). K/V convolve runs per slot
  (own KV + conv state per slot).
- **llama_model_loop_views (PR #389 llama-model.cpp):** later visits get *views* of the host-resident
  weights instead of sharing the tensor — the backend scheduler keys a host weight's device copy by
  tensor, so a shared tensor would pin all looped layers' GPU copies at once; a per-visit view gets its
  own copy, freed after that pass. Views keep name + buffer. strata-cpp ram-lock path needs the same
  trick: per-visit weight views in host memory.

**mix() exact math (inkling2.cpp:185-198):** states = concat([each block] + [cur], dim 1) as {n_embd, 1,
n_tokens} → rms_norm(1e-6); queries = repeat(query, n_tokens); scores = mul_mat(states, queries) →
soft_max_ext(scale 1/sqrt(n_embd), bias 0); out = mul_mat(cont(transpose(states)), scores).

**convolve() exact math (inkling2.cpp:208-245):** past = view_3d(history, n_conv−1, n_channels, n_seqs,
…, offset*nb[1]); x = concat(past, transpose(reshape_3d(cur, n_channels, n_seq_tokens, n_seqs)), dim 0);
store conv state (last n_conv−1 inputs); out = ssm_conv(x, weight) → reshape {n_channels, n_tokens};
dynamic (full-attn K/V): reshape_3d(dynamic, n_conv, n_heads, n_tokens); for lag 0..n_conv−1: tap =
view_3d(dynamic, 1, n_heads, n_tokens, lag*nb[0]); window = view_3d(x, n_seq_tokens, n_channels, n_seqs,
(n_conv−1−lag)*nb[0]) → cont(transpose) → reshape {n_channels/n_heads, n_heads, n_tokens}; out +=
reshape_2d(mul(window, tap), n_channels, n_tokens). Final: scale(cur + out, 1/sqrt((dynamic?2:1)*n_conv + 1)).

**Full-attn physical layers (pattern 0s at slots 5, 11, 17, 23, 29, 35, 41, 47):** phys **5, 11, 17, 23**
(slot 5 → phys 5, 11 → 11, 17 → 17, 23 → 11, 29 → 17, 35 → 11, 41 → 17, 47 → 23 under the PR #389
layout). blk.5/11/17/23 have the 3-column attn_res_query + k/v_conv_proj + q/k/v_inp_norm tensors;
blk.23's earlier "[2304]" reading was a parse error — it is {2304, 3}.

**Loop KV/conv state:** each of the 48 slots keeps its own KV cache + conv state. strata-cpp GpuModel::load
allocates KV per layer → need 48-slot layer array with physical weight sharing (weights per 24, state per 48).

## NVFP4 (down weights, 20 blocks; IQ4_NL on 4)

- block_nvfp4: 64 elements = 4 sub-blocks × 16; d[4] UE4M3 scales (1 per sub-block) + qs[32] packed E2M1 nibbles.
- dequant: v = kvalues_mxfp4[nibble] * ue4m3_to_fp32(d[s]); layout: yb[j] = v0*d, yb[j+8] = v1*d for j in 0..7 (first half / second half of sub-block).
- ue4m3_to_fp32: x==0 or 0x7F → 0; exp = (x>>3)&15, man = x&7; exp==0 → man*2^-9; else (1+man/8)*2^(exp-7); ×0.5.
- fp32_to_ue4m3: x<=0 → 0; clamp 448; exp = fp32_exp+7; subnormal man = round(x*512), else exp field; (see ggml-impl.h:530-560 for the full rounding).
- E2M1 values = kvalues_mxfp4 (16-entry table: ±0.5, ±1.5, ±2.5, ±3.5, ±4.5, ±6, ±7.5, ±10.5? — copy exact table from ggml).
- Per-expert F32 scale {128} multiplies the down output after mul_mat_id (per expert).
- llama.cpp uses GGML_PREC_BF16 accumulation for NVFP4 matmuls.

## Oracle

- unsloth prebuilt `b12588-mix-c0c3de8` (llama-server) — used for the bench runs. Scriptable: POST /completion with prompt + logprobs, or llama-cli greedy.
- Parity gate (from QFN port): same argmax ≥ 99% of positions, top-1 logprob within 0.05; greedy decode matches.

## strata-cpp integration points (verified this session)

- IQ3_XXS already in kernels.cu QTYPES (type 18, deq8<18>, QI<18>, MoE GEMV). IQ2_XXS/IQ2_S/IQ2_XS/IQ4_NL also present.
- **NVFP4 NOT in QTYPES** — new dequant path needed (type 40; E2M1 nibbles × UE4M3 sub-block scales).
- SWA supported: config.cpp:164-174 (sliding_window_pattern + sliding_window), gpu_model.cpp:774 (win passed to attn kernel).
- config.cpp reads `{arch}.{key}` — all inkling2.* keys parse with the existing opt/req helpers.
- Sinks supported (gpu_model.cpp:771-774, sink ptr passed to attn kernel) — but inkling2's is a projection (attn_sink_proj), not a bias vector.
- Ram lock: strata-cpp partial-residency lock exists (83.31 GB copy verified for Flash-Next); target quant Q8_0 ≈ 24.3 GiB.

## rel_pos semantics (confirmed from llama-kv-cache.cpp:2010)

`rel_pos[i*n_kv + j] = distance = pos_query[i] - pos_kv[j]`, clamped: `distance >= 0 && distance < extent ? distance : extent`. extent = n_relative_extent (1024) for full-attn layers, n_swa (1024) for SWA layers (both 1024 here). Empty cells → extent. For a decode step at position p: rel_pos[j] = min(p - pos[j], 1024) — the gather index into the padded bias table {16, 1025} (padded dim 1).

Attention bias: `kq_b[h, j] = Σ_r bias[r, rel_pos[j]] · R[r, h] − sink[h, t]`, added to the scores before softmax; the kernel takes a zero sink (adds exp(0)=1 to the denominator, no value vector) — the sink logit t is already folded into every bias entry as −t. strata-cpp's `attn_decode_q` sink pointer + zero values should give this directly IF the kernel adds exp(0) for a zero sink; otherwise pass a dummy −inf… verify in W2.

## strata-cpp design (W1)

- `ModelConfig` gains: `n_layer_phys`, `loop_start`, `loop_end`, `n_loop_passes`, `loop_gate_rank`, `situ_beta`, `situ_linear_beta`, `n_rel`, `n_rel_extent`, `f_rel_clip`, `f_qk_clip`, `n_log_scale_floor`, `f_log_scale_alpha`, `n_shortconv`, `f_final_softcap`, `n_attn_res_block`, `n_vocab_padded`, `n_vocab_unpadded`, `n_sconv_channels` (= 2*n_embd_k_gqa + 2*n_embd). `n_layer` = the slot count (48) — every per-slot allocation (KV, conv state) uses n_layer; weight access maps slot→phys via `layer_phys()`.
- `Arch` gains `kind` (Qwen | Flint | Inkling2) — gpu_model dispatches on it; inkling2 is not a qwen variant.
- `Weights::layer(il, suffix)`: for inkling2, `il >= n_layer_phys` remaps via `loop_layer()` (PR #389 layout: slots 18-29 → phys il−12, 30-47 → phys il−24) before the tensor lookup (blk.N names only exist for N < 24).
- `kExpertMats`: for inkling2 the fused `ffn_gate_up_exps.weight` replaces gate/up: the cache treats it as ONE mat {n_embd, 2*n_ff, n_expert}; the MoE GEMV emits {2*n_ff} rows, the activation splits them. ggml's mul_mat_id on {n_embd, 2*n_ff, n_expert} gives {2*n_ff, n_tokens} where rows 0..n_ff-1 = gate, n_ff..2*n_ff-1 = up (dim 1 of the weight). Verify in W2 with a parity check.
- Per-expert `.scale` tensors: F32 {n_expert} per block, applied after each expert GEMV (gate_up: before the activation; down: after) — gathered by the routed ids, exactly llama.cpp's w_s. The CPU worker applies the same scales. New scratch rows: `gs` (gate_up scale, k floats), `ds` (down scale, k floats).
- NVFP4 = Ty 40 in strata-cpp's Ty enum (quant.h), `block_bytes` 36 per 64 elems, `dequant_row` following kvalues_mxfp4[nibble] * ue4m3_to_fp32(scale). kernels.cu: `deq8<40>` (SBB = 4 blocks × 36 = 144 bytes per 256 elems), `QI<40>` (no MIN term — NVFP4 has no offset), `moe_gemv_nvfp4` via the existing ITYPES macro X(40, nvfp4, 144). BF16 accumulate (GGML_PREC_BF16): strata-cpp's MoE GEMV accumulates f32 — the reference used bf16 accumulation for NVFP4; if parity drifts, add a bf16-accumulate variant; measure first.
- GpuLayer (inkling2): q, k, v, o, gate(attn_gate), rel {2304, 384}, rel_bias {16, 1025 padded}, q_clip {128, 24}, rel_clip {16, 24}, sink_proj {2304, 24}, k_conv {4, 1024}, v_conv {4, 1024}, post_conv {4, 2304}, ffn_post_conv {4, 2304}, q_inp_norm/k_inp_norm/v_inp_norm (full-attn only), attn_norm (SWA only), res_q {2304, 1 or 3}, ffn_res_q {2304}, router, exp_probs_b {128}. Buf conv_state (3 * 6656 floats per slot). Slots 18..47 share the weight GpuTensors of their physical layer (two/three GpuLayer entries, same addresses) but own their KV + conv state. GpuLayer also gains the 12 loop-gate tensors (loop_down {4608, 64}, loop_gain/forget {64, 2304} — BF16) for the 12 core slots, plus the 2 global query tensors on the model.
- Expert cache: keyed by PHYSICAL layer (24 × 128 experts = 3072 entries). Pool bytes at Q8_K_XL: gate_up 2304*2304*0.5625 = 2.98 MB + down 1152*2304*0.5625 = 1.49 MB = 4.47 MB/expert × 128 = 572 MB per layer × 24 = 13.7 GB — exceeds the ~8 GB budget; the cache holds what fits, the rest goes to the CPU worker (as Flash-Next does). Decode maps slot→phys for the cache lookups (`cache_->layer(phys)`); routing tables are per physical layer (the two passes of one core block read the same table — the routed ids differ per pass, but the resident set is shared).
- Mailbox: per SLOT (48 mailboxes — each pass routes independently); `Mailbox::make(8, 2304)` stride × 48. `moe_route` reads the phys table, writes the slot's ids. `CpuWorker::create` over 48 moe slot indexes; the expert lookup inside cpu_moe remaps slot→phys for the weight pointers.
- Decode scratch (per token): states bank for mix() — block outputs of the current res-block group (up to 6 rows of n_embd) + cur = 7; messages bank {4, n_embd}; m_attn/m_kv {n_embd, 2} each (pass > 0); tau {1}; per-layer: qg {6144}, q {6144}, kraw/k/v {1024}, rel {384}, sink {24}, table {1025, 24} f32, kq_b {24, n_kv} f32, att {6144}, conv out rows, ctrl gate {64}, gain/forget {2304} ×4 ctrl, ge/ue {1152} (fused: {2304}), gs/ds {8}.

## W1 implemented (2026-11-21, acceptance PASSED)

Loader + config complete and verified on UD-Q8_K_XL (sc_load "ram": "lock", ctx 512):
`plan: weights+state+scratch 3698 MB (KV 96 MB for 512 tokens), expert slots 10807 MB, resident experts 2530/3072`,
`n_vocab=201024`, arch inkling2. Config dump matches the GGUF ground truth field for field (48 slots, 24/8 heads,
GQA 3:1, 40 SWA + 8 full, sigmoid router, scale 8.0, loop [6,18) × 3 passes, gate_rank 64).

Changes (all backed up in `backup/20261121_inkling_port/`):
- `quant.h/cpp`: `Ty::NVFP4 = 40`; `ue4m3_half` (ggml_ue4m3_to_fp32, ×0.5 built in); `nvfp4` block decoder
  (4 sub-blocks × 16, yb[j]=v0*d, yb[j+8]=v1*d); ty_from_u32(40), block_elems 64, block_bytes 36, ty_name.
- `config.h/cpp`: `Arch::Kind {Qwen, Flint, Inkling2}` + `ModelConfig::kind`; inkling2 fields (n_loop_passes,
  loop_start/end, loop_gate_rank, n_layer_phys, situ betas, n_rel/extent, f_rel_clip/f_qk_clip, log-scale
  floor/alpha, n_shortconv, f_final_softcap, n_attn_res_block, n_vocab_padded/unpadded); `req_arr_uniform`
  (head_count/head_count_kv are per-visit arrays); `loop_layer_of` = PR #389's loop_layer; inkling2 branch in
  from_gguf (rope_layer all false, router_sigmoid always true, loop validation, per-visit head consistency).
- `weights.cpp`: `Weights::layer` remaps every slot through `loop_layer_of` when looped (identity below
  loop_start; slots 18..41 remap to the core 6..17 — the remap is NOT gated on il >= n_layer_phys).
- `expert_cache.h/cpp`: fused gate_up detection (`ffn_gate_up_exps.weight` present, separate gate/up absent);
  `LayerExperts` gains mats[4] + n_mats + fused_gu + has_scales; the fused slot = [0] gate_up {n_embd, 2*n_ff},
  [1] down, [2]/[3] per-expert f32 scales (4 bytes each) when the file has them; `expert_slice` helper (the
  scales slice one f32 by hand — Tensor::expert would divide rows by n_expert); profile keyed by n_layer_phys.
- `cpu_moe.h/cpp`: `ExpertLayer` gains fused_gu + gate_up + gate_scale/down_scale; load_expert_layer fused
  branch; experts()/experts_multi() slice per-expert gate/up views from the fused tensor and multiply the
  dot outputs by the per-expert f32 scales (gate/up share one scale; down has its own). NVFP4 has no CPU
  integer kernel yet — batch_rows falls back to dequant + f32 dot (W2 adds the kernel).
- `cpu_worker.cpp`: the prefetch lists gate_up in place of gate+up for fused layers.
- `gpu_model.h/cpp`: `Inkling2Block` (rel, rel_bias, q_clip, rel_clip, sink_proj, k/v_conv + conv_proj,
  post_conv, ffn_post_conv, inp norms, res_q, ffn_res_q, exp_probs_b, loop gates ×4 controllers, conv_state);
  GpuModel gains loop_res_query_/loop_kv_res_query_ + tok_norm_/output_res_query_; the load branch per slot
  (SWA norm into attn_norm, loop gates only on core slots, per-slot KV + conv state); `lt` routes through
  Weights::layer (the remap); plan_bytes inkling2 branch; ExpertCache created over [0, n_layer_phys); the
  mailbox and CPU worker over all 48 slots; after_experts records routing on the slot's phys block.
- `build_cpu.ninja`: +advapi32.lib (the ram-lock privilege calls; the main build already had it).

Deviations from the recorded design (found while implementing, verified against the GGUF):
- The remap condition is `is_looped()` unconditional (loop_layer_of is identity below loop_start), NOT
  `il >= n_layer_phys` — slots 18..41 are loop passes BELOW n_layer_phys = 24 and must remap.
- attn_norm is missing on the full-attention blocks (5, 11, 17, 23 — the attn_res block pattern); they carry
  attn_{q,k,v}_inp_norm + attn_{k,v}_conv_proj instead. attn_res_query is {n_embd, 3} on the full-attn
  blocks and {n_embd} on the SWA blocks (absent on block 0).
- The experts' fused gate_up: rows 0..n_ff-1 = gate, n_ff..2*n_ff-1 = up per expert (dim 1 of the weight).

W2 next: the decode forward (mix/convolve/control on the GPU, rel-bias attention kernel, shortconv, situ,
loop gating, NVFP4 MoE GEMV) + parity vs the unsloth prebuilt.

## W1 audit fixes (2026-11-21, same session as the acceptance)

The audit pass over the W1 diff found and fixed four real issues (backup of the pre-fix files:
`backup/20261121_audit_w1/`):

1. **The expert cache read the WRONG blocks' experts** (latent until W2, then silently wrong output).
   `ExpertCache::create`/`place` are keyed by PHYSICAL block and called `Weights::layer(il, ..)`, whose
   looped-slot remap sends `il = 18..23` to `loop_layer_of(18..23) = 6..11..17` — so cache layers 18..23
   held blk.6..17's experts and blk.18..23's own experts were never placed. The GGUF proves they differ:
   blk.6 and blk.18 both carry `ffn_gate_up_exps.weight` (382,205,952 bytes each, data offsets
   19454574080 vs 7303951872). Fix: `Weights::layer_phys` (the lookup the remap lands on, no remap) and
   every `m.w.layer` in expert_cache.cpp became `m.w.layer_phys`. The CPU worker keeps `layer` (it is
   addressed by SLOT and the remap there is the point).
2. **The load uploaded duplicate weights for the 24 looped slots** (~1.34 GiB wasted VRAM, contradicting
   the recorded design "two/three GpuLayer entries, same addresses"). Measured: the GGUF's non-expert
   block weights are 1318.4 MiB for the 24 physical blocks but 2690.8 MiB summed over the 48 slots
   through the remap; the acceptance's `plan: weights+state+scratch 3698 MB` matches the duplicated
   sum (+ output 883 MiB + KV 96 + states/scratch), not the physical one. The waste also shrank the
   expert-cache budget (free VRAM minus margin). Fix: the first slot to play each PHYSICAL block
   uploads its weights and every later slot of that block copies the GpuTensors (`phys_layer` map);
   only the KV cache and the shortconv state are the slot's own. `plan_bytes` mirrors it exactly.
   **The sharing key must be the physical block, not `phys != il`:** no slot has `il == phys` in
   18..23 (slot 42 plays block 18), so the first draft — copy when `phys != il` — left blocks 18..23
   unuploaded and made slots 42..47 run blk.6..11's weights. The plan number caught it: the draft
   loaded at 2010 MB, exactly the Python replication of the broken tensor set (2009.0 MiB), while the
   corrected load measures 2326.0 MB against the replicated 2325.5 MiB (Δ = the 2 MiB arena rounding).
3. **`reset()` left the shortconv history stale** (the inkling2 branch memset only K/V). The conv
   history is recurrent state; a reset that skips it poisons the next conversation. Fix: reset also
   memsets `ink.conv_state` per slot.
4. **A decode on inkling2 fell into the generic qwen path** — its first op reads the empty `q_norm`
   (null device pointer), and past that it would produce silent garbage. Fix: `GpuModel::layer` rejects
   inkling2 with "inkling2 decode forward is not ported yet (W2)", and load forces `chunk = 0` for
   inkling2 (no batched prefill yet, W3) so prefill routes through step() and hits the same clear error.
   Also added `config`: inkling2 `vocab_size` must equal the token_embd row count (n_vocab drives the
   logits rows, n_vocab_padded the mask).

Verification (sc_load ram-lock reruns after the fixes, UD-Q8_K_XL ctx 512):
`plan: weights+state+scratch 2326 MB (KV 96 MB for 512 tokens), expert slots 12174 MB, resident experts
1999/3072`, load OK, n_vocab=201024 — the 2326 MB matches the independent Python replication of
plan_bytes (2325.5 MiB). `STRATA_LOAD_TRACE` pool lines prove the phys keying: pool 2 holds 15.19 MiB
slots (blk.20..22's BF16 experts) where the pre-fix cache would have shown blk.6's 4.27 MiB NVFP4
tensors there. Resident experts moved 2530 → 1999 because the cache now accounts the BF16 blocks at
their true 3.4x size and the reclaimed weight VRAM goes to the pool budget, not to fake NVFP4 slots.
pytest: 129 passed, 3 skipped (the documented baseline).

Completeness round (same day): `sci_model_config_json` emitted these 18 fields for every architecture,
which broke `tests/test_weights.py` ("config keys differ" — its oracle is a port of config.rs and has
no loop keys for the plain archs). The block is now emitted only when `c.is_looped()`; verified 0 loop
keys on the 14B qwen config JSON, all 18 on the UD-Q8_K_XL, and test_weights.py passes again.
Evidence: `build/ram_test/config_keys_console.log`; the acceptance console log is persisted at
`build/ram_test/sc_load_inkling_console.log`. Note: the files in `tests/` are script-style acceptance
tests (`python tests/test_<name>.py`, per each file's docstring), not a pytest or unittest suite.
