// C ABI of strata-cpp (declarations for src/capi.cpp).
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
//
// Every function that can fail returns 0 on success and a nonzero code otherwise; the text of the last
// failure is returned by sc_last_error() (thread-local, valid until the next call on the same thread).
// A model handle (void*) is one loaded model with its KV cache and recurrent state; it is used from one
// thread at a time.
#pragma once

#include <stdint.h>

#ifdef _WIN32
#define SC_EXPORT extern "C" __declspec(dllexport)
#else
#define SC_EXPORT extern "C"
#endif

// Version string, e.g. "strata-cpp 0.0.1 (port of RustingStrata 1.2.5 by Vasyl Trefilov)".
SC_EXPORT const char* sc_version(void);

// Text of the last error on this thread, empty when there is none.
SC_EXPORT const char* sc_last_error(void);

// "major.minor" of the NVRTC library the loader found, or the loader's message when none loads.
// Needs no device.
SC_EXPORT const char* sc_nvrtc_version(void);

// Compiles a 3-line kernel (`sc_fill`, y[i] = 2*i) with NVRTC for the compute capability of device 0,
// loads the CUBIN with the driver API on the device's primary context, launches it with cuLaunchKernel
// and copies the n results into out. Returns 0 on success; out[i] == 2.0f * i for 0 <= i < n.
// Allocates 4 * n bytes of VRAM.
SC_EXPORT int sc_selftest_kernel(float* out, int n);

// Loads a GGUF model onto the GPU for decoding. `options_json` is a flat JSON object (or null for the defaults):
//   "ctx"            KV cache size in tokens (default 4096)
//   "kv"             "f16" (the only layout ported so far)
//   "threads"        CPU threads of the expert passes and the worker (default: the physical cores, as Strata's GPU default)
//   "vram_margin_mb" VRAM kept free after the load (default 512, as the --vram-margin default of Strata)
//   "seed"           sampler seed for sc_decode_sample (default 1)
//   "device"         CUDA device ordinal (default 0)
//   "chunk"          prefill chunk in tokens (0: no batched prefill; default: the calibration file's, else 4096)
//   "short"          prompts shorter than this go token by token (default: the calibration file's, else 32)
// The calibration file %LOCALAPPDATA%usting_strata\strata-<model stem>.json is read for "chunk", "mtp" and "short"
// (the defaults of "chunk" and "short" here; "mtp" is only reported, MTP is not ported). The load prints Strata's
// `plan:` and `expert traffic:` lines to stderr.
// GLM-5.3-Flash (general.architecture "glm5-next" or "glm5next") runs the port of RustingStrata's glm/forward.rs instead:
// the CPU reference forward with the dense matvecs and a VRAM expert cache on the GPU (src/glm.h). For it "ctx" is the
// context limit (when not given: STRATA_GLM_CTX, else the trained context; the latent cache grows with the text),
// "threads", "device" (the default of STRATA_DS_DEVICES), "seed" and "ram" apply; "kv", "vram_margin_mb", "chunk" and
// "short" do not (the VRAM budget is STRATA_DS_GPU_EXPERTS GB or STRATA_DS_GPU_FRACTION of free VRAM, default 0.9;
// the prefill chunk is STRATA_GLM_CHUNK, default 512), "pin" is refused, and the calibration file is not read.
// STRATA_GLM_CPU keeps everything on the CPU. sc_save_profile writes nothing for it; sc_stats reports its own keys.
// Returns the handle, or null with sc_last_error() set.
SC_EXPORT void* sc_load(const char* gguf_path, const char* options_json);

// Vocabulary size: the length of a logits row.
SC_EXPORT int sc_n_vocab(void* h);

// Feeds `tok` at the current position and returns the logits of the next token (n_vocab floats into
// `logits_out`; null to skip the copy).
SC_EXPORT int sc_decode(void* h, int32_t tok, float* logits_out);

// Feeds `tok` and picks the next token with the sampler given as a flat JSON object: "temp" (0 = greedy), "top_k"
// (0 = no cut), "top_p" (1 = no cut), "min_p" (0 = off; keeps candidates with p >= min_p * p_max), "presence_penalty"
// (0 = off; subtracted from the recent logits — a positive penalty lowers the logit of tokens that already appeared,
// as OpenAI), "repeat_penalty" (1 = off), "recent" (an array of the last token ids the penalties see, most recent
// last; the window stratafreet/engine.py uses is the last 64 of prompt + generated), and "seed" (re-seeds the
// handle's xorshift64 state; the state carries over between calls). With temp 0 and no penalties the argmax is taken
// on the device (no logits copy); with penalties it is taken on the host after the penalties (engine.py parity).
// The token goes to `out`.
SC_EXPORT int sc_decode_sample(void* h, int32_t tok, const char* sampler_json, int32_t* out);

// Runs the `n` prompt tokens `ids` from the current position through the batched prefill (prompts shorter than the
// "short" threshold go token by token), leaving the model ready to decode the next token. Returns 0.
SC_EXPORT int sc_prefill(void* h, const int32_t* ids, int n);

// `sc_prefill`, then the logits of the last position (n_vocab floats into `logits_out`, which must have room).
SC_EXPORT int sc_prefill_logits(void* h, const int32_t* ids, int n, float* logits_out);

// `sc_prefill`, then the next token picked from the last position's logits with the sampler of sc_decode_sample
// ("temp" 0 and no penalties: argmax, first index on ties; with penalties they apply before the argmax). The token
// goes to `out`.
SC_EXPORT int sc_prefill_sample(void* h, const int32_t* ids, int n, const char* sampler_json, int32_t* out);

// Writes the routing profile (%LOCALAPPDATA%usting_strata\profile-<arch>-<L>x<E>-<n>.f32) when this run recorded
// routing. sc_free does the same.
SC_EXPORT int sc_save_profile(void* h);

// Clears the KV cache and the recurrent state (keep_tokens must be 0). A partial reset (keep_tokens > 0) needs the
// DeltaNet state snapshots of the speculative path and is not ported yet (W3-E2/E3).
SC_EXPORT int sc_reset(void* h, int32_t keep_tokens);

// JSON of the model's state and counters (layers, positions, expert residency, hits and misses), valid until the
// next call with the same handle.
SC_EXPORT const char* sc_stats(void* h);

// Frees the handle and its device memory.
SC_EXPORT void sc_free(void* h);
