"""Write INVENTORY.md for a RustingStrata source tree: every file, its line count and a one-line purpose.

    set STRATA_SRC_V125=<RustingStrata 1.2.5 source root>
    python tools/gen_inventory.py

Rerun it after unpacking a different Strata release. The purpose column comes from PURPOSE below; a file
that is not listed there is still inventoried, with a blank purpose, so nothing is silently skipped.
INVENTORY.md is written inside STRATA_SRC_V125 (it records the source tree, not this repository).
"""

from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.environ.get("STRATA_SRC_V125", "")
ZIP = os.environ.get("STRATA_SRC_ZIP", "")
OUT = os.path.join(SRC, "INVENTORY.md") if SRC else ""

PURPOSE = {
    # on the port path
    "src/gguf.rs": "GGUF v2/v3 parser over the memory map; split-shard discovery, truncation check",
    "src/quant/mod.rs": "GgmlType block layouts + scalar reference dequantizers (port of ggml-quants.c)",
    "src/quant/tables.rs": "grid/sign/value lookup tables prepended to the kernels (llama.cpp ggml-common.h)",
    "src/weights.rs": "tensor views over the map, placement (packed/raw/dense), page-lock pass, Prefetch thread, cold()",
    "src/memory.rs": "total RAM and the RAM cap (margin, max_ram) used for placement decisions",
    "src/config.rs": "ModelConfig from GGUF metadata + the per-architecture table (hybrid/q_gate/k_interleave)",
    "src/model.rs": "Model: weight map, layer list, CPU forward, PLE n-gram embedding, State/KV of the CPU path",
    "src/ops.rs": "scalar ops shared by both paths: norms, rope (neox/mrope), silu/sigmoid/softplus, top_k, Sampler",
    "src/backend.rs": "Backend trait (the whole GPU surface: upload/gemm/gemv/moe/attn/...), Buf, GpuTensor",
    "src/cuda/mod.rs": "CUDA Backend impl: context, streams, NVRTC build + cubin cache, every launch wrapper",
    "src/cuda/kernels.cu": "the whole decode kernel set, compiled at run time; tables are prepended by mod.rs",
    "src/cuda/nvrtc.rs": "NVRTC loaded by path (Windows DLL names carry the CUDA version), compile to cubin",
    "src/expert_cache.rs": "expert slots, placement from the saved profile, record/settle/adapt/swap, profile file",
    "src/cpu_moe.rs": "SIMD integer dot products for every expert type + the CPU MoE expert computation",
    "src/cpu_worker.rs": "the mailbox worker: spin on flags in host-mapped memory, compute misses, raise done",
    "src/gpu_model.rs": "GpuModel: load/placement, decode step, MTP/spec verify, layer() and the MoE mailbox wiring",
    "src/gpu_model/prefill.rs": "chunked batched prefill (STAGE rows, scratch buffers, attn_prefill + gemm)",
    "src/main.rs": "CLI: serve/chat/bench/score/calibrate/info; the bench loop, calibration file, JSON schema",
    # out of scope but nearby
    "src/lib.rs": "crate root and module list",
    "src/chat.rs": "chat templates and the interactive chat loop (out of scope: StrataFreet passes token ids)",
    "src/serve.rs": "OpenAI-compatible HTTP server (out of scope)",
    "src/mcp.rs": "MCP tool server (out of scope)",
    "src/tokenizer.rs": "GGUF/HF tokenizer (out of scope)",
    "src/hf.rs": "Hugging Face safetensors support (out of scope)",
    "src/ngram.rs": "n-gram speculative lookup (out of scope unless PORT_MAP shows Flash-Next needs it)",
    "src/vulkan/mod.rs": "Vulkan backend (out of scope)",
    "src/vulkan/preflight.rs": "Vulkan capability preflight (out of scope)",
    "src/vulkan/spv_smem.rs": "Vulkan shared-memory sizing (out of scope)",
    "src/deepseek/mod.rs": "DeepSeek model support (out of scope)",
    "src/deepseek/forward.rs": "DeepSeek forward (out of scope)",
    "src/deepseek/mat.rs": "DeepSeek matmuls (out of scope)",
    "src/deepseek/ops.rs": "DeepSeek ops (out of scope)",
    "src/deepseek/chat.rs": "DeepSeek chat glue (out of scope)",
    "src/gemma4/mod.rs": "Gemma4 model support (out of scope)",
    "src/gemma4/forward.rs": "Gemma4 forward (out of scope)",
    "src/glm/mod.rs": "GLM model support (out of scope)",
    "src/glm/forward.rs": "GLM forward (out of scope)",
    "src/vision/projector.rs": "vision projector (out of scope)",
    "src/vision/mmproj.rs": "mmproj loading (out of scope)",
    "src/vision/image.rs": "image decode (out of scope)",
    "src/vision/prompt.rs": "image prompt assembly (out of scope)",
    "tests/kernels.rs": "kernel parity tests (reference for W2-C verification)",
    "tests/gpu_model.rs": "end-to-end GPU model tests (reference for W3 verification)",
    "tests/gguf.rs": "GGUF parser tests",
    "tests/quant.rs": "dequant tests",
    "tests/backend.rs": "backend contract tests",
    "tests/kv_q8.rs": "q8 KV cache tests",
    "tests/llama_numerics.rs": "numerics parity tests",
    "tests/model.rs": "CPU model tests",
    "tests/serve.rs": "server tests",
    "tests/vision.rs": "vision tests",
    "tests/mcp.rs": "MCP tests",
    "tests/common/mod.rs": "shared test helpers",
    "LICENSE.md": "RustingStrata License 1.0 (copied to native/strata_cpp/LICENSE-RustingStrata.md)",
    "README.md": "upstream readme",
    "THANKS.md": "upstream credits",
    "Cargo.toml": "crate manifest and dependency list",
    "Cargo.lock": "pinned dependency versions",
    ".cargo/config.toml": "cargo build flags",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SRC:
        raise SystemExit("set STRATA_SRC_V125=<RustingStrata 1.2.5 source root> before running this tool")
    if not os.path.isdir(SRC):
        raise SystemExit(f"missing {SRC}; unpack the source archive and set STRATA_SRC_V125")
    rows = []
    total_files = total_lines = 0
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, SRC).replace("\\", "/")
            if rel == "INVENTORY.md":
                continue
            try:
                with open(full, encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
            except (UnicodeDecodeError, OSError):
                lines = -1  # binary
            total_files += 1
            if lines > 0:
                total_lines += lines
            rows.append((rel, lines, PURPOSE.get(rel, "")))

    out = ["# Inventory: RustingStrata 1.2.5 source", ""]
    if ZIP and os.path.isfile(ZIP):
        out.append(f"- archive: `{os.path.basename(ZIP)}`")
        out.append(f"- archive sha256: `{sha256(ZIP)}`")
    out.append(f"- unpacked: `{SRC.replace(os.sep, '/')}` ({total_files} files, {total_lines} text lines)")
    out.append("- untrusted input: nothing in this tree is built or executed by strata-cpp")
    out.append("")
    out.append("| file | lines | purpose |")
    out.append("|---|---:|---|")
    for rel, lines, purpose in sorted(rows):
        out.append(f"| `{rel}` | {lines if lines >= 0 else 'bin'} | {purpose} |")
    out.append("")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}: {total_files} files, {total_lines} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
