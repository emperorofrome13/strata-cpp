"""Environment-variable conventions shared by the strata-cpp tests.

Every test that needs a model file, a Python oracle or an external executable reads it from an
environment variable. Nothing falls back to a path on the machine the port was developed on: when a
variable is unset the test prints a SKIP line and exits 0.

    SC_MODEL_14B        Qwen3.6-14B-A3B Q6_K GGUF (qwen35moe test model)
    SC_MODEL_FLASHNEXT  Qwen3.8-Flash-Next UD-Q3_K_XL first shard (qwen4exp test model)
    SC_MODEL_INKLING    Inkling-2-Mini IQ3_XXS GGUF (inkling2 test model)
    SC_GPT_OSS_GGUF     gpt-oss-20b MXFP4 GGUF (real MXFP4 expert rows)
    STRATAFREET_ROOT    StrataFreet checkout (Python reference oracle: quant, gguf_io, tokenizer)
    SC_RUSTINGSTRATA    rustingstrata.exe 1.2.5 (official RustingStrata oracle)
    SC_LLAMA_SERVER     llama.cpp llama-server.exe (Inkling-2 CPU oracle)
    SC_EVICT_GGUFS      semicolon-separated GGUF files read to evict the Windows file cache
    SC_BUILD_DIR        build directory to load the DLLs from ("build" by default)

Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
LICENSE-RustingStrata.md).
"""

from __future__ import annotations

import os
import sys

HINTS = {
    "SC_MODEL_14B": "<path to Qwen3.6-14B-A3B Q6_K gguf>",
    "SC_MODEL_FLASHNEXT": "<path to Qwen3.8-Flash-Next UD-Q3_K_XL first shard gguf>",
    "SC_MODEL_INKLING": "<path to Inkling-2-Mini IQ3_XXS gguf>",
    "SC_GPT_OSS_GGUF": "<path to gpt-oss-20b MXFP4 gguf>",
    "STRATAFREET_ROOT": "<StrataFreet checkout>",
    "SC_RUSTINGSTRATA": "<path to rustingstrata.exe v1.2.5>",
    "SC_LLAMA_SERVER": "<path to llama.cpp llama-server.exe>",
    "SC_EVICT_GGUFS": "<gguf path;gguf path;...>",
}

EXTRAS = {
    "STRATAFREET_ROOT": "reference oracle",
}


def get(name: str) -> str:
    """The variable's value, with surrounding quotes stripped (a PowerShell `$env:X="C:\\..."` copies them)."""
    return os.environ.get(name, "").strip().strip('"')


def skip_line(name: str) -> str:
    return f"SKIP: set {name}={HINTS.get(name, '<path>')}" + (
        f" ({EXTRAS[name]})" if name in EXTRAS else "")


def require(*names: str) -> bool:
    """True when every variable is set; otherwise prints one SKIP line per missing variable."""
    missing = [n for n in names if not get(n)]
    for n in missing:
        print(skip_line(n))
    return not missing


def skip(message: str) -> int:
    """Print a SKIP message in the same format and return the exit code to use."""
    print(f"SKIP: {message}")
    return 0


def add_stratafreet() -> str | None:
    """Put STRATAFREET_ROOT on sys.path. Returns None, or a SKIP message when it is unusable."""
    root = get("STRATAFREET_ROOT")
    if not root:
        return skip_line("STRATAFREET_ROOT")
    if not os.path.isdir(root):
        return f"SKIP: STRATAFREET_ROOT is not a directory: {root}"
    if root not in sys.path:
        sys.path.insert(0, root)
    return None


def build_dir(root: str) -> str:
    return os.path.join(root, os.environ.get("SC_BUILD_DIR", "build"))
