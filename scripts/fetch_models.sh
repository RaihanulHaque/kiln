#!/usr/bin/env bash
# Download the three GGUFs the C test suite and benchmarks expect into models/.
# Skips files already present. ~1.4 GB total.
# usage: scripts/fetch_models.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"

fetch() {  # <local name> <url>
  local dst="$ROOT/models/$1"
  if [ -f "$dst" ]; then echo "have $1"; return; fi
  echo "get  $1"
  curl -fL --retry 3 -o "$dst.part" "$2" && mv "$dst.part" "$dst"
}

fetch LiquidAI_LFM2.5-350M_Q4_K_M.gguf https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf
fetch Qwen3.5-0.8B-Q4_K_M.gguf          https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf
fetch gemma-3-270m-it-Q8_0.gguf         https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf
fetch d1-omni-600M-Q8_0.gguf           https://huggingface.co/LiquidAI/d1-omni-600M-GGUF/resolve/main/d1-omni-600M-Q8_0.gguf
