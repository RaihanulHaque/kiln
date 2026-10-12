#!/usr/bin/env bash
# Download the GGUFs the C test suite and benchmarks expect into models/.
# Skips files already present. ~1.7 GB total; --media adds the image / audio encoders
# for the kiln_embed_media tests (~0.9 GB more).
# usage: scripts/fetch_models.sh [--media]
set -euo pipefail
# a mistyped flag must not look like a successful setup that then skips the media tests
for arg in "$@"; do
  [ "$arg" = "--media" ] || { echo "unknown argument: $arg" >&2; echo "usage: scripts/fetch_models.sh [--media]" >&2; exit 2; }
done
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
fetch embeddinggemma-2-UD-Q4_K_XL.gguf  https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-UD-Q4_K_XL.gguf
fetch nomic-embed-text-v1.5.Q4_K_M.gguf https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf

if [ "${1:-}" = "--media" ]; then
  J=https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-retrieval-GGUF/resolve/main
  fetch mmproj-embeddinggemma-2-Q8_0.gguf https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/mmproj-Q8_0.gguf
  # Jina v5 omni is CC BY-NC 4.0: fine for testing, not for a commercial app
  fetch jina-embeddings-v5-omni-nano-retrieval-Q4_K_M.gguf            "$J/jina-embeddings-v5-omni-nano-retrieval-Q4_K_M.gguf"
  fetch jina-embeddings-v5-omni-nano-retrieval-vision-mmproj-F16.gguf "$J/jina-embeddings-v5-omni-nano-retrieval-vision-mmproj-F16.gguf"
fi
