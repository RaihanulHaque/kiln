#!/usr/bin/env bash
# Run kiln-bench on the host and (if a device is attached) on Android, then plot.
# Every model in models/ is benchmarked the way it is used: chat models generate text, decision
# models answer questions, embedding models embed texts and, when their projector file is there,
# an image and an audio clip from tests/media; speech models write down two clips from tests/media.
# usage: scripts/bench.sh [runs] [tokens]
#   KILN_BENCH_ONLY=<substring>     only models whose file name contains it
#   KILN_BENCH_SKIP_HOST=1          device only (do not overwrite the host's results)
#   KILN_BENCH_DEVICE_TAG=<name>    file prefix for the device (default: its model name, lower case)
#   KILN_BENCH_DEVICE_ENV="A=b"     environment for kiln-bench on the device, e.g. KILN_CPU_MASK=fc
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${1:-5}"; N="${2:-256}"
OUT="$ROOT/docs/bench"; mkdir -p "$OUT"
PY="${KILN_PYTHON:-python3}"   # needs matplotlib + pandas
CPU_NAME="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -1 || true)"
HOST_TAG="${KILN_BENCH_TAG:-$(printf '%s' "${CPU_NAME:-host}" | tr -d ' ()' | tr '[:upper:]' '[:lower:]' | sed 's/^apple//')}"
DEV=/data/local/tmp/kiln

# Model files to benchmark: GGUFs and whisper.cpp's .bin files (Whisper, Parakeet). A projector
# ("mmproj") is an add-on to an embedding or speech model, not a model.
models() {
  local m
  for m in "$ROOT"/models/*.gguf "$ROOT"/models/*.bin; do
    [ -f "$m" ] || continue
    case "$(basename "$m")" in *mmproj*) continue ;; esac
    # the filter ignores case: file names are a mix (Qwen3.5..., LiquidAI_LFM2.5..., gemma-3...)
    if printf '%s' "$(basename "$m")" | grep -qiF -- "${KILN_BENCH_ONLY:-}"; then echo "$m"; fi
  done
}

# Extra arguments for an embedding or speech model: "<projector file> <media file>..." (file names
# only; the caller adds the directories; "-" for a model whose encoder is built in). Prints
# nothing for every other model.
media_for() {
  case "$1" in
    embeddinggemma-2-*)
      [ -f "$ROOT/models/mmproj-embeddinggemma-2-Q8_0.gguf" ] && echo "mmproj-embeddinggemma-2-Q8_0.gguf red_circle.png speech_train.wav" ;;
    jina-embeddings-v5-omni-nano-retrieval-Q4_K_M)   # its vision projector takes images only
      [ -f "$ROOT/models/jina-embeddings-v5-omni-nano-retrieval-vision-mmproj-F16.gguf" ] && echo "jina-embeddings-v5-omni-nano-retrieval-vision-mmproj-F16.gguf red_circle.png" ;;
    Qwen3-ASR-0.6B-Q8_0)   # a speech model: without its projector it would be benchmarked as a chat model
      [ -f "$ROOT/models/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf" ] && echo "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf speech_train.wav speech_recipe.wav" ;;
    pocket-tts-en)          # text to speech: the first media file is the voice to imitate
      [ -f "$ROOT/models/mmproj-pocket-tts-en.gguf" ] && echo "mmproj-pocket-tts-en.gguf speech_train.wav" ;;
    ggml-*)                 # Whisper and Parakeet: the encoder is in the file
      echo "- speech_train.wav speech_recipe.wav" ;;
  esac
  return 0
}

# media_args <name> <models dir> <media dir>: the --mmproj / --media arguments for kiln-bench,
# one per line, so that paths with spaces survive (read them into an array).
media_args() {
  local models_dir="$2" media_dir="$3" spec f
  spec="$(media_for "$1")"
  [ -n "$spec" ] || return 0
  # shellcheck disable=SC2086
  set -- $spec                      # $1 = projector file or "-", the rest = media files (names have no spaces)
  [ "$1" = "-" ] || printf '%s\n' "--mmproj" "$models_dir/$1"
  shift
  for f in "$@"; do printf '%s\n' "--media" "$media_dir/$f"; done
}

# Wait until the phone lets its cores run at full clock again (it caps them while it is hot), so
# every model starts from the same state. Gives up after 6 minutes and says so.
wait_cool() {
  local t=0
  while [ "$t" -lt 360 ]; do
    if adb shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do [ "$(cat $p/scaling_max_freq)" = "$(cat $p/cpuinfo_max_freq)" ] || exit 1; done' < /dev/null; then
      echo "cool: cores uncapped after ${t}s"; return 0
    fi
    sleep 10; t=$((t + 10))
  done
  echo "warning: cores still capped after ${t}s; this run starts warm"
}

while IFS= read -r m; do
  [ -z "${KILN_BENCH_SKIP_HOST:-}" ] || break
  name="$(basename "$m")"; name="${name%.*}"
  echo "=== $HOST_TAG $name ==="
  extra=()
  while IFS= read -r a; do extra+=("$a"); done < <(media_args "$name" "$ROOT/models" "$ROOT/tests/media")
  "$ROOT/build/host/kiln-bench" -m "$m" -r "$RUNS" -n "$N" --cooldown 3 -o "$OUT/${HOST_TAG}_$name.csv" ${extra[@]+"${extra[@]}"}
done < <(models)

if adb get-state >/dev/null 2>&1; then
  tag="${KILN_BENCH_DEVICE_TAG:-$(adb shell getprop ro.product.model | tr -d '\r ' | tr '[:upper:]' '[:lower:]')}"
  adb shell mkdir -p "$DEV/media"
  adb push "$ROOT/build/android-arm64/kiln-bench" "$ROOT/build/android-arm64/libkiln.so" "$DEV/" >/dev/null
  adb push "$ROOT"/tests/media/. "$DEV/media/" >/dev/null
  while IFS= read -r m; do
    file="$(basename "$m")"; name="${file%.*}"
    adb shell "[ -f $DEV/$file ]" < /dev/null || adb push "$m" "$DEV/" < /dev/null
    extra="$(media_args "$name" "$DEV" "$DEV/media" | tr '\n' ' ')"   # device paths have no spaces
    if [ -n "$extra" ]; then
      proj="$(media_for "$name" | cut -d' ' -f1)"
      [ "$proj" = "-" ] || adb shell "[ -f $DEV/$proj ]" < /dev/null || adb push "$ROOT/models/$proj" "$DEV/" < /dev/null
    fi
    echo "=== $tag $name ==="
    wait_cool
    adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ${KILN_BENCH_DEVICE_ENV:-} ./kiln-bench -m $file -r $RUNS -n $N --cooldown 10 -o $DEV/bench.csv $extra" < /dev/null
    adb pull "$DEV/bench.csv" "$OUT/${tag}_$name.csv" >/dev/null
    adb pull "$DEV/bench.runs.csv" "$OUT/${tag}_$name.runs.csv" >/dev/null
    sleep 30   # skin temperature lags the cores; wait_cool handles the rest before the next model
  done < <(models)
fi

"$PY" "$ROOT/scripts/bench_plot.py" "$OUT"
