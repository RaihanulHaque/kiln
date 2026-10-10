#!/usr/bin/env bash
# Run kiln-bench on the host and (if a device is attached) on Android, then plot.
# usage: scripts/bench.sh [runs] [tokens]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${1:-5}"; N="${2:-256}"
OUT="$ROOT/docs/bench"; mkdir -p "$OUT"
PY="${KILN_PYTHON:-$HOME/miniconda3/envs/ML/bin/python}"
HOST_TAG="$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr -d ' ' | tr '[:upper:]' '[:lower:]' || echo host)"
DEV=/data/local/tmp/kiln

for m in "$ROOT"/models/*.gguf; do
  name="$(basename "$m" .gguf)"
  echo "=== host $name ==="
  "$ROOT/build/host/kiln-bench" -m "$m" -r "$RUNS" -n "$N" --cooldown 3 -o "$OUT/${HOST_TAG}_$name.csv"
done

if adb get-state >/dev/null 2>&1; then
  tag="pixel8"; tag="$(adb shell getprop ro.product.model | tr -d '\r ' | tr '[:upper:]' '[:lower:]')"
  adb shell mkdir -p "$DEV"
  adb push "$ROOT/build/android-arm64/kiln-bench" "$ROOT/build/android-arm64/libkiln.so" "$DEV/" >/dev/null
  for m in "$ROOT"/models/*.gguf; do
    name="$(basename "$m" .gguf)"
    adb shell "[ -f $DEV/$name.gguf ]" || adb push "$m" "$DEV/"
    echo "=== $tag $name ==="
    adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./kiln-bench -m $name.gguf -r $RUNS -n $N --cooldown 10 -o $DEV/bench.csv"
    adb pull "$DEV/bench.csv" "$OUT/${tag}_$name.csv" >/dev/null
    sleep 60   # let the SoC cool so the next model starts from a similar temperature
  done
fi

"$PY" "$ROOT/scripts/bench_plot.py" "$OUT"
