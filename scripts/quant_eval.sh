#!/usr/bin/env bash
# Quantization feasibility: from one F16 GGUF, make several formats (with and without an
# importance matrix), measure how far each drifts from F16 (KL divergence on the host), and
# how fast each generates on the attached phone. Numbers from the first run are in docs/QUANT.md.
#
# usage: scripts/quant_eval.sh <model-F16.gguf> <out dir>
#   needs: third_party/llama.cpp built in build-host (llama-quantize, llama-imatrix,
#   llama-perplexity), wikitext-2 in <out dir>/wt (fetched if missing), and for the phone part
#   build/android-arm64/kiln-bench and an adb device.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$1"; OUT="$2"; mkdir -p "$OUT"
B="$ROOT/third_party/llama.cpp/build-host/bin"
NAME="$(basename "$SRC" .gguf)"; NAME="${NAME%-F16}"
[ -x "$B/llama-quantize" ] || { echo "build llama.cpp first: cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-host -DLLAMA_CURL=OFF && cmake --build third_party/llama.cpp/build-host --target llama-quantize llama-imatrix llama-perplexity -j" >&2; exit 1; }
if [ ! -f "$OUT/wt/wikitext-2-raw/wiki.test.raw" ]; then
  curl -sL -o "$OUT/wikitext-2-raw-v1.zip" https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  unzip -o -q "$OUT/wikitext-2-raw-v1.zip" -d "$OUT/wt"
fi
TEST="$OUT/wt/wikitext-2-raw/wiki.test.raw"; TRAIN="$OUT/wt/wikitext-2-raw/wiki.train.raw"
FORMATS="${FORMATS:-Q8_0 Q6_K Q4_K_M Q4_0 IQ4_XS}"
IMAT_FORMATS="${IMAT_FORMATS:-Q4_K_M Q4_0 IQ4_XS}"

echo "== quantize"
for t in $FORMATS; do
  [ -f "$OUT/$NAME-$t.gguf" ] || "$B/llama-quantize" "$SRC" "$OUT/$NAME-$t.gguf" "$t" 8 >/dev/null 2>&1
done
[ -f "$OUT/imatrix.gguf" ] || "$B/llama-imatrix" -m "$SRC" -f "$TRAIN" -c 512 --chunks 100 -o "$OUT/imatrix.gguf" >/dev/null 2>&1
for t in $IMAT_FORMATS; do
  [ -f "$OUT/$NAME-$t-imat.gguf" ] || "$B/llama-quantize" --imatrix "$OUT/imatrix.gguf" "$SRC" "$OUT/$NAME-$t-imat.gguf" "$t" 8 >/dev/null 2>&1
done

echo "== quality: KL divergence from F16 (wikitext-2 test, 40 chunks of 512)"
[ -f "$OUT/kl_base.bin" ] || "$B/llama-perplexity" -m "$SRC" -f "$TEST" -c 512 -b 512 --chunks 40 --kl-divergence-base "$OUT/kl_base.bin" >/dev/null 2>&1
printf '%-14s %10s %10s %12s\n' format MB mean_KLD same_top
for f in "$OUT"/$NAME-*.gguf; do
  t="$(basename "$f" .gguf)"; t="${t#$NAME-}"; [ "$t" = "F16" ] && continue
  line="$("$B/llama-perplexity" -m "$f" --kl-divergence-base "$OUT/kl_base.bin" --kl-divergence -c 512 -b 512 2>&1 | grep -E 'Mean    KLD|Same top p' | sed 's/.*I //' | tr '\n' ' ')"
  kld="$(echo "$line" | sed -E 's/.*Mean +KLD: +([0-9.]+).*/\1/')"; top="$(echo "$line" | sed -E 's/.*Same top p: +([0-9.]+).*/\1/')"
  printf '%-14s %10d %10s %11s%%\n' "$t" $(( $(stat -f%z "$f" 2>/dev/null || stat -c%s "$f") / 1000000 )) "$kld" "$top"
done

if adb get-state >/dev/null 2>&1 && [ -x "$ROOT/build/android-arm64/kiln-bench" ]; then
  echo "== speed on the device: tok/s per run (3 runs of 256 tokens), coolest run first"
  DEV=/data/local/tmp/kiln/q; adb shell mkdir -p "$DEV"
  adb push "$ROOT/build/android-arm64/kiln-bench" "$ROOT/build/android-arm64/libkiln.so" "$DEV/" >/dev/null
  for f in "$OUT"/$NAME-*.gguf; do
    t="$(basename "$f" .gguf)"; t="${t#$NAME-}"; [ "$t" = "F16" ] && continue
    adb shell "[ -f $DEV/$(basename "$f") ]" </dev/null || adb push "$f" "$DEV/" >/dev/null
    # wait until the phone lets the cores run at full clock again
    n=0; while [ $n -lt 30 ] && ! adb shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do [ "$(cat $p/scaling_max_freq)" = "$(cat $p/cpuinfo_max_freq)" ] || exit 1; done' </dev/null; do sleep 10; n=$((n+1)); done
    printf '%-14s ' "$t"
    adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV:/data/local/tmp/kiln ./kiln-bench -m $(basename "$f") -r 3 -n 256 --cooldown 8 -o $DEV/b.csv 2>&1 | grep '^chat,' | cut -d, -f9" </dev/null | tr '\n' ' '; echo
    sleep 20
  done
fi
