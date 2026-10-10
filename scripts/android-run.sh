#!/usr/bin/env bash
# Push the Android build + a GGUF to a connected device and run kiln-cli there.
# usage: scripts/android-run.sh path/to/model.gguf [kiln-cli args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${KILN_ANDROID_BUILD:-$ROOT/build/android-arm64}"
MODEL="${1:?model.gguf required}"; shift || true
DEV=/data/local/tmp/kiln
NDK="${ANDROID_NDK_HOME:-$(ls -d "$HOME"/Library/Android/sdk/ndk/* | sort | tail -1)}"
LIBCXX="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"

adb shell mkdir -p "$DEV"
adb push "$BUILD/libkiln.so" "$BUILD/kiln-cli" "$LIBCXX" "$DEV/" >/dev/null
if ! adb shell "[ -f $DEV/$(basename "$MODEL") ]"; then
  adb push "$MODEL" "$DEV/" 
fi
adb shell "cd $DEV && chmod +x kiln-cli && LD_LIBRARY_PATH=$DEV ./kiln-cli --probe"
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./kiln-cli -m $DEV/$(basename "$MODEL") $*"
