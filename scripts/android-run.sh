#!/usr/bin/env bash
# Push the Android build + a GGUF to a connected device and run kiln-cli there.
# usage: scripts/android-run.sh path/to/model.gguf [kiln-cli args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${KILN_ANDROID_BUILD:-$ROOT/build/android-arm64}"
MODEL="${1:?model.gguf required}"; shift || true
DEV=/data/local/tmp/kiln
# NDK default locations: Android Studio on macOS, then Linux
NDK="${ANDROID_NDK_HOME:-$(ls -d "$HOME"/Library/Android/sdk/ndk/* "$HOME"/Android/Sdk/ndk/* 2>/dev/null | sort | tail -1)}"
[ -d "$NDK" ] || { echo "Android NDK not found; set ANDROID_NDK_HOME" >&2; exit 1; }
# prebuilt host dir is darwin-x86_64 on macOS, linux-x86_64 on Linux
LIBCXX="$(ls "$NDK"/toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so | head -1)"

adb shell mkdir -p "$DEV"
adb push "$BUILD/libkiln.so" "$BUILD/kiln-cli" "$LIBCXX" "$DEV/" >/dev/null
if ! adb shell "[ -f $DEV/$(basename "$MODEL") ]"; then
  adb push "$MODEL" "$DEV/" 
fi
adb shell "cd $DEV && chmod +x kiln-cli && LD_LIBRARY_PATH=$DEV ./kiln-cli --probe"
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./kiln-cli -m $DEV/$(basename "$MODEL") $*"
