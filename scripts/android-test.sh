#!/usr/bin/env bash
# Push the Android build + test models to a connected device and run the test suite there.
# usage: scripts/android-test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${KILN_ANDROID_BUILD:-$ROOT/build/android-arm64}"
DEV=/data/local/tmp/kiln
# NDK default locations: Android Studio on macOS, then Linux
NDK="${ANDROID_NDK_HOME:-$(ls -d "$HOME"/Library/Android/sdk/ndk/* "$HOME"/Android/Sdk/ndk/* 2>/dev/null | sort | tail -1 || true)}"
[ -d "$NDK" ] || { echo "Android NDK not found; set ANDROID_NDK_HOME" >&2; exit 1; }
# prebuilt host dir is darwin-x86_64 on macOS, linux-x86_64 on Linux
LIBCXX="$(ls "$NDK"/toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so | head -1)"

adb shell mkdir -p "$DEV"
adb push "$BUILD/libkiln.so" "$BUILD/kiln-tests" "$LIBCXX" "$DEV/" >/dev/null
for m in "$ROOT"/models/*.gguf; do
  adb shell "[ -f $DEV/$(basename "$m") ]" || adb push "$m" "$DEV/"
done
# small images and audio clips for the kiln_embed_media tests
adb shell mkdir -p "$DEV/media"
adb push "$ROOT"/tests/media/. "$DEV/media/" >/dev/null
adb shell "cd $DEV && chmod +x kiln-tests && LD_LIBRARY_PATH=$DEV ./kiln-tests $DEV"
