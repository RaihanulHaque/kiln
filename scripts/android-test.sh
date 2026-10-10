#!/usr/bin/env bash
# Push the Android build + test models to a connected device and run the test suite there.
# usage: scripts/android-test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${KILN_ANDROID_BUILD:-$ROOT/build/android-arm64}"
DEV=/data/local/tmp/kiln
NDK="${ANDROID_NDK_HOME:-$(ls -d "$HOME"/Library/Android/sdk/ndk/* | sort | tail -1)}"
LIBCXX="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"

adb shell mkdir -p "$DEV"
adb push "$BUILD/libkiln.so" "$BUILD/kiln-tests" "$LIBCXX" "$DEV/" >/dev/null
for m in "$ROOT"/models/*.gguf; do
  adb shell "[ -f $DEV/$(basename "$m") ]" || adb push "$m" "$DEV/"
done
adb shell "cd $DEV && chmod +x kiln-tests && LD_LIBRARY_PATH=$DEV ./kiln-tests $DEV"
