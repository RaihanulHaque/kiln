#!/usr/bin/env bash
# Build the kiln core for iOS (arm64 device + arm64 simulator) as static libs, merge each into
# one libkiln.a, and wrap both in kiln.xcframework for the Flutter plugin's podspec.
# usage: scripts/build_ios.sh [--device-only]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/flutter/kiln_flutter/ios/Frameworks"
COMMON=(-DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_BUILD_TYPE=Release
        -DKILN_BUILD_SHARED=OFF -DKILN_BUILD_CLI=OFF -DKILN_BUILD_TESTS=OFF
        -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_OPENMP=OFF)

build() {  # $1 = build dir, rest = extra cmake args
  local dir="$1"; shift
  cmake -S "$ROOT" -B "$dir" "${COMMON[@]}" "$@" >/dev/null
  cmake --build "$dir" -j "$(sysctl -n hw.ncpu)" 2>&1 | grep -E "error|warning: .*deprecated" | grep -v "warning" || true
  # merge every static lib the build produced into one archive
  find "$dir" -name "*.a" -print0 | xargs -0 libtool -static -o "$dir/libkiln-merged.a" 2>&1 | grep -v "has no symbols" || true
  ls -la "$dir/libkiln-merged.a" | awk '{print "  " $9 ": " $5/1e6 " MB"}'
}

echo "== device (arm64)"
build "$ROOT/build/ios-device" -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_SYSROOT=iphoneos
ARGS=(-library "$ROOT/build/ios-device/libkiln-merged.a" -headers "$ROOT/include")

if [[ "${1:-}" != "--device-only" ]]; then
  echo "== simulator (arm64)"
  build "$ROOT/build/ios-sim" -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_SYSROOT=iphonesimulator
  ARGS+=(-library "$ROOT/build/ios-sim/libkiln-merged.a" -headers "$ROOT/include")
fi

rm -rf "$OUT/kiln.xcframework"; mkdir -p "$OUT"
xcodebuild -create-xcframework "${ARGS[@]}" -output "$OUT/kiln.xcframework" | tail -1
du -sh "$OUT/kiln.xcframework"
