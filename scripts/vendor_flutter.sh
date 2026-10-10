#!/usr/bin/env bash
# Copy the kiln core + pinned llama.cpp into flutter/kiln_flutter/src/core so the package
# builds standalone (pub.dev users do not have the monorepo). Prunes everything the build
# does not need to keep the package small. Idempotent. Run before `dart pub publish`.
#
# usage: scripts/vendor_flutter.sh [--check]     (--check: build an APK from the vendored tree)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/flutter/kiln_flutter"
DST="$PKG/src/core"

rm -rf "$DST"
mkdir -p "$DST/third_party/llama.cpp"

# --- kiln core
cp -R "$ROOT/include" "$ROOT/src" "$ROOT/CMakeLists.txt" "$ROOT/LICENSE" "$DST/"

# --- llama.cpp at the pinned submodule commit, via git archive (no .git, no untracked files)
LLAMA_SHA="$(git -C "$ROOT/third_party/llama.cpp" rev-parse HEAD)"
LLAMA_TAG="$(git -C "$ROOT/third_party/llama.cpp" describe --tags --exact-match 2>/dev/null || echo "$LLAMA_SHA")"
git -C "$ROOT/third_party/llama.cpp" archive --format=tar HEAD | tar -x -C "$DST/third_party/llama.cpp"

# --- prune what the CPU/Metal/Vulkan build never reads
L="$DST/third_party/llama.cpp"
rm -rf "$L"/{examples,tests,docs,models,tools,media,gguf-py,scripts,ci,.github,.devops,.devcontainer,grammars,prompts,pocs,requirements,poetry.lock,pyproject.toml,requirements.txt,convert_*.py,build-xcframework.sh,mypy.ini,pyrightconfig.json,flake.nix,flake.lock,SECURITY.md,CONTRIBUTING.md,AUTHORS} 2>/dev/null || true
# ggml backends we never enable on Android/Apple
for b in cuda hip musa sycl cann rpc kompute webgpu zdnn virtgpu openvino et blas hexagon; do rm -rf "$L/ggml/src/ggml-$b"; done
# tests inside common/vendor
rm -rf "$L/vendor/cpp-httplib/test" 2>/dev/null || true

# stamp
cat > "$DST/VENDORED.md" <<STAMP
Vendored copy of the kiln core. Do not edit here; edit the monorepo and re-run scripts/vendor_flutter.sh.

kiln:      $(git -C "$ROOT" describe --tags --always --dirty)
llama.cpp: $LLAMA_TAG ($LLAMA_SHA)
date:      $(date -u +%Y-%m-%dT%H:%MZ)
STAMP

SIZE="$(du -sh "$DST" | cut -f1)"
echo "vendored core -> $DST ($SIZE), llama.cpp $LLAMA_TAG"

if [[ "${1:-}" == "--check" ]]; then
  # Build the example APK with the monorepo root hidden, so only the vendored core can be used.
  echo "building example APK from the vendored core..."
  TMP="$(mktemp -d)"
  cp -R "$PKG" "$TMP/kiln_flutter"
  ( cd "$TMP/kiln_flutter/example" && flutter pub get >/dev/null && flutter build apk --release 2>&1 | grep -E "✓|error|Error" )
  ls -la "$TMP/kiln_flutter/example/build/app/outputs/flutter-apk/app-release.apk"
  rm -rf "$TMP"
fi
