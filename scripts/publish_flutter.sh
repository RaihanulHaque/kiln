#!/usr/bin/env bash
# Publish kiln_flutter to pub.dev. Publishing cannot be undone: a version stays there for good.
#
# Checks that the tree is clean, vendors the C core into the package, builds the example from
# exactly the files pub would upload, shows the dry run, and only then runs the real publish,
# which asks for confirmation (and a Google sign-in the first time).
#
# usage: scripts/publish_flutter.sh            the whole thing
#        scripts/publish_flutter.sh --dry-run  everything except the upload
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/flutter/kiln_flutter"
cleanup() { rm -rf "$PKG/src/core"; }   # while src/core exists the plugin builds it instead of the repo sources
trap cleanup EXIT

if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "error: uncommitted changes. Commit or stash them; publish from a clean tree." >&2
  git -C "$ROOT" status --short >&2
  exit 1
fi
VERSION="$(sed -n 's/^version: //p' "$PKG/pubspec.yaml")"
grep -q "^## $VERSION\$" "$PKG/CHANGELOG.md" || { echo "error: no '## $VERSION' entry in the plugin CHANGELOG.md" >&2; exit 1; }
echo "== kiln_flutter $VERSION at $(git -C "$ROOT" rev-parse --short HEAD) ($(git -C "$ROOT" branch --show-current))"

"$ROOT/scripts/vendor_flutter.sh" --check

echo "== dry run"
( cd "$PKG" && dart pub publish --dry-run ) | tail -8

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "dry run only; nothing was uploaded."
  exit 0
fi
echo "== publishing (you will be asked to confirm)"
( cd "$PKG" && dart pub publish )
