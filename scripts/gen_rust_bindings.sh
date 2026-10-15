#!/usr/bin/env bash
# Regenerates rust/kiln-sys/src/bindings.rs from include/kiln/kiln.h.
# Run after every change to kiln.h (as with ffigen for the Flutter plugin).
# Needs bindgen: cargo install bindgen-cli
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# libclang without the clang package has no stddef.h of its own; borrow the compiler's.
extra=()
if ! command -v clang >/dev/null && command -v gcc >/dev/null; then
    extra=(-- -I"$(gcc -print-file-name=include)")
fi
bindgen "$root/include/kiln/kiln.h" \
    --allowlist-function 'kiln_.*' \
    --allowlist-type 'kiln_.*' \
    --allowlist-var 'KILN_.*' \
    --no-doc-comments \
    --output "$root/rust/kiln-sys/src/bindings.rs" \
    "${extra[@]}"
echo "wrote rust/kiln-sys/src/bindings.rs"
