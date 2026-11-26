#!/usr/bin/env python3
# Compare encoder outputs in two directories (files from encprof --dump).
# Usage: cmp.py REF_DIR NEW_DIR
import sys
from pathlib import Path

import numpy as np


def load(p):
    b = Path(p).read_bytes()
    n_state, n_frames = np.frombuffer(b[:8], dtype=np.int32)
    return np.frombuffer(b[8:], dtype=np.float32).reshape(n_frames, n_state)


ref, new = Path(sys.argv[1]), Path(sys.argv[2])
worst_abs = worst_rel = 0.0
for f in sorted(ref.glob("*.enc")):
    a, b = load(f), load(new / f.name)
    if a.shape != b.shape:
        sys.exit(f"{f.name}: shape {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    rel = np.sqrt((d ** 2).mean() / (a ** 2).mean())
    worst_abs, worst_rel = max(worst_abs, d.max()), max(worst_rel, rel)
    print(f"{f.name:28s} max|d| {d.max():.2e}  rel rms {rel:.2e}  (|ref| max {np.abs(a).max():.1f})")
print(f"worst: max|d| {worst_abs:.2e}, rel rms {worst_rel:.2e}")
