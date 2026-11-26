#!/usr/bin/env python3
# Turn a LibriSpeech parquet file from Hugging Face into flac files plus a trans.txt
# Usage: unpack.py IN.parquet OUT_DIR
import sys
from pathlib import Path

import pyarrow.parquet as pq

src, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
t = pq.read_table(src, columns=["id", "audio", "text"]).to_pylist()
t.sort(key=lambda r: r["id"])
with open(out / "all.trans.txt", "w") as f:
    for r in t:
        (out / f"{r['id']}.flac").write_bytes(r["audio"]["bytes"])
        f.write(f"{r['id']} {r['text']}\n")
print(len(t), "clips in", out)
