#!/usr/bin/env python3
# Word error rate of a speech model run through kiln-cli on a LibriSpeech test set.
# Usage: eval.py MODEL LIBRISPEECH_SPLIT_DIR OUT_PREFIX [--limit N] [-t THREADS]
import argparse, json, re, subprocess, sys, time
from pathlib import Path

import jiwer

KILN = Path(__file__).resolve().parents[2] / "build/host/kiln-cli"
STAT = re.compile(r"([\d.]+) s of audio .* in (\d+) ms = .*\| encoder (\d+) ms, prompt \d+ ms, (\d+) tok written in (\d+) ms")


def norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return " ".join(s.split())


def load_split(root):
    items = []
    for trans in sorted(Path(root).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            items.append((uid, trans.parent / f"{uid}.flac", text))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-t", "--threads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()

    items = load_split(a.split)
    if a.limit:
        items = items[: a.limit]
    hyps, stats = [], []
    t0 = time.time()
    for i in range(0, len(items), a.batch):
        chunk = items[i : i + a.batch]
        cmd = [str(KILN), "-m", a.model]
        if a.threads:
            cmd += ["-t", str(a.threads)]
        for _, f, _ in chunk:
            cmd += ["--transcribe", str(f)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout.split("\n")
        if out and out[-1] == "":
            out = out[:-1]
        st = [m.groups() for m in map(STAT.search, r.stderr.splitlines()) if m]
        if r.returncode != 0 or len(out) != len(chunk) or len(st) != len(chunk):
            sys.exit(f"batch {i}: rc={r.returncode} lines={len(out)} stats={len(st)} want={len(chunk)}\n{r.stderr[-2000:]}")
        hyps += out
        stats += st
        print(f"\r{len(hyps)}/{len(items)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    refs = [norm(t) for _, _, t in items]
    hn = [norm(h) for h in hyps]
    wer = jiwer.wer(refs, hn)
    audio = sum(float(s[0]) for s in stats)
    total = sum(int(s[1]) for s in stats) / 1000
    enc = sum(int(s[2]) for s in stats) / 1000
    dec = sum(int(s[4]) for s in stats) / 1000
    res = {"model": Path(a.model).name, "split": Path(a.split).name, "n": len(items), "wer": wer,
           "audio_s": audio, "call_s": total, "encoder_s": enc, "decode_s": dec,
           "rtfx": audio / total, "wall_s": time.time() - t0}
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    with open(a.out + ".tsv", "w") as f:
        for (uid, _, ref), h in zip(items, hyps):
            f.write(f"{uid}\t{norm(ref)}\t{norm(h)}\t{h}\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
