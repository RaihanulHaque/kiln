#!/usr/bin/env python3
"""Compare several kiln-bench CSVs (e.g. a thread sweep) on one chart.

usage: python scripts/bench_compare.py docs/bench/threads 'pixel8_qwen_*.csv' out.png
Per config: tok/s per run (from the tokens column), mean fast-core MHz per run,
max CPU temp per run, and a summary table on stdout.
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def per_run(df):
    rows = []
    for run, g in df[df["phase"] == "generate"].groupby("run"):
        if run == 0 or len(g) < 2:
            continue
        dt = (g["t_ms"].iloc[-1] - g["t_ms"].iloc[0]) / 1000.0
        dtok = g["tokens"].iloc[-1] - g["tokens"].iloc[0]
        rows.append({
            "run": int(run),
            "tps": dtok / dt if dt > 0 else float("nan"),
            "mhz": g["freq_mhz"].mean(),
            "temp": g["temp_c"].max(),
            "skin": g["skin_c"].max() if "skin_c" in g else float("nan"),
            "cpu": g["cpu_pct"].mean(),
        })
    return pd.DataFrame(rows)


def label_of(path):
    m = re.search(r"_t(\d+)_(\w+)\.csv$", path)
    names = {"pin": "pinned, top cores", "mid": "pinned, A715 only", "nopin": "unpinned"}
    return f"{m.group(1)} thr, {names.get(m.group(2), m.group(2))}" if m else os.path.basename(path)


def main():
    d, pattern, out = sys.argv[1], sys.argv[2], sys.argv[3]
    files = sorted(glob.glob(os.path.join(d, pattern)), key=lambda p: (int(re.search(r"_t(\d+)_", p).group(1)) if re.search(r"_t(\d+)_", p) else 0, p))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    summary = []
    for f in files:
        r = per_run(pd.read_csv(f))
        lab = label_of(f)
        ls = "--" if "unpinned" in lab else "-"
        axes[0].plot(r["run"], r["tps"], ls, marker="o", label=lab)
        axes[1].plot(r["run"], r["mhz"], ls, marker="o", label=lab)
        axes[2].plot(r["run"], r["temp"], ls, marker="o", label=lab)
        summary.append({
            "config": lab,
            "run1 tok/s": r["tps"].iloc[0],
            "run8 tok/s": r["tps"].iloc[-1],
            "mean tok/s": r["tps"].mean(),
            "drop %": 100 * (1 - r["tps"].iloc[-1] / r["tps"].iloc[0]),
            "MHz run1→8": f"{r['mhz'].iloc[0]:.0f}→{r['mhz'].iloc[-1]:.0f}",
            "max temp": r["temp"].max(),
            "max skin": r["skin"].max(),
            "cpu %": r["cpu"].mean(),
        })
    axes[0].set_ylabel("gen tok/s"); axes[1].set_ylabel("fast-core MHz"); axes[2].set_ylabel("max CPU temp °C")
    axes[2].set_xlabel("run (same prompt, back to back)")
    for a in axes:
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(os.path.basename(out)[:-4])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    df = pd.DataFrame(summary)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    md = out[:-4] + ".md"
    with open(md, "w") as fh:
        fh.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, row in df.iterrows():
            fh.write("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in row) + " |\n")
    print("wrote", out, md)


if __name__ == "__main__":
    main()
