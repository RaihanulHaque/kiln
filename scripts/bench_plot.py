#!/usr/bin/env python3
"""Turn kiln-bench CSVs into charts and a summary table.

usage: python3 scripts/bench_plot.py docs/bench      (needs matplotlib + pandas)
Reads <device>_<model>.csv, writes <device>_<model>.png and summary.md next to them.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHASE_COLORS = {"load": "#f4a261", "prompt": "#e9c46a", "generate": "#2a9d8f", "cooldown": "#8d99ae"}


def plot_one(path, out_png):
    df = pd.read_csv(path)
    df["t_s"] = df["t_ms"] / 1000.0
    has_freq = df["freq_mhz"].max() > 0
    has_temp = df["temp_c"].max() > 0
    rows = 2 + int(has_freq) + int(has_temp)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.4 * rows), sharex=True)
    axes = list(axes)
    name = os.path.basename(path)[:-4]
    fig.suptitle(name, fontsize=12)

    def shade(ax):
        # shade phases
        prev = None
        start = 0
        for _, r in df.iterrows():
            if r["phase"] != prev:
                if prev in PHASE_COLORS:
                    ax.axvspan(start, r["t_s"], color=PHASE_COLORS[prev], alpha=0.18, lw=0)
                prev, start = r["phase"], r["t_s"]
        if prev in PHASE_COLORS:
            ax.axvspan(start, df["t_s"].iloc[-1], color=PHASE_COLORS[prev], alpha=0.18, lw=0)

    ax = axes[0]
    ax.plot(df["t_s"], df["rss_mb"], color="#264653", lw=1.5)
    ax.set_ylabel("RSS (MB)")
    shade(ax)
    ax = axes[1]
    ax.plot(df["t_s"], df["cpu_pct"], color="#e76f51", lw=1.2)
    ax.set_ylabel("process CPU %\n(100 = 1 core)")
    shade(ax)
    i = 2
    if has_freq:
        ax = axes[i]; i += 1
        ax.plot(df["t_s"], df["freq_mhz"], color="#457b9d", lw=1.2)
        ax.set_ylabel("fast-core freq (MHz)")
        shade(ax)
    if has_temp:
        ax = axes[i]; i += 1
        ax.plot(df["t_s"], df["temp_c"], color="#d62828", lw=1.2, label="max CPU cluster")
        if "skin_c" in df and df["skin_c"].max() > 0:
            ax.plot(df["t_s"], df["skin_c"], color="#f77f00", lw=1.2, label="skin")
        if "throttle" in df and df["throttle"].max() > 0:
            thr = df[df["throttle"] > 0]
            ax.scatter(thr["t_s"], thr["temp_c"], color="black", s=8, label="OS throttle status > 0", zorder=3)
        ax.set_ylabel("temp (°C)")
        ax.legend(fontsize=8, loc="lower right")
        shade(ax)
    axes[-1].set_xlabel("seconds")
    for a in axes:
        a.grid(alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in PHASE_COLORS.values()]
    axes[0].legend(handles, list(PHASE_COLORS), loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    gen = df[df["phase"] == "generate"]
    idle = df[(df["phase"] == "idle") & (df["run"] == 0)]
    return {
        "name": name,
        "baseline_rss": idle["rss_mb"].iloc[:4].mean() if len(idle) else float("nan"),
        "peak_rss": df["rss_mb"].max(),
        "gen_cpu_mean": gen["cpu_pct"].mean() if len(gen) else float("nan"),
        "gen_cpu_max": gen["cpu_pct"].max() if len(gen) else float("nan"),
        "freq_gen_mean": gen["freq_mhz"].mean() if has_freq and len(gen) else float("nan"),
        "temp_start": df["temp_c"].iloc[0] if has_temp else float("nan"),
        "temp_max": df["temp_c"].max() if has_temp else float("nan"),
        "mem_avail_min": df["mem_avail_mb"].min() if df["mem_avail_mb"].max() > 0 else float("nan"),
        "throttle_max": df["throttle"].max() if "throttle" in df else float("nan"),
    }


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "docs/bench"
    rows = []
    for csv in sorted(glob.glob(os.path.join(d, "*.csv"))):
        rows.append(plot_one(csv, csv[:-4] + ".png"))
        print("plotted", csv)
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("| run | baseline RSS | peak RSS | gen CPU mean | gen CPU max | gen freq | CPU temp start → max | OS throttle | min mem avail |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            def fmt(v, unit="", nd=0):
                return "–" if v != v else f"{v:.{nd}f}{unit}"
            f.write(f"| {r['name']} | {fmt(r['baseline_rss'],' MB')} | {fmt(r['peak_rss'],' MB')} | "
                    f"{fmt(r['gen_cpu_mean'],'%')} | {fmt(r['gen_cpu_max'],'%')} | {fmt(r['freq_gen_mean'],' MHz')} | "
                    f"{fmt(r['temp_start'],'',1)} → {fmt(r['temp_max'],' °C',1)} | {fmt(r['throttle_max'])} | {fmt(r['mem_avail_min'],' MB')} |\n")
    print(open(os.path.join(d, "summary.md")).read())


if __name__ == "__main__":
    main()
