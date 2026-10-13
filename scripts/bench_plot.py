#!/usr/bin/env python3
"""Turn kiln-bench CSVs into charts and a summary table.

usage: python3 scripts/bench_plot.py docs/bench      (needs matplotlib + pandas)
Reads <device>_<model>.csv (samples over time) and, when present, <device>_<model>.runs.csv
(one line per run), writes <device>_<model>.png and summary.md next to them.

A chat model's speed is tokens per second. Decision and embedding models do not generate
tokens, so theirs is the time for one call, in milliseconds. A speech model's is the time to
write down one clip.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHASE_COLORS = {"load": "#f4a261", "prompt": "#e9c46a", "generate": "#2a9d8f", "decide": "#2a9d8f",
                "embed": "#2a9d8f", "encode": "#6a4c93", "transcribe": "#2a9d8f", "cooldown": "#8d99ae"}
WORK_PHASES = ["generate", "decide", "embed", "encode", "transcribe"]       # the model is computing
SERIES_COLORS = ["#2a9d8f", "#6a4c93", "#e76f51", "#457b9d", "#bc6c25"]


def load_runs(path):
    """The per-run file, with the time of each run taken from the samples (same order)."""
    rp = path[:-4] + ".runs.csv"
    if not os.path.exists(rp):
        return None
    runs = pd.read_csv(rp)
    if runs.empty:
        return None
    # the `run` value the run has in the samples; files written before that column existed are in order
    runs["global_run"] = runs["sample_run"] if "sample_run" in runs else range(1, len(runs) + 1)
    return runs


def speed_text(runs):
    """'111 -> 90 tok/s' for chat, 'text 52 -> 61 ms' style for call series: first run -> last run."""
    if runs is None:
        return "–"
    parts = []
    for series, g in runs.groupby("series", sort=False):
        first, last = g.iloc[0], g.iloc[-1]
        if first["unit"] == "tok":
            parts.append(f"{first['per_s']:.0f} → {last['per_s']:.0f} tok/s")
        else:
            def ms(v):
                return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"
            label = series.replace("embed-", "").replace("transcribe-", "").replace("decide", "question")
            parts.append(f"{label}: {ms(first['mean_ms'])} → {ms(last['mean_ms'])}")
    return "<br>".join(parts)


def plot_one(path, out_png):
    df = pd.read_csv(path)
    df["t_s"] = df["t_ms"] / 1000.0
    has_freq = df["freq_mhz"].max() > 0
    has_temp = df["temp_c"].max() > 0
    runs = load_runs(path)
    rows = 2 + int(has_freq) + int(has_temp) + int(runs is not None)
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

    if runs is not None:
        # speed of each run, drawn at the time the run happened
        ax = axes[0]
        # only samples taken while the model was working: older files keep the run number
        # through the pause after a run, which would pull the marker to the right
        busy = df[(df["run"] > 0) & df["phase"].isin(WORK_PHASES + ["prompt"])]
        when = busy.groupby("run")["t_s"].mean()
        is_chat = (runs["unit"] == "tok").all()
        for k, (series, g) in enumerate(runs.groupby("series", sort=False)):
            xs = [when.get(r, float("nan")) for r in g["global_run"]]
            ys = g["per_s"] if is_chat else g["mean_ms"]
            ax.plot(xs, ys, "o-", color=SERIES_COLORS[k % len(SERIES_COLORS)], lw=1.5, ms=5,
                    label=None if is_chat else series.replace("embed-", "").replace("transcribe-", ""))
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        if is_chat:
            ax.set_ylabel("tokens / s\nper run")
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylabel("ms per call\n(mean of each run)")
            ax.set_yscale("log")
            ax.legend(fontsize=8, loc="upper right", ncol=3)
        ax.margins(y=0.25)
        shade(ax)
    off = int(runs is not None)                           # the speed panel, when there is one, is on top
    ax = axes[off]
    ax.plot(df["t_s"], df["rss_mb"], color="#264653", lw=1.5)
    ax.set_ylabel("RSS (MB)")
    shade(ax)
    ax = axes[off + 1]
    ax.plot(df["t_s"], df["cpu_pct"], color="#e76f51", lw=1.2)
    ax.set_ylabel("process CPU %\n(100 = 1 core)")
    shade(ax)
    i = off + 2
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
    seen = [p for p in PHASE_COLORS if (df["phase"] == p).any()]
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[p], alpha=0.4) for p in seen]
    axes[off].legend(handles, seen, loc="upper right", fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    gen = df[df["phase"].isin(WORK_PHASES)]
    idle = df[(df["phase"] == "idle") & (df["run"] == 0)]
    return {
        "name": name,
        "speed": speed_text(runs),
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
        if csv.endswith(".runs.csv"):
            continue
        rows.append(plot_one(csv, csv[:-4] + ".png"))
        print("plotted", csv)
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("Speed is the first run → the last run of each series. \"work\" is while the model is computing: "
                "generating, deciding, embedding text, encoding an image or audio clip, or writing down speech.\n\n")
        f.write("| run | speed, first → last run | baseline RSS | peak RSS | work CPU mean | work CPU max | work freq | CPU temp start → max | OS throttle | min mem avail |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            def fmt(v, unit="", nd=0):
                return "–" if v != v else f"{v:.{nd}f}{unit}"
            f.write(f"| {r['name']} | {r['speed']} | {fmt(r['baseline_rss'],' MB')} | {fmt(r['peak_rss'],' MB')} | "
                    f"{fmt(r['gen_cpu_mean'],'%')} | {fmt(r['gen_cpu_max'],'%')} | {fmt(r['freq_gen_mean'],' MHz')} | "
                    f"{fmt(r['temp_start'],'',1)} → {fmt(r['temp_max'],' °C',1)} | {fmt(r['throttle_max'])} | {fmt(r['mem_avail_min'],' MB')} |\n")
    print(open(os.path.join(d, "summary.md")).read())


if __name__ == "__main__":
    main()
