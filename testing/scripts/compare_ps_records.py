#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
 
import matplotlib.pyplot as plt
import pandas as pd
 
 
# ── shared helpers ────────────────────────────────────────────────────────────
 
def load_psrecord_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]) if len(parts) > 3 else None,
                ))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    return pd.DataFrame(rows, columns=["elapsed_sec", "cpu_pct", "real_mb", "virtual_mb"])
 
 
def summarize(df, label, sample, program):
    return {
        "sample":        sample,
        "program":       program,
        "label":         label,
        "duration_sec":  df["elapsed_sec"].max(),
        "avg_cpu_pct":   df["cpu_pct"].mean(),
        "peak_cpu_pct":  df["cpu_pct"].max(),
        "avg_real_mb":   df["real_mb"].mean(),
        "peak_real_mb":  df["real_mb"].max(),
    }
 
 
def collect_log_paths(inputs, pattern="*.log"):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, pattern))))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            matched = sorted(glob.glob(item))
            if matched:
                paths.extend(matched)
            else:
                print(f"Warning: '{item}' not found — skipping.", file=sys.stderr)
    return paths
 
 
def extract_sample_name(filename):
    """
    Strip the file extension, then remove any trailing _YYYYMMDD, _HHMMSS,
    or _YYYYMMDD_HHMMSS suffixes to get the sample name.
 
    Returns (display_name, match_key) where match_key has all underscores
    removed so that names that differ only by underscores (e.g.
    'sample_128_het005' vs 'sample_128het005') still match correctly.
 
    e.g. psrecord_sample_128_het005_20260918_110000
              display  → sample_128_het005
              key      → sample128het005
         psrecord_sample_128het005_20260918_110000
              display  → sample_128het005
              key      → sample128het005   ← same, so they pair up
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    # remove a leading "psrecord_" prefix if present (common naming pattern)
    stem = re.sub(r'^psrecord_', '', stem)
    # remove trailing _YYYYMMDD_HHMM(SS) or standalone _YYYYMMDD suffix
    stem = re.sub(r'(_\d{8}_\d{4,6}|_\d{6,8})$', '', stem)
    display = stem
    match_key = stem.replace('_', '').lower()
    return display, match_key
 
 
# ── general mode ──────────────────────────────────────────────────────────────
 
def run_general(args):
    log_paths = collect_log_paths(args.inputs, args.pattern)
    if not log_paths:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)
 
    datasets, summaries = [], []
    for path in log_paths:
        label = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_psrecord_log(path)
        except ValueError as e:
            print(f"Warning: {e} — skipping.", file=sys.stderr)
            continue
        datasets.append((label, df))
        summaries.append(summarize(df, label, label, ""))
 
    if not datasets:
        print("No valid data.", file=sys.stderr)
        sys.exit(1)
 
    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, df in datasets:
        ax_cpu.plot(df["elapsed_sec"], df["cpu_pct"], label=label, marker="o", markersize=3)
        ax_mem.plot(df["elapsed_sec"], df["real_mb"], label=label, marker="o", markersize=3)
 
    ax_cpu.set_ylabel("CPU (%)"); ax_cpu.set_title("CPU usage over time")
    ax_cpu.legend(fontsize=8); ax_cpu.grid(alpha=0.3)
    ax_mem.set_ylabel("Real memory (MB)"); ax_mem.set_xlabel("Elapsed time (s)")
    ax_mem.set_title("Memory usage over time"); ax_mem.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
 
    summary_df = pd.DataFrame(summaries).drop(columns=["sample","program"]).round(1)
    print("\nSummary:"); print(summary_df.to_string(index=False))
    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        print(f"\nSummary table saved to {args.csv}")
 
 
# ── paired mode ───────────────────────────────────────────────────────────────
 
def load_folder(folder, program_name, pattern="*.log"):
    """Load all logs in a folder.
 
    Returns a dict keyed by normalised match_key, with value
    (df, path, display_name) so we can show the original name in plots
    while matching on the underscore-stripped key.
    """
    logs = {}
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        display, key = extract_sample_name(path)
        try:
            df = load_psrecord_log(path)
            logs[key] = (df, path, display)
        except ValueError as e:
            print(f"Warning: {e} — skipping.", file=sys.stderr)
    return logs
 
 
def run_paired(args):
    name_a = args.name_a
    name_b = args.name_b
 
    logs_a = load_folder(args.program_a, name_a)
    logs_b = load_folder(args.program_b, name_b)
 
    # find samples present in both folders (matched on normalised key)
    matched = sorted(set(logs_a) & set(logs_b))
    only_a  = sorted(set(logs_a) - set(logs_b))
    only_b  = sorted(set(logs_b) - set(logs_a))
 
    if only_a:
        names = ', '.join(logs_a[k][2] for k in only_a)
        print(f"Warning: samples only in {name_a} (no match in {name_b}): {names}", file=sys.stderr)
    if only_b:
        names = ', '.join(logs_b[k][2] for k in only_b)
        print(f"Warning: samples only in {name_b} (no match in {name_a}): {names}", file=sys.stderr)
    if not matched:
        print("No matching samples found between the two folders.", file=sys.stderr)
        sys.exit(1)
 
    # use display name from folder A as the canonical label (they should be
    # the same sample, just with/without underscores)
    display_names = {key: logs_a[key][2] for key in matched}
    print(f"Matched {len(matched)} sample(s): {', '.join(display_names[k] for k in matched)}")
 
    n = len(matched)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    fig.suptitle(f"{name_a} vs {name_b} — per-sample comparison", fontsize=13, fontweight="bold")
 
    color_a = "#43ccca"   
    color_b = "#854ba6"   
 
    summaries = []
    for i, key in enumerate(matched):
        df_a, path_a, disp_a = logs_a[key]
        df_b, path_b, disp_b = logs_b[key]
        label = display_names[key]
 
        ax_cpu = axes[i, 0]
        ax_mem = axes[i, 1]
 
        # CPU subplot
        ax_cpu.plot(df_a["elapsed_sec"], df_a["cpu_pct"], color=color_a,
                    label=name_a, marker="o", markersize=3)
        ax_cpu.plot(df_b["elapsed_sec"], df_b["cpu_pct"], color=color_b,
                    label=name_b, marker="o", markersize=3)
        ax_cpu.set_title(f"{label} — CPU (%)")
        ax_cpu.set_ylabel("CPU (%)"); ax_cpu.set_xlabel("Elapsed time (s)")
        ax_cpu.legend(fontsize=8); ax_cpu.grid(alpha=0.3)
 
        # Memory subplot
        ax_mem.plot(df_a["elapsed_sec"], df_a["real_mb"], color=color_a,
                    label=name_a, marker="o", markersize=3)
        ax_mem.plot(df_b["elapsed_sec"], df_b["real_mb"], color=color_b,
                    label=name_b, marker="o", markersize=3)
        ax_mem.set_title(f"{label} — Memory (MB)")
        ax_mem.set_ylabel("Real memory (MB)"); ax_mem.set_xlabel("Elapsed time (s)")
        ax_mem.legend(fontsize=8); ax_mem.grid(alpha=0.3)
 
        summaries.append(summarize(df_a, os.path.basename(path_a), label, name_a))
        summaries.append(summarize(df_b, os.path.basename(path_b), label, name_b))
 
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
 
    # summary table: one row per sample per program, then a diff column
    summary_df = pd.DataFrame(summaries)[
        ["sample", "program", "duration_sec", "avg_cpu_pct", "peak_cpu_pct", "avg_real_mb", "peak_real_mb"]
    ].round(1)
 
    print("\nSummary:")
    print(summary_df.to_string(index=False))
 
    # print a simple delta table (B minus A) for quick comparison
    a_df = summary_df[summary_df["program"] == name_a].set_index("sample")
    b_df = summary_df[summary_df["program"] == name_b].set_index("sample")
    numeric_cols = ["duration_sec", "avg_cpu_pct", "peak_cpu_pct", "avg_real_mb", "peak_real_mb"]
    delta = (b_df[numeric_cols] - a_df[numeric_cols]).round(1)
    delta.columns = [f"Δ {c} ({name_b}−{name_a})" for c in numeric_cols]
    print(f"\nDelta ({name_b} minus {name_a}) — positive means {name_b} used more:")
    print(delta.to_string())
 
    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        delta_csv = args.csv.replace(".csv", "_delta.csv")
        delta.to_csv(delta_csv)
        print(f"\nSummary saved to {args.csv}")
        print(f"Delta table saved to {delta_csv}")
 
    overview_path = args.output.replace(".png", "_overview.png")
    plot_overview(summary_df, name_a, name_b, overview_path)
 
 
def plot_overview(summary_df, name_a, name_b, out_path):
    """
    Three outputs:
    1. Stacked panel bar chart (peak memory / runtime / peak CPU), one panel each.
    2. Combined bar+line chart using PEAK memory (bars, left axis) and runtime (line, right axis).
    3. Combined bar+line chart using AVERAGE memory (bars, left axis) and runtime (line, right axis).
    """
    import numpy as np
 
    color_a = "#43ccca"   
    color_b = "#854ba6"   
 
    df_a = summary_df[summary_df["program"] == name_a].set_index("sample")
    df_b = summary_df[summary_df["program"] == name_b].set_index("sample")
    samples = list(df_a.index)
    n = len(samples)
    x = np.arange(n)
 
    # ── 1. original stacked panel chart ──────────────────────────────────────
    metrics = [
        ("peak_real_mb", "Peak memory (MB)",  "Peak Memory per Sample"),
        ("duration_sec", "Runtime (s)",        "Runtime per Sample"),
        ("peak_cpu_pct", "Peak CPU (%)",       "Peak CPU per Sample"),
    ]
 
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(10, n * 1.2), 4 * len(metrics)))
    fig.suptitle(f"{name_a} vs {name_b} — overview across all samples",
                 fontsize=13, fontweight="bold")
 
    width = 0.35
    for ax, (col, ylabel, title) in zip(axes, metrics):
        vals_a = df_a[col].reindex(samples).values
        vals_b = df_b[col].reindex(samples).values
 
        ax.bar(x - width / 2, vals_a, width, label=name_a, color=color_a, alpha=0.85)
        ax.bar(x + width / 2, vals_b, width, label=name_b, color=color_b, alpha=0.85)
 
        mean_a = np.nanmean(vals_a)
        mean_b = np.nanmean(vals_b)
        ax.axhline(mean_a, color=color_a, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"{name_a} mean ({mean_a:.1f})")
        ax.axhline(mean_b, color=color_b, linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"{name_b} mean ({mean_b:.1f})")
 
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(samples, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Overview plot saved to {out_path}")
    plt.close()
 
    # ── 2 & 3. combined bar+line charts ──────────────────────────────────────
    combined_variants = [
        ("peak_real_mb", "Peak memory (MB)",    "peak"),
        ("avg_real_mb",  "Average memory (MB)", "avg"),
    ]
 
    for mem_col, mem_label, variant_tag in combined_variants:
        fig, ax_mem = plt.subplots(figsize=(max(10, n * 1.4), 5))
        ax_rt = ax_mem.twinx()
 
        bar_width = 0.2
        mem_a = df_a[mem_col].reindex(samples).values
        mem_b = df_b[mem_col].reindex(samples).values
        rt_a  = df_a["duration_sec"].reindex(samples).values
        rt_b  = df_b["duration_sec"].reindex(samples).values
 
        # bars (memory)
        ax_mem.bar(x - bar_width / 2, mem_a, bar_width,
                   label=f"{name_a} {mem_label}", color=color_a, alpha=0.7)
        ax_mem.bar(x + bar_width / 2, mem_b, bar_width,
                   label=f"{name_b} {mem_label}", color=color_b, alpha=0.7)
 
        # lines (runtime)
        ax_rt.plot(x, rt_a, color=color_a, linestyle="-", linewidth=2,
                   marker="o", markersize=6, label=f"{name_a} runtime")
        ax_rt.plot(x, rt_b, color=color_b, linestyle="-", linewidth=2,
                   marker="o", markersize=6, label=f"{name_b} runtime")
 
        ax_mem.set_ylabel(mem_label, fontsize=10)
        ax_rt.set_ylabel("Runtime (s)", fontsize=10)
        ax_mem.set_xticks(x)
        ax_mem.set_xticklabels(samples, rotation=35, ha="right", fontsize=8)
        ax_mem.grid(axis="y", alpha=0.3)
        ax_mem.set_axisbelow(True)
 
        #fig.suptitle(
        #    f"{name_a} vs {name_b} — {mem_label} (bars) & Runtime (line)",
        #    fontsize=12, fontweight="bold"
        #)
 
        # combined legend from both axes
        handles_mem, labels_mem = ax_mem.get_legend_handles_labels()
        handles_rt,  labels_rt  = ax_rt.get_legend_handles_labels()
        ax_mem.legend(handles_mem + handles_rt, labels_mem + labels_rt,
                      fontsize=8, loc="upper left")
 
        plt.tight_layout()
        combined_path = out_path.replace(".png", f"_combined_{variant_tag}mem.png")
        plt.savefig(combined_path, dpi=150)
        print(f"Combined ({variant_tag} memory) plot saved to {combined_path}")
        plt.close()
 
 
# ── entry point ───────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Compare psrecord logs. Use --program-a/--program-b for per-sample paired comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
 
    # general mode
    parser.add_argument("inputs", nargs="*", help="Log files or directories (general mode)")
    parser.add_argument("--pattern", default="*.log", help="Glob pattern for directory inputs (default: *.log)")
 
    # paired mode
    parser.add_argument("--program-a", metavar="DIR", help="Folder of logs for program A (paired mode)")
    parser.add_argument("--program-b", metavar="DIR", help="Folder of logs for program B (paired mode)")
    parser.add_argument("--name-a", default="ProgramA", help="Display name for program A")
    parser.add_argument("--name-b", default="ProgramB", help="Display name for program B")
 
    # shared
    parser.add_argument("-o", "--output", default="psrecord_comparison.png", help="Output plot path")
    parser.add_argument("--csv", default=None, help="Save summary table as CSV")
    parser.add_argument("--overview", action="store_true",
                        help="(paired mode) also save an overview bar chart across all samples")
 
    args = parser.parse_args()
 
    if args.program_a or args.program_b:
        if not (args.program_a and args.program_b):
            parser.error("--program-a and --program-b must both be provided together.")
        run_paired(args)
    elif args.inputs:
        run_general(args)
    else:
        parser.print_help()
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()