#!/usr/bin/env python3
"""Plotting functions for sv_pipeline_compare.py"""
 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
 
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
 
 
def plot_f1_by_type(per_type_df: pd.DataFrame, outpath: str):
    """Grouped bar chart of F1 (or precision/recall) by SV type, one group per pipeline.
    per_type_df: rows = (pipeline, svtype), columns include 'f1'
    Expected columns: pipeline, svtype, precision, recall, f1
    """
    pipelines = per_type_df["pipeline"].unique()
    svtypes = sorted(per_type_df["svtype"].unique())
    x = np.arange(len(svtypes))
    width = 0.8 / max(len(pipelines), 1)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, pl in enumerate(pipelines):
        sub = per_type_df[per_type_df["pipeline"] == pl].set_index("svtype").reindex(svtypes)
        ax.bar(x + i * width, sub["f1"].values, width, label=pl, color=PALETTE[i % len(PALETTE)])
 
    ax.set_xticks(x + width * (len(pipelines) - 1) / 2)
    ax.set_xticklabels(svtypes)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("SV detection F1 by type")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_confusion_matrix(matrix: pd.DataFrame, pipeline_name: str, outpath: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Called type")
    ax.set_ylabel("Truth type")
    ax.set_title(f"SV type confusion matrix — {pipeline_name}")
 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if val > 0:
                color = "white" if val > matrix.values.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color)
 
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_bland_altman(hf_df: pd.DataFrame, pipeline_name: str, outpath: str):
    """Bland-Altman plot: mean(true,called) on x, (called - true) on y."""
    x = hf_df["true_hf"].values
    y = hf_df["called_hf"].values
    mean_vals = (x + y) / 2
    diff_vals = y - x
    bias = np.mean(diff_vals)
    sd = np.std(diff_vals)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
 
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.7, color=PALETTE[0], edgecolor="k", linewidth=0.3)
    ax.axhline(bias, color="black", linestyle="-", linewidth=1, label=f"Bias = {bias:.3f}")
    ax.axhline(loa_upper, color="red", linestyle="--", linewidth=1, label=f"+1.96 SD = {loa_upper:.3f}")
    ax.axhline(loa_lower, color="red", linestyle="--", linewidth=1, label=f"-1.96 SD = {loa_lower:.3f}")
    ax.set_xlabel("Mean of true & called heteroplasmy")
    ax.set_ylabel("Called - True (heteroplasmy)")
    ax.set_title(f"Bland-Altman — {pipeline_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_true_vs_called_scatter(hf_dict: dict, outpath: str, colors: dict = None):
    """Overlay scatter of true vs. called heteroplasmy, one color per pipeline,
    with an unlabeled y=x reference line. hf_dict: {pipeline_name: hf_df}
    colors: optional {pipeline_name: matplotlib color}, falls back to PALETTE
    for any pipeline not given an explicit color.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (name, df) in enumerate(hf_dict.items()):
        color = (colors or {}).get(name, PALETTE[i % len(PALETTE)])
        ax.scatter(df["true_hf"], df["called_hf"], label=name,
                   color=color, alpha=0.75, edgecolor="k", linewidth=0.3)
    lims = [0, 1]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)  # y=x reference, not in legend
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True heteroplasmy fraction")
    ax.set_ylabel("Called heteroplasmy fraction")
    ax.set_title("True vs. called heteroplasmy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_burden_curve(merged_df: pd.DataFrame, sample: str, pipeline_name: str, outpath: str):
    """Line plot of true vs. called deletion burden across the mitochondrial genome
    for one sample. merged_df has columns: position, truth, called (from
    matcher.compare_burden_curves)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(merged_df["position"], merged_df["truth"], color="black", linewidth=1.3, label="Truth")
    ax.plot(merged_df["position"], merged_df["called"], color=PALETTE[1], linewidth=1.1,
            alpha=0.85, label=pipeline_name)
    ax.set_xlabel("mtDNA position")
    ax.set_ylabel("Deletion burden fraction")
    ax.set_title(f"Deletion burden — {sample}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_burden_error_summary(burden_metrics_df: pd.DataFrame, outpath: str):
    """Bar chart of MAE/RMSE per sample per pipeline for the burden-curve comparison."""
    pipelines = burden_metrics_df["pipeline"].unique()
    samples = sorted(burden_metrics_df["sample"].unique())
    x = np.arange(len(samples))
    width = 0.8 / max(len(pipelines), 1)
 
    fig, ax = plt.subplots(figsize=(max(8, len(samples) * 1.2), 5))
    for i, pl in enumerate(pipelines):
        sub = burden_metrics_df[burden_metrics_df["pipeline"] == pl].set_index("sample").reindex(samples)
        ax.bar(x + i * width, sub["MAE"].values, width, label=pl, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(pipelines) - 1) / 2)
    ax.set_xticklabels(samples, rotation=30, ha="right")
    ax.set_ylabel("MAE (deletion burden fraction)")
    ax.set_title("Burden curve accuracy by sample")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_resource_usage(resource_df: pd.DataFrame, outpath: str):
    """Bar charts comparing wall-clock time, CPU time, and peak memory across pipelines."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    pipelines = resource_df["pipeline"].values
 
    axes[0].bar(pipelines, resource_df["wall_clock_sec"], color=PALETTE[0])
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Wall-clock time")
 
    cpu_time = resource_df["user_time_sec"] + resource_df["sys_time_sec"]
    axes[1].bar(pipelines, cpu_time, color=PALETTE[1])
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Total CPU time (user + sys)")
 
    axes[2].bar(pipelines, resource_df["max_rss_kb"] / 1024, color=PALETTE[2])
    axes[2].set_ylabel("MB")
    axes[2].set_title("Peak memory (RSS)")
 
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
 
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_support_threshold_sweep(sweep_df: pd.DataFrame, pipeline_name: str, outpath: str):
    """Precision/recall/F1 vs. minimum read-support cutoff. sweep_df comes
    from matcher.support_threshold_sweep(): columns min_support, precision,
    recall, f1 (plus TP/FP/FN counts, not plotted here).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["min_support"], sweep_df["precision"], marker="o", label="Precision", color=PALETTE[0])
    ax.plot(sweep_df["min_support"], sweep_df["recall"], marker="o", label="Recall", color=PALETTE[1])
    ax.plot(sweep_df["min_support"], sweep_df["f1"], marker="o", label="F1", color=PALETTE[2], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Minimum read support to keep a call (log scale)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Accuracy vs. support threshold — {pipeline_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_tp_fp_support_distribution(match_df: pd.DataFrame, pipeline_name: str, outpath: str):
    """Strip plot comparing read support for TP calls vs. FP calls, to show
    directly whether false positives tend to be low-support noise or are
    genuinely well-supported (and therefore not fixable by a simple support
    filter). match_df comes from matcher.match_events() and must have a
    called_support column.
    """
    tp = match_df[(match_df["match_status"] == "TP") & match_df["called_support"].notna()]["called_support"]
    fp = match_df[(match_df["match_status"] == "FP") & match_df["called_support"].notna()]["called_support"]
 
    fig, ax = plt.subplots(figsize=(6, 5))
    rng = np.random.default_rng(0)
    if len(tp) > 0:
        x_tp = rng.normal(0, 0.05, size=len(tp))
        ax.scatter(x_tp, tp, color=PALETTE[2], alpha=0.7, edgecolor="k", linewidth=0.3, label=f"TP (n={len(tp)})")
    if len(fp) > 0:
        x_fp = rng.normal(1, 0.05, size=len(fp))
        ax.scatter(x_fp, fp, color=PALETTE[3], alpha=0.7, edgecolor="k", linewidth=0.3, label=f"FP (n={len(fp)})")
 
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["TP", "FP"])
    ax.set_ylabel("Read support (log scale)")
    ax.set_title(f"Read support: correct vs. false calls — {pipeline_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
 
 
def plot_f1_per_sample(per_sample_df: pd.DataFrame, pipeline_name: str, outpath: str):
    """Bar chart of F1 per sample, sorted ascending, colored/annotated by
    how many truth events that sample contains. Makes it visually obvious
    whether a low pooled F1 is coming from one busy, difficult sample or is
    spread evenly across many -- which the pooled 'Overall' number alone
    can't distinguish.
    """
    df = per_sample_df.sort_values("f1", na_position="first")
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 5))
    colors = [PALETTE[2] if v >= 0.8 else PALETTE[1] if v >= 0.5 else PALETTE[3] for v in df["f1"].fillna(0)]
    bars = ax.bar(range(len(df)), df["f1"].fillna(0), color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["sample"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"F1 by sample — {pipeline_name}")
    ax.grid(axis="y", alpha=0.3)
 
    for bar, n in zip(bars, df["n_truth_events"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={int(n)}",
                ha="center", va="bottom", fontsize=7, rotation=0)
 
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
