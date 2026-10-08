#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
 
import numpy as np
import pandas as pd
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matcher import (
    load_pipeline1_tsv,
    load_pipeline2_cluster_tsv,
    load_burden_tsv,
    attach_deletion_burden_estimate,
    collapse_overlapping_calls,
    match_events,
    summarize_matches,
    summarize_matches_weighted,
    summarize_matches_weighted_per_sample,
    summarize_matches_per_sample,
    macro_average_summary,
    summarize_by_type,
    truth_burden_curve,
    compare_burden_curves,
    sample_level_burden_totals,
    support_threshold_sweep,
    flag_colocated_fps,
    summarize_matches_adjusted,
    truth_samples_with_overlapping_dels,
)
from sv_plots import (
    plot_f1_by_type,
    plot_bland_altman,
    plot_true_vs_called_scatter,
    plot_burden_curve,
    plot_burden_error_summary,
    plot_resource_usage,
    plot_support_threshold_sweep,
    plot_tp_fp_support_distribution,
    plot_f1_per_sample,
)
 
 
# resource-usage log parser + heteroplasmy regression metrics (no VCF dependency)
from resource_and_het_metrics import parse_time_v_log, regression_metrics
 
 
EVENT_COMPARISON_COLS = ["TP", "FP", "FN", "precision", "recall", "f1", "accuracy"]
WEIGHTED_COMPARISON_COLS = ["TP_events", "FP_events", "FN_events", "event_precision", "recall",
                            "TP_reads", "FP_reads", "weighted_precision", "weighted_f1"]


def _merge_per_sample_comparison(own_df, other_df, own_name, other_name, expected_cols,
                                  outdir, out_filename, label):
    """Shared merge logic for --compare-with, used for both the plain event-count
    per-sample table and the read-weighted one -- same idea either way: line up
    both pipelines' per-sample metrics side by side on an outer join over 'sample',
    so a sample only one side has still shows up (as NaN on the other side) instead
    of silently vanishing. Returns the merged DataFrame, or None if the columns
    didn't overlap enough to be worth merging.
    """
    metric_cols = [c for c in expected_cols if c in own_df.columns and c in other_df.columns]
    missing = [c for c in expected_cols if c not in metric_cols]
    if len(metric_cols) < max(2, len(expected_cols) // 2):
        print(f"  [warning] {out_filename}: the file passed to --compare-with shares too few "
              f"columns with this run's {label} table to compare (missing: {missing}) -- make "
              f"sure --compare-with points at the other pipeline's matching {label} CSV. Skipped.")
        return None
    if missing:
        print(f"  [warning] --compare-with is missing column(s) {missing} that this run's "
              f"{label} table has -- comparison will only include {metric_cols}.")
    a = own_df[["sample"] + metric_cols].add_suffix(f"_{own_name}")
    a = a.rename(columns={f"sample_{own_name}": "sample"})
    b = other_df[["sample"] + metric_cols].add_suffix(f"_{other_name}")
    b = b.rename(columns={f"sample_{other_name}": "sample"})
    comparison_df = a.merge(b, on="sample", how="outer").sort_values("sample")
    check_col = metric_cols[0]
    n_a_only = comparison_df[f"{check_col}_{other_name}"].isna().sum()
    n_b_only = comparison_df[f"{check_col}_{own_name}"].isna().sum()
    if n_a_only or n_b_only:
        print(f"  [warning] {n_a_only} sample(s) only in {own_name}'s results, {n_b_only} "
              f"sample(s) only in {other_name}'s -- check both runs used the same --truth-file "
              f"and sample-name extraction.")
    comparison_df.to_csv(os.path.join(outdir, out_filename), index=False)
    print(f"\n=== Per-sample comparison ({label}): {own_name} vs {other_name} ===")
    print(comparison_df.to_string(index=False))
    return comparison_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-script", help="Path to your variant-generation script "
                                          "(builds the truth table; omit if using --truth-file instead)")
    ap.add_argument("--truth-file", help="Use an already-built truth TSV instead of building one from "
                                          "--sim-script. Must have columns: sample, svtype, start, end, "
                                          "wraps_origin, effective_heteroplasmy (event_label optional) -- "
                                          "i.e. the same schema truth_builder.py produces.")
    ap.add_argument("--mt-len", type=int, default=16569,
                     help="Genome length, only used with --truth-file (default 16569, standard mtDNA/rCRS)")
    ap.add_argument("--pipeline-name", required=True)
    ap.add_argument("--pipeline-type", required=True, choices=["per_cluster_het", "cluster_plus_burden"])
 
    ap.add_argument("--calls-file", help="[per_cluster_het] single TSV covering all samples")
    ap.add_argument("--calls-glob", help="[per_cluster_het] glob matching one TSV per sample (all in one folder); "
                                          "each file's own 'sample' column is used, so no filename regex needed")
 
    ap.add_argument("--cluster-glob", help="[cluster_plus_burden] glob for per-sample cluster TSVs")
    ap.add_argument("--burden-glob", help="[cluster_plus_burden] glob for per-sample burden TSVs")
    ap.add_argument("--sample-regex", help="[cluster_plus_burden] regex to pull the sample name (or its pieces) "
                                            "from the cluster filename -- see --sample-format if you need more "
                                            "than one capture group")
    ap.add_argument("--sample-format", default="{0}",
                     help="[cluster_plus_burden] Python .format() string built from --sample-regex's capture "
                          "groups (0-indexed), used when the filename doesn't literally contain the truth "
                          "sample name -- e.g. if truth has 'DEL_common_het005' but the filename has "
                          "'DEL_commonhet005' (missing underscore), use "
                          "--sample-regex '(.+?)het(\\d+)_cluster_table\\.tsv' "
                          "--sample-format '{0}_het{1}'. Default '{0}' just uses the first capture group as-is.")
    ap.add_argument("--burden-sample-regex", default=None,
                     help="[cluster_plus_burden] regex for extracting the sample name from BURDEN filenames, "
                          "if different from --sample-regex ")
    ap.add_argument("--burden-sample-format", default=None,
                     help="[cluster_plus_burden] --sample-format equivalent for --burden-sample-regex. "
                          "Defaults to --sample-format if not given.")
    ap.add_argument("--min-reciprocal-overlap", type=float, default=0.5)
    ap.add_argument("--collapse-overlap", type=float, default=0.8,
                     help="Merge same-svtype calls within a sample whose overlap (per --collapse-mode) is >= "
                          "this before matching, so redundant/jittered duplicate calls of one real event don't "
                          "get counted as separate false positives. Set to 0 or a negative number to disable. "
                          "Only used when --cluster-selection resolves to 'collapse_overlap' (see below).")
    ap.add_argument("--collapse-mode", default="reciprocal", choices=["reciprocal", "containment"],
                     help="How to score overlap for --collapse-overlap merging. 'reciprocal' (default): both "
                          "calls must be similarly sized -- use when redundant rows report roughly the same "
                          "span (e.g. breakpoint jitter of a few bp). 'containment': the shorter call just "
                          "needs to be almost entirely inside the longer one -- use when some rows are "
                          "genuinely shorter because the underlying read was shorter and didn't reach the far "
                          "breakpoint, not because they're a different event (a 500bp call fully inside a "
                          "5000bp call scores ~0.1 reciprocal but 1.0 containment, so reciprocal would wrongly "
                          "leave it unmerged and it would count as a false positive).")
    ap.add_argument("--max-support-ratio", type=float, default=None,
                     help="If set (e.g. 0.2), only merge two overlapping same-svtype calls when the "
                          "smaller one's support is at most this fraction of the larger one's -- i.e. "
                          "only a call that's clearly a low-support fragment gets absorbed into a bigger "
                          "one, never two comparably-supported calls. Without this, plain overlap-based "
                          "merging is transitive (A-B merge, B-C merge => A,B,C all merge), which can "
                          "silently fuse two DIFFERENT, comparably-supported real events into one if a "
                          "single low-support noise fragment happens to sit between them and overlaps "
                          "both -- verified on a real multi-deletion sample, where this caused several "
                          "real, well-supported deletion calls to vanish (scored as FN) even though they "
                          "didn't overlap each other enough to merge directly. Strongly recommended "
                          "(e.g. 0.2-0.3) whenever a sample can have multiple real, closely-spaced SV "
                          "events of the same type (a multi-deletion sample, say) -- leave unset only if "
                          "you're confident every sample has at most one real event per svtype.")
    ap.add_argument("--cluster-selection", default="auto",
                     choices=["auto", "dominant_only", "collapse_overlap", "all"],
                     help="[cluster_plus_burden] How to reduce multiple cluster rows per event to final calls. "
                          "'auto' (default) = 'collapse_overlap': every row is treated as a genuine call, and "
                          "same-svtype rows within a sample that overlap >= --collapse-overlap get merged into "
                          "one representative (the highest-read-count row -- this naturally recovers a 'dom' "
                          "row if your cluster TSV has one, without hard-filtering to only that row, since a "
                          "'dom' flag on some pipelines just marks the highest-read-count row within a group, "
                          "not a pipeline-selected final call -- the other rows are still real, independent "
                          "calls that need matching, not noise to discard). "
                          "'dominant_only': keep only rows where a 'type' column (distinct from 'orig_type') "
                          "equals 'dom', discarding the rest -- only use this if you've confirmed your pipeline "
                          "actually marks one row per locus as the sole final answer and the rest are internal "
                          "rejected candidates, not independent calls. "
                          "'all': use every row as-is, no merging.")
    ap.add_argument("--require-type-match", action="store_true", default=True)
    ap.add_argument("--time-log", help="Optional /usr/bin/time -v log for this pipeline's run")
    ap.add_argument("--outdir", default="sv_comparison_results")
    ap.add_argument("--compare-with", metavar="PER_SAMPLE_CSV",
                     help="Path to another run's per_sample_summary.csv OR per_sample_summary_weighted.csv "
                          "(produced by this same script, with a different --pipeline-name and --outdir) "
                          "to line up against this run's matching per-sample table side by side, one row "
                          "per sample that exists in the truth table for either run. Which of this run's "
                          "own two tables it's compared against (event-count TP/FP/FN/precision/recall/"
                          "f1/accuracy, or read-weighted TP_reads/FP_reads/weighted_precision/weighted_f1) "
                          "is auto-detected from --compare-with's own columns -- pass either file and you "
                          "get the matching comparison, never a mix of the two. Both runs must have been "
                          "scored against the same --truth-file for this to mean anything -- the merge "
                          "itself doesn't check that, since neither CSV records what truth file it came "
                          "from. Written to per_sample_comparison.csv or per_sample_comparison_weighted.csv "
                          "in this run's --outdir and printed alongside the usual per-sample table.")
    args = ap.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
 
    truth = pd.read_csv(args.truth_file, sep="\t")
    required_cols = {"sample", "svtype", "start", "end", "wraps_origin", "effective_heteroplasmy"}
    missing = required_cols - set(truth.columns)
    if missing:
        ap.error(f"--truth-file is missing required column(s): {sorted(missing)}")
    if truth["wraps_origin"].dtype == object:
        truth["wraps_origin"] = truth["wraps_origin"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    MT_LEN = args.mt_len
    print(f"Loaded {len(truth)} truth events for {truth['sample'].nunique()} samples from {args.truth_file}")
    truth.to_csv(os.path.join(args.outdir, "truth_events.tsv"), sep="\t", index=False)

    # Samples where DEL/DIMER truth events overlap each other break two of the burden-based
    # accuracy metrics (see truth_dels_overlap_in_sample()'s docstring): sample_level_burden_totals
    # already gates itself (returns NaN for these), but the per-TP heteroplasmy regression below
    # doesn't know about samples on its own, so it's filtered out explicitly further down. TP/FP/FN
    # detection accuracy and the burden curve comparison are both unaffected and still cover these
    # samples normally.
    overlapping_truth_samples = truth_samples_with_overlapping_dels(truth, MT_LEN)
    if overlapping_truth_samples:
        print(f"Note: {len(overlapping_truth_samples)} sample(s) have overlapping DEL/DIMER truth "
              f"events ({sorted(overlapping_truth_samples)}) -- sample-level total heteroplasmy and "
              f"per-TP heteroplasmy accuracy will exclude them (not meaningful when truth events "
              f"overlap); detection accuracy and the burden curve comparison are unaffected.")
 

    burden_metric_rows = []
    burden_total_region_rows = []
    burden_total_rows = []
    burden_curve_samples_plotted = 0
    match_dfs = []
 
    if args.pipeline_type == "per_cluster_het":
        if not (args.calls_file or args.calls_glob):
            ap.error("either --calls-file or --calls-glob is required for --pipeline-type per_cluster_het")
        if args.calls_file:
            calls = load_pipeline1_tsv(args.calls_file)
        else:
            call_files_raw = sorted(glob.glob(args.calls_glob))
            # A glob pattern can match directories as well as files (e.g.
            # "*results*" catching a "mitosalt_results_high" folder alongside the
            # TSVs inside it) -- silently trying to read a directory as a TSV
            # fails deep inside pandas with a confusing IsADirectoryError, so
            # filter those out here and say plainly what was skipped.
            call_files = [f for f in call_files_raw if os.path.isfile(f)]
            skipped_dirs = [f for f in call_files_raw if f not in call_files]
            if skipped_dirs:
                print(f"  [warning] --calls-glob matched {len(skipped_dirs)} non-file path(s), skipped: "
                      f"{skipped_dirs}")
            if not call_files:
                ap.error(f"--calls-glob matched no files: {args.calls_glob}")
            calls = pd.concat([load_pipeline1_tsv(f) for f in call_files], ignore_index=True)
            print(f"Loaded {len(calls)} calls from {len(call_files)} files matching {args.calls_glob}")
        samples_present = calls["sample"].unique()
        n_truth_samples = truth["sample"].nunique()
        n_zero_call_samples = len(set(truth["sample"]) - set(samples_present))

        # Evaluate against the FULL truth table -- do not filter to samples_present. A sample
        # with zero call rows is a genuine complete miss (all FN), not something to silently
        # exclude from evaluation; match_events() already handles generating the right FN rows
        # for truth samples absent from calls_df.
        truth_present = truth
        raw_calls_for_sweep = calls.copy()  # before collapsing, for support_threshold_sweep below
 
        collapse_thresh = args.collapse_overlap if args.collapse_overlap > 0 else None
        if collapse_thresh is not None:
            n_before = len(calls)
            calls = collapse_overlapping_calls(calls, MT_LEN, overlap_threshold=collapse_thresh, overlap_mode=args.collapse_mode, max_support_ratio=args.max_support_ratio)
            print(f"Collapsed {n_before} calls -> {len(calls)} after merging same-svtype overlap >= {collapse_thresh}")
 
        match_df = match_events(
            calls, truth_present, MT_LEN,
            min_reciprocal_overlap=args.min_reciprocal_overlap,
            require_type_match=args.require_type_match,
        )
        match_dfs.append(match_df)
 
    else:  # cluster_plus_burden
        if not (args.cluster_glob and args.burden_glob and args.sample_regex):
            ap.error("--cluster-glob, --burden-glob, and --sample-regex are all required for cluster_plus_burden")
 
        cluster_files = sorted(glob.glob(args.cluster_glob))
        pattern = re.compile(args.sample_regex)
        burden_pattern = re.compile(args.burden_sample_regex if args.burden_sample_regex else args.sample_regex)
        burden_format = args.burden_sample_format if args.burden_sample_format else args.sample_format
        raw_calls_list = []
        processed_samples = []
        n_samples_no_burden_match = 0

        # Map truth sample name -> cluster file, built from whatever files actually exist,
        # rather than driving the loop below off the file list directly. This matters: if we
        # loop "for cf in cluster_files", a truth sample this pipeline produced NO cluster
        # file for at all (crashed run, coverage too low, an early return with nothing
        # written) is simply never visited -- it contributes zero rows to match_df and then
        # gets dropped from truth_present too, so its truth events never become FN, they just
        # vanish from scoring entirely. That's a real double standard: the per_cluster_het
        # branch above evaluates against the FULL truth table unconditionally, so a sample
        # pipeline A produced nothing for correctly scores as all-FN there. Looping over truth
        # samples instead (falling back to an empty calls_df when no file matches) makes a
        # complete miss cost the same for both pipeline types.
        sample_to_file = {}
        for cf in cluster_files:
            m = pattern.search(os.path.basename(cf))
            if not m:
                print(f"  [skip] couldn't extract sample name from {cf}", file=sys.stderr)
                continue
            try:
                sample = args.sample_format.format(*m.groups())
            except (IndexError, KeyError) as e:
                ap.error(f"--sample-format {args.sample_format!r} doesn't fit the groups captured by "
                         f"--sample-regex from {os.path.basename(cf)} (captured: {m.groups()}): {e}")
            if sample in sample_to_file:
                print(f"  [warning] sample '{sample}' matched more than one cluster file "
                      f"({sample_to_file[sample]} and {cf}) -- using the first one found", file=sys.stderr)
                continue
            sample_to_file[sample] = cf

        for sample in sorted(truth["sample"].unique()):
            truth_sample = truth[truth["sample"] == sample]
            cf = sample_to_file.get(sample)
            if cf is not None:
                calls = load_pipeline2_cluster_tsv(cf, sample)
            else:
                print(f"  [warning] no cluster file found for truth sample '{sample}' -- "
                      f"scoring as a complete miss (zero calls, every truth event -> FN)")
                calls = pd.DataFrame(columns=["sample", "cluster_id", "svtype", "start", "end",
                                               "read_count", "support", "wraps_origin"])

            raw_calls_list.append(calls.copy())
            processed_samples.append(sample)

            use_dominant = args.cluster_selection == "dominant_only"
            if calls.empty:
                pass  # nothing to filter/collapse -- match_events handles an empty calls_df fine
            elif use_dominant:
                if "cluster_class" not in calls.columns:
                    ap.error("--cluster-selection dominant_only requires a 'type' column distinct from "
                             "'orig_type' in the cluster TSVs (the pipeline's own dominant-cluster flag)")
                n_before = len(calls)
                calls = calls[calls["cluster_class"] == "dom"]
                print(f"  [{sample}] using dominant-cluster rows only: {n_before} -> {len(calls)}")
            elif args.cluster_selection in ("auto", "collapse_overlap"):
                collapse_thresh = args.collapse_overlap if args.collapse_overlap > 0 else None
                if collapse_thresh is not None:
                    calls = collapse_overlapping_calls(calls, MT_LEN, overlap_threshold=collapse_thresh, overlap_mode=args.collapse_mode, max_support_ratio=args.max_support_ratio)
            # else args.cluster_selection == "all": use every row as-is, no filtering/collapsing

            # locate the matching burden file: try substring match first (fast path for the
            # common case), then fall back to applying the same --sample-regex/--sample-format
            # transform used for cluster filenames to burden filenames too, in case burden files
            # follow a similar naming convention with different filename mismatches (e.g. the same
            # missing-underscore issue --sample-format was built to fix for cluster files).
            all_burden_files = glob.glob(args.burden_glob)
            burden_candidates = [b for b in all_burden_files if sample in os.path.basename(b)]
            if not burden_candidates:
                for b in all_burden_files:
                    m2 = burden_pattern.search(os.path.basename(b))
                    if m2:
                        try:
                            candidate_sample = burden_format.format(*m2.groups())
                        except (IndexError, KeyError):
                            continue
                        if candidate_sample == sample:
                            burden_candidates = [b]
                            break
 
            if burden_candidates:
                burden_df = load_burden_tsv(burden_candidates[0])
                calls = attach_deletion_burden_estimate(calls, burden_df)
 
                curve = truth_burden_curve(sample, truth_sample, MT_LEN)
                metrics, merged = compare_burden_curves(curve, burden_df)
                metrics.update({"pipeline": args.pipeline_name, "sample": sample})
                burden_metric_rows.append(metrics)
 
                region_df, true_total, called_total = sample_level_burden_totals(sample, truth_sample, burden_df, MT_LEN)
                if not region_df.empty:
                    region_df["pipeline"] = args.pipeline_name
                    burden_total_region_rows.append(region_df)
                    burden_total_rows.append({
                        "pipeline": args.pipeline_name, "sample": sample,
                        "n_del_regions": len(region_df),
                        "true_hf": true_total, "called_hf": called_total,
                    })
 
                if burden_curve_samples_plotted < 5:  # cap number of example plots
                    plot_burden_curve(
                        merged, sample, args.pipeline_name,
                        os.path.join(args.outdir, f"burden_curve_{sample}.png"),
                    )
                    burden_curve_samples_plotted += 1
            else:
                n_samples_no_burden_match += 1
                print(f"  [warning] no burden file matched for sample '{sample}'")
 
            match_df = match_events(
                calls, truth_sample, MT_LEN,
                min_reciprocal_overlap=args.min_reciprocal_overlap,
                require_type_match=args.require_type_match,
            )
            match_dfs.append(match_df)
 
        raw_calls_for_sweep = pd.concat(raw_calls_list, ignore_index=True) if raw_calls_list else pd.DataFrame()
        # processed_samples now always equals every truth sample (see the loop above -- a
        # missing cluster file no longer skips the sample, it scores as a complete miss), so
        # this is truth in full, same as the per_cluster_het branch.
        truth_present = truth[truth["sample"].isin(processed_samples)]
        if n_samples_no_burden_match > 0:
            print(f"\n  [summary] {n_samples_no_burden_match}/{len(processed_samples)} samples had no "
                  f"matching burden file -- this is very likely why heteroplasmy accuracy is empty or "
                  f"incomplete below. See the [warning] lines above for which samples and why.")
 
    match_df = pd.concat(match_dfs, ignore_index=True)
    match_df = flag_colocated_fps(match_df, MT_LEN, overlap_threshold=args.min_reciprocal_overlap,
                                   overlap_mode=args.collapse_mode)
    match_df.to_csv(os.path.join(args.outdir, "matched_events.tsv"), sep="\t", index=False)
 
    # ---- Overall detection accuracy ----
    overall = summarize_matches(match_df)
    overall["pipeline"] = args.pipeline_name
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(os.path.join(args.outdir, "overall_summary.csv"), index=False)
    print("\n=== Overall SV detection accuracy (pooled / micro-averaged, strict typing) ===")
    print(overall_df.to_string(index=False))

    # ---- Read-weighted precision: same TP/FP rows, but a call's contribution to
    # precision is weighted by its read support rather than counted as one equal
    # unit. Matters when collapsing can't safely absorb low-support noise fragments
    # into their dominant cluster (e.g. multi-deletion samples, where several real
    # events genuinely overlap and a containment-style collapse would wrongly merge
    # distinct true events) -- those fragments still show up as their own FP rows,
    # but a 1-read fragment next to a 5774-read TP barely moves this number. ----
    weighted = summarize_matches_weighted(match_df)
    weighted["pipeline"] = args.pipeline_name
    weighted_df = pd.DataFrame([weighted])
    weighted_df.to_csv(os.path.join(args.outdir, "overall_summary_weighted.csv"), index=False)
    print("\n=== Read-weighted accuracy (FP/TP weighted by read support, not equal-count) ===")
    print(weighted_df.to_string(index=False))

    # ---- Adjusted accuracy: DEL calls at an already-correctly-found DIMER locus don't
    # count as a separate false positive (see flag_colocated_fps docstring for why this is
    # restricted to exactly that pairing, not a general "any wrong type near a TP" rule) ----
    n_colocated = int(match_df["co_located_with_tp"].sum()) if "co_located_with_tp" in match_df.columns else 0
    if n_colocated > 0:
        adjusted = summarize_matches_adjusted(match_df)
        adjusted["pipeline"] = args.pipeline_name
        pd.DataFrame([adjusted]).to_csv(os.path.join(args.outdir, "overall_summary_adjusted.csv"), index=False)
        print(f"\n=== Adjusted accuracy ({n_colocated} DEL call(s) at a correctly-found DIMER "
              f"locus excluded from FP) ===")
        print(pd.DataFrame([adjusted]).to_string(index=False))
 
    # ---- Per-sample detection accuracy + macro average ----
    # A sample with 12 truth events contributes 12x the TP/FP/FN of a sample with 1 to the
    # pooled numbers above -- this can make the pooled F1 look dominated by whichever sample
    # happens to have the most events, not necessarily the ones that matter most to you.
    # The per-sample table + macro average give every sample equal weight regardless of size.
    per_sample_df = summarize_matches_per_sample(match_df)
    per_sample_df["pipeline"] = args.pipeline_name
    per_sample_df.to_csv(os.path.join(args.outdir, "per_sample_summary.csv"), index=False)
    print("\n=== Per-sample detection accuracy ===")
    print(per_sample_df.to_string(index=False))
    plot_f1_per_sample(per_sample_df, args.pipeline_name, os.path.join(args.outdir, "f1_per_sample.png"))

    # ---- Per-sample read-weighted accuracy ----
    # The per-sample event-count table above can look alarming for a sample where a real,
    # well-supported call sits next to several low-support noise fragments (e.g. 2 TP / 14 FP
    # by event count) -- this table shows the same samples with FP/TP weighted by read support,
    # so you can see directly whether those FP rows are actually stray 1-2 read fragments (weighted
    # precision near 1.0) or genuine competing evidence (weighted precision noticeably lower),
    # without having to compute it by hand.
    per_sample_weighted_df = summarize_matches_weighted_per_sample(match_df)
    per_sample_weighted_df["pipeline"] = args.pipeline_name
    per_sample_weighted_df.to_csv(os.path.join(args.outdir, "per_sample_summary_weighted.csv"), index=False)
    print("\n=== Per-sample read-weighted accuracy ===")
    print(per_sample_weighted_df.to_string(index=False))

    # ---- Side-by-side comparison against another pipeline's run ----
    # Both per_sample_df (event-count) and per_sample_weighted_df (read-weighted) are computed
    # the exact same way regardless of --pipeline-type, off the same truth table -- so as long as
    # --compare-with points at the matching CSV from the other pipeline's run (scored against the
    # same --truth-file), these numbers are directly comparable row for row. --compare-with accepts
    # EITHER a per_sample_summary.csv or a per_sample_summary_weighted.csv: the schema is detected
    # from whichever set of expected column names overlaps better, and the comparison is built
    # against this run's matching table (event-count vs event-count, or weighted vs weighted) --
    # never mixed, since "precision" and "weighted_precision" aren't the same quantity.
    if args.compare_with:
        other_df = pd.read_csv(args.compare_with)
        if "pipeline" not in other_df.columns or other_df["pipeline"].nunique() != 1:
            print(f"  [warning] {args.compare_with} doesn't look like a single-pipeline "
                  f"per-sample summary CSV (missing or multi-valued 'pipeline' column) -- "
                  f"comparison merge skipped.")
        else:
            other_name = other_df["pipeline"].iloc[0]
            if other_name == args.pipeline_name:
                print(f"  [warning] --compare-with has the same --pipeline-name ('{other_name}') as "
                      f"this run -- rename one of them so the merged columns aren't ambiguous. "
                      f"Comparison merge skipped.")
            else:
                event_overlap = sum(c in other_df.columns for c in EVENT_COMPARISON_COLS)
                weighted_overlap = sum(c in other_df.columns for c in WEIGHTED_COMPARISON_COLS)
                if weighted_overlap > event_overlap:
                    _merge_per_sample_comparison(
                        per_sample_weighted_df, other_df, args.pipeline_name, other_name,
                        WEIGHTED_COMPARISON_COLS, args.outdir,
                        "per_sample_comparison_weighted.csv", "read-weighted")
                else:
                    _merge_per_sample_comparison(
                        per_sample_df, other_df, args.pipeline_name, other_name,
                        EVENT_COMPARISON_COLS, args.outdir,
                        "per_sample_comparison.csv", "event-count")

    macro = macro_average_summary(per_sample_df)
    macro["pipeline"] = args.pipeline_name
    pd.DataFrame([macro]).to_csv(os.path.join(args.outdir, "macro_average_summary.csv"), index=False)
    print("\n=== Macro-averaged accuracy (every sample weighted equally) ===")
    print(pd.DataFrame([macro]).to_string(index=False))
 
    n_p, n_r, n_f1 = macro["n_samples_with_defined_precision"], macro["n_samples_with_defined_recall"], macro["n_samples_with_defined_f1"]
 
    # ---- Per-type accuracy ----
    per_type_df = summarize_by_type(match_df)
    per_type_df["pipeline"] = args.pipeline_name
    per_type_df.to_csv(os.path.join(args.outdir, "per_type_summary.csv"), index=False)
    print("\n=== Per-SV-type accuracy ===")
    print(per_type_df.to_string(index=False))
    plot_f1_by_type(per_type_df, os.path.join(args.outdir, "f1_by_type.png"))
 
    # ---- Does read support distinguish real calls from false ones? ----
    # A plain F1 treats an 8000-read TP the same as a 1-read TP, and a
    # 1-read FP the same as an 8000-read FP -- these two diagnostics show
    # whether that's hiding something: (1) a direct TP-vs-FP support
    # comparison, (2) what precision/recall/F1 would look like if you
    # pre-filtered calls by a minimum support threshold.
    if "called_support" in match_df.columns and match_df["called_support"].notna().any():
        plot_tp_fp_support_distribution(match_df, args.pipeline_name,
                                         os.path.join(args.outdir, "tp_fp_support_distribution.png"))
 
        if not raw_calls_for_sweep.empty and "support" in raw_calls_for_sweep.columns:
            collapse_thresh_for_sweep = args.collapse_overlap if args.collapse_overlap > 0 else None
            sweep_df = support_threshold_sweep(
                raw_calls_for_sweep, truth_present, MT_LEN,
                min_reciprocal_overlap=args.min_reciprocal_overlap,
                require_type_match=args.require_type_match,
                collapse_overlap_threshold=collapse_thresh_for_sweep,
                collapse_overlap_mode=args.collapse_mode,
                max_support_ratio=args.max_support_ratio,
            )
            sweep_df["pipeline"] = args.pipeline_name
            sweep_df.to_csv(os.path.join(args.outdir, "support_threshold_sweep.csv"), index=False)
            print("\n=== Accuracy vs. minimum support threshold ===")
            print(sweep_df.to_string(index=False))
            plot_support_threshold_sweep(sweep_df, args.pipeline_name,
                                          os.path.join(args.outdir, "support_threshold_sweep.png"))
 
    # ---- Heteroplasmy accuracy (on matched TP events only) ----
    # Excludes samples with overlapping DEL/DIMER truth events -- a matched call's estimated
    # heteroplasmy (attach_deletion_burden_estimate) is a mean over the call's own span, which
    # picks up burden contributed by any OTHER overlapping truth event too, not just the one
    # it's being scored against here. See truth_dels_overlap_in_sample()'s docstring.
    tp_df = match_df[
        (match_df["match_status"] == "TP") & (~match_df["sample"].isin(overlapping_truth_samples))
    ].dropna(subset=["called_heteroplasmy"])
    n_tp_excluded_overlap = int(
        ((match_df["match_status"] == "TP") & (match_df["sample"].isin(overlapping_truth_samples))).sum()
    )
    if n_tp_excluded_overlap > 0:
        print(f"\n({n_tp_excluded_overlap} TP event(s) excluded from heteroplasmy accuracy below -- "
              f"their sample has overlapping DEL/DIMER truth events)")
    het_df = tp_df.rename(columns={"effective_heteroplasmy": "true_hf", "called_heteroplasmy": "called_hf"})
    het_df = het_df.astype({"true_hf": float, "called_hf": float})
    if not het_df.empty:
        het_metrics = regression_metrics(het_df)
        het_metrics["pipeline"] = args.pipeline_name
        pd.DataFrame([het_metrics]).to_csv(os.path.join(args.outdir, "heteroplasmy_accuracy_summary.csv"), index=False)
        print("\n=== Heteroplasmy accuracy (matched TP events) ===")
        print(pd.DataFrame([het_metrics]).to_string(index=False))
        plot_bland_altman(het_df, args.pipeline_name, os.path.join(args.outdir, "bland_altman_heteroplasmy.png"))
        plot_true_vs_called_scatter({args.pipeline_name: het_df}, os.path.join(args.outdir, "heteroplasmy_scatter.png"))
    else:
        print("\n(No heteroplasmy values available on matched events -- skipping heteroplasmy accuracy plots.)")
 
    # ---- Burden curve accuracy (cluster_plus_burden only) ----
    if burden_metric_rows:
        burden_metrics_df = pd.DataFrame(burden_metric_rows)
        burden_metrics_df.to_csv(os.path.join(args.outdir, "burden_curve_accuracy.csv"), index=False)
        print("\n=== Deletion burden curve accuracy (per sample) ===")
        print(burden_metrics_df.to_string(index=False))
        plot_burden_error_summary(burden_metrics_df, os.path.join(args.outdir, "burden_curve_mae_by_sample.png"))
 
    # ---- Sample-level total heteroplasmy: sum of per-region max burden vs. sum of truth ----
    # (handles samples with multiple non-overlapping DEL regions, e.g. major + minor arc,
    #  where a single genome-wide max would only capture one of them)
    if burden_total_rows:
        region_detail_df = pd.concat(burden_total_region_rows, ignore_index=True)
        region_detail_df.to_csv(os.path.join(args.outdir, "burden_total_region_detail.csv"), index=False)
 
        total_df = pd.DataFrame(burden_total_rows)
        total_df.to_csv(os.path.join(args.outdir, "burden_sample_totals.csv"), index=False)
        print("\n=== Sample-level total heteroplasmy (sum of per-DEL-region max burden) ===")
        print(total_df.to_string(index=False))
 
        total_df_clean = total_df.dropna(subset=["true_hf", "called_hf"])
        if not total_df_clean.empty:
            total_metrics = regression_metrics(total_df_clean)
            total_metrics["pipeline"] = args.pipeline_name
            pd.DataFrame([total_metrics]).to_csv(os.path.join(args.outdir, "burden_total_accuracy_summary.csv"), index=False)
            print("\n=== Sample-level total heteroplasmy accuracy ===")
            print(pd.DataFrame([total_metrics]).to_string(index=False))
            plot_bland_altman(total_df_clean, args.pipeline_name, os.path.join(args.outdir, "bland_altman_burden_totals.png"))
            plot_true_vs_called_scatter({args.pipeline_name: total_df_clean}, os.path.join(args.outdir, "burden_totals_scatter.png"))
 
    # ---- Resource usage ----
    if args.time_log:
        stats = parse_time_v_log(args.time_log)
        stats["pipeline"] = args.pipeline_name
        resource_df = pd.DataFrame([stats])
        resource_df.to_csv(os.path.join(args.outdir, "resource_usage.csv"), index=False)
        print("\n=== Resource usage ===")
        print(resource_df.to_string(index=False))
        # (bar chart needs >=2 pipelines to be meaningful side by side --
        #  run this script once per pipeline, then concat the resource_usage.csv
        #  files and call plot_resource_usage yourself, or see the combined
        #  example in the accompanying README for a 2-pipeline overlay.)
 
    print(f"\nAll tables and plots written to: {args.outdir}/")
 
 
if __name__ == "__main__":
    main()
