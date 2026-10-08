#!/usr/bin/env python3


import io
import os
import re
 
import numpy as np
import pandas as pd
 
 
# --------------------------------------------------------------------------
# Circular-aware interval overlap
# --------------------------------------------------------------------------
def _split_wrapped_interval(start, end, wraps, mt_len):
    """Return a list of non-wrapping (start, end) sub-intervals for one call/event.
    If wraps: split into [start, mt_len) and [0, end).
    """
    if wraps:
        return [(start, mt_len), (0, end)]
    return [(min(start, end), max(start, end))]
 
 
def _overlap_len(intervals_a, intervals_b):
    total = 0
    for a_s, a_e in intervals_a:
        for b_s, b_e in intervals_b:
            lo = max(a_s, b_s)
            hi = min(a_e, b_e)
            if hi > lo:
                total += hi - lo
    return total
 
 
def _intervals_overlap_any(intervals_a, intervals_b) -> bool:
    for a_s, a_e in intervals_a:
        for b_s, b_e in intervals_b:
            if min(a_e, b_e) > max(a_s, b_s):
                return True
    return False


def truth_dels_overlap_in_sample(sample_truth_df: pd.DataFrame, mt_len: int) -> bool:
    """Whether this sample's DEL/DIMER truth events overlap each other at all.

    sample_level_burden_totals() and the per-TP heteroplasmy accuracy metric
    (built from attach_deletion_burden_estimate()) both silently assume they
    don't. When DEL/DIMER truth events DO overlap (e.g. a scattered
    multi-deletion sample where several distinct deletion species share
    overlapping spans), a position's true burden is the SUM of every event
    covering it -- exactly what truth_burden_curve() computes. That breaks
    both of those metrics in different ways:
      - sample_level_burden_totals() sums each event's independently
        computed "max called burden within its own span" across events --
        wherever two truth spans overlap, that shared region's (already
        summed) called burden gets counted again for every event it
        overlaps, inflating called_total well above what was actually
        observed (verified: an otherwise-perfect caller on a real
        multi-deletion truth set here came out ~9x too high).
      - attach_deletion_burden_estimate() averages a matched call's OWN
        burden over its OWN span to estimate its heteroplasmy. If other
        truth events also overlap that span, the average picks up their
        contribution too, so it's no longer just describing the one truth
        event it's being scored against in the heteroplasmy regression/
        Bland-Altman plot (verified: came out ~12x too high in the same
        test).
    Neither problem touches compare_burden_curves(): it compares truth and
    called burden position-by-position and never attributes a single number
    to one truth event, so it stays valid regardless of overlap -- that's
    the metric to trust for samples flagged by this function.
    """
    sub = sample_truth_df[sample_truth_df["svtype"].isin(["DEL", "DIMER"])]
    events = list(sub[["start", "end", "wraps_origin"]].itertuples(index=False, name="Event"))
    for i in range(len(events)):
        ivs_i = _split_wrapped_interval(events[i].start, events[i].end, events[i].wraps_origin, mt_len)
        for j in range(i + 1, len(events)):
            ivs_j = _split_wrapped_interval(events[j].start, events[j].end, events[j].wraps_origin, mt_len)
            if _intervals_overlap_any(ivs_i, ivs_j):
                return True
    return False


def truth_samples_with_overlapping_dels(truth_df: pd.DataFrame, mt_len: int) -> set:
    """Set of sample names where DEL/DIMER truth events overlap each other --
    see truth_dels_overlap_in_sample() for why this matters."""
    flagged = set()
    for sample, sub in truth_df.groupby("sample"):
        if truth_dels_overlap_in_sample(sub, mt_len):
            flagged.add(sample)
    return flagged


def _overlap_metrics(a_start, a_end, a_wraps, b_start, b_end, b_wraps, mt_len):
    """Shared computation for both overlap metrics below. Returns (reciprocal, containment)."""
    ivs_a = _split_wrapped_interval(a_start, a_end, a_wraps, mt_len)
    ivs_b = _split_wrapped_interval(b_start, b_end, b_wraps, mt_len)
    len_a = sum(e - s for s, e in ivs_a)
    len_b = sum(e - s for s, e in ivs_b)
    if len_a == 0 or len_b == 0:
        return 0.0, 0.0
    ov = _overlap_len(ivs_a, ivs_b)
    reciprocal = min(ov / len_a, ov / len_b)
    containment = ov / min(len_a, len_b)
    return reciprocal, containment
 
 
def circular_reciprocal_overlap(a_start, a_end, a_wraps, b_start, b_end, b_wraps, mt_len):
    """Reciprocal overlap fraction between two (possibly origin-spanning) intervals.
    Returns min(overlap/len_a, overlap/len_b) in [0, 1], analogous to bedtools -f -r.
    Requires BOTH intervals to be similarly sized -- a short interval fully
    inside a much longer one scores low here even though it's fully contained.
    Appropriate for merging duplicate reports of the *same-sized* event.
    """
    r, _ = _overlap_metrics(a_start, a_end, a_wraps, b_start, b_end, b_wraps, mt_len)
    return r
 
 
def circular_containment_overlap(a_start, a_end, a_wraps, b_start, b_end, b_wraps, mt_len):
    """Containment fraction: overlap / min(len_a, len_b), in [0, 1]. Scores 1.0
    whenever the shorter interval is (almost) entirely inside the longer one,
    regardless of how much bigger the longer one is. Appropriate for merging
    calls where a shorter call is just a partial-support version of a longer
    one at the same breakpoint (e.g. a shorter sequencing read only reached
    part of the true deletion span) rather than a genuinely different event.
    """
    _, c = _overlap_metrics(a_start, a_end, a_wraps, b_start, b_end, b_wraps, mt_len)
    return c
 
 
def circular_breakpoint_distance(a_start, a_end, a_wraps, b_start, b_end, b_wraps, mt_len):
    """Min total breakpoint distance (|dstart| + |dend|), accounting for wraparound
    by comparing raw wrapped coordinates with modular distance.
    """
    def mod_dist(x, y):
        d = abs(x - y)
        return min(d, mt_len - d)
    return mod_dist(a_start, b_start) + mod_dist(a_end, b_end)
 
 
# --------------------------------------------------------------------------
# Matching calls to truth (per sample)
# --------------------------------------------------------------------------
def collapse_overlapping_calls(
    calls_df: pd.DataFrame,
    mt_len: int,
    overlap_threshold: float = 0.8,
    support_col: str = "support",
    heteroplasmy_agg: str = "representative",
    overlap_mode: str = "reciprocal",
    max_support_ratio: float = None,
) -> pd.DataFrame:
    """Merge redundant overlapping calls of the same svtype within a sample
    into one representative call, before matching against truth.
 
    Many cluster-based callers report one row per supporting read cluster
    rather than one row per final event, so several rows can describe the
    same real breakpoint with slightly different start/end positions (e.g.
    soft-clip position jitter). Left unmerged, only one of these rows can
    match the single truth event (matching is 1-to-1), and every other row
    gets counted as a separate false positive -- which misrepresents a
    single noisy-but-correct call as many wrong ones.
 
    Calls are greedily clustered (union-find) within each (sample, svtype)
    group: any two calls whose overlap (per `overlap_mode`) is >=
    overlap_threshold are merged into the same cluster, and clusters are
    transitively connected (A-B merge, B-C merge => A,B,C become one
    cluster) since jittered/truncated breakpoints from the same real event
    often don't all pairwise-overlap at a strict threshold, but chain
    together through shared overlap.

    WARNING -- transitive chaining can silently merge two genuinely
    DIFFERENT, comparably-supported real events if a low-support noise
    fragment happens to sit geometrically between them and overlaps both
    (e.g. a single stray read whose apparent span bridges two real,
    well-supported deletion clusters that don't overlap enough with EACH
    OTHER to merge on their own). This isn't hypothetical: verified on a
    real multi-deletion sample where a 1-read fragment bridged two ~300-
    and ~550-read real clusters, causing one of them to vanish as a
    separate call and its truth event to score FN, purely because of the
    bridge -- not a detection failure. Passing `max_support_ratio` (below)
    prevents this by refusing to merge two calls unless one is clearly
    much smaller than the other, which a plain overlap-threshold union
    can't distinguish on its own (both the real A-B pair and the noise
    A-fragment pair can satisfy the same overlap threshold).
 
    `overlap_mode` controls what counts as "the same event":
      - 'reciprocal' (default): both calls must be similarly *sized* --
        overlap / max(len_a, len_b) style. Appropriate when redundant
        duplicate rows all report roughly the same span (e.g. breakpoint
        position jitter of a few bp either way).
      - 'containment': the shorter call just needs to be (almost) entirely
        inside the longer one -- overlap / min(len_a, len_b). Appropriate
        when some rows are genuinely shorter *because the read behind them
        was shorter and didn't reach the far breakpoint*, not because
        they're a different event -- reciprocal overlap would wrongly
        reject these (a 500bp span inside a 5000bp span scores ~0.1
        reciprocal but 1.0 containment).
 
    The representative call's start/end/heteroplasmy are all taken directly
    from the row with the highest `support_col` value (default 'support' --
    alt.reads / read_count, whichever loader populated it); ties broken by
    largest span. This is the default because higher read support generally
    means a *more reliable* local heteroplasmy estimate, not less -- a
    handful of low-support duplicate rows (soft-clip position jitter from
    just a few reads each) shouldn't be allowed to drag down a well-supported
    estimate through averaging. `heteroplasmy_agg` can override this:
      - 'representative' (default): use the highest-support row's own value.
      - 'median': median across all merged rows -- more robust if you don't
        trust support as a reliability signal for your caller.
      - 'weighted_mean': support-weighted mean across all merged rows.
      - 'mean': plain unweighted mean.
 
    A note on data quality: heteroplasmy is expected to be a fraction in
    [0, 1] by this point (convert your loader's raw units -- e.g. percent --
    before this function runs). If any input row is outside that range after
    conversion, that's a real signal something's off upstream, and a warning
    is printed (not silently dropped/clipped).
 
    If overlap_threshold is None, this is a no-op (returns calls_df unchanged)
    -- useful if your pipeline already reports one row per final event.

    `max_support_ratio` (default None = off, preserves prior behaviour):
    when set (e.g. 0.2), two calls are only merged if the smaller one's
    support is at most this fraction of the larger one's (smaller/larger
    <= max_support_ratio) -- i.e. only a call that's CLEARLY a low-support
    fragment gets absorbed, never two comparably-supported calls, directly
    or through a chain. This changes the algorithm from transitive
    union-find to greedy support-descending absorption: calls are visited
    highest-support first; each unclaimed call becomes a new representative
    and claims every remaining unclaimed call that satisfies both the
    overlap threshold AND the support-ratio test; no representative can
    ever be absorbed into another (comparably-supported real events always
    stay separate calls), which also rules out the noise-bridge chaining
    problem described above by construction (a claimed low-support call is
    removed from the pool, so it can never link two representatives
    together). Requires `support_col` to be present; if it isn't, this
    falls back to the overlap-only union-find behaviour with a warning.
    """
    if overlap_threshold is None:
        return calls_df

    has_support = support_col in calls_df.columns
    has_het = "heteroplasmy" in calls_df.columns

    if has_het:
        bad_het = calls_df[(calls_df["heteroplasmy"] < 0) | (calls_df["heteroplasmy"] > 1)]
        if not bad_het.empty:
            print(f"  [warning] {len(bad_het)} call(s) have heteroplasmy outside [0,1] after unit "
                  f"conversion (e.g. {bad_het['heteroplasmy'].iloc[0]:.4f}) -- check the loader's "
                  f"percent/fraction handling, or this may be a genuine upstream data quality issue.")

    if max_support_ratio is not None and not has_support:
        print(f"  [warning] max_support_ratio={max_support_ratio} was set but '{support_col}' isn't "
              f"in calls_df -- falling back to plain overlap-threshold merging (transitive union-find), "
              f"which can wrongly bridge two distinct real events through a low-support fragment.")

    overlap_fn = circular_containment_overlap if overlap_mode == "containment" else circular_reciprocal_overlap

    kept_rows = []
    group_cols = ["sample", "svtype"] if "sample" in calls_df.columns else ["svtype"]
    for _, group in calls_df.groupby(group_cols):
        group = group.reset_index(drop=True)
        n = len(group)

        if max_support_ratio is not None and has_support:
            # Greedy support-descending absorption -- see docstring.
            order = sorted(range(n), key=lambda i: -group.loc[i, support_col])
            claimed = [False] * n
            clusters = {}
            for i in order:
                if claimed[i]:
                    continue
                claimed[i] = True
                ri = group.loc[i]
                members = [i]
                for j in order:
                    if claimed[j] or j == i:
                        continue
                    rj = group.loc[j]
                    if rj[support_col] > ri[support_col]:
                        continue
                    ratio = (rj[support_col] / ri[support_col]) if ri[support_col] > 0 else 0.0
                    if ratio > max_support_ratio:
                        continue
                    ov = overlap_fn(
                        ri["start"], ri["end"], ri.get("wraps_origin", False),
                        rj["start"], rj["end"], rj.get("wraps_origin", False),
                        mt_len,
                    )
                    if ov >= overlap_threshold:
                        claimed[j] = True
                        members.append(j)
                clusters[i] = members
        else:
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(n):
                for j in range(i + 1, n):
                    ri, rj = group.loc[i], group.loc[j]
                    ov = overlap_fn(
                        ri["start"], ri["end"], ri.get("wraps_origin", False),
                        rj["start"], rj["end"], rj.get("wraps_origin", False),
                        mt_len,
                    )
                    if ov >= overlap_threshold:
                        union(i, j)

            clusters = {}
            for i in range(n):
                clusters.setdefault(find(i), []).append(i)

        for idxs in clusters.values():
            sub = group.loc[idxs]
            if has_support:
                rep = sub.loc[sub[support_col].idxmax()].copy()
            else:
                rep = sub.loc[(sub["end"] - sub["start"]).abs().idxmax()].copy()
 
            if has_het and len(idxs) > 1 and heteroplasmy_agg != "representative":
                if heteroplasmy_agg == "median":
                    rep["heteroplasmy"] = sub["heteroplasmy"].median()
                elif heteroplasmy_agg == "weighted_mean" and has_support and sub[support_col].sum() > 0:
                    rep["heteroplasmy"] = np.average(sub["heteroplasmy"], weights=sub[support_col])
                elif heteroplasmy_agg == "mean":
                    rep["heteroplasmy"] = sub["heteroplasmy"].mean()
 
            rep["n_merged"] = len(idxs)
            kept_rows.append(rep)
 
    return pd.DataFrame(kept_rows).reset_index(drop=True)
 
 
def types_compatible(truth_svtype: str, called_svtype: str) -> bool:
    """Whether a called svtype counts as matching a truth svtype. Strict
    exact match -- a DIMER truth event requires an actual DIMER call, a
    plain DEL call at that locus does not count as a match (it's scored as
    a false positive, and the DIMER truth event as a false negative, if
    nothing DIMER-typed is found there). This is a deliberate choice: some
    users want a plain 'del' credited as partial evidence for a dimer
    (biologically its breakpoint is a real deletion junction), others want
    strict classification credit where only actually calling it a dimer
    counts as finding it. This function implements the strict version --
    a DIMER truth event still requires an actual DIMER call to count as TP
    (see flag_colocated_fps() for how the double-penalty on the co-located
    DEL call is handled separately, without changing this).
    """
    return truth_svtype == called_svtype
 
 
def match_events(
    calls_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    mt_len: int,
    min_reciprocal_overlap: float = 0.5,
    max_breakpoint_dist: int = None,
    require_type_match: bool = True,
    support_col: str = "support",
    overlap_tie_bucket: float = 0.01,
) -> pd.DataFrame:
    """Greedy best-match assignment between calls and truth events, done
    independently per sample. Returns a long-format DataFrame with one row
    per truth event or unmatched call:
 
        sample, match_status ('TP'/'FN'/'FP'), truth_svtype, called_svtype,
        truth_start, truth_end, called_start, called_end,
        reciprocal_overlap, breakpoint_dist,
        effective_heteroplasmy (truth), called_heteroplasmy (if present in calls_df)

    Candidates are ranked primarily by overlap, but overlap alone is a noisy
    tie-break when several calls are all near-perfect matches for the same
    truth event: a single-read fragment's apparent breakpoints can, purely
    by chance, land a hair closer to truth than a well-supported cluster's
    consensus breakpoints do (verified on a real multi-deletion sample: a
    1-read fragment beat a 917-read cluster by a reciprocal-overlap margin
    of 0.00014 -- noise in the estimate, not evidence). Left alone, that
    lets the greedy assignment hand the truth event to the noise fragment
    and wrongly score the real, well-supported cluster as a separate FP.
    `overlap_tie_bucket` (default 0.01) buckets overlap to that resolution
    before sorting, so overlaps within the same ~1% band are treated as
    equal and `support_col` (default 'support') decides between them --
    i.e. among calls that already clear the overlap threshold and are
    roughly equally good geometric matches, the one with more supporting
    evidence wins the truth event, which is what should happen. Overlap
    still fully controls which calls qualify at all (nothing below
    `min_reciprocal_overlap` is ever a candidate) and still decides between
    calls in different overlap buckets -- this only changes ties within a
    bucket. Set `overlap_tie_bucket=0` to disable and rank by overlap alone
    (the old behaviour). Falls back to overlap-only ranking with a warning
    if `support_col` isn't present in calls_df.
    """
    results = []
    samples = set(truth_df["sample"]).union(calls_df["sample"] if "sample" in calls_df else [])
 
    has_support = support_col in calls_df.columns
    if overlap_tie_bucket and not has_support:
        print(f"  [warning] match_events: '{support_col}' not in calls_df -- overlap ties will be "
              f"broken arbitrarily instead of by support (a low-support fragment could beat a "
              f"well-supported call purely on overlap noise)")

    for sample in samples:
        t_sub = truth_df[truth_df["sample"] == sample].reset_index(drop=True)
        c_sub = calls_df[calls_df["sample"] == sample].reset_index(drop=True) if "sample" in calls_df else calls_df.reset_index(drop=True)
 
        # score all truth x call pairs (optionally restricted to matching svtype)
        candidates = []
        for ti, t in t_sub.iterrows():
            for ci, c in c_sub.iterrows():
                if require_type_match and not types_compatible(t["svtype"], c["svtype"]):
                    continue
                ov = circular_reciprocal_overlap(
                    t["start"], t["end"], t["wraps_origin"],
                    c["start"], c["end"], c.get("wraps_origin", False),
                    mt_len,
                )
                bd = circular_breakpoint_distance(
                    t["start"], t["end"], t["wraps_origin"],
                    c["start"], c["end"], c.get("wraps_origin", False),
                    mt_len,
                )
                ok = ov >= min_reciprocal_overlap
                if max_breakpoint_dist is not None:
                    ok = ok and bd <= max_breakpoint_dist
                if ok:
                    if overlap_tie_bucket and has_support:
                        bucket = round(ov / overlap_tie_bucket)
                        sup = c.get(support_col, 0)
                        sup = sup if pd.notna(sup) else 0
                        candidates.append((bucket, sup, ov, -bd, ti, ci))
                    else:
                        candidates.append((ov, -bd, ti, ci))
 
        # best overlap-bucket first, then (if support-aware) highest support, then
        # exact overlap, then smallest breakpoint distance
        candidates.sort(reverse=True)
        used_truth, used_calls = set(), set()
        for *_, ov, neg_bd, ti, ci in candidates:
            if ti in used_truth or ci in used_calls:
                continue
            used_truth.add(ti)
            used_calls.add(ci)
            t, c = t_sub.loc[ti], c_sub.loc[ci]
            results.append({
                "sample": sample, "match_status": "TP",
                "truth_svtype": t["svtype"], "called_svtype": c["svtype"],
                "truth_start": t["start"], "truth_end": t["end"],
                "called_start": c["start"], "called_end": c["end"],
                "reciprocal_overlap": ov, "breakpoint_dist": -neg_bd,
                "effective_heteroplasmy": t["effective_heteroplasmy"],
                "called_heteroplasmy": c.get("heteroplasmy", np.nan),
                "called_support": c.get("support", np.nan),
                "event_label": t.get("event_label", None),
            })
 
        for ti, t in t_sub.iterrows():
            if ti not in used_truth:
                results.append({
                    "sample": sample, "match_status": "FN",
                    "truth_svtype": t["svtype"], "called_svtype": None,
                    "truth_start": t["start"], "truth_end": t["end"],
                    "called_start": None, "called_end": None,
                    "reciprocal_overlap": None, "breakpoint_dist": None,
                    "effective_heteroplasmy": t["effective_heteroplasmy"],
                    "called_heteroplasmy": None,
                    "called_support": None,
                    "event_label": t.get("event_label", None),
                })
        for ci, c in c_sub.iterrows():
            if ci not in used_calls:
                results.append({
                    "sample": sample, "match_status": "FP",
                    "truth_svtype": None, "called_svtype": c["svtype"],
                    "truth_start": None, "truth_end": None,
                    "called_start": c["start"], "called_end": c["end"],
                    "reciprocal_overlap": None, "breakpoint_dist": None,
                    "effective_heteroplasmy": None,
                    "called_heteroplasmy": c.get("heteroplasmy", np.nan),
                    "called_support": c.get("support", np.nan),
                    "event_label": None,
                })
 
    return pd.DataFrame(results)
 
 
def match_events_locus_only(
    calls_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    mt_len: int,
    min_reciprocal_overlap: float = 0.5,
    max_breakpoint_dist: int = None,
) -> pd.DataFrame:
    """Same matching as match_events(), but ignoring svtype entirely
    (require_type_match=False) -- 'was something found at this locus at
    all', regardless of whether it was classified with the right type.
 
    This is a secondary, additive view, not a replacement for the strict
    typed match_df: it exists for cases like DIMER truth where a caller's
    plain 'del' call at the same locus is genuine partial evidence (it
    found the right place, just not the right classification) but you
    still want the primary TP/FP/FN accounting to require exact type match.
    Use locus-level recall/precision from this alongside
    build_type_confusion_matrix() on the same result to see both "was it
    found" and "was it labeled correctly" without conflating the two.
    """
    return match_events(
        calls_df, truth_df, mt_len,
        min_reciprocal_overlap=min_reciprocal_overlap,
        max_breakpoint_dist=max_breakpoint_dist,
        require_type_match=False,
    )
 
 
def build_type_confusion_matrix(locus_match_df: pd.DataFrame) -> pd.DataFrame:
    """From a locus-only (type-ignoring) match_df, build a truth_svtype x
    called_svtype confusion matrix. TP rows show truth type vs. whatever
    type actually won that locus (may differ from truth -- that's the
    point). FN rows (nothing found at that locus at all, any type) appear
    as an extra 'MISSED' column; FP rows (a call with no truth event at
    that locus) appear as an extra 'NO_TRUTH' row.
    """
    rows = []
    tp = locus_match_df[locus_match_df["match_status"] == "TP"]
    for _, r in tp.iterrows():
        rows.append((r["truth_svtype"], r["called_svtype"]))
    fn = locus_match_df[locus_match_df["match_status"] == "FN"]
    for _, r in fn.iterrows():
        rows.append((r["truth_svtype"], "MISSED"))
    fp = locus_match_df[locus_match_df["match_status"] == "FP"]
    for _, r in fp.iterrows():
        rows.append(("NO_TRUTH", r["called_svtype"]))
 
    df = pd.DataFrame(rows, columns=["truth_type", "called_type"])
    if df.empty:
        return pd.DataFrame()
    return pd.crosstab(df["truth_type"], df["called_type"])
 
 
def flag_colocated_fps(
    match_df: pd.DataFrame,
    mt_len: int,
    overlap_threshold: float = 0.5,
    overlap_mode: str = "reciprocal",
) -> pd.DataFrame:
    """Mark FP rows that are a DEL call sitting on the TRUE coordinates of
    a DIMER truth event in the same sample -- whether or not that DIMER
    event was itself matched by an actual DIMER-typed call.

    Deliberately narrow, not a general "any wrong-type call near any TP"
    rule: a DEL and a DIMER share the same essential breakpoint (a dimer's
    junction genuinely is a deletion junction -- see types_compatible()'s
    docstring for the same reasoning), so a caller's plain 'del' call at a
    dimer's locus is redundant evidence about a locus it found correctly at
    the breakpoint level, not a distinct false event. No other type pair
    gets this treatment -- e.g. a DUP call near a DEL TP is NOT flagged,
    since there's no equivalent shared-breakpoint justification for that
    combination. Only FP rows with called_svtype == 'DEL' overlapping a
    DIMER truth event (TP or FN) are ever flagged.

    Credit is granted against BOTH a matched DIMER TP and an unmatched DIMER
    FN, deliberately -- some pipelines never attempt dimer classification at
    all and label everything at a deletion-like junction 'del' (no DIMER
    call is ever produced, so the DIMER truth event is always FN under
    strict type matching). Without crediting the FN case too, such a
    pipeline would be double-penalized for every dimer in the truth set: the
    DIMER truth event scores FN (correctly -- it genuinely never called it a
    dimer, and that's a real classification gap worth recording), and the
    DEL call that found the exact same breakpoint would ALSO score as its
    own separate FP, as if it were unrelated, unsupported evidence at a
    completely different locus. The FN/recall accounting is untouched here
    either way; only the redundant FP is suppressed.

    Flagging lets you report an "adjusted" precision that doesn't
    double-penalize for this specific case, without changing the strict
    TP/FP/FN counts (a DIMER truth event still requires an actual DIMER
    call to count as TP; only the FP accounting for a coincidentally-
    overlapping DEL call at that same locus is affected).
    """
    match_df = match_df.copy()
    match_df["co_located_with_tp"] = False
    overlap_fn = circular_containment_overlap if overlap_mode == "containment" else circular_reciprocal_overlap

    for sample, sub in match_df.groupby("sample"):
        # any DIMER truth event is eligible to grant credit -- whether it was
        # matched (TP) or missed entirely (FN, e.g. a pipeline that never
        # emits a DIMER-typed call) -- and only to DEL FPs. truth_start/
        # truth_end are populated on both TP and FN rows (a TP's are the
        # matched truth event's own coordinates, an FN's are the truth
        # event it never got a call for), so the same overlap check works
        # for either.
        dimer_rows = sub[(sub["match_status"].isin(["TP", "FN"])) & (sub["truth_svtype"] == "DIMER")]
        if dimer_rows.empty:
            continue
        del_fp_idx = sub[(sub["match_status"] == "FP") & (sub["called_svtype"] == "DEL")].index
        for idx in del_fp_idx:
            fp = match_df.loc[idx]
            for _, dimer_row in dimer_rows.iterrows():
                ov = overlap_fn(
                    fp["called_start"], fp["called_end"], False,
                    dimer_row["truth_start"], dimer_row["truth_end"], False,
                    mt_len,
                )
                if ov >= overlap_threshold:
                    match_df.loc[idx, "co_located_with_tp"] = True
                    break

    return match_df
 
 
def summarize_matches_adjusted(match_df: pd.DataFrame) -> dict:
    """Like summarize_matches(), but excludes FP rows flagged
    co_located_with_tp (by flag_colocated_fps()) from the FP count and from
    precision. TP and FN are unaffected -- this only changes how "wrong
    type at an already-correctly-found locus" gets scored, not whether a
    truth event counts as found.
    """
    tp = (match_df["match_status"] == "TP").sum()
    fp_df = match_df[match_df["match_status"] == "FP"]
    if "co_located_with_tp" in match_df.columns:
        fp = (~fp_df["co_located_with_tp"]).sum()
        fp_excluded = fp_df["co_located_with_tp"].sum()
    else:
        fp = len(fp_df)
        fp_excluded = 0
    fn = (match_df["match_status"] == "FN").sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    # "Accuracy" here is TP / (TP + FP + FN), the standard variant-calling stand-in for
    # classification accuracy (see summarize_matches() below for why plain accuracy
    # doesn't apply to event calling).
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "FP_excluded_colocated": fp_excluded,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def summarize_matches(match_df: pd.DataFrame) -> dict:
    tp = (match_df["match_status"] == "TP").sum()
    fp = (match_df["match_status"] == "FP").sum()
    fn = (match_df["match_status"] == "FN").sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    # There's no "true negative" for event calling -- there's no fixed inventory of
    # non-events a caller could correctly *not* call, so classification accuracy
    # (TP+TN)/(TP+TN+FP+FN) doesn't apply here. TP/(TP+FP+FN) is the metric variant
    # calling benchmarks (hap.py, truvari, GA4GH) use in its place: it's the Jaccard
    # index / intersection-over-union between the call set and the truth set, and,
    # like F1, it penalizes FP and FN equally, but unlike F1 it isn't rescaled by 2x
    # in the numerator, so a caller with equal FP+FN always scores lower on this than
    # on F1 for the same TP/FP/FN. It's included as the closest honest analogue to
    # "accuracy" for this kind of data, not literal classification accuracy.
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy}
 
 
def summarize_matches_weighted(match_df: pd.DataFrame, support_col: str = "called_support") -> dict:
    """Like summarize_matches(), but precision is computed from summed read
    support rather than counting each TP/FP row as one equal unit.

    This is a different question from event-count precision: not "how many
    distinct calls were right vs wrong" but "of all the read-level evidence
    this pipeline reported, what fraction of it pointed at a real event".
    A single stray read that clustered into its own tiny FP row barely
    moves this number; a 5774-read TP dominates it.

    This matters specifically because collapsing overlapping same-svtype
    calls (collapse_overlapping_calls, overlap_mode='containment') is NOT
    safe to use for this -- containment will happily merge several
    genuinely distinct, closely-spaced real events (e.g. a multi-deletion
    sample where truth has many overlapping-but-different deletions) into
    one, silently discarding real TPs/FNs. Event-count matching therefore
    has to stay strict (no containment collapsing) so distinct events stay
    distinct -- which means low-support noise fragments *will* land as
    their own FP rows. Rather than trying to guess which of those are
    "real" via collapsing, this function just makes sure they're weighted
    the way they should be: a 1-read fragment next to a 5774-read TP
    barely registers, without ever having to decide whether to discard it.

    recall is intentionally NOT read-weighted -- there's no read-support
    analogue for a truth event nothing was called at (an FN has no
    `called_support`), so recall stays the standard event-count TP/(TP+FN).

    Returns both the event-count and read-weighted numbers side by side so
    neither view is hidden by the other.
    """
    tp_df = match_df[match_df["match_status"] == "TP"]
    fp_df = match_df[match_df["match_status"] == "FP"]
    fn_df = match_df[match_df["match_status"] == "FN"]

    tp_events, fp_events, fn_events = len(tp_df), len(fp_df), len(fn_df)
    event_precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else float("nan")
    recall = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else float("nan")

    tp_reads = tp_df[support_col].fillna(0).sum() if support_col in tp_df.columns else float("nan")
    fp_reads = fp_df[support_col].fillna(0).sum() if support_col in fp_df.columns else float("nan")
    weighted_precision = (
        tp_reads / (tp_reads + fp_reads) if (tp_reads + fp_reads) > 0 else float("nan")
    )
    weighted_f1 = (
        2 * weighted_precision * recall / (weighted_precision + recall)
        if (weighted_precision + recall) > 0 and not (np.isnan(weighted_precision) or np.isnan(recall))
        else float("nan")
    )

    return {
        "TP_events": tp_events, "FP_events": fp_events, "FN_events": fn_events,
        "event_precision": event_precision, "recall": recall,
        "TP_reads": tp_reads, "FP_reads": fp_reads,
        "weighted_precision": weighted_precision, "weighted_f1": weighted_f1,
    }


def summarize_matches_weighted_per_sample(match_df: pd.DataFrame, support_col: str = "called_support") -> pd.DataFrame:
    """summarize_matches_weighted(), but broken out one row per sample.

    This is the per-sample analogue of summarize_matches_per_sample() --
    same idea, but reporting the read-weighted precision/F1 alongside the
    raw event counts for each sample individually. This is what actually
    answers "does this sample's FP list look bad" without having to
    hand-average anything: a sample can show e.g. 2 TP / 14 FP by event
    count (alarming) and still show weighted_precision=0.998 (because
    those 14 FP rows are almost all 1-2 read noise fragments next to two
    genuinely large TP calls) -- both numbers are real, they're just
    answering different questions ("how many distinct calls were right"
    vs. "how much of the read-level evidence pointed at a real event").
    """
    rows = []
    for sample, sub in match_df.groupby("sample"):
        s = summarize_matches_weighted(sub, support_col=support_col)
        s["sample"] = sample
        rows.append(s)
    cols = ["sample", "TP_events", "FP_events", "FN_events", "event_precision", "recall",
            "TP_reads", "FP_reads", "weighted_precision", "weighted_f1"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def summarize_matches_per_sample(match_df: pd.DataFrame) -> pd.DataFrame:
    """Precision/recall/F1 computed independently for each sample. A sample
    with 12 truth events and a sample with 1 truth event each get exactly
    one row here, regardless of how many TP/FP/FN they individually
    contributed -- this is what macro_average_summary() then averages over,
    so no single busy sample can dominate just by having more events in it.
    """
    rows = []
    for sample, sub in match_df.groupby("sample"):
        s = summarize_matches(sub)
        s["sample"] = sample
        s["n_truth_events"] = (sub["match_status"] != "FP").sum()  # TP + FN = truth events in this sample
        rows.append(s)
    return pd.DataFrame(rows)
 
 
def macro_average_summary(per_sample_df: pd.DataFrame) -> dict:
    """Unweighted mean of precision/recall/F1 across samples (macro average),
    as opposed to summarize_matches() on the pooled match_df (micro average,
    which implicitly weights each sample by its own event count).
 
    IMPORTANT: precision, recall, and F1 can each be undefined (NaN) for
    different samples, for different reasons -- most commonly, a sample
    where the pipeline made zero calls at all has undefined precision
    (0/0) and undefined F1, but recall is still defined (0 / n_truth = 0.0,
    since there's essentially always at least one truth event per sample).
    Averaging each column independently via nanmean therefore averages
    precision and F1 over a SMALLER set of samples (only ones with at
    least one call) than recall (essentially all samples) -- which can make
    precision/F1 look deceptively good relative to recall, purely because
    the pipeline's complete misses got silently excluded from precision/F1
    but correctly dragged recall down. This function reports the valid
    sample count for each metric explicitly so that discrepancy is visible
    rather than hidden, and also reports a 'consistent' version that only
    averages over samples where all three are defined, so all three numbers
    describe the exact same set of samples if you want a directly
    comparable trio (at the cost of dropping some samples entirely).
    """
    n_total = len(per_sample_df)
    n_valid_p = int(per_sample_df["precision"].notna().sum())
    n_valid_r = int(per_sample_df["recall"].notna().sum())
    n_valid_f1 = int(per_sample_df["f1"].notna().sum())
    n_valid_acc = int(per_sample_df["accuracy"].notna().sum()) if "accuracy" in per_sample_df.columns else 0

    subset_cols = ["precision", "recall", "f1"] + (["accuracy"] if "accuracy" in per_sample_df.columns else [])
    all_defined = per_sample_df.dropna(subset=subset_cols)
    n_consistent = len(all_defined)

    # A third option: precision/F1/accuracy are only undefined because the pipeline
    # made zero calls in that sample (TP+FP==0, so TP+FP+FN==FN>0 too) -- that's a
    # real failure, not a case where the metric is inapplicable. Treating it as 0
    # (rather than excluding the sample, as the "consistent" version above does)
    # keeps every sample in the average without silently hiding complete misses.
    filled = per_sample_df.copy()
    filled["precision"] = filled["precision"].fillna(0.0)
    filled["f1"] = filled["f1"].fillna(0.0)
    if "accuracy" in filled.columns:
        filled["accuracy"] = filled["accuracy"].fillna(0.0)

    result = {
        "n_samples": n_total,
        "n_samples_with_defined_precision": n_valid_p,
        "n_samples_with_defined_recall": n_valid_r,
        "n_samples_with_defined_f1": n_valid_f1,
        "n_samples_with_defined_accuracy": n_valid_acc,
        "macro_precision": np.nanmean(per_sample_df["precision"]) if n_total else float("nan"),
        "macro_recall": np.nanmean(per_sample_df["recall"]) if n_total else float("nan"),
        "macro_f1": np.nanmean(per_sample_df["f1"]) if n_total else float("nan"),
        "macro_accuracy": np.nanmean(per_sample_df["accuracy"]) if n_total and "accuracy" in per_sample_df.columns else float("nan"),
        "n_samples_consistent_denominator": n_consistent,
        "macro_precision_consistent": all_defined["precision"].mean() if n_consistent else float("nan"),
        "macro_recall_consistent": all_defined["recall"].mean() if n_consistent else float("nan"),
        "macro_f1_consistent": all_defined["f1"].mean() if n_consistent else float("nan"),
        "macro_accuracy_consistent": all_defined["accuracy"].mean() if (n_consistent and "accuracy" in all_defined.columns) else float("nan"),
        "macro_precision_zerofilled": filled["precision"].mean() if n_total else float("nan"),
        "macro_f1_zerofilled": filled["f1"].mean() if n_total else float("nan"),
        "macro_accuracy_zerofilled": filled["accuracy"].mean() if (n_total and "accuracy" in filled.columns) else float("nan"),
    }
    return result
 
 
def support_threshold_sweep(
    calls_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    mt_len: int,
    thresholds: list = None,
    min_reciprocal_overlap: float = 0.5,
    require_type_match: bool = True,
    collapse_overlap_threshold: float = None,
    collapse_overlap_mode: str = "reciprocal",
    max_support_ratio: float = None,
) -> pd.DataFrame:
    """Precision/recall/F1 as a function of a minimum read-support cutoff,
    answering the question 'does read count actually separate real calls
    from noise, and if so, how much would filtering on it help?'
 
    A single aggregate F1 treats an 8000-read call and a 1-read call as
    equally right or equally wrong -- this sweep instead shows what
    precision/recall/F1 would be if you discarded every call below each
    threshold, so you can see directly whether false positives cluster at
    low support (and could be filtered out cheaply) or whether they're
    genuinely well-supported too (in which case support alone can't fix
    precision, and it's a real detection problem, not a filtering problem).
 
    Collapsing (if collapse_overlap_threshold is given) is re-run at each
    threshold on the already-filtered calls, matching how the main pipeline
    would behave if that threshold were used as a hard pre-filter.
    """
    if thresholds is None:
        max_support = calls_df["support"].max() if "support" in calls_df.columns and not calls_df.empty else 1
        # log-ish spaced thresholds covering 1 up to the max observed support
        thresholds = sorted(set([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
                                 + [int(max_support)]))
        thresholds = [t for t in thresholds if t <= max_support] or [1]
 
    rows = []
    for thresh in thresholds:
        filtered = calls_df[calls_df["support"] >= thresh].copy() if "support" in calls_df.columns else calls_df.copy()
        if collapse_overlap_threshold is not None and not filtered.empty:
            filtered = collapse_overlapping_calls(
                filtered, mt_len, overlap_threshold=collapse_overlap_threshold,
                overlap_mode=collapse_overlap_mode, max_support_ratio=max_support_ratio,
            )
        match_df = match_events(
            filtered, truth_df, mt_len,
            min_reciprocal_overlap=min_reciprocal_overlap,
            require_type_match=require_type_match,
        )
        summary = summarize_matches(match_df)
        summary["min_support"] = thresh
        rows.append(summary)
 
    return pd.DataFrame(rows)
 
 
def summarize_by_type(match_df: pd.DataFrame) -> pd.DataFrame:
    """Per-svtype precision/recall/F1. Uses truth_svtype for TP/FN rows and
    called_svtype for FP rows (a FP has no truth type)."""
    rows = []
    types = set(match_df["truth_svtype"].dropna()) | set(match_df["called_svtype"].dropna())
    for svtype in sorted(types):
        tp = ((match_df["match_status"] == "TP") & (match_df["truth_svtype"] == svtype)).sum()
        fn = ((match_df["match_status"] == "FN") & (match_df["truth_svtype"] == svtype)).sum()
        fp = ((match_df["match_status"] == "FP") & (match_df["called_svtype"] == svtype)).sum()
        # also count TP rows where a *different* truth type got matched to this called type as a type-confusion FP
        # (only relevant if require_type_match=False was used upstream; with type-matching TP always has truth==called)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
        rows.append({"svtype": svtype, "TP": tp, "FP": fp, "FN": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)
 
 
# --------------------------------------------------------------------------
# Pipeline 1 loader: per-cluster TSV with heteroplasmy + final.event/start/end
# --------------------------------------------------------------------------
def load_pipeline1_tsv(path: str) -> pd.DataFrame:
    # The C parser's read(nbytes) can fail outright on some network filesystems
    # (seen on HPC scratch/NFS mounts) with "Calling read(nbytes) on source
    # failed" even though the file itself is perfectly well-formed -- it's a
    # C-engine/filesystem interaction issue, not bad data. Fall back to the
    # slower pure-Python engine, which reads the file differently and doesn't
    # hit this. Read the whole file into memory first as a second line of
    # defense, in case the C engine's failure was specifically about streaming
    # reads from that filesystem rather than parsing per se.
    try:
        df = pd.read_csv(path, sep="\t")
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(path, sep="\t", engine="python")
        except pd.errors.ParserError:
            with open(path, "r") as f:
                text = f.read()
            df = pd.read_csv(io.StringIO(text), sep="\t", engine="python")
    # sample column may have a path-like prefix, e.g. "indel/DEL_and_DUP_33_67_het005"
    df["sample"] = df["sample"].apply(lambda s: os.path.basename(str(s)))
    out = pd.DataFrame({
        "sample": df["sample"],
        "cluster_id": df["cluster.id"],
        "svtype": df["final.event"].str.upper(),
        "start": df["final.start"],
        "end": df["final.end"],
        # This pipeline reports heteroplasmy as a percent (e.g. 1.1828 means 1.18%,
        # not 118%), so divide by 100 to match the truth table's 0-1 fraction scale.
        "heteroplasmy": df["heteroplasmy"] / 100.0,
        "support": df["alt.reads"] if "alt.reads" in df.columns else np.nan,
    })
    # If a caller reports start > end, that's the natural way an origin-spanning
    # breakpoint shows up (e.g. start=13442, end=8462) -- infer wraparound from it.
    out["wraps_origin"] = out["start"] > out["end"]

    # DUP is the exception: this pipeline's final.start/final.end for a duplication
    # encode the RETAINED (non-duplicated) arc going the long way around the circle,
    # not the duplicated segment itself -- the opposite of del.start.range/del.end.range,
    # which give the actual duplication breakpoints directly. Confirmed quantitatively:
    # a real example has del.size=10000 (from del.start.range/del.end.range) but
    # final.size=6569, and 10000 + 6569 == 16569 == mt_len exactly -- final.start/
    # final.end is the complement arc. Truth records DUP events as the short,
    # non-wrapping span between the two breakpoints (see truth_burden_curve's docstring
    # and prior verification against DUP_large_het005), so to match that convention the
    # two breakpoints just need reordering: min() and max() of final.start/final.end
    # give the actual duplicated span's start/end directly, non-wrapping, regardless of
    # which one this pipeline happened to report as "start". Left uncorrected, every DUP
    # call from this pipeline is loaded at the wrong location entirely (a large complement
    # region instead of the real ~10000bp duplicated span), which registers as both an FN
    # (truth's real region has no matching call) and an FP (the wrongly-placed region
    # matches nothing) for every single DUP truth event.
    is_dup = out["svtype"] == "DUP"
    if is_dup.any():
        s, e = out.loc[is_dup, "start"], out.loc[is_dup, "end"]
        out.loc[is_dup, "start"] = np.minimum(s, e)
        out.loc[is_dup, "end"] = np.maximum(s, e)
        out.loc[is_dup, "wraps_origin"] = False
    return out
 
 
# --------------------------------------------------------------------------
# Pipeline 2 loader: cluster TSV (no sample/heteroplasmy column) + burden TSV
# --------------------------------------------------------------------------
def load_pipeline2_cluster_tsv(path: str, sample: str) -> pd.DataFrame:
    """`sample` must be supplied since this file format doesn't self-report it
    (typically one cluster file per sample — pass the sample name derived
    from the filename or directory).
 
    Uses the `orig_type` column for svtype if present, falling back to
    `type` otherwise. Some pipeline outputs use `type` for a cluster-level
    classification that isn't the SV type itself (e.g. 'dom' for a dominant/
    majority cluster vs 'del'/'dup'), while `orig_type` carries the actual
    del/dup call -- using the wrong column silently mismatches every such
    row against truth.
    """
    empty_cols = ["sample", "cluster_id", "svtype", "start", "end", "read_count", "wraps_origin"]
    try:
        df = pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=empty_cols)
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    type_col = "orig_type" if "orig_type" in df.columns else "type"
    out = pd.DataFrame({
        "sample": sample,
        "cluster_id": df["cluster"],
        "svtype": df[type_col].str.upper(),
        "start": df["start"],
        "end": df["end"],
        "read_count": df["read_count"],
        "support": df["read_count"],
    })
    # If there's a separate 'type' column distinct from orig_type, it may carry a
    # pipeline-internal classification (e.g. 'dom' for the chosen/dominant cluster
    # at a locus vs 'del' for other candidate sub-clusters it considered but didn't
    # select) rather than the SV type itself. Preserve it as cluster_class so the
    # caller can decide whether to filter on it.
    if "orig_type" in df.columns and "type" in df.columns:
        out["cluster_class"] = df["type"]
    # Same wraparound inference as pipeline1: start > end implies origin-spanning.
    out["wraps_origin"] = out["start"] > out["end"]
    return out
 
 
def load_burden_tsv(path: str) -> pd.DataFrame:
    """Load a per-position deletion burden file:
        chrom  position  deletion_burden  coverage  deletion_percent
    Handles either real tabs or the run-together whitespace shown in some
    pasted examples by falling back to a flexible whitespace split.
 
    Adds a `deletion_fraction` column (deletion_percent / 100), since the
    truth table's effective_heteroplasmy is a 0-1 fraction and this pipeline
    reports deletion_percent as an actual percent (e.g. 45.20 means 45.2%,
    not 4520%). Downstream code should use `deletion_fraction`, not
    `deletion_percent`, when comparing against truth.
    """
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] < 5:
            raise ValueError("too few columns, retry with whitespace split")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = [c.strip() for c in df.columns]
    df["deletion_fraction"] = df["deletion_percent"].astype(float) / 100.0
    return df
 
 
def attach_deletion_burden_estimate(calls_df: pd.DataFrame, burden_df: pd.DataFrame) -> pd.DataFrame:
    """For DEL and DIMER calls with no direct heteroplasmy column (pipeline 2),
    estimate heteroplasmy as the mean deletion_fraction across the call's
    start-end span from the burden file. DIMER is included alongside DEL
    because a dimer's breakpoint region is genuinely deleted in the mutant
    molecule (same reasoning as truth_burden_curve() treating DIMER as
    deletion signal) -- a burden file has no separate way to represent
    "dimer" and shouldn't be expected to. DUP calls are left as NaN (burden
    file only tracks deletion signal, not duplication) — compare those via
    the cluster read_count instead, or via a separate duplication-coverage-
    fold metric if your pipeline reports one.
    """
    calls_df = calls_df.copy()
    ests = []
    for _, row in calls_df.iterrows():
        if row["svtype"] not in ("DEL", "DIMER"):
            ests.append(np.nan)
            continue
        s, e = min(row["start"], row["end"]), max(row["start"], row["end"])
        span = burden_df[(burden_df["position"] >= s) & (burden_df["position"] <= e)]
        ests.append(span["deletion_fraction"].mean() if not span.empty else np.nan)
    calls_df["heteroplasmy"] = ests
    return calls_df
 
 
# --------------------------------------------------------------------------
# Genome-wide deletion burden curve: truth vs. called
# --------------------------------------------------------------------------
def truth_burden_curve(sample: str, truth_df: pd.DataFrame, mt_len: int) -> np.ndarray:
    """Per-position expected deletion fraction for one sample: sum of
    effective_heteroplasmy over all deletion-contributing truth events
    (svtype DEL or DIMER -- a dimer's breakpoint span is deleted in the
    mutant molecule just like a plain DEL, so it contributes to the burden
    signal the same way) covering that position (handling origin-spanning
    events by wrapping).
    """
    curve = np.zeros(mt_len)
    sub = truth_df[(truth_df["sample"] == sample) & (truth_df["svtype"].isin(["DEL", "DIMER"]))]
    for _, row in sub.iterrows():
        ivs = _split_wrapped_interval(row["start"], row["end"], row["wraps_origin"], mt_len)
        for s, e in ivs:
            curve[int(s):int(e)] += row["effective_heteroplasmy"]
    return curve
 
 
def max_deletion_burden_in_region(burden_df: pd.DataFrame, start, end, wraps_origin: bool, mt_len: int) -> float:
    """Max called deletion_fraction within one truth event's span (handles
    origin-spanning regions by checking both wrapped sub-intervals)."""
    ivs = _split_wrapped_interval(start, end, wraps_origin, mt_len)
    mask = pd.Series(False, index=burden_df.index)
    for s, e in ivs:
        # truth start/end are 0-indexed half-open [s, e); burden `position` is 1-indexed
        mask |= (burden_df["position"] >= s + 1) & (burden_df["position"] <= e)
    if not mask.any():
        return float("nan")
    return burden_df.loc[mask, "deletion_fraction"].max()
 
 
def sample_level_burden_totals(sample: str, truth_df: pd.DataFrame, burden_df: pd.DataFrame, mt_len: int) -> tuple:
    """For one sample, find every distinct DEL truth region and take the max
    called burden within each one separately, then sum across regions.
 
    This handles samples with multiple non-overlapping deletions (e.g. one
    in the major arc, one in the minor arc) correctly: a single genome-wide
    max would only capture whichever deletion happens to be larger, silently
    dropping the other. Taking the max *per truth region* and summing
    recovers the total mutant load across all of them -- the same thing as
    manually adding the major-arc max + minor-arc max, just generalized to
    however many non-overlapping DEL regions a sample actually has (1, 2, or
    more), driven by the truth table rather than hardcoded arc boundaries.
 
    Returns (region_detail_df, true_total, called_total). true_total is the
    sum of effective_heteroplasmy across the sample's deletion-contributing
    truth events (DEL and DIMER); called_total is the sum of per-region
    called maxima (NaN regions are excluded from the sum and noted in the
    detail table, not silently zeroed).

    This assumes the sample's DEL/DIMER truth regions DON'T overlap each
    other (per-region maxima summed independently double-counts any shared
    region otherwise -- see truth_dels_overlap_in_sample() for why, and the
    ~9x inflation seen on a real overlapping multi-deletion truth set even
    with a perfect caller). If they do overlap, called_total (and true_total,
    for consistency, since the two aren't comparable if only one is
    computed) come back as NaN rather than a silently wrong number -- use
    compare_burden_curves() for these samples instead, it isn't affected.
    """
    sub = truth_df[(truth_df["sample"] == sample) & (truth_df["svtype"].isin(["DEL", "DIMER"]))]
    rows = []
    for _, r in sub.iterrows():
        called_max = max_deletion_burden_in_region(burden_df, r["start"], r["end"], r["wraps_origin"], mt_len)
        rows.append({
            "sample": sample,
            "event_label": r.get("event_label", None),
            "start": r["start"], "end": r["end"],
            "true_heteroplasmy": r["effective_heteroplasmy"],
            "called_max_heteroplasmy": called_max,
        })
    region_df = pd.DataFrame(rows)
    if region_df.empty:
        return region_df, 0.0, float("nan")

    if truth_dels_overlap_in_sample(sub, mt_len):
        print(f"  [warning] sample '{sample}': DEL/DIMER truth regions overlap -- "
              f"sample-level total heteroplasmy is not valid here (summing per-region maxima "
              f"double-counts the shared region), reporting true_total/called_total as NaN. "
              f"See compare_burden_curves() / the burden curve accuracy table instead.")
        region_df["overlapping_truth_regions"] = True
        return region_df, float("nan"), float("nan")

    region_df["overlapping_truth_regions"] = False
    true_total = region_df["true_heteroplasmy"].sum()
    called_total = region_df["called_max_heteroplasmy"].sum(skipna=True)
    if region_df["called_max_heteroplasmy"].isna().any():
        n_missing = region_df["called_max_heteroplasmy"].isna().sum()
        print(f"  [warning] sample '{sample}': {n_missing} DEL region(s) had no burden data in range "
              f"-- excluded from called_total (may understate it)")
    return region_df, true_total, called_total
 
 
def compare_burden_curves(truth_curve: np.ndarray, burden_df: pd.DataFrame) -> dict:
    """Align truth_curve (0-indexed array, length mt_len) to the burden file's
    `position` column (assumed 1-indexed) and compute MAE/RMSE/Pearson r.
    Returns the metrics dict plus a merged DataFrame for plotting.
    """
    from scipy.stats import pearsonr
 
    pos = burden_df["position"].astype(int).values
    called = burden_df["deletion_fraction"].astype(float).values
    truth_vals = truth_curve[pos - 1]  # position is 1-indexed
 
    merged = pd.DataFrame({"position": pos, "truth": truth_vals, "called": called})
    mae = np.mean(np.abs(truth_vals - called))
    rmse = np.sqrt(np.mean((truth_vals - called) ** 2))
    if len(truth_vals) > 1 and np.std(truth_vals) > 0 and np.std(called) > 0:
        r, _ = pearsonr(truth_vals, called)
    else:
        r = float("nan")
    return {"MAE": mae, "RMSE": rmse, "pearson_r": r, "n_positions": len(pos)}, merged
