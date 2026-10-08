#!/usr/bin/env python3
"""
resource_and_het_metrics.py
 
Two independent, format-agnostic utilities used by run_comparison2.py:
 
  1. parse_time_v_log()   -- parse a `/usr/bin/time -v <cmd>` log into a dict
                              of wall-clock time, CPU time, and peak memory.
  2. regression_metrics()  -- MAE, RMSE, Pearson r, and Lin's CCC between
                              true and called heteroplasmy values.
 
Nothing here reads or writes VCFs -- the truth table comes from
truth_builder.py (built directly from your variant-generation script) and
pipeline calls are read from TSVs by matcher.py.
"""
 
import re
 
import numpy as np
import pandas as pd
 
 
def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient between two arrays."""
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    sxy = np.mean((x - mx) * (y - my))
    return (2 * sxy) / (vx + vy + (mx - my) ** 2)
 
 
def regression_metrics(df: pd.DataFrame) -> dict:
    """MAE, RMSE, Pearson r, Lin's CCC between a DataFrame's true_hf and
    called_hf columns (heteroplasmy fraction, 0-1)."""
    from scipy.stats import pearsonr
 
    if df.empty:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "pearson_r": np.nan, "CCC": np.nan}
 
    x = df["true_hf"].astype(float).values
    y = df["called_hf"].astype(float).values
    mae = np.mean(np.abs(x - y))
    rmse = np.sqrt(np.mean((x - y) ** 2))
    r, _ = pearsonr(x, y)
    ccc = lins_ccc(x, y)
    return {"n": len(df), "MAE": mae, "RMSE": rmse, "pearson_r": r, "CCC": ccc}
 
 
def parse_time_v_log(path: str) -> dict:
    """Parse a `/usr/bin/time -v <cmd>` stderr log into a dict of resource stats.
    Wrap each pipeline like:  /usr/bin/time -v ./run_pipeline.sh 2> timeA.log
    """
    with open(path) as f:
        text = f.read()
 
    def grab(pattern, cast=float, default=np.nan):
        m = re.search(pattern, text)
        if not m:
            return default
        return cast(m.group(1))
 
    # Elapsed time can be [h:]mm:ss.ss
    elapsed_str = grab(r"Elapsed \(wall clock\) time.*: ([\d:.]+)", cast=str, default=None)
    elapsed_sec = np.nan
    if elapsed_str:
        parts = [float(p) for p in elapsed_str.split(":")]
        if len(parts) == 3:
            elapsed_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            elapsed_sec = parts[0] * 60 + parts[1]
        else:
            elapsed_sec = parts[0]
 
    return {
        "wall_clock_sec": elapsed_sec,
        "user_time_sec": grab(r"User time \(seconds\): ([\d.]+)"),
        "sys_time_sec": grab(r"System time \(seconds\): ([\d.]+)"),
        "cpu_percent": grab(r"Percent of CPU this job got: (\d+)", cast=float),
        "max_rss_kb": grab(r"Maximum resident set size.*: (\d+)", cast=float),
    }
 