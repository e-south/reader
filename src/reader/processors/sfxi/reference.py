"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/sfxi/reference.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Literal

import pandas as pd


def _reduce(series: pd.Series, stat: Literal["mean", "median"]) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.median()) if stat == "median" else float(s.mean())


def compute_reference_table(
    per_corner: pd.DataFrame,
    *,
    design_by: List[str],
    batch_col: str,
    ref_genotype: str,
    scope: Literal["batch", "global"] = "batch",
    stat: Literal["mean", "median"] = "mean",
) -> pd.DataFrame:
    """
    Build a table of reference b00/b10/b01/b11 values.

    Returns a DataFrame:
      scope == "batch": columns = [batch_col, 'b00','b10','b01','b11']
      scope == "global": one row with columns ['b00','b10','b01','b11']
    """
    if not design_by:
        raise ValueError("design_by must contain at least one column to locate the reference")

    label_col = design_by[0]
    ref_rows = per_corner[per_corner[label_col].astype(str) == str(ref_genotype)].copy()
    if ref_rows.empty:
        raise ValueError(f"Reference genotype {ref_genotype!r} has no rows in per-corner table.")

    idx = [batch_col] if scope == "batch" else []
    mean = ref_rows.pivot_table(index=idx, columns="corner", values="y_mean", aggfunc=lambda s: _reduce(s, stat))

    # Rename columns to b00..b11 and reset index
    mean = mean.rename(columns={"00": "b00", "10": "b10", "01": "b01", "11": "b11"}).reset_index()
    return mean
