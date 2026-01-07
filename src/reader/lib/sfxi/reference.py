"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/sfxi/reference.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def _reduce(series: pd.Series, stat: Literal["mean", "median"]) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.median()) if stat == "median" else float(s.mean())


def resolve_reference_design_id(
    df: pd.DataFrame,
    *,
    design_by: list[str],
    ref_label: str | None,
) -> str | None:
    """
    Deterministically resolve the configured reference label to the *raw* design label.
    Policy:
      1) If ref_label matches the raw label column exactly → use it.
      2) Else, if <label>_alias exists and matches exactly → map to the single raw label.
      3) Else → raise a clear error (no silent fallback).
    """
    if ref_label is None:
        return None
    label_col = design_by[0] if design_by else "design_id"
    alias_col = f"{label_col}_alias"
    want = str(ref_label)

    if label_col in df.columns and df[label_col].astype(str).eq(want).any():
        return want  # exact raw match

    if alias_col in df.columns:
        pairs = df[[label_col, alias_col]].dropna().astype(str).drop_duplicates()
        matches = pairs.loc[pairs[alias_col] == want, label_col].unique()
        if len(matches) == 1:
            return str(matches[0])  # unique alias→raw mapping
        if len(matches) > 1:
            raise ValueError(
                f"sfxi: reference.{label_col} {want!r} resolves via {alias_col!r} to multiple {label_col!r} values: "
                f"{sorted(map(str, matches))!r}"
            )

    # diagnostics (short previews)
    raw_vals = df[label_col].astype(str).unique().tolist() if label_col in df.columns else []
    alias_vals = df[alias_col].astype(str).unique().tolist() if alias_col in df.columns else []
    raw_prev = ", ".join(sorted(raw_vals)[:8]) + (" …" if len(raw_vals) > 8 else "")
    alias_prev = ", ".join(sorted(alias_vals)[:8]) + (" …" if len(alias_vals) > 8 else "")
    raise ValueError(
        f"sfxi: reference.{label_col} {want!r} not found under {label_col!r} or {alias_col!r}.\n"
        f"   available raw:   [{raw_prev}]\n"
        f"   available alias: [{alias_prev or '—'}]"
    )


def compute_reference_table(
    per_corner: pd.DataFrame,
    *,
    design_by: list[str],
    ref_design_id: str,
    stat: Literal["mean", "median"] = "mean",
) -> pd.DataFrame:
    """
    Build a table of reference b00/b10/b01/b11 values.

    Returns a DataFrame:
      one row with columns ['b00','b10','b01','b11']
    """
    if not design_by:
        raise ValueError("design_by must contain at least one column to locate the reference")

    label_col = design_by[0]
    ref_rows = per_corner[per_corner[label_col].astype(str) == str(ref_design_id)].copy()
    if ref_rows.empty:
        raise ValueError(f"Reference design_id {ref_design_id!r} has no rows in per-corner table.")

    agg = ref_rows.groupby("corner")["y_mean"].agg(lambda s: _reduce(s, stat))
    mean = pd.DataFrame([agg.to_dict()])
    # Rename columns to b00..b11
    mean = mean.rename(columns={"00": "b00", "10": "b10", "01": "b01", "11": "b11"})
    return mean
