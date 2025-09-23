"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/sfxi/math.py

Compute the vec8 = [v00,v10,v01,v11, y00*,y10*,y01*,y11*] per (design x batch).

Deafult computation:

- v* are computed from the LOGIC CHANNEL (e.g., YFP/CFP):
    u_i = log2(max(r_i, eps_ratio))
    if max(u)-min(u) <= eps_range: v_i = 0.25
    else: v = (u - u_min) / (u_max - u_min)

- y* are computed from the INTENSITY CHANNEL (e.g., YFP/OD600):
    y*_i = log2( (b_i + eps_abs) / max(anchor_i, eps_ref) )

Anchors are computed from the INTENSITY per-corner table for the reference genotype.

Author(s): Eric J. South (updated)
--------------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_log2(x: np.ndarray | float, eps: float) -> np.ndarray | float:
    return np.log2(np.maximum(x, eps))


def _logic_minmax_from_four(
    vals: Tuple[float, float, float, float],
    *,
    eps_ratio: float,
    eps_range: float
) -> Tuple[np.ndarray, float, bool]:
    """
    Given four raw logic values (typically ratios), compute:
      - u = log2(vals guarded by eps_ratio)
      - if span is tiny (flat), set v = 0.25 for all states
      - else min-max to [0,1]
    Returns (v, r, flat_flag) where r is linear dynamic range (max/min).
    """
    a = np.array(vals, dtype=float)
    u = _safe_log2(a, eps_ratio)
    umin, umax = float(np.min(u)), float(np.max(u))
    span = umax - umin
    flat = bool(span <= eps_range)

    if flat:
        v = np.full(4, 0.25, dtype=float)
    else:
        v = np.clip((u - umin) / span, 0.0, 1.0)

    # dynamic range in linear space (helpful diagnostic)
    M = float(np.max(np.maximum(a, eps_ratio)))
    m = float(np.min(np.maximum(a, eps_ratio)))
    r = (M / m) if M > 0 and m > 0 else 1.0
    return v.astype(float), r, flat


def compute_vec8(
    *,
    points_logic: pd.DataFrame,           # b00..b11 from LOGIC channel
    points_intensity: pd.DataFrame,       # b00..b11 from INTENSITY channel
    per_corner_intensity: pd.DataFrame,   # per-corner table for anchors
    design_by: List[str],
    batch_col: str,
    reference_genotype: Optional[str],
    reference_scope: str,
    reference_stat: str,
    eps_ratio: float,
    eps_range: float,
    eps_ref: float,
    eps_abs: float,
) -> pd.DataFrame:
    """
    Build vec8 per (design×batch). Logic and intensity channels are decoupled.
    """
    # Build reference (anchor) table from INTENSITY per-corner rows
    ref_tab: Optional[pd.DataFrame] = None
    if reference_genotype:
        label_col = design_by[0]
        ref_rows = per_corner_intensity[
            per_corner_intensity[label_col].astype(str) == str(reference_genotype)
        ].copy()
        if not ref_rows.empty:
            grp = [batch_col, "corner"] if reference_scope == "batch" else ["corner"]
            agg_fun = "median" if reference_stat == "median" else "mean"
            ref_tab = (
                ref_rows
                .groupby(grp)["y_mean"]
                .agg(agg_fun)
                .reset_index()
                .rename(columns={"y_mean": "anchor_mean"})
            )

    # Align logic/intensity points on design_by + batch
    idx_cols = design_by + [batch_col]
    L = points_logic.set_index(idx_cols)
    I = points_intensity.set_index(idx_cols)  # noqa
    merged = L[["b00","b10","b01","b11"]].join(
        I[["b00","b10","b01","b11"]],
        how="inner",
        lsuffix="_logic",
        rsuffix="_intensity",
    ).reset_index()

    out_rows: List[Dict[str, object]] = []

    for _, row in merged.iterrows():
        # Logic vector from LOGIC channel
        v, r_logic, flat = _logic_minmax_from_four(
            (
                float(row["b00_logic"]),
                float(row["b10_logic"]),
                float(row["b01_logic"]),
                float(row["b11_logic"]),
            ),
            eps_ratio=eps_ratio, eps_range=eps_range,
        )

        # Anchors from INTENSITY per-corner table
        anchors = {"00": np.nan, "10": np.nan, "01": np.nan, "11": np.nan}
        if ref_tab is not None and not ref_tab.empty:
            if reference_scope == "batch":
                batch_val = row[batch_col]
                sub = ref_tab[ref_tab[batch_col] == batch_val]
                if not sub.empty:
                    for _, rr in sub.iterrows():
                        anchors[str(rr["corner"])] = float(rr["anchor_mean"])
            else:
                for _, rr in ref_tab.iterrows():
                    anchors[str(rr["corner"])] = float(rr["anchor_mean"])

        # Intensity y* from INTENSITY channel against anchors
        def ystar(b: float, a: float) -> float:
            denom = max(eps_ref, float(a) if not np.isnan(a) else eps_ref)
            val = (b + eps_abs) / denom
            val = max(val, eps_ratio)
            return float(np.log2(val))

        y00 = ystar(float(row["b00_intensity"]), anchors["00"])
        y10 = ystar(float(row["b10_intensity"]), anchors["10"])
        y01 = ystar(float(row["b01_intensity"]), anchors["01"])
        y11 = ystar(float(row["b11_intensity"]), anchors["11"])

        rec: Dict[str, object] = {c: row[c] for c in idx_cols}
        rec.update(
            v00=float(v[0]), v10=float(v[1]), v01=float(v[2]), v11=float(v[3]),
            y00_star=y00, y10_star=y10, y01_star=y01, y11_star=y11,
            r_logic=r_logic,
            flat_logic=flat,
        )
        out_rows.append(rec)

    return pd.DataFrame.from_records(out_rows)
