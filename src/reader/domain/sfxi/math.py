"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domain/sfxi/math.py

SFXI math: build vec8 from logic/intensity per-corner points.

Compute the vec8 = [v00,v10,v01,v11, y00*,y10*,y01*,y11*] per design.

- v* are computed from the LOGIC CHANNEL (e.g., YFP/CFP):
    # r_logic and v computation are derived from the LOGIC channel corner means.
    # r_logic is the dynamic range on the linear scale (after ε guard),
    # while v is obtained by log2 + min-max on the *log* scale.
    # Spec-aligned: use eps_range as eta in the denominator when not flat.
    if max(u)-min(u) <= eps_range: v_i = 0.25
    else: v = (u - u_min) / (u_max - u_min + eps_range)

- y* are computed from the INTENSITY CHANNEL (e.g., YFP/OD600):
    y_linear_i = (b_i + eps_abs) / max(anchor_i + ref_add_alpha, eps_ref)
    y*_i       = log2( max(y_linear_i + log2_offset_delta, eps_ratio) )

Anchors are computed from the INTENSITY per-corner table for the reference design_id.

Author(s): Eric J. South (updated)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


def _safe_log2(x: np.ndarray | float, eps: float) -> np.ndarray | float:
    return np.log2(np.maximum(x, eps))


def _logic_minmax_from_four(
    vals: tuple[float, float, float, float], *, eps_ratio: float, eps_range: float
) -> tuple[np.ndarray, float, bool, float, float, float, str, str]:
    a = np.array(vals, dtype=float)
    # ε‑guarded linear values (used for r_logic = max/min)
    a_guard = np.maximum(a, eps_ratio)
    # log2 for min–max shape mapping
    u = _safe_log2(a_guard, eps_ratio)
    umin, umax = float(np.min(u)), float(np.max(u))
    span = umax - umin
    flat = bool(span <= eps_range)
    if flat:
        v = np.full(4, 0.25, dtype=float)
    else:
        denom = span + float(eps_range)
        v = np.clip((u - umin) / denom, 0.0, 1.0)
    M = float(np.max(a_guard))
    m = float(np.min(a_guard))
    r = (M / m) if M > 0 and m > 0 else 1.0
    # Corner labels for clarity in logs/diagnostics
    corners = np.array(["00", "10", "01", "11"])
    cmax = str(corners[int(np.argmax(a_guard))])
    cmin = str(corners[int(np.argmin(a_guard))])
    # Return v, r, flat, plus diagnostics:
    #   M (max), m (min) on linear scale; span on log2 scale; and which corners hit them.
    return v.astype(float), r, flat, M, m, span, cmax, cmin


def compute_vec8(
    *,
    points_logic: pd.DataFrame,  # b00..b11 from LOGIC channel
    points_intensity: pd.DataFrame,  # b00..b11 from INTENSITY channel
    per_corner_intensity: pd.DataFrame,  # per-corner table for anchors
    design_by: list[str],
    reference_design_id: str | None,
    reference_stat: str,
    eps_ratio: float,
    eps_range: float,
    eps_ref: float,  # hard lower bound for (A + α)
    eps_abs: float,  # small add to numerator b_i (absolute intensity)
    ref_add_alpha: float,  # α
    log2_offset_delta: float,  # δ
) -> pd.DataFrame:
    label_col = design_by[0]

    # anchors from INTENSITY per-corner table for the reference design_id
    ref_tab: pd.DataFrame | None = None
    if reference_design_id:
        ref_rows = per_corner_intensity[per_corner_intensity[label_col].astype(str) == str(reference_design_id)].copy()
        if not ref_rows.empty:
            agg_fun = "median" if reference_stat == "median" else "mean"
            ref_tab = (
                ref_rows.groupby(["corner"])["y_mean"]
                .agg(agg_fun)
                .reset_index()
                .rename(columns={"y_mean": "anchor_mean"})
            )

    idx_cols = design_by
    L = points_logic.set_index(idx_cols)
    I_vec = points_intensity.set_index(idx_cols)
    merged = (
        L[["b00", "b10", "b01", "b11"]]
        .join(I_vec[["b00", "b10", "b01", "b11"]], how="inner", lsuffix="_logic", rsuffix="_intensity")
        .reset_index()
    )

    out_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        v, r_logic, flat, rmax, rmin, span_log2, cmax, cmin = _logic_minmax_from_four(
            (float(row["b00_logic"]), float(row["b10_logic"]), float(row["b01_logic"]), float(row["b11_logic"])),
            eps_ratio=eps_ratio,
            eps_range=eps_range,
        )

        anchors: dict[str, float] = {"00": np.nan, "10": np.nan, "01": np.nan, "11": np.nan}
        if ref_tab is not None and not ref_tab.empty:
            for _, rr in ref_tab.iterrows():
                anchors[str(rr["corner"])] = float(rr["anchor_mean"])

        def ystar(b: float, a: float) -> float:
            # Spec-aligned:
            #   denom = max(a + α, eps_ref)
            #   y_linear = (b + eps_abs) / denom
            #   y* = log2( max(y_linear + δ, eps_ratio) )
            if np.isnan(a):
                raise ValueError("SFXI: missing anchor for a corner; check reference configuration")
            denom = max(float(a) + float(ref_add_alpha), float(eps_ref))
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError(f"SFXI: invalid anchor denominator (max(A+alpha, eps_ref))={denom}")
            y_linear = (float(b) + float(eps_abs)) / denom
            log_arg = y_linear + float(log2_offset_delta)
            # Guard only for the log argument (assertive, no silent backfills elsewhere)
            return float(np.log2(np.maximum(log_arg, float(eps_ratio))))

        y00 = ystar(float(row["b00_intensity"]), anchors["00"])
        y10 = ystar(float(row["b10_intensity"]), anchors["10"])
        y01 = ystar(float(row["b01_intensity"]), anchors["01"])
        y11 = ystar(float(row["b11_intensity"]), anchors["11"])

        rec: dict[str, object] = {c: row[c] for c in idx_cols}
        rec.update(
            v00=float(v[0]),
            v10=float(v[1]),
            v01=float(v[2]),
            v11=float(v[3]),
            y00_star=y00,
            y10_star=y10,
            y01_star=y01,
            y11_star=y11,
            r_logic=r_logic,
            flat_logic=flat,
            # Self-describing diagnostics for r_logic:
            r_logic_min=float(rmin),
            r_logic_max=float(rmax),
            logic_span_log2=float(span_log2),
            r_logic_corner_min=cmin,
            r_logic_corner_max=cmax,
        )
        out_rows.append(rec)

    df = pd.DataFrame.from_records(out_rows)

    # Keep numerics as float for stability and downstream math.
    float_cols = [
        "v00",
        "v10",
        "v01",
        "v11",
        "y00_star",
        "y10_star",
        "y01_star",
        "y11_star",
        "r_logic",
        "r_logic_min",
        "r_logic_max",
        "logic_span_log2",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Contract compliance: flat_logic must be a proper boolean dtype.
    if "flat_logic" in df.columns:
        df["flat_logic"] = df["flat_logic"].astype(bool)
        if not is_bool_dtype(df["flat_logic"]):
            raise TypeError(
                f"SFXI internal error: expected boolean dtype for 'flat_logic', got {df['flat_logic'].dtype!r}"
            )

    return df
