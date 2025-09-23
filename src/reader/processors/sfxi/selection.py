"""
--------------------------------------------------------------------------------
<reader project>
reader/processors/sfxi/selection.py

Time selection + cornerization/aggregation for SFXI.

Author(s): Eric J. South (rewired)
--------------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = ["position", "time", "channel", "value", "treatment"]


@dataclass(frozen=True)
class CornerizeResult:
    # one row per (design×batch×corner)
    per_corner: pd.DataFrame
    # wide one row per (design×batch) with b00.., sd00.., n00..
    points: pd.DataFrame
    # bookkeeping
    chosen_times: Dict[object, float]  # batch -> time
    dropped_batches: List[object]


def _normalize(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.casefold()


def _enforce_columns(df: pd.DataFrame, design_by: List[str], batch_col: str) -> None:
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"SFXI: tidy data missing required columns: {miss}")
    m2 = [c for c in design_by if c not in df.columns]
    if m2:
        raise ValueError(f"SFXI: missing design_by columns: {m2}")
    if batch_col not in df.columns:
        raise ValueError(f"SFXI: missing batch column '{batch_col}'")


def _pick_time_for_batch(times: np.ndarray, target: Optional[float], mode: str) -> Optional[float]:
    if times.size == 0:
        return None
    times = np.unique(times.astype(float))
    if target is None:
        return float(np.max(times))  # last available if not specified
    if mode == "exact":
        hits = times[np.isclose(times, float(target), rtol=0, atol=1e-12)]
        return float(hits[0]) if hits.size else None
    if mode == "nearest":
        return float(times[np.argmin(np.abs(times - float(target)))])
    if mode == "last_before":
        candidates = times[times <= float(target)]
        return float(candidates.max()) if candidates.size else None
    if mode == "first_after":
        candidates = times[times >= float(target)]
        return float(candidates.min()) if candidates.size else None
    raise ValueError(f"Unknown mode {mode}")


def select_times(
    df: pd.DataFrame,
    *,
    channel: str,
    treatment_map: Dict[str, str],
    case_sensitive: bool,
    batch_col: str,
    target_time_h: Optional[float],
    time_mode: str,
    tolerance_h: Optional[float],
    per_batch: bool,
    on_missing: str,
) -> Tuple[pd.DataFrame, Dict[object, float], List[object]]:
    """
    Decide the snapshot time per batch (or globally) and filter df to those rows.
    Returns (filtered_df, chosen_times, dropped_batches)
    """
    # Filter to channel and mapped treatments only
    work = df[df["channel"] == channel].copy()
    if not case_sensitive:
        work["_t_norm"] = _normalize(work["treatment"])
        mapped = [str(v).strip().casefold() for v in treatment_map.values()]
        work = work[work["_t_norm"].isin(mapped)].copy()
    else:
        mapped = [str(v) for v in treatment_map.values()]
        work = work[work["treatment"].astype(str).isin(mapped)].copy()

    if work.empty:
        raise ValueError("SFXI: No rows for the requested channel/treatment_map combination.")

    chosen: Dict[object, float] = {}
    dropped: List[object] = []

    if per_batch:
        for b, g in work.groupby(batch_col, dropna=False):
            t = _pick_time_for_batch(g["time"].to_numpy(dtype=float), target_time_h, time_mode)
            if t is None or (tolerance_h is not None and target_time_h is not None and abs(t - target_time_h) > tolerance_h):
                if on_missing == "error":
                    raise ValueError(f"SFXI: could not choose time for batch={b!r} (mode={time_mode}, target={target_time_h}, tol={tolerance_h})")
                dropped.append(b)
                continue
            chosen[b] = float(t)
        if not chosen and on_missing == "error":
            raise ValueError("SFXI: no batches met the time selection rule.")
        mask = work[batch_col].map(chosen).notna() & np.isclose(work["time"].astype(float), work[batch_col].map(chosen).astype(float), rtol=0, atol=1e-9)
        out = work[mask].copy()
    else:
        t = _pick_time_for_batch(work["time"].to_numpy(dtype=float), target_time_h, time_mode)
        if t is None or (tolerance_h is not None and target_time_h is not None and abs(t - target_time_h) > tolerance_h):
            if on_missing == "error":
                raise ValueError("SFXI: could not choose a global time.")
            return work.iloc[0:0].copy(), {}, []
        chosen["__global__"] = float(t)
        out = work[np.isclose(work["time"].astype(float), float(t), rtol=0, atol=1e-9)].copy()

    return out, chosen, dropped


def cornerize_and_aggregate(
    df: pd.DataFrame,
    *,
    design_by: List[str],
    batch_col: str,
    treatment_map: Dict[str, str],
    case_sensitive: bool,
    time_column: str,
    channel: str,
    target_time_h: Optional[float],
    time_mode: str,
    time_tolerance_h: Optional[float],
    time_per_batch: bool,
    on_missing_time: str,
    require_all_corners_per_design: bool,
) -> CornerizeResult:
    """Main entry: choose times, map labels → corners, aggregate replicates, pivot wide."""
    _enforce_columns(df, design_by, batch_col)

    # Rename custom time col to 'time' if needed
    if time_column != "time":
        if time_column not in df.columns:
            raise ValueError(f"SFXI: time column '{time_column}' not found in tidy data.")
        df = df.rename(columns={time_column: "time"})

    snap, chosen, dropped = select_times(
        df,
        channel=channel,
        treatment_map=treatment_map,
        case_sensitive=case_sensitive,
        batch_col=batch_col,
        target_time_h=target_time_h,
        time_mode=time_mode,
        tolerance_h=time_tolerance_h,
        per_batch=time_per_batch,
        on_missing=on_missing_time,
    )

    if time_per_batch and dropped and on_missing_time == "drop-all":
        snap = snap.iloc[0:0].copy()
        chosen = {}

    if snap.empty:
        raise ValueError("SFXI: no rows remain after time selection.")

    # Reverse map value -> corner
    rev: Dict[str, str] = {}
    seen = set()
    for k in ("00", "10", "01", "11"):
        v = treatment_map[k]
        key = str(v) if case_sensitive else str(v).strip().casefold()
        if key in rev:
            raise ValueError(f"SFXI: duplicate treatment_map value: {v!r}")
        rev[key] = k
        seen.add(v)

    corner_col = "_t_norm" if not case_sensitive else "treatment"
    snap["corner"] = (snap[corner_col].astype(str) if corner_col in snap.columns else snap["treatment"].astype(str))
    if not case_sensitive:
        snap["corner"] = snap["corner"].str.strip().str.casefold()
    snap["corner"] = snap["corner"].map(rev)

    # Ensure one time per (design×batch×corner) by construction via selection; still sanity
    chk = snap.groupby(design_by + [batch_col, "corner"])["time"].nunique().reset_index(name="n_times")
    bad = chk[chk["n_times"] > 1]
    if not bad.empty:
        raise ValueError("SFXI: more than one time within a (design x batch x corner) after selection; check tidy data.")

    # Numeric aggregation of replicates
    def _agg_mean(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if s.size else float("nan")
    def _agg_sd(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.std(ddof=1)) if s.size >= 2 else 0.0
    def _agg_n(series: pd.Series) -> int:
        return int(pd.to_numeric(series, errors="coerce").dropna().size)

    grp_cols = design_by + [batch_col, "corner"]
    g = snap.groupby(grp_cols, dropna=False)
    per_corner = g.agg(time=("time", "first"),
                      y_mean=("value", _agg_mean),
                      y_sd=("value", _agg_sd),
                      y_n=("value", _agg_n)).reset_index()

    # Pivot to wide
    idx_cols = design_by + [batch_col]
    m = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_mean", aggfunc="first")
    s = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_sd",   aggfunc="first")
    n = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_n",    aggfunc="first")

    req = ["00", "10", "01", "11"]
    if require_all_corners_per_design:
        missing_groups = []
        for idx, row in m.iterrows():
            miss = [c for c in req if pd.isna(row.get(c))]
            if miss:
                key = idx if isinstance(idx, tuple) else (idx,)
                key_str = ", ".join(f"{c}={v!r}" for c, v in zip(idx_cols, key))
                missing_groups.append(f"{key_str} → missing corners: {miss}")
        if missing_groups:
            raise ValueError("SFXI: incomplete corner set for some (design×batch):\n" + "\n".join(missing_groups[:40]))

    points = (
        m[req].rename(columns={"00":"b00","10":"b10","01":"b01","11":"b11"})
        .join(s[req].rename(columns={"00":"sd00","10":"sd10","01":"sd01","11":"sd11"}))
        .join(n[req].rename(columns={"00":"n00", "10":"n10", "01":"n01", "11":"n11"}))
        .reset_index()
    )
    return CornerizeResult(per_corner=per_corner, points=points, chosen_times=chosen, dropped_batches=dropped)
