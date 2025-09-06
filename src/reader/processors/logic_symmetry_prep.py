"""
--------------------------------------------------------------------------------
<reader project>
src/reader/processors/logic_symmetry_prep.py

Prep helper for logic-symmetry:

Select one time per (design…, batch, treatment) from time-series data so
that the logic_symmetry plotter's snapshot contract is satisfied.

Author(s): Eric J. South (new)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _norm_case(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.casefold()


def _choose_time(times: np.ndarray, *, mode: str, target: Optional[float], tol: float) -> float:
    times = np.asarray(sorted(times), dtype=float)
    if times.size == 0:
        raise ValueError("No times available in group")
    if mode == "first":
        return float(times.min())
    if mode == "last":
        return float(times.max())
    if mode == "median":
        med = float(np.median(times))
        return float(times[np.abs(times - med).argmin()])
    if mode == "exact":
        if target is None:
            raise ValueError("prep.mode='exact' requires target_time")
        # accept within tolerance, but require an exact timestamp match within tol
        diffs = np.abs(times - float(target))
        if diffs.min() > tol:
            raise ValueError(f"Requested exact time {target} not present within ±{tol} h")
        return float(times[diffs.argmin()])
    if mode == "nearest":
        if target is None:
            raise ValueError("prep.mode='nearest' requires target_time")
        diffs = np.abs(times - float(target))
        if diffs.min() > tol:
            raise ValueError(f"No time within ±{tol} h of target {target} h")
        return float(times[diffs.argmin()])
    raise ValueError(f"Unknown prep.mode '{mode}'")


def prepare_for_logic_symmetry(
    df: pd.DataFrame,
    *,
    response_channel: str,
    design_by: List[str],
    batch_col: str,
    treatment_map: Dict[str, str],
    mode: str = "last",
    target_time: Optional[float] = None,
    tolerance: float = 0.51,
    align_corners: bool = False,
    case_sensitive: bool = True,
    time_column: str = "time",
) -> pd.DataFrame:
    """
    Return a filtered DataFrame in which, for each (design…, batch, treatment),
    exactly ONE time remains—selected according to *mode*.

    Parameters
    ----------
    response_channel : str
        Channel to filter on before selection.
    design_by : list[str]
        Columns that define a design (e.g., ["genotype"]).
    batch_col : str
        Name of the batch column (must be numeric).
    treatment_map : dict
        Mapping {'00','10','01','11'} → exact labels in the data.
        Only those labels are kept.
    mode : {'last','first','median','nearest','exact'}
        Strategy for picking one time per group.
    target_time : float, optional
        Used by 'nearest'/'exact' modes.
    tolerance : float
        Allowed deviation from target_time for 'nearest'/'exact' (in hours).
    align_corners : bool
        If True, choose one **common** time per (design,batch) and then
        pick per-treatment rows nearest to that common time within tolerance.
    case_sensitive : bool
        Whether treatment label matching is case sensitive.
    time_column : str
        Name of the time column.

    Raises
    ------
    ValueError
        If any (design,batch,treatment) group cannot produce a single time
        according to the selection policy.

    Notes
    -----
    • This does not compute metrics; it only filters to a single time per group.
    • The logic_symmetry plotter will still validate corner completeness, etc.
    """
    required = set(design_by + [batch_col, "channel", time_column, "treatment", "value"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    # Filter to channel and mapped treatments
    work = df[df["channel"] == response_channel].copy()
    tvals = list(treatment_map.values())
    if not case_sensitive:
        work["_t_norm"] = _norm_case(work["treatment"])
        tvals_norm = [str(v).strip().casefold() for v in tvals]
        work = work[work["_t_norm"].isin(tvals_norm)].copy()
        work["_corner"] = work["_t_norm"].map({str(v).strip().casefold(): k for k, v in treatment_map.items()})
    else:
        work = work[work["treatment"].astype(str).isin([str(v) for v in tvals])].copy()
        work["_corner"] = work["treatment"].map({v: k for k, v in treatment_map.items()})

    if work.empty:
        raise ValueError("No rows remain after filtering to response_channel and treatment_map labels")

    # sanity: batch numeric
    try:
        pd.to_numeric(work[batch_col])
    except Exception:
        raise ValueError(f"Batch column '{batch_col}' must be numeric (0,1,2,...)")

    keys_db = design_by + [batch_col]
    keys_dbt = keys_db + ["_corner"]

    # Align corners to a **common** time per (design,batch) if requested
    if align_corners:
        if mode not in {"nearest", "first", "last", "median", "exact"}:
            raise ValueError(f"Unsupported prep.mode {mode!r} for align_corners")

        selected_rows = []
        failures = []

        for keys, gdb in work.groupby(keys_db, dropna=False):
            # choose an anchor time for this (design,batch)
            all_times = gdb[time_column].unique().astype(float)
            try:
                if mode in {"nearest", "exact"}:
                    anchor = _choose_time(all_times, mode=mode, target=target_time, tol=tolerance)
                else:
                    anchor = _choose_time(all_times, mode=mode, target=None, tol=tolerance)
            except Exception as exc:
                failures.append(f"{dict(zip(keys_db, keys))} → {exc}")
                continue

            ok = True
            for corner, g in gdb.groupby("_corner", dropna=False):
                # for each treatment, pick one row nearest to anchor within tol
                tvals = g[time_column].astype(float).values
                if tvals.size == 0:
                    ok = False
                    break
                diffs = np.abs(tvals - anchor)
                if diffs.min() > tolerance:
                    ok = False
                    break
                pick_time = float(tvals[diffs.argmin()])
                selected_rows.append(
                    g.loc[g[time_column] == pick_time].iloc[0]
                )
            if not ok:
                failures.append(f"{dict(zip(keys_db, keys))} → no per-corner time within ±{tolerance} h of anchor={anchor}")

        if failures:
            raise ValueError("logic_symmetry_prep align_corners failed for some groups:\n  " + "\n  ".join(failures[:50]))

        out = pd.DataFrame(selected_rows)
        out = out.drop(columns=["_t_norm"], errors="ignore")
        return out

    # Otherwise: pick one time **per (design,batch,corner)** independently
    chosen = []
    failures = []
    for keys, g in work.groupby(keys_dbt, dropna=False):
        times = g[time_column].astype(float).values
        try:
            if mode in {"nearest", "exact"}:
                tsel = _choose_time(times, mode=mode, target=target_time, tol=tolerance)
            else:
                tsel = _choose_time(times, mode=mode, target=None, tol=tolerance)
        except Exception as exc:
            failures.append(f"{dict(zip(keys_dbt, keys))} → {exc}")
            continue

        chosen.append((keys, tsel))

    if failures:
        raise ValueError("logic_symmetry_prep failed to select a time for some groups:\n  " + "\n  ".join(failures[:50]))

    # Build mask to keep chosen (exact matches)
    keep = pd.Series(False, index=work.index)
    for (keys, tsel) in chosen:
        mask = (work[keys_dbt] == pd.Series(keys, index=keys_dbt)).all(axis=1) & (work[time_column].astype(float) == float(tsel))
        keep |= mask

    out = work.loc[keep].copy()
    return out.drop(columns=["_t_norm"], errors="ignore")
