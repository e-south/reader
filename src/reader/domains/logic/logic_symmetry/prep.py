"""Logic-symmetry prep helper: pick one time per (design x batch x treatment)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reader.domains.logic.treatment_columns import choose_treatment_column, normalize_treatment_series


def _choose_time(times: np.ndarray, *, mode: str, target: float | None, tol: float) -> float:
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
    design_by: list[str],
    batch_col: str,
    treatment_map: dict[str, str],
    treatment_column: str | None = None,
    mode: str = "last",
    target_time: float | None = None,
    tolerance: float = 0.51,
    align_corners: bool = False,
    case_sensitive: bool = True,
    time_column: str = "time",
) -> pd.DataFrame:
    required = set(design_by + [batch_col, "channel", time_column, "value"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    work = df[df["channel"] == response_channel].copy()
    treatment_col = choose_treatment_column(
        work,
        treatment_map,
        case_sensitive=case_sensitive,
        preferred=treatment_column,
    )
    tvals = list(treatment_map.values())
    if not case_sensitive:
        work["_t_norm"] = normalize_treatment_series(work[treatment_col])
        tvals_norm = [str(v).strip().casefold() for v in tvals]
        work = work[work["_t_norm"].isin(tvals_norm)].copy()
        work["_corner"] = work["_t_norm"].map({str(v).strip().casefold(): k for k, v in treatment_map.items()})
    else:
        work = work[work[treatment_col].astype(str).isin([str(v) for v in tvals])].copy()
        work["_corner"] = work[treatment_col].map({v: k for k, v in treatment_map.items()})

    if work.empty:
        raise ValueError("No rows remain after filtering to response_channel and treatment_map labels")

    try:
        pd.to_numeric(work[batch_col])
    except Exception as err:
        raise ValueError(f"Batch column '{batch_col}' must be numeric (0,1,2,...)") from err

    keys_db = design_by + [batch_col]
    keys_dbt = keys_db + ["_corner"]

    if align_corners:
        selected_rows = []
        failures = []
        for keys, gdb in work.groupby(keys_db, dropna=False):
            all_times = gdb[time_column].unique().astype(float)
            try:
                anchor = _choose_time(all_times, mode=mode, target=target_time, tol=tolerance)
            except Exception as exc:
                failures.append(f"{dict(zip(keys_db, keys, strict=False))} → {exc}")
                continue
            ok = True
            for _, g in gdb.groupby("_corner", dropna=False):
                tvals = g[time_column].astype(float).values
                if tvals.size == 0:
                    ok = False
                    break
                diffs = np.abs(tvals - anchor)
                if diffs.min() > tolerance:
                    ok = False
                    break
                pick_time = float(tvals[diffs.argmin()])
                selected_rows.append(g.loc[g[time_column] == pick_time].iloc[0])
            if not ok:
                failures.append(
                    f"{dict(zip(keys_db, keys, strict=False))} → no per-corner time within ±{tolerance} h of anchor={anchor}"
                )
        if failures:
            raise ValueError("logic_symmetry_prep align_corners failed:\n  " + "\n  ".join(failures[:50]))
        return pd.DataFrame(selected_rows).drop(columns=["_t_norm"], errors="ignore")

    chosen = []
    failures = []
    for keys, g in work.groupby(keys_dbt, dropna=False):
        times = g[time_column].astype(float).values
        try:
            tsel = _choose_time(times, mode=mode, target=target_time, tol=tolerance)
        except Exception as exc:
            failures.append(f"{dict(zip(keys_dbt, keys, strict=False))} → {exc}")
            continue
        chosen.append((keys, tsel))

    if failures:
        raise ValueError("logic_symmetry_prep failed:\n  " + "\n  ".join(failures[:50]))

    keep = pd.Series(False, index=work.index)
    for keys, tsel in chosen:
        mask = (work[keys_dbt] == pd.Series(keys, index=keys_dbt)).all(axis=1) & (
            work[time_column].astype(float) == float(tsel)
        )
        keep |= mask

    return work.loc[keep].copy().drop(columns=["_t_norm"], errors="ignore")
