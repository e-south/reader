"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domain/sfxi/selection.py

SFXI selection: time picking + cornerization + replicate aggregation.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLS = ["position", "time", "channel", "value", "treatment"]


@dataclass(frozen=True)
class CornerizeResult:
    per_corner: pd.DataFrame  # one row per (design×corner)
    points: pd.DataFrame  # wide one row per (design) with b00.., sd00.., n00..
    chosen_time: float
    time_warning: str | None = None


def _normalize(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.casefold()


def _choose_treatment_column(df: pd.DataFrame, treatment_map: dict[str, str], *, case_sensitive: bool) -> str:
    candidates = [c for c in ("treatment", "treatment_alias") if c in df.columns]
    if not candidates:
        raise ValueError("SFXI: neither 'treatment' nor 'treatment_alias' present in tidy data.")

    def _score(col: str) -> int:
        s = df[col].astype(str)
        if case_sensitive:
            want = {str(v) for v in treatment_map.values()}
        else:
            want = {str(v).strip().casefold() for v in treatment_map.values()}
            s = s.str.strip().str.casefold()
        return int(s.isin(list(want)).sum())

    scores = {c: _score(c) for c in candidates}
    # Prefer raw 'treatment' on ties — aliases are cosmetic.
    best = max(scores, key=lambda c: (scores[c], c == "treatment"))
    return best


def _enforce_columns(df: pd.DataFrame, design_by: list[str]) -> None:
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"SFXI: tidy data missing required columns: {miss}")
    m2 = [c for c in design_by if c not in df.columns]
    if m2:
        raise ValueError(f"SFXI: missing design_by columns: {m2}")


def _pick_time(times: np.ndarray, target: float | None, mode: str) -> float | None:
    if times.size == 0:
        return None
    times = np.unique(times.astype(float))
    if target is None:
        return float(np.max(times))
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
    raise ValueError(f"Unknown time mode {mode!r}")


def select_times(
    df: pd.DataFrame,
    *,
    channel: str,
    treatment_map: dict[str, str],
    case_sensitive: bool,
    target_time_h: float | None,
    time_mode: str,
    tolerance_h: float | None,
    on_missing: str,
) -> tuple[pd.DataFrame, float]:
    work = df[df["channel"] == channel].copy()
    # Decide which column to match against (raw preferred; alias tolerated).
    treatment_col = _choose_treatment_column(work, treatment_map, case_sensitive=case_sensitive)
    if case_sensitive:
        mapped = [str(v) for v in treatment_map.values()]
        work = work[work[treatment_col].astype(str).isin(mapped)].copy()
    else:
        mapped = [str(v).strip().casefold() for v in treatment_map.values()]
        norm_col = "__norm_treatment"
        work[norm_col] = _normalize(work[treatment_col])
        work = work[work[norm_col].isin(mapped)].copy()
        work["_t_norm"] = work[norm_col]

    if work.empty:
        # Helpful, assertive diagnostics
        present_unfiltered = df[df["channel"] == channel]
        present_vals = (
            present_unfiltered[treatment_col].astype(str).dropna().unique().tolist()
            if treatment_col in present_unfiltered.columns
            else []
        )
        preview_present = ", ".join(map(str, present_vals[:8])) + (" …" if len(present_vals) > 8 else "")
        preview_expected = ", ".join(map(str, list(treatment_map.values())))
        raise ValueError(
            "SFXI: no rows matched the requested channel/treatment_map.\n"
            f"  channel: {channel!r}\n"
            f"  using treatment column: {treatment_col!r}\n"
            f"  expected (from treatment_map): [{preview_expected}]\n"
            f"  present  (in data at channel): [{preview_present}]\n"
            "Hints: ensure aliases map raw treatments to the configured labels, or update treatment_map to "
            "match the values in the chosen treatment column."
        )

    t = _pick_time(work["time"].to_numpy(dtype=float), target_time_h, time_mode)
    if t is None:
        if on_missing == "error":
            raise ValueError("SFXI: could not choose a global time.")
        return work.iloc[0:0].copy(), float("nan")
    out = work[np.isclose(work["time"].astype(float), float(t), rtol=0, atol=1e-9)].copy()
    return out, float(t)


def cornerize_and_aggregate(
    df: pd.DataFrame,
    *,
    design_by: list[str],
    treatment_map: dict[str, str],
    case_sensitive: bool,
    time_column: str,
    channel: str,
    target_time_h: float | None,
    time_mode: str,
    time_tolerance_h: float | None,
    on_missing_time: str,
    require_all_corners_per_design: bool,
) -> CornerizeResult:
    _enforce_columns(df, design_by)

    if time_column != "time":
        if time_column not in df.columns:
            raise ValueError(f"SFXI: time column '{time_column}' not found.")
        df = df.rename(columns={time_column: "time"})

    snap, chosen_time = select_times(
        df,
        channel=channel,
        treatment_map=treatment_map,
        case_sensitive=case_sensitive,
        target_time_h=target_time_h,
        time_mode=time_mode,
        tolerance_h=time_tolerance_h,
        on_missing=on_missing_time,
    )
    time_warning: str | None = None
    if time_tolerance_h is not None and target_time_h is not None:
        tol = float(time_tolerance_h)
        tgt = float(target_time_h)
        delta = abs(float(chosen_time) - tgt)
        if delta > tol:
            time_warning = (
                f"Config requested time={tgt:.3f} h (mode={time_mode}, tol={tol:.3f} h) "
                f"but closest time was {float(chosen_time):.3f} h (|Δ|={delta:.3f} h); "
                f"using {float(chosen_time):.3f} h."
            )

    if snap.empty:
        raise ValueError("SFXI: no rows remain after time selection.")

    rev: dict[str, str] = {}
    for k in ("00", "10", "01", "11"):
        v = treatment_map[k]
        key = str(v) if case_sensitive else str(v).strip().casefold()
        if key in rev:
            raise ValueError(f"SFXI: duplicate treatment_map value: {v!r}")
        rev[key] = k

    # Map treatments → {00,10,01,11} using the same column we matched on.
    corner_source = _choose_treatment_column(snap, treatment_map, case_sensitive=case_sensitive)
    if case_sensitive:
        corner_values = snap[corner_source].astype(str)
        rev_keys = {str(k): v for k, v in rev.items()}
    else:
        corner_values = _normalize(snap[corner_source])
        rev_keys = {str(k).strip().casefold(): v for k, v in rev.items()}
    snap["corner"] = corner_values.map(rev_keys)

    group_cols = design_by + ["corner"]
    chk = snap.groupby(group_cols)["time"].nunique().reset_index(name="n_times")
    bad = chk[chk["n_times"] > 1]
    if not bad.empty:
        raise ValueError("SFXI: more than one time within (design×corner) after selection.")

    def _agg_mean(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.mean()) if s.size else float("nan")

    def _agg_sd(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.std(ddof=1)) if s.size >= 2 else 0.0

    def _agg_n(series: pd.Series) -> int:
        return int(pd.to_numeric(series, errors="coerce").dropna().size)

    grp_cols = design_by + ["corner"]
    g = snap.groupby(grp_cols, dropna=False)
    per_corner = g.agg(
        time=("time", "first"), y_mean=("value", _agg_mean), y_sd=("value", _agg_sd), y_n=("value", _agg_n)
    ).reset_index()

    idx_cols = design_by
    m = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_mean", aggfunc="first")
    s = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_sd", aggfunc="first")
    n = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_n", aggfunc="first")

    req = ["00", "10", "01", "11"]
    if require_all_corners_per_design:
        missing_groups = []
        for idx, row in m.iterrows():
            miss = [c for c in req if pd.isna(row.get(c))]
            if miss:
                key = idx if isinstance(idx, tuple) else (idx,)
                missing_groups.append(f"{dict(zip(idx_cols, key, strict=False))} → missing corners {miss}")
        if missing_groups:
            raise ValueError("SFXI: incomplete corner sets:\n" + "\n".join(missing_groups[:40]))

    points = (
        m[req]
        .rename(columns={"00": "b00", "10": "b10", "01": "b01", "11": "b11"})
        .join(s[req].rename(columns={"00": "sd00", "10": "sd10", "01": "sd01", "11": "sd11"}))
        .join(n[req].rename(columns={"00": "n00", "10": "n10", "01": "n01", "11": "n11"}))
        .reset_index()
    )
    return CornerizeResult(
        per_corner=per_corner,
        points=points,
        chosen_time=float(chosen_time),
        time_warning=time_warning,
    )
