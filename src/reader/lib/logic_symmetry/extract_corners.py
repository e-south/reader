"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/logic_symmetry/extract_corners.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

LOG = logging.getLogger(__name__)

REQUIRED_BASE_COLS = ["position", "time", "channel", "value", "treatment"]


@dataclass(frozen=True)
class MappingConfig:
    treatment_map: dict[str, str]  # keys: "00","10","01","11" → exact data labels
    case_sensitive: bool
    design_by: list[str]
    batch_col: str
    response_channel: str
    replicate_stat: str  # "mean" | "median"


def _assert_required_columns(df: pd.DataFrame, cfg: MappingConfig) -> None:
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")
    meta_missing = [c for c in cfg.design_by if c not in df.columns]
    if meta_missing:
        raise ValueError(f"Missing design_by columns: {meta_missing}")
    if cfg.batch_col not in df.columns:
        raise ValueError(f"Missing required batch column '{cfg.batch_col}'")

    # Batch must be numeric ordinal
    try:
        pd.to_numeric(df[cfg.batch_col])
    except Exception as err:
        raise ValueError(f"Batch column '{cfg.batch_col}' must be numeric (0,1,2,...)") from err

    if cfg.response_channel not in df["channel"].unique().tolist():
        raise ValueError(f"response_channel '{cfg.response_channel}' not present in 'channel' column")


def _normalize(s: pd.Series) -> pd.Series:
    # normalization for case-insensitive matching
    return s.astype(str).str.strip().str.casefold()


def _rep_agg(series: pd.Series, how: str) -> float:
    if how == "median":
        return float(series.median())
    return float(series.mean())


def _sd(series: pd.Series) -> float:
    # population-like SD with ddof=1 when n>1; 0 otherwise
    n = series.size
    if n <= 1:
        return 0.0
    return float(series.std(ddof=1))


def _fail_if_multiple_times(df: pd.DataFrame, group_cols: list[str]) -> None:
    """
    Enforce: exactly ONE time per (design..., batch, treatment).
    If violations exist, raise with the list of offending groups and their times.
    """
    chk = df.groupby(group_cols)["time"].nunique().reset_index(name="n_times")
    bad = chk[chk["n_times"] > 1]
    if not bad.empty:
        offenders = []
        for _, row in bad.iterrows():
            filt = (df[group_cols] == row[group_cols]).all(axis=1)
            times = sorted(df.loc[filt, "time"].unique().tolist())
            key_str = ", ".join(f"{c}={row[c]!r}" for c in group_cols)
            offenders.append(f"{key_str} → times={times}")
        msg = "Snapshot violation: more than one 'time' for some (design…, batch, treatment).\n" + "\n".join(
            offenders[:50]
        )
        raise ValueError(msg)


def resolve_and_aggregate(df: pd.DataFrame, cfg: MappingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        points_df: one row per (design_by…, batch) with corner means, SDs, counts
        per_corner_df: per (design…, batch, corner) aggregated table
    """
    _assert_required_columns(df, cfg)

    # Filter to the channel
    before = len(df)
    df = df[df["channel"] == cfg.response_channel].copy()
    LOG.info("• extract: filtered to channel %r → %d/%d rows", cfg.response_channel, len(df), before)
    if df.empty:
        raise ValueError(f"No rows for response_channel '{cfg.response_channel}'")

    # Keep only rows whose treatment matches one of the mapped labels
    map_vals = list(cfg.treatment_map.values())
    if not cfg.case_sensitive:
        df["_t_norm"] = _normalize(df["treatment"])
        map_norm = [str(v).strip().casefold() for v in map_vals]
        df = df[df["_t_norm"].isin(map_norm)].copy()
    else:
        df = df[df["treatment"].astype(str).isin([str(v) for v in map_vals])].copy()
    LOG.info(
        "• extract: kept rows matching treatment_map labels → %d rows; unique treatments kept=%d",
        len(df),
        df["treatment"].nunique(),
    )

    if df.empty:
        counts = df["treatment"].value_counts().to_string() if "treatment" in df.columns else "(no treatment column)"
        raise ValueError(f"No rows match any treatment_map labels. Check spelling/case.\nAvailable counts:\n{counts}")

    # Add 'corner' column via reverse lookup (value→corner)
    rev = {}
    seen = set()
    for corner in ("00", "10", "01", "11"):
        val = cfg.treatment_map.get(corner)
        if val in seen:
            raise ValueError(f"Duplicate treatment_map values detected: {val!r} is used by multiple corners")
        seen.add(val)
        rev[str(val) if cfg.case_sensitive else str(val).strip().casefold()] = corner

    if cfg.case_sensitive:
        df["corner"] = df["treatment"].astype(str).map(lambda x: rev.get(x))
    else:
        df["corner"] = _normalize(df["treatment"]).map(lambda x: rev.get(x))

    # Enforce snapshot rule per (design…, batch, treatment/corner)
    group_cols = cfg.design_by + [cfg.batch_col, "corner"]
    _fail_if_multiple_times(df, group_cols)

    # Aggregate replicates per corner at that unique time
    agg_rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        # There is exactly one time by construction; keep it
        t_unique = float(g["time"].iloc[0])
        val = pd.to_numeric(g["value"], errors="coerce").dropna()
        if val.empty:
            key_str = ", ".join(f"{c}={k!r}" for c, k in zip(group_cols, keys, strict=False))
            raise ValueError(f"Non-numeric or missing 'value' for group: {key_str}")
        mean_or_med = _rep_agg(val, cfg.replicate_stat)
        sd = _sd(val)
        n = int(val.size)
        record = dict(zip(group_cols, keys, strict=False))
        record.update(time=t_unique, y_mean=mean_or_med, y_sd=sd, y_n=n)
        agg_rows.append(record)

    per_corner = pd.DataFrame.from_records(agg_rows)
    LOG.info("• extract: per-corner aggregates = %d rows (design×batch×corner)", len(per_corner))

    # Pivot to wide per (design…, batch)
    idx_cols = cfg.design_by + [cfg.batch_col]
    pivot_mean = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_mean", aggfunc="first")
    pivot_sd = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_sd", aggfunc="first")
    pivot_n = per_corner.pivot_table(index=idx_cols, columns="corner", values="y_n", aggfunc="first")

    # Ensure all four corners present for every group (fail-fast)
    required = ["00", "10", "01", "11"]
    missing_groups = []
    for idx, row in pivot_mean.iterrows():
        missing = [c for c in required if pd.isna(row.get(c))]
        if missing:
            key_str = ", ".join(
                f"{c}={v!r}" for c, v in zip(idx_cols, (idx if isinstance(idx, tuple) else (idx,)), strict=False)
            )
            missing_groups.append(f"{key_str} → missing corners: {missing}")
    if missing_groups:
        LOG.error("• extract: incomplete corner set for some groups (%d shown):", len(missing_groups))
        for line in missing_groups[:10]:
            LOG.error("    %s", line)
        msg = "Incomplete corner set: some (design…, batch) groups lack one or more of {00,10,01,11}.\n" + "\n".join(
            missing_groups[:50]
        )
        raise ValueError(msg)

    pivot_mean = pivot_mean[required].rename(columns={"00": "b00", "10": "b10", "01": "b01", "11": "b11"})
    pivot_sd = pivot_sd[required].rename(columns={"00": "sd00", "10": "sd10", "01": "sd01", "11": "sd11"})
    pivot_n = pivot_n[required].rename(columns={"00": "n00", "10": "n10", "01": "n01", "11": "n11"}).astype(int)

    points = pivot_mean.join(pivot_sd, how="left").join(pivot_n, how="left").reset_index()

    LOG.info("• extract: wide table rows (designxbatch) = %d", len(points))
    return points, per_corner
