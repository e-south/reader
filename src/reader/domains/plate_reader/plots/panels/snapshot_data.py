from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import polars as pl

from reader.domains.plate_reader.analysis.timepoints import nearest_time_per_key


@dataclass(frozen=True)
class SnapshotSelection:
    rows: pd.DataFrame
    time_used: float
    fell_back: bool
    fallback_delta: float | None = None
    fallback_times_preview: str | None = None


def select_snapshot_rows(
    *,
    df: pd.DataFrame,
    target_time: float,
    keys: Sequence[str],
    channel: str,
    tolerance: float,
) -> SnapshotSelection:
    snapped = nearest_time_per_key(df, target_time=float(target_time), keys=list(keys), tol=float(tolerance))
    snapped = snapped[snapped["channel"].astype(str) == str(channel)].copy()
    if snapped.empty:
        fallback = nearest_time_per_key(df, target_time=float(target_time), keys=list(keys), tol=float("inf"))
        fallback = fallback[fallback["channel"].astype(str) == str(channel)].copy()
        if fallback.empty:
            return SnapshotSelection(rows=fallback, time_used=float(target_time), fell_back=True)
        times_used = pd.to_numeric(fallback["time"], errors="coerce").dropna()
        unique_times = sorted(times_used.unique().tolist())
        representative_time = unique_times[0] if len(unique_times) == 1 else float(pd.Series(unique_times).median())
        preview = ", ".join(f"{time:.2f}" for time in unique_times[:6]) + (" …" if len(unique_times) > 6 else "")
        return SnapshotSelection(
            rows=fallback,
            time_used=float(times_used.median()) if not times_used.empty else float(target_time),
            fell_back=True,
            fallback_delta=abs(float(representative_time) - float(target_time)),
            fallback_times_preview=preview,
        )

    times_used = pd.to_numeric(snapped["time"], errors="coerce").dropna()
    time_used = float(times_used.median()) if not times_used.empty else float(target_time)
    return SnapshotSelection(rows=snapped, time_used=time_used, fell_back=False)


def summarize_snapshot_values(
    *,
    df: pd.DataFrame,
    group_cols: Sequence[str],
    dispersion: str,
) -> pd.DataFrame:
    stats_pl = (
        pl.from_pandas(df)
        .with_columns(
            pl.when(pl.col("value").cast(pl.Float64, strict=False).is_nan())
            .then(None)
            .otherwise(pl.col("value").cast(pl.Float64, strict=False))
            .alias("value")
        )
        .group_by(list(group_cols))
        .agg(
            pl.col("value").count().alias("n"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").median().alias("median"),
            pl.col("value").std(ddof=1).alias("std"),
        )
    )

    if dispersion == "iqr":
        quantiles = (
            pl.from_pandas(df)
            .group_by(list(group_cols))
            .agg(
                pl.col("value").quantile(0.25, interpolation="linear").alias("q1"),
                pl.col("value").quantile(0.75, interpolation="linear").alias("q3"),
            )
        )
        stats_pl = stats_pl.join(quantiles, on=list(group_cols), how="left")

    return stats_pl.sort(list(group_cols)).to_pandas(use_pyarrow_extension_array=False)
