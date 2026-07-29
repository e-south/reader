"""Plate-reader timepoint selection and nearest-snapshot helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl


def nearest_time_per_key(df: pd.DataFrame, *, target_time: float, keys: Sequence[str], tol: float) -> pd.DataFrame:
    work = pl.from_pandas(df.reset_index(drop=True)).with_row_index("__row__")
    time_col = "time"
    time_expr = pl.col(time_col).cast(pl.Float64, strict=False)
    time_expr = pl.when(time_expr.is_nan()).then(None).otherwise(time_expr)
    work = work.with_columns(time_expr.alias(time_col))
    work = work.filter(pl.col(time_col).is_not_null())
    work = work.with_columns((pl.col(time_col) - float(target_time)).abs().alias("__dt__"))
    work = work.with_columns(pl.col("__dt__").min().over(list(keys)).alias("__dt_min__"))
    work = work.filter(pl.col("__dt__") == pl.col("__dt_min__"))
    work = work.sort("__row__").unique(subset=list(keys), keep="first")
    work = work.filter(pl.col("__dt__") <= float(tol))
    work = work.drop(["__dt__", "__dt_min__", "__row__"])
    return work.to_pandas(use_pyarrow_extension_array=False)


def choose_nearest_time(
    times: Sequence[object] | np.ndarray,
    *,
    target_time: float,
    tol: float | None,
    where: str,
    logger: logging.Logger | None = None,
) -> float:
    cleaned = pd.to_numeric(pd.Series(times), errors="coerce").dropna().to_numpy(dtype=float)
    if cleaned.size == 0:
        raise ValueError(f"{where}: no valid time values")
    unique_times = np.asarray(sorted(np.unique(cleaned)), dtype=float)
    diffs = np.abs(unique_times - float(target_time))
    chosen_index = int(np.argmin(diffs))
    chosen_time = float(unique_times[chosen_index])
    chosen_delta = float(diffs[chosen_index])
    if tol is not None and chosen_delta > float(tol) and logger is not None:
        logger.info(
            "[warn]%s[/warn] • requested t=%.2f h; nearest available t=%.2f h (Δ=%.2f h) — using nearest",
            where,
            float(target_time),
            chosen_time,
            chosen_delta,
        )
    return chosen_time


def infer_acquisition_transition_time_h(df: pd.DataFrame, *, time_col: str) -> float | None:
    """Return the first time in a later workbook acquisition segment.

    A sheet transition is acquisition provenance. It does not identify a
    biological intervention unless a separate, explicit event contract says
    that it does.
    """
    if "sheet_index" not in df.columns:
        return None
    sheet_values = pd.to_numeric(df["sheet_index"], errors="coerce").dropna()
    if sheet_values.empty:
        return None
    min_sheet = float(sheet_values.min())
    sheet_series = pd.to_numeric(df["sheet_index"], errors="coerce")
    times = pd.to_numeric(df.loc[sheet_series > min_sheet, time_col], errors="coerce").dropna()
    if times.empty:
        return None
    return float(times.min())


__all__ = ["choose_nearest_time", "infer_acquisition_transition_time_h", "nearest_time_per_key"]
