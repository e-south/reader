from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _retron_sponge_presentation as retron_presentation
from .common import bootstrap_mean_interval


def trace_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    return grouped_trace_summary_frames(df).get((), empty_trace_summary_frame())


def grouped_trace_summary_frames(
    df: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
) -> dict[tuple[object, ...], pd.DataFrame]:
    if df.empty:
        return {}
    frame = df.copy()
    frame["time_from_stress"] = pd.to_numeric(frame["time_from_stress"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["time_from_stress"].notna() & frame["value"].notna()].copy()
    if frame.empty:
        return {}
    key_columns = list(group_columns)
    grouped_columns = [*key_columns, "time_from_stress"] if key_columns else ["time_from_stress"]
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | object]] = []
    for keys, series in frame.groupby(grouped_columns, dropna=False, sort=True)["value"]:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_values = key_values[: len(key_columns)] if key_columns else ()
        time_value = key_values[-1]
        mean, lower, upper = bootstrap_mean_interval(
            series.to_numpy(dtype=float, copy=False),
            ci=95.0,
            ci_boot=100,
            rng=rng,
        )
        record = {
            "time_from_stress": float(time_value),
            "mean": mean,
            "lower": lower,
            "upper": upper,
        }
        record.update(dict(zip(key_columns, group_values, strict=False)))
        rows.append(record)
    if not rows:
        return {}
    summary = pd.DataFrame(rows)
    interval_columns = ["time_from_stress", "mean", "lower", "upper"]
    if not key_columns:
        return {(): summary[interval_columns].sort_values("time_from_stress", kind="stable").reset_index(drop=True)}
    out: dict[tuple[object, ...], pd.DataFrame] = {}
    for keys, group in summary.groupby(key_columns, dropna=False, sort=False):
        group_key = keys if isinstance(keys, tuple) else (keys,)
        out[group_key] = group[interval_columns].sort_values("time_from_stress", kind="stable").reset_index(drop=True)
    return out


def empty_trace_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["time_from_stress", "mean", "lower", "upper"])


def trace_display_bounds(
    trace: pd.DataFrame | None,
    *,
    max_post_stress_hours: float,
) -> tuple[float, float] | None:
    if trace is None or trace.empty or "time_from_stress" not in trace.columns:
        return None
    values = pd.to_numeric(trace["time_from_stress"], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lower = min(float(np.min(finite)), 0.0)
    upper_cap = max(0.0, float(max_post_stress_hours))
    positive = finite[finite >= 0.0]
    observed_upper = float(np.max(positive)) if positive.size else 0.0
    upper = max(observed_upper, upper_cap) if upper_cap > 0.0 else observed_upper
    if upper <= lower:
        upper = float(np.max(finite))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def annotate_primary_window(ax: plt.Axes, trace: pd.DataFrame, *, stress_condition: str | None) -> None:
    span = retron_presentation.primary_window_span_bounds(trace, stress_condition=stress_condition)
    if span is None:
        return
    start, end = span
    ax.axvspan(start, end, color="#f3b4b0", alpha=0.14, zorder=0.15, linewidth=0.0)


def annotate_stress_addition(ax) -> None:
    if any(text.get_text() == "Stress addition" for text in ax.texts):
        return
    x_limits = ax.get_xlim()
    if len(x_limits) != 2 or not np.isfinite(x_limits).all() or not (x_limits[0] <= 0.0 <= x_limits[1]):
        return
    ax.annotate(
        "Stress addition",
        xy=(0.0, 0.08),
        xycoords=ax.get_xaxis_transform(),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#666666",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.25},
        zorder=3.5,
    )
