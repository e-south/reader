"""Trace summaries and compact endpoint components for time-series review."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .plot_style import LEGEND_SIZE, PANEL_TITLE_SIZE, style_data_axis
from .sources import STATE_ORDER
from .visual_labels import (
    STATE_COLORS,
    STATE_MARKERS,
)


def signal_rows(
    traces: pd.DataFrame,
    *,
    signal_kind: str,
    design_id: str,
    reference_id: str,
) -> pd.DataFrame:
    signal = traces.loc[traces["signal_kind"].astype(str).eq(signal_kind)].copy()
    signal = signal.loc[signal["design_id"].astype(str).isin({design_id, reference_id})]
    if signal_kind != "magnitude":
        signal = signal.loc[signal["design_id"].astype(str).eq(design_id)]
    if signal.empty:
        raise ValueError(f"review traces contain no {signal_kind!r} observations for {design_id!r}.")
    return signal


def trace_interval(trace: pd.DataFrame, *, log_transform: bool, confidence_level: float) -> pd.DataFrame:
    work = trace.loc[:, ["time_from_event_h", "value"]].copy()
    values = work["value"].to_numpy(dtype=float)
    if log_transform:
        if np.any(values <= 0.0):
            raise ValueError("review traces contain non-positive ratios.")
        work["value"] = np.log2(values)
    tail = (1.0 - confidence_level) / 2.0
    summary = work.groupby("time_from_event_h")["value"].quantile([tail, 0.5, 1.0 - tail]).unstack()
    summary.columns = ["lower", "median", "upper"]
    return summary.reset_index()


def style_trajectory_axis(
    axis: plt.Axes,
    *,
    title: str,
    ylabel: str,
    event_label: str,
    uncertainty: float,
    selected: pd.Series,
    annotate_spans: bool = False,
    show_event_uncertainty: bool = True,
) -> None:
    if show_event_uncertainty:
        event_span = axis.axvspan(-uncertainty, uncertainty, color="#9ca3af", alpha=0.20, zorder=1)
        event_span.set_gid("event-time-uncertainty-window")
    response_window = axis.axvspan(
        float(selected["window_start_event_h"]),
        float(selected["window_end_event_h"]),
        color="#f59e0b",
        alpha=0.11,
        zorder=1,
    )
    response_window.set_gid("selected-response-window")
    event_line = axis.axvline(0.0, color="#111827", linewidth=0.9, zorder=3)
    event_line.set_gid("recorded-event-time")
    axis.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="semibold")
    axis.set_xlabel(f"Hours from {event_label.lower()}")
    axis.set_ylabel(ylabel)
    axis.set_box_aspect(1.0)
    style_data_axis(axis, grid_axis="both")
    if annotate_spans:
        axis.legend(
            handles=[
                Patch(facecolor="#9ca3af", alpha=0.38, edgecolor="none", label="Event interval"),
                Patch(facecolor="#f59e0b", alpha=0.28, edgecolor="none", label="Selected window"),
            ],
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#d1d5db",
            fontsize=LEGEND_SIZE,
            handlelength=1.0,
            handletextpad=0.45,
            labelspacing=0.25,
            borderaxespad=0.35,
        )


def legend_handles(
    *,
    state_labels: dict[str, object],
) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=STATE_COLORS[state],
            marker=STATE_MARKERS[state],
            markersize=4,
            linewidth=2,
            label=str(state_labels[state]),
        )
        for state in STATE_ORDER
    ]


__all__ = [
    "legend_handles",
    "signal_rows",
    "style_trajectory_axis",
    "trace_interval",
]
