"""Axes-level components for the promoter-evidence publication figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import style_data_axis
from .review_time_series_components import signal_rows, style_trajectory_axis, trace_interval
from .sources import STATE_ORDER
from .visual_labels import (
    STATE_COLORS,
    STATE_MARKERS,
    anchored_fluorescence_axis_label,
    condition_ticks,
    response_axis_label,
)


def draw_trajectory_axis(
    axis: plt.Axes,
    *,
    traces: pd.DataFrame,
    signal_kind: str,
    design_id: str,
    reference_id: str,
    confidence_level: float,
    uncertainty: float,
    selected: pd.Series,
    event_label: str,
    title: str,
    ylabel: str,
) -> None:
    rows = signal_rows(traces, signal_kind=signal_kind, design_id=design_id, reference_id=reference_id)
    for (source_design, state), trace in rows.groupby(["design_id", "state"], sort=True):
        is_anchor = str(source_design) == reference_id and design_id != reference_id
        summary = trace_interval(
            trace,
            log_transform=signal_kind != "growth",
            confidence_level=confidence_level,
        )
        if not is_anchor:
            band = axis.fill_between(
                summary["time_from_event_h"],
                summary["lower"],
                summary["upper"],
                color=STATE_COLORS[str(state)],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            band.set_gid("replicate-interval")
        (line,) = axis.plot(
            summary["time_from_event_h"],
            summary["median"],
            color=STATE_COLORS[str(state)],
            linestyle="--" if is_anchor else "-",
            linewidth=1.2 if is_anchor else 1.8,
            marker=STATE_MARKERS[str(state)],
            markersize=3,
            markevery=0.10,
            markerfacecolor="white" if is_anchor else STATE_COLORS[str(state)],
            markeredgewidth=0.7,
            zorder=4,
        )
        line.set_gid("response-window-median")
    style_trajectory_axis(
        axis,
        title="",
        ylabel=ylabel,
        event_label=event_label,
        uncertainty=uncertainty,
        selected=selected,
    )
    axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
    axis.set_box_aspect(0.78)


def draw_handoff_axis(
    axis: plt.Axes,
    *,
    selected: pd.Series,
    display: dict[str, object],
    prefix: str,
) -> None:
    x = np.arange(len(STATE_ORDER))
    values = np.asarray([selected[f"{prefix}{state}"] for state in STATE_ORDER], dtype=float)
    bootstrap = np.asarray([selected[f"{prefix}{state}_bootstrap_sd"] for state in STATE_ORDER], dtype=float)
    event = np.asarray([selected[f"{prefix}{state}_event_half_range"] for state in STATE_ORDER], dtype=float)
    event_marks = axis.vlines(x, values - event, values + event, color="#9ca3af", linewidth=6, alpha=0.38, zorder=2)
    event_marks.set_gid("event-time-sensitivity")
    bootstrap_marks = axis.vlines(
        x,
        values - bootstrap,
        values + bootstrap,
        color=[STATE_COLORS[state] for state in STATE_ORDER],
        linewidth=1.8,
        zorder=3,
    )
    bootstrap_marks.set_gid("bootstrap-uncertainty")
    for index, state in enumerate(STATE_ORDER):
        points = axis.scatter(
            x[index],
            values[index],
            color=STATE_COLORS[state],
            marker=STATE_MARKERS[state],
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        points.set_gid(f"handoff-value-{state}")
    axis.axhline(0, color="#111827", linewidth=0.8, zorder=1)
    axis.set_xticks(x, condition_ticks(display, width=11), fontsize=7.2)
    if prefix == "r":
        axis.set_title("D1  Response handoff, r_i", loc="left", fontsize=10, fontweight="semibold")
        axis.set_ylabel(response_axis_label(display))
    else:
        axis.set_title("D2  pDual-10-relative fluorescence, b_i", loc="left", fontsize=10, fontweight="semibold")
        axis.set_ylabel(anchored_fluorescence_axis_label(display))
    style_data_axis(axis, grid_axis="y")


__all__ = ["draw_handoff_axis", "draw_trajectory_axis"]
