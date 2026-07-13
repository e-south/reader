"""Trace summaries and compact endpoint components for time-series review."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .plot_style import style_data_axis
from .sources import STATE_ORDER
from .visual_labels import (
    STATE_COLORS,
    STATE_MARKERS,
    anchored_fluorescence_axis_label,
    condition_ticks,
    response_axis_label,
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
) -> None:
    axis.axvspan(-uncertainty, uncertainty, color="#9ca3af", alpha=0.20, zorder=1)
    axis.axvspan(
        float(selected["window_start_event_h"]),
        float(selected["window_end_event_h"]),
        color="#f59e0b",
        alpha=0.11,
        zorder=1,
    )
    axis.axvline(0.0, color="#111827", linewidth=0.9, zorder=3)
    axis.set_title(title)
    axis.set_xlabel(f"Hours from {event_label.lower()}")
    axis.set_ylabel(ylabel)
    axis.set_box_aspect(0.9)
    style_data_axis(axis, grid_axis="both")


def draw_handoff_axis(axis: plt.Axes, *, selected: pd.Series, display: dict[str, object]) -> None:
    x = np.arange(len(STATE_ORDER))
    width = 0.34
    response = np.asarray([selected[f"r{state}"] for state in STATE_ORDER], dtype=float)
    fluorescence = np.asarray([selected[f"b{state}"] for state in STATE_ORDER], dtype=float)
    axis.bar(
        x - width / 2.0,
        response,
        width,
        yerr=_combined_error(selected, "r"),
        capsize=3,
        color="#2563eb",
        label=f"Response r_i: {response_axis_label(display)}",
        zorder=3,
    )
    axis.bar(
        x + width / 2.0,
        fluorescence,
        width,
        yerr=_combined_error(selected, "b"),
        capsize=3,
        color="#0f766e",
        label=f"Anchored fluorescence b_i: {anchored_fluorescence_axis_label(display)}",
        zorder=3,
    )
    axis.axhline(0.0, color="#111827", linewidth=0.9, zorder=2)
    axis.set_xticks(x, condition_ticks(display, width=14))
    axis.set_ylabel("Window-reduced value (log2 units)")
    axis.set_title("The response window reduces the trajectories to eight condition-specific values", pad=38)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=2,
        fontsize=7.5,
    )
    style_data_axis(axis, grid_axis="y")


def legend_handles(
    *,
    state_labels: dict[str, object],
    reference_id: str,
    include_anchor: bool,
    confidence_level: float,
    event_label: str,
) -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = [
        *(
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
        ),
        Patch(facecolor="#2563eb", alpha=0.14, label=f"Central {confidence_level:.0%} of design wells"),
    ]
    if include_anchor:
        handles.append(
            Line2D(
                [],
                [],
                color="#111827",
                linewidth=1.5,
                linestyle="--",
                label=f"{reference_id} median anchor",
            )
        )
    handles.extend(
        [
            Patch(facecolor="#9ca3af", alpha=0.20, label=f"{event_label} interval"),
            Patch(facecolor="#f59e0b", alpha=0.14, label="Response window"),
        ]
    )
    return handles


def _combined_error(selected: pd.Series, prefix: str) -> np.ndarray:
    bootstrap = np.asarray([selected[f"{prefix}{state}_bootstrap_sd"] for state in STATE_ORDER], dtype=float)
    event = np.asarray([selected[f"{prefix}{state}_event_half_range"] for state in STATE_ORDER], dtype=float)
    return np.hypot(bootstrap, event)


__all__ = ["draw_handoff_axis", "legend_handles", "signal_rows", "style_trajectory_axis", "trace_interval"]
