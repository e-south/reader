"""Trajectory panels for cross-experiment response-window review."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd

from .plot_style import style_data_axis
from .review_cross_experiment_contract import CrossExperimentContext
from .review_time_series_components import signal_rows, trace_interval
from .visual_labels import (
    STATE_COLORS,
    channels,
    magnitude_ratio_label,
    response_axis_label,
    response_ratio_label,
)

_LINE_STYLES = ("-", "--", "-.", ":")
_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<")


def draw_cross_experiment_trajectories(
    axes: list[plt.Axes],
    *,
    selected: pd.DataFrame,
    context: CrossExperimentContext,
    traces: pd.DataFrame,
    display: Mapping[str, object],
) -> None:
    channel_labels = channels(display)
    specs = (
        ("growth", "A  Growth trajectory", str(channel_labels["growth"]), False),
        ("response", f"B  {response_ratio_label(display)} response", response_axis_label(display), True),
        (
            "magnitude",
            "C  Fluorescence + anchor",
            f"log₂({magnitude_ratio_label(display)})",
            True,
        ),
    )
    start = float(selected["window_start_event_h"].iloc[0])
    end = float(selected["window_end_event_h"].iloc[0])
    for axis, (signal_kind, title, ylabel, log_transform) in zip(axes, specs, strict=True):
        for experiment_index, experiment_id in enumerate(context.experiment_order):
            experiment_traces = traces.loc[
                traces["experiment_id"].astype(str).eq(experiment_id) & traces["state"].astype(str).eq(context.state)
            ].copy()
            signal = signal_rows(
                experiment_traces,
                signal_kind=signal_kind,
                design_id=context.design_id,
                reference_id=context.reference_id,
            )
            expected_sources = (
                {context.design_id, context.reference_id} if signal_kind == "magnitude" else {context.design_id}
            )
            observed_sources = set(signal["design_id"].astype(str))
            if observed_sources != expected_sources:
                raise ValueError(
                    f"experiment {experiment_id!r} lacks exact {signal_kind} trace sources: "
                    f"expected={sorted(expected_sources)}, observed={sorted(observed_sources)}."
                )
            for source_design, trace in signal.groupby("design_id", sort=True):
                _draw_trace(
                    axis,
                    trace=trace,
                    is_anchor=str(source_design) == context.reference_id,
                    state=context.state,
                    experiment_index=experiment_index,
                    log_transform=log_transform,
                    confidence_level=context.confidence_level,
                )
            span = axis.axvspan(
                -context.event_uncertainty[experiment_id],
                context.event_uncertainty[experiment_id],
                color="#9ca3af",
                alpha=0.08,
                zorder=1,
            )
            span.set_gid("experiment-event-interval")
        window = axis.axvspan(start, end, color="#f59e0b", alpha=0.11, zorder=1)
        window.set_gid("selected-response-window")
        axis.axvline(0.0, color="#111827", linewidth=0.9, zorder=3)
        axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
        axis.set_xlabel(f"Hours from {str(display['event_label']).lower()}")
        axis.set_ylabel(ylabel)
        axis.set_box_aspect(1.0)
        style_data_axis(axis, grid_axis="both")
    axes[0].text(
        0.02,
        0.97,
        "Gray: experiment event intervals\nAmber: selected window",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#475569",
    )


def experiment_line_style(index: int) -> str:
    return _LINE_STYLES[index % len(_LINE_STYLES)]


def experiment_marker(index: int) -> str:
    return _MARKERS[index % len(_MARKERS)]


def _draw_trace(
    axis: plt.Axes,
    *,
    trace: pd.DataFrame,
    is_anchor: bool,
    state: str,
    experiment_index: int,
    log_transform: bool,
    confidence_level: float,
) -> None:
    summary = trace_interval(trace, log_transform=log_transform, confidence_level=confidence_level)
    color = "#64748b" if is_anchor else STATE_COLORS[state]
    band = axis.fill_between(
        summary["time_from_event_h"],
        summary["lower"],
        summary["upper"],
        color=color,
        alpha=0.05 if is_anchor else 0.11,
        linewidth=0.0,
        zorder=2,
    )
    band.set_gid("cross-experiment-anchor-interval" if is_anchor else "cross-experiment-interval")
    (line,) = axis.plot(
        summary["time_from_event_h"],
        summary["median"],
        color=color,
        linewidth=1.1 if is_anchor else 1.8,
        linestyle=experiment_line_style(experiment_index),
        marker=experiment_marker(experiment_index),
        markersize=3.0,
        markevery=0.14,
        markerfacecolor="white" if is_anchor else color,
        markeredgewidth=0.7,
        zorder=4,
    )
    line.set_gid("cross-experiment-median")


__all__ = [
    "draw_cross_experiment_trajectories",
    "experiment_line_style",
    "experiment_marker",
]
