"""Axes-level components for the promoter-evidence publication figure."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .censor_display import annotate_bound_glyph
from .plot_style import AXIS_LABEL_SIZE, PANEL_TITLE_SIZE, TICK_LABEL_SIZE, style_data_axis
from .review_replicates import draw_horizontal_replicate_summary
from .review_time_series_components import signal_rows, style_trajectory_axis, trace_interval
from .sources import STATE_ORDER
from .visual_labels import STATE_COLORS, STATE_MARKERS

_SUBSCRIPT_DIGITS = str.maketrans("01", "₀₁")


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
    annotate_spans: bool = False,
) -> None:
    rows = signal_rows(traces, signal_kind=signal_kind, design_id=design_id, reference_id=reference_id)
    for (source_design, state), trace in rows.groupby(["design_id", "state"], sort=True):
        is_anchor = str(source_design) == reference_id and design_id != reference_id
        summary = trace_interval(
            trace,
            log_transform=signal_kind != "growth",
            confidence_level=confidence_level,
        )
        band = axis.fill_between(
            summary["time_from_event_h"],
            summary["lower"],
            summary["upper"],
            color=STATE_COLORS[str(state)],
            alpha=0.06 if is_anchor else 0.14,
            linewidth=0,
            zorder=2,
        )
        band.set_gid("anchor-replicate-interval" if is_anchor else "replicate-interval")
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
        annotate_spans=annotate_spans,
        show_event_uncertainty=False,
    )
    axis.set_title(
        textwrap.fill(title, width=28),
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="normal",
        linespacing=1.15,
    )
    axis.set_box_aspect(1.0)
    _style_promoter_axis_text(axis)


def draw_eight_value_handoff_axis(
    axis: plt.Axes,
    *,
    selected: pd.Series,
    replicate_rows: pd.DataFrame,
) -> None:
    axis.set_gid("promoter-evidence-response-window-phenotype")
    components = tuple((prefix, state) for prefix in ("r", "b") for state in STATE_ORDER)
    y = np.asarray((8.0, 7.0, 6.0, 5.0, 3.0, 2.0, 1.0, 0.0))
    values = np.asarray([selected[f"{prefix}{state}"] for prefix, state in components], dtype=float)
    ci_low = np.asarray(
        [selected[f"{prefix}{state}_ci_low"] for prefix, state in components],
        dtype=float,
    )
    ci_high = np.asarray(
        [selected[f"{prefix}{state}_ci_high"] for prefix, state in components],
        dtype=float,
    )
    event = np.asarray(
        [selected[f"{prefix}{state}_event_half_range"] for prefix, state in components],
        dtype=float,
    )
    colors = [STATE_COLORS[state] for _prefix, state in components]

    event_marks = axis.hlines(
        y,
        values - event,
        values + event,
        color="#9ca3af",
        linewidth=6,
        alpha=0.38,
        zorder=2,
    )
    event_marks.set_gid("event-time-sensitivity")
    bootstrap_marks = axis.hlines(
        y,
        ci_low,
        ci_high,
        color=colors,
        linewidth=1.8,
        zorder=3,
    )
    bootstrap_marks.set_gid("bootstrap-uncertainty")
    for index, (prefix, state) in enumerate(components):
        state_replicates = replicate_rows.loc[
            replicate_rows["component"].astype(str).eq(f"{prefix}{state}"), "value"
        ].to_numpy(dtype=float)
        draw_horizontal_replicate_summary(
            axis,
            y=float(y[index]),
            values=state_replicates,
            summary=float(values[index]),
            state=state,
            component=f"{prefix}{state}",
        )
        annotate_bound_glyph(
            axis,
            row=selected,
            component=f"{prefix}{state}",
            xy=(values[index], y[index]),
            xytext=(0, 6),
            ha="center",
            va="bottom",
        )

    axis.axvline(0, color="#111827", linewidth=0.8, zorder=1)
    axis.axhline(4.0, color="#d1d5db", linewidth=0.8, zorder=1)
    axis.set_yticks(
        y,
        [f"{prefix}{state.translate(_SUBSCRIPT_DIGITS)}" for prefix, state in components],
        fontsize=9,
    )
    axis.set_ylim(-0.75, 8.75)
    _set_phenotype_x_limits(
        axis,
        values=values,
        ci_low=ci_low,
        ci_high=ci_high,
        event=event,
        replicate_rows=replicate_rows,
    )
    axis.set_xlabel("Reduced value (log₂ units)")
    axis.set_title(
        "Response-window\nphenotype",
        loc="center",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="normal",
    )
    axis.set_box_aspect(1.0)
    style_data_axis(axis, grid_axis="x")
    _style_promoter_axis_text(axis)


def _set_phenotype_x_limits(
    axis: plt.Axes,
    *,
    values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    event: np.ndarray,
    replicate_rows: pd.DataFrame,
) -> None:
    replicate_values = pd.to_numeric(replicate_rows.get("value"), errors="coerce").to_numpy(dtype=float)
    finite_replicates = replicate_values[np.isfinite(replicate_values)]
    lower_candidates = [0.0, *ci_low.tolist(), *(values - event).tolist(), *finite_replicates.tolist()]
    upper_candidates = [0.0, *ci_high.tolist(), *(values + event).tolist(), *finite_replicates.tolist()]
    data_low = float(np.min(lower_candidates))
    data_high = float(np.max(upper_candidates))
    data_span = max(data_high - data_low, 1.0)
    left = data_low - 0.08 * data_span
    data_right = data_high + 0.08 * data_span
    axis.set_xlim(left, data_right)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))


def _style_promoter_axis_text(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    axis.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)


__all__ = [
    "draw_eight_value_handoff_axis",
    "draw_trajectory_axis",
]
