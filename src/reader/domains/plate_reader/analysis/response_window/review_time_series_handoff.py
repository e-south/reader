"""Square handoff and support panels for response-window review."""

from __future__ import annotations

from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .censor_display import annotate_bound_glyph, censor_qc_line
from .plot_style import style_data_axis
from .sources import STATE_ORDER
from .visual_labels import (
    STATE_COLORS,
    channels,
    magnitude_ratio_label,
    response_ratio_label,
    response_summary_label,
)


def draw_reduced_value_axis(
    axis: plt.Axes,
    *,
    selected: pd.Series,
    display: dict[str, object],
    prefix: str,
    replicate_rows: pd.DataFrame,
) -> None:
    x = np.arange(len(STATE_ORDER))
    values = np.asarray([selected[f"{prefix}{state}"] for state in STATE_ORDER], dtype=float)
    ci_low = np.asarray([selected[f"{prefix}{state}_ci_low"] for state in STATE_ORDER], dtype=float)
    ci_high = np.asarray([selected[f"{prefix}{state}_ci_high"] for state in STATE_ORDER], dtype=float)
    event = np.asarray([selected[f"{prefix}{state}_event_half_range"] for state in STATE_ORDER], dtype=float)
    event_marks = axis.vlines(
        x,
        values - event,
        values + event,
        color="#9ca3af",
        linewidth=6,
        alpha=0.38,
        zorder=2,
    )
    event_marks.set_gid("event-time-sensitivity")
    bootstrap_marks = axis.vlines(
        x,
        ci_low,
        ci_high,
        color=[STATE_COLORS[state] for state in STATE_ORDER],
        linewidth=1.8,
        zorder=3,
    )
    bootstrap_marks.set_gid("bootstrap-uncertainty")
    for index, state in enumerate(STATE_ORDER):
        state_replicates = replicate_rows.loc[
            replicate_rows["component"].astype(str).eq(f"{prefix}{state}"), "value"
        ].to_numpy(dtype=float)
        if len(state_replicates):
            offsets = _replicate_offsets(len(state_replicates))
            points = axis.scatter(
                x[index] + offsets,
                state_replicates,
                s=18,
                facecolors="white",
                edgecolors="#94a3b8",
                marker="o",
                linewidths=0.8,
                zorder=4,
            )
            points.set_gid(f"replicate-values-{prefix}{state}")
        summary = axis.hlines(
            values[index],
            x[index] - 0.16,
            x[index] + 0.16,
            color=STATE_COLORS[state],
            linewidth=2.4,
            zorder=5,
        )
        summary.set_gid(f"handoff-summary-{prefix}{state}")
        annotate_bound_glyph(
            axis,
            row=selected,
            component=f"{prefix}{state}",
            xy=(x[index], values[index]),
            xytext=(7, 0),
            ha="left",
            va="center",
        )
    axis.axhline(0.0, color="#111827", linewidth=0.9, zorder=2)
    axis.set_xticks(x, STATE_ORDER)
    axis.set_xlabel("Condition")
    if prefix == "r":
        axis.set_title("D  Response handoff, rᵢ", loc="left", fontsize=10, fontweight="semibold")
        axis.set_ylabel(f"log₂({response_ratio_label(display)})")
    elif prefix == "b":
        reference_id = channels(display)["reference_design_id"]
        axis.set_title(
            f"E  {reference_id}-relative fluorescence, bᵢ",
            loc="left",
            fontsize=10,
            fontweight="semibold",
        )
        axis.set_ylabel(f"{reference_id}-relative\nlog₂({magnitude_ratio_label(display)})")
    else:
        raise ValueError(f"unknown response-window handoff component: {prefix!r}.")
    axis.set_box_aspect(1.0)
    style_data_axis(axis, grid_axis="y")


def draw_window_support_axis(
    axis: plt.Axes,
    *,
    selected: pd.Series,
    display: dict[str, object],
    event_time_uncertainty_h: float,
    reference_counts: dict[str, int],
) -> None:
    start = float(selected["window_start_event_h"])
    end = float(selected["window_end_event_h"])
    confidence_level = float(selected["confidence_level"])
    reference_id = channels(display)["reference_design_id"]
    raw_lines = (
        f"Selected window  {start:g}–{end:g} h after {str(display['event_label']).lower()}",
        f"Reduction  {response_summary_label(selected)}",
        f"Event timing  ±{event_time_uncertainty_h:.2g} h",
        "Wells (design/reference for bᵢ)  "
        + " · ".join(f"{state} {int(selected[f'n{state}'])}/{reference_counts[state]}" for state in STATE_ORDER),
        "Trace coverage  "
        f"≥{int(selected['min_observed_points_per_trace'])} points · "
        f"max gap {float(selected['max_interior_gap_h']):.2g} h",
        f"Trajectories  median · central {confidence_level:.0%} well interval",
        f"Trajectory style  solid/filled: selected design · dashed/hollow: {reference_id} anchor",
        f"Plot key  hollow circles: observed rᵢ wells · colored line: {str(selected['replicate_stat']).lower()}",
        f"Intervals  thin color: central {confidence_level:.0%} bootstrap · thick gray: event-time sensitivity",
        censor_qc_line(selected),
        "bᵢ uses independent design/reference aggregates; no paired replicate points",
    )
    lines = [
        line
        for raw_line in raw_lines
        for line in wrap(
            raw_line,
            width=52,
            subsequent_indent="  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
    ]
    axis.set_box_aspect(1.0)
    axis.set_axis_off()
    axis.set_title("F  Window and support", loc="left", fontsize=10, fontweight="semibold")
    axis.add_patch(
        Rectangle(
            (0.02, 0.05),
            0.96,
            0.88,
            transform=axis.transAxes,
            fill=False,
            edgecolor="#cbd5e1",
            linewidth=0.9,
        )
    )
    axis.text(
        0.07,
        0.86,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        linespacing=1.28,
        color="#1f2937",
    )


def _replicate_offsets(count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("response-window handoff requires observed replicate wells.")
    if count == 1:
        return np.asarray([0.0])
    return np.linspace(-0.11, 0.11, count)


__all__ = ["draw_reduced_value_axis", "draw_window_support_axis"]
