"""Endpoint and evidence-boundary panels for cross-experiment review."""

from __future__ import annotations

from collections.abc import Mapping
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .plot_style import style_data_axis
from .review_cross_experiment_contract import CrossExperimentContext
from .review_replicates import draw_horizontal_replicate_summary
from .visual_labels import (
    STATE_COLORS,
    magnitude_ratio_label,
    response_ratio_label,
    response_summary_label,
)


def draw_cross_experiment_summary(
    axis: plt.Axes,
    *,
    selected: pd.DataFrame,
    prefix: str,
    context: CrossExperimentContext,
    replicate_rows: Mapping[str, pd.DataFrame],
    display: Mapping[str, object],
) -> None:
    selected_by_experiment = selected.set_index("experiment_id")
    for y, experiment_id in enumerate(context.experiment_order):
        row = selected_by_experiment.loc[experiment_id]
        value = float(row[f"{prefix}{context.state}"])
        event = float(row[f"{prefix}{context.state}_event_half_range"])
        event_mark = axis.hlines(
            y,
            value - event,
            value + event,
            color="#9ca3af",
            linewidth=6,
            alpha=0.38,
            zorder=2,
        )
        event_mark.set_gid("event-time-sensitivity")
        bootstrap_mark = axis.hlines(
            y,
            float(row[f"{prefix}{context.state}_ci_low"]),
            float(row[f"{prefix}{context.state}_ci_high"]),
            color=STATE_COLORS[context.state],
            linewidth=1.8,
            zorder=3,
        )
        bootstrap_mark.set_gid("bootstrap-uncertainty")
        values = np.asarray([], dtype=float)
        if prefix == "r":
            values = (
                replicate_rows[experiment_id]
                .loc[
                    replicate_rows[experiment_id]["component"].astype(str).eq(f"r{context.state}"),
                    "value",
                ]
                .to_numpy(dtype=float)
            )
        draw_horizontal_replicate_summary(
            axis,
            y=float(y),
            values=values,
            summary=value,
            state=context.state,
            component=f"{prefix}{context.state}",
        )
    axis.axvline(0.0, color="#111827", linewidth=0.9, zorder=2)
    axis.set_yticks(
        np.arange(len(context.experiment_order)),
        [context.plot_experiment_labels[experiment_id] for experiment_id in context.experiment_order],
    )
    axis.invert_yaxis()
    if prefix == "r":
        axis.set_title("D  Response, rᵢ", loc="left", fontsize=10, fontweight="semibold")
        axis.set_xlabel(f"log₂({response_ratio_label(display)})\nrᵢ")
    elif prefix == "b":
        axis.set_title("E  Anchored fluorescence, bᵢ", loc="left", fontsize=10, fontweight="semibold")
        axis.set_xlabel(f"{context.reference_id}-relative log₂({magnitude_ratio_label(display)})\nbᵢ")
        axis.tick_params(axis="y", labelleft=False)
    else:
        raise ValueError(f"unknown cross-experiment component: {prefix!r}.")
    axis.set_box_aspect(1.0)
    style_data_axis(axis, grid_axis="x")


def draw_cross_experiment_support(
    axis: plt.Axes,
    *,
    selected: pd.DataFrame,
    context: CrossExperimentContext,
    reference_counts: Mapping[str, Mapping[str, int]],
) -> None:
    first = selected.iloc[0]
    selected_by_experiment = selected.set_index("experiment_id")
    interval_percent = 100.0 * context.confidence_level
    interval_label = f"{interval_percent:g}%"
    raw_lines = [
        f"Reader design  {context.design_id}",
        f"Condition  {context.state_label} ({context.state})",
        f"Response summary  {response_summary_label(first)}",
        f"Evidence  {len(context.experiment_order)} Reader experiments; shown separately",
        "Wells (design/reference for bᵢ)",
        *(
            f"  {context.plot_experiment_labels[experiment_id]}  "
            f"{int(selected_by_experiment.loc[experiment_id, f'n{context.state}'])}/"
            f"{reference_counts[experiment_id][context.state]} · "
            f"event ±{context.event_uncertainty[experiment_id]:.2g} h"
            for experiment_id in context.experiment_order
        ),
        "Marks  hollow circles: observed rᵢ wells; vertical line: experiment summary",
        f"Intervals  thin color: central {interval_label} bootstrap; thick gray: event-time sensitivity",
        "Trajectories are pointwise replicate summaries; endpoints reduce each well, then aggregate.",
        "No cross-experiment aggregation or comparability decision is made.",
    ]
    lines = [
        line
        for raw_line in raw_lines
        for line in wrap(
            raw_line,
            width=46,
            subsequent_indent="  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
    ]
    axis.set_box_aspect(1.0)
    axis.set_axis_off()
    axis.set_title("F  Evidence boundary", loc="left", fontsize=10, fontweight="semibold")
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
        fontsize=6.4,
        linespacing=1.2,
        color="#1f2937",
    )


__all__ = ["draw_cross_experiment_summary", "draw_cross_experiment_support"]
