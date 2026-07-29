"""Cross-experiment evidence figure for one response-window Reader design."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .review_cross_experiment_contract import prepare_cross_experiment_context
from .review_cross_experiment_summaries import (
    draw_cross_experiment_summary,
    draw_cross_experiment_support,
)
from .review_cross_experiment_trajectories import (
    draw_cross_experiment_trajectories,
    experiment_line_style,
    experiment_marker,
)
from .review_replicates import reference_replicate_counts, response_replicate_rows
from .visual_labels import STATE_COLORS


def cross_experiment_state_figure(
    *,
    selected: pd.DataFrame,
    state: str,
    experiment_labels: Mapping[str, str],
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    events: pd.DataFrame,
    display: dict[str, object],
) -> plt.Figure:
    """Render separate experiment evidence for one exact design and condition."""

    selected, context = prepare_cross_experiment_context(
        selected=selected,
        state=state,
        experiment_labels=experiment_labels,
        events=events,
        display=display,
    )
    response_replicates: dict[str, pd.DataFrame] = {}
    reference_counts: dict[str, dict[str, int]] = {}
    for row in selected.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        experiment_id = str(row.experiment_id)
        response_replicates[experiment_id] = response_replicate_rows(
            selected=row_series,
            wells=wells,
            experiment_id=experiment_id,
            design_id=context.design_id,
            reduction_id=context.reduction_id,
        )
        reference_counts[experiment_id] = reference_replicate_counts(
            selected=row_series,
            wells=wells,
            experiment_id=experiment_id,
            reduction_id=context.reduction_id,
        )

    figure = plt.figure(figsize=(11.4, 8.4), constrained_layout=True)
    figure.set_gid(f"response-window-cross-experiment:{context.design_id}:{context.state}:{context.reduction_id}")
    grid = figure.add_gridspec(2, 3)
    trajectory_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    response_summary_axis = figure.add_subplot(grid[1, 0])
    fluorescence_summary_axis = figure.add_subplot(grid[1, 1])
    support_axis = figure.add_subplot(grid[1, 2])

    draw_cross_experiment_trajectories(
        trajectory_axes,
        selected=selected,
        context=context,
        traces=traces,
        display=display,
    )
    draw_cross_experiment_summary(
        response_summary_axis,
        selected=selected,
        prefix="r",
        context=context,
        replicate_rows=response_replicates,
        display=display,
    )
    draw_cross_experiment_summary(
        fluorescence_summary_axis,
        selected=selected,
        prefix="b",
        context=context,
        replicate_rows=response_replicates,
        display=display,
    )
    draw_cross_experiment_support(
        support_axis,
        selected=selected,
        context=context,
        reference_counts=reference_counts,
    )
    figure.legend(
        handles=_legend_handles(context),
        loc="outside lower center",
        ncol=min(len(context.experiment_order) + 1, 4),
        frameon=False,
        fontsize=7.2,
    )
    figure.suptitle(
        f"{context.design_id}: evidence across Reader experiments for {context.state_label} ({context.state})",
        fontsize=12,
        fontweight="semibold",
    )
    return figure


def _legend_handles(context) -> list[Line2D]:
    handles = [
        Line2D(
            [],
            [],
            color=STATE_COLORS[context.state],
            linestyle=experiment_line_style(index),
            marker=experiment_marker(index),
            markersize=4,
            linewidth=1.8,
            label=context.plot_experiment_labels[experiment_id],
        )
        for index, experiment_id in enumerate(context.experiment_order)
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="#64748b",
            marker="o",
            markerfacecolor="white",
            markersize=4,
            linewidth=1.2,
            label=f"{context.reference_id} anchor (magnitude only)",
        )
    )
    return handles


__all__ = ["cross_experiment_state_figure"]
