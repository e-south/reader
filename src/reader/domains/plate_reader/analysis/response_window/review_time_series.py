"""Time-series figure for event-relative response-window review."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .review_replicates import reference_replicate_counts, response_replicate_rows
from .review_time_series_components import (
    legend_handles,
    signal_rows,
    style_trajectory_axis,
    trace_interval,
)
from .review_time_series_handoff import (
    draw_reduced_value_axis,
    draw_window_support_axis,
)
from .visual_labels import STATE_COLORS, STATE_MARKERS


def time_series_figure(
    *,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
    selected: pd.Series,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    events: pd.DataFrame,
    display: dict[str, object],
) -> plt.Figure:
    event = events.loc[events["experiment_id"].astype(str).eq(experiment_id)]
    if len(event) != 1:
        raise ValueError(f"experiment {experiment_id!r} must have exactly one event record.")
    event_row = event.iloc[0]
    reference_id = str(selected["reference_design_id"])
    experiment_traces = traces.loc[traces["experiment_id"].astype(str).eq(experiment_id)].copy()
    replicate_rows = response_replicate_rows(
        selected=selected,
        wells=wells,
        experiment_id=experiment_id,
        design_id=design_id,
        reduction_id=reduction_id,
    )
    reference_counts = reference_replicate_counts(
        selected=selected,
        wells=wells,
        experiment_id=experiment_id,
        reduction_id=reduction_id,
    )
    channels = display["channels"]
    state_labels = display["state_labels"]
    if not isinstance(channels, dict) or not isinstance(state_labels, dict):
        raise ValueError("validated response-window display is malformed.")
    confidence_level = float(selected["confidence_level"])
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("response-window confidence level must lie strictly between zero and one.")

    figure = plt.figure(figsize=(11.4, 8.4), constrained_layout=True)
    figure.set_gid(f"response-window:{reduction_id}")
    grid = figure.add_gridspec(2, 3)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    response_axis = figure.add_subplot(grid[1, 0])
    fluorescence_axis = figure.add_subplot(grid[1, 1])
    support_axis = figure.add_subplot(grid[1, 2])
    response_ratio = _spaced(channels["response_ratio"])
    magnitude_ratio = _spaced(channels["magnitude_ratio"])
    specs = (
        ("growth", "A  Growth by condition", str(channels["growth"]), False),
        ("response", f"B  {response_ratio} response", f"log2({response_ratio})", True),
        (
            "magnitude",
            f"C  {magnitude_ratio} with {reference_id} anchor",
            f"log2({magnitude_ratio})",
            True,
        ),
    )
    uncertainty = float(event_row["event_time_uncertainty_h"])
    for index, (axis, (signal_kind, title, ylabel, log_transform)) in enumerate(zip(axes, specs, strict=True)):
        signal = signal_rows(
            experiment_traces,
            signal_kind=signal_kind,
            design_id=design_id,
            reference_id=reference_id,
        )
        for (source_design, state), trace in signal.groupby(["design_id", "state"], sort=True):
            is_anchor = str(source_design) == reference_id and design_id != reference_id
            summary = trace_interval(trace, log_transform=log_transform, confidence_level=confidence_level)
            band = axis.fill_between(
                summary["time_from_event_h"],
                summary["lower"],
                summary["upper"],
                color=STATE_COLORS[str(state)],
                alpha=0.06 if is_anchor else 0.14,
                linewidth=0.0,
                zorder=2,
            )
            band.set_gid("anchor-replicate-interval" if is_anchor else "replicate-interval")
            (line,) = axis.plot(
                summary["time_from_event_h"],
                summary["median"],
                color=STATE_COLORS[str(state)],
                linewidth=1.8 if not is_anchor else 1.2,
                linestyle="--" if is_anchor else "-",
                marker=STATE_MARKERS[str(state)],
                markersize=3.2,
                markevery=0.10,
                markerfacecolor="white" if is_anchor else STATE_COLORS[str(state)],
                markeredgewidth=0.7,
                zorder=4,
            )
            line.set_gid("response-window-median")
        style_trajectory_axis(
            axis,
            title=title,
            ylabel=ylabel,
            event_label=str(display["event_label"]),
            uncertainty=uncertainty,
            selected=selected,
            annotate_spans=index == 0,
        )
    draw_reduced_value_axis(
        response_axis,
        selected=selected,
        display=display,
        prefix="r",
        replicate_rows=replicate_rows,
    )
    draw_reduced_value_axis(
        fluorescence_axis,
        selected=selected,
        display=display,
        prefix="b",
        replicate_rows=replicate_rows,
    )
    draw_window_support_axis(
        support_axis,
        selected=selected,
        display=display,
        event_time_uncertainty_h=uncertainty,
        reference_counts=reference_counts,
    )
    figure.legend(
        handles=legend_handles(
            state_labels=state_labels,
        ),
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=7.2,
    )
    figure.suptitle(
        "The selected post-stress interval connects observed trajectories to the eight-value handoff",
        fontsize=12,
        fontweight="semibold",
    )
    return figure


def _spaced(value: object) -> str:
    return str(value).replace("/", " / ")


__all__ = ["time_series_figure"]
