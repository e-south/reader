"""Exact well-level evidence for response-window review figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .sources import STATE_ORDER
from .visual_labels import STATE_COLORS


def draw_vertical_replicate_summary(
    axis: Axes,
    *,
    x: float,
    values: np.ndarray,
    summary: float,
    state: str,
    component: str,
) -> None:
    """Draw observed values around an x position and a horizontal summary line."""

    if len(values):
        offsets = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else np.asarray([0.0])
        points = axis.scatter(
            x + offsets,
            values,
            s=22,
            facecolors="white",
            edgecolors="#94a3b8",
            linewidths=0.8,
            zorder=4,
        )
        points.set_gid(f"replicate-values-{component}")
    summary_line = axis.hlines(
        summary,
        x - 0.17,
        x + 0.17,
        color=STATE_COLORS[state],
        linewidth=2.5,
        zorder=5,
    )
    summary_line.set_gid(f"handoff-summary-{component}")


def draw_horizontal_replicate_summary(
    axis: Axes,
    *,
    y: float,
    values: np.ndarray,
    summary: float,
    state: str,
    component: str,
) -> None:
    """Draw observed values around a y position and a vertical summary line."""

    if len(values):
        offsets = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else np.asarray([0.0])
        points = axis.scatter(
            values,
            y + offsets,
            s=18,
            facecolors="white",
            edgecolors="#94a3b8",
            linewidths=0.8,
            zorder=4,
        )
        points.set_gid(f"replicate-values-{component}")
    summary_line = axis.vlines(
        summary,
        y - 0.16,
        y + 0.16,
        color=STATE_COLORS[state],
        linewidth=2.4,
        zorder=5,
    )
    summary_line.set_gid(f"handoff-summary-{component}")


def response_replicate_rows(
    *,
    selected: pd.Series,
    wells: pd.DataFrame,
    experiment_id: str,
    design_id: str,
    reduction_id: str,
) -> pd.DataFrame:
    """Return observed design-well values on the published response scale.

    The anchored fluorescence summary compares independent design and
    reference aggregates, so it deliberately has no per-well ``b`` points.
    """

    required = {
        "experiment_id",
        "design_id",
        "reduction_id",
        "state",
        "position",
        "response_well",
    }
    missing = sorted(required - set(wells.columns))
    if missing:
        raise ValueError(f"response-window replicate evidence is missing columns: {missing}.")

    replicate_stat = str(selected["replicate_stat"])
    if replicate_stat == "median":
        aggregate = np.median
    elif replicate_stat == "mean":
        aggregate = np.mean
    else:
        raise ValueError(f"unknown response-window replicate statistic: {replicate_stat!r}.")

    selection = wells.loc[
        wells["experiment_id"].astype(str).eq(experiment_id)
        & wells["reduction_id"].astype(str).eq(reduction_id)
        & wells["design_id"].astype(str).eq(design_id)
    ].copy()
    records: list[dict[str, object]] = []
    for state in STATE_ORDER:
        design_state = selection.loc[selection["state"].astype(str).eq(state)].sort_values("position", kind="mergesort")
        expected = int(selected[f"n{state}"])
        if len(design_state) != expected:
            raise ValueError(
                f"response-window replicate count disagrees for {design_id}:{state}: "
                f"observed={len(design_state)}, published={expected}."
            )
        for row in design_state.itertuples(index=False):
            records.append(
                {
                    "component": f"r{state}",
                    "state": state,
                    "position": str(row.position),
                    "value": float(row.response_well),
                }
            )

    result = pd.DataFrame.from_records(records)
    for state in STATE_ORDER:
        values = result.loc[result["component"].eq(f"r{state}"), "value"].to_numpy(dtype=float)
        observed = float(aggregate(values))
        published = float(selected[f"r{state}"])
        if not np.isclose(observed, published, rtol=1e-10, atol=1e-10):
            raise ValueError(
                f"response-window replicate summary disagrees for {design_id}:r{state}: "
                f"observed={observed}, published={published}."
            )
    return result


def reference_replicate_counts(
    *,
    selected: pd.Series,
    wells: pd.DataFrame,
    experiment_id: str,
    reduction_id: str,
) -> dict[str, int]:
    """Count independent reference wells contributing to each ``b`` summary."""

    reference_id = str(selected["reference_design_id"])
    selection = wells.loc[
        wells["experiment_id"].astype(str).eq(experiment_id)
        & wells["reduction_id"].astype(str).eq(reduction_id)
        & wells["design_id"].astype(str).eq(reference_id)
    ]
    counts = {
        state: int(selection.loc[selection["state"].astype(str).eq(state), "position"].nunique())
        for state in STATE_ORDER
    }
    missing = [state for state, count in counts.items() if count < 1]
    if missing:
        raise ValueError(f"response-window reference {reference_id!r} lacks well support for states: {missing}.")
    return counts


__all__ = [
    "draw_horizontal_replicate_summary",
    "draw_vertical_replicate_summary",
    "reference_replicate_counts",
    "response_replicate_rows",
]
