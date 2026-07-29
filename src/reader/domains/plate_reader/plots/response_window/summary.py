"""Selection and rendering for the canonical response-window summary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reader.plotting.style import use_style

COMPONENT_COLUMNS = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")
RESPONSE_COLUMNS = COMPONENT_COLUMNS[:4]
MAGNITUDE_COLUMNS = COMPONENT_COLUMNS[4:]
STATE_LABELS = tuple(column[1:] for column in RESPONSE_COLUMNS)


@dataclass(frozen=True)
class ResponseWindowSummaryMatrix:
    """Ordered labels and values consumed by the summary heatmap."""

    row_labels: tuple[str, ...]
    component_labels: tuple[str, ...]
    values: np.ndarray


def response_window_summary_matrix(
    designs: pd.DataFrame,
    *,
    primary_reduction_id: str,
) -> ResponseWindowSummaryMatrix:
    """Select one reduction's non-reference designs and validate plot values."""

    selected = designs.loc[
        designs["reduction_id"].astype(str).eq(primary_reduction_id) & ~designs["is_reference"].astype(bool)
    ].copy()
    if selected.empty:
        raise ValueError(f"response-window plot has no non-reference rows for reduction {primary_reduction_id!r}")

    selected = selected.sort_values(["experiment_id", "design_id"], kind="stable")
    row_labels = tuple(selected["experiment_id"].astype(str) + " :: " + selected["design_id"].astype(str))
    values = selected.loc[:, COMPONENT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("response-window summary requires finite component values")

    return ResponseWindowSummaryMatrix(
        row_labels=row_labels,
        component_labels=COMPONENT_COLUMNS,
        values=values,
    )


def render_response_window_summary(
    designs: pd.DataFrame,
    *,
    primary_reduction_id: str,
    title: str,
):
    """Render the selected response and anchored-magnitude components."""

    summary = response_window_summary_matrix(designs, primary_reduction_id=primary_reduction_id)
    with use_style(
        {
            "axes_grid": False,
            "figure_figsize": (11.0, max(4.5, 0.32 * len(summary.row_labels) + 2.0)),
        }
    ):
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.colors import TwoSlopeNorm  # noqa: PLC0415

        figure, axes = plt.subplots(1, 2, sharey=True, constrained_layout=True)
        response_values = summary.values[:, : len(RESPONSE_COLUMNS)]
        magnitude_values = summary.values[:, len(RESPONSE_COLUMNS) :]
        families = (
            (axes[0], "Response", "response", response_values),
            (axes[1], "Anchored magnitude", "anchored magnitude", magnitude_values),
        )
        for axis, panel_title, colorbar_label, values in families:
            limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
            image = axis.imshow(
                values,
                aspect="auto",
                cmap="coolwarm",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            )
            axis.set_xticks(range(len(STATE_LABELS)), labels=STATE_LABELS)
            axis.set_title(panel_title)
            axis.set_xlabel("state")
            figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.8)
        axes[0].set_yticks(range(len(summary.row_labels)), labels=summary.row_labels)
        axes[0].set_ylabel("source :: design")
        figure.suptitle(title)
    return figure


__all__ = [
    "COMPONENT_COLUMNS",
    "MAGNITUDE_COLUMNS",
    "RESPONSE_COLUMNS",
    "ResponseWindowSummaryMatrix",
    "STATE_LABELS",
    "render_response_window_summary",
    "response_window_summary_matrix",
]
