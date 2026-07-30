"""Selection and rendering for the canonical response-window summary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reader.plotting.style import use_style

from .schema import COMPONENT_COLUMNS, MAGNITUDE_COLUMNS, RESPONSE_COLUMNS, STATE_ORDER


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
    experiment_ids: list[str] | tuple[str, ...] | None = None,
    design_ids: list[str] | tuple[str, ...] | None = None,
    maximum_rows: int = 64,
) -> ResponseWindowSummaryMatrix:
    """Select one reduction's non-reference designs and validate plot values."""

    experiment_selection = _identity_filter(experiment_ids, label="experiment_ids")
    design_selection = _identity_filter(design_ids, label="design_ids")
    if isinstance(maximum_rows, bool) or not isinstance(maximum_rows, int) or maximum_rows < 1:
        raise ValueError("response-window summary maximum_rows must be a positive integer")
    selected = designs.loc[
        designs["reduction_id"].astype(str).eq(primary_reduction_id) & ~designs["is_reference"].astype(bool)
    ].copy()
    if experiment_selection is not None:
        selected = selected.loc[selected["experiment_id"].astype(str).isin(experiment_selection)]
    if design_selection is not None:
        selected = selected.loc[selected["design_id"].astype(str).isin(design_selection)]
    if selected.empty:
        raise ValueError(
            f"response-window plot has no non-reference rows for reduction {primary_reduction_id!r} "
            "under the configured experiment_ids/design_ids filters"
        )
    if len(selected) > maximum_rows:
        raise ValueError(
            f"response-window summary selected {len(selected)} rows, exceeding maximum_rows={maximum_rows}; "
            "configure experiment_ids/design_ids or raise maximum_rows explicitly"
        )

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
    experiment_ids: list[str] | tuple[str, ...] | None = None,
    design_ids: list[str] | tuple[str, ...] | None = None,
    maximum_rows: int = 64,
):
    """Render the selected response and anchored-magnitude components."""

    summary = response_window_summary_matrix(
        designs,
        primary_reduction_id=primary_reduction_id,
        experiment_ids=experiment_ids,
        design_ids=design_ids,
        maximum_rows=maximum_rows,
    )
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
            axis.set_xticks(range(len(STATE_ORDER)), labels=STATE_ORDER)
            axis.set_title(panel_title)
            axis.set_xlabel("state")
            figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.8)
        axes[0].set_yticks(range(len(summary.row_labels)), labels=summary.row_labels)
        axes[0].set_ylabel("source :: design")
        figure.suptitle(title)
    return figure


def _identity_filter(values: list[str] | tuple[str, ...] | None, *, label: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"response-window summary {label} must contain non-empty identities")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"response-window summary {label} must not contain duplicates")
    return normalized


__all__ = [
    "COMPONENT_COLUMNS",
    "MAGNITUDE_COLUMNS",
    "RESPONSE_COLUMNS",
    "ResponseWindowSummaryMatrix",
    "STATE_ORDER",
    "render_response_window_summary",
    "response_window_summary_matrix",
]
