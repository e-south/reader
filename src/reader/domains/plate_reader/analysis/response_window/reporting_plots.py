"""Static publication figures for response-window review bundles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plot_style import save_publication_figure, style_data_axis
from .sources import STATE_ORDER
from .visual_labels import (
    anchored_fluorescence_axis_label,
    channels,
    condition_ticks,
    reduction_label,
    response_axis_label,
    response_ratio_label,
)


def write_event_plot(events: pd.DataFrame, *, display: dict[str, object], out_dir: Path) -> dict[str, str]:
    event_label = str(display["event_label"])
    title = f"Every experiment has a bounded {event_label.lower()} interval"
    ordered = events.sort_values("experiment_id", kind="mergesort").reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    y = np.arange(len(ordered))
    lower = ordered["event_interval_start_assay_h"].to_numpy(dtype=float)
    upper = ordered["event_interval_end_assay_h"].to_numpy(dtype=float)
    estimate = ordered["event_time_estimate_assay_h"].to_numpy(dtype=float)
    axis.hlines(y, lower, upper, color="#64748b", linewidth=5, zorder=3)
    axis.scatter(estimate, y, color="#111827", marker="|", s=140, linewidths=2, zorder=4)
    axis.set_yticks(y, [_short_experiment(value) for value in ordered["experiment_id"]])
    axis.set_xlabel("Acquisition time (h)")
    axis.set_title(title)
    axis.legend(
        handles=[
            Line2D([], [], color="#64748b", linewidth=5, label="Last pre-event to first post-event"),
            Line2D([], [], color="#111827", marker="|", linestyle="none", markersize=12, label="Event estimate"),
        ],
        frameon=False,
        loc="upper right",
    )
    style_data_axis(axis, grid_axis="x")
    path = out_dir / "plots" / "event_intervals.png"
    save_publication_figure(figure, path)
    plt.close(figure)
    return plot_manifest_row(
        plot_id="event_intervals",
        tier="assay_contract",
        title=title,
        premise="Event-relative summaries require an explicit intervention interval for every experiment.",
        decision_value="Shows whether event declarations are chronological, bounded, and comparable across sources.",
        rationale="The window origin must be measured provenance rather than inferred from worksheet order.",
        alt_text=f"Horizontal intervals for {len(ordered)} experiments show the last pre-event and first post-event acquisition times, with the declared {event_label.lower()} estimate marked inside each interval.",
        non_claim_boundary="The interval locates the intervention; it does not establish when a biological response begins.",
        data_table="tables/events.parquet",
        path="plots/event_intervals.png",
    )


def write_handoff_plot(
    primary: pd.DataFrame,
    *,
    display: dict[str, object],
    out_dir: Path,
) -> dict[str, str]:
    ordered = primary.sort_values(["experiment_id", "design_id"], kind="mergesort")
    response_ratio = response_ratio_label(display)
    reference_id = channels(display)["reference_design_id"]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, max(7.0, len(ordered) * 0.18)),
        sharey=True,
        constrained_layout=True,
    )
    row_labels = [
        f"{str(experiment)[4:8]} | {design}"
        for experiment, design in zip(ordered["experiment_id"], ordered["design_id"], strict=True)
    ]
    _heatmap_panel(
        figure,
        axes[0],
        ordered.loc[:, [f"r{state}" for state in STATE_ORDER]].to_numpy(dtype=float),
        xlabels=condition_ticks(display, width=14),
        ylabels=row_labels,
        title=f"{response_ratio} response",
        colorbar_label=response_axis_label(display),
    )
    _heatmap_panel(
        figure,
        axes[1],
        ordered.loc[:, [f"b{state}" for state in STATE_ORDER]].to_numpy(dtype=float),
        xlabels=condition_ticks(display, width=14),
        ylabels=None,
        title=f"Fluorescence relative to {reference_id}",
        colorbar_label=anchored_fluorescence_axis_label(display),
    )
    figure.suptitle("The handoff preserves response and anchored fluorescence by condition")
    path = out_dir / "plots" / "handoff_matrix.png"
    save_publication_figure(figure, path)
    plt.close(figure)
    return plot_manifest_row(
        plot_id="handoff_matrix",
        tier="handoff",
        title="The handoff preserves response and anchored fluorescence by condition",
        premise="The Reader handoff preserves four response states and four reference-relative fluorescence states without per-design min-max scaling.",
        decision_value="Makes source shifts, state structure, and extreme values visible before study binding.",
        rationale="The downstream objective should receive measured state summaries rather than an assay-specific scalar score.",
        alt_text=f"Two aligned heatmaps show {response_ratio} response and same-state {reference_id}-relative {channels(display)['magnitude_ratio']} fluorescence for every experiment and design in the primary post-event window.",
        non_claim_boundary="Rows from repeated experiments are not independent candidate labels and have not been study-aggregated.",
        data_table="tables/primary_handoff.csv",
        path="plots/handoff_matrix.png",
    )


def write_stability_plot(
    stability: pd.DataFrame,
    *,
    display: dict[str, object],
    out_dir: Path,
) -> dict[str, str]:
    metadata = stability.drop_duplicates("reduction_id").set_index("reduction_id")
    response_ratio = response_ratio_label(display)
    reference_id = channels(display)["reference_design_id"]
    maximum_disagreement = float((1.0 - stability["spearman_to_primary"]).max())
    disagreement_limit = max(0.15, np.ceil(maximum_disagreement * 20.0) / 20.0)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), sharey=True, constrained_layout=True)
    row_labels = [reduction_label(metadata.loc[reduction_id]) for reduction_id in metadata.index]
    for axis, family, title, colorbar_label in (
        (
            axes[0],
            "r",
            f"{response_ratio} rank change",
            f"Rank disagreement (1 - Spearman) for {response_ratio}",
        ),
        (
            axes[1],
            "b",
            "Anchored fluorescence rank change",
            f"Rank disagreement (1 - Spearman) for fluorescence vs {reference_id}",
        ),
    ):
        components = [f"{family}{state}" for state in STATE_ORDER]
        pivot = stability.loc[stability["component"].isin(components)].pivot(
            index="reduction_id",
            columns="component",
            values="spearman_to_primary",
        )
        pivot = pivot.reindex(index=metadata.index, columns=components)
        _heatmap_panel(
            figure,
            axis,
            1.0 - pivot.to_numpy(dtype=float),
            xlabels=condition_ticks(display, width=14),
            ylabels=row_labels if family == "r" else None,
            title=title,
            colorbar_label=colorbar_label,
            fixed_limit=(0.0, disagreement_limit),
            annotate=True,
        )
    figure.suptitle("Prespecified reductions preserve most candidate ordering by condition")
    path = out_dir / "plots" / "reduction_stability.png"
    save_publication_figure(figure, path)
    plt.close(figure)
    return plot_manifest_row(
        plot_id="reduction_stability",
        tier="method_review",
        title="Prespecified reductions preserve most candidate ordering by condition",
        premise="A promoted reduction should not reverse response or anchored-fluorescence ordering under nearby, prespecified summaries.",
        decision_value="Shows which state components are sensitive to window, integration, or response-basis choices.",
        rationale="Method choice should be based on stable measured behavior, not the best downstream model score on one corpus.",
        alt_text=f"Two heatmaps show rank disagreement, calculated as one minus Spearman correlation with the primary reduction, for {response_ratio} response and {reference_id}-relative fluorescence across four conditions and {len(metadata)} prespecified reductions; zero means unchanged ordering.",
        non_claim_boundary="High rank agreement does not prove that the primary window is biologically optimal.",
        data_table="tables/reduction_stability.csv",
        path="plots/reduction_stability.png",
    )


def _heatmap_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    values: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str] | None,
    title: str,
    colorbar_label: str,
    fixed_limit: tuple[float, float] | None = None,
    annotate: bool = False,
) -> None:
    if fixed_limit is None:
        limit = max(float(np.quantile(np.abs(values), 0.98)), 1.0e-6)
        vmin, vmax = -limit, limit
        cmap = "coolwarm"
    else:
        vmin, vmax = fixed_limit
        cmap = "viridis"
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(values.shape[1]), xlabels)
    if ylabels is not None:
        axis.set_yticks(np.arange(values.shape[0]), ylabels, fontsize=7)
    axis.set_title(title)
    if annotate:
        midpoint = (vmin + vmax) / 2.0
        for row_index, column_index in np.ndindex(values.shape):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < midpoint else "#111827",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label(colorbar_label)


def plot_manifest_row(
    *,
    plot_id: str,
    tier: str,
    title: str,
    premise: str,
    decision_value: str,
    rationale: str,
    alt_text: str,
    non_claim_boundary: str,
    data_table: str,
    path: str,
) -> dict[str, str]:
    if not title or title.endswith("."):
        raise ValueError(f"plot title must be a non-empty sentence without a terminal period: {title!r}.")
    return {
        "plot_id": plot_id,
        "tier": tier,
        "title": title,
        "premise": premise,
        "decision_value": decision_value,
        "rationale": rationale,
        "alt_text": alt_text,
        "non_claim_boundary": non_claim_boundary,
        "data_table": data_table,
        "path": path,
    }


def _short_experiment(value: object) -> str:
    text = str(value)
    date = text[:8]
    suffix = text[9:].replace("_", " ")
    return f"{date[:4]}-{date[4:6]}-{date[6:]} | {suffix}"


__all__ = [
    "plot_manifest_row",
    "write_event_plot",
    "write_handoff_plot",
    "write_stability_plot",
]
