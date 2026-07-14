"""Endpoint, sensitivity, and quality figures for response-window review."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import style_data_axis
from .sources import STATE_ORDER
from .visual_labels import (
    STATE_COLORS,
    anchored_fluorescence_axis_label,
    anchored_fluorescence_uncertainty_axis_label,
    channels,
    condition_ticks,
    reduction_label,
    response_axis_label,
    response_ratio_label,
    response_uncertainty_axis_label,
)


def state_summary_figure(*, selected: pd.Series, display: dict[str, object]) -> plt.Figure:
    display_channels = channels(display)
    reference_id = display_channels["reference_design_id"]
    response_ratio = response_ratio_label(display)
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.9), constrained_layout=True)
    x = np.arange(len(STATE_ORDER))
    specs = (
        (axes[0], "r", f"{response_ratio} response", response_axis_label(display)),
        (
            axes[1],
            "b",
            f"Fluorescence relative to {reference_id}",
            anchored_fluorescence_axis_label(display),
        ),
    )
    for axis, prefix, title, ylabel in specs:
        values = np.asarray([selected[f"{prefix}{state}"] for state in STATE_ORDER], dtype=float)
        bootstrap = np.asarray([selected[f"{prefix}{state}_bootstrap_sd"] for state in STATE_ORDER], dtype=float)
        event = np.asarray([selected[f"{prefix}{state}_event_half_range"] for state in STATE_ORDER], dtype=float)
        axis.bar(x, values, color=[STATE_COLORS[state] for state in STATE_ORDER], alpha=0.88, zorder=3)
        axis.errorbar(
            x,
            values,
            yerr=np.hypot(bootstrap, event),
            fmt="none",
            ecolor="#111827",
            capsize=4,
            zorder=4,
        )
        axis.axhline(0.0, color="#111827", linewidth=0.9, zorder=2)
        axis.set_xticks(x, condition_ticks(display, width=14))
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_box_aspect(0.9)
        style_data_axis(axis, grid_axis="y")
    figure.suptitle("The selected window preserves response and anchored fluorescence by condition")
    return figure


def reduction_sensitivity_figure(*, rows: pd.DataFrame, display: dict[str, object]) -> plt.Figure:
    if rows.empty:
        raise ValueError("reduction sensitivity requires at least one response-window row.")
    rows = rows.copy()
    rows["role_order"] = rows["reduction_role"].astype(str).map({"primary": 0, "sensitivity": 1})
    rows = rows.sort_values(["role_order", "reduction_id"], kind="mergesort")
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 6.0), sharey=True, constrained_layout=True)
    row_labels = [reduction_label(row._asdict()) for row in rows.itertuples(index=False)]
    _heatmap_panel(
        figure,
        axes[0],
        rows.loc[:, [f"r{state}" for state in STATE_ORDER]].to_numpy(dtype=float),
        xlabels=condition_ticks(display, width=14),
        ylabels=row_labels,
        title=f"{response_ratio_label(display)} response",
        colorbar_label=response_axis_label(display),
    )
    _heatmap_panel(
        figure,
        axes[1],
        rows.loc[:, [f"b{state}" for state in STATE_ORDER]].to_numpy(dtype=float),
        xlabels=condition_ticks(display, width=14),
        ylabels=None,
        title=f"Fluorescence relative to {channels(display)['reference_design_id']}",
        colorbar_label=anchored_fluorescence_axis_label(display),
    )
    figure.suptitle("Prespecified reductions retain the same condition-level response structure")
    return figure


def quality_figure(*, selected: pd.Series, selected_wells: pd.DataFrame, display: dict[str, object]) -> plt.Figure:
    if selected_wells.empty:
        raise ValueError("selected response-window row has no well-level evidence.")
    x = np.arange(len(STATE_ORDER))
    replicates = selected_wells.groupby("state")["position"].nunique().reindex(STATE_ORDER).to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.5), constrained_layout=True)
    width = 0.36
    for axis, prefix, title, ylabel in (
        (
            axes[0],
            "r",
            f"{response_ratio_label(display)} uncertainty",
            response_uncertainty_axis_label(display),
        ),
        (
            axes[1],
            "b",
            "Anchored fluorescence uncertainty",
            anchored_fluorescence_uncertainty_axis_label(display),
        ),
    ):
        bootstrap = np.asarray([selected[f"{prefix}{state}_bootstrap_sd"] for state in STATE_ORDER], dtype=float)
        event = np.asarray([selected[f"{prefix}{state}_event_half_range"] for state in STATE_ORDER], dtype=float)
        axis.bar(
            x - width / 2,
            bootstrap,
            width,
            color="#2563eb",
            label="Replicate bootstrap",
            zorder=3,
        )
        axis.bar(x + width / 2, event, width, color="#f59e0b", label="Event interval", zorder=3)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_xticks(
            x,
            condition_ticks(display, width=10),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
            fontsize=7,
        )
        axis.set_box_aspect(1.0)
        style_data_axis(axis, grid_axis="y")
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[2].bar(x, replicates, color=[STATE_COLORS[state] for state in STATE_ORDER], zorder=3)
    axes[2].set_ylabel("Independent wells")
    axes[2].set_title("Replicate support")
    axes[2].set_xticks(
        x,
        condition_ticks(display, width=10),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=7,
    )
    axes[2].set_box_aspect(1.0)
    style_data_axis(axes[2], grid_axis="y")
    figure.legend(legend_handles, legend_labels, loc="outside lower center", ncols=2, frameon=False)
    figure.suptitle("The handoff exposes uncertainty and replicate support for every condition")
    return figure


def measured_response_examples_figure(*, rows: pd.DataFrame, display: dict[str, object]) -> plt.Figure:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.8, max(5.8, 0.72 * len(rows) + 2.0)),
        sharey=True,
        constrained_layout=True,
    )
    labels = rows["experiment_id"].astype(str).str.split("_").str[0] + " | " + rows["example_label"].astype(str)
    _heatmap_panel(
        figure,
        axes[0],
        rows.loc[:, [f"r{state}" for state in STATE_ORDER]].to_numpy(dtype=float),
        xlabels=condition_ticks(display, width=14),
        ylabels=labels.tolist(),
        title=f"{response_ratio_label(display)} response",
        colorbar_label=response_axis_label(display),
    )
    _heatmap_panel(
        figure,
        axes[1],
        rows.loc[:, [f"b{state}" for state in STATE_ORDER]].to_numpy(dtype=float),
        xlabels=condition_ticks(display, width=14),
        ylabels=None,
        title=f"Fluorescence relative to {channels(display)['reference_design_id']}",
        colorbar_label=anchored_fluorescence_axis_label(display),
    )
    figure.suptitle("Measured response examples provide direction checks across all four conditions")
    return figure


def _heatmap_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    values: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str] | None,
    title: str,
    colorbar_label: str,
) -> None:
    limit = max(float(np.quantile(np.abs(values), 0.98)), 1.0)
    image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="equal")
    axis.set_xticks(
        np.arange(values.shape[1]),
        xlabels,
        rotation=50,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )
    if ylabels is not None:
        axis.set_yticks(np.arange(values.shape[0]), ylabels)
    axis.set_title(title)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            color = "white" if abs(values[row_index, column_index]) > limit * 0.55 else "#111827"
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label(colorbar_label)


__all__ = [
    "quality_figure",
    "measured_response_examples_figure",
    "reduction_sensitivity_figure",
    "state_summary_figure",
]
