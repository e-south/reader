"""Static assay-quality figures for response-window review bundles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader.domains.plate_reader.analysis.response_window.sources import STATE_ORDER

from .plot_style import save_publication_figure, style_data_axis
from .reporting_plots import plot_manifest_row
from .visual_labels import (
    anchored_fluorescence_axis_label,
    anchored_fluorescence_uncertainty_axis_label,
    channels,
    condition_ticks,
    response_axis_label,
    response_ratio_label,
    response_uncertainty_axis_label,
)


def write_repeat_plot(
    repeated: pd.DataFrame,
    *,
    display: dict[str, object],
    out_dir: Path,
) -> dict[str, str]:
    reference_id = channels(display)["reference_design_id"]
    response_ratio = response_ratio_label(display)
    figure_height = 5.6 if repeated.empty else max(5.6, repeated["design_id"].nunique() * 0.42)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, figure_height),
        sharey=True,
        constrained_layout=True,
    )
    if repeated.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No repeated designs", ha="center", va="center")
            axis.set_axis_off()
    else:
        order = repeated.groupby("design_id")["absolute_deviation"].max().sort_values(ascending=False).index
        for axis, prefix, title, xlabel in (
            (
                axes[0],
                "r",
                f"{response_ratio} response",
                f"Absolute deviation from design median\nin {response_axis_label(display)}",
            ),
            (
                axes[1],
                "b",
                "Anchored fluorescence",
                f"Absolute deviation from design median\nin {anchored_fluorescence_axis_label(display)}",
            ),
        ):
            family = repeated.loc[repeated["component"].astype(str).str.startswith(prefix)]
            data = [family.loc[family["design_id"].eq(design), "absolute_deviation"].to_numpy() for design in order]
            axis.boxplot(
                data,
                tick_labels=order,
                vert=False,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "white", "edgecolor": "#374151", "zorder": 3},
                medianprops={"color": "#d97706", "linewidth": 1.5, "zorder": 4},
                whiskerprops={"color": "#374151", "zorder": 3},
                capprops={"color": "#374151", "zorder": 3},
            )
            axis.invert_yaxis()
            axis.set_xlabel(xlabel)
            axis.set_title(title)
            axis.set_box_aspect(1.0)
            style_data_axis(axis, grid_axis="x")
        axes[0].set_ylabel("Repeated design")
    figure.suptitle("Repeated experiments reveal candidate-specific source variation")
    path = out_dir / "plots" / "repeated_design_agreement.png"
    save_publication_figure(figure, path)
    plt.close(figure)
    return plot_manifest_row(
        plot_id="repeated_design_agreement",
        tier="assay_qc",
        title="Repeated experiments reveal candidate-specific source variation",
        premise="Repeated experiments must agree well enough for one candidate-level label to be meaningful.",
        decision_value="Identifies designs whose response or anchored-fluorescence values require explicit aggregation or exclusion.",
        rationale="Treating repeated experiments as independent labels would understate uncertainty and distort model support.",
        alt_text=f"Two horizontal-boxplot panels separate {response_ratio} response deviations from {reference_id}-relative fluorescence deviations for each repeatedly measured design, ordered by the largest observed deviation.",
        non_claim_boundary="Differences can reflect biological, plate, timing, or processing variation; this plot does not assign a cause.",
        data_table="tables/repeated_design_agreement.csv",
        path="plots/repeated_design_agreement.png",
    )


def write_uncertainty_plot(
    uncertainty: pd.DataFrame,
    *,
    display: dict[str, object],
    out_dir: Path,
) -> dict[str, str]:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=True)
    x = np.arange(len(STATE_ORDER))
    width = 0.36
    for axis, family, title, ylabel in (
        (
            axes[0],
            "response",
            f"{response_ratio_label(display)} uncertainty",
            response_uncertainty_axis_label(display),
        ),
        (
            axes[1],
            "anchored_fluorescence",
            "Anchored fluorescence uncertainty",
            anchored_fluorescence_uncertainty_axis_label(display),
        ),
    ):
        rows = uncertainty.loc[uncertainty["family"].eq(family)].set_index("state").reindex(STATE_ORDER)
        axis.bar(
            x - width / 2,
            rows["p90_bootstrap_sd"],
            width,
            label="Bootstrap SD",
            color="#2563eb",
            zorder=3,
        )
        axis.bar(
            x + width / 2,
            rows["p90_event_half_range"],
            width,
            label="Event-time sensitivity (max bound deviation)",
            color="#f59e0b",
            zorder=3,
        )
        axis.set_xticks(x, condition_ticks(display, width=14))
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_box_aspect(0.9)
        style_data_axis(axis, grid_axis="y")
    axes[0].legend(frameon=False)
    figure.suptitle("Replicate and event-timing uncertainty differ by signal and condition")
    path = out_dir / "plots" / "uncertainty_sources.png"
    save_publication_figure(figure, path)
    plt.close(figure)
    return plot_manifest_row(
        plot_id="uncertainty_sources",
        tier="assay_qc",
        title="Replicate and event-timing uncertainty differ by signal and condition",
        premise="Replicate variation and event-time uncertainty are distinct contributions to endpoint uncertainty.",
        decision_value="Shows which source limits the precision of each response or anchored-fluorescence state.",
        rationale="The handoff must preserve uncertainty rather than presenting one reduced number as exact assay truth.",
        alt_text=f"Two grouped-bar panels show the ninetieth-percentile bootstrap standard deviation and event-time sensitivity half-range for {response_ratio_label(display)} response and {channels(display)['reference_design_id']}-relative {channels(display)['magnitude_ratio']} fluorescence in all four conditions.",
        non_claim_boundary="These empirical review summaries are not calibrated measurement-error distributions.",
        data_table="tables/uncertainty_summary.csv",
        path="plots/uncertainty_sources.png",
    )


__all__ = ["write_repeat_plot", "write_uncertainty_plot"]
