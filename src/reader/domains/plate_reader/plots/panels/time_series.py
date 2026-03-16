"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/panels/time_series.py

Shared time-series drawing primitives.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "H"]


def marker_map_for_levels(levels: Sequence[str]) -> dict[str, str]:
    return {str(level): _MARKERS[idx % len(_MARKERS)] for idx, level in enumerate(levels)}


def _maybe_log(ax, enable: bool) -> None:
    if enable:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        if ymin <= 0:
            ax.set_ylim(bottom=max(1e-12, ymin))


def draw_time_series_panel(
    ax,
    *,
    data: pd.DataFrame,
    x_col: str,
    hue_col: str,
    hue_levels: list[str],
    color_map: dict[str, str],
    marker_map: dict[str, str],
    show_replicates: bool,
    ci: float,
    ci_alpha: float,
    line_alpha: float,
    mean_marker_alpha: float,
    replicate_alpha: float,
    add_sheet_lines: bool,
    sheet_lines: list[float] | None,
    sheet_line_kwargs: dict | None,
    log_y: bool,
    xlabel: str,
    ylabel: str,
    legend_loc: str,
    show_legend: bool,
    marked_time: float | None = None,
    marked_time_kwargs: dict | None = None,
) -> None:
    ax.grid(False)
    ax.yaxis.grid(True, which="major")
    ax.xaxis.grid(True, which="major")

    if show_replicates:
        for hue in hue_levels:
            rr = data[data[hue_col].astype(str) == str(hue)]
            ax.scatter(
                rr[x_col],
                rr["value"],
                s=18,
                alpha=replicate_alpha,
                zorder=3,
                linewidths=0.0,
                edgecolors="none",
                marker=marker_map[hue],
                c=color_map[hue],
            )

    sns.lineplot(
        data=data,
        x=x_col,
        y="value",
        hue=hue_col,
        hue_order=hue_levels,
        estimator="mean",
        errorbar=("ci", float(ci)),
        err_style="band",
        err_kws={"alpha": float(ci_alpha)},
        lw=1.8,
        alpha=line_alpha,
        legend=False,
        ax=ax,
        palette=[color_map[h] for h in hue_levels],
        marker=None,
        zorder=1,
    )

    means = data.groupby([hue_col, x_col], dropna=False)["value"].mean().reset_index()
    for hue in hue_levels:
        mm = means[means[hue_col].astype(str) == str(hue)]
        ax.scatter(
            mm[x_col],
            mm["value"],
            s=36,
            zorder=2.5,
            marker=marker_map[hue],
            alpha=mean_marker_alpha,
            edgecolors="none",
            linewidths=0.0,
            c=color_map[hue],
        )

    if add_sheet_lines and sheet_lines:
        style = {"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.8, "alpha": 0.9, "zorder": 0.5}
        style.update(sheet_line_kwargs or {})
        for sheet_x in sheet_lines:
            ax.axvline(float(sheet_x), **style)

    if marked_time is not None:
        style = {"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.9, "alpha": 1.0, "zorder": 0.8}
        style.update(marked_time_kwargs or {})
        ax.axvline(float(marked_time), **style)

    _maybe_log(ax, log_y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                color=color_map[hue],
                marker=marker_map[hue],
                markersize=7,
                linestyle="-",
                linewidth=1.8,
                alpha=mean_marker_alpha,
                label=str(hue),
            )
            for hue in hue_levels
        ]
        ax.legend(handles=handles, loc=legend_loc, title=None)
