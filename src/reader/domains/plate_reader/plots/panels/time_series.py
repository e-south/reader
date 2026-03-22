"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/panels/time_series.py

Shared time-series drawing primitives.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..common import bootstrap_mean_interval

_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "H"]


def marker_map_for_levels(levels: Sequence[str]) -> dict[str, str]:
    return {str(level): _MARKERS[idx % len(_MARKERS)] for idx, level in enumerate(levels)}


def _maybe_log(ax, enable: bool) -> None:
    if enable:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        if ymin <= 0:
            ax.set_ylim(bottom=max(1e-12, ymin))


def _summarize_time_series_lines(
    *,
    data: pd.DataFrame,
    x_col: str,
    hue_col: str,
    hue_levels: Sequence[str],
    segment_col: str | None,
    ci: float,
    ci_boot: int,
    ci_seed: int,
) -> dict[str, pd.DataFrame]:
    if data.empty:
        return {}

    columns = [x_col, hue_col, "value"]
    if segment_col is not None and segment_col in data.columns:
        columns.append(segment_col)
    plot_data = data.loc[:, columns].copy()
    plot_data[hue_col] = plot_data[hue_col].astype(str)
    active_segment_col = segment_col if segment_col is not None and segment_col in plot_data.columns else None
    if active_segment_col is not None:
        plot_data[active_segment_col] = plot_data[active_segment_col].astype(str)
    else:
        active_segment_col = "__segment_id"
        plot_data[active_segment_col] = "segment"
    plot_data["value"] = pd.to_numeric(plot_data["value"], errors="coerce")
    grouped = plot_data.groupby([hue_col, active_segment_col, x_col], dropna=False, sort=True)["value"]
    rng = np.random.default_rng(int(ci_seed))

    summaries: dict[str, dict[str, list[object]]] = {
        str(hue): {x_col: [], "segment": [], "mean": [], "lower": [], "upper": []} for hue in hue_levels
    }
    for (hue, segment_id, x_value), series in grouped:
        hue_key = str(hue)
        payload = summaries.get(hue_key)
        if payload is None:
            continue
        values = series.to_numpy(dtype=float, copy=False)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        mean, lower, upper = bootstrap_mean_interval(values, ci=ci, ci_boot=ci_boot, rng=rng)
        payload[x_col].append(x_value)
        payload["segment"].append(str(segment_id))
        payload["mean"].append(mean)
        payload["lower"].append(lower)
        payload["upper"].append(upper)

    out: dict[str, pd.DataFrame] = {}
    for hue in hue_levels:
        payload = summaries[str(hue)]
        if not payload[x_col]:
            continue
        out[str(hue)] = pd.DataFrame(payload).sort_values(["segment", x_col], kind="stable")
    return out


def draw_time_series_panel(
    ax,
    *,
    data: pd.DataFrame,
    x_col: str,
    hue_col: str,
    hue_levels: list[str],
    color_map: dict[str, str],
    marker_map: dict[str, str],
    segment_col: str | None,
    show_replicates: bool,
    ci: float,
    ci_alpha: float,
    ci_boot: int,
    ci_seed: int,
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
    legend_label_map: Mapping[str, str] | None = None,
    marked_time: float | None = None,
    marked_time_kwargs: dict | None = None,
    line_width: float = 1.8,
    mean_marker_size: float = 36.0,
    replicate_marker_size: float = 18.0,
    axis_label_size: float = 10.0,
    tick_label_size: float = 8.0,
    legend_fontsize: float = 8.0,
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
                s=replicate_marker_size,
                alpha=replicate_alpha,
                zorder=3,
                linewidths=0.0,
                edgecolors="none",
                marker=marker_map[hue],
                c=color_map[hue],
            )

    summary_by_hue = _summarize_time_series_lines(
        data=data,
        x_col=x_col,
        hue_col=hue_col,
        hue_levels=hue_levels,
        segment_col=segment_col,
        ci=ci,
        ci_boot=ci_boot,
        ci_seed=ci_seed,
    )
    for hue in hue_levels:
        mm = summary_by_hue.get(str(hue))
        if mm is None or mm.empty:
            continue
        for _, segment_df in mm.groupby("segment", sort=False):
            if float(ci) > 0:
                ax.fill_between(
                    segment_df[x_col],
                    segment_df["lower"],
                    segment_df["upper"],
                    alpha=float(ci_alpha),
                    color=color_map[hue],
                    linewidth=0.0,
                    zorder=1,
                )
            ax.plot(
                segment_df[x_col],
                segment_df["mean"],
                color=color_map[hue],
                linewidth=line_width,
                alpha=line_alpha,
                zorder=1.5,
            )
            ax.scatter(
                segment_df[x_col],
                segment_df["mean"],
                s=mean_marker_size,
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
    ax.set_xlabel(xlabel, fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                color=color_map[hue],
                marker=marker_map[hue],
                markersize=7,
                linestyle="-",
                linewidth=line_width,
                alpha=mean_marker_alpha,
                label=(str(legend_label_map.get(str(hue), hue)) if legend_label_map is not None else str(hue)),
            )
            for hue in hue_levels
        ]
        ax.legend(handles=handles, loc=legend_loc, title=None, frameon=False, fontsize=legend_fontsize)
