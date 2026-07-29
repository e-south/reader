from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.lines import Line2D

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import PaletteBook, use_style

from ..ordering import order_levels
from .common import alias_column, best_subplot_grid, plot_figure, pretty_name, require_columns, warn_if_empty
from .grouping import GroupMatch, resolve_groups
from .panels import draw_time_series_panel, marker_map_for_levels


def plot_time_series(
    *,
    df: pd.DataFrame,
    x: str,
    y: list[str] | None,
    hue: str,
    channels: list[str] | None,
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch,
    fig_kwargs: dict | None,
    add_sheet_line: bool,
    sheet_line_kwargs: dict | None,
    log_transform: bool | list[str],
    time_window: list[float] | None,
    palette_book: PaletteBook | None,
    ci: float = 95.0,
    ci_alpha: float = 0.15,
    ci_boot: int = 100,
    ci_seed: int = 0,
    legend_loc: str = "upper right",
    show_replicates: bool = False,
    filename: str | None = None,
    xlabel: str | None = None,
    ylabel_map: Mapping[str, str] | None = None,
    hue_label_map: Mapping[str, str] | None = None,
    shared_legend: bool = False,
) -> list[PlotFigure]:
    """
    Time-series plotting with one figure per group (default: per genotype[(_alias)]),
    subplots across channels, mean lines with CI bands, and *vertical gray background
    grid bands* behind the time axis (alternating between tick intervals).
    """
    xcol = alias_column(df, x)
    line_alpha = float((fig_kwargs or {}).get("line_alpha", 0.85))
    mean_marker_alpha = float((fig_kwargs or {}).get("mean_marker_alpha", 0.75))
    replicate_alpha = float((fig_kwargs or {}).get("replicate_alpha", 0.30))
    axis_label_size = float((fig_kwargs or {}).get("axis_label_size", 10.0))
    title_fontsize = float((fig_kwargs or {}).get("title_fontsize", axis_label_size))
    tick_label_size = float((fig_kwargs or {}).get("tick_label_size", 8.0))
    legend_fontsize = float((fig_kwargs or {}).get("legend_fontsize", 8.0))
    legend_marker_size = float((fig_kwargs or {}).get("legend_marker_size", 7.0))
    mean_marker_size = float((fig_kwargs or {}).get("mean_marker_size", 36.0))
    replicate_marker_size = float((fig_kwargs or {}).get("replicate_marker_size", 18.0))
    line_width = float((fig_kwargs or {}).get("line_width", 1.8))

    require_columns(df, [xcol, "channel", "value"], where="time_series")

    # Channel roster
    y_feats = (list(y) if y else list(channels or [])) or sorted(df["channel"].astype(str).unique().tolist())

    # Base frame
    base_pl = pl.from_pandas(df)
    value_expr = pl.col("value").cast(pl.Float64, strict=False)
    value_expr = pl.when(value_expr.is_nan()).then(None).otherwise(value_expr)
    base_pl = base_pl.with_columns(value_expr.alias("value"))
    if time_window:
        lo, hi = float(time_window[0]), float(time_window[1])
        x_num = pl.col(xcol).cast(pl.Float64, strict=False)
        base_pl = base_pl.filter((x_num >= lo) & (x_num <= hi))
    base = base_pl.to_pandas(use_pyarrow_extension_array=False)
    if warn_if_empty(base, where="time_series", detail="after time_window filter"):
        return []

    group_col = alias_column(base, group_on) if group_on else None
    hue_col = alias_column(base, hue)
    if hue_col and hue_col not in base.columns:
        raise ValueError(
            f"time_series: missing hue column {hue_col!r}. "
            "Either set hue=None/another column or add metadata via merge/sample_map."
        )
    if group_col and group_col not in base.columns:
        raise ValueError(f"time_series: missing group_on column {group_col!r}")
    # Drop None-like labels
    if group_col:
        mask = base[group_col].notna() & (
            ~base[group_col].astype(str).str.strip().str.lower().isin({"none", "nan", ""})
        )
        base = base.loc[mask].copy()
        if warn_if_empty(base, where="time_series", detail="after group_on filter"):
            return []
    if group_col:
        universe = order_levels(base[group_col].astype(str).unique().tolist())
        fig_groups = (
            resolve_groups(universe, pool_sets, match=pool_match) if pool_sets else [(g, [g]) for g in universe]
        )
    else:
        fig_groups = [("all", [None])]

    segment_columns = (
        ["acquisition_segment_id"]
        if "acquisition_segment_id" in base.columns
        else [column for column in ("plate_id", "source_file", "sheet_name", "sheet_index") if column in base.columns]
    )
    segment_col: str | None = None
    if segment_columns:
        segment_col = "__plot_segment"
        segment_frame = base[segment_columns].copy()
        for column in segment_columns:
            segment_frame[column] = segment_frame[column].astype(str)
        base[segment_col] = segment_frame.agg("::".join, axis=1)

    available_channels = sorted(base["channel"].astype(str).unique().tolist())
    explicit_channels = list(y) if y else (list(channels) if channels else [])
    if explicit_channels:
        missing = [str(c) for c in explicit_channels if str(c) not in available_channels]
        if missing:
            raise ValueError(f"time_series: requested channels not found: {missing}. Available: {available_channels}")

    # Optional sheet-change lines
    sheet_lines = None
    if add_sheet_line and "sheet_index" in base.columns:
        # mark transitions between sheets: skip the first start
        starts = sorted(base.groupby("sheet_index")[xcol].min().dropna().tolist())
        sheet_lines = starts[1:] if len(starts) > 1 else []

    # Colors for hue
    def _colors(n: int) -> list[str]:
        if palette_book:
            if n == 1:
                pal = palette_book.colors(2)
                first = (pal[0] or "").lower()
                return [pal[1]] if first in {"#000000", "black"} else [pal[0]]
            return palette_book.colors(n)
        cyc = PaletteBook(name="colorblind").colors(max(2, n))
        if n == 1 and str(cyc[0]).lower() in {"#000000", "black"} and len(cyc) > 1:
            return [cyc[1]]
        return cyc[:n]

    # Per-figure drawing
    figures: list[PlotFigure] = []
    for label, members in fig_groups:
        d = base.copy()
        if group_col and members != [None]:
            d = d[d[group_col].astype(str).isin(members)]
        if d.empty:
            continue

        hue_levels = order_levels(d[hue_col].astype(str).unique().tolist())
        marker_map = marker_map_for_levels(hue_levels)
        colors = _colors(len(hue_levels))
        color_map = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels)}
        legend_label_map = {str(key): str(value) for key, value in (hue_label_map or {}).items()} or None
        rows, cols = best_subplot_grid(len(y_feats))

        with use_style(rc=(fig_kwargs or {}).get("rc"), color_cycle=colors):
            custom_figsize = (fig_kwargs or {}).get("figsize")
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=tuple(custom_figsize) if custom_figsize is not None else (cols * 5, rows * 5),
                constrained_layout=False,
            )
            axes = np.atleast_1d(axes).ravel()

            # Optional rasterization threshold for heavy artists (replicate dots)
            rz = (fig_kwargs or {}).get("rasterize_zorder", None)
            if rz is not None:
                for ax in axes:
                    ax.set_rasterization_zorder(float(rz))

            fig.suptitle(
                f"{label}",
                y=0.97 if shared_legend else 0.985,
                fontweight="normal",
                x=0.5,
                ha="center",
                fontsize=title_fontsize,
            )

            for idx, ch in enumerate(y_feats):
                ax = axes[idx]
                sub = d[d["channel"].astype(str) == ch].copy()
                if sub.empty:
                    ax.set_visible(False)
                    continue

                # Start clean; show horizontal + vertical major grid lines
                draw_time_series_panel(
                    ax,
                    data=sub,
                    x_col=xcol,
                    hue_col=hue_col,
                    hue_levels=hue_levels,
                    color_map=color_map,
                    marker_map=marker_map,
                    segment_col=segment_col,
                    show_replicates=show_replicates,
                    ci=ci,
                    ci_alpha=ci_alpha,
                    ci_boot=ci_boot,
                    ci_seed=ci_seed,
                    line_alpha=line_alpha,
                    mean_marker_alpha=mean_marker_alpha,
                    replicate_alpha=replicate_alpha,
                    add_sheet_lines=bool(sheet_lines),
                    sheet_lines=sheet_lines,
                    sheet_line_kwargs=sheet_line_kwargs,
                    log_y=(ch in log_transform) if isinstance(log_transform, list) else bool(log_transform),
                    xlabel=(
                        str(xlabel)
                        if xlabel is not None
                        else ("Time (h)" if str(x).lower() == "time" else pretty_name(str(xcol)))
                    ),
                    ylabel=(str(ylabel_map.get(str(ch), ch)) if ylabel_map is not None else str(ch)),
                    legend_loc=legend_loc,
                    show_legend=(idx == 0 and not shared_legend),
                    legend_label_map=legend_label_map,
                    line_width=line_width,
                    mean_marker_size=mean_marker_size,
                    replicate_marker_size=replicate_marker_size,
                    axis_label_size=axis_label_size,
                    tick_label_size=tick_label_size,
                    legend_fontsize=legend_fontsize,
                    legend_marker_size=legend_marker_size,
                )
                with suppress(Exception):
                    ax.set_box_aspect(1.0)

            # Hide extra axes if any
            for j in range(len(y_feats), len(axes)):
                axes[j].set_visible(False)

            if shared_legend and hue_levels:
                handles = [
                    Line2D(
                        [0],
                        [0],
                        color=color_map[hue],
                        marker=marker_map[hue],
                        markersize=legend_marker_size,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=mean_marker_alpha,
                        label=(str(legend_label_map.get(str(hue), hue)) if legend_label_map is not None else str(hue)),
                    )
                    for hue in hue_levels
                ]
                fig.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.94),
                    ncol=max(1, min(4, len(handles))),
                    frameon=False,
                    title=None,
                    fontsize=legend_fontsize,
                )
                fig.subplots_adjust(
                    top=float((fig_kwargs or {}).get("top", 0.82)),
                    bottom=float((fig_kwargs or {}).get("bottom", 0.10)),
                    left=float((fig_kwargs or {}).get("left", 0.08)),
                    right=float((fig_kwargs or {}).get("right", 0.98)),
                    wspace=float((fig_kwargs or {}).get("wspace", 0.22)),
                    hspace=float((fig_kwargs or {}).get("hspace", 0.24)),
                )
            else:
                fig.subplots_adjust(
                    top=float((fig_kwargs or {}).get("top", 0.92)),
                    bottom=float((fig_kwargs or {}).get("bottom", 0.10)),
                    left=float((fig_kwargs or {}).get("left", 0.08)),
                    right=float((fig_kwargs or {}).get("right", 0.98)),
                    wspace=float((fig_kwargs or {}).get("wspace", 0.22)),
                    hspace=float((fig_kwargs or {}).get("hspace", 0.24)),
                )

            # Allow file type override via fig.ext ("pdf" | "png" | "svg", etc.)
            group_tag = None
            if group_col and members != [None]:
                group_tag = f"__{str(group_col)}={str(label)}"
            stub = (f"{filename}{group_tag}" if group_tag else filename) if filename else f"ts__{label}"
            figures.append(plot_figure(fig=fig, filename=stub, fig_kwargs=fig_kwargs))
    return figures
