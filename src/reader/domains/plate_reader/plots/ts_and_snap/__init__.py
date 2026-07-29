"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/ts_and_snap/__init__.py

Two-panel figure: (left) time series, (right) snapshot barplot,
driven by the same group selection.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import DEFAULT_RC as _RC
from reader.plotting.style import PaletteBook, use_style

from ..common import colors_for, emit_plot_figure
from ..grouping import GroupMatch
from ..panels import (
    draw_snapshot_panel,
    draw_time_series_panel,
    marker_map_for_levels,
)
from .planning import (
    prepare_composite_inputs,
    prepare_snapshot_panel_data,
    prepare_time_series_panel_data,
    resolve_level_order,
    resolve_paired_hue_levels,
)

_FIG_STYLE_KEYS = {
    "axis_label_size",
    "tick_label_size",
    "legend_fontsize",
    "legend_marker_size",
    "line_width",
    "mean_marker_size",
    "mean_marker_every",
    "replicate_marker_size",
    "style_legend_loc",
    "style_legend_title",
    "snap_tick_rotation",
    "ts_title",
    "snap_title",
    "line_alpha",
    "mean_marker_alpha",
    "replicate_alpha",
    "suptitle_y",
}

# -------------------------------- main API --------------------------------


def plot_ts_and_snap(
    *,
    df: pd.DataFrame,
    output_dir: Path | None,
    # grouping
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch = "exact",
    group_layout: str = "separate",
    # time series (left)
    ts_x: str = "time",
    ts_channel: str,
    ts_hue: str,
    ts_style: str | None = None,
    order_hue: list[str] | None = None,
    order_style: list[str] | None = None,
    ts_time_window: list[float] | None = None,
    ts_add_sheet_line: bool = False,
    ts_sheet_line_kwargs: dict | None = None,
    ts_mark_snap_time: bool = False,
    ts_snap_line_kwargs: dict | None = None,
    ts_log_transform: bool | list[str] = False,
    ts_ci: float = 95.0,
    ts_ci_alpha: float = 0.15,
    ts_ci_boot: int = 100,
    ts_ci_seed: int = 0,
    ts_show_replicates: bool = False,
    ts_legend_loc: str = "upper right",
    # snapshot (right)
    snap_x: str = "treatment",
    snap_channel: str | None = None,  # defaults to ts_channel
    snap_hue: str | None = None,  # defaults to None (gray bars, white dots)
    order_x: list[str] | None = None,
    order_snap_hue: list[str] | None = None,
    snap_time: float = 0.0,
    snap_agg: str = "mean",  # "mean" | "median"
    snap_err: str = "sem",  # "sem" | "iqr" | "none"
    snap_time_tolerance: float = 0.51,
    snap_show_legend: bool = False,
    snap_legend_loc: str = "upper right",
    snap_color_by_x: bool = False,
    square_panels: bool = False,
    title: str | None = None,
    # figure/style
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    palette_book: PaletteBook | None = None,
) -> list[PlotFigure]:
    """
    Render time-series/snapshot pairs for each selected group. By default each
    group receives its own two-panel figure; ``group_layout="paired_row"``
    composes every group pair into one horizontal figure.

    Hue handling:
      • Left  requires `ts_hue`
      • Right uses `snap_hue` if provided; when `snap_hue == ts_hue`, colors are shared
      • Otherwise snapshot bars are gray with white replicate dots (no legend by default)
    """
    if snap_agg not in {"mean", "median"}:
        raise ValueError("snap_agg must be 'mean' or 'median'")
    if snap_err not in {"sem", "iqr", "none"}:
        raise ValueError("snap_err must be 'sem', 'iqr', or 'none'")
    if group_layout not in {"separate", "paired_row"}:
        raise ValueError("group_layout must be 'separate' or 'paired_row'")
    if snap_color_by_x and snap_hue is not None:
        raise ValueError("snap_color_by_x requires snap_hue to be omitted")

    fig_kwargs = fig_kwargs or {}

    prepared = prepare_composite_inputs(
        df=df,
        group_on=group_on,
        pool_sets=pool_sets,
        pool_match=pool_match,
        ts_x=ts_x,
        ts_channel=ts_channel,
        ts_hue=ts_hue,
        ts_style=ts_style,
        ts_time_window=ts_time_window,
        snap_x=snap_x,
        snap_channel=snap_channel,
        snap_hue=snap_hue,
    )
    if prepared is None:
        return []
    ts_x_col = prepared.ts_x_col
    group_col = prepared.group_col
    ts_hue_col = prepared.ts_hue_col
    ts_style_col = prepared.ts_style_col
    snap_x_col = prepared.snap_x_col
    snap_hue_col = prepared.snap_hue_col
    ch_ts = prepared.ts_channel
    ch_snap = prepared.snap_channel
    group_frames = prepared.groups

    figures: list[PlotFigure] = []
    snap_fallbacks: list[dict[str, object]] = []

    def _draw_group_pair(
        *,
        label: str,
        d: pd.DataFrame,
        ts_d: pd.DataFrame,
        ax_ts,
        ax_snap,
        paired_row: bool,
        show_ts_legend: bool,
        shared_hue_levels: list[str] | None = None,
        shared_color_map: dict[str, str] | None = None,
    ):
        """Draw one group pair into caller-owned axes."""

        time_series_data = prepare_time_series_panel_data(
            frame=ts_d,
            channel=ch_ts,
            x_col=ts_x_col,
            add_sheet_lines=ts_add_sheet_line,
        )
        ts = time_series_data.frame

        # TS hue levels + colors
        hue_levels_ts = (
            shared_hue_levels
            if shared_hue_levels is not None
            else resolve_level_order(
                observed=ts[ts_hue_col].astype(str).unique().tolist(), configured=order_hue, name="order_hue"
            )
        )
        style_levels_ts = (
            resolve_level_order(
                observed=ts[ts_style_col].astype(str).unique().tolist(),
                configured=order_style,
                name="order_style",
            )
            if ts_style_col
            else []
        )
        marker_map = marker_map_for_levels(hue_levels_ts)
        colors = colors_for(len(hue_levels_ts), palette_book)
        color_map = shared_color_map or {h: colors[i % len(colors)] for i, h in enumerate(hue_levels_ts)}

        # If snapshot uses the same hue column, reuse colors; otherwise compute locally on demand
        def _snap_color_map(
            hue_levels_snap: list[str],
            *,
            color_map=color_map,
            colors=colors,
        ) -> dict[str, str]:
            if snap_hue_col and snap_hue_col == ts_hue_col:
                return {h: color_map.get(h, colors[i % len(colors)]) for i, h in enumerate(hue_levels_snap)}
            snap_colors = colors_for(len(hue_levels_snap), palette_book)
            return {h: snap_colors[i % len(snap_colors)] for i, h in enumerate(hue_levels_snap)}

        # ---- Left: time series (mean ± CI) ----
        if not ts.empty:
            draw_time_series_panel(
                ax_ts,
                data=ts,
                x_col=ts_x_col,
                hue_col=ts_hue_col,
                hue_levels=hue_levels_ts,
                style_col=ts_style_col,
                style_levels=style_levels_ts,
                color_map=color_map,
                marker_map=marker_map,
                segment_col=time_series_data.segment_col,
                show_replicates=ts_show_replicates,
                ci=ts_ci,
                ci_alpha=ts_ci_alpha,
                ci_boot=ts_ci_boot,
                ci_seed=ts_ci_seed,
                line_alpha=float(fig_kwargs.get("line_alpha", 0.85)),
                mean_marker_alpha=float(fig_kwargs.get("mean_marker_alpha", 0.75)),
                replicate_alpha=float(fig_kwargs.get("replicate_alpha", 0.30)),
                add_sheet_lines=ts_add_sheet_line,
                sheet_lines=time_series_data.sheet_lines,
                sheet_line_kwargs=ts_sheet_line_kwargs,
                log_y=(ch_ts in ts_log_transform) if isinstance(ts_log_transform, list) else bool(ts_log_transform),
                xlabel=("Time (h)" if str(ts_x).lower() == "time" else ts_x_col),
                ylabel=ch_ts,
                legend_loc=str(ts_legend_loc),
                show_legend=show_ts_legend,
                style_legend_loc=str(fig_kwargs.get("style_legend_loc", "lower right")),
                style_legend_title=(
                    str(fig_kwargs["style_legend_title"]) if fig_kwargs.get("style_legend_title") is not None else None
                ),
                marked_time=(float(snap_time) if ts_mark_snap_time else None),
                marked_time_kwargs=ts_snap_line_kwargs,
                line_width=float(fig_kwargs.get("line_width", 1.8)),
                mean_marker_size=float(fig_kwargs.get("mean_marker_size", 36.0)),
                mean_marker_every=int(fig_kwargs.get("mean_marker_every", 1)),
                replicate_marker_size=float(fig_kwargs.get("replicate_marker_size", 18.0)),
                axis_label_size=float(fig_kwargs.get("axis_label_size", 10.0)),
                tick_label_size=float(fig_kwargs.get("tick_label_size", 8.0)),
                legend_fontsize=float(fig_kwargs.get("legend_fontsize", 8.0)),
                legend_marker_size=float(fig_kwargs.get("legend_marker_size", 7.0)),
            )
            configured_ts_title = fig_kwargs.get("ts_title")
            if paired_row:
                title_parts = [str(label)]
                if configured_ts_title is not None:
                    title_parts.append(str(configured_ts_title))
                ax_ts.set_title("\n".join(title_parts), fontweight="normal")
            elif configured_ts_title is not None:
                ax_ts.set_title(str(configured_ts_title), fontweight="normal")
            if ts_time_window:
                ax_ts.set_xlim(float(ts_time_window[0]), float(ts_time_window[1]))

        # ---- Right: snapshot barplot ----
        snapshot_data = prepare_snapshot_panel_data(
            frame=d,
            group_col=group_col,
            snap_x_col=snap_x_col,
            snap_hue_col=snap_hue_col,
            snap_channel=ch_snap,
            snap_time=snap_time,
            snap_time_tolerance=snap_time_tolerance,
            snap_err=snap_err,
            order_x=order_x,
            order_snap_hue=order_snap_hue,
            order_hue=order_hue,
            ts_hue_col=ts_hue_col,
        )
        if snapshot_data is not None and snapshot_data.fell_back:
            snap_fallbacks.append(
                {
                    "label": str(label),
                    "times": snapshot_data.fallback_times_preview,
                    "delta": snapshot_data.fallback_delta,
                }
            )
        if snapshot_data is not None:
            hue_levels_snap = snapshot_data.hue_order
            resolved_snap_x_order = snapshot_data.x_order
            configured_snap_title = fig_kwargs.get("snap_title")
            snap_title = (
                f"{configured_snap_title} (t={snapshot_data.time_used:.2f} h)"
                if configured_snap_title is not None
                else f"t={snapshot_data.time_used:.2f} h"
            )
            if paired_row:
                snap_title = f"{label}\n{snap_title}"
            x_color_map = None
            if snap_color_by_x:
                if snap_x_col == ts_hue_col:
                    x_color_map = {value: color_map[value] for value in resolved_snap_x_order if value in color_map}
                else:
                    snap_x_colors = colors_for(len(resolved_snap_x_order), palette_book)
                    x_color_map = {
                        value: snap_x_colors[idx % len(snap_x_colors)]
                        for idx, value in enumerate(resolved_snap_x_order)
                    }
            draw_snapshot_panel(
                ax_snap,
                snapped=snapshot_data.frame,
                stats=snapshot_data.stats,
                x_col=snap_x_col,
                hue_col=snap_hue_col,
                agg=snap_agg,
                err=snap_err,
                palette_book=palette_book,
                show_legend=snap_show_legend,
                legend_loc=str(snap_legend_loc),
                title=snap_title,
                ylabel=ch_snap,
                color_map=_snap_color_map(hue_levels_snap),
                x_color_map=x_color_map,
                x_order=resolved_snap_x_order,
                hue_order=(hue_levels_snap if snap_hue_col else None),
                tick_rotation=float(fig_kwargs.get("snap_tick_rotation", 45.0)),
                axis_label_size=float(fig_kwargs.get("axis_label_size", 10.0)),
                tick_label_size=float(fig_kwargs.get("tick_label_size", 8.0)),
                legend_fontsize=float(fig_kwargs.get("legend_fontsize", 8.0)),
            )

        if square_panels:
            ax_ts.set_box_aspect(1.0)
            ax_snap.set_box_aspect(1.0)

    def _figure_kwargs(*, pair_count: int) -> dict:
        fkw = dict(fig_kwargs)
        if "figsize" not in fkw:
            base_w, base_h = _RC["figure_figsize"]
            fkw["figsize"] = (base_w * 2.0 * pair_count, base_h)
        return {key: value for key, value in fkw.items() if key not in {"rc", "ext", *_FIG_STYLE_KEYS}}

    if group_layout == "paired_row":
        figure_hue_levels = resolve_paired_hue_levels(
            groups=group_frames,
            hue_col=ts_hue_col,
            configured=order_hue,
        )
        figure_colors = colors_for(len(figure_hue_levels), palette_book)
        figure_color_map = {
            hue: figure_colors[index % len(figure_colors)] for index, hue in enumerate(figure_hue_levels)
        }
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=figure_colors):
            fig, axes = plt.subplots(1, 2 * len(group_frames), **_figure_kwargs(pair_count=len(group_frames)))
            for idx, group in enumerate(group_frames):
                ax_ts, ax_snap = axes[2 * idx : 2 * idx + 2]
                _draw_group_pair(
                    label=group.label,
                    d=group.snapshot,
                    ts_d=group.time_series,
                    ax_ts=ax_ts,
                    ax_snap=ax_snap,
                    paired_row=True,
                    show_ts_legend=(idx == 0),
                    shared_hue_levels=figure_hue_levels,
                    shared_color_map=figure_color_map,
                )
            if title:
                fig.suptitle(str(title), y=float(fig_kwargs.get("suptitle_y", 1.04)))
            stub = filename or f"ts_snap__{ch_snap}__paired_row"
            figures.extend(emit_plot_figure(fig=fig, filename=stub, output_dir=output_dir, fig_kwargs=fig_kwargs))
    else:
        for group in group_frames:
            hue_levels = resolve_level_order(
                observed=group.time_series[ts_hue_col].astype(str).unique().tolist(),
                configured=order_hue,
                name="order_hue",
            )
            figure_colors = colors_for(len(hue_levels), palette_book)
            with use_style(rc=fig_kwargs.get("rc"), color_cycle=figure_colors):
                fig, axes = plt.subplots(1, 2, **_figure_kwargs(pair_count=1))
                ax_ts, ax_snap = axes
                _draw_group_pair(
                    label=group.label,
                    d=group.snapshot,
                    ts_d=group.time_series,
                    ax_ts=ax_ts,
                    ax_snap=ax_snap,
                    paired_row=False,
                    show_ts_legend=True,
                )
                fig.suptitle(
                    str(title if title is not None else group.label),
                    y=float(fig_kwargs.get("suptitle_y", 1.04)),
                )
                group_tag = f"__{str(group_col)}={str(group.label)}" if group_col and group.members != (None,) else None
                if filename:
                    stub = f"{filename}{group_tag}" if group_tag else filename
                else:
                    base = f"ts_snap__{ch_snap}"
                    stub = f"{base}{group_tag}" if group_tag else f"{base}__{group.label}"
                figures.extend(emit_plot_figure(fig=fig, filename=stub, output_dir=output_dir, fig_kwargs=fig_kwargs))
    if snap_fallbacks:
        log = logging.getLogger("reader")
        sample = []
        for row in snap_fallbacks[:3]:
            sample.append(f"{row['label']}: times={row['times']} Δ≈{float(row['delta']):.2f} h")
        extra = f" (+{len(snap_fallbacks) - 3} more)" if len(snap_fallbacks) > 3 else ""
        log.info(
            "[warn]ts_and_snap:snapshot[/warn] • requested t=%.2f h; no rows within ±%.2f h for %d group(s) — "
            "using nearest available per key (examples: %s%s)",
            float(snap_time),
            float(snap_time_tolerance),
            len(snap_fallbacks),
            "; ".join(sample),
            extra,
        )

    return figures
