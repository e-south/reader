"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/ts_and_snap.py

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
import polars as pl

from reader.plotting.sinks import PlotFigure
from reader.plotting.style import DEFAULT_RC as _RC
from reader.plotting.style import PaletteBook, use_style

from ..ordering import order_levels
from .common import alias_column, colors_for, emit_plot_figure, require_columns, warn_if_empty
from .grouping import GroupMatch, resolve_groups
from .panels import (
    draw_snapshot_panel,
    draw_time_series_panel,
    marker_map_for_levels,
    select_snapshot_rows,
    summarize_snapshot_values,
)

# -------------------------------- main API --------------------------------


def plot_ts_and_snap(
    *,
    df: pd.DataFrame,
    output_dir: Path | None,
    # grouping
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch = "exact",
    # time series (left)
    ts_x: str = "time",
    ts_channel: str,
    ts_hue: str,
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
    snap_time: float = 0.0,
    snap_agg: str = "mean",  # "mean" | "median"
    snap_err: str = "sem",  # "sem" | "iqr" | "none"
    snap_time_tolerance: float = 0.51,
    snap_show_legend: bool = False,
    snap_legend_loc: str = "upper right",
    # figure/style
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    palette_book: PaletteBook | None = None,
) -> list[PlotFigure]:
    """
    Render one figure per group value (if group_on is set), each with two subplots:
      • Left  = time series (mean ± CI) for `ts_channel`
      • Right = snapshot barplot for `snap_channel` (defaults to `ts_channel`) at `snap_time`

    Hue handling:
      • Left  requires `ts_hue`
      • Right uses `snap_hue` if provided; when `snap_hue == ts_hue`, colors are shared
      • Otherwise snapshot bars are gray with white replicate dots (no legend by default)
    """
    if snap_agg not in {"mean", "median"}:
        raise ValueError("snap_agg must be 'mean' or 'median'")
    if snap_err not in {"sem", "iqr", "none"}:
        raise ValueError("snap_err must be 'sem', 'iqr', or 'none'")

    fig_kwargs = fig_kwargs or {}

    # Resolve columns (prefer *_alias when present)
    ts_x_col = alias_column(df, ts_x)
    group_col = alias_column(df, group_on) if group_on else None
    ts_hue_col = alias_column(df, ts_hue)
    snap_x_col = alias_column(df, snap_x)
    snap_hue_col = alias_column(df, snap_hue) if snap_hue else None

    ch_ts = str(ts_channel)
    ch_snap = str(snap_channel if snap_channel else ts_channel)

    required = ["time", "channel", "value", "position", ts_x_col, ts_hue_col, snap_x_col]
    if group_col:
        required.append(group_col)
    if snap_hue_col:
        required.append(snap_hue_col)
    require_columns(df, required, where="ts_and_snap")

    # Base numerics
    work_pl = pl.from_pandas(df)
    value_expr = pl.col("value").cast(pl.Float64, strict=False)
    value_expr = pl.when(value_expr.is_nan()).then(None).otherwise(value_expr)
    time_expr = pl.col("time").cast(pl.Float64, strict=False)
    time_expr = pl.when(time_expr.is_nan()).then(None).otherwise(time_expr)
    work_pl = work_pl.with_columns(
        value_expr.alias("value"),
        time_expr.alias("time"),
    )
    if ts_time_window:
        lo, hi = float(ts_time_window[0]), float(ts_time_window[1])
        ts_x_num = pl.col(ts_x_col).cast(pl.Float64, strict=False)
        work_pl = work_pl.filter((ts_x_num >= lo) & (ts_x_num <= hi))
    work = work_pl.to_pandas(use_pyarrow_extension_array=False)
    if warn_if_empty(work, where="ts_and_snap", detail="after time_window filter"):
        return []

    available_channels = sorted(work["channel"].astype(str).unique().tolist())
    if ch_ts not in available_channels:
        raise ValueError(f"ts_and_snap: ts_channel {ch_ts!r} not in data. Available: {available_channels}")
    if ch_snap not in available_channels:
        raise ValueError(f"ts_and_snap: snap_channel {ch_snap!r} not in data. Available: {available_channels}")

    # Figure iteration over groups
    if group_col:
        universe = order_levels(work[group_col].astype(str).unique().tolist())
        fig_groups = (
            resolve_groups(universe, pool_sets, match=pool_match) if pool_sets else [(g, [g]) for g in universe]
        )
    else:
        fig_groups = [("all", [None])]

    figures: list[PlotFigure] = []
    snap_fallbacks: list[dict[str, object]] = []
    for label, members in fig_groups:
        d = work.copy()
        if group_col and members != [None]:
            d = d[d[group_col].astype(str).isin(members)]
        if d.empty:
            continue

        # TS hue levels + colors
        hue_levels_ts = order_levels(d[ts_hue_col].astype(str).unique().tolist())
        marker_map = marker_map_for_levels(hue_levels_ts)
        colors = colors_for(len(hue_levels_ts), palette_book)
        color_map = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels_ts)}

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

        with use_style(rc=(fig_kwargs or {}).get("rc"), color_cycle=colors):
            # If user didn't override figsize, widen it for a side-by-side layout.
            fkw = dict(fig_kwargs)
            if "figsize" not in fkw:
                # Use the base height and double the width
                base_w, base_h = _RC["figure_figsize"]
                fkw["figsize"] = (base_w * 2.0, base_h)

            fig, axes = plt.subplots(1, 2, **{k: v for k, v in fkw.items() if k not in {"rc", "ext"}})
            ax_ts, ax_snap = axes  # left, right

            # ---- Left: time series (mean ± CI) ----
            ts = d[d["channel"].astype(str) == ch_ts].copy()
            if not ts.empty:
                sheet_lines = None
                if ts_add_sheet_line and "sheet_index" in ts.columns:
                    starts = sorted(ts.groupby("sheet_index")[ts_x_col].min().dropna().tolist())
                    sheet_lines = starts[1:] if len(starts) > 1 else []

                draw_time_series_panel(
                    ax_ts,
                    data=ts,
                    x_col=ts_x_col,
                    hue_col=ts_hue_col,
                    hue_levels=hue_levels_ts,
                    color_map=color_map,
                    marker_map=marker_map,
                    show_replicates=ts_show_replicates,
                    ci=ts_ci,
                    ci_alpha=ts_ci_alpha,
                    ci_boot=ts_ci_boot,
                    ci_seed=ts_ci_seed,
                    line_alpha=float(fig_kwargs.get("line_alpha", 0.85)),
                    mean_marker_alpha=float(fig_kwargs.get("mean_marker_alpha", 0.75)),
                    replicate_alpha=float(fig_kwargs.get("replicate_alpha", 0.30)),
                    add_sheet_lines=ts_add_sheet_line,
                    sheet_lines=sheet_lines,
                    sheet_line_kwargs=ts_sheet_line_kwargs,
                    log_y=(ch_ts in ts_log_transform) if isinstance(ts_log_transform, list) else bool(ts_log_transform),
                    xlabel=("Time (h)" if str(ts_x).lower() == "time" else ts_x_col),
                    ylabel=ch_ts,
                    legend_loc=str(ts_legend_loc),
                    show_legend=True,
                    marked_time=(float(snap_time) if ts_mark_snap_time else None),
                    marked_time_kwargs=ts_snap_line_kwargs,
                )

            # ---- Right: snapshot barplot ----
            snap = d.copy()
            key_cols = [c for c in [group_col, snap_x_col, snap_hue_col, "channel", "position"] if c]
            selection = select_snapshot_rows(
                df=snap,
                target_time=float(snap_time),
                keys=key_cols,
                channel=ch_snap,
                tolerance=float(snap_time_tolerance),
            )
            snapped = selection.rows
            t_used = selection.time_used
            if selection.fell_back and not snapped.empty:
                snap_fallbacks.append(
                    {
                        "label": str(label),
                        "times": selection.fallback_times_preview or "",
                        "delta": float(selection.fallback_delta or 0.0),
                    }
                )
            if not snapped.empty:
                base_group_cols: list[str] = [snap_x_col] + ([snap_hue_col] if snap_hue_col else [])
                stats = summarize_snapshot_values(df=snapped, group_cols=base_group_cols, err=snap_err)
                hue_levels_snap = (
                    order_levels(stats[snap_hue_col].astype(str).unique().tolist()) if snap_hue_col else ["_single"]
                )
                draw_snapshot_panel(
                    ax_snap,
                    snapped=snapped,
                    stats=stats,
                    x_col=snap_x_col,
                    hue_col=snap_hue_col,
                    agg=snap_agg,
                    err=snap_err,
                    palette_book=palette_book,
                    show_legend=snap_show_legend,
                    legend_loc=str(snap_legend_loc),
                    title=f"t={t_used:.2f} h",
                    ylabel=ch_snap,
                    color_map=_snap_color_map(hue_levels_snap),
                )

            # ---- figure title + save ----
            fig.suptitle(f"{label}", y=float(fig_kwargs.get("suptitle_y", 1.04)))
            # ----- Unique, descriptive filenames -----
            # If grouping is active, append "<group_col>=<label>" to make per-group files distinct.
            group_tag = None
            if group_col and members != [None]:
                # Example: "__genotype=araBADp"  or "__genotype=Ara_related" for pooled sets
                group_tag = f"__{str(group_col)}={str(label)}"

            if filename:
                # Respect the base name but *append* the group tag when present
                stub = f"{filename}{group_tag}" if group_tag else filename
            else:
                base = f"ts_snap__{ch_snap}"
                # Backward‑compatible default still includes label; enhanced with group_col when available
                stub = f"{base}{group_tag}" if group_tag else f"{base}__{label}"
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
