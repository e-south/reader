"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/snapshot_barplot/__init__.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reader.core.plot_sinks import PlotFigure
from reader.core.plot_style import PaletteBook, use_style

from ...ordering import order_levels
from ..common import alias_column, best_subplot_grid, colors_for, emit_plot_figure, require_columns, warn_if_empty
from ..grouping import GroupMatch
from ..panels import draw_snapshot_panel, select_snapshot_rows, summarize_snapshot_values
from .planning import build_figure_groups, compute_shared_ylim, resolve_panel_configuration

_PanelBy = Literal["channel", "x", "group"]
_FileBy = Literal["auto", "group", "channel", "x"]


def _figure_stub(
    *,
    filename: str | None,
    panel_by: _PanelBy,
    group_col: str | None,
    fig_label: str,
    members: list[str | None],
    selected_channel: str | None,
    multiple_files: bool,
) -> str:
    if filename:
        if not multiple_files:
            return filename
        suffix_parts: list[str] = []
        if panel_by == "group" and selected_channel is not None:
            suffix_parts.append(f"channel={selected_channel}")
        elif panel_by == "channel":
            key = members[0] if group_col and members and members[0] is not None else fig_label
            suffix_parts.append(f"group={key}")
        else:
            suffix_parts.append(f"panel={fig_label}")
        return f"{filename}__{'__'.join(suffix_parts)}"
    if panel_by == "group":
        return f"snap__grp__{selected_channel}"
    if panel_by == "channel":
        key = members[0] if group_col and members and members[0] is not None else fig_label
        return f"snap__ch__{key}"
    return f"snap__x__{selected_channel}__{fig_label}"


def plot_snapshot_barplot(
    *,
    df: pd.DataFrame,
    output_dir: Path | None,
    x: str,
    y: list[str] | str,
    hue: str | None,
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    time: float,
    pool_match: GroupMatch = "exact",
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    palette_book: PaletteBook | None = None,
    agg: str = "mean",
    err: str = "sem",
    time_tolerance: float = 0.51,
    panel_by: _PanelBy = "channel",
    channel_select: str | None = None,
    file_by: _FileBy = "auto",
    show_legend: bool = False,
    legend_loc: str = "upper right",
) -> list[PlotFigure]:
    fig_kwargs = fig_kwargs or {}

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    if err not in {"sem", "iqr", "none"}:
        raise ValueError("err must be 'sem', 'iqr', or 'none'")
    if file_by not in {"auto", "channel"}:
        raise ValueError("snapshot_barplot: file_by supports only 'auto' or 'channel'")

    x_col = alias_column(df, x)
    hue_col = alias_column(df, hue) if hue else None
    group_col = alias_column(df, group_on) if group_on else None
    y_list = [y] if isinstance(y, str) else list(y)
    if panel_by == "channel" and channel_select:
        y_list = [channel_select]

    required = ["time", "channel", "value", "position", x_col]
    if hue_col:
        required.append(hue_col)
    if group_col:
        required.append(group_col)
    require_columns(df, required, where="snapshot_barplot")

    work = df.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")

    if panel_by == "channel":
        available = sorted(work["channel"].astype(str).unique().tolist())
        missing = [str(channel) for channel in y_list if str(channel) not in available]
        if missing:
            raise ValueError(f"snapshot_barplot: requested channels not found: {missing}. Available: {available}")
        work = work[work["channel"].astype(str).isin([str(channel) for channel in y_list])].copy()
    if warn_if_empty(work, where="snapshot_barplot", detail="after channel filter"):
        return []

    if panel_by in {"x", "group"} and not channel_select:
        raise ValueError("snapshot_barplot: channel_select is required when panel_by != 'channel'")
    if channel_select and panel_by in {"x", "group"}:
        available = sorted(work["channel"].astype(str).unique().tolist())
        if str(channel_select) not in available:
            raise ValueError(f"snapshot_barplot: channel_select {channel_select!r} not in data. Available: {available}")

    selected_channel_default = None
    if panel_by in {"x", "group"}:
        selected_channel_default = str(
            channel_select if channel_select else (y if isinstance(y, str) else (y_list[0] if y_list else None))
        )
        if not selected_channel_default:
            raise ValueError(
                f"snapshot_barplot: panel_by={panel_by!r} requires an explicit channel via channel_select or y."
            )

    key_cols = [column for column in [group_col, x_col, hue_col, "channel", "position"] if column]
    target_channels = list(y_list) if panel_by == "channel" else [selected_channel_default]
    selections = [
        select_snapshot_rows(
            df=work,
            target_time=float(time),
            keys=key_cols,
            channel=str(channel),
            tolerance=float(time_tolerance),
        )
        for channel in target_channels
    ]
    snapped_parts = [selection.rows for selection in selections if not selection.rows.empty]
    if not snapped_parts:
        logging.getLogger("reader").info(
            "[warn]snapshot_barplot[/warn] • no rows available at any time — skipping figure"
        )
        return []
    fallback_notes = [selection for selection in selections if selection.fell_back and not selection.rows.empty]
    if fallback_notes:
        preview = "; ".join(
            f"{target_channels[idx]}: times={selection.fallback_times_preview} Δ≈{float(selection.fallback_delta or 0.0):.2f} h"
            for idx, selection in enumerate(selections)
            if selection.fell_back and not selection.rows.empty
        )
        logging.getLogger("reader").info(
            "[warn]snapshot_barplot[/warn] • requested t=%.2f h; no rows within ±%.2f h — "
            "using nearest available per key (%s)",
            float(time),
            float(time_tolerance),
            preview,
        )
    snapped = pd.concat(snapped_parts, ignore_index=True)
    snapped["value"] = pd.to_numeric(snapped["value"], errors="coerce")
    times_used = pd.to_numeric(snapped["time"], errors="coerce").dropna()
    t_used = float(times_used.median()) if not times_used.empty else float(time)

    if group_col:
        mask = snapped[group_col].notna() & (
            ~snapped[group_col].astype(str).str.strip().str.lower().isin({"none", "nan", ""})
        )
        snapped = snapped.loc[mask].copy()
        if snapped.empty:
            return []

    base_group_cols = [column for column in [group_col, x_col, hue_col, "channel"] if column]
    seen_cols: set[str] = set()
    base_group_cols = [column for column in base_group_cols if not (column in seen_cols or seen_cols.add(column))]
    stats = summarize_snapshot_values(df=snapped, group_cols=base_group_cols, err=err)

    fig_groups = build_figure_groups(
        stats=stats,
        group_col=group_col,
        panel_by=panel_by,
        pool_sets=pool_sets,
        pool_match=pool_match,
    )

    iterate_channels = panel_by == "group" and file_by == "channel"
    channels_for_files: list[str | None] = (
        (y_list if iterate_channels else [None]) if y_list else ([None] if not iterate_channels else [])
    )
    if iterate_channels and not channels_for_files:
        return []

    figures: list[PlotFigure] = []
    for fig_label, members in fig_groups:
        for ch_for_file in channels_for_files:
            panels, selected_channel = resolve_panel_configuration(
                panel_by=panel_by,
                members=members,
                y_list=y_list,
                group_col=group_col,
                channel_select=channel_select,
                selected_channel_default=selected_channel_default,
                ch_for_file=ch_for_file,
                stats=stats,
                x_col=x_col,
            )
            rows, cols = best_subplot_grid(len(panels))
            hue_levels_union = order_levels(stats[hue_col].astype(str).unique().tolist()) if hue_col else ["_single"]
            colors = colors_for(max(1, len(hue_levels_union)), palette_book)
            color_map = {level: colors[index % len(colors)] for index, level in enumerate(hue_levels_union)}

            with use_style(rc=fig_kwargs.get("rc"), color_cycle=colors):
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
                axes = np.atleast_1d(axes).ravel()
                rasterize_zorder = fig_kwargs.get("rasterize_zorder", None)
                if rasterize_zorder is not None:
                    for ax in axes:
                        ax.set_rasterization_zorder(float(rasterize_zorder))

                if panel_by == "channel" and group_col and members and members[0] is not None:
                    fig.suptitle(f"{members[0]} • t={t_used:.2f} h", y=float(fig_kwargs.get("suptitle_y", 1.04)))
                elif panel_by == "group":
                    fig.suptitle(f"{selected_channel} • t={t_used:.2f} h", y=float(fig_kwargs.get("suptitle_y", 1.04)))
                else:
                    fig.suptitle(f"{fig_label} • t={t_used:.2f} h", y=float(fig_kwargs.get("suptitle_y", 1.04)))

                y_lo_glob, y_hi_glob = (
                    compute_shared_ylim(
                        stats=stats,
                        snapped=snapped,
                        panels=panels,
                        panel_by=panel_by,
                        selected_channel=str(selected_channel),
                        group_col=group_col,
                        x_col=x_col,
                        agg=agg,
                        err=err,
                    )
                    if selected_channel is not None
                    else (None, None)
                )

                for ax_idx, panel in enumerate(panels):
                    ax = axes[ax_idx]
                    if panel_by == "channel":
                        gval = members[0] if (group_col and members and members[0] is not None) else None
                        sbar = stats[stats["channel"].astype(str) == str(panel)].copy()
                        srep = snapped[snapped["channel"].astype(str) == str(panel)].copy()
                        if gval is not None:
                            sbar = sbar[sbar[group_col].astype(str) == str(gval)]
                            srep = srep[srep[group_col].astype(str) == str(gval)]
                        channel_key = str(panel)
                    elif panel_by == "group":
                        sbar = stats[
                            (stats["channel"].astype(str) == str(selected_channel))
                            & (stats[group_col].astype(str) == str(panel))
                        ].copy()
                        srep = snapped[
                            (snapped["channel"].astype(str) == str(selected_channel))
                            & (snapped[group_col].astype(str) == str(panel))
                        ].copy()
                        channel_key = str(selected_channel)
                    else:
                        sbar = stats[
                            (stats["channel"].astype(str) == str(selected_channel))
                            & (stats[x_col].astype(str) == str(panel))
                        ].copy()
                        srep = snapped[
                            (snapped["channel"].astype(str) == str(selected_channel))
                            & (snapped[x_col].astype(str) == str(panel))
                        ].copy()
                        channel_key = str(selected_channel)

                    if sbar.empty:
                        ax.set_visible(False)
                        continue

                    title = str(panel) if panel_by == "group" else None
                    draw_snapshot_panel(
                        ax,
                        snapped=srep,
                        stats=sbar,
                        x_col=x_col,
                        hue_col=hue_col,
                        agg=agg,
                        err=err,
                        palette_book=palette_book,
                        show_legend=show_legend,
                        legend_loc=legend_loc,
                        title=title,
                        ylabel=str(channel_key if panel_by != "channel" else panel),
                        color_map=(color_map if hue_col else None),
                    )
                    with suppress(Exception):
                        ax.set_box_aspect(1.0)

                for index in range(len(panels), len(axes)):
                    axes[index].set_visible(False)

                if y_hi_glob is not None:
                    for axis in axes[: len(panels)]:
                        if axis.get_visible():
                            axis.set_ylim(y_lo_glob if y_lo_glob is not None else axis.get_ylim()[0], y_hi_glob)

                stub = _figure_stub(
                    filename=filename,
                    panel_by=panel_by,
                    group_col=group_col,
                    fig_label=str(fig_label),
                    members=members,
                    selected_channel=str(selected_channel) if selected_channel is not None else None,
                    multiple_files=len(fig_groups) * len(channels_for_files) > 1,
                )
                figures.extend(emit_plot_figure(fig=fig, filename=stub, output_dir=output_dir, fig_kwargs=fig_kwargs))
    return figures
