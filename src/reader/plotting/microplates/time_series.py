"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotting/microplates/time_series.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .base import (
    GroupMatch,
    alias_column,
    best_subplot_grid,
    pretty_name,
    resolve_groups,
    save_figure,
)
from .style import PaletteBook, use_style

# Fixed palette of distinct marker shapes (mapped 1:1 to hue levels)
_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "H"]


def _maybe_log(ax, enable: bool):
    if enable:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        if ymin <= 0:
            ax.set_ylim(bottom=max(1e-12, ymin))


def plot_time_series(
    *,
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir,
    x: str,
    y: list[str] | None,
    hue: str,
    channels: list[str] | None,
    subplots: str | None = None,  # kept for API parity (ignored: always subplots per channel)
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
    legend_loc: str = "upper right",
    show_replicates: bool = False,
) -> list[Path]:
    """
    Time-series plotting with one figure per group (default: per design_id[(_alias)]),
    subplots across channels, mean lines with CI bands, and *vertical gray background
    grid bands* behind the time axis (alternating between tick intervals).
    """
    xcol = alias_column(df, x)
    line_alpha = float((fig_kwargs or {}).get("line_alpha", 0.85))
    mean_marker_alpha = float((fig_kwargs or {}).get("mean_marker_alpha", 0.75))
    replicate_alpha = float((fig_kwargs or {}).get("replicate_alpha", 0.30))

    # Channel roster
    y_feats = (list(y) if y else list(channels or [])) or sorted(df["channel"].astype(str).unique().tolist())

    # Base frame
    base = df.copy()
    base["value"] = pd.to_numeric(base["value"], errors="coerce")
    if time_window:
        lo, hi = float(time_window[0]), float(time_window[1])
        base = base[
            (pd.to_numeric(base[xcol], errors="coerce") >= lo) & (pd.to_numeric(base[xcol], errors="coerce") <= hi)
        ].copy()
    if base.empty:
        return []

    group_col = alias_column(base, group_on) if group_on else None
    hue_col = alias_column(base, hue)
    # Drop None-like labels
    if group_col:
        mask = base[group_col].notna() & (
            ~base[group_col].astype(str).str.strip().str.lower().isin({"none", "nan", ""})
        )
        base = base.loc[mask].copy()
    if group_col:
        universe = base[group_col].astype(str).unique().tolist()
        fig_groups = (
            resolve_groups(universe, pool_sets, match=pool_match) if pool_sets else [(g, [g]) for g in universe]
        )
    else:
        fig_groups = [("all", [None])]

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
        cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not cyc:
            raise RuntimeError("No color cycle available; configure Matplotlib rc or provide a PaletteBook.")
        if n == 1 and str(cyc[0]).lower() in {"#000000", "black"} and len(cyc) > 1:
            return [cyc[1]]
        return cyc[:n]

    # Per-figure drawing
    saved: list[Path] = []
    for label, members in fig_groups:
        d = base.copy()
        if group_col and members != [None]:
            d = d[d[group_col].astype(str).isin(members)]
        if d.empty:
            continue

        hue_levels = list(d[hue_col].astype(str).unique())
        marker_map = {h: _MARKERS[i % len(_MARKERS)] for i, h in enumerate(hue_levels)}
        colors = _colors(len(hue_levels))
        color_map = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels)}
        rows, cols = best_subplot_grid(len(y_feats))

        with use_style(rc=(fig_kwargs or {}).get("rc"), color_cycle=colors):
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()

            # Optional rasterization threshold for heavy artists (replicate dots)
            rz = (fig_kwargs or {}).get("rasterize_zorder", None)
            if rz is not None:
                for ax in axes:
                    ax.set_rasterization_zorder(float(rz))

            fig.suptitle(f"{label}", y=1.04, fontweight="bold")

            for idx, ch in enumerate(y_feats):
                ax = axes[idx]
                sub = d[d["channel"].astype(str) == ch].copy()
                if sub.empty:
                    ax.set_visible(False)
                    continue

                # Start clean; show horizontal + vertical major grid lines
                ax.grid(False)
                ax.yaxis.grid(True, which="major")
                ax.xaxis.grid(True, which="major")

                # Optional replicate dots
                if show_replicates:
                    for h in hue_levels:
                        rr = sub[sub[hue_col].astype(str) == h]
                        ax.scatter(
                            rr[xcol],
                            rr["value"],
                            s=18,
                            alpha=replicate_alpha,
                            zorder=3,  # zorder=3 to rasterize if rz > 2
                            linewidths=0.0,
                            edgecolors="none",  # ← no marker edge
                            marker=marker_map[h],
                            c=color_map[h],
                        )

                sns.lineplot(
                    data=sub,
                    x=xcol,
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

                # Mean points (bold)
                means = sub.groupby([hue_col, xcol], dropna=False)["value"].mean().reset_index()
                for h in hue_levels:
                    mm = means[means[hue_col].astype(str) == h]
                    ax.scatter(
                        mm[xcol],
                        mm["value"],
                        s=36,
                        zorder=2.5,
                        marker=marker_map[h],
                        alpha=mean_marker_alpha,
                        edgecolors="none",
                        linewidths=0.0,  # ← remove white edge
                        c=color_map[h],
                    )

                # Sheet markers
                if sheet_lines:
                    style = {"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.8, "alpha": 0.9, "zorder": 0.5}
                    style.update(sheet_line_kwargs or {})
                    for sx in sheet_lines:
                        ax.axvline(float(sx), **style)

                # Log selection
                if isinstance(log_transform, list):
                    ax.set_yscale("log" if (ch in log_transform) else "linear")
                else:
                    _maybe_log(ax, bool(log_transform))

                # Labels & legend
                ax.set_xlabel("Time (h)" if str(x).lower() == "time" else pretty_name(str(xcol)))
                ax.set_ylabel(str(ch))
                with suppress(Exception):
                    ax.set_box_aspect(1.0)

                # Custom legend — place it on the first subplot only
                if idx == 0:
                    handles = [
                        Line2D(
                            [0],
                            [0],
                            color=color_map[h],
                            marker=marker_map[h],
                            markersize=7,
                            linestyle="-",
                            linewidth=1.8,
                            alpha=mean_marker_alpha,
                            label=str(h),
                        )
                        for h in hue_levels
                    ]
                    ax.legend(handles=handles, loc=legend_loc, title=None)

            # Hide extra axes if any
            for j in range(len(y_feats), len(axes)):
                axes[j].set_visible(False)

            # Allow file type override via fig.ext ("pdf" | "png" | "svg", etc.)
            ext = str((fig_kwargs or {}).get("ext", "pdf")).lower()
            stub = f"ts__{label}"
            saved.append(save_figure(fig, Path(output_dir), stub, ext=ext))
            plt.close(fig)
    return saved
