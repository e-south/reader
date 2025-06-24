"""
--------------------------------------------------------------------------------
<reader project>
reader/plotters/time_series.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from reader.utils.plot_style import PaletteBook

__all__ = ["plot_time_series"]


def plot_time_series(
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    channels: Optional[List[str]] = None,
    x: str,
    y: Optional[Union[List[str], str]] = None,
    hue: str,
    groupby: Optional[str] = None,
    subplots: Optional[str] = None,
    groups: Optional[List[Dict[str, List[str]]]] = None,
    iterate_genotypes: bool = False,
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    channel_col: str = "channel",
    value_col: str = "value",
    add_sheet_line: bool = False,
    log_transform: Union[List[str], bool] = False,
    time_window: Optional[List[float]] = None,
    palette_book: Optional[PaletteBook] = None,
    **_kwargs,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ─── time window filter ─────────────────────────────────────────────
    if time_window is not None:
        try:
            start, end = time_window
            df = df[df[x].between(start, end)]
            blanks = blanks[blanks[x].between(start, end)]
        except Exception:
            warnings.warn(f"Invalid time_window: {time_window}, skipping filter.")

    # ─── determine y-columns ────────────────────────────────────────────
    if y is None:
        if channels is None:
            raise ValueError("Must supply either 'y' or 'channels' for time_series")
        ys = channels
    else:
        ys = [y] if isinstance(y, str) else y

    # ─── style & rc ────────────────────────────────────────────────────
    fig_kwargs = fig_kwargs or {}
    rc = fig_kwargs.get("rc", {})
    dpi = fig_kwargs.get("dpi", 300)
    sns.set_style(fig_kwargs.get("seaborn_style", "ticks"), rc=rc)

    # ─── sheet transition lines ─────────────────────────────────────────
    base_markers = ["o", "s", "^", "X", "D", "P", "*"]
    transitions: List[float] = []
    if add_sheet_line and "sheet" in df.columns:
        mins = df.groupby("sheet")[x].min().sort_index()
        transitions = mins.iloc[1:].tolist()

    log_list = log_transform if isinstance(log_transform, list) else []

    # ─── helper to size subplots ───────────────────────────────────────
    def get_figsize(n: int) -> tuple[float, float]:
        cols = min(3, n)
        rows = math.ceil(n / cols)
        return tuple(fig_kwargs.get("figsize", (5 * cols, 4 * rows)))

    def _numeric_prefix(s: str) -> float:
        m = re.match(r"^(\d+\.?\d*)", str(s))
        return float(m.group(1)) if m else float("inf")

    # ─── styling for each panel ────────────────────────────────────────
    def style(ax: plt.Axes, col: str, subdf: pd.DataFrame):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time (h)", fontweight="bold")
        ylabel = f"log2({col})" if col in log_list else col
        ax.set_ylabel(ylabel, fontweight="bold")
        vals = subdf[value_col]
        if not vals.empty:
            ax.set_ylim(vals.min(), vals.max())

    def mark(ax: plt.Axes):
        for t in transitions:
            ax.axvline(t, color="gray", linestyle="--", linewidth=1)

    # ─── FACETED BY GROUP ───────────────────────────────────────────────
    if subplots == "group" and groups and groupby:
        for grp_map in groups:
            name, vals = next(iter(grp_map.items()))
            subdf = df[df[groupby].isin(vals)]
            levels = sorted(subdf[hue].dropna().unique())
            pal = (
                palette_book(levels, key=f"time_series/{name}")
                if palette_book
                else dict(zip(levels, sns.color_palette(fig_kwargs.get("palette", "colorblind"), len(levels))))
            )
            markers = (base_markers * math.ceil(len(levels) / len(base_markers)))[: len(levels)]
            n = len(ys)

            fig, axes = plt.subplots(
                math.ceil(n / min(3, n)),
                min(3, n),
                figsize=get_figsize(n),
                dpi=dpi,
                constrained_layout=True,
            )
            axes_flat = axes.flatten()

            for i, col in enumerate(ys):
                ax = axes_flat[i]
                dsub = subdf[subdf[channel_col] == col].copy()
                if col in log_list:
                    dsub[value_col] = np.log2(dsub[value_col].astype(float))

                sns.lineplot(
                    data=dsub,
                    x=x,
                    y=value_col,
                    hue=hue,
                    palette=pal,
                    estimator="mean",
                    errorbar="sd",
                    ax=ax,
                    legend=False,
                )
                pts = dsub.groupby([x, hue])[value_col].mean().reset_index()
                if col in log_list:
                    pts[value_col] = np.log2(pts[value_col].astype(float))

                sns.scatterplot(
                    data=pts,
                    x=x,
                    y=value_col,
                    hue=hue,
                    style=hue,
                    palette=pal,
                    markers=markers,
                    legend=(i == 0),
                    ax=ax,
                    s=30,
                    alpha=0.6,
                )

                style(ax, col, dsub)
                if add_sheet_line:
                    mark(ax)

                if i == 0:
                    handles, labs = ax.get_legend_handles_labels()
                    pairs = sorted(zip(labs, handles), key=lambda lh: _numeric_prefix(lh[0]))
                    if pairs:
                        lbls, hnds = zip(*pairs)
                        ax.legend(hnds, lbls, loc="upper left", frameon=False, fontsize="small")

            for ax in axes_flat[n:]:
                ax.axis("off")

            fig.suptitle(name, fontweight="bold")
            out_file = filename or f"time_series_grouped_{name}.pdf"
            fig.savefig(out / out_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    # ─── SINGLE-PANEL FALLBACK ──────────────────────────────────────────
    else:
        for col in ys:
            subdf = df[df[channel_col] == col]
            levels = sorted(subdf[hue].dropna().unique())
            pal = (
                palette_book(levels, key=f"time_series/{col}")
                if palette_book
                else dict(zip(levels, sns.color_palette(fig_kwargs.get("palette", "colorblind"), len(levels))))
            )
            markers = (base_markers * math.ceil(len(levels) / len(base_markers)))[: len(levels)]

            fig, ax = plt.subplots(1, 1, figsize=get_figsize(len(ys)), dpi=dpi, constrained_layout=True)
            dsub = subdf.copy()
            if col in log_list:
                dsub[value_col] = np.log2(dsub[value_col].astype(float))

            sns.lineplot(
                data=dsub,
                x=x,
                y=value_col,
                hue=hue,
                palette=pal,
                estimator="mean",
                errorbar="sd",
                ax=ax,
                legend=False,
            )
            pts = dsub.groupby([x, hue])[value_col].mean().reset_index()
            if col in log_list:
                pts[value_col] = np.log2(pts[value_col].astype(float))

            sns.scatterplot(
                data=pts,
                x=x,
                y=value_col,
                hue=hue,
                style=hue,
                palette=pal,
                markers=markers,
                legend=True,
                ax=ax,
                s=30,
                alpha=0.6,
            )

            style(ax, col, dsub)
            if add_sheet_line:
                mark(ax)

            handles, labs = ax.get_legend_handles_labels()
            pairs = sorted(zip(labs, handles), key=lambda lh: _numeric_prefix(lh[0]))
            if pairs:
                lbls, hnds = zip(*pairs)
                ax.legend(hnds, lbls, loc="upper left", frameon=False, fontsize="small")

            out_file = filename or f"time_series_{col}.pdf"
            fig.savefig(out / out_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)