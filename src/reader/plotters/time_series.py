"""
--------------------------------------------------------------------------------
<reader project>

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import re
import math
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="The markers list has more values",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Glyph.*missing from font",
    category=UserWarning
)


def plot_time_series(
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Union[str, Path],
    x: str,
    y: List[str] | str,
    hue: str,
    groupby_col: Optional[str] = None,
    subplots: Optional[str] = None,
    groups: Optional[List[Dict[str, List[str]]]] = None,
    iterate_genotypes: bool = False,
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    channel_col: str = "channel",
    value_col: str = "value",
    add_sheet_line: bool = False,
    log_transform: List[str] | bool = False,
):
    """
    Plot time series with optional log2 transform on a selected subset of channels.

    Parameters
    ----------
    df : DataFrame of measurements
    blanks : DataFrame of blanks (unused here)
    output_dir : output directory path
    x : column name for the x-axis (e.g. 'time')
    y : one or more channels to plot
    hue : column name for grouping/hue
    subplots : 'group' for per-group panels, else single per channel
    groups : list of {group_name: [values]} to facet by group
    add_sheet_line : whether to draw vertical separators at sheet transitions
    log_transform : list of channel names to log2-transform, or False
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ensure y is a list
    ys = y if isinstance(y, list) else [y]
    dpi = fig_kwargs.get("dpi", 300) if fig_kwargs else 300

    # prepare style & palette
    sns.set_style(fig_kwargs.get("seaborn_style", "ticks") if fig_kwargs else "ticks")
    base_pal = fig_kwargs.get("palette", "colorblind") if fig_kwargs else "colorblind"
    levels = sorted(df[hue].dropna().unique())
    pal = dict(zip(levels, sns.color_palette(base_pal, len(levels))))

    # markers
    base_markers = ["o", "s", "^", "X", "D", "P", "*"]
    repeat = math.ceil(len(levels) / len(base_markers))
    markers = (base_markers * repeat)[: len(levels)]

    # compute sheet transitions
    transitions: List[float] = []
    if add_sheet_line and "sheet" in df.columns:
        mins = df.groupby("sheet")[x].min().sort_index()
        transitions = mins.iloc[1:].tolist()

    # interpret log_transform param
    log_list = log_transform if isinstance(log_transform, list) else []

    def get_figsize(n: int) -> tuple[float, float]:
        cols = min(3, n)
        rows = math.ceil(n / cols)
        if fig_kwargs and "figsize" in fig_kwargs:
            return tuple(fig_kwargs["figsize"])
        return (5 * cols, 4 * rows)

    def _numeric_prefix(s: str) -> float:
        m = re.match(r"^(\d+)", s)
        return float(m.group(1)) if m else float("inf")

    def style(ax: plt.Axes, col: str, data_sub: pd.DataFrame):
        try:
            ax.set_box_aspect(1)
        except Exception:
            ax.set_aspect("equal", adjustable="box")
        vals = data_sub[value_col]
        if not vals.empty:
            ax.set_ylim(vals.min(), vals.max())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(x, fontweight="bold")
        ylabel = f"log2({col})" if col in log_list else col
        ax.set_ylabel(ylabel, fontweight="bold")

    def mark(ax: plt.Axes):
        for t in transitions:
            ax.axvline(t, color="gray", linestyle="--", linewidth=1)

    # -------------------------------------------------------------------------
    # plotting: grouped subplots?
    # -------------------------------------------------------------------------
    if subplots == "group" and groups and groupby_col:
        for grp_map in groups:
            name, vals = next(iter(grp_map.items()))
            subdf = df[df[groupby_col].isin(vals)]
            n = len(ys)

            fig, axes = plt.subplots(
                math.ceil(n / min(3, n)),
                min(3, n),
                figsize=get_figsize(n),
                dpi=dpi,
                constrained_layout=True,
            )
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

            for i, col in enumerate(ys):
                ax = axes_flat[i]
                data_sub = subdf[subdf[channel_col] == col].copy()

                # optionally transform
                if col in log_list:
                    data_sub[value_col] = np.log2(data_sub[value_col].astype(float))

                sns.lineplot(
                    data=data_sub,
                    x=x,
                    y=value_col,
                    hue=hue,
                    palette=pal,
                    estimator="mean",
                    errorbar=("ci", 95),
                    ax=ax,
                    legend=False,
                )

                pts = (
                    data_sub.groupby([x, hue])[value_col]
                    .mean()
                    .reset_index()
                )
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
                    legend=(i == n - 1),
                    ax=ax,
                    s=30,
                    alpha=0.5,
                )

                style(ax, col, data_sub)
                if add_sheet_line:
                    mark(ax)

                if i == n - 1:
                    handles, labels = ax.get_legend_handles_labels()
                    pairs = list(zip(labels, handles))
                    pairs.sort(key=lambda lh: _numeric_prefix(lh[0]))
                    sorted_labels, sorted_handles = zip(*pairs) if pairs else (labels, handles)
                    ax.legend(
                        sorted_handles,
                        sorted_labels,
                        loc="upper left",
                        frameon=False,
                        fontsize="small",
                    )

            for ax in axes_flat[n:]:
                ax.axis("off")

            fig.suptitle(f"Group: {name}", fontweight="bold")
            out_file = filename or f"time_series_grouped_{name}.pdf"
            fig.savefig(out / out_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    # -------------------------------------------------------------------------
    # plotting: single panel per y
    # -------------------------------------------------------------------------
    else:
        for col in ys:
            fig, ax = plt.subplots(
                1,
                1,
                figsize=get_figsize(len(ys)),
                dpi=dpi,
                constrained_layout=True,
            )
            data_sub = df[df[channel_col] == col].copy()

            if col in log_list:
                data_sub[value_col] = np.log2(data_sub[value_col].astype(float))

            sns.lineplot(
                data=data_sub,
                x=x,
                y=value_col,
                hue=hue,
                palette=pal,
                estimator="mean",
                errorbar=("ci", 95),
                ax=ax,
                legend=False,
            )

            pts = (
                data_sub.groupby([x, hue])[value_col]
                .mean()
                .reset_index()
            )
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
                alpha=0.5,
            )

            style(ax, col, data_sub)
            if add_sheet_line:
                mark(ax)

            handles, labels = ax.get_legend_handles_labels()
            pairs = list(zip(labels, handles))
            pairs.sort(key=lambda lh: _numeric_prefix(lh[0]))
            sorted_labels, sorted_handles = zip(*pairs) if pairs else (labels, handles)
            ax.legend(
                sorted_handles,
                sorted_labels,
                loc="upper left",
                frameon=False,
                fontsize="small",
            )

            out_file = filename or f"time_series_{col}.pdf"
            fig.savefig(out / out_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
