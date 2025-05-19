"""
--------------------------------------------------------------------------------
<reader project>

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_barplot_snapshot(
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Union[str, Path],
    x: str,
    y: Union[str, List[str]],
    hue: str,
    groups: List[Dict[str, List[str]]],
    time: float = 0.0,
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
):
    """
    Grouped barplot snapshot

    - df: tidy DataFrame with 'time', 'channel' and 'value' columns
    - groups: list of {group_name: [genotypes...]} specifying panel order
    - y: single channel name or list of channel names to plot (one figure per y)
    - hue: column for bar and point colors
    - time: snapshot time (hours); picks closest if exact match not found
    - fig_kwargs: {figsize: [w,h], dpi: d, seaborn_style: ..., palette: ...}
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, message="errwidth")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ys = [y] if isinstance(y, str) else y

    style = fig_kwargs.get("seaborn_style", "ticks") if fig_kwargs else "ticks"
    palette_name = fig_kwargs.get("palette", "colorblind") if fig_kwargs else "colorblind"
    sns.set_style(style)

    # build a full palette
    all_hues = sorted(df[hue].dropna().unique())
    full_palette = dict(zip(all_hues, sns.color_palette(palette_name, len(all_hues))))

    n_groups = len(groups)
    cols = min(3, n_groups)
    rows = math.ceil(n_groups / cols)

    # helper to extract numeric (including decimal) prefix for sorting
    def _numeric_prefix(s: str) -> float:
        m = re.match(r"^(\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else float("inf")

    for ycol in ys:
        # 1) determine a common y-axis upper limit across all panels
        global_max = 0.0
        for grp in groups:
            genos = next(iter(grp.values()))
            sub = df[df["genotype"].isin(genos)]
            if "time" in sub:
                closest = min(sub["time"].unique(), key=lambda t: abs(t - time))
                sub = sub[sub["time"] == closest]
            sub_y = sub[sub["channel"] == ycol]
            if not sub_y.empty:
                global_max = max(global_max, sub_y["value"].max())
        y_upper = global_max * 1.05 if global_max > 0 else 1.0

        safe_y = ycol.replace("/", "_")
        figsize = tuple(fig_kwargs.get("figsize", [5 * cols, 4 * rows])) if fig_kwargs else (5 * cols, 4 * rows)
        dpi = fig_kwargs.get("dpi", 300) if fig_kwargs else 300

        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, constrained_layout=True)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, grp in enumerate(groups):
            name, genos = next(iter(grp.items()))
            ax = axes_flat[idx]

            sub = df[df["genotype"].isin(genos)]
            if "time" in sub:
                closest = min(sub["time"].unique(), key=lambda t: abs(t - time))
                sub = sub[sub["time"] == closest]
            else:
                closest = time

            sub_y = sub[sub["channel"] == ycol]
            unique_hues = sorted(sub_y[hue].dropna().unique(), key=_numeric_prefix)
            dodge_flag = len(unique_hues) > 1

            # draw bars in hue_order
            sns.barplot(
                data=sub_y,
                x=x, y="value", hue=hue,
                hue_order=unique_hues,
                palette={h: full_palette[h] for h in unique_hues},
                errorbar=("ci", 95), capsize=0.1, err_kws={"linewidth": 1},
                dodge=dodge_flag, ax=ax, legend=False,
            )

            # overlay points in same hue_order
            sns.stripplot(
                data=sub_y,
                x=x, y="value", hue=hue,
                hue_order=unique_hues,
                dodge=dodge_flag, jitter=0.1,
                palette={h: "white" for h in unique_hues},
                edgecolor="darkgray", linewidth=0.5, size=6,
                ax=ax, legend=False,
            )

            # styling
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("", fontweight="bold")  # remove x‐axis label
            ax.set_ylabel(ycol, fontweight="bold")
            ax.set_title(name, fontweight="bold", pad=6)
            ax.text(
                0.5, 0.98,
                f"{closest:.1f} h",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize="small", fontweight="normal",
            )
            ax.set_ylim(0, y_upper)

            # build & draw legend in same order
            legend_handles = [
                Patch(facecolor=full_palette[h], label=h)
                for h in unique_hues
            ]
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    loc="center left",
                    frameon=False,
                    fontsize="x-small",
                    title=None,
                )

        # hide any extra subplots
        for extra_ax in axes_flat[n_groups:]:
            extra_ax.axis("off")

        out_file = filename or f"grouped_barplot_snapshot_{safe_y}.pdf"
        fig.savefig(out / out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
