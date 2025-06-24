"""
--------------------------------------------------------------------------------
<reader project>
reader/plotters/snapshot_multi_feature.py

Draw (optionally multi-metric) bar/strip plots *per* genotype group.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = ["plot_snapshot_multi_feature"]

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="errwidth")


def _sort_treatments(labels: List[str]) -> List[str]:
    """
    Sort labels by their first numeric component (ascending), then
    non-numeric labels alphabetically.
    """
    def key(lbl: str):
        m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", lbl)
        if m:
            return (0, float(m.group(1)))
        return (1, lbl.lower())

    return sorted(labels, key=key)


def plot_snapshot_multi_feature(
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    channels: Optional[List[str]] = None,
    x: str,
    y: Union[str, List[str]],
    hue: str,
    groups: List[Dict[str, List[str]]],
    time: float = 0.0,
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    palette_book=None,
    **_kwargs,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_kwargs = fig_kwargs or {}
    rc = fig_kwargs.get("rc", {})
    sns.set_style(fig_kwargs.get("seaborn_style", "ticks"), rc=rc)

    metrics = [y] if isinstance(y, str) else list(y)
    gray = "#BFBFBF"

    for grp in groups:
        name, genos = next(iter(grp.items()))
        subdf = df[df[x].isin(genos)]
        if subdf.empty:
            warnings.warn(f"No data for group '{name}'")
            continue

        # numeric-asc sort by extracted number in label
        treatments = _sort_treatments(subdf[hue].dropna().astype(str).unique())
        pal = {t: gray for t in treatments}

        n = len(metrics)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        figsize = tuple(fig_kwargs.get("figsize", [5 * cols, 4 * rows]))
        dpi = fig_kwargs.get("dpi", 300)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, constrained_layout=True)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            data_all = subdf[subdf["channel"] == metric]
            if data_all.empty:
                ax.axis("off")
                continue

            sel_t = data_all.loc[(data_all["time"] - time).abs().idxmin(), "time"]
            sel_data = data_all[data_all["time"] == sel_t]

            sns.barplot(
                data=sel_data,
                x=hue,
                y="value",
                hue=hue,
                order=treatments,
                hue_order=treatments,
                palette=pal,
                dodge=False,
                capsize=0.1,
                err_kws={"linewidth": 1},
                ax=ax,
                legend=False,
            )
            sns.stripplot(
                data=sel_data,
                x=hue,
                y="value",
                order=treatments,
                color="white",
                edgecolor="0.6",
                size=5,
                linewidth=0.4,
                jitter=True,
                ax=ax,
                legend=False,
                zorder=10,
            )

            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel(metric, fontweight="bold")
            ax.set_xlabel("")
            ax.set_xticklabels(treatments, rotation=45, ha="right")

        for extra in axes_flat[len(metrics):]:
            extra.axis("off")

        fig.suptitle(f"{name}  (t ≈ {time:.2f} h)", fontweight="bold")
        out_file = filename or f"grouped_bar_by_group_{name}.pdf"
        fig.savefig(out / out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)