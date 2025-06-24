"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotters/distributions.py

Density & blank distributions per channel

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

__all__ = ["plot_distributions"]


def _build_group_column(
    df: pd.DataFrame,
    *,
    groups: List[Dict[str, List[str]]],
    groupby_col: str,
    new_col: str = "__group",
) -> pd.DataFrame:
    mapping: Dict[str, str] = {}
    for grp in groups:
        label, members = next(iter(grp.items()))
        for m in members:
            mapping[m] = label

    out = df.copy()
    out[new_col] = out[groupby_col].map(mapping)
    return out.dropna(subset=[new_col])


def _get_palette(
    labels: List[str],
    *,
    base_palette: str,
) -> Dict[str, tuple[float, float, float]]:
    # simple fallback palette from seaborn
    colours = sns.color_palette(base_palette, len(labels))
    return dict(zip(labels, colours))


def plot_distributions(
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    channels: List[str],
    groups: Optional[List[Dict[str, List[str]]]] = None,
    groupby_col: str = "genotype",
    channel_col: str = "channel",
    value_col: str = "value",
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    **_,  # ignore extra args
) -> None:
    """
    KDE density grid; hue = *group* if YAML `groups:` supplied.

    • One grid of 3-column subplots, one per channel.
    • x-axis label = channel name.
    • Single legend in the top-right of the first subplot.
    • Figure suptitle = PDF filename.
    """
    fig_kwargs = fig_kwargs or {}
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Prepare data
    df = df.copy()
    blanks = blanks.copy()
    df[value_col]     = pd.to_numeric(df[value_col], errors="coerce")
    blanks[value_col] = pd.to_numeric(blanks[value_col], errors="coerce")

    # Determine hue levels
    hue_col = "__group"
    if groups:
        df     = _build_group_column(df, groups=groups, groupby_col=groupby_col, new_col=hue_col)
        blanks = _build_group_column(blanks, groups=groups, groupby_col=groupby_col, new_col=hue_col)
        hue_labels = sorted(df[hue_col].unique())
    else:
        # fallback: just distinguish Data vs Blank
        df[hue_col]     = "Data"
        blanks[hue_col] = "Blank"
        hue_labels = ["Data", "Blank"]

    # Build a simple palette
    palette = _get_palette(
        labels=hue_labels,
        base_palette=fig_kwargs.get("palette", "colorblind"),
    )

    # Figure layout
    sns.set_style(fig_kwargs.get("seaborn_style", "ticks"))
    n = len(channels)
    cols = 3
    rows = (n + cols - 1) // cols
    figsize = tuple(fig_kwargs.get("figsize", (4*cols, 4*rows)))
    dpi     = fig_kwargs.get("dpi", 300)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axes_flat = axes.flatten()

    # Plot each channel
    for idx, ch in enumerate(channels):
        ax = axes_flat[idx]
        data_sub  = df    [df   [channel_col] == ch]
        blank_sub = blanks[blanks[channel_col] == ch]

        # filled curves for data groups
        sns.kdeplot(
            data=data_sub,
            x=value_col,
            hue=hue_col,
            palette=palette,
            fill=True,
            common_norm=False,
            linewidth=1,
            alpha=0.4,
            ax=ax,
            legend=False,
        )

        # dashed outlines for blanks (same hue mapping)
        if not blank_sub.empty:
            sns.kdeplot(
                data=blank_sub,
                x=value_col,
                hue=hue_col,
                palette=palette,
                fill=False,
                common_norm=False,
                linestyle="--",
                linewidth=1,
                alpha=0.8,
                ax=ax,
                legend=False,
            )

        # clean styling
        try:
            ax.set_box_aspect(1)
        except AttributeError:
            ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(ch)
        ax.set_ylabel("")

    # turn off any extra axes
    for ax in axes_flat[n:]:
        ax.axis("off")

    # build manual legend handles
    legend_handles = [
        Line2D([0], [0], color=palette[label], lw=2, alpha=0.6, label=label)
        for label in hue_labels
    ]
    # draw legend in first subplot
    axes_flat[0].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize="small",
        title=None,
    )

    # supertitle = filename
    out_file = (filename or "distributions.pdf")

    plt.tight_layout()
    fig.savefig(out_dir / out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)