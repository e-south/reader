"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotters/distributions.py

Density & blank distributions per channel

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

def plot_distributions(
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Union[str, Path],
    channels: List[str],
    channel_col: str = "channel",
    value_col: str = "value",
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
):
    """
    Squared density plots arranged in a 3xN grid (wrap after 3 columns).
    Legend shown only on the first subplot.
    """
    df = df.copy()
    blanks = blanks.copy()

    # Convert values to numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    blanks[value_col] = pd.to_numeric(blanks[value_col], errors='coerce')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Apply style overrides
    if fig_kwargs:
        if "seaborn_style" in fig_kwargs:
            sns.set_style(fig_kwargs.pop("seaborn_style"))
        if "palette" in fig_kwargs:
            sns.set_palette(fig_kwargs.pop("palette"))

    n = len(channels)
    # Compute grid: 3 columns, wrap into rows
    cols = 3
    rows = (n + cols - 1) // cols

    figsize = tuple(fig_kwargs.get("figsize", (4 * cols, 4 * rows))) if fig_kwargs else (4 * cols, 4 * rows)
    dpi     = fig_kwargs.get("dpi", 300) if fig_kwargs else 300

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axes_flat = axes.flatten()

    for idx, ch in enumerate(channels):
        ax = axes_flat[idx]
        data = df[df[channel_col] == ch].dropna(subset=[value_col])
        sns.kdeplot(data=data, x=value_col, ax=ax, label="Data", fill=True)

        bdata = blanks[blanks[channel_col] == ch].dropna(subset=[value_col])
        if not bdata.empty:
            sns.kdeplot(
                data=bdata,
                x=value_col,
                ax=ax,
                label="Blank",
                fill=False,
                linestyle="--",
            )

        # Square aspect ratio
        try:
            ax.set_box_aspect(1)
        except AttributeError:
            ax.set_aspect('equal', adjustable='box')

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(ch)
        ax.set_xlabel(value_col)

        # Show legend only on first subplot
        if idx == 0:
            ax.legend(frameon=False)
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    # Turn off any unused axes
    for ax in axes_flat[n:]:
        ax.axis('off')

    plt.tight_layout()

    out_file = filename or "distributions.pdf"
    fig.savefig(out / out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)