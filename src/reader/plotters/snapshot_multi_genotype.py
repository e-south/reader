"""
--------------------------------------------------------------------------------
<reader project>
reader/plotters/snapshot_multi_genotype.py

Snapshot bar-plots, with nested panels if a group contains ≥2 genotypes.

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
from matplotlib.gridspec import GridSpec

from reader.utils.plot_style import PaletteBook

__all__ = ["plot_snapshot_multi_genotype"]

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="errwidth")

# ────────────────────────────── helpers ───────────────────────────────────────

def _closest_time(df: pd.DataFrame, ch: str, tgt: float) -> float:
    """Return the available time-point closest to *tgt* for channel *ch*."""
    avail = df.loc[df["channel"] == ch, "time"].unique()
    if len(avail) == 0:
        raise ValueError(f"No data for channel {ch!r}")
    return min(avail, key=lambda t: abs(t - tgt))


def _ymax(df: pd.DataFrame, genos: List[str], sel_t: float, ch: str) -> float:
    """
    Maximum finite y-value for the given *genos* at *sel_t* & channel *ch*.
    Returns 0.0 when data are absent or all-NaN.
    """
    slab = df[
        (df["genotype"].isin(genos))
        & (df["time"] == sel_t)
        & (df["channel"] == ch)
    ]
    vmax = slab["value"].max(skipna=True) if not slab.empty else 0.0
    return vmax if math.isfinite(vmax) else 0.0


def _numeric_prefix(s: str) -> float:
    """Extract leading numeric value from string for ordering; fallback to +∞."""
    m = re.match(r"^([-+]?\d*\.?\d+)", s)
    return float(m.group(1)) if m else float("inf")


def _sanitize_time(t: float) -> str:
    """Return a clean token for *t* suitable for filenames (e.g. 1.25 → '1p25')."""
    token = f"{t:g}".rstrip("0").rstrip(".")  # avoid scientific notation & trailing 0s
    return token.replace(".", "p") or "0"


# ─────────────────────────────── main ────────────────────────────────────────

def plot_snapshot_multi_genotype(
    df: pd.DataFrame,
    blanks: pd.DataFrame,  # unused – kept for API symmetry
    output_dir: Union[str, Path],
    *,
    channels: Optional[List[str]] = None,  # retained for API compatibility
    x: str,
    y: Union[str, List[str]],
    hue: str,
    groups: List[Dict[str, List[str]]],
    time: float = 0.0,
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    palette_book: Optional[PaletteBook] = None,
    **_,
) -> None:
    """Draw snapshot bar-plots at *time* and embed *time* from the **config** in
    the output filename.

    * The **requested** time (from the config) is used for the filename so that
      names are predictable (e.g. `…_t14h.pdf`).
    * If the actual data lack that exact time-point the nearest available value
      is plotted, and a warning is issued, but the filename still reflects the
      requested snapshot.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ys = [y] if isinstance(y, str) else list(y)
    fig_kwargs = fig_kwargs or {}

    # seaborn style + rc overrides ------------------------------------------------
    style = fig_kwargs.get("seaborn_style", "ticks")
    rc = fig_kwargs.get("rc", {})
    sns.set_style(style, rc=rc)

    for ycol in ys:
        # ───────── determine which time-point to plot ──────────
        sel_t = _closest_time(df, ycol, time)
        if sel_t != time:
            warnings.warn(
                f"Requested {time} h but using closest available {sel_t} h for channel {ycol}"
            )

        # ────────── figure scaffold ──────────
        n_groups = len(groups)
        ncols = min(3, n_groups)
        nrows = math.ceil(n_groups / ncols)
        figsize = tuple(fig_kwargs.get("figsize", (4 * ncols, 3.5 * nrows)))
        dpi = fig_kwargs.get("dpi", 300)

        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs_root = GridSpec(nrows, ncols, figure=fig)

        # ────────── per-group panels ──────────
        for gi, grp in enumerate(groups):
            _, genos = next(iter(grp.items()))
            r, c = divmod(gi, ncols)

            grp_treats = sorted(
                df.loc[
                    (df["genotype"].isin(genos))
                    & (df["time"] == sel_t)
                    & (df["channel"] == ycol),
                    hue,
                ].dropna().unique(),
                key=_numeric_prefix,
            )

            # palette per group
            pal = (
                palette_book(grp_treats, key=f"snapshot/{ycol}/{gi}")
                if palette_book
                else dict(
                    zip(
                        grp_treats,
                        sns.color_palette(fig_kwargs.get("palette", "colorblind"), len(grp_treats)),
                    )
                )
            )

            ymax = max(_ymax(df, genos, sel_t, ycol), 0) * 1.05
            inner = gs_root[r, c].subgridspec(1, len(genos), wspace=0.02, hspace=0.0)

            first_ax: Optional[plt.Axes] = None
            for si, geno in enumerate(genos):
                ax = fig.add_subplot(inner[0, si], sharey=first_ax or None)
                slab = df[
                    (df["genotype"] == geno)
                    & (df["time"] == sel_t)
                    & (df["channel"] == ycol)
                ]
                if slab.empty:
                    ax.set_axis_off()
                    continue

                treats_here = [t for t in grp_treats if t in slab[hue].unique()]

                sns.barplot(
                    data=slab,
                    x=hue,
                    y="value",
                    order=treats_here,
                    hue=hue,
                    palette={t: pal[t] for t in treats_here},
                    dodge=False,
                    width=0.6,
                    capsize=0.1,
                    errorbar=("ci", 95),
                    err_kws={"linewidth": 1},
                    legend=False,
                    ax=ax,
                )
                sns.stripplot(
                    data=slab,
                    x=hue,
                    y="value",
                    order=treats_here,
                    color="white",
                    edgecolor="grey",
                    linewidth=0.4,
                    size=5,
                    jitter=0.1,
                    legend=False,
                    ax=ax,
                )

                ax.set_ylim(0, ymax)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_xlabel("")
                ax.yaxis.grid(True)

                if si == 0:
                    ax.set_ylabel(ycol, fontweight="bold")
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

                ax.set_title(geno, fontsize="small", fontweight="bold", pad=6)
                ax.tick_params(axis="x", rotation=45, labelsize="small", pad=6)
                for lbl in ax.get_xticklabels():
                    lbl.set_ha("right")

                first_ax = first_ax or ax

        # ────────── save figure ──────────
        if filename is None:
            time_token = _sanitize_time(time)  # ← use *requested* time for naming
            fname = f"snapshot_bar_{ycol.replace('/', '_')}_t{time_token}h.pdf"
        else:
            fname = filename

        out_path = out / fname
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
