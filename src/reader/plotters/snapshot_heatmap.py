"""
--------------------------------------------------------------------------------
<reader project>
reader/plotters/snapshot_heatmap.py

Square heat-map of one snapshot (single time-point).

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import pandas as pd
import seaborn as sns

__all__ = ["plot_snapshot_heatmap"]

# ───────────────────────────── helpers ─────────────────────────────────────

def _closest_time(df: pd.DataFrame, ch: str, tgt: float) -> float:
    """Return the available time‑point closest to *tgt* for channel *ch*."""
    avail = df.loc[df["channel"] == ch, "time"].unique()
    if len(avail) == 0:
        raise ValueError(f"No data for channel {ch!r}")
    return float(min(avail, key=lambda t: abs(t - tgt)))


def _sanitize_time(t: float) -> str:
    """Return filename‑safe token (e.g. 1.25 → "1p25")."""
    tok = f"{t:g}".rstrip("0").rstrip(".")
    return tok.replace(".", "p") or "0"


def _resolve_cmap(cmap: Any | None) -> Any:
    """Return a usable Colormap from *cmap* (string | list | Colormap | None).

    * **String** – tries seaborn palette name then mpl colormap; if those fail and
      the string is a valid colour (e.g. "seagreen"), a two‑step gradient from
      white → colour is built.
    * **List/tuple** – treated as discrete palette → converted to cmap.
    * **None** – defaults to "rocket_r".
    """
    if cmap is None:
        cmap = "rocket_r"

    # Already a colormap object
    if hasattr(cmap, "__call__"):
        return cmap

    # Sequence of colours → continuous cmap
    if isinstance(cmap, (list, tuple)):
        return sns.color_palette(cmap, as_cmap=True)

    # String handling ------------------------------------------------------
    if isinstance(cmap, str):
        # 1) seaborn palette name
        try:
            return sns.color_palette(cmap, as_cmap=True)
        except (ValueError, TypeError):
            pass
        # 2) matplotlib colormap name
        try:
            return mplcm.get_cmap(cmap)
        except ValueError:
            pass
        # 3) treat string as single colour -> white→colour gradient
        try:
            color_rgba = mplcolors.to_rgba(cmap)
            return mplcolors.LinearSegmentedColormap.from_list("single", [(1,1,1,0), color_rgba])
        except ValueError as err:
            raise ValueError(f"Could not resolve cmap '{cmap}'.") from err

    raise TypeError("Unsupported cmap specification: " + repr(cmap))


# ───────────────────────────── main fn ─────────────────────────────────────

def plot_snapshot_heatmap(
    df: pd.DataFrame,
    blanks: pd.DataFrame,               # unused – kept for API symmetry
    output_dir: Union[str, Path],
    *,
    channel: str,                       # metric to colour tiles
    time: float,                        # requested snapshot (h)
    x: str = "treatment",              # column → X‑axis categories
    y: str = "genotype",               # column → Y‑axis categories
    order_x: Optional[List[str]] = None,
    order_y: Optional[List[str]] = None,
    cmap: Any | None = None,            # override via function arg if desired
    square: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fig_kwargs: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    **_,
) -> None:
    """Draw a heat‑map snapshot with user‑selectable colour‑map.

    In the YAML config you can set, e.g.:

    ```yaml
    fig:
      cmap: seagreen      # single colour → gradient (white → seagreen)
      cbar_shrink: 0.75   # adjust colour‑bar height (default 0.8)
      cbar_pad: 0.02      # spacing between heat‑map and colour‑bar
    ```
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_kwargs = fig_kwargs or {}
    sns.set_style(fig_kwargs.get("seaborn_style", "ticks"), rc=fig_kwargs.get("rc", {}))

    # — snapshot selection —
    sel_t = _closest_time(df, channel, time)
    if sel_t != time:
        warnings.warn(
            f"Requested {time} h but using closest available {sel_t} h for channel {channel}"
        )

    snap = df[(df["time"] == sel_t) & (df["channel"] == channel)].copy()
    if snap.empty:
        raise RuntimeError(f"No rows for channel={channel!r} at time={sel_t}")

    # — average replicates —
    snap_val = snap.groupby([y, x], as_index=False)["value"].mean()

    # — axis ordering —
    if order_x is None:
        order_x = sorted(snap_val[x].dropna().unique().tolist())
    if order_y is None:
        order_y = sorted(snap_val[y].dropna().unique().tolist())

    pivot = (
        snap_val.pivot(index=y, columns=x, values="value")
        .reindex(index=order_y, columns=order_x)
    )

    # — figure sizing —
    n_y, n_x = len(order_y), len(order_x)
    cell = fig_kwargs.get("cell_size", 0.5)
    fig_w = max(cell * n_x, 4)
    fig_h = max(cell * n_y, 4)
    if square:
        fig_w = fig_h = max(fig_w, fig_h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_kwargs.get("dpi", 300))

    # resolve colour‑map & colour‑bar kwargs --------------------------------
    cmap_final = _resolve_cmap(fig_kwargs.get("cmap", cmap))
    cbar_shrink = fig_kwargs.get("cbar_shrink", 0.8)
    cbar_pad = fig_kwargs.get("cbar_pad", 0.02)

    sns.heatmap(
        pivot,
        cmap=cmap_final,
        cbar_kws={"label": channel, "shrink": cbar_shrink, "pad": cbar_pad},
        linewidths=0.5,
        linecolor="white",
        square=square,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )

    # — styling tweaks —
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    ax.set_title(
        f"Compound-promoter crosstalk ({sel_t:g} h)",
        fontweight="bold",
        pad=12,
    )

    # — save figure —
    if filename is None:
        fname = f"snapshot_heatmap_{channel.replace('/', '_')}_t{_sanitize_time(time)}h.pdf"
    else:
        fname = filename

    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)
