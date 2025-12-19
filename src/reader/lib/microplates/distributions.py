"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/distributions.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base import (
    GroupMatch,
    alias_column,
    best_subplot_grid,
    resolve_groups,
    save_figure,
    smart_string_numeric_key,
)
from .style import PaletteBook, use_style


def _colors_for(n: int, palette_book: Optional[PaletteBook]) -> List[str]:
    if palette_book:
        pal = palette_book.colors(max(2, n))
        # avoid pure black as the only color in single‑trace figures
        if n == 1 and str(pal[0]).lower() in {"#000000", "black"}:
            return [pal[1]]
        return pal[:n]
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cyc:
        raise RuntimeError("No color cycle available; configure Matplotlib rc or provide a PaletteBook.")
    if n == 1 and str(cyc[0]).lower() in {"#000000", "black"} and len(cyc) > 1:
        return [cyc[1]]
    return cyc[:n]


def _figure_groups(
    *,
    df: pd.DataFrame,
    group_on: Optional[str],
    pool_sets: Optional[List[Dict[str, List[str]]]],
    pool_match: GroupMatch,
) -> List[Tuple[str, List[str]]]:
    """
    Return a list of (figure_label, [member_values]) to iterate files.
    - When pool_sets is provided, each declared set label becomes a file label,
      with its members listed.
    - Otherwise: one file per distinct value of group_on.
    - If group_on is None: single file ("all", [None]).
    """
    if not group_on:
        return [("all", [None])]
    gcol = str(group_on)
    universe = sorted(df[gcol].astype(str).unique().tolist(), key=smart_string_numeric_key)
    if pool_sets:
        resolved = resolve_groups(universe, pool_sets, match=pool_match)
        return resolved or [("all", universe)]
    return [(v, [v]) for v in universe]


def plot_distributions(
    *,
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    output_dir: Path | str,
    channels: List[str],
    # modern grouping knobs
    group_on: Optional[str] = "genotype",
    pool_sets: Optional[List[Dict[str, List[str]]]] = None,
    pool_match: GroupMatch = "exact",
    # layout
    panel_by: str = "channel",  # "channel" (default) | "group"
    hue: Optional[str] = None,
    legend_loc: str = "upper left",
    # style / output
    fig_kwargs: Optional[Dict] = None,
    filename: Optional[str] = None,
    palette_book: Optional[PaletteBook] = None,
) -> None:
    """
    Distribution histograms with modern semantics:
      • Auto‑alias columns: prefers '<group_on>_alias' transparently.
      • `group_on` + `pool_sets` replace legacy 'groupby_col' & 'groups'.
      • Default: panel_by='channel' (one subplot per channel) and a separate
        output file for each `group_on` value (e.g., per genotype).
    """
    fig_kwargs = fig_kwargs or {}
    fill_alpha = float(fig_kwargs.get("kde_fill_alpha", 0.18))

    # --- resolve columns (assertive, no silent fallbacks) ---
    ch_list = [str(c) for c in channels]
    work = df[df["channel"].astype(str).isin(ch_list)].copy()
    if work.empty:
        return
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value"])

    gcol = alias_column(work, group_on) if group_on else None
    if panel_by not in {"channel", "group"}:
        raise ValueError("panel_by must be 'channel' or 'group'")

    # --- figure groups (decides how many files we emit) ---
    fig_groups = _figure_groups(
        df=(work if not gcol else work.rename(columns={gcol: str(gcol)})),
        group_on=(str(gcol) if gcol else None),
        pool_sets=pool_sets,
        pool_match=pool_match,
    )

    # --- two modes: panel by channel (default) or by group value ---
    if panel_by == "channel":
        rows, cols = best_subplot_grid(len(ch_list))
        for label, members in fig_groups:
            legend_shown = False
            sub = work.copy()
            if gcol and members != [None]:
                sub = sub[sub[gcol].astype(str).isin(members)]
            if sub.empty:
                continue

            # overlay colors per member (multiple overlays if members>1)
            colors = _colors_for(max(1, len(members)), palette_book)
            with use_style(rc=fig_kwargs.get("rc"), color_cycle=colors):
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
                axes = np.atleast_1d(axes).ravel()
                fig.suptitle(str(label), y=float(fig_kwargs.get("suptitle_y", 1.04)))

                for j, ch in enumerate(ch_list):
                    if j >= len(axes):
                        break
                    ax = axes[j]
                    dch = sub[sub["channel"].astype(str) == ch]
                    if dch.empty:
                        ax.set_visible(False)
                        continue

                    if hue:
                        hue_col = alias_column(dch, hue)
                        hue_levels = sorted(dch[hue_col].astype(str).unique().tolist(), key=smart_string_numeric_key)
                        colors = _colors_for(max(1, len(hue_levels)), palette_book)
                        cmap = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels)}
                        place_legend_here = (not legend_shown) and (len(hue_levels) > 1)
                        for h in hue_levels:
                            dd = dch[dch[hue_col].astype(str) == h]
                            if dd.empty:
                                continue
                            sns.kdeplot(
                                data=dd, x="value",
                                ax=ax, lw=1.8, fill=True, alpha=fill_alpha,
                                common_norm=False,
                                # Only the legend host needs labeled artists.
                                label=(str(h) if place_legend_here else None),
                                color=cmap[h]
                            )
                        if place_legend_here:
                            ax.legend(loc=legend_loc, title=None)
                            legend_shown = True
                    else:
                        # Single overall KDE if no hue given
                        sns.kdeplot(data=dch, x="value", ax=ax, lw=1.8, fill=True, alpha=fill_alpha)

                    # optional blanks median
                    if not blanks.empty:
                        b = blanks[blanks["channel"].astype(str) == ch]
                        if not b.empty:
                            med = float(pd.to_numeric(b["value"], errors="coerce").median())
                            ax.axvline(med, ls="--", lw=1.0, alpha=0.6)

                    ax.set_xlabel(str(ch))     # more informative than "value"
                    ax.set_ylabel("density")   # not "count"

                # hide extras if grid > panels
                for k in range(len(ch_list), len(axes)):
                    axes[k].set_visible(False)

                ext = str(fig_kwargs.get("ext", "pdf")).lower()
                # Ensure user-specified filename remains unique per file
                if filename:
                    stub = f"{filename}__{(str(gcol) + '=' if gcol else '')}{label}"
                else:
                    stub = f"distrib__{label}"
                save_figure(fig, Path(output_dir), stub, ext=ext)
                plt.close(fig)

    else:  # panel_by == "group"
        if not gcol:
            raise ValueError("panel_by='group' requires 'group_on'")
        # Panels are individual group values (optionally restricted by pool_sets)
        # Single fixed channel required (or unambiguous)
        if len(ch_list) != 1:
            raise ValueError("panel_by='group' expects exactly one channel in 'channels'")
        ch = ch_list[0]
        # flatten members across all figure groups into a unique, ordered panel list
        members_union: List[str] = []
        seen: set[str] = set()
        for _, members in fig_groups:
            for v in members:
                if v is not None and v not in seen:
                    members_union.append(v)
                    seen.add(v)
        if not members_union:
            return
        rows, cols = best_subplot_grid(len(members_union))

        sub = work[work["channel"].astype(str) == ch]
        if sub.empty:
            return

        colors = _colors_for(1, palette_book)
        with use_style(rc=fig_kwargs.get("rc"), color_cycle=colors):
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            fig.suptitle(str(ch), y=float(fig_kwargs.get("suptitle_y", 1.04)))
            for j, gv in enumerate(members_union):
                ax = axes[j]
                dd = sub[sub[gcol].astype(str) == str(gv)]
                vals = pd.to_numeric(dd["value"], errors="coerce").dropna()
                if vals.empty:
                    ax.set_visible(False)
                    continue
                sns.kdeplot(data=dd, x="value", ax=ax, lw=1.8, fill=True, alpha=fill_alpha)
                if not blanks.empty:
                    b = blanks[blanks["channel"].astype(str) == ch]
                    if not b.empty:
                        med = float(pd.to_numeric(b["value"], errors="coerce").median())
                        ax.axvline(med, ls="--", lw=1.0, alpha=0.6)
                ax.set_xlabel(str(ch))
                ax.set_ylabel("density")

            for k in range(len(members_union), len(axes)):
                axes[k].set_visible(False)

            ext = str(fig_kwargs.get("ext", "pdf")).lower()
            if filename:
                stub = f"{filename}__{ch}"
            else:
                stub = f"distrib__{ch}"
            save_figure(fig, Path(output_dir), stub, ext=ext)
            plt.close(fig)
