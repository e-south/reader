from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from reader_workbench.plotting.sinks import PlotFigure
from reader_workbench.plotting.style import PaletteBook, use_style

from ..ordering import order_levels
from .common import (
    alias_column,
    best_subplot_grid,
    colors_for,
    exact_observation_counts,
    plot_figure,
    require_columns,
    select_exact_observations,
    warn_if_empty,
)
from .grouping import GroupMatch, resolve_groups


def _figure_groups(
    *,
    df: pd.DataFrame,
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch,
) -> list[tuple[str, list[str]]]:
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
    universe = order_levels(df[gcol].astype(str).unique().tolist())
    if pool_sets:
        resolved = resolve_groups(universe, pool_sets, match=pool_match)
        return resolved or [("all", universe)]
    return [(v, [v]) for v in universe]


def plot_distributions(
    *,
    df: pd.DataFrame,
    blanks: pd.DataFrame,
    channels: list[str],
    # grouping
    group_on: str | None = "design_id",
    pool_sets: list[dict[str, list[str]]] | None = None,
    pool_match: GroupMatch = "exact",
    # layout
    panel_by: str = "channel",  # "channel" (default) | "group"
    hue: str | None = None,
    legend_loc: str = "upper left",
    # style / output
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    palette_book: PaletteBook | None = None,
) -> list[PlotFigure]:
    """
    Distribution histograms:
      • Auto‑alias columns: prefers '<group_on>_alias' transparently.
      • `group_on` selects partitions; `pool_sets` names explicit collections.
      • Default: panel_by='channel' (one subplot per channel) and a separate
        output file for each `group_on` value (e.g., per design_id).
    """
    fig_kwargs = fig_kwargs or {}
    figures: list[PlotFigure] = []
    fill_alpha = float(fig_kwargs.get("kde_fill_alpha", 0.18))

    # --- resolve columns (assertive, no silent fallbacks) ---
    ch_list = [str(c) for c in channels]
    require_columns(df, ["channel", "value"], where="distributions")
    selected_work = df[df["channel"].astype(str).isin(ch_list)].copy()
    if warn_if_empty(selected_work, where="distributions", detail="after channel filter"):
        return []
    gcol = alias_column(selected_work, group_on) if group_on else None
    if gcol and gcol not in selected_work.columns:
        raise ValueError(f"distributions: missing group_on column {gcol!r}")
    if gcol:
        group_values = selected_work[gcol]
        valid_group = group_values.notna() & ~group_values.astype(str).str.strip().str.casefold().isin(
            {"", "nan", "none"}
        )
        selected_work = selected_work.loc[valid_group].copy()
        if warn_if_empty(selected_work, where="distributions", detail="after group_on identity filter"):
            return []

    work = selected_work
    work = select_exact_observations(work, where="distributions")
    if warn_if_empty(work, where="distributions", detail="after exact-observation filter"):
        return []

    selected_blanks = blanks
    blank_work = selected_blanks
    if not blanks.empty:
        require_columns(blanks, ["channel", "value"], where="distribution blanks")
        selected_blanks = blanks[blanks["channel"].astype(str).isin(ch_list)].copy()
        blank_work = select_exact_observations(selected_blanks, where="distribution blanks")

    if hue:
        hcol = alias_column(work, hue)
        if hcol not in work.columns:
            raise ValueError(f"distributions: missing hue column {hcol!r}")
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
            colors = colors_for(max(1, len(members)), palette_book)
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
                        hue_levels = order_levels(dch[hue_col].astype(str).unique().tolist())
                        colors = colors_for(max(1, len(hue_levels)), palette_book)
                        cmap = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels)}
                        place_legend_here = (not legend_shown) and (len(hue_levels) > 1)
                        for h in hue_levels:
                            dd = dch[dch[hue_col].astype(str) == h]
                            if dd.empty:
                                continue
                            sns.kdeplot(
                                data=dd,
                                x="value",
                                ax=ax,
                                lw=1.8,
                                fill=True,
                                alpha=fill_alpha,
                                common_norm=False,
                                # Only the legend host needs labeled artists.
                                label=(str(h) if place_legend_here else None),
                                color=cmap[h],
                            )
                        if place_legend_here:
                            ax.legend(loc=legend_loc, title=None)
                            legend_shown = True
                    else:
                        # Single overall KDE if no hue given
                        sns.kdeplot(data=dch, x="value", ax=ax, lw=1.8, fill=True, alpha=fill_alpha)

                    # optional blanks median
                    if not blank_work.empty:
                        b = blank_work[blank_work["channel"].astype(str) == ch]
                        if not b.empty:
                            med = float(pd.to_numeric(b["value"], errors="coerce").median())
                            ax.axvline(med, ls="--", lw=1.0, alpha=0.6)

                    ax.set_xlabel(str(ch))  # more informative than "value"
                    ax.set_ylabel("density")  # not "count"

                # hide extras if grid > panels
                for k in range(len(ch_list), len(axes)):
                    axes[k].set_visible(False)

                # Ensure user-specified filename remains unique per file
                stub = f"{filename}__{str(gcol) + '=' if gcol else ''}{label}" if filename else f"distrib__{label}"
                source_sub = selected_work.copy()
                if gcol and members != [None]:
                    source_sub = source_sub[source_sub[gcol].astype(str).isin(members)]
                observed_count, omitted_count = exact_observation_counts(
                    pd.concat([source_sub, selected_blanks], ignore_index=True)
                )
                description = None
                if omitted_count:
                    description = (
                        f"Selected measurements: {observed_count} observed, {omitted_count} "
                        "omitted as bounded or non-finite; bounded or non-finite rows were omitted "
                        "before density estimation."
                    )
                figures.append(
                    plot_figure(
                        fig=fig,
                        filename=stub,
                        fig_kwargs=fig_kwargs,
                        description=description,
                    )
                )

    else:  # panel_by == "group"
        if not gcol:
            raise ValueError("panel_by='group' requires 'group_on'")
        # Panels are individual group values (optionally restricted by pool_sets)
        # Single fixed channel required (or unambiguous)
        if len(ch_list) != 1:
            raise ValueError("panel_by='group' expects exactly one channel in 'channels'")
        ch = ch_list[0]
        # flatten members across all figure groups into a unique, ordered panel list
        members_union: list[str] = []
        seen: set[str] = set()
        for _, members in fig_groups:
            for v in members:
                if v is not None and v not in seen:
                    members_union.append(v)
                    seen.add(v)
        if not members_union:
            return []
        rows, cols = best_subplot_grid(len(members_union))

        sub = work[work["channel"].astype(str) == ch]
        if sub.empty:
            return []

        colors = colors_for(1, palette_book)
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
                if not blank_work.empty:
                    b = blank_work[blank_work["channel"].astype(str) == ch]
                    if not b.empty:
                        med = float(pd.to_numeric(b["value"], errors="coerce").median())
                        ax.axvline(med, ls="--", lw=1.0, alpha=0.6)
                ax.set_xlabel(str(ch))
                ax.set_ylabel("density")

            for k in range(len(members_union), len(axes)):
                axes[k].set_visible(False)

            stub = f"{filename}__{ch}" if filename else f"distrib__{ch}"
            observed_count, omitted_count = exact_observation_counts(
                pd.concat(
                    [
                        selected_work[selected_work["channel"].astype(str) == ch],
                        selected_blanks[selected_blanks["channel"].astype(str) == ch],
                    ],
                    ignore_index=True,
                )
            )
            description = None
            if omitted_count:
                description = (
                    f"Selected measurements: {observed_count} observed, {omitted_count} "
                    "omitted as bounded or non-finite; bounded or non-finite rows were omitted "
                    "before density estimation."
                )
            figures.append(
                plot_figure(
                    fig=fig,
                    filename=stub,
                    fig_kwargs=fig_kwargs,
                    description=description,
                )
            )
    return figures
