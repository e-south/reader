"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/snapshot_barplot.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import (
    GroupMatch,
    alias_column,
    best_subplot_grid,
    nearest_time_per_key,
    require_columns,
    resolve_groups,
    save_figure,
    smart_grouped_dose_key,
    smart_string_numeric_key,
    warn_if_empty,
)
from .style import PaletteBook, use_style

_SubplotBy = Literal["channel", "x", "group"]
_PanelBy = Literal["channel", "x", "group"]
_FileBy = Literal["auto", "group", "channel", "x"]

# ---------------------------- small helpers ----------------------------


def _order_levels(levels: list[str]) -> list[str]:
    # Prefer grouping by common prefix (text before the first number) and
    # ordering by normalized dose within that group. Falls back to numeric‑first ordering.
    try:
        return sorted(levels, key=smart_grouped_dose_key)
    except Exception:
        return sorted(levels, key=smart_string_numeric_key)


def _colors_for(n: int, palette_book: PaletteBook | None) -> list[str]:
    if palette_book:
        if n == 1:
            pal = palette_book.colors(2)
            return [pal[1]] if (pal and str(pal[0]).lower() in {"#000000", "black"}) else [pal[0]]
        return palette_book.colors(n)
    cyc = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cyc:
        raise RuntimeError("No color cycle available; configure Matplotlib rcParams or provide a PaletteBook.")
    if n == 1 and str(cyc[0]).lower() in {"#000000", "black"} and len(cyc) > 1:
        return [cyc[1]]
    return cyc[:n]


def _asym_yerr(lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    """Matplotlib bar(): asymmetric yerr is a 2×N array [lower; upper]."""
    return np.vstack([np.asarray(lower, float), np.asarray(upper, float)])


# ---------------------------- main function ----------------------------


def plot_snapshot_barplot(
    *,
    df: pd.DataFrame,
    output_dir,
    x: str,
    y: list[str] | str,
    hue: str | None,
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    time: float,
    pool_match: GroupMatch = "exact",
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    palette_book: PaletteBook | None = None,
    agg: str = "mean",  # default: mean
    err: str = "sem",  # default: sem  (also supports "iqr" | "none")
    time_tolerance: float = 0.51,
    panel_by: _PanelBy = "channel",
    channel_select: str | None = None,  # required when panel_by == "x"
    file_by: _FileBy = "auto",
    show_legend: bool = False,
    legend_loc: str = "upper right",
) -> None:
    """
    Snapshot barplot with replicate dots.

    Pragmatics:
      • Bar heights are computed from replicates (default: mean).
      • Error bars use Matplotlib's native yerr so whiskers are *anchored* to the bars.
      • Replicate dot clouds are overlaid *after* error bars.
      • IQR with agg='median' draws asymmetric whiskers (q1..median..q3).
        IQR with agg='mean' draws symmetric ±IQR/2 around the mean.

    Modes:
      - panel_by == "channel": one subplot per channel in `y`.
      - panel_by == "group":   one subplot per group value (from `group_on`), fixed channel.
      - panel_by == "x":       one subplot per x-level, fixed channel.
    """
    fig_kwargs = fig_kwargs or {}

    # Validate knobs early (assertive, no fallbacks)
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")
    if err not in {"sem", "iqr", "none"}:
        raise ValueError("err must be 'sem', 'iqr', or 'none'")

    # Resolve columns (prefer *_alias)
    x_col = alias_column(df, x)
    hue_col = alias_column(df, hue) if hue else None
    group_col = alias_column(df, group_on) if group_on else None
    y_list = [y] if isinstance(y, str) else list(y)
    required = ["time", "channel", "value", "position", x_col]
    if hue_col:
        required.append(hue_col)
    if group_col:
        required.append(group_col)
    require_columns(df, required, where="snapshot_barplot")

    # Base frame
    work = df.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")

    # Restrict to requested channels in the common case
    if panel_by == "channel":
        available = sorted(work["channel"].astype(str).unique().tolist())
        missing = [str(c) for c in y_list if str(c) not in available]
        if missing:
            raise ValueError(f"snapshot_barplot: requested channels not found: {missing}. Available: {available}")
        work = work[work["channel"].astype(str).isin([str(c) for c in y_list])].copy()
    if warn_if_empty(work, where="snapshot_barplot", detail="after channel filter"):
        return

    if panel_by in {"x", "group"} and not channel_select:
        raise ValueError("snapshot_barplot: channel_select is required when panel_by != 'channel'")
    if channel_select:
        available = sorted(work["channel"].astype(str).unique().tolist())
        if str(channel_select) not in available:
            raise ValueError(
                f"snapshot_barplot: channel_select {channel_select!r} not in data. Available: {available}"
            )

    # Pick nearest-time replicates per (groupby?, x, hue?, channel, position)
    key_cols: list[str] = [c for c in [group_col, x_col, hue_col, "channel", "position"] if c]
    snapped = nearest_time_per_key(work, target_time=float(time), keys=key_cols, tol=time_tolerance)
    if snapped.empty:
        log = logging.getLogger("reader")
        # Fall back to the nearest rows per key (no tolerance), with a clear warning.
        snapped = nearest_time_per_key(work, target_time=float(time), keys=key_cols, tol=float("inf"))
        if snapped.empty:
            # Truly nothing to plot at any time
            log.info("[warn]snapshot_barplot[/warn] • no rows available at any time — skipping figure")
            return
        # Summarize what we used
        times_used = pd.to_numeric(snapped["time"], errors="coerce").dropna()
        uniq = sorted(times_used.unique().tolist())
        # report a representative delta (median of used times vs target)
        t_rep = uniq[0] if len(uniq) == 1 else float(pd.Series(uniq).median())
        delta = abs(float(t_rep) - float(time))
        preview = ", ".join(f"{t:.2f}" for t in uniq[:6]) + (" …" if len(uniq) > 6 else "")
        log.info(
            "[warn]snapshot_barplot[/warn] • requested t=%.2f h; no rows within ±%.2f h — "
            "using nearest available per key (times=%s; Δ≈%.2f h)",
            float(time),
            float(time_tolerance),
            preview,
            float(delta),
        )

    snapped["value"] = pd.to_numeric(snapped["value"], errors="coerce")

    # Drop None-like group labels
    if group_col:
        mask = snapped[group_col].notna() & (
            ~snapped[group_col].astype(str).str.strip().str.lower().isin({"none", "nan", ""})
        )
        snapped = snapped.loc[mask].copy()
        if snapped.empty:
            return

    # -------------- aggregate once: heights + error stats from replicates --------------

    base_group_cols: list[str] = [c for c in [group_col, x_col, hue_col, "channel"] if c]

    # Aggregate on DataFrameGroupBy (named aggregations require DataFrame groupby)
    stats = (
        snapped.groupby(base_group_cols, dropna=False)
        .agg(
            n=("value", "count"),
            mean=("value", "mean"),
            median=("value", "median"),
            std=("value", "std"),
            sem=("value", "sem"),
        )
        .reset_index()
    )

    # Quantiles for IQR (merged later, only used when err="iqr")
    if err == "iqr":
        g_series = snapped.groupby(base_group_cols, dropna=False)["value"]
        q = g_series.quantile([0.25, 0.75]).unstack(-1).reset_index().rename(columns={0.25: "q1", 0.75: "q3"})
        stats = stats.merge(q, on=base_group_cols, how="left")

    def _row_for(
        tab: pd.DataFrame,
        *,
        x_val: str,
        ch: str,
        hue_val: str | None,
        group_val: str | None,
    ) -> pd.Series | None:
        sub = tab[(tab[x_col].astype(str) == str(x_val)) & (tab["channel"].astype(str) == str(ch))]
        if hue_col:
            sub = sub[sub[hue_col].astype(str) == str(hue_val)]
        if group_col and (group_val is not None):
            sub = sub[sub[group_col].astype(str) == str(group_val)]
        if sub.empty:
            return None
        return sub.iloc[0]

    # -------------- figure grouping --------------

    def _figure_groups() -> list[tuple[str, list[str]]]:
        """
        Returns a list of (figure_label, [members]) where `members` are group values.
        Semantics:
          • panel_by == "group" → one figure with panels over the union of selected groups.
          • otherwise           → one figure per group value (restricted to union if pool_sets present).
        """
        if not group_col:
            return [("all", [None])]

        universe = sorted(stats[group_col].astype(str).unique().tolist(), key=smart_string_numeric_key)
        if panel_by == "group":
            if pool_sets:
                resolved = resolve_groups(universe, pool_sets, match=pool_match)
                ordered: list[str] = []
                seen: set[str] = set()
                for _, vals in resolved:
                    for v in vals:
                        if v in universe and v not in seen:
                            ordered.append(v)
                            seen.add(v)
                return [("all", ordered or universe)]
            return [("all", universe)]

        # panel_by != "group": draw a separate figure per group value
        if pool_sets:
            resolved = resolve_groups(universe, pool_sets, match=pool_match)
            # Flatten union while preserving declared order
            union: list[str] = []
            seen: set[str] = set()
            for _, vals in resolved:
                for v in vals:
                    if v in universe and v not in seen:
                        union.append(v)
                        seen.add(v)
            return [(g, [g]) for g in (union or universe)]
        return [(g, [g]) for g in universe]

    fig_groups = _figure_groups()

    # -------------- figure iteration strategy --------------
    # Optional: one file per channel when comparing groups.
    iterate_channels = panel_by == "group" and file_by == "channel"
    channels_for_files: list[str | None] = (
        (y_list if iterate_channels else [None]) if y_list else ([None] if not iterate_channels else [])
    )
    if iterate_channels and not channels_for_files:
        # No channels available to iterate — nothing to draw.
        return

    for fig_label, members in fig_groups:
        for ch_for_file in channels_for_files:
            # Determine panels for this figure and the "selected_channel" when needed
            if panel_by == "channel":
                panels = y_list
                selected_channel: str | None = None
            elif panel_by == "group":
                if not group_col:
                    raise ValueError("panel_by='group' requires 'group_on'")
                # Choose the channel to visualize across all panels (groups).
                selected_channel = (str(ch_for_file) if ch_for_file is not None else None) or (
                    channel_select if channel_select else (y if isinstance(y, str) else (y_list[0] if y_list else None))
                )
                if not selected_channel:
                    raise ValueError(
                        "panel_by='group' requires an explicit channel: "
                        "set 'channel_select' or pass a single string to 'y'."
                    )
                panels = list(members)  # one panel per group value
            else:  # panel_by == "x"
                # Fix a channel and create one panel per x-level.
                selected_channel = (
                    channel_select if channel_select else (y if isinstance(y, str) else (y_list[0] if y_list else None))
                )
                if not selected_channel:
                    raise ValueError(
                        "panel_by='x' requires an explicit channel: "
                        "set 'channel_select' or pass a single string to 'y'."
                    )
                sub_stats = stats[stats["channel"].astype(str) == str(selected_channel)]
                x_levels = sub_stats[x_col].astype(str).unique().tolist()
                panels = _order_levels(x_levels)

            rows, cols = best_subplot_grid(len(panels))

            # Hue levels present union (for stable colors across the figure)
            hue_levels_union = (
                sorted(stats[hue_col].astype(str).unique().tolist(), key=smart_string_numeric_key)
                if hue_col
                else ["_single"]
            )
        colors = _colors_for(max(1, len(hue_levels_union)), palette_book)
        color_map = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels_union)}

        # Global width so every panel uses identical bar widths
        num_hues = len(hue_levels_union) if hue_col else 1
        has_hue_global = hue_col is not None and num_hues > 1
        width_global = 0.8 if not has_hue_global else min(0.85 / max(num_hues, 1), 0.8)

        with use_style(rc=fig_kwargs.get("rc"), color_cycle=colors):
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()

            # Optional rasterization threshold for heavy artists
            rz = fig_kwargs.get("rasterize_zorder", None)
            if rz is not None:
                for ax in axes:
                    ax.set_rasterization_zorder(float(rz))

            # Title
            if panel_by == "channel" and group_col and members and members[0] is not None:
                fig.suptitle(f"{members[0]} • t≈{float(time):.2f} h", y=float(fig_kwargs.get("suptitle_y", 1.04)))
            elif panel_by == "group":
                fig.suptitle(f"{selected_channel} • t≈{float(time):.2f} h", y=float(fig_kwargs.get("suptitle_y", 1.04)))
            else:
                fig.suptitle(f"{fig_label} • t≈{float(time):.2f} h", y=float(fig_kwargs.get("suptitle_y", 1.04)))

            # --- optional: compute global y-lims when all panels share a channel ---
            unify_same_channel = (panel_by in ("group", "x")) and len(panels) > 1
            y_lo_glob: float | None = None
            y_hi_glob: float | None = None
            if unify_same_channel:
                # Pre-compute across all panels before drawing
                def _collect_limits_for_subset(
                    sbar_sub: pd.DataFrame, srep_sub: pd.DataFrame
                ) -> tuple[float | None, float | None]:
                    vals = (
                        pd.to_numeric(sbar_sub[agg], errors="coerce").dropna()
                        if not sbar_sub.empty
                        else pd.Series([], dtype=float)
                    )
                    vmin = float(vals.min()) if not vals.empty else None
                    vmax = float(vals.max()) if not vals.empty else None
                    # include error band upper bound
                    if err == "sem" and "sem" in sbar_sub.columns:
                        top = (
                            pd.to_numeric(sbar_sub[agg], errors="coerce")
                            + pd.to_numeric(sbar_sub["sem"], errors="coerce")
                        ).dropna()
                        vmax = max(vmax or -np.inf, float(top.max())) if not top.empty else vmax
                    elif err == "iqr" and {"q1", "q3"}.issubset(sbar_sub.columns):
                        if agg == "median":
                            top = pd.to_numeric(sbar_sub["q3"], errors="coerce").dropna()
                            vmax = max(vmax or -np.inf, float(top.max())) if not top.empty else vmax
                        else:
                            half = 0.5 * (
                                pd.to_numeric(sbar_sub["q3"], errors="coerce")
                                - pd.to_numeric(sbar_sub["q1"], errors="coerce")
                            )
                            top = (pd.to_numeric(sbar_sub[agg], errors="coerce") + half).dropna()
                            vmax = max(vmax or -np.inf, float(top.max())) if not top.empty else vmax
                    # include replicate scatter extremum for safety
                    if not srep_sub.empty and "value" in srep_sub.columns:
                        rv = pd.to_numeric(srep_sub["value"], errors="coerce").dropna()
                        if not rv.empty:
                            vmin = min(vmin if vmin is not None else float("inf"), float(rv.min()))
                            vmax = max(vmax if vmax is not None else float("-inf"), float(rv.max()))
                    return vmin, vmax

                lo_vals: list[float] = []
                hi_vals: list[float] = []
                for panel in panels:
                    if panel_by == "group":
                        sbar_sub = stats[
                            (stats["channel"].astype(str) == str(selected_channel))
                            & (stats[group_col].astype(str) == str(panel))
                        ]
                        srep_sub = snapped[
                            (snapped["channel"].astype(str) == str(selected_channel))
                            & (snapped[group_col].astype(str) == str(panel))
                        ]
                    else:  # panel_by == "x"
                        sbar_sub = stats[
                            (stats["channel"].astype(str) == str(selected_channel))
                            & (stats[x_col].astype(str) == str(panel))
                        ]
                        srep_sub = snapped[
                            (snapped["channel"].astype(str) == str(selected_channel))
                            & (snapped[x_col].astype(str) == str(panel))
                        ]
                    vmin, vmax = _collect_limits_for_subset(sbar_sub, srep_sub)
                    if vmin is not None:
                        lo_vals.append(vmin)
                    if vmax is not None:
                        hi_vals.append(vmax)
                if hi_vals:
                    y_lo_glob = min(0.0, float(np.nanmin(lo_vals))) if lo_vals else 0.0
                    y_hi_glob = float(np.nanmax(hi_vals))
                    # gentle headroom
                    pad = 0.05 * (y_hi_glob - y_lo_glob if y_hi_glob > y_lo_glob else max(1.0, y_hi_glob or 1.0))
                    y_hi_glob = y_hi_glob + pad

            # ---------- panel loop ----------
            for ax_idx, panel in enumerate(panels):
                ax = axes[ax_idx]

                # Subset stats & replicates for this panel; determine x ordering
                if panel_by == "channel":
                    gval = members[0] if (group_col and members and members[0] is not None) else None
                    sbar = stats[stats["channel"].astype(str) == str(panel)].copy()
                    srep = snapped[snapped["channel"].astype(str) == str(panel)].copy()
                    if gval is not None:
                        sbar = sbar[sbar[group_col].astype(str) == str(gval)]
                        srep = srep[srep[group_col].astype(str) == str(gval)]
                    x_levels = sbar[x_col].astype(str).unique().tolist()
                    x_order = _order_levels(x_levels)
                    channel_key = str(panel)
                    group_val_for_panel = gval
                elif panel_by == "group":
                    sbar = stats[
                        (stats["channel"].astype(str) == str(selected_channel))
                        & (stats[group_col].astype(str) == str(panel))
                    ].copy()
                    srep = snapped[
                        (snapped["channel"].astype(str) == str(selected_channel))
                        & (snapped[group_col].astype(str) == str(panel))
                    ].copy()
                    x_levels = sbar[x_col].astype(str).unique().tolist()
                    x_order = _order_levels(x_levels)
                    channel_key = str(selected_channel)
                    group_val_for_panel = str(panel)
                else:  # panel_by == "x"
                    sbar = stats[
                        (stats["channel"].astype(str) == str(selected_channel))
                        & (stats[x_col].astype(str) == str(panel))
                    ].copy()
                    srep = snapped[
                        (snapped["channel"].astype(str) == str(selected_channel))
                        & (snapped[x_col].astype(str) == str(panel))
                    ].copy()
                    x_order = [str(panel)]
                    channel_key = str(selected_channel)
                    group_val_for_panel = None

                if sbar.empty:
                    ax.set_visible(False)
                    continue

                # Hue levels present in this panel (preserve stable order)
                if hue_col:
                    hue_levels = sorted(sbar[hue_col].astype(str).unique().tolist(), key=smart_string_numeric_key)
                else:
                    hue_levels = ["_single"]

                # Layout: centers for x categories, and per-hue offsets
                n_x = len(x_order)
                base_pos = np.arange(n_x, dtype=float)
                width = width_global
                # Center bars within each x slot even when a subset of hues is present in this panel
                n_here = len(hue_levels) if (hue_col and len(hue_levels) > 0) else 1
                offsets = (
                    (np.arange(n_here) - (n_here - 1) / 2.0) * width_global
                    if (hue_col and n_here > 1)
                    else np.array([0.0])
                )
                hue_index_panel = {h: i for i, h in enumerate(hue_levels)}

                # Axes cosmetics: horizontal grid only (cleaner)
                ax.grid(False)
                ax.yaxis.grid(True, which="major")
                ax.xaxis.grid(False)

                # Keep legend patches per hue to avoid duplicates
                legend_handles: dict[str, object] = {}

                # Draw bars ONE x-category at a time → simple, robust alignment
                for j, xv in enumerate(x_order):
                    x_center = base_pos[j]
                    for _i, hval in enumerate(hue_levels):
                        row = _row_for(
                            sbar,
                            x_val=xv,
                            ch=channel_key,
                            hue_val=(hval if hue_col else None),
                            group_val=group_val_for_panel,
                        )
                        if row is None or not np.isfinite(float(row[agg])):
                            continue

                        height = float(row[agg])

                        # Error bars (native yerr)
                        yerr = None
                        if err == "sem":
                            ev = float(row.get("sem", np.nan))
                            yerr = None if not np.isfinite(ev) else ev
                        elif err == "iqr":
                            q1 = row.get("q1", np.nan)
                            q3 = row.get("q3", np.nan)
                            if np.isfinite(q1) and np.isfinite(q3):
                                if agg == "median":
                                    lo = max(height - float(q1), 0.0)
                                    hi = max(float(q3) - height, 0.0)
                                    yerr = _asym_yerr([lo], [hi])
                                else:  # mean → symmetric ±IQR/2
                                    half = max(0.5 * (float(q3) - float(q1)), 0.0)
                                    yerr = half
                        # errorbar style (compatible across Matplotlib versions)
                        error_kw = {"capsize": 3, "elinewidth": 1.0, "alpha": 0.9} if yerr is not None else None

                        xpos = x_center + offsets[hue_index_panel[hval]]
                        bars = ax.bar(
                            [xpos],
                            [height],
                            width=width,
                            color=color_map[hval] if hue_col else "#D9D9D9",
                            edgecolor="#C0C0C0",
                            zorder=1,
                            yerr=yerr,
                            **({"error_kw": error_kw} if error_kw is not None else {}),
                            label=(str(hval) if (show_legend and hval not in legend_handles) else None),
                        )
                        if show_legend and hval not in legend_handles and len(bars.patches) > 0:
                            legend_handles[hval] = bars.patches[0]

                        # Replicate dots (overlay AFTER bars/errorbars)
                        rr = srep[(srep[x_col].astype(str) == str(xv))]
                        if hue_col:
                            rr = rr[rr[hue_col].astype(str) == str(hval)]
                        rr = rr[rr["channel"].astype(str) == channel_key]
                        if not rr.empty:
                            rng = np.random.default_rng()
                            has_hue_here = hue_col is not None and len(hue_levels) > 1
                            jitter = float(width) * (0.08 if has_hue_here else 0.12)
                            xj = xpos + (rng.random(len(rr)) - 0.5) * (2.0 * jitter)
                            ax.scatter(
                                xj,
                                rr["value"],
                                s=34,
                                zorder=3,
                                # White fill; light-gray edge for clarity on gray bars
                                facecolors="#FFFFFF",
                                edgecolors="#C0C0C0",
                                linewidths=0.7,
                                # Preserve color legend when hue is used (bar color carries hue)
                                color=None,
                            )

                # Tick labels and titles
                ax.set_xticks(base_pos)
                ax.set_xticklabels(x_order, rotation=45, ha="right")
                if panel_by == "group":
                    ax.set_title(str(panel))
                ax.set_xlabel("")
                ax.set_ylabel(str(channel_key if panel_by != "channel" else panel))
                with suppress(Exception):
                    ax.set_box_aspect(1.0)

                if show_legend and hue_col and len(hue_levels) > 1 and legend_handles:
                    ax.legend(
                        handles=list(legend_handles.values()),
                        labels=list(legend_handles.keys()),
                        loc=legend_loc,
                        title=None,
                    )

            # Hide any unused axes (when grid > number of panels)
            for k in range(len(panels), len(axes)):
                axes[k].set_visible(False)

            # Apply unified y-limits when applicable
            if unify_same_channel and (y_hi_glob is not None):
                for _ax in axes[: len(panels)]:
                    if _ax.get_visible():
                        _ax.set_ylim(y_lo_glob if y_lo_glob is not None else _ax.get_ylim()[0], y_hi_glob)

            # Save (PDF by default)
            ext = str(fig_kwargs.get("ext", "pdf")).lower()
            dpi = fig_kwargs.get("dpi", None)
            if filename:
                stub = filename
            else:
                if panel_by == "group":
                    # When iterating channels, selected_channel may change per file.
                    stub = f"snap__grp__{selected_channel}"
                elif panel_by == "channel":
                    key = members[0] if group_col and members and members[0] is not None else fig_label
                    stub = f"snap__ch__{key}"
                else:  # by x
                    stub = f"snap__x__{selected_channel}__{fig_label}"
            save_figure(fig, Path(output_dir), stub, ext=ext, dpi=dpi)
            plt.close(fig)
