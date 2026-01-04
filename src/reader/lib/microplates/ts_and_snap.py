"""
--------------------------------------------------------------------------------
<reader project>
src/reader/lib/microplates/ts_and_snap.py

Two-panel figure: (left) time series, (right) snapshot barplot,
driven by the same group selection.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from reader.core.plot_sinks import PlotFigure

from .base import (
    GroupMatch,
    alias_column,
    nearest_time_per_key,
    require_columns,
    resolve_groups,
    save_figure,
    smart_grouped_dose_key,
    smart_string_numeric_key,
    warn_if_empty,
)
from .style import _DEFAULT_RC as _RC
from .style import PaletteBook, use_style

# -------- small helpers (kept local to avoid cross-module imports) --------

_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "H"]


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


def _order_levels(levels: list[str]) -> list[str]:
    try:
        return sorted(levels, key=smart_grouped_dose_key)
    except Exception:
        return sorted(levels, key=smart_string_numeric_key)


# -------------------------------- main API --------------------------------


def plot_ts_and_snap(
    *,
    df: pd.DataFrame,
    output_dir: Path | None,
    # grouping
    group_on: str | None,
    pool_sets: list[dict[str, list[str]]] | None,
    pool_match: GroupMatch = "exact",
    # time series (left)
    ts_x: str = "time",
    ts_channel: str,
    ts_hue: str,
    ts_time_window: list[float] | None = None,
    ts_add_sheet_line: bool = False,
    ts_sheet_line_kwargs: dict | None = None,
    ts_mark_snap_time: bool = False,
    ts_snap_line_kwargs: dict | None = None,
    ts_log_transform: bool | list[str] = False,
    ts_ci: float = 95.0,
    ts_ci_alpha: float = 0.15,
    ts_show_replicates: bool = False,
    ts_legend_loc: str = "upper right",
    # snapshot (right)
    snap_x: str = "treatment",
    snap_channel: str | None = None,  # defaults to ts_channel
    snap_hue: str | None = None,  # defaults to None (gray bars, white dots)
    snap_time: float = 0.0,
    snap_agg: str = "mean",  # "mean" | "median"
    snap_err: str = "sem",  # "sem" | "iqr" | "none"
    snap_time_tolerance: float = 0.51,
    snap_show_legend: bool = False,
    snap_legend_loc: str = "upper right",
    # figure/style
    fig_kwargs: dict | None = None,
    filename: str | None = None,
    palette_book: PaletteBook | None = None,
) -> list[PlotFigure]:
    """
    Render one figure per group value (if group_on is set), each with two subplots:
      • Left  = time series (mean ± CI) for `ts_channel`
      • Right = snapshot barplot for `snap_channel` (defaults to `ts_channel`) at `snap_time`

    Hue handling:
      • Left  requires `ts_hue`
      • Right uses `snap_hue` if provided; when `snap_hue == ts_hue`, colors are shared
      • Otherwise snapshot bars are gray with white replicate dots (no legend by default)
    """
    if snap_agg not in {"mean", "median"}:
        raise ValueError("snap_agg must be 'mean' or 'median'")
    if snap_err not in {"sem", "iqr", "none"}:
        raise ValueError("snap_err must be 'sem', 'iqr', or 'none'")

    fig_kwargs = fig_kwargs or {}

    # Resolve columns (prefer *_alias when present)
    ts_x_col = alias_column(df, ts_x)
    group_col = alias_column(df, group_on) if group_on else None
    ts_hue_col = alias_column(df, ts_hue)
    snap_x_col = alias_column(df, snap_x)
    snap_hue_col = alias_column(df, snap_hue) if snap_hue else None

    ch_ts = str(ts_channel)
    ch_snap = str(snap_channel if snap_channel else ts_channel)

    required = ["time", "channel", "value", "position", ts_x_col, ts_hue_col, snap_x_col]
    if group_col:
        required.append(group_col)
    if snap_hue_col:
        required.append(snap_hue_col)
    require_columns(df, required, where="ts_and_snap")

    # Base numerics
    work = df.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    if ts_time_window:
        lo, hi = float(ts_time_window[0]), float(ts_time_window[1])
        work = work[
            (pd.to_numeric(work[ts_x_col], errors="coerce") >= lo)
            & (pd.to_numeric(work[ts_x_col], errors="coerce") <= hi)
        ].copy()
    if warn_if_empty(work, where="ts_and_snap", detail="after time_window filter"):
        return []

    available_channels = sorted(work["channel"].astype(str).unique().tolist())
    if ch_ts not in available_channels:
        raise ValueError(f"ts_and_snap: ts_channel {ch_ts!r} not in data. Available: {available_channels}")
    if ch_snap not in available_channels:
        raise ValueError(f"ts_and_snap: snap_channel {ch_snap!r} not in data. Available: {available_channels}")

    # Figure iteration over groups
    if group_col:
        universe = work[group_col].astype(str).unique().tolist()
        fig_groups = (
            resolve_groups(universe, pool_sets, match=pool_match) if pool_sets else [(g, [g]) for g in universe]
        )
    else:
        fig_groups = [("all", [None])]

    figures: list[PlotFigure] = []
    for label, members in fig_groups:
        d = work.copy()
        if group_col and members != [None]:
            d = d[d[group_col].astype(str).isin(members)]
        if d.empty:
            continue

        # TS hue levels + colors
        hue_levels_ts = list(d[ts_hue_col].astype(str).unique())
        marker_map = {h: _MARKERS[i % len(_MARKERS)] for i, h in enumerate(hue_levels_ts)}
        colors = _colors_for(len(hue_levels_ts), palette_book)
        color_map = {h: colors[i % len(colors)] for i, h in enumerate(hue_levels_ts)}

        # If snapshot uses the same hue column, reuse colors; otherwise compute locally on demand
        def _snap_color_map(
            hue_levels_snap: list[str],
            *,
            color_map=color_map,
            colors=colors,
        ) -> dict[str, str]:
            if snap_hue_col and snap_hue_col == ts_hue_col:
                return {h: color_map.get(h, colors[i % len(colors)]) for i, h in enumerate(hue_levels_snap)}
            snap_colors = _colors_for(len(hue_levels_snap), palette_book)
            return {h: snap_colors[i % len(snap_colors)] for i, h in enumerate(hue_levels_snap)}

        with use_style(rc=(fig_kwargs or {}).get("rc"), color_cycle=colors):
            # If user didn't override figsize, widen it for a side-by-side layout.
            fkw = dict(fig_kwargs)
            if "figsize" not in fkw:
                # Use the base height and double the width
                base_w, base_h = _RC["figure_figsize"]
                fkw["figsize"] = (base_w * 2.0, base_h)

            fig, axes = plt.subplots(1, 2, **{k: v for k, v in fkw.items() if k not in {"rc", "ext"}})
            ax_ts, ax_snap = axes  # left, right

            # ---- Left: time series (mean ± CI) ----
            ts = d[d["channel"].astype(str) == ch_ts].copy()
            if not ts.empty:
                ax = ax_ts
                # grid cosmetics
                ax.grid(False)
                ax.yaxis.grid(True, which="major")
                ax.xaxis.grid(True, which="major")

                # Replicates (optional)
                if ts_show_replicates:
                    for h in hue_levels_ts:
                        rr = ts[ts[ts_hue_col].astype(str) == h]
                        ax.scatter(
                            rr[ts_x_col],
                            rr["value"],
                            s=18,
                            alpha=float(fig_kwargs.get("replicate_alpha", 0.30)),
                            zorder=3,
                            linewidths=0.0,
                            edgecolors="none",
                            marker=marker_map[h],
                            c=color_map[h],
                        )

                # Mean line + CI band
                sns.lineplot(
                    data=ts,
                    x=ts_x_col,
                    y="value",
                    hue=ts_hue_col,
                    hue_order=hue_levels_ts,
                    estimator="mean",
                    errorbar=("ci", float(ts_ci)),
                    err_style="band",
                    err_kws={"alpha": float(ts_ci_alpha)},
                    lw=1.8,
                    alpha=float(fig_kwargs.get("line_alpha", 0.85)),
                    legend=False,
                    ax=ax,
                    palette=[color_map[h] for h in hue_levels_ts],
                    marker=None,
                    zorder=1,
                )

                # Mean points (accent)
                means = ts.groupby([ts_hue_col, ts_x_col], dropna=False)["value"].mean().reset_index()
                for h in hue_levels_ts:
                    mm = means[means[ts_hue_col].astype(str) == h]
                    ax.scatter(
                        mm[ts_x_col],
                        mm["value"],
                        s=36,
                        zorder=2.5,
                        marker=marker_map[h],
                        alpha=float(fig_kwargs.get("mean_marker_alpha", 0.75)),
                        edgecolors="none",
                        linewidths=0.0,
                        c=color_map[h],
                    )

                # Sheet change markers (optional)
                if ts_add_sheet_line and "sheet_index" in ts.columns:
                    starts = sorted(ts.groupby("sheet_index")[ts_x_col].min().dropna().tolist())
                    trans = starts[1:] if len(starts) > 1 else []
                    style = {"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.8, "alpha": 0.9, "zorder": 0.5}
                    style.update(ts_sheet_line_kwargs or {})
                    for sx in trans:
                        ax.axvline(float(sx), **style)

                # Optional: show a vertical marker at the snapshot time used on the right
                if ts_mark_snap_time:
                    sstyle = {"color": "#9E9E9E", "linestyle": "--", "linewidth": 0.9, "alpha": 1.0, "zorder": 0.8}
                    sstyle.update(ts_snap_line_kwargs or {})
                    ax.axvline(float(snap_time), **sstyle)

                # Log y?
                if isinstance(ts_log_transform, list):
                    ax.set_yscale("log" if (ch_ts in ts_log_transform) else "linear")
                else:
                    if bool(ts_log_transform):
                        ax.set_yscale("log")

                ax.set_xlabel("Time (h)" if str(ts_x).lower() == "time" else ts_x_col)
                ax.set_ylabel(ch_ts)

                # Legend on left only
                handles = [
                    Line2D(
                        [0],
                        [0],
                        color=color_map[h],
                        marker=marker_map[h],
                        markersize=7,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=float(fig_kwargs.get("mean_marker_alpha", 0.75)),
                        label=str(h),
                    )
                    for h in hue_levels_ts
                ]
                ax.legend(handles=handles, loc=str(ts_legend_loc), title=None)

            # ---- Right: snapshot barplot ----
            snap = d.copy()
            # Pick nearest-time replicate per key at snap_time
            key_cols = [c for c in [group_col, snap_x_col, snap_hue_col, "channel", "position"] if c]
            snapped = nearest_time_per_key(
                snap, target_time=float(snap_time), keys=key_cols, tol=float(snap_time_tolerance)
            )
            snapped = snapped[snapped["channel"].astype(str) == ch_snap].copy()
            if snapped.empty:
                log = logging.getLogger("reader")
                fb = nearest_time_per_key(snap, target_time=float(snap_time), keys=key_cols, tol=float("inf"))
                snapped = fb[fb["channel"].astype(str) == ch_snap].copy()
                if not snapped.empty:
                    uniq = sorted(pd.to_numeric(snapped["time"], errors="coerce").dropna().unique().tolist())
                    t_rep = uniq[0] if len(uniq) == 1 else float(pd.Series(uniq).median())
                    delta = abs(float(t_rep) - float(snap_time))
                    preview = ", ".join(f"{t:.2f}" for t in uniq[:6]) + (" …" if len(uniq) > 6 else "")
                    log.info(
                        "[warn]ts_and_snap:snapshot[/warn] • requested t=%.2f h; no rows within ±%.2f h — "
                        "using nearest available per key (times=%s; Δ≈%.2f h)",
                        float(snap_time),
                        float(snap_time_tolerance),
                        preview,
                        float(delta),
                    )
            if not snapped.empty:
                ax = ax_snap

                # aggregate to bar heights + errors
                base_group_cols: list[str] = [snap_x_col] + ([snap_hue_col] if snap_hue_col else [])
                stats = (
                    snapped.groupby(base_group_cols, dropna=False)["value"]
                    .agg(n="count", mean="mean", median="median", std="std", sem="sem")
                    .reset_index()
                )

                # IQR if requested
                if snap_err == "iqr":
                    q = (
                        snapped.groupby(base_group_cols, dropna=False)["value"]
                        .quantile([0.25, 0.75])
                        .unstack(-1)
                        .reset_index()
                        .rename(columns={0.25: "q1", 0.75: "q3"})
                    )
                    stats = stats.merge(q, on=base_group_cols, how="left")

                # x ordering + hue levels + colors
                x_levels = stats[snap_x_col].astype(str).unique().tolist()
                x_order = _order_levels(x_levels)
                hue_levels_snap = (
                    sorted(stats[snap_hue_col].astype(str).unique().tolist(), key=smart_string_numeric_key)
                    if snap_hue_col
                    else ["_single"]
                )
                color_map_snap = _snap_color_map(hue_levels_snap)

                # layout
                n_x = len(x_order)
                base_pos = np.arange(n_x, dtype=float)
                num_hues = len(hue_levels_snap) if snap_hue_col else 1
                has_hue = snap_hue_col is not None and num_hues > 1
                width = 0.8 if not has_hue else min(0.85 / max(num_hues, 1), 0.8)
                offsets = (np.arange(num_hues) - (num_hues - 1) / 2.0) * width if has_hue else np.array([0.0])
                hue_index = {h: i for i, h in enumerate(hue_levels_snap)}

                # cosmetics
                ax.grid(False)
                ax.yaxis.grid(True, which="major")
                ax.xaxis.grid(False)

                legend_handles: dict[str, object] = {}

                # draw bars one x at a time
                for j, xv in enumerate(x_order):
                    x_center = base_pos[j]
                    for h in hue_levels_snap:
                        sub = stats[stats[snap_x_col].astype(str) == str(xv)]
                        if snap_hue_col:
                            sub = sub[sub[snap_hue_col].astype(str) == str(h)]
                        if sub.empty:
                            continue
                        row = sub.iloc[0]
                        height = float(row[snap_agg])

                        # yerr
                        yerr = None
                        if snap_err == "sem":
                            ev = float(row.get("sem", np.nan))
                            yerr = None if not np.isfinite(ev) else ev
                        elif snap_err == "iqr":
                            q1 = row.get("q1", np.nan)
                            q3 = row.get("q3", np.nan)
                            if np.isfinite(q1) and np.isfinite(q3):
                                if snap_agg == "median":
                                    lo = max(height - float(q1), 0.0)
                                    hi = max(float(q3) - height, 0.0)
                                    yerr = np.vstack([[lo], [hi]])  # asym
                                else:
                                    half = max(0.5 * (float(q3) - float(q1)), 0.0)
                                    yerr = half

                        error_kw = {"capsize": 3, "elinewidth": 1.0, "alpha": 0.9} if yerr is not None else None
                        xpos = x_center + offsets[hue_index[h]]
                        bar_color = color_map_snap[h] if snap_hue_col else "#D9D9D9"
                        bars = ax.bar(
                            [xpos],
                            [height],
                            width=width,
                            color=bar_color,
                            edgecolor="#C0C0C0",
                            zorder=1,
                            yerr=yerr,
                            **({"error_kw": error_kw} if error_kw else {}),
                            label=(str(h) if (snap_show_legend and h not in legend_handles) else None),
                        )
                        if snap_show_legend and h not in legend_handles and len(bars.patches) > 0:
                            legend_handles[h] = bars.patches[0]

                        # replicate dots
                        rr = snapped[snapped[snap_x_col].astype(str) == str(xv)]
                        if snap_hue_col:
                            rr = rr[rr[snap_hue_col].astype(str) == str(h)]
                        if not rr.empty:
                            rng = np.random.default_rng()
                            jitter = float(width) * (0.08 if has_hue else 0.12)
                            xj = xpos + (rng.random(len(rr)) - 0.5) * (2.0 * jitter)
                            ax.scatter(
                                xj,
                                rr["value"],
                                s=34,
                                zorder=3,
                                facecolors="#FFFFFF",
                                edgecolors="#C0C0C0",
                                linewidths=0.7,
                                color=None,
                            )

                ax.set_xticks(base_pos)
                ax.set_xticklabels(x_order, rotation=45, ha="right")
                ax.set_xlabel("")
                ax.set_ylabel(ch_snap)
                ax.set_title(f"t≈{float(snap_time):.2f} h", fontweight="normal")

                if snap_show_legend and snap_hue_col and len(hue_levels_snap) > 1 and legend_handles:
                    ax.legend(
                        handles=list(legend_handles.values()),
                        labels=list(legend_handles.keys()),
                        loc=str(snap_legend_loc),
                        title=None,
                    )

            # ---- figure title + save ----
            fig.suptitle(f"{label}", y=float(fig_kwargs.get("suptitle_y", 1.04)))
            ext = str((fig_kwargs or {}).get("ext", "pdf")).lower()
            dpi = (fig_kwargs or {}).get("dpi", None)

            # ----- Unique, descriptive filenames -----
            # If grouping is active, append "<group_col>=<label>" to make per-group files distinct.
            group_tag = None
            if group_col and members != [None]:
                # Example: "__genotype=araBADp"  or "__genotype=Ara_related" for pooled sets
                group_tag = f"__{str(group_col)}={str(label)}"

            if filename:
                # Respect the base name but *append* the group tag when present
                stub = f"{filename}{group_tag}" if group_tag else filename
            else:
                base = f"ts_snap__{ch_snap}"
                # Backward‑compatible default still includes label; enhanced with group_col when available
                stub = f"{base}{group_tag}" if group_tag else f"{base}__{label}"
            if output_dir is None:
                figures.append(PlotFigure(fig=fig, filename=stub, ext=ext, dpi=dpi))
            else:
                save_figure(fig, Path(output_dir), stub, ext=ext, dpi=dpi)
                plt.close(fig)
    return figures
