"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/panels/snapshot.py

Shared snapshot selection and drawing primitives.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from reader.core.plot_style import PaletteBook
from reader.domains.plate_reader.analysis.timepoints import nearest_time_per_key
from reader.domains.plate_reader.ordering import order_levels
from reader.domains.plate_reader.plots.common import colors_for


@dataclass(frozen=True)
class SnapshotSelection:
    rows: pd.DataFrame
    time_used: float
    fell_back: bool
    fallback_delta: float | None = None
    fallback_times_preview: str | None = None


def select_snapshot_rows(
    *,
    df: pd.DataFrame,
    target_time: float,
    keys: Sequence[str],
    channel: str,
    tolerance: float,
) -> SnapshotSelection:
    snapped = nearest_time_per_key(df, target_time=float(target_time), keys=list(keys), tol=float(tolerance))
    snapped = snapped[snapped["channel"].astype(str) == str(channel)].copy()
    if snapped.empty:
        fallback = nearest_time_per_key(df, target_time=float(target_time), keys=list(keys), tol=float("inf"))
        fallback = fallback[fallback["channel"].astype(str) == str(channel)].copy()
        if fallback.empty:
            return SnapshotSelection(rows=fallback, time_used=float(target_time), fell_back=True)
        times_used = pd.to_numeric(fallback["time"], errors="coerce").dropna()
        unique_times = sorted(times_used.unique().tolist())
        t_rep = unique_times[0] if len(unique_times) == 1 else float(pd.Series(unique_times).median())
        preview = ", ".join(f"{t:.2f}" for t in unique_times[:6]) + (" …" if len(unique_times) > 6 else "")
        return SnapshotSelection(
            rows=fallback,
            time_used=float(times_used.median()) if not times_used.empty else float(target_time),
            fell_back=True,
            fallback_delta=abs(float(t_rep) - float(target_time)),
            fallback_times_preview=preview,
        )

    times_used = pd.to_numeric(snapped["time"], errors="coerce").dropna()
    time_used = float(times_used.median()) if not times_used.empty else float(target_time)
    return SnapshotSelection(rows=snapped, time_used=time_used, fell_back=False)


def summarize_snapshot_values(
    *,
    df: pd.DataFrame,
    group_cols: Sequence[str],
    err: str,
) -> pd.DataFrame:
    stats_pl = (
        pl.from_pandas(df)
        .with_columns(
            pl.when(pl.col("value").cast(pl.Float64, strict=False).is_nan())
            .then(None)
            .otherwise(pl.col("value").cast(pl.Float64, strict=False))
            .alias("value")
        )
        .group_by(list(group_cols))
        .agg(
            pl.col("value").count().alias("n"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").median().alias("median"),
            pl.col("value").std(ddof=1).alias("std"),
        )
        .with_columns(
            pl.when(pl.col("n") > 0)
            .then(pl.col("std") / pl.col("n").cast(pl.Float64).sqrt())
            .otherwise(None)
            .alias("sem")
        )
    )

    if err == "iqr":
        q_pl = (
            pl.from_pandas(df)
            .group_by(list(group_cols))
            .agg(
                pl.col("value").quantile(0.25, interpolation="linear").alias("q1"),
                pl.col("value").quantile(0.75, interpolation="linear").alias("q3"),
            )
        )
        stats_pl = stats_pl.join(q_pl, on=list(group_cols), how="left")

    return stats_pl.sort(list(group_cols)).to_pandas(use_pyarrow_extension_array=False)


def draw_snapshot_panel(
    ax,
    *,
    snapped: pd.DataFrame,
    stats: pd.DataFrame,
    x_col: str,
    hue_col: str | None,
    agg: str,
    err: str,
    palette_book: PaletteBook | None,
    show_legend: bool,
    legend_loc: str,
    title: str | None,
    ylabel: str,
    color_map: dict[str, str] | None = None,
) -> None:
    x_levels = stats[x_col].astype(str).unique().tolist()
    x_order = order_levels(x_levels)
    hue_levels = order_levels(stats[hue_col].astype(str).unique().tolist()) if hue_col else ["_single"]
    if color_map is None:
        if hue_col:
            colors = colors_for(len(hue_levels), palette_book)
            color_map = {hue: colors[idx % len(colors)] for idx, hue in enumerate(hue_levels)}
        else:
            color_map = {"_single": "#D9D9D9"}

    n_x = len(x_order)
    base_pos = np.arange(n_x, dtype=float)
    num_hues = len(hue_levels) if hue_col else 1
    has_hue = hue_col is not None and num_hues > 1
    width = 0.8 if not has_hue else min(0.85 / max(num_hues, 1), 0.8)
    offsets = (np.arange(num_hues) - (num_hues - 1) / 2.0) * width if has_hue else np.array([0.0])
    hue_index = {hue: idx for idx, hue in enumerate(hue_levels)}

    ax.grid(False)
    ax.yaxis.grid(True, which="major")
    ax.xaxis.grid(False)

    legend_handles: dict[str, object] = {}
    for j, x_value in enumerate(x_order):
        x_center = base_pos[j]
        for hue in hue_levels:
            sub = stats[stats[x_col].astype(str) == str(x_value)]
            if hue_col:
                sub = sub[sub[hue_col].astype(str) == str(hue)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            height = float(row[agg])

            yerr = None
            if err == "sem":
                err_value = float(row.get("sem", np.nan))
                yerr = None if not np.isfinite(err_value) else err_value
            elif err == "iqr":
                q1 = row.get("q1", np.nan)
                q3 = row.get("q3", np.nan)
                if np.isfinite(q1) and np.isfinite(q3):
                    if agg == "median":
                        lower = max(height - float(q1), 0.0)
                        upper = max(float(q3) - height, 0.0)
                        yerr = np.vstack([[lower], [upper]])
                    else:
                        yerr = max(0.5 * (float(q3) - float(q1)), 0.0)

            error_kw = {"capsize": 3, "elinewidth": 1.0, "alpha": 0.9} if yerr is not None else None
            xpos = x_center + offsets[hue_index[hue]]
            bar_color = color_map[hue] if hue_col else "#D9D9D9"
            bars = ax.bar(
                [xpos],
                [height],
                width=width,
                color=bar_color,
                edgecolor="#C0C0C0",
                zorder=1,
                yerr=yerr,
                **({"error_kw": error_kw} if error_kw else {}),
                label=(str(hue) if (show_legend and hue not in legend_handles) else None),
            )
            if show_legend and hue not in legend_handles and len(bars.patches) > 0:
                legend_handles[hue] = bars.patches[0]

            rr = snapped[snapped[x_col].astype(str) == str(x_value)]
            if hue_col:
                rr = rr[rr[hue_col].astype(str) == str(hue)]
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
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="normal")

    if show_legend and hue_col and len(hue_levels) > 1 and legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc=str(legend_loc),
            title=None,
        )
