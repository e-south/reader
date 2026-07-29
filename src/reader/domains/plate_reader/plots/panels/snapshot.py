"""
--------------------------------------------------------------------------------
<reader project>
src/reader/domains/plate_reader/plots/panels/snapshot.py

Shared snapshot selection and drawing primitives.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reader.domains.plate_reader.ordering import order_levels
from reader.domains.plate_reader.plots.common import colors_for
from reader.plotting.style import PaletteBook


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
    x_color_map: dict[str, str] | None = None,
    x_order: list[str] | None = None,
    hue_order: list[str] | None = None,
    tick_rotation: float = 45.0,
    axis_label_size: float | None = None,
    tick_label_size: float | None = None,
    legend_fontsize: float | None = None,
    replicate_seed: int = 0,
) -> None:
    x_levels = stats[x_col].astype(str).unique().tolist()
    resolved_x_order = [str(value) for value in x_order] if x_order is not None else order_levels(x_levels)
    observed_x = set(map(str, x_levels))
    missing_x = [value for value in resolved_x_order if value not in observed_x]
    omitted_x = [value for value in observed_x if value not in resolved_x_order]
    if missing_x:
        raise ValueError(f"snapshot: x order includes missing label(s): {missing_x}")
    if omitted_x:
        raise ValueError(f"snapshot: x order omits observed label(s): {sorted(omitted_x)}")
    if len(set(resolved_x_order)) != len(resolved_x_order):
        raise ValueError("snapshot: x order contains duplicate labels")
    observed_hue_levels = order_levels(stats[hue_col].astype(str).unique().tolist()) if hue_col else ["_single"]
    hue_levels = [str(value) for value in hue_order] if hue_col and hue_order is not None else observed_hue_levels
    observed_hue = set(observed_hue_levels)
    missing_hue = [value for value in hue_levels if value not in observed_hue]
    omitted_hue = [value for value in observed_hue if value not in hue_levels]
    if missing_hue:
        raise ValueError(f"snapshot: hue order includes missing label(s): {missing_hue}")
    if omitted_hue:
        raise ValueError(f"snapshot: hue order omits observed label(s): {sorted(omitted_hue)}")
    if len(set(hue_levels)) != len(hue_levels):
        raise ValueError("snapshot: hue order contains duplicate labels")
    if color_map is None:
        if hue_col:
            colors = colors_for(len(hue_levels), palette_book)
            color_map = {hue: colors[idx % len(colors)] for idx, hue in enumerate(hue_levels)}
        else:
            color_map = {"_single": "#D9D9D9"}

    n_x = len(resolved_x_order)
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
    rng = np.random.default_rng(int(replicate_seed))
    for j, x_value in enumerate(resolved_x_order):
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
                        yerr = [[lower], [upper]]
                    else:
                        yerr = max(0.5 * (float(q3) - float(q1)), 0.0)

            error_kw = {"capsize": 3, "elinewidth": 1.0, "alpha": 0.9} if yerr is not None else None
            xpos = x_center + offsets[hue_index[hue]]
            bar_color = color_map[hue] if hue_col else ((x_color_map or {}).get(str(x_value), "#D9D9D9"))
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
    horizontal_alignment = "center" if float(tick_rotation) == 0.0 else "right"
    tick_kwargs = {"fontsize": tick_label_size} if tick_label_size is not None else {}
    ax.set_xticklabels(resolved_x_order, rotation=float(tick_rotation), ha=horizontal_alignment, **tick_kwargs)
    ax.set_xlabel("")
    ylabel_kwargs = {"fontsize": axis_label_size} if axis_label_size is not None else {}
    ax.set_ylabel(ylabel, **ylabel_kwargs)
    if tick_label_size is not None:
        ax.tick_params(axis="y", labelsize=tick_label_size)
    if title:
        ax.set_title(title, fontweight="normal")

    if show_legend and hue_col and len(hue_levels) > 1 and legend_handles:
        legend_kwargs = {"fontsize": legend_fontsize} if legend_fontsize is not None else {}
        ax.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc=str(legend_loc),
            title=None,
            frameon=False,
            **legend_kwargs,
        )
