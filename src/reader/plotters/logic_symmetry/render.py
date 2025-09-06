"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plotters/logic_symmetry/render.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

from .overlay import OverlayStyle, generate_overlay_points


@dataclass(frozen=True)
class VisualConfig:
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    grid: bool
    color: str
    annotate_designs: bool = False
    design_label_col: Optional[str] = None
    label_fontsize: int = 9
    label_offset: float = 0.02


def _color_map_for_hue(series: pd.Series) -> Dict[object, str]:
    cats = pd.Categorical(series.astype(str))
    n = len(cats.categories)
    if n == 0:
        return {}
    palette = sns.color_palette("colorblind", n_colors=n)
    return {cat: to_hex(palette[i]) for i, cat in enumerate(cats.categories)}


def _u_to_color(u: float) -> str:
    u = float(np.clip(u, 0.0, 1.0))
    g = 1.0 - u            # 0 (white) at no activation, 1 (black) at full
    return to_hex((g, g, g))


def _draw_tile_strip(ax, L: float, A: float, u: List[float], style: OverlayStyle, *, zorder: float = 0.3):
    """Draw a centered 4-square horizontal strip encoding u00,u10,u01,u11."""
    cell_w = float(style.tile_cell_w)
    cell_h = float(style.tile_cell_h)
    gap    = float(style.tile_gap)
    total_w = 4 * cell_w + 3 * gap
    x0 = float(L) - total_w / 2.0
    y0 = float(A) - cell_h / 2.0

    # 00,10,01,11 left→right
    for i, uu in enumerate([u[0], u[1], u[2], u[3]]):
        xi = x0 + i * (cell_w + gap)
        rect = Rectangle(
            (xi, y0), cell_w, cell_h,
            facecolor=_u_to_color(uu),
            edgecolor=style.edge_color,
            linewidth=style.tile_edge_width,
            alpha=style.alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)


def draw_scatter(
    points: pd.DataFrame,
    *,
    hue_col: Optional[str],
    visuals: VisualConfig,
    uncertainty_mode: str,
    overlay_cfg: Optional[OverlayStyle] = None,
    overlay_gate_set: Optional[str] = None,
    title: Optional[str] = None,
    figsize=(7, 6), dpi=300
):
    sns.set_style("ticks")

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(*visuals.xlim)
    ax.set_ylim(*visuals.ylim)
    ax.set_xlabel("Logic (L)", fontsize=14)
    ax.set_ylabel("Asymmetry (A)", fontsize=14)

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="both", which="major", labelsize=12)

    if visuals.grid:
        ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.6)
    sns.despine(ax=ax, top=True, right=True)

    # Prepare colors for experimental points
    if hue_col and points[hue_col].notna().any():
        cmap = _color_map_for_hue(points[hue_col])
        colors = points[hue_col].astype(str).map(cmap).fillna(visuals.color)
    else:
        cmap = None
        colors = pd.Series([visuals.color] * len(points), index=points.index)

    # ——— Ideal overlay (dots or tiles) ———
    if overlay_cfg and overlay_gate_set:
        ol = generate_overlay_points(overlay_gate_set).copy()
        xmin, xmax = visuals.xlim
        ymin, ymax = visuals.ylim

        if str(overlay_cfg.mode).lower() == "tiles":
            # Stack multiple strips at the same (L,A)
            for (L, A), g in ol.groupby(["L", "A"], sort=False):
                n = len(g)
                if overlay_cfg.tiles_stack_multiple and n > 1:
                    dy = overlay_cfg.tile_cell_h + overlay_cfg.label_line_height
                    offsets = [(i - (n - 1) / 2.0) * dy for i in range(n)]
                else:
                    offsets = [0.0] * n

                for off, (_, row) in zip(offsets, g.iterrows()):
                    _draw_tile_strip(ax, float(L), float(A) + float(off), list(row["u"]), overlay_cfg, zorder=0.3)

            # Labels: keep the existing stacked-text behavior
            if overlay_cfg.show_labels:
                for (L, A), g in ol.groupby(["L", "A"], sort=False):
                    labels = sorted(g["label"].astype(str).tolist())
                    n = len(labels)
                    above = True
                    if A + overlay_cfg.label_offset + (n - 1) * overlay_cfg.label_line_height > (ymax - 0.01):
                        above = False
                    if above:
                        y0 = A + overlay_cfg.label_offset
                        dy = overlay_cfg.label_line_height
                        va = "bottom"
                    else:
                        y0 = A - overlay_cfg.label_offset
                        dy = -overlay_cfg.label_line_height
                        va = "top"

                    for i, text in enumerate(labels):
                        yy = np.clip(y0 + i * dy, ymin + 0.005, ymax - 0.005)
                        ax.text(
                            float(L), float(yy), text,
                            fontsize=9, ha="center", va=va,
                            alpha=min(1.0, float(overlay_cfg.alpha) + 0.25),
                            zorder=0.35,
                        )
        else:
            # Legacy: circles at the ideal coordinates
            for (L, A), g in ol.groupby(["L", "A"], sort=False):
                ax.scatter(
                    [L], [A],
                    s=overlay_cfg.size,
                    facecolors=overlay_cfg.face_color,
                    edgecolors=overlay_cfg.edge_color,
                    alpha=overlay_cfg.alpha,
                    marker="o",
                    linewidths=0.9,
                    zorder=0
                )
                if overlay_cfg.show_labels:
                    labels = sorted(g["label"].astype(str).tolist())
                    n = len(labels)
                    above = True
                    if A + overlay_cfg.label_offset + (n - 1) * overlay_cfg.label_line_height > (ymax - 0.01):
                        above = False
                    if above:
                        y0 = A + overlay_cfg.label_offset
                        dy = overlay_cfg.label_line_height
                        va = "bottom"
                    else:
                        y0 = A - overlay_cfg.label_offset
                        dy = -overlay_cfg.label_line_height
                        va = "top"
                    for i, text in enumerate(labels):
                        yy = np.clip(y0 + i * dy, ymin + 0.005, ymax - 0.005)
                        ax.text(
                            float(L), float(yy), text,
                            fontsize=9, ha="center", va=va,
                            alpha=min(1.0, float(overlay_cfg.alpha) + 0.25),
                            zorder=0
                        )

    # ——— Experimental points ———
    for marker, sub in points.groupby("shape_value"):
        if uncertainty_mode == "halo":
            k = 1.5
            halo_size = sub["size_value"] * (1.0 + k * sub.get("cv", pd.Series(0.0, index=sub.index)).clip(0, 0.5))
            ax.scatter(
                sub["L"], sub["A"],
                s=halo_size,
                c=colors.loc[sub.index],
                alpha=np.minimum(sub["alpha_value"] * 0.35, 0.5),
                marker=marker,
                edgecolor="none",
                linewidths=0.0,
                zorder=1
            )

        if uncertainty_mode == "errorbars":
            err = (sub.get("cv", pd.Series(0.0, index=sub.index)).clip(0, 0.5)) * 0.3
            ax.errorbar(
                sub["L"], sub["A"],
                xerr=err, yerr=err,
                fmt="none",
                ecolor="#9e9e9e",
                elinewidth=0.7,
                capsize=2,
                alpha=0.7,
                zorder=1.4
            )

        ax.scatter(
            sub["L"], sub["A"],
            s=sub["size_value"],
            c=colors.loc[sub.index],
            alpha=sub["alpha_value"],
            marker=marker,
            edgecolor="k",
            linewidths=0.5,
            zorder=2
        )

    if visuals.annotate_designs:
        label_col = visuals.design_label_col
        if label_col is None or label_col not in points.columns:
            label_col = next((c for c in ["genotype", "design", "construct"] if c in points.columns), None)
        if label_col is not None:
            for _, r in points.iterrows():
                txt = r.get(label_col)
                if pd.isna(txt):
                    continue
                ax.text(
                    float(r["L"]), float(r["A"]) + float(visuals.label_offset),
                    str(txt),
                    fontsize=int(visuals.label_fontsize),
                    ha="center", va="bottom",
                    alpha=0.95, zorder=4
                )

    first_legend = None
    if cmap is not None:
        handles = []
        from matplotlib.lines import Line2D
        for cat, col in cmap.items():
            handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=col,
                                  label=str(cat), markersize=8))
        first_legend = ax.legend(title=hue_col, handles=handles, loc="best", frameon=False)
        if first_legend is not None:
            frame = first_legend.get_frame()
            frame.set_facecolor('none'); frame.set_edgecolor('none'); frame.set_alpha(0)

    if points["shape_value"].nunique() > 1:
        from matplotlib.lines import Line2D
        sh_handles = []
        for marker in points["shape_value"].unique().tolist():
            sh_handles.append(Line2D([0], [0], marker=marker, color='k', markerfacecolor='none',
                                     label=str(marker), markersize=8, linewidth=0))
        second_legend = ax.legend(title=(hue_col or "shape"), handles=sh_handles, loc="upper left", frameon=False)
        if second_legend is not None:
            frame = second_legend.get_frame()
            frame.set_facecolor('none'); frame.set_edgecolor('none'); frame.set_alpha(0)
        if first_legend is not None:
            ax.add_artist(first_legend)

    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax
