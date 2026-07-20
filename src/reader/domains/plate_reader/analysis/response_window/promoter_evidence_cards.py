"""Compact shared legend for the promoter-evidence figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .plot_style import LEGEND_SIZE
from .sources import STATE_ORDER
from .visual_labels import STATE_COLORS, STATE_MARKERS


def draw_header_axis(
    axis: plt.Axes,
    *,
    state_labels: dict[str, object],
    reference_id: str,
    window_start_h: float,
    window_end_h: float,
) -> None:
    axis.set_axis_off()
    axis.set_gid("promoter-evidence-header")
    axis.legend(
        handles=_figure_legend(
            state_labels=state_labels,
            reference_id=reference_id,
            window_start_h=window_start_h,
            window_end_h=window_end_h,
        ),
        loc="center",
        ncol=7,
        frameon=False,
        fontsize=LEGEND_SIZE,
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.45,
    )


def _figure_legend(
    *,
    state_labels: dict[str, object],
    reference_id: str,
    window_start_h: float,
    window_end_h: float,
) -> list[Line2D | Patch]:
    handles = [
        Line2D(
            [],
            [],
            color=STATE_COLORS[state],
            marker=STATE_MARKERS[state],
            linewidth=1.8,
            label=str(state_labels[state]),
        )
        for state in STATE_ORDER
    ]
    handles.extend(
        (
            Line2D(
                [],
                [],
                color="#475569",
                marker="o",
                markerfacecolor="#475569",
                linewidth=1.8,
                label="Selected design",
            ),
            Line2D(
                [],
                [],
                color="#475569",
                marker="o",
                markerfacecolor="white",
                linestyle="--",
                linewidth=1.2,
                label=f"{reference_id} reference",
            ),
            Patch(
                facecolor="#f59e0b",
                alpha=0.28,
                edgecolor="none",
                label=f"{window_start_h:g}–{window_end_h:g} h summary window",
            ),
        )
    )
    return handles


__all__ = ["draw_header_axis"]
