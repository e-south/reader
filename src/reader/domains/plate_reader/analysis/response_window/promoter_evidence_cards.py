"""Compact header and shared legend for the promoter-evidence figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plot_style import LEGEND_SIZE
from .sources import STATE_ORDER
from .visual_labels import STATE_COLORS, STATE_MARKERS


def draw_header_axis(
    axis: plt.Axes,
    *,
    state_labels: dict[str, object],
    reference_id: str,
) -> None:
    axis.set_axis_off()
    axis.set_gid("promoter-evidence-header")
    axis.legend(
        handles=_figure_legend(state_labels=state_labels, reference_id=reference_id),
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=LEGEND_SIZE,
        borderaxespad=0,
    )


def _figure_legend(
    *,
    state_labels: dict[str, object],
    reference_id: str,
) -> list[Line2D]:
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
                label=f"{reference_id} anchor",
            ),
        )
    )
    return handles


__all__ = ["draw_header_axis"]
