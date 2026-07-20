"""Theme-independent publication styling for response-window figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

AXIS_LABEL_SIZE = 10.0
LEGEND_SIZE = 8.5
PANEL_TITLE_SIZE = 11.0
TICK_LABEL_SIZE = 10.0


def style_data_axis(axis: plt.Axes, *, grid_axis: str | None = None) -> None:
    """Keep quiet grids behind marks and standardize review axes."""

    axis.set_axisbelow(True)
    if grid_axis is not None:
        axis.grid(axis=grid_axis, color="#e5e7eb", linewidth=0.7, zorder=0)
    for line in (*axis.get_xgridlines(), *axis.get_ygridlines()):
        line.set_zorder(0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def apply_publication_style(figure: plt.Figure) -> plt.Figure:
    figure.patch.set_facecolor("white")
    if figure._suptitle is not None:
        figure._suptitle.set_color("#111827")
        if figure.get_constrained_layout():
            figure.get_layout_engine().set(h_pad=0.06)
    for axis in figure.axes:
        axis.set_facecolor("white")
        axis.set_axisbelow(True)
        axis.title.set_color("#111827")
        axis.xaxis.label.set_color("#111827")
        axis.yaxis.label.set_color("#111827")
        axis.tick_params(colors="#111827")
        for spine in axis.spines.values():
            spine.set_color("#6b7280")
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("white")
            for text in legend.get_texts():
                text.set_color("#111827")
    for legend in figure.legends:
        legend.get_frame().set_facecolor("white")
        for text in legend.get_texts():
            text.set_color("#111827")
    return figure


def save_publication_figure(figure: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    apply_publication_style(figure)
    figure.savefig(path, dpi=dpi, facecolor="white", transparent=False, bbox_inches="tight")


__all__ = [
    "AXIS_LABEL_SIZE",
    "LEGEND_SIZE",
    "PANEL_TITLE_SIZE",
    "TICK_LABEL_SIZE",
    "apply_publication_style",
    "save_publication_figure",
    "style_data_axis",
]
