"""Family definitions and evidence key for the response-window phenotype."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .plot_style import HANDOFF_FAMILY_LABEL_SIZE


def draw_handoff_family_axis(axis: plt.Axes, *, width_ratio: float) -> None:
    """Explain the two phenotype families in a gutter beside the data panel."""

    if not 0.0 < width_ratio <= 1.0:
        raise ValueError("phenotype-family gutter width ratio must lie in (0, 1].")
    axis.set_gid("promoter-evidence-response-window-phenotype-families")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.75, 8.75)
    axis.set_box_aspect(1.0 / width_ratio)
    axis.set_axis_off()
    _draw_family_bracket(
        axis,
        y_bottom=5.0,
        y_top=8.0,
        text="Response rᵢ\nlog₂(YFP/CFP)",
        gid="handoff-family-response",
    )
    _draw_family_bracket(
        axis,
        y_bottom=0.0,
        y_top=3.0,
        text=("Signal bᵢ\nlog₂(YFP/OD600)\nrelative to\nsame-state\npDual-10"),
        gid="handoff-family-fluorescence",
    )


def _draw_family_bracket(
    axis: plt.Axes,
    *,
    y_bottom: float,
    y_top: float,
    text: str,
    gid: str,
) -> None:
    """Label one handoff family using axes-relative horizontal placement."""

    stem_x = 0.10
    cap_x = 0.01
    (bracket,) = axis.plot(
        [cap_x, stem_x, stem_x, cap_x],
        [y_top, y_top, y_bottom, y_bottom],
        color="#64748b",
        linewidth=1.0,
        transform=axis.get_yaxis_transform(),
        clip_on=False,
        zorder=5,
    )
    bracket.set_gid(f"{gid}-bracket")
    label = axis.text(
        0.15,
        (y_bottom + y_top) / 2.0,
        text,
        color="#334155",
        fontsize=HANDOFF_FAMILY_LABEL_SIZE,
        ha="left",
        va="center",
        linespacing=1.08,
        transform=axis.get_yaxis_transform(),
        clip_on=False,
        zorder=6,
    )
    label.set_gid(f"{gid}-label")


__all__ = ["draw_handoff_family_axis"]
