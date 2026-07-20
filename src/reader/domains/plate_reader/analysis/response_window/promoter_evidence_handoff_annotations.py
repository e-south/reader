"""Family definitions and evidence key for the promoter-evidence handoff."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plot_style import TICK_LABEL_SIZE


def draw_handoff_family_axis(axis: plt.Axes) -> None:
    """Explain the two handoff families beside, rather than inside, the data axes."""

    axis.set_gid("promoter-evidence-handoff-families")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.75, 8.75)
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
        text=("Fluorescence bᵢ\nlog₂(YFP/OD600)\nrelative to\nsame-state\npDual-10"),
        gid="handoff-family-fluorescence",
    )


def handoff_legend_handles(
    *,
    replicate_stat: str,
    confidence_level: float,
    event_label: str,
) -> list[Line2D]:
    """Return one compact key for the three handoff evidence layers."""

    return [
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#94a3b8",
            markersize=5,
            label="Observed response wells",
        ),
        Line2D(
            [],
            [],
            color="#64748b",
            marker="|",
            markeredgewidth=2.0,
            markersize=9,
            linewidth=1.8,
            label=f"{replicate_stat.capitalize()} + {confidence_level:.0%} bootstrap",
        ),
        Line2D(
            [],
            [],
            color="#9ca3af",
            linewidth=6,
            alpha=0.5,
            label=f"Value range covering earliest/latest recorded {_event_time_label(event_label)} times",
        ),
    ]


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
        0.52,
        (y_bottom + y_top) / 2.0,
        text,
        color="#334155",
        fontsize=TICK_LABEL_SIZE,
        ha="center",
        va="center",
        linespacing=1.08,
        transform=axis.get_yaxis_transform(),
        clip_on=False,
        zorder=6,
    )
    label.set_gid(f"{gid}-label")


def _event_time_label(event_label: str) -> str:
    words = event_label.strip().lower().split()
    if not words:
        raise ValueError("promoter-evidence event label must not be empty.")
    return "-".join(words)


__all__ = ["draw_handoff_family_axis", "handoff_legend_handles"]
